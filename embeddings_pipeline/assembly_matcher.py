"""Assembly-to-assembly similarity retrieval on top of HOOPS Embeddings.

The corpus stores one embedding vector per body, using the source file path as the
id (repeated once per body). A multi-body assembly therefore appears as
several rows that share the same id, and the number of rows for an id is that
assembly's body/part count.

``AssemblyMatcher`` groups those per-body vectors by assembly and, given a query
assembly, returns the most similar assemblies in the corpus via a two-stage pipeline:

1. Stage 1 - candidate generation: per query body, a fast vector-store shortlist.
2. Stage 2 - verification: an optimal one-to-one (Hungarian) part matching between
   the query parts and each candidate's parts, with optional TF-IDF rare-part
   weighting so common hardware (fasteners, washers) does not dominate.

The final score blends that geometric match with a bag-of-parts composition score
(how alike the two assemblies' mix and counts of parts are); see ``search``.
"""

from __future__ import annotations

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from hoops_ai.ml.embeddings import Embedding


def _l2norm(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class AssemblyMatcher:
    """Assembly-to-assembly retrieval built on HOOPS per-body embeddings.

    Parameters
    ----------
    searcher : CADSearch
        The loaded index. Powers the fast Stage-1 candidate shortlist and, via its
        shape model, embeds out-of-corpus query files at search time. When
        ``embedding_batch`` is omitted, the corpus vectors are pulled from it too.
    embedding_batch : EmbeddingBatch, optional
        The corpus (vectors + ids), as returned by ``CADSearch.load_shape_index``.
        Held in memory and used to (a) cluster parts for the IDF rarity weights and
        (b) reconstruct candidate part vectors during Stage-2 scoring. Defaults to
        ``searcher._get_shape_embedding_batch()`` when not provided.
    min_bodies_for_assembly : int
        An id with at least this many rows is treated as an assembly.

    Notes
    -----
    The shape model and corpus are currently read from private CADSearch attributes
    (``_shape_model``, ``_get_shape_embedding_batch``) as a temporary workaround;
    public accessors are planned for a future release.
    """

    def __init__(self, searcher, embedding_batch=None, min_bodies_for_assembly: int = 2):
        self.searcher = searcher
        # Workaround: reuse the searcher's shape model and stored corpus instead of
        # taking them as separate arguments. Swap to public accessors once CADSearch
        # exposes them.
        self.embedder = getattr(searcher, "_shape_model", None)
        if embedding_batch is None:
            embedding_batch = searcher._get_shape_embedding_batch()

        ids = list(embedding_batch.ids)
        self._model_id = embedding_batch.model
        self._vectors = _l2norm(embedding_batch.values)            # (n_bodies, dim), normalized
        self._asm_rows = defaultdict(list)                          # id -> [row indices]
        for row, fid in enumerate(ids):
            self._asm_rows[fid].append(row)
        self._counts = Counter(ids)
        self._n_docs = max(len(self._counts), 1)
        self.assemblies = [fid for fid, n in self._counts.items() if n >= min_bodies_for_assembly]

        # row -> file id (used by IDF)
        self._row_aid = [None] * self._vectors.shape[0]
        for fid, rows in self._asm_rows.items():
            for r in rows:
                self._row_aid[r] = fid

        # IDF state (lazy)
        self._idf_ready = False
        self._row_to_cluster = None
        self._idf_per_cluster = None
        self._centroids = None          # cluster centroids, for weighting query parts
        self._bop = {}                  # id -> bag-of-parts TF-IDF histogram (composition)

    # ---------- TF-IDF rare-part weighting ----------
    def build_part_rarity_weights(self, n_clusters: Optional[int] = None,
                                  niter: int = 10, seed: int = 42) -> None:
        """Build inverse-document-frequency (IDF) rarity weights for parts.

        Cluster corpus bodies so similar parts share a cluster; a cluster's document
        frequency is the number of distinct files containing it, and rare clusters get a
        higher IDF weight. ``n_clusters`` defaults to ~N/8 capped at 4096.

        Uses FAISS k-means: a heavily optimized C++ routine that clusters hundreds of
        thousands of body vectors in seconds (orders of magnitude faster than a pure
        Python k-means for large corpora).
        """
        import faiss

        N = self._vectors.shape[0]
        k = max(1, min(int(n_clusters or min(max(64, N // 8), 4096)), N))

        X = np.ascontiguousarray(self._vectors, dtype=np.float32)
        km = faiss.Kmeans(X.shape[1], k, niter=niter, seed=seed, verbose=False)
        km.train(X)
        _, assign = km.index.search(X, 1)            # nearest centroid per body
        row_to_cluster = assign.reshape(-1).astype(np.int64)

        # document frequency = number of distinct files touching each cluster
        files_per_cluster = defaultdict(set)
        for row, c in enumerate(row_to_cluster):
            files_per_cluster[int(c)].add(self._row_aid[row])
        idf = np.array(
            [np.log(1.0 + self._n_docs / max(len(files_per_cluster.get(c, ())), 1))
             for c in range(k)],
            dtype=np.float32,
        )

        self._row_to_cluster = row_to_cluster
        self._idf_per_cluster = idf
        self._idf_ready = True

        # Keep centroids (normalized) so out-of-corpus query parts can be assigned a
        # cluster -> IDF weight, and precompute each file's bag-of-parts signature.
        self._centroids = _l2norm(np.asarray(km.centroids).reshape(k, X.shape[1]))
        self._bop = {fid: self._bag_of_parts(row_to_cluster[rows])
                     for fid, rows in self._asm_rows.items()}

    # ---------- cluster / composition helpers ----------
    def _clusters_for_vectors(self, V: np.ndarray):
        """Nearest-centroid cluster id per row of V (None if IDF not built)."""
        if not self._idf_ready or self._centroids is None:
            return None
        return np.argmax(V @ self._centroids.T, axis=1)

    def _bag_of_parts(self, clusters) -> dict:
        """TF-IDF histogram over part-clusters for one assembly (L2-normalized).

        Treats each assembly as a 'document' and each part-cluster as a 'word': the
        weight of a cluster is (part count in cluster) x IDF, then L2-normalized so the
        cosine of two histograms measures how alike two assemblies' *compositions* are
        (their mix and multiplicity of parts), independent of geometric correspondence.
        """
        vec = defaultdict(float)
        for c in clusters:
            ci = int(c)
            vec[ci] += float(self._idf_per_cluster[ci])
        norm = float(np.sqrt(sum(v * v for v in vec.values()))) or 1.0
        return {c: v / norm for c, v in vec.items()}

    def _bop_sim(self, q_bop: dict, fid: str) -> float:
        """Cosine similarity between a query bag-of-parts and a stored one."""
        c_bop = self._bop.get(fid)
        if not q_bop or not c_bop:
            return 0.0
        if len(q_bop) > len(c_bop):
            q_bop, c_bop = c_bop, q_bop
        return float(sum(w * c_bop.get(c, 0.0) for c, w in q_bop.items()))

    def _part_weights(self, rows, use_idf: bool) -> np.ndarray:
        if not use_idf or not self._idf_ready:
            return np.ones(len(rows), dtype=np.float32)
        return self._idf_per_cluster[self._row_to_cluster[np.asarray(rows)]]

    # ---------- per-candidate scoring ----------
    def _score_candidate(self, Q, w_q, q_bop, fid, sim_thresh, method, use_idf,
                         coverage_mode, bop_weight):
        rows = self._asm_rows[fid]
        C = self._vectors[rows]                       # (N, dim), normalized
        S = Q @ C.T                                   # (M, N) cosine similarities
        M, N = S.shape
        w_c = self._part_weights(rows, use_idf)       # (N,) candidate-part IDF

        # --- pick matched part pairs ---
        if method == "voting":
            best_j = S.argmax(axis=1)
            covered = S[np.arange(M), best_j] >= sim_thresh
            pairs = [(int(i), int(best_j[i])) for i in np.where(covered)[0]]
        else:  # hungarian: optimal one-to-one matching (SciPy linear assignment)
            r, c = linear_sum_assignment(-S)
            pairs = [(i, j) for i, j in zip(r.tolist(), c.tolist()) if S[i, j] >= sim_thresh]

        # --- geometric match score: quality x rarity-aware coverage ---
        if pairs:
            num = sum(S[i, j] * w_c[j] for i, j in pairs)
            den = sum(w_c[j] for i, j in pairs) or 1.0
            match_quality = num / den

            matched_mass = sum(0.5 * (w_q[i] + w_c[j]) for i, j in pairs)
            total_q = float(w_q.sum())
            total_c = float(w_c.sum())
            if coverage_mode == "containment":        # how much of the query is present
                denom = total_q
            elif coverage_mode == "jaccard":          # overlap / union
                denom = total_q + total_c - matched_mass
            else:                                     # "symmetric" (default)
                denom = max(total_q, total_c)
            coverage = min(matched_mass / (denom or 1.0), 1.0)
            geom = float(match_quality * coverage)
            matched = len(pairs)
        else:
            coverage, geom, matched = 0.0, 0.0, 0

        # --- composition score (bag-of-parts) and blend ---
        eff_bw = bop_weight if q_bop else 0.0         # no blend when no composition signature
        bop = self._bop_sim(q_bop, fid) if eff_bw > 0 else 0.0
        score = (1.0 - eff_bw) * geom + eff_bw * bop
        return {"score": float(score), "geom_score": geom, "bop_sim": float(bop),
                "coverage": float(coverage), "matched": matched, "M": M, "N": N}

    # ---------- query body de-duplication ----------
    @staticmethod
    def _dedupe_rows(Qn, decimals: int = 4):
        """Return indices of unique query bodies (near-identical rows merged)."""
        _, idx = np.unique(Qn.round(decimals), axis=0, return_index=True)
        return sorted(idx.tolist())

    # ---------- public search ----------
    def search(self, query_path, top_k=5, candidate_k=10, sim_thresh=0.80,
               method="hungarian", use_idf=True, candidate_mode="search",
               assemblies_only=True, query_embeddings=None, n_jobs=None,
               dedupe_query=True, reuse_index_vectors=True,
               coverage_mode="symmetric", bop_weight=0.3) -> List[dict]:
        """Return the top-K most similar assemblies to ``query_path``.

        Parameters
        ----------
        query_embeddings : list[Embedding] or None
            Pre-computed per-body embeddings of the query. When provided, the query is
            not embedded again.
        reuse_index_vectors : bool
            If the query path is already in the index, reuse its stored body vectors
            instead of re-embedding the file.
        n_jobs : int or None
            Thread pool size for Stage-2 scoring. None/1 = serial.
        dedupe_query : bool
            Merge near-identical query bodies before the Stage-1 vector-store queries.
        candidate_mode : "search" (fast vector-store shortlist) or "all" (brute-force
            every assembly - accurate for small corpora).
        assemblies_only : bool
            If True, only multi-body files are scored/returned.
        coverage_mode : "symmetric" (matched / larger side), "containment" (matched /
            query size - find assemblies that *contain* the query), or "jaccard".
        bop_weight : float in [0, 1]
            Blend weight of the bag-of-parts composition score against the geometric
            Hungarian score. 0 = geometry only; higher favors matching part mix/counts.
        """
        if use_idf and not self._idf_ready:
            self.build_part_rarity_weights()

        # --- obtain query body embeddings (in priority order) ---
        if query_embeddings is not None:
            query_bodies = query_embeddings
        elif reuse_index_vectors and query_path in self._asm_rows:
            rows = self._asm_rows[query_path]
            query_bodies = [
                Embedding(values=self._vectors[r], model=self._model_id,
                          dim=int(self._vectors.shape[1]), id=query_path)
                for r in rows
            ]
        else:
            if self.embedder is None:
                raise ValueError(
                    f"Query '{query_path}' is not in the corpus and no shape model is "
                    "available from the searcher to embed it."
                )
            query_bodies = self.embedder.embed_shape(query_path)
        Q = _l2norm(np.stack([b.values for b in query_bodies]))   # (M, dim)

        # Per-query-part rarity weights and composition signature (one cluster pass).
        q_clusters = self._clusters_for_vectors(Q)
        if q_clusters is not None:
            w_q = self._idf_per_cluster[q_clusters]
            q_bop = self._bag_of_parts(q_clusters) if bop_weight > 0 else {}
        else:
            w_q = np.ones(Q.shape[0], dtype=np.float32)
            q_bop = {}

        # --- Stage 1: candidate assemblies ---
        display_meta = {}
        if candidate_mode == "all":
            candidates = set(self.assemblies if assemblies_only else self._asm_rows.keys())
        else:
            rep_idx = list(self._dedupe_rows(Q)) if dedupe_query else list(range(len(query_bodies)))
            candidates = set()
            for i in rep_idx:
                emb = query_bodies[i]
                hits = self.searcher.search_by_embedding(
                    emb, top_k=candidate_k, num_candidates=max(50, candidate_k))
                for h in hits:
                    candidates.add(h.id)
                    prev = display_meta.get(h.id)
                    if prev is None or h.score > prev[0]:
                        display_meta[h.id] = (h.score, h.metadata)
        candidates.discard(query_path)

        # --- Stage 2: verify + score (optionally threaded) ---
        cand_list = [fid for fid in candidates
                     if fid in self._asm_rows and not (assemblies_only and self._counts[fid] < 2)]

        def _work(fid):
            res = self._score_candidate(Q, w_q, q_bop, fid, sim_thresh, method, use_idf,
                                        coverage_mode, bop_weight)
            res["assembly"] = fid
            res["n_parts"] = self._counts[fid]
            res["display_metadata"] = (display_meta.get(fid) or (None, None))[1]
            return res

        if n_jobs and n_jobs != 1 and len(cand_list) > 1:
            with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
                scored = list(ex.map(_work, cand_list))
        else:
            scored = [_work(fid) for fid in cand_list]

        scored.sort(key=lambda d: (d["score"], d["coverage"]), reverse=True)
        return scored[:top_k]

    # ---------- demo helper (optional) ----------
    def find_demo_queries(self, n: int = 5, top_k: int = 8, good_score: float = 0.5,
                          min_parts: int = 8, max_parts: int = 200,
                          sample_size: int = 200, seed: int = 0) -> List[str]:
        """Suggest assemblies that make good demo queries (a strong top-K neighborhood).

        Samples in-corpus assemblies within a part-count band and ranks each by how good
        its whole top-``top_k`` result set is, not just its single best match - so the
        chosen queries showcase several similar assemblies at once. The ranking key is
        (number of results clearing ``good_score``, then mean score of those results), and
        only queries that return a full top-``top_k`` set are considered. Returns the top
        ``n`` ids. This is a convenience for building notebook galleries; it is not needed
        for normal retrieval.
        """
        import random

        rng = random.Random(seed)
        pool = [fid for fid in self.assemblies if min_parts <= self._counts[fid] <= max_parts]
        sample = rng.sample(pool, min(sample_size, len(pool)))

        ranked = []
        for q in sample:
            res = self.search(q, top_k=top_k, candidate_k=10, method="hungarian",
                              use_idf=True, n_jobs=8)
            if len(res) < top_k:
                continue
            good = [r["score"] for r in res if r["score"] >= good_score]
            if not good:
                continue
            mean_good = sum(good) / len(good)
            ranked.append((len(good), mean_good, q))
        ranked.sort(reverse=True)
        return [q for _, _, q in ranked[:n]]
