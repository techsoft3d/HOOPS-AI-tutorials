"""Synthetic PLM-style metadata for the TMCAD demo dataset.

This module shows how to connect the HOOPS AI Context Layer to an external
metadata system (PLM, ERP, or any database). It is split into two parts so the
integration boundary is obvious:

* ``OnDemandContextProvider`` — the thin adapter you implement. It subclasses
  ``ContextProvider`` and implements only the three methods the Context Layer
  calls. In your own project this is where you translate those calls into your
  PLM/ERP client's read/write/schema endpoints.
* ``SyntheticPLM`` — a stand-in for that external system. It generates stable,
  hash-derived records so the tutorial is reproducible without a real backend.
  You would delete this class and point the adapter at your real client.

Use it like this::

    from database import OnDemandContextProvider

    provider = OnDemandContextProvider()        # wraps a SyntheticPLM backend
    metadata = provider.get_contexts([h.id for h in hits])
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from hoops_ai.ml.context_layer import ContextProvider

MATERIALS = ["Steel", "Aluminum", "Titanium", "Stainless Steel"]
PROCESSES = ["CNC Milling", "Turning", "5-Axis Machining"]

BASE_MATERIAL_COST = {
    "Steel": 40, "Aluminum": 55, "Titanium": 150,
    "Stainless Steel": 90,
}
PROCESS_MULTIPLIER = {
    "CNC Milling": 1.0,
    "Turning": 0.9,
    "5-Axis Machining": 1.8,
}





def _hash(name: str, salt: str) -> int:
    return int(hashlib.md5(f"{salt}:{name}".encode("utf-8")).hexdigest()[:8], 16)


def compute_true_cost_from_equation(material: str, process: str, internal_features: int) -> float:
    base = BASE_MATERIAL_COST[material] ** 1.5 / 10
    feat = 1 + (0.08 * internal_features) ** 2.2
    return base * feat * PROCESS_MULTIPLIER[process]


def metadata_for(path: str) -> dict:
    """Return deterministic synthetic metadata for one CAD file path."""
    name = Path(path).stem
    material = MATERIALS[_hash(name, "mat") % len(MATERIALS)]
    process = PROCESSES[_hash(name, "proc") % len(PROCESSES)]
    features = 5 + _hash(name, "feat") % 15
    noise = (_hash(name, "noise") % 200) / 1000.0 - 0.10   # +/- 10 %
    cost = round(compute_true_cost_from_equation(material, process, features) * (1 + noise), 2)
    return {
        "Material": material,
        "Process": process,
        "InternalFeatures": features,
        "Cost": cost,
    }


def _iter_ids(items):
    """Yield CAD-file ids from a mixed iterable.

    Accepts: VectorHit, plain strings/paths, or nested iterables of either
    (so ``search_by_shape`` results work whether they're flat or batched).
    """
    for item in items:
        if hasattr(item, "id"):
            yield item.id
        elif isinstance(item, (str, Path)):
            yield str(item)
        else:
            yield from _iter_ids(item)


def build_records(hits_or_paths) -> dict[str, dict]:
    """Build a ``{path: metadata}`` dict from VectorHits or CAD file paths."""
    
    return {pid: metadata_for(pid) for pid in _iter_ids(hits_or_paths)}


class SyntheticPLM:
    """Stand-in for an external PLM / ERP system.

    This class plays the role your production metadata store plays: given a list
    of part ids it returns whatever records it holds, and it accepts writes. In
    your own project you would NOT reimplement this — you would call your
    existing PLM/ERP client instead. It exists here only so the tutorial runs
    without a real backend.

    Records are hash-derived from the file *stem*, so the same id always returns
    the same metadata across runs and machines. To stay realistic, the backend
    deliberately leaves some records incomplete — like a PLM where many parts
    are not yet fully tagged:

    * **The first id in a query is always returned empty.** This forces the
      predictor to aggregate context from the neighbours instead of copying the
      top hit's tags — the whole point of the Context Layer.
    * **The remaining ids follow a 50 / 30 / 20 mix:** ~50 % full records,
      ~30 % only ``Material`` + ``Process`` (textual tags entered, numeric
      estimates pending), ~20 % empty.
    * **~60 % of queries land in "consensus mode":** a single
      ``(Material, Process)`` pair is pinned on ~50 % of the hit-set (with
      ``Cost`` recomputed to stay consistent), giving the predictor enough
      agreement to reach ready_to_propose. The rest stay varied so you can also
      observe needs_review / insufficient_evidence outputs.

    Explicit writes via :meth:`store` always win over the synthetic record.
    """

    NUMERIC_FIELDS: tuple[str, ...] = ("Cost",)

    _FULL_BUCKET_CEILING = 6      # 0-5 → ~50 % full
    _PARTIAL_BUCKET_CEILING = 8   # 6-7 → ~30 % partial (Material + Process)
    # 8-9 → ~20 % empty
    _CONSENSUS_QUERY_CEILING = 6  # hit-set bucket 0-5 → ~60 % of queries
    _CONSENSUS_RECORD_CEILING = 5 # within a consensus query, bucket 0-4 → ~50 % share

    def __init__(self) -> None:
        self._written: dict[str, dict] = {}

    def fetch(self, part_ids: Sequence[str]) -> dict[str, dict]:
        """Return ``{part_id: record}`` for the requested ids, in order.

        This is the read endpoint. Your real backend would issue one batched
        query here and return the rows it finds (omitting unknown ids).
        """
        ids_list = list(part_ids)
        consensus_tags = self._consensus_tags(ids_list)
        out: dict[str, dict] = {}
        for index, pid in enumerate(ids_list):
            if index == 0:
                # Strip the top-scoring hit so the predictor has to
                # aggregate from the rest. Explicit overrides still win.
                record: dict = {}
            else:
                record = self._shape_for(pid)
                if (
                    consensus_tags
                    and record
                    and self._missingness_bucket(pid) <= self._CONSENSUS_RECORD_CEILING
                ):
                    record = self._apply_consensus(pid, record, consensus_tags)
            overrides = self._written.get(pid)
            if overrides:
                record = {**record, **overrides}
            out[pid] = record
        return out

    def store(self, updates: Mapping[str, dict]) -> None:
        """Persist explicit per-part overrides (the write endpoint)."""
        for pid, payload in updates.items():
            self._written.setdefault(str(pid), {}).update(payload)

    # --- synthetic data generation (your real backend already owns the data) ---

    @staticmethod
    def _missingness_bucket(pid: str) -> int:
        """Stable 0\u20139 bucket per file stem; controls metadata completeness."""
        return _hash(Path(pid).stem, "missing") % 10

    @classmethod
    def _shape_for(cls, pid: str) -> dict:
        """Return the synthetic record for ``pid`` after applying the
        per-file missingness rule. Always returns a dict (possibly empty).
        """
        bucket = cls._missingness_bucket(pid)
        if bucket >= cls._PARTIAL_BUCKET_CEILING:
            return {}
        record = metadata_for(pid)
        if bucket >= cls._FULL_BUCKET_CEILING:
            # Partial: drop the numeric / shape-intrinsic keys, keep
            # only the textual tags a human is likely to have entered.
            record = {
                "Material": record["Material"],
                "Process": record["Process"],
            }
        return record

    @classmethod
    def _consensus_tags(cls, ids: Sequence[str]) -> dict | None:
        """Return a single ``(Material, Process)`` to share across the
        hit-set, or ``None`` when this query stays in varied mode.
        """
        if not ids:
            return None
        fingerprint = "|".join(sorted(str(pid) for pid in ids))
        if _hash(fingerprint, "cluster") % 10 >= cls._CONSENSUS_QUERY_CEILING:
            return None
        return {
            "Material": MATERIALS[_hash(fingerprint, "consensus-mat") % len(MATERIALS)],
            "Process": PROCESSES[_hash(fingerprint, "consensus-proc") % len(PROCESSES)],
        }

    @classmethod
    def _apply_consensus(cls, pid: str, record: dict, tags: dict) -> dict:
        """Replace ``Material`` / ``Process`` with the consensus pair and
        re-derive ``Cost`` from them so the cluster stays self-consistent.
        """
        merged = {**record, **tags}
        features = merged.get("InternalFeatures")
        if features is not None:
            noise = (_hash(Path(pid).stem, "noise") % 200) / 1000.0 - 0.10
            merged["Cost"] = round(
                compute_true_cost_from_equation(tags["Material"], tags["Process"], features) * (1 + noise),
                2,
            )
        return merged


class OnDemandContextProvider(ContextProvider):
    """Adapter exposing a PLM/ERP backend to the HOOPS AI Context Layer.

    This is the only class you implement for your own system. The Context Layer
    calls exactly three methods on a ``ContextProvider``; each one here is a
    thin translation to the backend:

    * :meth:`get_contexts`     -> ``backend.fetch``  (read metadata by part id)
    * :meth:`set_contexts`     -> ``backend.store``  (write metadata back)
    * :meth:`list_numeric_keys` -> which keys are numeric (so numeric
      aggregation rules fire instead of categorical ones)

    In production, swap ``SyntheticPLM`` for your real PLM/ERP client and
    forward these three calls to its read / write / schema endpoints — the rest
    of the Context Layer is unchanged.
    """

    def __init__(
        self,
        backend: SyntheticPLM | None = None,
        numeric_keys: Sequence[str] | None = None,
    ) -> None:
        self._backend = backend if backend is not None else SyntheticPLM()
        self._numeric_keys = (
            tuple(numeric_keys)
            if numeric_keys is not None
            else tuple(self._backend.NUMERIC_FIELDS)
        )

    def get_contexts(self, part_ids: Sequence[str]) -> Mapping[str, dict]:
        return self._backend.fetch(list(_iter_ids(part_ids)))

    def set_contexts(self, updates: Mapping[str, dict]) -> None:
        self._backend.store(updates)

    def list_numeric_keys(self) -> Sequence[str]:
        return self._numeric_keys
