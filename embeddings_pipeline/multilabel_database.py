"""Synthetic multi-label catalog for the TMCAD multi-labeling demo.

This module is the multi-label sibling of ``database.py``. Instead of predicting
a single field (``Material`` / ``Process`` / ``Cost``), it attaches a *set* of
functional categories to each mechanical part — a multi-label problem. A part
can belong to several groups at once (e.g. a flanged bushing is ``Rotational``,
``Sealing`` and ``Mounting`` all at the same time), exactly like the furniture
example where one product is ``Accessories`` + ``Casegoods`` + ``Storage``.

It provides the data-access side of the demo (the aggregation ``MultiLabelRule``
lives in the notebook so it can be read and edited there):

* ``MultiLabelContextProvider`` — the thin ``ContextProvider`` adapter you
  implement. The Context Layer calls only ``get_contexts`` / ``set_contexts`` /
  ``list_numeric_keys`` on it. In your own project this forwards to your real
  catalog / PLM client.
* ``SyntheticCatalog`` — a stand-in for that external system. It fabricates
  deterministic, hash-derived multi-label records so the tutorial runs with no
  backend. You would delete this and point the adapter at your real client.

Use it like this::

    from multilabel_database import MultiLabelContextProvider

    provider = MultiLabelContextProvider()
    records = provider.get_contexts([h.id for h in hits])
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from hoops_ai.ml.context_layer import ContextProvider

# Functional groups a mechanical part can belong to. Unlike a classic
# single-label taxonomy these categories deliberately overlap: a single part
# routinely carries two or three of them at once.
CATEGORY_KEY = "Categories"
TAGS: tuple[str, ...] = (
    "Rotational",
    "Fastening",
    "Structural",
    "Housing",
    "Sealing",
    "Mounting",
)


def _hash(name: str, salt: str) -> int:
    return int(hashlib.md5(f"{salt}:{name}".encode("utf-8")).hexdigest()[:8], 16)


def _iter_ids(items: Iterable) -> Iterable[str]:
    """Yield CAD-file ids from a mixed iterable.

    Accepts VectorHit objects, plain strings/paths, or nested iterables of
    either, so it works whether ``search_by_shape`` results are flat or batched.
    """
    for item in items:
        if hasattr(item, "id"):
            yield item.id
        elif isinstance(item, (str, Path)):
            yield str(item)
        else:
            yield from _iter_ids(item)


class SyntheticCatalog:
    """Stand-in for an external product catalog / PLM holding multi-label tags.

    Records are hash-derived from the file stem, so the same id always returns
    the same categories across runs and machines. To stay realistic the backend
    leaves some records incomplete, mimicking a catalog where many parts are not
    yet fully tagged:

    * **The top hit is always returned empty.** The query's nearest geometric
      twin has no tags yet, forcing the predictor to aggregate categories from
      the rest of the neighbourhood instead of copying the best match.
    * **~20 % of the remaining parts are empty**, the rest carry tags.
    * **~70 % of queries land in a "shape family"**: a shared set of 2-3
      categories is pinned across most of the neighbourhood (with light per-part
      noise — a tag dropped here, an extra tag added there), so the predictor can
      recover a confident multi-label set. The other ~30 % stay divided, so you
      can also observe lower-confidence, sparser predictions.

    Explicit writes via :meth:`store` always win over the synthetic record.
    """

    _EMPTY_BUCKET_CEILING = 8       # buckets 8-9 -> ~20 % of parts empty
    _CONSENSUS_QUERY_CEILING = 7    # fingerprint bucket 0-6 -> ~70 % of queries

    def __init__(self) -> None:
        self._written: dict[str, dict] = {}

    def fetch(self, part_ids: Sequence[str]) -> dict[str, dict]:
        """Return ``{part_id: record}`` for the requested ids, in order."""
        ids_list = list(part_ids)
        family = self._family_tags(ids_list)
        out: dict[str, dict] = {}
        for index, pid in enumerate(ids_list):
            if index == 0:
                record: dict = {}
            else:
                record = self._record_for(pid, family)
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
        return _hash(Path(pid).stem, "missing") % 10

    @staticmethod
    def _own_tags(pid: str) -> list[str]:
        """One or two intrinsic categories for a part, from its stem."""
        stem = Path(pid).stem
        first = TAGS[_hash(stem, "tag-a") % len(TAGS)]
        tags = [first]
        if _hash(stem, "tag-b") % 2 == 0:
            second = TAGS[_hash(stem, "tag-b-pick") % len(TAGS)]
            if second != first:
                tags.append(second)
        return tags

    @classmethod
    def _record_for(cls, pid: str, family: list[str] | None) -> dict:
        if cls._missingness_bucket(pid) >= cls._EMPTY_BUCKET_CEILING:
            return {}
        if family is not None:
            tags = set(family)
            stem = Path(pid).stem
            if len(tags) > 1 and _hash(stem, "drop") % 4 == 0:
                ordered = sorted(tags)
                tags.discard(ordered[_hash(stem, "which") % len(ordered)])
            if _hash(stem, "add") % 3 == 0:
                tags.update(cls._own_tags(pid)[:1])
        else:
            tags = set(cls._own_tags(pid))
        return {CATEGORY_KEY: sorted(tags)}

    @classmethod
    def _family_tags(cls, ids: Sequence[str]) -> list[str] | None:
        """Return the 2-3 categories shared across a neighbourhood, or None."""
        if not ids:
            return None
        fingerprint = "|".join(sorted(str(pid) for pid in ids))
        if _hash(fingerprint, "family") % 10 >= cls._CONSENSUS_QUERY_CEILING:
            return None
        count = 2 + (_hash(fingerprint, "family-count") % 2)
        picked: list[str] = []
        salt = 0
        while len(picked) < count and salt < 50:
            candidate = TAGS[_hash(fingerprint, f"family-{salt}") % len(TAGS)]
            if candidate not in picked:
                picked.append(candidate)
            salt += 1
        return picked


class MultiLabelContextProvider(ContextProvider):
    """Adapter exposing a multi-label catalog to the HOOPS AI Context Layer.

    The Context Layer calls exactly three methods on a ``ContextProvider``:

    * :meth:`get_contexts`      -> ``backend.fetch``  (read categories by id)
    * :meth:`set_contexts`      -> ``backend.store``  (write categories back)
    * :meth:`list_numeric_keys` -> declares no numeric keys (all categorical)

    In production, swap ``SyntheticCatalog`` for your real catalog / PLM client
    and forward these calls to its read / write endpoints — the rest of the
    Context Layer is unchanged.
    """

    def __init__(self, backend: SyntheticCatalog | None = None) -> None:
        self._backend = backend if backend is not None else SyntheticCatalog()

    def get_contexts(self, part_ids: Sequence[str]) -> Mapping[str, dict]:
        return self._backend.fetch(list(_iter_ids(part_ids)))

    def set_contexts(self, updates: Mapping[str, dict]) -> None:
        self._backend.store(updates)

    def list_numeric_keys(self) -> Sequence[str]:
        return ()
