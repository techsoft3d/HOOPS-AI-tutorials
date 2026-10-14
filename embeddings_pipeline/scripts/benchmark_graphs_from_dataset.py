"""Build the training graph files (.pt) from a merged .dataset instead of during encoding, and time it.

The flow writes one graph per body while it encodes. The merged dataset holds the same arrays, so
the graphs can also be built afterwards. This script:

1. opens the dataset with DatasetExplorer and maps each body name to its file code (infoset),
2. loads each group once and finds the rows of every body from its file id codes,
3. builds each graph with EmbeddingFlowModel's own conversion and writes <name>.pt to
   on_demand_graph_data/ in the flow directory,
4. times DatasetExplorer.file_dataset on a sample of bodies, the per-body API,
5. compares every new graph with the one the flow wrote in graph_data/.

Usage:
    python scripts/benchmark_graphs_from_dataset.py --flow-dir out/flows/HOOPS_Embedding_Training
"""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import hoops_ai
from hoops_ai.dataset import DatasetExplorer
from hoops_ai.ml.EXPERIMENTAL import EmbeddingFlowModel
from hoops_ai.ml.EXPERIMENTAL.embedding_encoding import EmbeddingEncodingConfig

# Merged group -> arrays the graph conversion reads from a body.
GROUP_ARRAYS: dict[str, tuple[str, ...]] = {
    "faces": ("face_discretization", "face_types", "face_areas"),
    "edges": ("edge_u_grids", "edge_types", "edge_convexities", "edge_dihedral_angles"),
    "graph": ("edges_source", "edges_destination", "num_nodes"),
    "reranker": ("reranker_feature_vector", "oriented_bounding_box"),
}


@dataclass
class GroupRows:
    """Arrays of one group, with each body's row range, keyed by file id code."""

    arrays: dict[str, np.ndarray]
    ranges: dict[int, tuple[int, int]]


class BodyArrays:
    """One body's arrays served like a per-body storage, so the model's conversion reads them unchanged."""

    def __init__(self, arrays: dict[str, Any]) -> None:
        self._arrays = arrays
        self._group = SimpleNamespace(list=lambda: [self])

    def load_data(self, key: str) -> Any:
        return self._arrays[key]

    def get_keys(self) -> list[str]:
        return list(self._arrays)

    def get_store_group(self) -> SimpleNamespace:
        return self._group


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="on_demand_graph_data")
    parser.add_argument("--limit", type=int, default=0, help="Bodies to build; 0 builds all.")
    parser.add_argument("--naive-sample", type=int, default=100, help="Bodies timed with DatasetExplorer.file_dataset.")
    return parser.parse_args()


def load_group(explorer: DatasetExplorer, group: str) -> GroupRows:
    """Load the arrays of a group and the row range of every file id code in it."""
    dataset = explorer.get_group_data(group)
    arrays = {name: dataset[name].values for name in GROUP_ARRAYS[group]}
    codes = dataset[f"file_id_code_{group}"].values
    order = np.argsort(codes, kind="stable")
    if not np.array_equal(order, np.arange(codes.size)):
        # Rows are folded in code order, so this only runs for datasets written another way.
        codes = codes[order]
        arrays = {name: values[order] for name, values in arrays.items()}
    unique_codes, starts, counts = np.unique(codes, return_index=True, return_counts=True)
    ranges = {int(code): (int(start), int(start + count)) for code, start, count in zip(unique_codes, starts, counts)}
    return GroupRows(arrays=arrays, ranges=ranges)


def body_arrays(groups: dict[str, GroupRows], code: int) -> BodyArrays:
    """The arrays of one body, in the shapes and names a per-body encoded store holds."""
    arrays: dict[str, Any] = {}
    for group in ("faces", "edges"):
        start, end = groups[group].ranges.get(code, (0, 0))
        for name, values in groups[group].arrays.items():
            arrays[name] = values[start:end]
    # Only the first body of a file has reranker rows; the others had no such keys in their store.
    if code in groups["reranker"].ranges:
        start, end = groups["reranker"].ranges[code]
        for name, values in groups["reranker"].arrays.items():
            arrays[name] = values[start:end]
    start, end = groups["graph"].ranges.get(code, (0, 0))
    graph = groups["graph"].arrays
    num_nodes = int(graph["num_nodes"][start]) if end > start else len(arrays["face_types"])
    arrays["graph"] = {
        "edges": {"source": graph["edges_source"][start:end], "destination": graph["edges_destination"][start:end]},
        "num_nodes": num_nodes,
    }
    return BodyArrays(arrays)


def time_naive(explorer: DatasetExplorer, codes: list[int]) -> float:
    """Seconds per body to read a body's groups with DatasetExplorer.file_dataset."""
    started = time.perf_counter()
    for code in codes:
        for group in GROUP_ARRAYS:
            explorer.file_dataset(code, group).load()
    return (time.perf_counter() - started) / max(1, len(codes))


def _as_dict(value: Any) -> dict[str, Any]:
    return value.to_dict() if hasattr(value, "to_dict") else dict(value)


def graph_differences(new_file: Path, old_file: Path) -> list[str]:
    """Names of the fields that differ between two graph files."""
    new, old = torch.load(new_file, weights_only=False), torch.load(old_file, weights_only=False)
    differences: list[str] = []
    for part in ("data", "extra"):
        new_fields, old_fields = _as_dict(new[part]), _as_dict(old[part])
        if new_fields.keys() != old_fields.keys():
            differences.append(f"{part} keys {sorted(set(new_fields) ^ set(old_fields))}")
        for key in new_fields.keys() & old_fields.keys():
            a, b = new_fields[key], old_fields[key]
            same = (a.dtype == b.dtype and torch.equal(a, b)) if torch.is_tensor(a) and torch.is_tensor(b) else a == b
            if not same:
                differences.append(f"{part}.{key}")
    return differences


def main() -> int:
    args = _arguments()
    hoops_ai.set_license(os.environ["HOOPS_AI_LICENSE"], validate=True, silent=True)
    flow_dir = args.flow_dir.resolve()
    output = flow_dir / args.output_name
    output.mkdir(exist_ok=True)

    started = time.perf_counter()
    explorer = DatasetExplorer(flow_output_file=str(next(flow_dir.glob("*.flow"))), dask_client_params={"disable": True})
    info = explorer.get_file_info_all()
    bodies: list[tuple[str, int]] = [(str(name), int(code)) for name, code in zip(info["name"], info["id"])]
    if args.limit > 0:
        bodies = bodies[: args.limit]
    open_s = time.perf_counter() - started

    started = time.perf_counter()
    groups = {group: load_group(explorer, group) for group in GROUP_ARRAYS}
    load_s = time.perf_counter() - started

    model = EmbeddingFlowModel(result_dir=str(output), log_file=str(output / "flow.log"), **EmbeddingEncodingConfig().model_kwargs())
    build_s = save_s = 0.0
    for name, code in bodies:
        started = time.perf_counter()
        handler = next(iter(model._iter_encoded_graph_handlers(body_arrays(groups, code))))
        build_s += time.perf_counter() - started
        started = time.perf_counter()
        handler.save_graph(str(output / f"{name}.pt"))
        save_s += time.perf_counter() - started

    naive_codes = [code for _, code in bodies[: args.naive_sample]]
    naive_per_body = time_naive(explorer, naive_codes) if naive_codes else 0.0
    explorer.close(close_dask=False)

    started = time.perf_counter()
    mismatched: dict[str, list[str]] = {}
    missing = 0
    for name, _ in bodies:
        original = flow_dir / "graph_data" / f"{name}.pt"
        if not original.is_file():
            missing += 1
            continue
        differences = graph_differences(output / f"{name}.pt", original)
        if differences:
            mismatched[name] = differences
    compare_s = time.perf_counter() - started

    count = len(bodies)
    total = open_s + load_s + build_s + save_s
    print(f"Bodies: {count}")
    print(f"Open DatasetExplorer and infoset: {open_s:7.1f} s")
    print(f"Load the four groups once:        {load_s:7.1f} s")
    print(f"Build graphs:                     {build_s:7.1f} s  ({build_s / count * 1000:.2f} ms/body)")
    print(f"Write .pt files:                  {save_s:7.1f} s  ({save_s / count * 1000:.2f} ms/body)")
    print(f"Total, bulk path:                 {total:7.1f} s  ({count / total:.0f} bodies/s)")
    print(f"DatasetExplorer.file_dataset per body ({len(naive_codes)} sampled): {naive_per_body * 1000:.0f} ms "
          f"-> {naive_per_body * count / 60:.1f} min for {count} bodies")
    print(f"Compared with graph_data in {compare_s:.1f} s: {count - missing - len(mismatched)} identical, "
          f"{len(mismatched)} different, {missing} without an original")
    for name, differences in list(mismatched.items())[:10]:
        print(f"  {name}: {', '.join(differences)}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
