"""Compare file-backed and in-memory inference from one retained Zarr encoding."""

import argparse
import cProfile
import copy
import json
import os
import pathlib
import pstats
import shutil
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from collections.abc import Iterator, MutableMapping
from datetime import datetime, timezone
from typing import Any, Callable


TUTORIAL_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = TUTORIAL_ROOT / "packages" / "trained_ml_models" / "ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt"
DEFAULT_MODEL_NAME = "HOOPS Embeddings SIGNAL preview"


@dataclass(frozen=True)
class ArrayLayout:
    key: str
    shape: list[int]
    dtype: str
    bytes: int
    chunks: list[int]


@dataclass(frozen=True)
class ReadMeasurement:
    body: str
    field: str
    repetition: int
    seconds: float
    arrays: list[ArrayLayout]


@dataclass(frozen=True)
class ConversionMeasurement:
    body: str
    mode: str
    repetition: int
    seconds: float


class ReadOnlyMetadataCache(MutableMapping[str, bytes]):
    """Cache only Zarr metadata, including missing keys, within one body read."""

    def __init__(self, source: Any, max_bytes: int = 1024 * 1024) -> None:
        self.source = source
        self.max_bytes = max_bytes
        self.cached_bytes = 0
        self.cache: dict[str, bytes | None] = {}

    def __getitem__(self, key: str) -> bytes:
        if key.rsplit("/", 1)[-1] not in {".zarray", ".zgroup", ".zattrs"}:
            return self.source[key]
        if key in self.cache:
            value = self.cache[key]
            if value is None:
                raise KeyError(key)
            return value
        try:
            value = self.source[key]
        except KeyError:
            if len(self.cache) < 256:
                self.cache[key] = None
            raise
        if self.cached_bytes + len(value) <= self.max_bytes and len(self.cache) < 256:
            self.cache[key] = value
            self.cached_bytes += len(value)
        return value

    def __setitem__(self, key: str, value: bytes) -> None:
        raise TypeError("Read-only benchmark cache")

    def __delitem__(self, key: str) -> None:
        raise TypeError("Read-only benchmark cache")

    def __iter__(self) -> Iterator[str]:
        return iter(self.source)

    def __len__(self) -> int:
        return len(self.source)


def _profile_conversion(records: list[Any], run_dir: pathlib.Path, repetitions: int) -> None:
    """Compare identical body arrays with and without storage reads, one body at a time."""
    import torch
    import zarr

    from hoops_ai.ml.EXPERIMENTAL.flow_model_embedding import EmbeddingFlowModel
    from hoops_ai.storage.datastorage.memory_storage_handler import MemoryStorage
    from hoops_ai.storage.datastorage.zarr_storage_handler import OptStorage

    model = EmbeddingFlowModel()
    fields = (
        "graph", "face_discretization", "edge_u_grids", "face_types", "face_areas",
        "reranker_feature_vector", "oriented_bounding_box",
    )
    reads: list[ReadMeasurement] = []
    conversions: list[ConversionMeasurement] = []
    profiles = {mode: cProfile.Profile() for mode in (
        "disk", "preloaded", "no_connectivity", "direct_optional", "metadata_cache",
    )}
    body_paths = sorted({path for record in records for path in record.body_store_paths})

    def convert(storage: Any, mode: str) -> list[Any]:
        if mode == "metadata_cache":
            storage = copy.copy(storage)
            storage.root = zarr.open_group(store=ReadOnlyMetadataCache(storage.store), mode="r")
            storage._store_group = type(storage._store_group)(storage)
        return model._convert_encoded_data_to_graph_handlers(
            storage,
            include_connectivity=mode in {"disk", "preloaded"},
            direct_optional_checks=mode in {"direct_optional", "metadata_cache"},
        )

    def array_layouts(node: Any, key: str) -> list[ArrayLayout]:
        if hasattr(node, "shape"):
            return [ArrayLayout(key, list(node.shape), str(node.dtype), int(node.nbytes), list(node.chunks))]
        layouts: list[ArrayLayout] = []
        for child_key in sorted(node.keys()):
            layouts.extend(array_layouts(node[child_key], f"{key}/{child_key}"))
        return layouts

    for body_index, body_path in enumerate(body_paths):
        disk = OptStorage(body_path)
        memory = MemoryStorage()
        keys = set(disk.get_keys())
        selected_fields = [field for field in fields if field in keys]
        body_key = str(pathlib.Path(body_path).relative_to(run_dir / "encoded"))
        layouts_by_field: dict[str, list[ArrayLayout]] = {}
        for field in selected_fields:
            started = time.perf_counter()
            values = disk.load_data(field)
            elapsed = time.perf_counter() - started
            layouts_by_field[field] = array_layouts(disk.root[field], field)
            reads.append(ReadMeasurement(body_key, field, -1, elapsed, layouts_by_field[field]))
            memory.save_data(field, values)

        disk_handler = model._convert_encoded_data_to_graph_handlers(disk)[0]
        memory_handler = model._convert_encoded_data_to_graph_handlers(memory)[0]
        for attribute in ("x", "edge_attr", "edge_index"):
            torch.testing.assert_close(
                getattr(disk_handler.data, attribute), getattr(memory_handler.data, attribute),
                rtol=0, atol=0,
            )
        assert disk_handler.data.num_nodes == memory_handler.data.num_nodes
        assert disk_handler.extra_dict.keys() == memory_handler.extra_dict.keys()
        for key in disk_handler.extra_dict:
            torch.testing.assert_close(disk_handler.extra_dict[key], memory_handler.extra_dict[key], rtol=0, atol=0)
        disk_sample = model._prepare_model_input_from_handler(disk_handler, include_duplicate_signature=False)
        memory_sample = model._prepare_model_input_from_handler(memory_handler, include_duplicate_signature=False)
        for disk_value, memory_value in zip(disk_sample, memory_sample, strict=True):
            if disk_value is None:
                assert memory_value is None
            else:
                torch.testing.assert_close(disk_value, memory_value, rtol=0, atol=0)
        fast_handler = model._convert_encoded_data_to_graph_handlers(disk, include_connectivity=False)[0]
        fast_sample = model._prepare_model_input_from_handler(fast_handler, include_duplicate_signature=False)
        assert fast_handler.data.num_nodes == disk_handler.data.num_nodes
        for disk_value, fast_value in zip(disk_sample, fast_sample, strict=True):
            if disk_value is None:
                assert fast_value is None
            else:
                torch.testing.assert_close(disk_value, fast_value, rtol=0, atol=0)
        for candidate in ("direct_optional", "metadata_cache"):
            candidate_handler = convert(disk, candidate)[0]
            candidate_sample = model._prepare_model_input_from_handler(
                candidate_handler, include_duplicate_signature=False,
            )
            for expected, actual in zip(disk_sample, candidate_sample, strict=True):
                if expected is None:
                    assert actual is None
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            del candidate_handler, candidate_sample
        del disk_handler, memory_handler, fast_handler, disk_sample, memory_sample, fast_sample

        for repetition in range(repetitions):
            field_order = selected_fields if repetition % 2 == 0 else list(reversed(selected_fields))
            for field in field_order:
                started = time.perf_counter()
                values = disk.load_data(field)
                elapsed = time.perf_counter() - started
                reads.append(ReadMeasurement(body_key, field, repetition, elapsed, layouts_by_field[field]))
                del values
            mode_names = list(profiles)
            offset = repetition % len(mode_names)
            modes = mode_names[offset:] + mode_names[:offset]
            for mode in modes:
                storage = memory if mode == "preloaded" else disk
                started = time.perf_counter()
                handlers = convert(storage, mode)
                elapsed = time.perf_counter() - started
                conversions.append(ConversionMeasurement(body_key, mode, repetition, elapsed))
                del handlers

        for mode in profiles:
            storage = memory if mode == "preloaded" else disk
            profiles[mode].runcall(convert, storage, mode)
        if (body_index + 1) % 100 == 0:
            print(f"Profiled {body_index + 1}/{len(body_paths)} bodies", flush=True)

    summary: dict[str, Any] = {"bodies": len(body_paths), "repetitions": repetitions, "parity": "exact"}
    for mode in profiles:
        totals = [sum(item.seconds for item in conversions if item.mode == mode and item.repetition == repetition)
                  for repetition in range(repetitions)]
        summary[mode] = {"seconds_per_pass": totals, "median_seconds": statistics.median(totals)}
        profiles[mode].dump_stats(str(run_dir / f"conversion_{mode}.prof"))
        with (run_dir / f"conversion_{mode}.txt").open("w", encoding="utf-8") as handle:
            pstats.Stats(profiles[mode], stream=handle).strip_dirs().sort_stats("cumulative").print_stats(60)
    summary["warm_reads_seconds_by_field"] = {
        field: statistics.median([
            sum(item.seconds for item in reads if item.field == field and item.repetition == repetition)
            for repetition in range(repetitions)
        ]) for field in fields
    }
    payload = {"summary": summary, "encoded_records": [asdict(record) for record in records],
               "reads": [asdict(item) for item in reads],
               "conversions": [asdict(item) for item in conversions]}
    (run_dir / "conversion_profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Conversion profile: {run_dir / 'conversion_profile.json'}", flush=True)


@dataclass(frozen=True)
class Result:
    device: str
    mode: str
    batch_size: int
    concurrency: int
    preparation_workers: int | None
    bodies: int
    failures: int
    conversion_seconds: float
    inference_seconds: float
    total_seconds: float
    bodies_per_second: float
    mean_gpu_utilization_percent: float | None
    peak_private_memory_gb: float
    peak_cuda_memory_gb: float | None
    consumer_wait_seconds: float | None
    forward_seconds: float | None
    storage_open_seconds: float | None
    graph_conversion_seconds: float | None
    model_preparation_seconds: float | None
    gpu_starvation_ratio: float | None
    peak_live_prepared_batches: int | None


class ResourceSampler:
    def __init__(self, torch_module: Any, device: str) -> None:
        import psutil

        self._process = psutil.Process()
        self._torch = torch_module
        self._device = device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._private_bytes: list[int] = []
        self._gpu_utilization: list[float] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, name="phase-two-sampler")
        self._thread.start()

    def stop(self) -> tuple[float, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        peak_gb = max(self._private_bytes, default=0) / 1024**3
        mean_gpu = sum(self._gpu_utilization) / len(self._gpu_utilization) if self._gpu_utilization else None
        return peak_gb, mean_gpu

    def _sample(self) -> None:
        while not self._stop.wait(0.1):
            memory = self._process.memory_info()
            self._private_bytes.append(int(getattr(memory, "private", memory.rss)))
            if self._device == "cuda":
                try:
                    self._gpu_utilization.append(float(self._torch.cuda.utilization()))
                except (AttributeError, RuntimeError, OSError):
                    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--repeat-largest",
        type=int,
        default=0,
        help="Repeat the record with the highest total node count this many times.",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--profile-conversion", action="store_true")
    parser.add_argument("--profile-repetitions", type=int, default=5)
    parser.add_argument("--profile-sample-spread", action="store_true",
                        help="Select evenly spaced sorted CAD paths instead of the first paths.")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--modes", nargs="+", choices=("memory", "file"), default=["memory", "file"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--memory-depths", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--memory-workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--file-readers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--file-convert-once",
        action="store_true",
        help="Convert the selected source once, then repeat its graph paths for batch-size tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=TUTORIAL_ROOT / "embeddings_pipeline" / "out" / "in_memory_phase_two",
    )
    return parser.parse_args()


def _measure(
    operation: Callable[[], tuple[int, int, dict[str, Any] | None]],
    torch_module: Any,
    device: str,
) -> tuple[float, float, float | None, float | None, int, int, dict[str, Any] | None]:
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
        torch_module.cuda.synchronize()
    sampler = ResourceSampler(torch_module, device)
    sampler.start()
    started = time.perf_counter()
    bodies, failures, metrics = operation()
    if device == "cuda":
        torch_module.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak_gb, mean_gpu = sampler.stop()
    peak_cuda_gb = torch_module.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None
    return seconds, peak_gb, mean_gpu, peak_cuda_gb, bodies, failures, metrics


def main() -> None:
    args = _parse_args()
    if args.profile_repetitions < 1:
        raise ValueError("--profile-repetitions must be positive.")
    import torch

    import hoops_ai
    from hoops_ai.flowmanager.tasks.parallel_executor import ParallelExecutor
    from hoops_ai.ml.embeddings import HOOPSEmbeddings
    from hoops_ai.ml.embeddings.batch_encode_task import BatchEncodeToGraphTask
    from hoops_ai.storage import CADFileRetriever, LocalStorageProvider

    license_key = os.environ.get("HOOPS_AI_LICENSE")
    if not license_key:
        raise RuntimeError("HOOPS_AI_LICENSE environment variable is required.")
    if "cuda" in args.devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    hoops_ai.set_license(license_key, validate=True)

    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=args.dataset),
        formats=[".stp", ".step", ".iges", ".igs"],
    )
    all_cad_files = sorted(str(path) for path in retriever.get_file_list())
    cad_files = all_cad_files[: args.limit]
    if args.profile_sample_spread and 1 < args.limit < len(all_cad_files):
        cad_files = [all_cad_files[index * (len(all_cad_files) - 1) // (args.limit - 1)]
                     for index in range(args.limit)]
    run_name = datetime.now(timezone.utc).strftime("phase_two_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_name
    encoded_dir = run_dir / "encoded"
    encoded_dir.mkdir(parents=True, exist_ok=True)

    specifications = {
        "graph_dir": str(encoded_dir),
        "generate_images": False,
        "file_size_bucketing": False,
        "time_limit_overall": 1800.0,
    }
    executor = ParallelExecutor(max_workers=args.workers, parallel_task_kwargs=specifications)
    encode_started = time.perf_counter()
    try:
        task = executor.execute(BatchEncodeToGraphTask, cad_files, force_sequential=False, specifications=specifications)
    finally:
        executor.close_pool()
    encode_seconds = time.perf_counter() - encode_started
    encoded_records = [
        record
        for item in task.results
        if item.get("error") is None
        for record in (item.get("result") or [])
    ]
    if not encoded_records:
        raise RuntimeError("No CAD files were encoded successfully.")

    if args.profile_conversion:
        try:
            _profile_conversion(encoded_records, run_dir, args.profile_repetitions)
        finally:
            shutil.rmtree(encoded_dir, ignore_errors=True)
        return

    largest_record = max(encoded_records, key=lambda record: sum(record.num_nodes_by_body))
    repetition_count = args.repeat_largest if args.repeat_largest > 0 else 1
    records = [largest_record] * repetition_count if args.repeat_largest > 0 else encoded_records
    largest_total_nodes = sum(largest_record.num_nodes_by_body)
    if args.repeat_largest > 0:
        print(
            f"Repeating largest encoded file {largest_record.cad_path} "
            f"({largest_total_nodes:.0f} total nodes, {len(largest_record.body_store_paths)} bodies) "
            f"{repetition_count} times.",
            flush=True,
        )

    if args.model_name not in HOOPSEmbeddings.list_available_models():
        HOOPSEmbeddings.register_model(args.model_name, str(args.checkpoint.resolve()))

    results: list[Result] = []
    graph_paths: list[str] = []
    graph_loads: list[float] = []
    file_conversion_seconds = 0.0
    file_conversion_failures = 0

    for device in args.devices:
        embedder = HOOPSEmbeddings(model=args.model_name, device=device)
        warmup_records = records[: min(8, len(records))]
        embedder._embed_records_in_memory(
            warmup_records,
            batch_size=max(1, len(warmup_records)),
            balance_batches=True,
            show_progress=False,
        )

        if "memory" in args.modes:
            for batch_size in args.batch_sizes:
                for depth in args.memory_depths:
                    worker_counts = args.memory_workers if depth > 0 else [1]
                    for preparation_workers in worker_counts:
                        def run_memory() -> tuple[int, int, dict[str, Any]]:
                            rows, _, conversion_errors, inference_errors, metrics = embedder._embed_records_in_memory(
                                records,
                                batch_size=batch_size,
                                balance_batches=True,
                                show_progress=False,
                                prefetch_depth=depth,
                                preparation_workers=preparation_workers,
                                collect_metrics=True,
                            )
                            return len(rows), len(conversion_errors) + len(inference_errors), metrics.to_dict()

                        seconds, peak_gb, mean_gpu, peak_cuda_gb, bodies, failures, metrics = _measure(
                            run_memory, torch, device
                        )
                        results.append(Result(
                        device=device,
                        mode="memory",
                        batch_size=batch_size,
                        concurrency=depth,
                        preparation_workers=preparation_workers,
                        bodies=bodies,
                        failures=failures,
                        conversion_seconds=float(metrics["prepare_seconds"]),
                        inference_seconds=float(metrics["forward_seconds"]),
                        total_seconds=seconds,
                        bodies_per_second=bodies / seconds,
                        mean_gpu_utilization_percent=mean_gpu,
                        peak_private_memory_gb=peak_gb,
                        peak_cuda_memory_gb=peak_cuda_gb,
                        consumer_wait_seconds=float(metrics["consumer_wait_seconds"]),
                        forward_seconds=float(metrics["forward_seconds"]),
                        storage_open_seconds=float(metrics["storage_open_seconds"]),
                        graph_conversion_seconds=float(metrics["graph_conversion_seconds"]),
                        model_preparation_seconds=float(metrics["model_preparation_seconds"]),
                        gpu_starvation_ratio=float(metrics["gpu_starvation_ratio"]),
                        peak_live_prepared_batches=int(metrics["peak_live_prepared_batches"]),
                        ))
                        print(asdict(results[-1]), flush=True)

        if "file" in args.modes and not graph_paths:
            conversion_started = time.perf_counter()
            conversion_records = [largest_record] if args.file_convert_once else records
            graph_meta, conversion_errors = embedder._convert_records_to_graphs(conversion_records, encoded_dir)
            file_conversion_seconds = time.perf_counter() - conversion_started
            file_conversion_failures = len(conversion_errors)
            graph_paths = list(graph_meta) * repetition_count
            graph_loads = [graph_meta[path].num_nodes for path in graph_meta] * repetition_count

        if "file" not in args.modes:
            continue

        for batch_size in args.batch_sizes:
            for readers in args.file_readers:
                def run_file() -> tuple[int, int, None]:
                    rows, errors = embedder._embed_graph_rows(
                        graph_paths,
                        batch_size=batch_size,
                        show_progress=False,
                        num_workers=readers,
                        item_loads=graph_loads,
                        prefetch_factor=2 if readers > 0 else None,
                    )
                    return len(rows), file_conversion_failures + len(errors), None

                seconds, peak_gb, mean_gpu, peak_cuda_gb, bodies, failures, _ = _measure(
                    run_file, torch, device
                )
                total_seconds = file_conversion_seconds + seconds
                results.append(Result(
                    device=device,
                    mode="file",
                    batch_size=batch_size,
                    concurrency=readers,
                    preparation_workers=None,
                    bodies=bodies,
                    failures=failures,
                    conversion_seconds=file_conversion_seconds,
                    inference_seconds=seconds,
                    total_seconds=total_seconds,
                    bodies_per_second=bodies / total_seconds,
                    mean_gpu_utilization_percent=mean_gpu,
                    peak_private_memory_gb=peak_gb,
                    peak_cuda_memory_gb=peak_cuda_gb,
                    consumer_wait_seconds=None,
                    forward_seconds=None,
                    storage_open_seconds=None,
                    graph_conversion_seconds=None,
                    model_preparation_seconds=None,
                    gpu_starvation_ratio=None,
                    peak_live_prepared_batches=None,
                ))
                print(asdict(results[-1]), flush=True)

        del embedder
        if device == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "metadata": {
            "dataset": str(args.dataset.resolve()),
            "requested_files": len(cad_files),
            "encoded_files": len(encoded_records),
            "benchmark_files": len(records),
            "repeated_largest": args.repeat_largest,
            "largest_cad_path": largest_record.cad_path,
            "largest_total_nodes": largest_total_nodes,
            "largest_body_count": len(largest_record.body_store_paths),
            "modes": args.modes,
            "file_convert_once": args.file_convert_once,
            "encoding_failures": len(task.errors),
            "encoding_seconds": encode_seconds,
            "workers": args.workers,
        },
        "results": [asdict(result) for result in results],
    }
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results: {results_path}")
    shutil.rmtree(encoded_dir, ignore_errors=True)


if __name__ == "__main__":
    main()