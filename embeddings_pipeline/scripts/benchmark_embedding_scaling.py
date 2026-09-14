"""Benchmark TMCAD encoding workers and balanced embedding inference batches.

This script separates the two stages introduced by ML-191:

1. Encode the same CAD corpus with several process counts and retain one .data cache.
2. Convert that cache to PyG graphs once, load one model, and replay GPU inference
   with several balanced batch sizes.

Heavy imports stay inside main so Windows spawn workers do not import torch.
"""

import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


CAD_SUFFIXES = {".stp", ".step", ".iges", ".igs"}
DEFAULT_MODEL_NAME = "HOOPS Embeddings SIGNAL preview"
DEFAULT_WORKERS = [4, 8, 12, 16, 20]
DEFAULT_BATCH_SIZES = [16, 32, 64, 128]
TUTORIAL_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    TUTORIAL_ROOT
    / "packages"
    / "trained_ml_models"
    / "ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt"
)


@dataclass(frozen=True)
class EncodingResult:
    """One phase-one worker scaling measurement."""

    workers: int
    files: int
    encoded_files: int
    bodies: int
    failed: int
    seconds: float
    files_per_second: float


@dataclass(frozen=True)
class InferenceResult:
    """One balanced inference batch-size measurement."""

    batch_size: int
    reader_workers: int
    prefetch_factor: int | None
    bodies: int
    batches: int
    failed: int
    seconds: float | None
    bodies_per_second: float | None
    max_batch_nodes: float
    mean_batch_nodes: float
    peak_allocated_gb: float | None
    peak_reserved_gb: float | None
    status: str
    error: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TMCAD encoding scaling and balanced batched inference."
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=os.environ.get("PATH_DATASET_TMCAD"),
        help="TMCAD root. Defaults to PATH_DATASET_TMCAD.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=DEFAULT_CHECKPOINT,
        help="SIGNAL checkpoint path.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument(
        "--inference-reader-workers",
        type=int,
        nargs="+",
        default=[0],
        help="DataLoader reader workers to benchmark. Values above zero prefetch graph batches.",
    )
    parser.add_argument(
        "--prefetch-factors",
        type=int,
        nargs="+",
        default=[2],
        help="Batches queued per reader. Ignored when the reader count is zero.",
    )
    parser.add_argument(
        "--reuse-graph-dir",
        type=pathlib.Path,
        default=None,
        help="Skip CAD encoding and reuse converted .pt graphs from an earlier run.",
    )
    parser.add_argument(
        "--reuse-workers",
        type=int,
        default=None,
        help="Worker run whose .data stores feed inference. Defaults to max(--workers).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic subset size. Default uses the full dataset.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cuda")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=TUTORIAL_ROOT / "embeddings_pipeline" / "out" / "scaling",
    )
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--keep-all-encoding-caches",
        action="store_true",
        help="Keep .data stores from every worker run instead of only --reuse-workers.",
    )
    parser.add_argument(
        "--resume-results",
        type=pathlib.Path,
        default=None,
        help="Resume an inference matrix from an existing results.json file.",
    )
    parser.add_argument("--point-result", type=pathlib.Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.reuse_graph_dir is None:
        if args.dataset is None:
            raise ValueError("Set PATH_DATASET_TMCAD or pass --dataset.")
        if not args.dataset.is_dir():
            raise FileNotFoundError(f"TMCAD directory not found: {args.dataset}")
    elif not args.reuse_graph_dir.is_dir():
        raise FileNotFoundError(f"Graph directory not found: {args.reuse_graph_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SIGNAL checkpoint not found: {args.checkpoint}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")
    if not args.workers or any(worker <= 1 for worker in args.workers):
        raise ValueError("Every --workers value must be greater than one.")
    if not args.batch_sizes or any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("Every --batch-sizes value must be positive.")
    if not args.inference_reader_workers or any(
        workers < 0 for workers in args.inference_reader_workers
    ):
        raise ValueError("Every --inference-reader-workers value must be non-negative.")
    if not args.prefetch_factors or any(factor <= 0 for factor in args.prefetch_factors):
        raise ValueError("Every --prefetch-factors value must be positive.")

    if args.reuse_graph_dir is None:
        reuse_workers = args.reuse_workers or max(args.workers)
        if reuse_workers not in args.workers:
            raise ValueError("--reuse-workers must also be present in --workers.")


def _load_cad_files(dataset: pathlib.Path, limit: int | None) -> list[str]:
    from hoops_ai.storage import CADFileRetriever, LocalStorageProvider

    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=dataset),
        formats=sorted(CAD_SUFFIXES),
    )
    cad_files = sorted(str(pathlib.Path(path).resolve()) for path in retriever.get_file_list())
    if limit is not None:
        cad_files = cad_files[:limit]
    if not cad_files:
        raise RuntimeError(f"No supported CAD files found under {dataset}")
    return cad_files


def _write_csv(path: pathlib.Path, rows: list[Any]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_results(
    run_dir: pathlib.Path,
    metadata: dict[str, Any],
    encoding_rows: list[EncodingResult],
    inference_rows: list[InferenceResult],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "encoding_results": [asdict(row) for row in encoding_rows],
        "inference_results": [asdict(row) for row in inference_rows],
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(run_dir / "encoding_scaling.csv", encoding_rows)
    _write_csv(run_dir / "inference_batch_scaling.csv", inference_rows)


def _configuration_key(row: InferenceResult) -> tuple[int, int, int | None]:
    return row.batch_size, row.reader_workers, row.prefetch_factor


def _classify_subprocess_failure(output: str) -> str:
    normalized = output.lower()
    if "winerror 1455" in normalized or "paging file is too small" in normalized:
        return "worker_start_commit_failure"
    if "dataloader worker" in normalized and "exited unexpectedly" in normalized:
        return "worker_process_failure"
    if "outofmemoryerror" in normalized or "out of memory" in normalized:
        return "oom"
    return "subprocess_failure"


def _run_isolated_inference_matrix(args: argparse.Namespace) -> None:
    if args.resume_results is not None:
        results_path = args.resume_results.resolve()
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        run_dir = results_path.parent
        metadata = payload.get("metadata", {})
        encoding_rows = [EncodingResult(**row) for row in payload.get("encoding_results", [])]
        inference_rows = [InferenceResult(**row) for row in payload.get("inference_results", [])]
    else:
        run_name = datetime.now(timezone.utc).strftime("tmcad_scaling_%Y%m%dT%H%M%SZ")
        run_dir = args.output_dir.resolve() / run_name
        results_path = run_dir / "results.json"
        metadata = {
            "reused_graph_dir": str(args.reuse_graph_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "model_name": args.model_name,
            "device": "cuda" if args.device == "gpu" else args.device,
            "inference_batch_sizes": args.batch_sizes,
            "inference_reader_workers": args.inference_reader_workers,
            "inference_prefetch_factors": args.prefetch_factors,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "isolated_subprocesses": True,
        }
        encoding_rows = []
        inference_rows = []

    run_dir.mkdir(parents=True, exist_ok=True)
    completed = {_configuration_key(row) for row in inference_rows}
    point_root = run_dir / ".points"
    point_root.mkdir(exist_ok=True)

    for batch_size in args.batch_sizes:
        for reader_workers in args.inference_reader_workers:
            factors = args.prefetch_factors if reader_workers > 0 else [None]
            for prefetch_factor in factors:
                key = batch_size, reader_workers, prefetch_factor
                if key in completed:
                    print(f"skip completed batch={batch_size} readers={reader_workers} prefetch={prefetch_factor}")
                    continue

                point_name = f"b{batch_size}_r{reader_workers}_p{prefetch_factor or 0}"
                point_result = point_root / f"{point_name}.json"
                command = [
                    sys.executable,
                    str(pathlib.Path(__file__).resolve()),
                    "--reuse-graph-dir", str(args.reuse_graph_dir.resolve()),
                    "--checkpoint", str(args.checkpoint.resolve()),
                    "--model-name", args.model_name,
                    "--batch-sizes", str(batch_size),
                    "--inference-reader-workers", str(reader_workers),
                    "--prefetch-factors", str(prefetch_factor or 1),
                    "--device", args.device,
                    "--output-dir", str(point_root / point_name),
                    "--point-result", str(point_result),
                ]
                if args.limit is not None:
                    command.extend(["--limit", str(args.limit)])
                if args.show_progress:
                    command.append("--show-progress")

                print(f"run batch={batch_size} readers={reader_workers} prefetch={prefetch_factor}", flush=True)
                completed_process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                output = "\n".join((completed_process.stdout, completed_process.stderr)).strip()
                if completed_process.returncode == 0 and point_result.is_file():
                    row = InferenceResult(**json.loads(point_result.read_text(encoding="utf-8")))
                else:
                    status = _classify_subprocess_failure(output)
                    row = InferenceResult(
                        batch_size=batch_size,
                        reader_workers=reader_workers,
                        prefetch_factor=prefetch_factor,
                        bodies=0,
                        batches=0,
                        failed=0,
                        seconds=None,
                        bodies_per_second=None,
                        max_batch_nodes=0.0,
                        mean_batch_nodes=0.0,
                        peak_allocated_gb=None,
                        peak_reserved_gb=None,
                        status=status,
                        error=output[-12000:] or f"Child exited with code {completed_process.returncode}",
                    )
                inference_rows.append(row)
                completed.add(key)
                _write_results(run_dir, metadata, encoding_rows, inference_rows)
                print(f"recorded status={row.status} bodies/s={row.bodies_per_second or 0.0:.2f}", flush=True)

    successful = [row for row in inference_rows if row.status == "ok"]
    if successful:
        recommended = max(successful, key=lambda row: row.bodies_per_second or 0.0)
        metadata["best_measured_batch_size"] = recommended.batch_size
        metadata["best_measured_reader_workers"] = recommended.reader_workers
        metadata["best_measured_prefetch_factor"] = recommended.prefetch_factor
        metadata["best_measured_bodies_per_second"] = recommended.bodies_per_second
    _write_plot(run_dir, encoding_rows, inference_rows)
    _write_results(run_dir, metadata, encoding_rows, inference_rows)
    print(f"Results: {results_path}")


def _batch_load_stats(
    loads: list[float],
    batch_size: int,
    sampler_class: type,
) -> tuple[int, float, float]:
    sampler = sampler_class(loads, batch_size=batch_size, shuffle=False, drop_last=False)
    batch_loads = [sum(loads[index] for index in indices) for indices in sampler]
    if not batch_loads:
        return 0, 0.0, 0.0
    return len(batch_loads), max(batch_loads), sum(batch_loads) / len(batch_loads)


def _load_graph_paths_and_loads(
    graph_dir: pathlib.Path,
    limit: int | None,
    torch_module: Any,
) -> tuple[list[str], list[float]]:
    graph_paths = sorted(str(path.resolve()) for path in graph_dir.glob("*.pt"))
    if limit is not None:
        graph_paths = graph_paths[:limit]
    if not graph_paths:
        raise RuntimeError(f"No .pt graphs found under {graph_dir}")

    item_loads: list[float] = []
    for graph_path in graph_paths:
        payload = torch_module.load(graph_path, map_location="cpu", weights_only=False)
        item_loads.append(float(payload["data"].num_nodes))
    return graph_paths, item_loads


def _write_plot(
    run_dir: pathlib.Path,
    encoding_rows: list[EncodingResult],
    inference_rows: list[InferenceResult],
) -> pathlib.Path:
    import matplotlib.pyplot as plt

    figure, (encode_axis, inference_axis) = plt.subplots(1, 2, figsize=(14, 5))

    encode_axis.plot(
        [row.workers for row in encoding_rows],
        [row.files_per_second for row in encoding_rows],
        "o-",
    )
    encode_axis.set_xlabel("encoding workers")
    encode_axis.set_ylabel("CAD files/s")
    encode_axis.set_title("CPU encoding scaling")
    encode_axis.grid(alpha=0.3)

    successful = [row for row in inference_rows if row.status == "ok"]
    series = sorted(
        {(row.batch_size, row.prefetch_factor) for row in successful},
        key=lambda value: (value[0], value[1] if value[1] is not None else 0),
    )
    for batch_size, prefetch_factor in series:
        batch_rows = [
            row for row in successful
            if row.batch_size == batch_size and row.prefetch_factor == prefetch_factor
        ]
        inference_axis.plot(
            [row.reader_workers for row in batch_rows],
            [row.bodies_per_second for row in batch_rows],
            "s-",
            label=f"batch {batch_size}, prefetch {prefetch_factor or 'none'}",
        )
    inference_axis.set_xlabel("DataLoader reader workers")
    inference_axis.set_ylabel("bodies/s", color="tab:red")
    inference_axis.tick_params(axis="y", labelcolor="tab:red")
    inference_axis.set_title("Prefetched GPU inference scaling")
    inference_axis.grid(alpha=0.3)
    inference_axis.legend()

    memory_axis = inference_axis.twinx()
    for batch_size in sorted({row.batch_size for row in successful}):
        batch_rows = [row for row in successful if row.batch_size == batch_size]
        memory_axis.plot(
            [row.reader_workers for row in batch_rows],
            [row.peak_reserved_gb for row in batch_rows],
            "^--",
            alpha=0.5,
        )
    memory_axis.set_ylabel("peak CUDA reserved (GB)", color="tab:blue")
    memory_axis.tick_params(axis="y", labelcolor="tab:blue")

    figure.suptitle("TMCAD two-phase embedding scaling")
    figure.tight_layout()
    plot_path = run_dir / "scaling.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    if args.reuse_graph_dir is not None and args.point_result is None:
        _run_isolated_inference_matrix(args)
        return

    license_key = os.environ.get("HOOPS_AI_LICENSE")
    if not license_key:
        raise RuntimeError("HOOPS_AI_LICENSE environment variable is required.")

    # These imports are intentionally local. Spawned Windows workers re-execute only
    # this module's top-level code and therefore remain free of torch/model imports.
    import numpy as np
    import torch

    import hoops_ai
    from hoops_ai.dataset import DatasetLoader
    from hoops_ai.dataset.torch_adapter import BalancedLoadBatchSampler
    from hoops_ai.flowmanager.tasks.parallel_executor import ParallelExecutor
    from hoops_ai.ml.embeddings import HOOPSEmbeddings
    from hoops_ai.ml.embeddings.batch_encode_task import BatchEncodeTask, EncodedRecord
    from hoops_ai.ml.embeddings.batch_graph_task import BatchGraphTask
    from hoops_ai.ml.embeddings.embedding_batch_pipeline import EmbeddingBatchPipeline

    device = "cuda" if args.device == "gpu" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    hoops_ai.set_license(license_key, validate=True)
    cad_files = [] if args.reuse_graph_dir else _load_cad_files(args.dataset, args.limit)
    reuse_workers = args.reuse_workers or max(args.workers)
    run_name = datetime.now(timezone.utc).strftime("tmcad_scaling_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_name
    cache_root = run_dir / "encoded"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "dataset": str(args.dataset.resolve()) if args.dataset else None,
        "reused_graph_dir": str(args.reuse_graph_dir.resolve()) if args.reuse_graph_dir else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "model_name": args.model_name,
        "device": device,
        "torch_version": torch.__version__,
        "logical_cpus": os.cpu_count(),
        "file_count": len(cad_files),
        "workers": args.workers,
        "reuse_workers": reuse_workers,
        "inference_batch_sizes": args.batch_sizes,
        "inference_reader_workers": args.inference_reader_workers,
        "inference_prefetch_factors": args.prefetch_factors,
        "balance_inference_batches": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        metadata.update({
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_total_gb": total_bytes / 1024**3,
            "gpu_free_at_start_gb": free_bytes / 1024**3,
        })

    encoding_rows: list[EncodingResult] = []
    inference_rows: list[InferenceResult] = []
    retained_records: list[EncodedRecord] = []
    retained_graph_dir = args.reuse_graph_dir or cache_root / f"workers_{reuse_workers}"
    metadata["encoding_errors_by_workers"] = {}

    if args.reuse_graph_dir:
        print(f"Reusing graphs: {args.reuse_graph_dir}")
    else:
        print(f"TMCAD files: {len(cad_files):,}")
        print(f"Encoding workers: {args.workers}")
    print(
        f"Inference batches: {args.batch_sizes}; readers: "
        f"{args.inference_reader_workers}; prefetch: {args.prefetch_factors} "
        f"({device}, balanced by num_nodes)"
    )

    for workers in [] if args.reuse_graph_dir else args.workers:
        graph_dir = cache_root / f"workers_{workers}"
        shutil.rmtree(graph_dir, ignore_errors=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        specifications: dict[str, Any] = {
            "graph_dir": str(graph_dir),
            "generate_images": False,
            "file_size_bucketing": False,
            "time_limit_overall": 1800.0,
        }
        executor = ParallelExecutor(max_workers=workers, parallel_task_kwargs=specifications)
        started = time.perf_counter()
        try:
            task = executor.execute(
                BatchEncodeTask,
                cad_files,
                force_sequential=False,
                specifications=specifications,
            )
        finally:
            executor.close_pool()
        seconds = time.perf_counter() - started
        records = [
            record
            for result in sorted(task.results, key=lambda result: result.get("item_index", 0))
            if result.get("error") is None
            for record in (result.get("result") or [])
        ]
        bodies = sum(len(record.body_store_paths) for record in records)
        metadata["encoding_errors_by_workers"][str(workers)] = [
            error.get("error", str(error)) for error in task.errors
        ]
        row = EncodingResult(
            workers=workers,
            files=len(cad_files),
            encoded_files=len(records),
            bodies=bodies,
            failed=len(task.errors),
            seconds=seconds,
            files_per_second=len(records) / seconds if seconds else 0.0,
        )
        encoding_rows.append(row)
        print(
            f"encode workers={workers:>2}: {row.files_per_second:>6.2f} files/s, "
            f"{row.failed} failed, {row.seconds:.1f}s",
            flush=True,
        )

        _write_results(run_dir, metadata, encoding_rows, inference_rows)
        if workers == reuse_workers:
            retained_records = records
        elif not args.keep_all_encoding_caches:
            shutil.rmtree(graph_dir, ignore_errors=True)

    if not args.reuse_graph_dir and not retained_records:
        raise RuntimeError(f"The {reuse_workers}-worker encoding run produced no reusable records.")

    if args.model_name not in HOOPSEmbeddings.list_available_models():
        HOOPSEmbeddings.register_model(args.model_name, str(args.checkpoint.resolve()))

    model_load_started = time.perf_counter()
    embedder = HOOPSEmbeddings(model=args.model_name, device=device)
    pipeline = EmbeddingBatchPipeline(embedder._embeddings_model, embedder.model_id)
    metadata["model_load_seconds"] = time.perf_counter() - model_load_started
    metadata["model_load_count"] = 1

    if args.reuse_graph_dir:
        load_started = time.perf_counter()
        graph_paths, item_loads = _load_graph_paths_and_loads(
            retained_graph_dir,
            args.limit,
            torch,
        )
        metadata["graph_load_scan_seconds"] = time.perf_counter() - load_started
        metadata["file_count"] = len(graph_paths)
    else:
        convert_started = time.perf_counter()
        graph_task = BatchGraphTask(embedder._embeddings_model.flowmodel, retained_graph_dir)
        graph_task.execute({"encoded_records": retained_records})
        prepared = graph_task.outputs["prepared_graphs"]
        graph_meta, conversion_errors = prepared.graph_meta, prepared.errors
        metadata["graph_conversion_seconds"] = time.perf_counter() - convert_started
        metadata["graph_conversion_failures"] = len(conversion_errors)
        metadata["graph_conversion_errors"] = conversion_errors
        graph_paths = list(graph_meta)
        item_loads = [float(graph_meta[path].num_nodes) for path in graph_paths]
    if not graph_paths or not all(np.isfinite(item_loads)):
        raise RuntimeError("No converted graphs with valid num_nodes loads are available.")

    warmup_count = min(min(args.batch_sizes), len(graph_paths))
    pipeline.predict_dataset(
        DatasetLoader(graph_files=graph_paths[:warmup_count], skip_unloadable_files=False),
        batch_size=warmup_count,
        show_progress=False,
        num_workers=0,
        item_loads=item_loads[:warmup_count],
    )
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    for batch_size in args.batch_sizes:
        for reader_workers in args.inference_reader_workers:
          for prefetch_factor in args.prefetch_factors if reader_workers > 0 else [None]:
            batch_count, max_batch_nodes, mean_batch_nodes = _batch_load_stats(
                item_loads,
                batch_size,
                BalancedLoadBatchSampler,
            )
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            started = time.perf_counter()
            try:
                rows, errors = pipeline.predict_dataset(
                    DatasetLoader(graph_files=graph_paths, skip_unloadable_files=False),
                    batch_size=batch_size,
                    show_progress=args.show_progress,
                    num_workers=reader_workers,
                    item_loads=item_loads,
                    prefetch_factor=prefetch_factor,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                result = InferenceResult(
                    batch_size=batch_size,
                    reader_workers=reader_workers,
                    prefetch_factor=prefetch_factor,
                    bodies=len(rows),
                    batches=batch_count,
                    failed=len(errors),
                    seconds=seconds,
                    bodies_per_second=len(rows) / seconds if seconds else 0.0,
                    max_batch_nodes=max_batch_nodes,
                    mean_batch_nodes=mean_batch_nodes,
                    peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
                    peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3 if device == "cuda" else None,
                    status="ok",
                    error=None,
                )
                del rows
            except torch.cuda.OutOfMemoryError as exc:
                result = InferenceResult(
                    batch_size=batch_size,
                    reader_workers=reader_workers,
                    prefetch_factor=prefetch_factor,
                    bodies=0,
                    batches=batch_count,
                    failed=len(graph_paths),
                    seconds=None,
                    bodies_per_second=None,
                    max_batch_nodes=max_batch_nodes,
                    mean_batch_nodes=mean_batch_nodes,
                    peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                    peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
                    status="oom",
                    error=str(exc),
                )
            finally:
                if device == "cuda":
                    torch.cuda.empty_cache()

            inference_rows.append(result)
            print(
                f"infer batch={batch_size:>3} readers={reader_workers} "
                f"prefetch={prefetch_factor}: status={result.status} "
                f"bodies/s={result.bodies_per_second or 0.0:>7.2f} "
                f"peak_reserved={result.peak_reserved_gb or 0.0:>5.2f} GB "
                f"max_batch_nodes={result.max_batch_nodes:.0f}",
                flush=True,
            )
            _write_results(run_dir, metadata, encoding_rows, inference_rows)

    successful = [row for row in inference_rows if row.status == "ok"]
    if successful:
        recommended = max(successful, key=lambda row: row.bodies_per_second or 0.0)
        metadata["best_measured_batch_size"] = recommended.batch_size
        metadata["best_measured_reader_workers"] = recommended.reader_workers
        metadata["best_measured_prefetch_factor"] = recommended.prefetch_factor
        metadata["best_measured_bodies_per_second"] = recommended.bodies_per_second

    plot_path = _write_plot(run_dir, encoding_rows, inference_rows)
    _write_results(run_dir, metadata, encoding_rows, inference_rows)
    if args.point_result is not None:
        args.point_result.parent.mkdir(parents=True, exist_ok=True)
        args.point_result.write_text(json.dumps(asdict(inference_rows[0]), indent=2), encoding="utf-8")
    print(f"Results: {run_dir / 'results.json'}")
    print(f"Plot:    {plot_path}")
    print(f"Cache:   {retained_graph_dir}")


if __name__ == "__main__":
    main()