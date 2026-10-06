"""Benchmark the public GPU embedding API across worker and batch-size settings.

Each point runs in an isolated child process and records end-to-end and public phase timings.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import csv
import json
import logging
import os
import pathlib
import subprocess
import sys
from threading import Event, Thread
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


CAD_FORMATS = [".stp", ".step", ".iges", ".igs"]
DEFAULT_MODEL_NAME = "HOOPS Embeddings SIGNAL preview"
DEFAULT_WORKERS = [24]
DEFAULT_BATCH_SIZES = [32, 64, 128, 256]
TUTORIAL_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    TUTORIAL_ROOT
    / "packages"
    / "trained_ml_models"
    / "ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt"
)


@dataclass
class _RunResources:
    parent_peak_rss_bytes: int = 0
    workers_peak_rss_bytes: int = 0
    combined_peak_rss_bytes: int = 0
    memory_samples: int = 0
    sampling_error: str | None = None


@contextmanager
def _sample_memory(resources: _RunResources) -> Iterator[None]:
    import psutil

    parent = psutil.Process()
    stopped = Event()

    def sample() -> None:
        try:
            while not stopped.is_set():
                parent_rss = parent.memory_info().rss
                worker_rss = 0
                for child in parent.children(recursive=True):
                    try:
                        worker_rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        logging.getLogger(__name__).debug("Worker exited during RSS sampling")
                resources.parent_peak_rss_bytes = max(resources.parent_peak_rss_bytes, parent_rss)
                resources.workers_peak_rss_bytes = max(resources.workers_peak_rss_bytes, worker_rss)
                resources.combined_peak_rss_bytes = max(resources.combined_peak_rss_bytes, parent_rss + worker_rss)
                resources.memory_samples += 1
                stopped.wait(0.1)
        except Exception as exc:
            resources.sampling_error = str(exc)
            logging.getLogger(__name__).exception("Memory sampler failed")

    thread = Thread(target=sample, name="benchmark-memory")
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


@dataclass(frozen=True)
class ScalingPoint:
    """One end-to-end embed_shape_batch measurement."""

    workers: int
    batch_size: int
    requested_files: int
    embedded_bodies: int
    successful_files: int
    failed_files: int
    seconds: float | None
    input_files_per_second: float | None
    successful_files_per_second: float | None
    status: str
    error: str | None
    phase_seconds: dict[str, float] | None = None
    resources: dict[str, Any] | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the public GPU embedding API across worker and batch-size settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=os.environ.get("PATH_DATASET_TMCAD"),
        help="TMCAD root. Defaults to PATH_DATASET_TMCAD.",
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS,
                        help="Worker counts. Legacy releases load a GPU model per worker; reduce if memory is limited.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inference-graph-mode", choices=("file", "memory"), default="memory")
    parser.add_argument("--encoding-storage-mode", choices=("directory", "packed"), default="packed")
    parser.add_argument("--encoding-pack-max-files", type=int, default=8)
    parser.add_argument(
        "--inference-batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
        help="Inference batch sizes. Each worker/batch-size pair runs in a fresh process.",
    )
    parser.add_argument("--inference-memory-prefetch-depth", type=int, default=2)
    parser.add_argument("--inference-memory-workers", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=TUTORIAL_ROOT / "embeddings_pipeline" / "out" / "embed_shape_batch_scaling",
    )
    parser.add_argument("--resume-results", type=pathlib.Path, default=None)
    parser.add_argument("--point-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--point-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--point-result", type=pathlib.Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--point-errors", type=pathlib.Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _benchmark_points(workers: list[int], batch_sizes: list[int]) -> list[tuple[int, int]]:
    return [
        (worker_count, batch_size)
        for worker_count in workers
        for batch_size in batch_sizes
    ]


def _validate_args(args: argparse.Namespace) -> None:
    if args.dataset is None:
        raise ValueError("Set PATH_DATASET_TMCAD or pass --dataset.")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"TMCAD directory not found: {args.dataset}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.workers or any(workers <= 1 for workers in args.workers):
        raise ValueError("Every worker count must be greater than one.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if not args.inference_batch_sizes or any(size <= 0 for size in args.inference_batch_sizes):
        raise ValueError("Every --inference-batch-sizes value must be positive.")
    if args.inference_memory_prefetch_depth < 0:
        raise ValueError("--inference-memory-prefetch-depth must be non-negative.")
    if args.inference_memory_workers <= 0:
        raise ValueError("--inference-memory-workers must be positive.")
    if args.encoding_pack_max_files <= 0:
        raise ValueError("--encoding-pack-max-files must be positive.")
    if args.encoding_storage_mode == "packed" and args.inference_graph_mode != "memory":
        raise ValueError("Packed benchmarks require memory inference.")


def _load_cad_files(dataset: pathlib.Path, limit: int | None) -> list[str]:
    from hoops_ai.storage import CADFileRetriever, LocalStorageProvider

    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=dataset),
        formats=CAD_FORMATS,
    )
    cad_files = sorted(str(pathlib.Path(path).resolve()) for path in retriever.get_file_list())
    if limit is not None:
        cad_files = cad_files[:limit]
    if not cad_files:
        raise RuntimeError(f"No supported CAD files found under {dataset}")
    return cad_files


def _run_point(args: argparse.Namespace) -> None:
    import torch
    import numpy as np

    import hoops_ai
    from hoops_ai.ml.embeddings import HOOPSEmbeddings

    license_key = os.environ.get("HOOPS_AI_LICENSE")
    if not license_key:
        raise RuntimeError("HOOPS_AI_LICENSE environment variable is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment.")

    hoops_ai.set_license(license_key, validate=True)
    cad_files = _load_cad_files(args.dataset, args.limit)
    if args.model_name not in HOOPSEmbeddings.list_available_models():
        HOOPSEmbeddings.register_model(args.model_name, str(args.checkpoint.resolve()))

    embedder = HOOPSEmbeddings(model=args.model_name, device="cuda")
    resources = _RunResources()
    with _sample_memory(resources):
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = embedder.embed_shape_batch(
            cad_files,
            num_workers=args.point_workers,
            show_progress=sys.stderr.isatty(),
            specifications={
                "inference_graph_mode": args.inference_graph_mode,
                "encoding_storage_mode": args.encoding_storage_mode,
                "encoding_pack_max_files": args.encoding_pack_max_files,
                "inference_batch_size": args.point_batch_size,
                "inference_prefetch_depth": args.inference_memory_prefetch_depth,
                "inference_prepare_workers": args.inference_memory_workers,
                "inference_num_workers": 2,
                "inference_prefetch_factor": 2,
                "balance_inference_batches": True,
                "generate_images": False,
                "inference_phase_diagnostics": True,
            },
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started

    failed_files = int(batch.metadata.get("failed_count", 0))
    successful_files = max(0, len(cad_files) - failed_files)
    point = ScalingPoint(
        workers=args.point_workers,
        batch_size=args.point_batch_size,
        requested_files=len(cad_files),
        embedded_bodies=len(batch.ids),
        successful_files=successful_files,
        failed_files=failed_files,
        seconds=seconds,
        input_files_per_second=len(cad_files) / seconds,
        successful_files_per_second=successful_files / seconds,
        status="ok",
        error=None,
        phase_seconds=batch.metadata.get("phase_seconds"),
        resources=asdict(resources),
    )
    args.point_result.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.point_result.with_suffix(".npz"),
        ids=np.asarray(batch.ids, dtype=str), values=batch.values,
        inputs=np.asarray(cad_files, dtype=str),
    )
    args.point_result.write_text(json.dumps(asdict(point), indent=2), encoding="utf-8")
    if args.point_errors is not None:
        errors = batch.metadata.get("errors", [])
        args.point_errors.parent.mkdir(parents=True, exist_ok=True)
        args.point_errors.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(
        f"workers={point.workers} batch_size={point.batch_size} time={point.seconds:.1f}s "
        f"input_files/s={point.input_files_per_second:.2f} "
        f"successful_files/s={point.successful_files_per_second:.2f} "
        f"failed={point.failed_files}",
        flush=True,
    )


def _write_results(
    results_path: pathlib.Path,
    metadata: dict[str, Any],
    points: list[ScalingPoint],
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": [asdict(point) for point in points]}
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if points:
        with (results_path.parent / "embed_shape_batch_scaling.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0])))
            writer.writeheader()
            writer.writerows(asdict(point) for point in points)


def _write_toshi_plot(
    run_dir: pathlib.Path,
    points: list[ScalingPoint],
    throughput_field: str,
    filename: str,
    title: str,
) -> pathlib.Path:
    import matplotlib.pyplot as plt

    figure, time_axis = plt.subplots(figsize=(8, 4.5))
    throughput_axis = time_axis.twinx()
    lines = []
    for workers in sorted({point.workers for point in points if point.status == "ok"}):
        successful = sorted(
            (point for point in points if point.status == "ok" and point.workers == workers),
            key=lambda point: point.batch_size,
        )
        batch_sizes = [point.batch_size for point in successful]
        lines.extend(time_axis.plot(
            batch_sizes, [point.seconds for point in successful], "o-",
            label=f"time, workers={workers}",
        ))
        lines.extend(throughput_axis.plot(
            batch_sizes, [getattr(point, throughput_field) for point in successful], "s--",
            label=f"files/s, workers={workers}",
        ))

    time_axis.set_xticks(sorted({point.batch_size for point in points}))
    time_axis.set_xlabel("inference_batch_size")
    time_axis.set_ylabel("time (s)", color="#1f77b4")
    throughput_axis.set_ylabel("files/s", color="#e52521")
    time_axis.tick_params(axis="y", labelcolor="#1f77b4")
    throughput_axis.tick_params(axis="y", labelcolor="#e52521")
    time_axis.grid(alpha=0.3)
    time_axis.set_title(title)
    time_axis.legend(lines, [line.get_label() for line in lines], loc="upper center")
    figure.tight_layout()

    plot_path = run_dir / filename
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def _run_controller(args: argparse.Namespace) -> None:
    if args.resume_results is not None:
        results_path = args.resume_results.resolve()
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        run_dir = results_path.parent
        metadata = payload["metadata"]
        if metadata.get("benchmark_schema_version") != 4:
            raise ValueError("Older results cannot be resumed with batch-size measurements.")
        for key in ("dataset", "checkpoint", "model_name", "limit", "inference_graph_mode",
                    "encoding_storage_mode", "encoding_pack_max_files", "inference_batch_sizes",
                    "inference_memory_prefetch_depth", "inference_memory_workers"):
            value = getattr(args, key)
            if isinstance(value, pathlib.Path):
                value = str(value.resolve())
            if metadata.get(key) != value:
                raise ValueError(f"Cannot resume with a different {key}.")
        points = [ScalingPoint(**point) for point in payload.get("results", [])]
    else:
        run_name = datetime.now(timezone.utc).strftime("gpu_embed_shape_batch_%Y%m%dT%H%M%SZ")
        run_dir = args.output_dir.resolve() / run_name
        results_path = run_dir / "results.json"
        metadata = {
            "benchmark_schema_version": 4,
            "dataset": str(args.dataset.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "model_name": args.model_name,
            "device": "cuda",
            "workers": args.workers,
            "limit": args.limit,
            "inference_graph_mode": args.inference_graph_mode,
            "encoding_storage_mode": args.encoding_storage_mode,
            "encoding_pack_max_files": args.encoding_pack_max_files,
            "inference_batch_sizes": args.inference_batch_sizes,
            "inference_memory_prefetch_depth": args.inference_memory_prefetch_depth,
            "inference_memory_workers": args.inference_memory_workers,
            "memory_sample_interval_seconds": 0.1,
            "timing_policy": "Wall time around embed_shape_batch plus CUDA synchronization; includes API cleanup and worker setup, excludes parent model loading and output serialization.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        points = []

    completed_points = {(point.workers, point.batch_size) for point in points}
    log_dir = run_dir / "logs"
    error_dir = run_dir / "errors"
    point_dir = run_dir / ".points"
    log_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)

    for workers, batch_size in _benchmark_points(args.workers, args.inference_batch_sizes):
        if (workers, batch_size) in completed_points:
            print(f"skip completed workers={workers} batch_size={batch_size}", flush=True)
            continue

        point_name = f"workers_{workers}_batch_{batch_size}"
        point_result = point_dir / f"{point_name}.json"
        point_errors = error_dir / f"{point_name}.json"
        command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--dataset", str(args.dataset.resolve()),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--model-name", args.model_name,
            "--workers", str(workers),
            "--inference-graph-mode", args.inference_graph_mode,
            "--encoding-storage-mode", args.encoding_storage_mode,
            "--encoding-pack-max-files", str(args.encoding_pack_max_files),
            "--inference-batch-sizes", str(batch_size),
            "--inference-memory-prefetch-depth", str(args.inference_memory_prefetch_depth),
            "--inference-memory-workers", str(args.inference_memory_workers),
            "--point-workers", str(workers),
            "--point-batch-size", str(batch_size),
            "--point-result", str(point_result),
            "--point-errors", str(point_errors),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])

        log_path = log_dir / f"{point_name}.log"
        print(
            f"run embed_shape_batch workers={workers} batch_size={batch_size}; log={log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        if process.returncode == 0 and point_result.is_file():
            point = ScalingPoint(**json.loads(point_result.read_text(encoding="utf-8")))
        else:
            error = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            point = ScalingPoint(
                workers=workers,
                batch_size=batch_size,
                requested_files=0,
                embedded_bodies=0,
                successful_files=0,
                failed_files=0,
                seconds=None,
                input_files_per_second=None,
                successful_files_per_second=None,
                status="failed",
                error=error or f"Child exited with code {process.returncode}",
            )
        points.append(point)
        completed_points.add((workers, batch_size))
        _write_results(results_path, metadata, points)
        print(
            f"recorded workers={workers} batch_size={batch_size} status={point.status} "
            f"time={point.seconds or 0.0:.1f}s files/s={point.input_files_per_second or 0.0:.2f}",
            flush=True,
        )

    successful = [point for point in points if point.status == "ok"]
    if not successful:
        raise RuntimeError(f"No successful points. See logs under {log_dir}")

    requested_count = successful[0].requested_files
    best_input = max(successful, key=lambda point: point.input_files_per_second or 0.0)
    best_successful = max(successful, key=lambda point: point.successful_files_per_second or 0.0)
    metadata["best_input_throughput"] = {
        "workers": best_input.workers, "batch_size": best_input.batch_size,
    }
    metadata["best_successful_throughput"] = {
        "workers": best_successful.workers, "batch_size": best_successful.batch_size,
    }
    input_plot = _write_toshi_plot(
        run_dir,
        points,
        "input_files_per_second",
        "gpu_embed_shape_batch_input_scaling.png",
        f"GPU embed_shape_batch - inference batch-size scaling (n={requested_count:,})",
    )
    successful_plot = _write_toshi_plot(
        run_dir,
        points,
        "successful_files_per_second",
        "gpu_embed_shape_batch_successful_scaling.png",
        "GPU embed_shape_batch - successful files scaling",
    )
    _write_results(results_path, metadata, points)
    print(f"Results: {results_path}")
    print(f"Input plot: {input_plot}")
    print(f"Successful plot: {successful_plot}")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if args.point_workers is not None:
        if args.point_result is None or args.point_batch_size is None:
            raise ValueError("--point-result and --point-batch-size are required with --point-workers.")
        _run_point(args)
        return
    _run_controller(args)


if __name__ == "__main__":
    main()