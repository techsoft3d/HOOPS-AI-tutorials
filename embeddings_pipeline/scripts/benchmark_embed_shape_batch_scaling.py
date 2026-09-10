"""Benchmark end-to-end GPU embed_shape_batch scaling and create Toshi-style plots."""

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


CAD_FORMATS = [".stp", ".step", ".iges", ".igs"]
DEFAULT_MODEL_NAME = "HOOPS Embeddings SIGNAL preview"
DEFAULT_WORKERS = [4, 8, 12, 16, 20]
TUTORIAL_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    TUTORIAL_ROOT
    / "packages"
    / "trained_ml_models"
    / "ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt"
)


@dataclass(frozen=True)
class ScalingPoint:
    """One end-to-end embed_shape_batch measurement."""

    workers: int
    requested_files: int
    embedded_bodies: int
    successful_files: int
    failed_files: int
    seconds: float | None
    input_files_per_second: float | None
    successful_files_per_second: float | None
    status: str
    error: str | None
    pipeline_metrics: dict[str, float] | None = None
    inference_metrics: dict[str, float | int] | None = None
    encoding_storage_mode: str = "directory"
    encoding_pack_max_bodies: int = 32
    encoding_pack_max_bytes: int = 64 * 1024**2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full GPU embed_shape_batch scaling with default specifications."
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=os.environ.get("PATH_DATASET_TMCAD"),
        help="TMCAD root. Defaults to PATH_DATASET_TMCAD.",
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inference-graph-mode", choices=("file", "memory"), default="memory")
    parser.add_argument("--encoding-storage-mode", choices=("directory", "packed"), default="packed")
    parser.add_argument("--encoding-pack-max-bodies", type=int, default=32)
    parser.add_argument("--encoding-pack-max-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--inference-memory-prefetch-depth", type=int, default=2)
    parser.add_argument("--inference-memory-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=TUTORIAL_ROOT / "embeddings_pipeline" / "out" / "embed_shape_batch_scaling",
    )
    parser.add_argument("--resume-results", type=pathlib.Path, default=None)
    parser.add_argument("--point-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--point-result", type=pathlib.Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--point-errors", type=pathlib.Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


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
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive.")
    if args.inference_memory_prefetch_depth < 0:
        raise ValueError("--inference-memory-prefetch-depth must be non-negative.")
    if args.inference_memory_workers <= 0:
        raise ValueError("--inference-memory-workers must be positive.")
    if args.encoding_pack_max_bodies < 0 or args.encoding_pack_max_bytes < 0:
        raise ValueError("Pack rotation targets must be non-negative.")
    if args.encoding_storage_mode == "packed" and args.inference_graph_mode != "memory":
        raise ValueError("Packed benchmarks require memory inference.")


def _load_cad_files(dataset: pathlib.Path, limit: int | None) -> list[str]:
    from hoops_ai.storage import CADFileRetriever, LocalStorageProvider

    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=dataset),
        formats=CAD_FORMATS,
    )
    cad_files = sorted(str(path) for path in retriever.get_file_list())
    if limit is not None:
        cad_files = cad_files[:limit]
    if not cad_files:
        raise RuntimeError(f"No supported CAD files found under {dataset}")
    return cad_files


def _run_point(args: argparse.Namespace) -> None:
    import torch

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
    torch.cuda.synchronize()
    started = time.perf_counter()
    batch = embedder.embed_shape_batch(
        cad_files,
        num_workers=args.point_workers,
        show_progress=True,
        specifications={
            "inference_graph_mode": args.inference_graph_mode,
            "encoding_storage_mode": args.encoding_storage_mode,
            "encoding_pack_max_bodies": args.encoding_pack_max_bodies,
            "encoding_pack_max_bytes": args.encoding_pack_max_bytes,
            "inference_batch_size": args.inference_batch_size,
            "inference_memory_prefetch_depth": args.inference_memory_prefetch_depth,
            "inference_memory_workers": args.inference_memory_workers,
            "collect_inference_metrics": True,
        },
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    failed_files = int(batch.metadata.get("failed_count", 0))
    successful_files = max(0, len(cad_files) - failed_files)
    point = ScalingPoint(
        workers=args.point_workers,
        requested_files=len(cad_files),
        embedded_bodies=len(batch.ids),
        successful_files=successful_files,
        failed_files=failed_files,
        seconds=seconds,
        input_files_per_second=len(cad_files) / seconds,
        successful_files_per_second=successful_files / seconds,
        status="ok",
        error=None,
        pipeline_metrics=batch.metadata.get("pipeline_metrics"),
        inference_metrics=batch.metadata.get("inference_metrics"),
        encoding_storage_mode=args.encoding_storage_mode,
        encoding_pack_max_bodies=args.encoding_pack_max_bodies,
        encoding_pack_max_bytes=args.encoding_pack_max_bytes,
    )
    args.point_result.parent.mkdir(parents=True, exist_ok=True)
    args.point_result.write_text(json.dumps(asdict(point), indent=2), encoding="utf-8")
    if args.point_errors is not None:
        errors = batch.metadata.get("errors", [])
        args.point_errors.parent.mkdir(parents=True, exist_ok=True)
        args.point_errors.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(
        f"workers={point.workers} time={point.seconds:.1f}s "
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

    successful = sorted((point for point in points if point.status == "ok"), key=lambda point: point.workers)
    workers = [point.workers for point in successful]
    seconds = [point.seconds for point in successful]
    throughput = [getattr(point, throughput_field) for point in successful]

    figure, time_axis = plt.subplots(figsize=(8, 4.5))
    throughput_axis = time_axis.twinx()
    time_line = time_axis.plot(workers, seconds, "o-", color="#1f77b4", label="time (s)")
    throughput_line = throughput_axis.plot(
        workers,
        throughput,
        "s--",
        color="#e52521",
        label="files/s",
    )

    best_index = max(range(len(throughput)), key=throughput.__getitem__)
    time_axis.axvline(workers[best_index], color="#777777", linestyle=":", alpha=0.8)
    time_axis.set_xticks(workers)
    time_axis.set_xlabel("num_workers")
    time_axis.set_ylabel("time (s)", color="#1f77b4")
    throughput_axis.set_ylabel("files/s", color="#e52521")
    time_axis.tick_params(axis="y", labelcolor="#1f77b4")
    throughput_axis.tick_params(axis="y", labelcolor="#e52521")
    time_axis.grid(alpha=0.3)
    time_axis.set_title(title)
    lines = time_line + throughput_line
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
        for key, default in (("encoding_storage_mode", "directory"),
                     ("encoding_pack_max_bodies", 32),
                     ("encoding_pack_max_bytes", 64 * 1024**2)):
            if metadata.get(key, default) != getattr(args, key):
                raise ValueError(f"Cannot resume with a different {key}.")
        points = [ScalingPoint(**point) for point in payload.get("results", [])]
    else:
        run_name = datetime.now(timezone.utc).strftime("gpu_embed_shape_batch_%Y%m%dT%H%M%SZ")
        run_dir = args.output_dir.resolve() / run_name
        results_path = run_dir / "results.json"
        metadata = {
            "dataset": str(args.dataset.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "model_name": args.model_name,
            "device": "cuda",
            "workers": args.workers,
            "limit": args.limit,
            "inference_graph_mode": args.inference_graph_mode,
            "encoding_storage_mode": args.encoding_storage_mode,
            "encoding_pack_max_bodies": args.encoding_pack_max_bodies,
            "encoding_pack_max_bytes": args.encoding_pack_max_bytes,
            "inference_batch_size": args.inference_batch_size,
            "inference_memory_prefetch_depth": args.inference_memory_prefetch_depth,
            "inference_memory_workers": args.inference_memory_workers,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        points = []

    completed_workers = {point.workers for point in points}
    log_dir = run_dir / "logs"
    error_dir = run_dir / "errors"
    point_dir = run_dir / ".points"
    log_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)

    for workers in args.workers:
        if workers in completed_workers:
            print(f"skip completed workers={workers}", flush=True)
            continue

        point_result = point_dir / f"workers_{workers}.json"
        point_errors = error_dir / f"workers_{workers}.json"
        command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--dataset", str(args.dataset.resolve()),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--model-name", args.model_name,
            "--workers", str(workers),
            "--inference-graph-mode", args.inference_graph_mode,
            "--encoding-storage-mode", args.encoding_storage_mode,
            "--encoding-pack-max-bodies", str(args.encoding_pack_max_bodies),
            "--encoding-pack-max-bytes", str(args.encoding_pack_max_bytes),
            "--inference-batch-size", str(args.inference_batch_size),
            "--inference-memory-prefetch-depth", str(args.inference_memory_prefetch_depth),
            "--inference-memory-workers", str(args.inference_memory_workers),
            "--point-workers", str(workers),
            "--point-result", str(point_result),
            "--point-errors", str(point_errors),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])

        log_path = log_dir / f"workers_{workers}.log"
        print(f"run embed_shape_batch workers={workers}; log={log_path}", flush=True)
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
        completed_workers.add(workers)
        _write_results(results_path, metadata, points)
        print(
            f"recorded workers={workers} status={point.status} "
            f"time={point.seconds or 0.0:.1f}s files/s={point.input_files_per_second or 0.0:.2f}",
            flush=True,
        )

    successful = [point for point in points if point.status == "ok"]
    if not successful:
        raise RuntimeError(f"No successful points. See logs under {log_dir}")

    requested_count = successful[0].requested_files
    metadata["best_input_throughput_workers"] = max(
        successful, key=lambda point: point.input_files_per_second or 0.0
    ).workers
    metadata["best_successful_throughput_workers"] = max(
        successful, key=lambda point: point.successful_files_per_second or 0.0
    ).workers
    input_plot = _write_toshi_plot(
        run_dir,
        points,
        "input_files_per_second",
        "gpu_embed_shape_batch_input_scaling.png",
        f"GPU embed_shape_batch - num_workers scaling (n={requested_count:,})",
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
        if args.point_result is None:
            raise ValueError("--point-result is required with --point-workers.")
        _run_point(args)
        return
    _run_controller(args)


if __name__ == "__main__":
    main()