"""Tests for the embedding batch-size benchmark controller."""

import csv
import json
from pathlib import Path

from embeddings_pipeline.scripts.benchmark_embed_shape_batch_scaling import (
    ScalingPoint,
    _benchmark_points,
    _write_results,
)


def test_benchmark_points_build_cartesian_matrix() -> None:
    assert _benchmark_points([20], [32, 64, 128, 256]) == [
        (20, 32),
        (20, 64),
        (20, 128),
        (20, 256),
    ]


def test_write_results_preserves_batch_size_and_phase_timings(tmp_path: Path) -> None:
    point = ScalingPoint(
        workers=20,
        batch_size=64,
        requested_files=100,
        embedded_bodies=99,
        successful_files=99,
        failed_files=1,
        seconds=10.0,
        input_files_per_second=10.0,
        successful_files_per_second=9.9,
        status="ok",
        error=None,
        phase_seconds={
            "total": 9.9,
            "encoding": 8.0,
            "preparation": 1.5,
            "inference": 0.8,
            "cleanup": 0.1,
        },
    )
    results_path = tmp_path / "results.json"

    _write_results(results_path, {"benchmark_schema_version": 4}, [point])

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["batch_size"] == 64
    assert payload["results"][0]["phase_seconds"]["inference"] == 0.8
    with (tmp_path / "embed_shape_batch_scaling.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["batch_size"] == "64"
    assert "'inference': 0.8" in row["phase_seconds"]
