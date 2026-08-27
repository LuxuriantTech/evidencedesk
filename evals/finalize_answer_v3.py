"""Create one fail-closed, attested final runtime for answer/extraction v3.

This runner is deliberately separate from the bounded strategy comparison: it
accepts only the pre-registered winner, always performs three calibration and
three selection repetitions, and never resumes or overwrites an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from evals.freeze import (
    _development_target_gate,
    _validate_comparison_raw,
    _validate_strategy_comparison,
)
from evals.recalculate import recalculate_metrics
from evals.runner import _engine_fingerprint
from evals.schema_versions import RECALCULATED_EVALUATION_SCHEMA
from evals.strategy_benchmark_v3 import (
    CORPUS_PATH,
    PARTITION_MANIFESTS,
    PLAN_PATH,
    QUALITY_METRICS,
    _aggregate,
    _core_decision_digest,
    _select_configuration,
    run_once,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_PATH = (
    ROOT / "artifacts/evaluations/development_v3/strategy_v3/comparison.json"
)


class FinalizationError(RuntimeError):
    """The frozen development runtime cannot be safely created."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"cannot read comparison artifact: {path}") from exc
    if not isinstance(value, dict):
        raise FinalizationError("comparison artifact must be a JSON object")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _selected_configuration(comparison: dict[str, Any]) -> dict[str, Any]:
    if comparison.get("schema_version") != "evidencedesk-strategy-comparison-v3":
        raise FinalizationError("comparison artifact has an unsupported schema")
    entries = comparison.get("comparison")
    if not isinstance(entries, dict) or not entries:
        raise FinalizationError("comparison artifact has no configurations")
    try:
        derived = _select_configuration(entries)
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalizationError(
            "comparison cannot be evaluated by the pre-registered rule"
        ) from exc
    declared = comparison.get("selected_configuration")
    if not isinstance(derived, str) or declared != derived:
        raise FinalizationError(
            "comparison selected configuration differs from pre-registered selection"
        )
    selected = entries.get(derived)
    if not isinstance(selected, dict) or not isinstance(selected.get("configuration"), dict):
        raise FinalizationError("selected configuration is invalid")
    configuration = selected["configuration"]
    if configuration.get("id") != derived:
        raise FinalizationError("selected configuration id does not match comparison key")
    return cast(dict[str, Any], configuration)


def _validate_comparison_provenance(
    comparison: dict[str, Any], *, engine_fingerprint: str
) -> None:
    expected = {
        "engine_fingerprint": engine_fingerprint,
        "experiment_plan_sha256": _sha256(PLAN_PATH),
        "corpus_sha256": _sha256(CORPUS_PATH),
        "partition_manifest_sha256": {
            partition: _sha256(path) for partition, path in PARTITION_MANIFESTS.items()
        },
        "repetitions": 3,
    }
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise FinalizationError(f"comparison {key} does not match the current engine run")


def _runtime_aggregate(paths: list[Path]) -> dict[str, Any]:
    try:
        rooted_paths = [
            path.resolve() if path.is_absolute() else (ROOT / path).resolve()
            for path in paths
        ]
        return _aggregate(rooted_paths)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise FinalizationError("final runtime aggregate could not be calculated") from exc


def _recalculate(raw_path: Path, output_path: Path) -> dict[str, Any]:
    raw = _load(raw_path)
    recalculated = {
        "schema_version": RECALCULATED_EVALUATION_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "raw_artifact_sha256": _sha256(raw_path),
        **recalculate_metrics(raw),
    }
    _write_json(output_path, recalculated)
    return recalculated


def finalize(*, comparison_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run and attest the sole final runtime; fail before writes when invalid."""

    if output_dir.exists():
        raise FinalizationError(f"refusing to overwrite evaluation output: {output_dir}")
    engine_fingerprint = _engine_fingerprint()
    comparison = _load(comparison_path)
    _validate_comparison_provenance(
        comparison,
        engine_fingerprint=engine_fingerprint,
    )
    _validate_strategy_comparison(
        comparison,
        root=ROOT,
        expected_fingerprint=engine_fingerprint,
    )
    configuration = _selected_configuration(comparison)

    output_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, list[Path]] = {"calibration": [], "selection": []}
    recalculations: dict[str, dict[str, Any]] = {}
    try:
        for partition in ("calibration", "selection"):
            for repetition in (1, 2, 3):
                raw = run_once(configuration, partition, repetition)
                if raw.get("engine_fingerprint") != engine_fingerprint:
                    raise FinalizationError("run engine fingerprint differs from frozen runtime")
                _validate_comparison_raw(
                    raw,
                    expected_fingerprint=engine_fingerprint,
                    expected_manifest_sha256=_sha256(PARTITION_MANIFESTS[partition]),
                    expected_corpus_sha256=_sha256(CORPUS_PATH),
                    expected_configuration=configuration,
                    expected_partition=partition,
                    expected_repetition=repetition,
                )
                raw_path = output_dir / f"{partition}-r{repetition}.json"
                _write_json(raw_path, raw)
                paths[partition].append(raw_path)
                recalculation_path = output_dir / f"{partition}-r{repetition}-recalculated.json"
                recalculations[f"{partition}_r{repetition}"] = _recalculate(
                    raw_path, recalculation_path
                )

        aggregates = {
            partition: _runtime_aggregate(partition_paths)
            for partition, partition_paths in paths.items()
        }
        source_entries = cast(dict[str, Any], comparison["comparison"])
        selected_id = str(configuration["id"])
        source = cast(dict[str, Any], source_entries[selected_id])
        core_reproduction: dict[str, Any] = {}
        for partition in ("calibration", "selection"):
            original = source.get(partition)
            if not isinstance(original, dict) or not isinstance(original.get("artifacts"), list):
                raise FinalizationError(f"selected comparison lacks {partition} artifacts")
            original_paths = [ROOT / item["path"] for item in original["artifacts"]]
            frozen_digests = [_core_decision_digest(_load(path)) for path in paths[partition]]
            original_digests = [_core_decision_digest(_load(path)) for path in original_paths]
            core_reproduction[partition] = {
                "all_match": frozen_digests == original_digests,
                "frozen_runtime_core_decision_sha256": frozen_digests,
                "original_core_decision_sha256": original_digests,
            }
        if not all(value["all_match"] for value in core_reproduction.values()):
            raise FinalizationError("final runtime decisions do not reproduce selected comparison")

        summary = {
            "schema_version": "evidencedesk-development-v3-frozen-runtime-summary-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "comparison_path": _artifact_path(comparison_path),
            "comparison_sha256": _sha256(comparison_path),
            "engine_fingerprint": engine_fingerprint,
            "selected_configuration": configuration,
            "repetitions": 3,
            "quality_metrics": list(QUALITY_METRICS),
            "aggregates": aggregates,
            "core_decision_reproduction": core_reproduction,
            "development_target_gate": _development_target_gate(
                aggregates["selection"]["quality"]
            ),
            "recalculation": {
                f"{partition}_r1_path": _artifact_path(
                    output_dir / f"{partition}-r1-recalculated.json"
                )
                for partition in ("calibration", "selection")
            }
            | {
                f"{partition}_r1_sha256": _sha256(
                    output_dir / f"{partition}-r1-recalculated.json"
                )
                for partition in ("calibration", "selection")
            },
        }
        _write_json(output_dir / "summary.json", summary)
        return summary
    except Exception:
        # Artifacts from a failed attempt remain inspectable but are never reused:
        # the output directory itself blocks a silent retry or overwrite.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = finalize(comparison_path=args.comparison, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "summary": _artifact_path(args.output_dir / "summary.json"),
                "gate": summary["development_target_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
