"""用一条 2024—2025 开发样本跑通 A1→A2→A4→A5，不读取真值账本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copper_mas.agents.runtime import fixed_offline_plan, run_offline_graph  # noqa: E402
from copper_mas.contracts.cards import (  # noqa: E402
    AsOfAdmissionCardV2,
    ForecastRequestV2,
    ObservationCardV2,
    ObservationItemV2,
)
from copper_mas.models.predictors import load_selected_predictor  # noqa: E402


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_list(value: object) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    parsed = json.loads(str(value))
    return tuple(str(item) for item in parsed)


def _json_counts(value: object) -> dict[str, int]:
    parsed = json.loads(str(value))
    return {str(key): int(item) for key, item in parsed.items()}


def run_smoke(input_dir: Path, output_dir: Path) -> dict[str, object]:
    sources = {
        "features": input_dir / "core_feature_matrix_v2.csv",
        "admission": input_dir / "as_of_admission_card_core_v2.csv",
        "index": input_dir / "training_evaluation_index_v2.csv",
        "selected_model": PROJECT_ROOT / "models/selected_model_manifest_v1.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P3 smoke 输入缺失: {missing}")

    index = pd.read_csv(sources["index"], low_memory=False)
    eligible = index.loc[index["tier1_core_primary_prospective"].astype(str).str.lower() == "true"]
    if eligible.empty:
        raise ValueError("没有 tier1_core_primary_prospective 样本")
    origin_event_id = str(eligible.sort_values("origin_recorded_at").iloc[0]["origin_event_id"])

    feature_frame = pd.read_csv(sources["features"], low_memory=False)
    feature_match = feature_frame.loc[
        feature_frame["origin_event_id"].astype(str) == origin_event_id
    ]
    admission_frame = pd.read_csv(sources["admission"], low_memory=False)
    admission_match = admission_frame.loc[
        admission_frame["origin_event_id"].astype(str) == origin_event_id
    ]
    if len(feature_match) != 1 or len(admission_match) != 1:
        raise ValueError("smoke 样本的特征或准入卡不唯一")
    feature_row = feature_match.iloc[0].to_dict()
    raw = admission_match.iloc[0]
    decision_at = pd.Timestamp(raw["decision_at"]).to_pydatetime()
    admission = AsOfAdmissionCardV2(
        origin_event_id=origin_event_id,
        decision_at=decision_at,
        feature_cutoff_at=pd.Timestamp(raw["feature_cutoff_at"]).to_pydatetime(),
        admission_status=str(raw["admission_status"]),
        reason_codes=_json_list(raw["reason_codes"]),
        feature_group_counts=_json_counts(raw["feature_group_counts"]),
        missing_feature_groups=_json_list(raw["missing_feature_groups"]),
        source_quality_warnings=_json_list(raw["source_quality_warnings"]),
    )
    observations = ObservationCardV2(
        origin_event_id=origin_event_id,
        decision_at=decision_at,
        observations=(
            ObservationItemV2(
                canonical_tag="origin_cu_g_l",
                value=float(feature_row["origin_cu_g_l"]),
                unit="g/L",
                event_at=decision_at,
                available_at=decision_at,
                matched_anchor_at=decision_at,
                age_hours=0,
                missing=False,
                quality_code="KNOWN_CURRENT_RESULT",
            ),
            ObservationItemV2(
                canonical_tag="origin_as_mg_l",
                value=float(feature_row["origin_as_mg_l"]),
                unit="mg/L",
                event_at=decision_at,
                available_at=decision_at,
                matched_anchor_at=decision_at,
                age_hours=0,
                missing=False,
                quality_code="KNOWN_CURRENT_RESULT",
            ),
        ),
    )
    request = ForecastRequestV2(
        request_id=f"smoke::{origin_event_id}",
        admission=admission,
        observations=observations,
    )
    predictor = load_selected_predictor(sources["selected_model"])
    plan = fixed_offline_plan(
        run_id="P3_OFFLINE_SMOKE_V1",
        numeric_predictor_ref=predictor.model_id,
        post_freeze=False,
    )
    result = run_offline_graph(
        plan=plan,
        request=request,
        feature_row=feature_row,
        predict=predictor,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "workflow_plan_v1.json": result.plan,
        "process_mode_card_v2.json": result.process_mode,
        "prediction_card_v2.json": result.prediction,
        "audit_card_v2.json": result.audit,
        "resource_ledger_v1.json": result.resources,
    }
    for filename, card in outputs.items():
        (output_dir / filename).write_text(
            json.dumps(card.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest: dict[str, object] = {
        "run_id": plan.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_scope": "single_2024_2025_development_smoke",
        "outcome_ledger_read": False,
        "external_2026_read": False,
        "llm_calls_made": result.resources.total_calls,
        "graph_order": list(plan.ordered_nodes),
        "numeric_predictor_ref": plan.numeric_predictor_ref,
        "audit_passed": result.audit.passed,
        "source_sha256": {path.name: _hash(path) for path in sources.values()},
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data/development_2024_2025",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/v1_offline_smoke_v1",
    )
    args = parser.parse_args()
    manifest = run_smoke(args.input_dir, args.output_dir)
    safe_summary = {
        "run_id": manifest["run_id"],
        "audit_passed": manifest["audit_passed"],
        "llm_calls_made": manifest["llm_calls_made"],
        "external_2026_read": manifest["external_2026_read"],
    }
    print(json.dumps(safe_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
