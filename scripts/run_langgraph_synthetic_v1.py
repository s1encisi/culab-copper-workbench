"""运行一条不含工厂数据的 V1/LangGraph V2 合成一致性样例。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copper_langgraph_v2.bridge import (  # noqa: E402
    AsOfAdmissionCardV2,
    ForecastRequestV2,
    ObservationCardV2,
    ObservationItemV2,
    fixed_offline_plan,
)
from copper_langgraph_v2.graph import (  # noqa: E402
    LangGraphRunner,
    persistence_predictor,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/langgraph_synthetic_v1",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    decision = datetime(2025, 1, 1, 8)
    admission = AsOfAdmissionCardV2(
        origin_event_id="synthetic-event",
        decision_at=decision,
        feature_cutoff_at=decision,
        admission_status="ADMITTED",
    )
    request = ForecastRequestV2(
        request_id="synthetic-request",
        admission=admission,
        observations=ObservationCardV2(
            origin_event_id="synthetic-event",
            decision_at=decision,
            observations=(
                ObservationItemV2(
                    canonical_tag="origin_cu_g_l",
                    value=5.25,
                    available_at=decision,
                    event_at=decision,
                    matched_anchor_at=decision,
                    age_hours=0,
                    missing=False,
                ),
            ),
        ),
    )
    plan = fixed_offline_plan(
        run_id="synthetic-v2-demo",
        numeric_predictor_ref="PERSISTENCE_CURRENT_RESULT_V1",
    )
    features = {
        "origin_cu_g_l": 5.25,
        "origin_as_mg_l": 1200.0,
        "phase1_stage12_current_ka__t_minus_0h": 1.0,
        "phase2_stage12_current_ka__t_minus_0h": 0.0,
        "stage3_current_a__t_minus_0h": 1.0,
        "stage4_current_a__t_minus_0h": 0.0,
    }
    checkpoint = output_dir / "checkpoints.sqlite"
    if checkpoint.exists():
        checkpoint.unlink()
    with LangGraphRunner(checkpoint_path=checkpoint, predict=persistence_predictor) as runner:
        result = runner.invoke(plan=plan, request=request, feature_row=features)
        replay = runner.invoke(plan=plan, request=request, feature_row=features)
    output = {
        "trace": result["trace"],
        "prediction": result["prediction"],
        "audit_passed": result["audit"]["passed"],
        "blocked": result["blocked"],
        "llm_calls": result["resources"]["total_calls"],
        "estimated_cost_cny": result["resources"]["total_estimated_cost_cny"],
        "idempotent_replay": replay["idempotent_replay"],
        "checkpoint_path": str(checkpoint),
        "uses_2026_data": False,
        "network_calls": 0,
    }
    target = output_dir / "langgraph_v2_demo_v1.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
