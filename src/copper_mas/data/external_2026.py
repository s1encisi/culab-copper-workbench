"""2026 外部时间留出集的锁定规范化、事件配对与隔离账本。

本模块有意不提供任何模型拟合、阈值搜索或指标计算功能。2026 Cu/As
只在内存中用于构造“当前已知结果”输入和独立的下一事件真值账本；所有
命令行输出均只能包含行数、时间覆盖、质量门数量和哈希。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from copper_mas.contracts.cards import (
    CONTRACT_ID,
    AsOfAdmissionCardV2,
    EvaluationEligibilityLedgerV2,
    OutcomeLedgerV2,
)
from copper_mas.data.asof import backward_asof_snapshots, summarize_feature_anchors
from copper_mas.data.assembly import assemble_final_admission_cards
from copper_mas.data.leakage import assert_no_future_information


SCHEMA_VERSION = "2.0"
EXTERNAL_YEAR = 2026
EXPECTED_EVENT_COUNT = 683
FROZEN_COMMON_CUTOFF = pd.Timestamp("2026-07-17 08:53:00")
ASSUMED_SAMPLE_DELAY = pd.Timedelta(hours=2)

ELECTRO_MEASUREMENT_COLUMNS = [
    "west_current_ka",
    "west_voltage_v",
    "west_avg_cell_voltage_v",
    "west_active_cells",
    "east_current_ka",
    "east_voltage_v",
    "east_avg_cell_voltage_v",
    "east_active_cells",
    "stage12_decopper_current_ka",
    "stage12_decopper_voltage_v",
    "stage12_decopper_avg_cell_voltage_v",
    "stage12_decopper_active_cells",
]

ELECTRO_CORE_COLUMNS = [
    "west_current_ka",
    "west_voltage_v",
    "west_active_cells",
    "east_current_ka",
    "east_voltage_v",
    "east_active_cells",
    "stage12_decopper_current_ka",
    "stage12_decopper_voltage_v",
    "stage12_decopper_active_cells",
]

STAGE34_CORE_COLUMNS = [
    "feed_flow_m3_h",
    "stage3_temperature_c",
    "stage3_current_a",
    "stage3_voltage_v",
    "stage3_flow_m3_h",
    "stage4_temperature_c",
    "stage4_current_a",
    "stage4_voltage_v",
    "stage4_flow_m3_h",
]


@dataclass(frozen=True)
class ExternalEventArtifactsV2:
    event_metadata: pd.DataFrame
    origin_known_results: pd.DataFrame
    event_pair_index: pd.DataFrame
    as_of_admission_cards: pd.DataFrame
    evaluation_ledger: pd.DataFrame
    outcome_ledger: pd.DataFrame
    partition_manifest: pd.DataFrame


@dataclass(frozen=True)
class ObservationArtifactsV2:
    snapshot_long: pd.DataFrame
    feature_matrix: pd.DataFrame
    coverage: pd.DataFrame
    final_admission_cards: pd.DataFrame
    external_evaluation_index: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_pair_id(origin_event_id: str, target_event_id: str | None) -> str:
    target_token = target_event_id or "FINAL_AUDIT_NO_OUTCOME"
    payload = f"{CONTRACT_ID}|{origin_event_id}|{target_token}".encode("utf-8")
    return "pair_" + hashlib.sha256(payload).hexdigest()[:20]


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ", timespec="seconds")


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_legacy_parser(workspace_root: Path) -> ModuleType:
    """只读加载既有解析器，并禁止其写回旧目录。"""

    parser_path = workspace_root / "03_分析结果/可复现脚本/rebuild_factory_data.py"
    if not parser_path.is_file():
        raise FileNotFoundError(f"既有解析器不存在: {parser_path}")
    spec = importlib.util.spec_from_file_location("factory_legacy_parser_readonly", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载既有解析器: {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STUDY_YEARS = {EXTERNAL_YEAR}
    module.RAW = workspace_root / "01_原始数据/人工初步整理数据"
    # 所调用的旧函数中只有三四段解析器会主动写 CSV；替换为 no-op，确保旧目录不变。
    module.write_csv = lambda _frame, _path: None
    return module


def parse_target_events_2026(
    legacy: ModuleType,
    *,
    enforce_expected_contract: bool = True,
) -> pd.DataFrame:
    """解析原液罐 2026 录入事件；仅返回 Cu/As 与必要血缘，不做统计。"""

    path = legacy.RAW / "2026电解二系统-液体样.xlsx"
    long = legacy.parse_two_row_lab(
        path,
        "废铜液原液罐（二期）-9",
        "WASTE_COPPER_LIQUOR_RAW_TANK",
        "SHARED_DOWNSTREAM",
        ["sample_id", "sample_name", "crew", "production_date", "time"],
        5,
        110,
    )
    long = long.loc[long["analyte"].isin(["Cu", "As"])].copy()
    if long.empty:
        raise ValueError("2026 原液罐目标表未解析出 Cu/As")

    canonical_rows: list[pd.Series] = []
    for (_, _), group in long.groupby(
        ["liquid_sample_key", "analyte"], sort=False, dropna=False
    ):
        representative = group.sort_values(
            ["source_rank", "source_file", "source_row"], kind="mergesort"
        ).iloc[-1].copy()
        values = group[["canonical_value", "canonical_unit"]].drop_duplicates()
        times = group["sample_datetime"].drop_duplicates()
        conflict = len(values) > 1 or len(times) > 1
        representative["source_value_or_time_conflict_flag"] = conflict
        representative["analysis_eligible"] = not conflict
        if conflict:
            representative["canonical_value"] = np.nan
        canonical_rows.append(representative)
    canonical = pd.DataFrame(canonical_rows)

    metadata_columns = [
        "sample_id",
        "sample_name_raw",
        "sample_datetime",
        "event_time_parse_eligible",
        "crew_raw",
        "crew",
        "source_file",
        "source_sheet",
        "source_row",
    ]
    metadata = (
        canonical.sort_values("sample_datetime", kind="mergesort")
        .groupby("liquid_sample_key", as_index=False, dropna=False)[metadata_columns]
        .first()
    )
    quality = canonical.groupby("liquid_sample_key", as_index=False, dropna=False).agg(
        source_quality_eligible=("analysis_eligible", "all"),
        source_conflict=("source_value_or_time_conflict_flag", "any"),
    )
    values = canonical.pivot_table(
        index="liquid_sample_key",
        columns="analyte",
        values="canonical_value",
        aggfunc="first",
    ).reset_index()
    values.columns.name = None
    events = metadata.merge(quality, on="liquid_sample_key", validate="one_to_one").merge(
        values, on="liquid_sample_key", how="left", validate="one_to_one"
    )
    for analyte in ("Cu", "As"):
        if analyte not in events:
            events[analyte] = np.nan
    events = events.rename(
        columns={"sample_datetime": "recorded_at", "Cu": "current_cu", "As": "current_as"}
    )
    events["recorded_at"] = pd.to_datetime(events["recorded_at"], errors="raise")
    events = events.loc[events["recorded_at"].dt.year.eq(EXTERNAL_YEAR)].copy()
    events["available_at"] = events["recorded_at"]
    events["sample_at_assumed"] = events["recorded_at"] - ASSUMED_SAMPLE_DELAY
    events["event_id"] = [
        legacy.stable_id("target", key, timestamp)
        for key, timestamp in zip(events["liquid_sample_key"], events["recorded_at"])
    ]
    events = events.sort_values(["recorded_at", "event_id"], kind="mergesort").reset_index(drop=True)
    events["sequence_index"] = events.index.astype(int)

    if events["event_id"].duplicated().any():
        raise ValueError("2026 目标稳定事件 ID 不唯一")
    if events["recorded_at"].duplicated().any():
        raise ValueError("2026 目标 recorded_at 不唯一，不能严格相邻配对")
    if enforce_expected_contract:
        if len(events) != EXPECTED_EVENT_COUNT:
            raise ValueError(f"2026 目标事件数量合同不一致: {len(events)}")
        if events["recorded_at"].max() != FROZEN_COMMON_CUTOFF:
            raise ValueError("2026 目标最大录入时刻与冻结共同截止不一致")
        if events[["current_cu", "current_as"]].isna().any().any():
            raise ValueError("2026 目标 Cu/As 存在缺失，不能构造完整外部账本")
    return events


def trim_phase1_trailing_template_rows(
    frame: pd.DataFrame,
    *,
    measurement_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """只删除最后真实信号之后连续的全零/空模板；内部零值原样保留。"""

    columns = measurement_columns or ELECTRO_MEASUREMENT_COLUMNS
    if frame.empty:
        return frame.copy(), 0
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"一期模板裁剪缺少测量列: {sorted(missing)}")
    ordered = frame.sort_values(["source_row", "timestamp"], kind="mergesort").reset_index(drop=True)
    numeric = ordered[columns].apply(pd.to_numeric, errors="coerce")
    template = numeric.isna() | numeric.eq(0)
    all_zero_or_blank = template.all(axis=1)
    real_positions = np.flatnonzero((~all_zero_or_blank).to_numpy())
    if len(real_positions) == 0:
        raise ValueError("一期整流记录没有任何非零真实信号，拒绝把全表当模板删除")
    last_real = int(real_positions[-1])
    removed = len(ordered) - last_real - 1
    kept = ordered.iloc[: last_real + 1].copy()
    return kept, removed


def _canonicalize_electro(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cleaned: list[pd.DataFrame] = []
    for frame in frames:
        part = frame.copy()
        part.attrs.clear()
        cleaned.append(part)
    raw = pd.concat(cleaned, ignore_index=True).sort_values(
        ["system", "timestamp", "source_file", "source_row"], kind="mergesort"
    )
    canonical_rows: list[pd.Series] = []
    for (_, _), group in raw.groupby(["system", "timestamp"], sort=False, dropna=False):
        representative = group.iloc[0].copy()
        exact = len(group) > 1 and len(group[ELECTRO_MEASUREMENT_COLUMNS].drop_duplicates()) == 1
        conflict = len(group) > 1 and not exact
        representative["timestamp_source_record_count"] = len(group)
        representative["timestamp_exact_duplicate_flag"] = exact
        representative["timestamp_conflict_flag"] = conflict
        representative["source_record_resolution"] = (
            "CONFLICT_GROUP_VALUES_WITHHELD"
            if conflict
            else "EXACT_DUPLICATES_COLLAPSED"
            if exact
            else "SINGLE_SOURCE_ROW"
        )
        if conflict:
            representative[ELECTRO_MEASUREMENT_COLUMNS] = np.nan
            representative["source_row"] = np.nan
        canonical_rows.append(representative)
    frame = pd.DataFrame(canonical_rows).reset_index(drop=True)
    frame.insert(
        0,
        "operation_event_id",
        [
            "er_"
            + hashlib.sha256(f"{system}|{pd.Timestamp(ts).isoformat()}".encode("utf-8")).hexdigest()[:20]
            for system, ts in zip(frame["system"], frame["timestamp"])
        ],
    )

    for area in ("west", "east", "stage12_decopper"):
        current = pd.to_numeric(frame[f"{area}_current_ka"], errors="coerce")
        voltage = pd.to_numeric(frame[f"{area}_voltage_v"], errors="coerce")
        active = pd.to_numeric(frame[f"{area}_active_cells"], errors="coerce")
        source_avg = pd.to_numeric(frame[f"{area}_avg_cell_voltage_v"], errors="coerce")
        recomputed_avg = pd.Series(np.where(active > 0, voltage / active, np.nan), index=frame.index)
        avg_bad = (
            (source_avg.notna() & ~source_avg.between(0, 5))
            | (recomputed_avg.notna() & ~recomputed_avg.between(0, 5))
            | (source_avg.eq(0) & recomputed_avg.gt(0))
        )
        positive = pd.concat([current.gt(0), voltage.gt(0), active.gt(0)], axis=1).any(axis=1)
        nonpositive = pd.concat(
            [
                current.le(0) & current.notna(),
                voltage.le(0) & voltage.notna(),
                active.le(0) & active.notna(),
            ],
            axis=1,
        ).any(axis=1)
        frame[f"{area}_state_signal_conflict_flag"] = positive & nonpositive
        frame[f"{area}_implausible_flag"] = (
            (current.notna() & ~current.between(0, 100))
            | (voltage.notna() & ~voltage.between(0, 300))
            | (active.notna() & ~active.between(0, 1000))
            | avg_bad
        )

    implausible = [f"{area}_implausible_flag" for area in ("west", "east", "stage12_decopper")]
    conflicts = [
        f"{area}_state_signal_conflict_flag" for area in ("west", "east", "stage12_decopper")
    ]
    frame["any_implausible_value_flag"] = frame[implausible].any(axis=1)
    frame["any_state_signal_conflict_flag"] = frame[conflicts].any(axis=1)
    frame["analysis_eligible"] = ~(
        frame["timestamp_conflict_flag"]
        | frame["any_implausible_value_flag"]
        | frame["any_state_signal_conflict_flag"]
        | frame["time_grid_off_flag"]
    )
    return frame


def parse_electro_2026(
    legacy: ModuleType,
    *,
    cutoff: pd.Timestamp = FROZEN_COMMON_CUTOFF,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """解析一期/二期整流；一期尾部模板裁剪发生在截止过滤之前。"""

    phase1 = legacy.parse_rectifier_rows(
        legacy.RAW / "2026电解一系统运行记录和脱铜往返读数累计.xls",
        "整流运行原始记录",
        "PHASE_1",
        {EXTERNAL_YEAR},
    )
    phase2 = legacy.parse_rectifier_rows(
        legacy.RAW / "2026电解二系统运行记录和脱铜往返读数累计.xls",
        "电解整流运行日志",
        "PHASE_2",
        {EXTERNAL_YEAR},
    )
    phase1_trimmed, removed = trim_phase1_trailing_template_rows(phase1)
    canonical = _canonicalize_electro([phase1_trimmed, phase2])
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], errors="raise")
    canonical = canonical.loc[
        canonical["timestamp"].dt.year.eq(EXTERNAL_YEAR) & canonical["timestamp"].le(cutoff)
    ].copy()
    canonical = canonical.sort_values(["system", "timestamp"], kind="mergesort").reset_index(drop=True)
    safe_columns = [
        "operation_event_id",
        "timestamp",
        "year",
        "system",
        *ELECTRO_CORE_COLUMNS,
        "source_file",
        "source_sheet",
        "source_row",
        "source_date_raw",
        "date_recovery_method",
        "time_cell_raw",
        "time_recovery_method",
        "date_block_size",
        "time_cell_missing_flag",
        "time_cell_conflict_flag",
        "time_grid_off_flag",
        "timestamp_source_record_count",
        "timestamp_exact_duplicate_flag",
        "timestamp_conflict_flag",
        "source_record_resolution",
        "west_state_signal_conflict_flag",
        "east_state_signal_conflict_flag",
        "stage12_decopper_state_signal_conflict_flag",
        "west_implausible_flag",
        "east_implausible_flag",
        "stage12_decopper_implausible_flag",
        "any_implausible_value_flag",
        "any_state_signal_conflict_flag",
        "analysis_eligible",
    ]
    canonical = canonical[safe_columns]
    audit = {
        "phase1_trailing_template_rows_removed": int(removed),
        "phase1_rows_at_or_before_cutoff": int(canonical["system"].eq("PHASE_1").sum()),
        "phase2_rows_at_or_before_cutoff": int(canonical["system"].eq("PHASE_2").sum()),
    }
    return canonical, audit


def parse_stage34_2026(
    legacy: ModuleType,
    *,
    cutoff: pd.Timestamp = FROZEN_COMMON_CUTOFF,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """按既有 analysis_eligible 保守排除所有重复时间戳组，不自动拼接。"""

    parsed = legacy.parse_stage34_operation().copy()
    parsed["timestamp"] = pd.to_datetime(parsed["timestamp"], errors="raise")
    parsed = parsed.loc[
        parsed["timestamp"].dt.year.eq(EXTERNAL_YEAR) & parsed["timestamp"].le(cutoff)
    ].copy()
    duplicate = parsed["timestamp_duplicate_or_conflict_flag"].fillna(False).astype(bool)
    if parsed.loc[duplicate, "analysis_eligible"].fillna(False).astype(bool).any():
        raise AssertionError("三四段重复时间戳组不得进入 analysis_eligible")
    safe_columns = [
        "event_id",
        "timestamp",
        "year",
        *STAGE34_CORE_COLUMNS,
        "source_file",
        "source_sheet",
        "source_row",
        "date_recovery_method",
        "time_recovery_method",
        "time_cell_raw",
        "date_block_size",
        "time_cell_missing_flag",
        "time_cell_conflict_flag",
        "timestamp_order_flag",
        "no_process_measurement_flag",
        "stage3_state_signal_conflict_flag",
        "stage4_state_signal_conflict_flag",
        "stage3_hydraulic_electrical_disagreement_review_flag",
        "stage4_hydraulic_electrical_disagreement_review_flag",
        "stage3_state_strict",
        "stage4_state_strict",
        "mode_strict",
        "routing_status_strict",
        "any_implausible_value_flag",
        "timestamp_record_count",
        "timestamp_duplicate_or_conflict_flag",
        "analysis_eligible",
    ]
    result = parsed[safe_columns].sort_values(["timestamp", "source_row"], kind="mergesort")
    result = result.reset_index(drop=True)
    audit = {
        "stage34_rows_at_or_before_cutoff": int(len(result)),
        "stage34_duplicate_or_conflict_rows_excluded": int(duplicate.sum()),
        "stage34_analysis_eligible_rows": int(result["analysis_eligible"].sum()),
    }
    return result, audit


def build_external_event_artifacts(events: pd.DataFrame) -> ExternalEventArtifactsV2:
    """构造 current-result -> next-result；下一真值只进入 outcome_ledger。"""

    required = {
        "event_id",
        "recorded_at",
        "available_at",
        "sample_at_assumed",
        "current_cu",
        "current_as",
        "source_quality_eligible",
        "source_conflict",
        "event_time_parse_eligible",
        "source_file",
        "source_sheet",
        "source_row",
    }
    if missing := required - set(events.columns):
        raise ValueError(f"2026 目标事件缺列: {sorted(missing)}")
    ordered = events.sort_values(["recorded_at", "event_id"], kind="mergesort").reset_index(drop=True)
    ordered["sequence_index"] = ordered.index.astype(int)

    metadata_rows: list[dict[str, Any]] = []
    known_rows: list[dict[str, Any]] = []
    admission_rows: list[dict[str, Any]] = []
    for event in ordered.itertuples(index=False):
        complete = bool(pd.notna(event.current_cu) and pd.notna(event.current_as))
        parse_ok = bool(event.event_time_parse_eligible)
        warnings: list[str] = []
        reasons: list[str] = []
        if not complete:
            reasons.append("CURRENT_RESULT_INCOMPLETE")
        if not parse_ok:
            reasons.append("CURRENT_SOURCE_TIME_FLAG_INELIGIBLE")
        if not bool(event.source_quality_eligible):
            warnings.append("CURRENT_SOURCE_QUALITY_WARNING")
        if bool(event.source_conflict):
            warnings.append("CURRENT_SOURCE_CONFLICT_WARNING")
        status = "REJECTED" if reasons else "WARNING" if warnings else "ADMITTED"

        metadata_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_id": CONTRACT_ID,
                "sequence_index": int(event.sequence_index),
                "origin_event_id": str(event.event_id),
                "recorded_at": _iso(event.recorded_at),
                "available_at": _iso(event.available_at),
                "sample_at_assumed": _iso(event.sample_at_assumed),
                "year": EXTERNAL_YEAR,
                "source_quality_eligible": bool(event.source_quality_eligible),
                "source_conflict": bool(event.source_conflict),
                "source_file": event.source_file,
                "source_sheet": event.source_sheet,
                "source_row": int(event.source_row),
            }
        )
        known_rows.append(
            {
                "origin_event_id": str(event.event_id),
                "recorded_at": _iso(event.recorded_at),
                "origin_cu_g_l": float(event.current_cu) if pd.notna(event.current_cu) else np.nan,
                "origin_as_mg_l": float(event.current_as) if pd.notna(event.current_as) else np.nan,
                "origin_result_quality_eligible": bool(event.source_quality_eligible),
                "available_at": _iso(event.available_at),
                "sample_at_assumed": _iso(event.sample_at_assumed),
                "source_file": event.source_file,
                "source_sheet": event.source_sheet,
                "source_row": int(event.source_row),
            }
        )
        card = AsOfAdmissionCardV2(
            origin_event_id=str(event.event_id),
            decision_at=pd.Timestamp(event.recorded_at).to_pydatetime(),
            feature_cutoff_at=pd.Timestamp(event.recorded_at).to_pydatetime(),
            admission_status=status,
            reason_codes=tuple(reasons),
            feature_group_counts={"current_waste_tank_result": int(complete)},
            missing_feature_groups=(),
            source_quality_warnings=tuple(warnings),
        )
        assert_no_future_information(card, decision_at=card.decision_at)
        payload = card.model_dump(mode="json")
        for field in (
            "reason_codes",
            "feature_group_counts",
            "missing_feature_groups",
            "source_quality_warnings",
        ):
            payload[field] = _json_cell(payload[field])
        admission_rows.append(payload)

    pair_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    for index, origin in ordered.iterrows():
        target = ordered.iloc[index + 1] if index + 1 < len(ordered) else None
        target_id = str(target["event_id"]) if target is not None else None
        pair_id = stable_pair_id(str(origin["event_id"]), target_id)
        pair_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_id": CONTRACT_ID,
                "sequence_index": int(index),
                "pair_id": pair_id,
                "origin_event_id": str(origin["event_id"]),
                "origin_recorded_at": _iso(origin["recorded_at"]),
                "origin_available_at": _iso(origin["available_at"]),
                "origin_sample_at_assumed": _iso(origin["sample_at_assumed"]),
                "target_event_id": target_id,
                "target_recorded_at": _iso(target["recorded_at"]) if target is not None else None,
                "target_available_at": _iso(target["available_at"]) if target is not None else None,
                "target_sample_at_assumed": (
                    _iso(target["sample_at_assumed"]) if target is not None else None
                ),
                "origin_year": EXTERNAL_YEAR,
                "target_year": EXTERNAL_YEAR if target is not None else pd.NA,
                "cross_year_pair": False,
                "pair_status": (
                    "LABELED_STRICT_ADJACENT" if target is not None else "FINAL_EVENT_AUDIT_NO_LABEL"
                ),
            }
        )
        if target is None:
            continue
        lead = (target["recorded_at"] - origin["recorded_at"]).total_seconds() / 3600
        if lead <= 0:
            raise ValueError("2026 相邻事件录入时间跨度必须为正")
        prospective = bool(target["sample_at_assumed"] > origin["recorded_at"])
        quality = bool(target["source_quality_eligible"])
        conflict = bool(target["source_conflict"])
        complete = bool(pd.notna(target["current_cu"]) and pd.notna(target["current_as"]))
        reasons: list[str] = []
        if not prospective:
            reasons.append("NON_PROSPECTIVE_UNDER_2H_ASSUMPTION")
        if not quality:
            reasons.append("TARGET_SOURCE_QUALITY_INELIGIBLE")
        if conflict:
            reasons.append("TARGET_SOURCE_CONFLICT")
        if not complete:
            reasons.append("TARGET_OUTCOME_INCOMPLETE")
        eligible = prospective and quality and not conflict and complete
        ledger = EvaluationEligibilityLedgerV2(
            pair_id=pair_id,
            origin_event_id=str(origin["event_id"]),
            target_event_id=str(target["event_id"]),
            decision_at=pd.Timestamp(origin["recorded_at"]).to_pydatetime(),
            target_recorded_at=pd.Timestamp(target["recorded_at"]).to_pydatetime(),
            target_sample_at_assumed=pd.Timestamp(target["sample_at_assumed"]).to_pydatetime(),
            lead_to_recorded_hours=float(lead),
            prospectively_sampled=prospective,
            target_source_quality_eligible=quality,
            evaluation_eligible=eligible,
            evaluation_reason_codes=tuple(reasons),
        )
        eval_payload = ledger.model_dump(mode="json")
        eval_payload["evaluation_reason_codes"] = _json_cell(
            eval_payload["evaluation_reason_codes"]
        )
        evaluation_rows.append(eval_payload)
        if complete:
            outcome_rows.append(
                OutcomeLedgerV2(
                    pair_id=pair_id,
                    target_cu_g_l=float(target["current_cu"]),
                    target_as_mg_l=float(target["current_as"]),
                ).model_dump(mode="json")
            )

    pairs = pd.DataFrame(pair_rows)
    admissions = pd.DataFrame(admission_rows)
    evaluations = pd.DataFrame(evaluation_rows)
    outcomes = pd.DataFrame(outcome_rows)
    outcome_pairs = set(outcomes["pair_id"].astype(str))
    admission_lookup = admissions.set_index("origin_event_id")["admission_status"].to_dict()
    eligibility_lookup = evaluations.set_index("pair_id")["evaluation_eligible"].to_dict()
    partitions = pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "contract_id": CONTRACT_ID,
                "partition_id": "EXTERNAL_2026_LOCKED",
                "partition_role": "EXTERNAL_TEMPORAL_HOLDOUT",
                "origin_event_id": row.origin_event_id,
                "pair_id": row.pair_id,
                "decision_at": row.origin_recorded_at,
                "origin_year": EXTERNAL_YEAR,
                "admission_status": admission_lookup[row.origin_event_id],
                "has_sealed_outcome": row.pair_id in outcome_pairs,
                "primary_evaluation_eligible": bool(eligibility_lookup.get(row.pair_id, False)),
                "row_usage": (
                    "LOCKED_EXTERNAL_EVALUATION"
                    if bool(eligibility_lookup.get(row.pair_id, False))
                    else "AUDIT_OR_SENSITIVITY_ONLY"
                ),
            }
            for row in pairs.itertuples(index=False)
        ]
    )
    return ExternalEventArtifactsV2(
        event_metadata=pd.DataFrame(metadata_rows),
        origin_known_results=pd.DataFrame(known_rows),
        event_pair_index=pairs,
        as_of_admission_cards=admissions,
        evaluation_ledger=evaluations,
        outcome_ledger=outcomes,
        partition_manifest=partitions,
    )


def _apply_filters(frame: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for column, expected in filters.items():
        if column not in result:
            raise ValueError(f"过程表缺少筛选字段: {column}")
        if isinstance(expected, bool):
            normalized = result[column].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
            result = result.loc[normalized.eq(expected)]
        else:
            result = result.loc[result[column].eq(expected)]
    return result.copy()


def build_external_observation_artifacts(
    event_artifacts: ExternalEventArtifactsV2,
    process_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> ObservationArtifactsV2:
    """仅用 decision_at 及之前过程记录构造与开发集同构的 27 变量快照。"""

    events = event_artifacts.event_metadata[["origin_event_id", "recorded_at"]].rename(
        columns={"recorded_at": "decision_at"}
    )
    events["decision_at"] = pd.to_datetime(events["decision_at"], errors="raise")
    all_snapshots: list[pd.DataFrame] = []
    feature_matrices: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []

    for group_name, group in config["groups"].items():
        records = _apply_filters(process_frames[group_name], group.get("row_filter", {}))
        mapping = {source: meta["tag"] for source, meta in group["features"].items()}
        required = [
            group["source_event_id_column"],
            group["source_time_column"],
            "source_file",
            "source_sheet",
            "source_row",
            *mapping.keys(),
        ]
        if missing := set(required) - set(records.columns):
            raise ValueError(f"{group_name} 缺少字段: {sorted(missing)}")
        records = records[required].rename(columns=mapping)
        snapshots = backward_asof_snapshots(
            events,
            records,
            record_id_column=group["source_event_id_column"],
            record_time_column=group["source_time_column"],
            anchor_offsets_hours=config["anchor_offsets_hours"],
            tolerance_hours=float(config["backward_tolerance_hours"]),
        )
        snapshots.insert(2, "feature_group", group_name)
        all_snapshots.append(snapshots)
        feature_matrices.append(
            summarize_feature_anchors(
                snapshots,
                event_id_column="origin_event_id",
                feature_columns=list(mapping.values()),
            )
        )
        coverage = snapshots.groupby("origin_event_id", as_index=False).agg(
            matched_anchor_count=("missing_record", lambda values: int((~values).sum())),
            unique_matched_record_count=("matched_record_id", "nunique"),
            maximum_anchor_match_age_hours=("anchor_match_age_hours", "max"),
        )
        coverage["feature_group"] = group_name
        coverage["group_4_of_7_ready"] = coverage["unique_matched_record_count"].ge(
            int(config["minimum_unique_records_per_group"])
        )
        coverage_parts.append(coverage)

    snapshots = pd.concat(all_snapshots, ignore_index=True, sort=False)
    future = snapshots["matched_at"].notna() & (
        (snapshots["matched_at"] > snapshots["anchor_at"])
        | (snapshots["available_at"] > snapshots["decision_at"])
    )
    if future.any():
        raise AssertionError("2026 外部快照出现未来过程记录")

    feature_matrix = events.copy()
    for matrix in feature_matrices:
        feature_matrix = feature_matrix.merge(
            matrix, on="origin_event_id", how="left", validate="one_to_one"
        )
    feature_matrix = feature_matrix.merge(
        event_artifacts.origin_known_results[
            [
                "origin_event_id",
                "origin_cu_g_l",
                "origin_as_mg_l",
                "origin_result_quality_eligible",
            ]
        ],
        on="origin_event_id",
        how="left",
        validate="one_to_one",
    )

    coverage_long = pd.concat(coverage_parts, ignore_index=True)
    coverage_wide = coverage_long.pivot(
        index="origin_event_id",
        columns="feature_group",
        values=["matched_anchor_count", "unique_matched_record_count", "group_4_of_7_ready"],
    )
    coverage_wide.columns = [f"{group}__{metric}" for metric, group in coverage_wide.columns]
    coverage_wide = coverage_wide.reset_index().merge(
        events, on="origin_event_id", how="left", validate="one_to_one"
    )
    ready_columns = [column for column in coverage_wide if column.endswith("__group_4_of_7_ready")]
    coverage_wide["all_three_groups_4_of_7_ready"] = coverage_wide[ready_columns].all(axis=1)

    final_cards = assemble_final_admission_cards(
        event_artifacts.as_of_admission_cards, coverage_wide
    )
    pairs = event_artifacts.event_pair_index[
        ["pair_id", "origin_event_id", "origin_recorded_at", "pair_status"]
    ]
    evaluations = event_artifacts.evaluation_ledger[
        ["pair_id", "prospectively_sampled", "evaluation_eligible"]
    ]
    external_index = pairs.merge(
        final_cards[
            ["origin_event_id", "admission_status", "core_process_4_of_7_ready"]
        ],
        on="origin_event_id",
        how="left",
        validate="many_to_one",
    ).merge(evaluations, on="pair_id", how="left", validate="one_to_one")
    outcome_pairs = set(event_artifacts.outcome_ledger["pair_id"].astype(str))
    external_index["sealed_label_available"] = external_index["pair_id"].astype(str).isin(
        outcome_pairs
    )
    external_index["partition_role"] = "EXTERNAL_TEMPORAL_HOLDOUT"
    external_index["runtime_prediction_available"] = external_index["admission_status"].ne(
        "REJECTED"
    )
    external_index["tier1_core_external_evaluation"] = (
        external_index["runtime_prediction_available"]
        & external_index["core_process_4_of_7_ready"].astype(bool)
        & external_index["evaluation_eligible"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
    )
    forbidden_value_columns = {
        "target_cu_g_l",
        "target_as_mg_l",
        "observed_cu_g_l",
        "observed_as_mg_l",
    }
    if forbidden_value_columns & set(external_index.columns):
        raise AssertionError("外部主索引不得包含目标真值列")
    return ObservationArtifactsV2(
        snapshot_long=snapshots,
        feature_matrix=feature_matrix,
        coverage=coverage_wide,
        final_admission_cards=final_cards,
        external_evaluation_index=external_index,
    )
