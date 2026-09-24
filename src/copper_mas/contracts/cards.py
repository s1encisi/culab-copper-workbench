"""V2 数据卡：预测时工件与事后账本必须分离。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_ID = "DHC_EVENT_SEQUENCE_V2_20260826"


class StrictCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AsOfAdmissionCardV2(StrictCard):
    """A1 在 decision_at 当时能够生成的准入卡。"""

    schema_version: Literal["2.0"] = "2.0"
    contract_id: Literal[CONTRACT_ID] = CONTRACT_ID
    origin_event_id: str = Field(min_length=1)
    decision_at: datetime
    feature_cutoff_at: datetime
    admission_status: Literal["ADMITTED", "WARNING", "REJECTED"]
    reason_codes: tuple[str, ...] = ()
    feature_group_counts: dict[str, int] = Field(default_factory=dict)
    missing_feature_groups: tuple[str, ...] = ()
    source_quality_warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_as_of_boundary(self) -> AsOfAdmissionCardV2:
        if self.feature_cutoff_at > self.decision_at:
            raise ValueError("feature_cutoff_at 不能晚于 decision_at")
        if any(value < 0 for value in self.feature_group_counts.values()):
            raise ValueError("feature_group_counts 不能为负数")
        return self


class ObservationItemV2(StrictCard):
    canonical_tag: str = Field(min_length=1)
    value: float | int | str | bool | None = None
    unit: str | None = None
    event_at: datetime | None = None
    available_at: datetime | None = None
    matched_anchor_at: datetime | None = None
    age_hours: float | None = Field(default=None, ge=0)
    missing: bool
    source_file: str | None = None
    source_sheet: str | None = None
    source_row: int | None = Field(default=None, ge=1)
    quality_code: str = "UNREVIEWED"

    @model_validator(mode="after")
    def validate_missingness(self) -> ObservationItemV2:
        if self.missing and self.value is not None:
            raise ValueError("missing=true 时 value 必须为空")
        if not self.missing and self.available_at is None:
            raise ValueError("非缺失观测必须有 available_at")
        return self


class ObservationCardV2(StrictCard):
    schema_version: Literal["2.0"] = "2.0"
    contract_id: Literal[CONTRACT_ID] = CONTRACT_ID
    origin_event_id: str = Field(min_length=1)
    decision_at: datetime
    observations: tuple[ObservationItemV2, ...]

    @model_validator(mode="after")
    def validate_all_observations_are_known(self) -> ObservationCardV2:
        future = [
            item.canonical_tag
            for item in self.observations
            if item.available_at is not None and item.available_at > self.decision_at
        ]
        if future:
            raise ValueError(f"发现 decision_at 之后才可用的观测: {future[:5]}")
        return self


class ForecastRequestV2(StrictCard):
    """允许进入 A2/A4 的唯一预测请求，不含任何下一事件信息。"""

    schema_version: Literal["2.0"] = "2.0"
    request_id: str = Field(min_length=1)
    admission: AsOfAdmissionCardV2
    observations: ObservationCardV2

    @model_validator(mode="after")
    def validate_identity(self) -> ForecastRequestV2:
        if self.admission.origin_event_id != self.observations.origin_event_id:
            raise ValueError("准入卡与观察卡的 origin_event_id 不一致")
        if self.admission.decision_at != self.observations.decision_at:
            raise ValueError("准入卡与观察卡的 decision_at 不一致")
        if self.admission.admission_status == "REJECTED":
            raise ValueError("REJECTED 准入卡不能形成预测请求")
        return self


class EvaluationEligibilityLedgerV2(StrictCard):
    """事后评价资格；绝不能进入 ForecastRequestV2。"""

    schema_version: Literal["2.0"] = "2.0"
    contract_id: Literal[CONTRACT_ID] = CONTRACT_ID
    pair_id: str = Field(min_length=1)
    origin_event_id: str = Field(min_length=1)
    target_event_id: str = Field(min_length=1)
    decision_at: datetime
    target_recorded_at: datetime
    target_sample_at_assumed: datetime
    lead_to_recorded_hours: float = Field(gt=0)
    prospectively_sampled: bool
    target_source_quality_eligible: bool
    evaluation_eligible: bool
    evaluation_reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_temporal_order(self) -> EvaluationEligibilityLedgerV2:
        if self.target_recorded_at <= self.decision_at:
            raise ValueError("target_recorded_at 必须晚于 decision_at")
        expected = (self.target_recorded_at - self.decision_at).total_seconds() / 3600
        if abs(expected - self.lead_to_recorded_hours) > 1e-6:
            raise ValueError("lead_to_recorded_hours 与时间戳不一致")
        return self


class OutcomeLedgerV2(StrictCard):
    """隔离的下一目标真值账本。"""

    schema_version: Literal["2.0"] = "2.0"
    contract_id: Literal[CONTRACT_ID] = CONTRACT_ID
    pair_id: str = Field(min_length=1)
    target_cu_g_l: float
    target_as_mg_l: float


class ProcessModeCardV2(StrictCard):
    schema_version: Literal["2.0"] = "2.0"
    contract_id: Literal[CONTRACT_ID] = CONTRACT_ID
    origin_event_id: str
    decision_at: datetime
    mode_code: str
    confidence: float = Field(ge=0, le=1)
    evidence_observation_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class PredictionCardV2(StrictCard):
    schema_version: Literal["2.0"] = "2.0"
    contract_id: Literal[CONTRACT_ID] = CONTRACT_ID
    request_id: str
    origin_event_id: str
    decision_at: datetime
    frozen_model_id: str
    predicted_cu_g_l: float
    predicted_as_mg_l: float
    cu_interval_low: float | None = None
    cu_interval_high: float | None = None
    as_interval_low: float | None = None
    as_interval_high: float | None = None
    warnings: tuple[str, ...] = ()


class AuditCardV2(StrictCard):
    schema_version: Literal["2.0"] = "2.0"
    contract_id: Literal[CONTRACT_ID] = CONTRACT_ID
    artifact_id: str
    passed: bool
    check_codes: tuple[str, ...]
    failure_codes: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
