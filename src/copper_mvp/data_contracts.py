"""G1 contracts: explicit historical clocks, units, and immutable label revisions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from copper_mas.data.leakage import find_forbidden_paths
from copper_mvp.common import digest, utc_now

SOURCE_TIMEZONE = ZoneInfo("Asia/Shanghai")
UNITS = {"cu": "g/L", "as": "mg/L"}


def source_time(value) -> datetime:
    """Naive source timestamps have the frozen contract's Shanghai semantics."""
    if value is None:
        raise ValueError("缺少时间")
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SOURCE_TIMEZONE)
    return parsed.astimezone(timezone.utc)


def envelope(schema: str, payload: dict, source_kind: str) -> dict:
    content = {"schema_version": schema, "project_id": "copper-research",
               "source_kind": source_kind, "payload": payload}
    signature = digest(content)
    return {**content, "id": schema + ":" + signature, "content_hash": signature,
            "created_at": utc_now(), "created_by": "local-data-service",
            "trace_id": "evidence:" + signature}


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TargetSpec(StrictRecord):
    name: Literal["cu", "as"]
    unit: Literal["g/L", "mg/L"]

    @model_validator(mode="after")
    def valid_unit(self):
        if self.unit != UNITS[self.name]:
            raise ValueError("目标单位不匹配")
        return self


class TaskSpec(StrictRecord):
    schema_version: Literal["task-spec.g1"] = "task-spec.g1"
    task_id: Literal["copper-next-recorded-result"] = "copper-next-recorded-result"
    evaluation_mode: Literal["historical_replay"] = "historical_replay"
    event_id: str = Field(min_length=1, max_length=180)
    decision_at: AwareDatetime
    feature_cutoff_at: AwareDatetime
    horizon: Literal["next_recorded_event"] = "next_recorded_event"
    targets: tuple[TargetSpec, TargetSpec] = (
        TargetSpec(name="cu", unit="g/L"), TargetSpec(name="as", unit="mg/L"))
    input_columns: tuple[str, ...]
    dataset_version: str
    feature_spec_version: str
    timing_contract_id: str
    timing_contract_sha256: str
    model_scope: Literal["oof_replay"] = "oof_replay"
    source_timezone: Literal["Asia/Shanghai"] = "Asia/Shanghai"
    observed_sample_time: None = None
    observed_sample_time_missing_reason: Literal["not_recorded"] = "not_recorded"
    assumed_sample_time: AwareDatetime
    sample_delay_is_assumption: Literal[True] = True
    inputs_are_control_setpoints: Literal[False] = False

    @field_validator("decision_at", "feature_cutoff_at", "assumed_sample_time")
    @classmethod
    def utc_fields(cls, value):
        return source_time(value)

    @model_validator(mode="after")
    def valid_scope(self):
        if self.decision_at != self.feature_cutoff_at:
            raise ValueError("特征截止必须等于该事件决策时点")
        if self.assumed_sample_time != self.decision_at - timedelta(hours=2):
            raise ValueError("假定采样时间必须按冻结合同单独标记")
        if tuple(t.name for t in self.targets) != ("cu", "as"):
            raise ValueError("任务目标必须为 Cu、As")
        if not self.input_columns or len(set(self.input_columns)) != len(self.input_columns):
            raise ValueError("输入字段缺失或重复")
        if find_forbidden_paths(dict.fromkeys(self.input_columns)):
            raise ValueError("输入字段含未来信息")
        return self


class FeatureValue(StrictRecord):
    value: float | None
    unit: str
    available_at: AwareDatetime
    missing_reason: str | None = None

    @model_validator(mode="after")
    def explicit_missing(self):
        if (self.value is None) != (self.missing_reason is not None):
            raise ValueError("缺失数值必须提供原因，非缺失数值不得标记缺失")
        return self


class LabelRecord(StrictRecord):
    schema_version: Literal["label-record.g1"] = "label-record.g1"
    event_id: str
    pair_id: str
    target: Literal["cu", "as"]
    horizon: Literal["next_recorded_event"] = "next_recorded_event"
    value: float | None = Field(ge=0)
    unit: Literal["g/L", "mg/L"]
    decision_at: AwareDatetime
    available_at: AwareDatetime
    observed_sample_time: AwareDatetime | None = None
    observed_sample_time_missing_reason: str | None = "not_recorded"
    assumed_sample_time: AwareDatetime
    revision: int = Field(default=1, ge=1)
    supersedes: str | None = None
    quality_eligible: bool
    missing_reason: str | None = None
    source_version: str
    source_kind: Literal["development_ledger", "synthetic_fixture"] = "development_ledger"

    @field_validator("decision_at", "available_at", "observed_sample_time", "assumed_sample_time")
    @classmethod
    def utc_fields(cls, value):
        return source_time(value) if value is not None else None

    @model_validator(mode="after")
    def valid_record(self):
        if self.unit != UNITS[self.target] or self.available_at <= self.decision_at:
            raise ValueError("标签单位或可得时间不符合下一事件合同")
        if self.assumed_sample_time > self.available_at:
            raise ValueError("假定采样时间晚于标签可得时间")
        if (self.value is None) != (self.missing_reason is not None):
            raise ValueError("标签缺失状态与原因不一致")
        if (self.observed_sample_time is None) != (self.observed_sample_time_missing_reason is not None):
            raise ValueError("实际采样时间缺失状态与原因不一致")
        if self.observed_sample_time is not None and self.observed_sample_time > self.available_at:
            raise ValueError("实际采样时间晚于标签可得时间")
        if (self.revision == 1) != (self.supersedes is None):
            raise ValueError("标签修订必须关联前一版本")
        return self

    @property
    def record_id(self) -> str:
        return "label:" + digest(self.model_dump(mode="json"))

    def as_dict(self) -> dict:
        return {"id": self.record_id, **self.model_dump(mode="json")}


class ReplayQuery(StrictRecord):
    as_of: AwareDatetime | None = None
    model_profile: Literal["Persistence", "DeltaRidge", "DeltaHGB"] = "Persistence"
    bundle_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class LabelQuery(StrictRecord):
    as_of: AwareDatetime
    days: int = Field(default=30, ge=1, le=3650)
    limit: int = Field(default=60, ge=1, le=3000)
    minimum: int = Field(default=60, ge=1, le=3000)
