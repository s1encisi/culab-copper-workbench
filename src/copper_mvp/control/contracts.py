"""Exact command inputs for the synthetic current-control fixture."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEVICE = "mock.cell3"
POLICY = "MOCK_PROTOCOL_FIXTURE.v1"
POINTS = ("mock.cell3.current_setpoint", "mock.cell3.aux_current_setpoint")
NOTICE = "Mock 环境，真实工厂未连接"
TERMINAL = {"VERIFIED", "PARTIAL", "FAILED", "REJECTED", "CANCELLED", "EXPIRED"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Target(StrictModel):
    point_id: Literal["mock.cell3.current_setpoint", "mock.cell3.aux_current_setpoint"]
    value: float
    unit: Literal["A"] = "A"


class ProposalInput(StrictModel):
    request_key: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    device_id: Literal["mock.cell3"] = DEVICE
    operation: Literal["set_absolute"] = "set_absolute"
    targets: list[Target] = Field(min_length=1, max_length=2)
    ramp_seconds: float = Field(default=1, gt=0, le=30)
    ttl_seconds: float = Field(default=30, gt=0, le=120)
    tolerance: float = Field(default=1, gt=0, le=1)
    settling_deadline_seconds: float = Field(default=10, ge=3, le=60)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def distinct(self):
        if len({p.point_id for p in self.targets}) != len(self.targets):
            raise ValueError("点位不能重复")
        return self


class ApprovalInput(StrictModel):
    payload_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["approve", "reject"] = "approve"
    reason: str = Field(default="", max_length=500)


class TickInput(StrictModel):
    seconds: float = Field(gt=0, le=60)


class FaultInput(StrictModel):
    mode: Literal["remote", "manual", "maintenance"] | None = None
    connected: bool | None = None
    running: bool | None = None
    estop_latched: bool | None = None
    interlocks_ok: bool | None = None
    reject_writes: bool | None = None
    partial_write_count: int | None = Field(default=None, ge=0, le=2)
    ack_loss: bool | None = None
    stuck_pv: bool | None = None
    bad_quality: bool | None = None
    stale: bool | None = None
    readback_bias: float | None = Field(default=None, ge=-20, le=20)
    sensor_bias: float | None = Field(default=None, ge=-20, le=20)
    noise: float | None = Field(default=None, ge=0, le=2)
    ack_delay_seconds: float | None = Field(default=None, ge=0, le=20)
    write_delay_seconds: float | None = Field(default=None, ge=0, le=20)
    pv_delay_seconds: float | None = Field(default=None, ge=0, le=20)
    restart: bool = False
    reset_estop: bool = False


class DispatchInput(StrictModel):
    command: dict
    payload_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    fencing_token: int = Field(ge=1)
    lease_until: float
