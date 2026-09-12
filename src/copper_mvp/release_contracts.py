"""Explicit artifact, shadow and release requests; authority comes from the server."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

TASK_ID="copper-next-recorded-result"
TARGET_UNITS={"cu":"g/L","as":"mg/L"}
BASELINE_ID="builtin-persistence"
LIFECYCLE=("candidate","registered","runnable","tested","benchmarked","shadow","approved","retired","quarantined")


class StrictModel(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)


class ArtifactRequest(StrictModel):
    request_key:str=Field(min_length=1,max_length=100,pattern=r"^[A-Za-z0-9_.:-]+$")
    source_kind:Literal["model_comparison","ensemble_study"]
    source_id:str=Field(pattern=r"^[a-f0-9]{32}$")
    method_id:str=Field(min_length=1,max_length=80)
    target:Literal["cu","as"]
    seed:int|None=Field(default=None,ge=0,lt=2**31)


class ShadowRequest(StrictModel):
    request_key:str=Field(min_length=1,max_length=100,pattern=r"^[A-Za-z0-9_.:-]+$")
    samples_per_fold:int=Field(default=8,ge=2,le=32)


class ReleaseRequest(StrictModel):
    request_key:str=Field(min_length=1,max_length=100,pattern=r"^[A-Za-z0-9_.:-]+$")
    candidate_id:str=Field(min_length=1,max_length=80)
    shadow_id:str|None=Field(default=None,pattern=r"^[a-f0-9]{32}$")
    target:Literal["cu","as"]
    purpose:Literal["forecast","scenario_prediction"]="forecast"
    expected_version:int=Field(ge=0)
    ttl_seconds:int=Field(default=900,ge=30,le=3600)
    reason:str=Field(min_length=1,max_length=500)


class ReleaseApproval(StrictModel):
    payload_hash:str=Field(pattern=r"^[a-f0-9]{64}$")
    decision:Literal["approve","reject"]="approve"
    reason:str=Field(default="",max_length=500)


class RollbackRequest(StrictModel):
    request_key:str=Field(min_length=1,max_length=100,pattern=r"^[A-Za-z0-9_.:-]+$")
    target:Literal["cu","as"]
    expected_version:int=Field(ge=0)
    destination_release_id:str|None=Field(default=None,pattern=r"^[a-f0-9]{32}$")
    reason:str=Field(min_length=1,max_length=500)


class ForecastRequest(StrictModel):
    request_key:str=Field(min_length=1,max_length=100,pattern=r"^[A-Za-z0-9_.:-]+$")
    event_id:str=Field(min_length=1,max_length=180)
    selection:Literal["current_research","as_of_event"]="current_research"
