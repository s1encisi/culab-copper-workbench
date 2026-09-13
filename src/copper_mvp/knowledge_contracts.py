"""Versioned document metadata; document text never grants execution authority."""
from __future__ import annotations

from typing import Literal
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from copper_mvp.access import PROJECT


class DocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")
    project_id: str = PROJECT
    title: str = Field(min_length=1, max_length=200)
    author: str = Field(min_length=1, max_length=160)
    source: str = Field(min_length=1, max_length=500)
    authorization: str = Field(min_length=1, max_length=500)
    classification: Literal["public", "internal", "confidential"] = "internal"
    license: str = Field(min_length=1, max_length=300)
    version_label: str = Field(min_length=1, max_length=80)
    format: Literal["md", "docx", "pdf"]
    effective_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    readers: list[str] = Field(default_factory=list, max_length=100)
    read_roles: list[Literal["researcher", "viewer"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_times(self):
        if self.expires_at is not None and self.expires_at <= self.effective_at:
            raise ValueError("失效时间必须晚于生效时间")
        return self


class DocumentAccess(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readers: list[str] = Field(default_factory=list, max_length=100)
    read_roles: list[Literal["researcher", "viewer"]] = Field(default_factory=list)
    revoked: bool = False


class DocumentReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accepted: bool
    note: str = Field(min_length=1, max_length=1000)


class KnowledgeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    as_of: AwareDatetime | None = None
    limit: int = Field(default=5, ge=1, le=20)
    context_tokens: int = Field(default=6000, ge=100, le=6000)
