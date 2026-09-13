"""Versioned document metadata; document text never grants execution authority."""
from __future__ import annotations

from typing import Literal, Annotated
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
    expected_parse_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    accepted: bool
    note: str = Field(min_length=1, max_length=1000)


class KnowledgeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    as_of: AwareDatetime | None = None
    limit: int = Field(default=5, ge=1, le=20)
    context_tokens: int = Field(default=6000, ge=100, le=6000)


class OCRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_parse_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    pages: list[Annotated[int, Field(ge=1)]] | None = Field(default=None, min_length=1, max_length=20)
    dpi: Literal[120, 180, 240] = 180


class CorrectedRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["paragraph", "heading", "table", "equation", "image"]
    text: str = Field(default="", max_length=30000)
    heading: str = Field(default="", max_length=500)
    bbox: tuple[float, float, float, float] | None = None
    headers: list[str] = Field(default_factory=list, max_length=100)
    rows: list[list[str]] = Field(default_factory=list, max_length=1000)
    footnotes: list[str] = Field(default_factory=list, max_length=100)
    merged_cells: list[dict] = Field(default_factory=list, max_length=100)
    definitions: str = Field(default="", max_length=6000)


class BlockCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: int = Field(ge=0)
    regions: list[CorrectedRegion] = Field(min_length=1, max_length=100)


class DocumentCorrections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_parse_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corrections: list[BlockCorrection] = Field(min_length=1, max_length=100)
    note: str = Field(min_length=1, max_length=1000)
