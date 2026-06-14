from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    ok: bool = True
    data: Any = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    song_id: str = "2001"
    start_phrase: int = 1
    max_phrases: int | None = None
    assembly_mode: Literal["concat", "timeline"] = "timeline"
    normalize: bool = True
    infer_class: str = "cascade"
    f0_naturalize: bool = True
    voice_model: str | None = None
    index_file: str | None = None
    f0_method: str = "rmvpe"
    f0_up_key: int = 0
    index_rate: float = 0.5
    protect: float = 0.33
    filter_radius: int = 3
    resample_sr: int = 0


class JobSnapshot(BaseModel):
    job_id: str
    name: str
    status: str
    command: str
    start_time: str | None = None
    end_time: str | None = None
    returncode: int | None = None
    log_path: str
