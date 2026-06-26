"""Pydantic models for pipeline JSON and runtime Settings."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pipeline JSON models (produced by SEER — shape is fixed)
# ---------------------------------------------------------------------------

class QuestionOption(BaseModel):
    value: str
    label: str
    ic_passes: int | None = None


class Question(BaseModel):
    question_id: int
    key: str
    version: int
    version_id: int
    label: str
    help_text: str = ""
    question_type: Literal["boolean", "integer", "float", "text", "categorical"]
    allow_multiple: bool = False
    options: list[QuestionOption] = Field(default_factory=list)


class Paper(BaseModel):
    paper_id: int
    title: str
    abstract: str = ""
    split: str = ""


# Batching can be one of four forms:
#   "per_question" | "all" | {"size": N} | [["k1","k2"], ["k3"]]
BatchingConfig = Union[
    Literal["per_question", "all"],
    dict[Literal["size"], int],
    list[list[str]],
]


class RunConfig(BaseModel):
    concurrency: int = 8
    per_provider_rpm: float | None = None
    drop_params: bool = False
    citation_max_error_rate: float = 0.05
    citation_max_ellipsis_gap: int = 600
    text_source: Literal["full_text", "abstract"] = "full_text"
    batching: BatchingConfig = "per_question"
    temperature: float = 0.0
    reasoning_effort: str | None = None
    format_model: str | None = None
    format_model_provider: str | None = None
    format_model_params: dict[str, Any] = Field(default_factory=dict)
    format_structured_output: bool = True
    model_params: dict[str, Any] = Field(default_factory=dict)
    cache: bool = False
    cache_first: Literal["text", "questions"] = "text"
    cache_ttl: Literal["5m", "1h"] = "1h"
    system_prompt: str | None = None
    batch_p1: bool = False
    batch_p2: bool = False
    fail_fast: bool = False


class ExperimentRun(BaseModel):
    run_id: int
    name: str
    model_name: str
    model_provider: str
    config: RunConfig = Field(default_factory=RunConfig)


class PipelineConfig(BaseModel):
    review_id: int
    setup_id: int
    api_base: str        # ends in /api
    api_token: str
    papers: list[Paper]
    questions: list[Question]
    runs: list[ExperimentRun]

    @field_validator("api_base")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


# ---------------------------------------------------------------------------
# Runtime Settings (from TOML / env — infra knobs SEER doesn't supply)
# ---------------------------------------------------------------------------

class ProviderSettings(BaseModel):
    api_key_env: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    base_url: str | None = None

    def resolved_api_key(self) -> str | None:
        import os
        return self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)


class RuntimeSettings(BaseModel):
    store_path: str = "./.seer_state.db"
    p1_dump_dir: str = "./tmp"


# ---------------------------------------------------------------------------
# Run-level defaults (settings.toml [run_defaults] — overrides RunConfig
# code defaults but is itself overridden by explicit per-run RunConfig values)
# ---------------------------------------------------------------------------

class RunDefaults(BaseModel):
    """Global defaults for RunConfig fields, settable in settings.toml [run_defaults].

    Hierarchy (least → most specific):
      RunConfig code defaults → settings.toml [run_defaults] → pipeline JSON runs[*].config
    """
    concurrency: int | None = None
    per_provider_rpm: float | None = None
    drop_params: bool | None = None
    citation_max_error_rate: float | None = None
    citation_max_ellipsis_gap: int | None = None
    text_source: Literal["full_text", "abstract"] | None = None
    batching: BatchingConfig | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    format_model: str | None = None
    format_model_provider: str | None = None
    format_model_params: dict[str, Any] | None = None
    format_structured_output: bool | None = None
    model_params: dict[str, Any] | None = None
    cache: bool | None = None
    cache_first: Literal["text", "questions"] | None = None
    cache_ttl: Literal["5m", "1h"] | None = None
    system_prompt: str | None = None
    batch_p1: bool | None = None
    batch_p2: bool | None = None
    fail_fast: bool | None = None


def effective_run_config(run_config: RunConfig, defaults: RunDefaults) -> RunConfig:
    """Merge run_defaults (from settings.toml) with per-run RunConfig.

    Fields explicitly set in the pipeline JSON always win over run_defaults.
    run_defaults fill in for fields that are absent from the pipeline JSON.
    """
    base = defaults.model_dump(exclude_none=True)
    override = run_config.model_dump(exclude_unset=True)
    return RunConfig.model_validate({**base, **override})


class Settings(BaseModel):
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    run_defaults: RunDefaults = Field(default_factory=RunDefaults)

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        import os
        import tomllib

        if path is None:
            path = os.environ.get("SEER_SETTINGS", "settings.toml")

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            return cls.model_validate(data)
        except FileNotFoundError:
            return cls()

    @classmethod
    def defaults(cls) -> "Settings":
        return cls()
