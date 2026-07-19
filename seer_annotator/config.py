"""Pydantic models for pipeline JSON and runtime Settings."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Generous enough not to cut off legitimate long reasoning calls; the point is
# to bound the "hangs forever" case when a provider stalls, not to be tight.
DEFAULT_REQUEST_TIMEOUT = 600.0


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
    is_ic: bool = False  # inclusion-criteria flag; only meaningful for dispute questions


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
    batching: BatchingConfig = "all"
    temperature: float | None = None
    reasoning_effort: str | None = None
    format_model: str | None = None
    format_model_provider: str | None = None
    format_temperature: float | None = None
    format_model_params: dict[str, Any] = Field(default_factory=dict)
    format_structured_output: bool = True
    model_params: dict[str, Any] = Field(default_factory=dict)
    cache: bool = False
    cache_first: Literal["text", "questions"] = "questions"
    cache_ttl: Literal["5m", "1h"] = "1h"
    system_prompt: str | None = None
    chunk_papers: int = 10
    batch_p1: bool = False
    batch_p2: bool = False
    fail_fast: bool = False
    request_timeout: float | None = None


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
# Dispute-set pipeline JSON models (arbitration — shape is fixed, see
# overhaul-work/adjudication-contract.md Part B on the SEER side)
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    rater_key: str
    value: Any = None
    comment: str = ""
    cited_text: str = ""
    source_answer_id: int | None = None


class DisputeItem(BaseModel):
    dispute_item_id: int
    paper_id: int
    paper_title: str
    abstract: str = ""
    question_key: str
    version_id: int
    candidates: list[Candidate] = Field(default_factory=list)


class ArbiterRunConfig(BaseModel):
    """Tunables for one ArbiterRun. Deliberately separate from RunConfig — no
    chunk-papers-of-fixed-questions concept, and arbitration has its own knobs
    (text_source's candidates_only mode, batching semantics, rater anonymization)."""

    concurrency: int = 8
    per_provider_rpm: float | None = None
    drop_params: bool = False
    citation_max_error_rate: float = 0.05
    citation_max_ellipsis_gap: int = 600
    # "candidates_only" skips OCR/abstract entirely — the adjudicator reasons
    # only from the candidates' stated value/comment/cited_text.
    text_source: Literal["full_text", "abstract", "candidates_only"] = "full_text"
    # Reuses the same BatchingConfig shape as annotation's `batching`, applied
    # per-paper against that paper's disputed questions. "all" (default) puts
    # every disputed question for a paper in one Pass-1 call.
    batching: BatchingConfig = "all"
    anonymize_raters: bool = True
    temperature: float | None = None
    reasoning_effort: str | None = None
    format_model: str | None = None
    format_model_provider: str | None = None
    format_temperature: float | None = None
    format_model_params: dict[str, Any] = Field(default_factory=dict)
    format_structured_output: bool = True
    model_params: dict[str, Any] = Field(default_factory=dict)
    cache: bool = False
    cache_first: Literal["text", "questions"] = "text"
    cache_ttl: Literal["5m", "1h"] = "1h"
    system_prompt: str | None = None
    chunk_papers: int = 10
    batch_p1: bool = False
    batch_p2: bool = False
    fail_fast: bool = False
    request_timeout: float | None = None


class ArbiterRun(BaseModel):
    run_id: int
    name: str
    model_name: str
    model_provider: str
    config: ArbiterRunConfig = Field(default_factory=ArbiterRunConfig)


class DisputePipelineConfig(BaseModel):
    review_id: int
    review_name: str = ""
    dispute_set_id: int
    dispute_set_name: str = ""
    api_base: str
    api_token: str
    rater_keys: list[str] = Field(default_factory=list)
    runs: list[ArbiterRun]
    questions: list[Question]
    disputes: list[DisputeItem]

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
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

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
    format_temperature: float | None = None
    format_model_params: dict[str, Any] | None = None
    format_structured_output: bool | None = None
    model_params: dict[str, Any] | None = None
    cache: bool | None = None
    cache_first: Literal["text", "questions"] | None = None
    cache_ttl: Literal["5m", "1h"] | None = None
    system_prompt: str | None = None
    chunk_papers: int | None = None
    batch_p1: bool | None = None
    batch_p2: bool | None = None
    fail_fast: bool | None = None
    request_timeout: float | None = None


def effective_run_config(run_config: RunConfig, defaults: RunDefaults) -> RunConfig:
    """Merge run_defaults (from settings.toml) with per-run RunConfig.

    Fields explicitly set in the pipeline JSON always win over run_defaults.
    run_defaults fill in for fields that are absent from the pipeline JSON.
    """
    base = defaults.model_dump(exclude_none=True)
    override = run_config.model_dump(exclude_unset=True)
    return RunConfig.model_validate({**base, **override})


class ArbiterRunDefaults(BaseModel):
    """Global defaults for ArbiterRunConfig fields, settable in settings.toml [arbiter_run_defaults].

    Same three-level hierarchy as RunDefaults/RunConfig, applied to arbitration runs.
    """
    model_config = ConfigDict(extra="forbid")

    concurrency: int | None = None
    per_provider_rpm: float | None = None
    drop_params: bool | None = None
    citation_max_error_rate: float | None = None
    citation_max_ellipsis_gap: int | None = None
    text_source: Literal["full_text", "abstract", "candidates_only"] | None = None
    batching: BatchingConfig | None = None
    anonymize_raters: bool | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    format_model: str | None = None
    format_model_provider: str | None = None
    format_temperature: float | None = None
    format_model_params: dict[str, Any] | None = None
    format_structured_output: bool | None = None
    model_params: dict[str, Any] | None = None
    cache: bool | None = None
    cache_first: Literal["text", "questions"] | None = None
    cache_ttl: Literal["5m", "1h"] | None = None
    system_prompt: str | None = None
    chunk_papers: int | None = None
    batch_p1: bool | None = None
    batch_p2: bool | None = None
    fail_fast: bool | None = None
    request_timeout: float | None = None


def effective_arbiter_config(run_config: ArbiterRunConfig, defaults: ArbiterRunDefaults) -> ArbiterRunConfig:
    """Merge arbiter_run_defaults (from settings.toml) with per-run ArbiterRunConfig.

    Mirrors effective_run_config(): fields explicit in the pipeline JSON always win.
    """
    base = defaults.model_dump(exclude_none=True)
    override = run_config.model_dump(exclude_unset=True)
    return ArbiterRunConfig.model_validate({**base, **override})


class Settings(BaseModel):
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    run_defaults: RunDefaults = Field(default_factory=RunDefaults)
    arbiter_run_defaults: ArbiterRunDefaults = Field(default_factory=ArbiterRunDefaults)

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
