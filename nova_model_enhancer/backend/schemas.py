"""Explicit request/response models for every endpoint that takes a body."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    application: str
    version: str
    schema_version: int
    workspace_writable: bool
    model_families_available: dict[str, bool]
    detail: str = ""


class CompatibilityRequest(BaseModel):
    trust_local_package: bool = Field(
        ...,
        description="Must be true. Loading a .pkl executes code from the uploaded package.",
    )
    actor: str = ""


class TargetEncoding(BaseModel):
    voice_values: list[str] = ["0"]
    non_voice_values: list[str] = ["1"]


class ReadinessDecisions(BaseModel):
    date_column: str
    target_mode: Literal["existing", "derive_from_subtask"]
    target_column: str | None = None
    target_encoding: TargetEncoding = TargetEncoding()
    dedup_mode: Literal["full_row", "key_columns", "none"]
    dedup_keys: list[str] = []
    subtask_mappings: list[dict[str, Any]] = []
    subtask_keywords: list[str] = []
    allow_unmapped_default: bool = False
    acknowledge_model_output_target: bool = False
    historical_window_days: int | None = None
    # Explicit training window. Either bound may stand alone. When both a range
    # and a days-back value are present the range wins, and the snapshot
    # manifest records which one was applied.
    date_from: str | None = None
    date_to: str | None = None
    approver: str

    @field_validator("date_from", "date_to")
    @classmethod
    def _valid_date(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        import pandas as pd

        parsed = pd.to_datetime(str(value).strip(), errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"'{value}' is not a date this application can read.")
        return str(value).strip()

    @model_validator(mode="after")
    def _range_is_ordered(self):
        if self.date_from and self.date_to:
            import pandas as pd

            if pd.to_datetime(self.date_from) > pd.to_datetime(self.date_to):
                raise ValueError("The training window starts after it ends.")
        return self

    @field_validator("approver")
    @classmethod
    def _approver_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An approver name is required to record this decision.")
        return value.strip()


class WeightPreviewRequest(BaseModel):
    strategy: dict[str, Any]


class WeightApprovalRequest(BaseModel):
    strategy: dict[str, Any]
    approver: str
    notes: str = ""

    @field_validator("approver")
    @classmethod
    def _approver_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An approver name is required to save a weight strategy.")
        return value.strip()


class SplitConfig(BaseModel):
    mode: Literal["temporal", "random"] = "temporal"
    train_pct: int = 70
    val_pct: int = 15
    test_pct: int = 15
    stratify: bool = True
    seed: int = 42


class TrainingRequest(BaseModel):
    split: SplitConfig = SplitConfig()
    n_trials: int | None = None
    timeout_seconds: int | None = None
    n_jobs: int = -1
    seed: int = 42
    second_family: str | None = None
    include_baseline: bool = True
    # Off by default: the backtest is roughly a third of a run's cost. When it
    # is skipped the promotion gate reports stability as not assessed rather
    # than treating its absence as a pass.
    # Sequential by default: on a 4-core machine splitting cores between
    # candidates measured slower, not faster. Raise it only on a box with
    # more cores than one model fit can use.
    max_parallel_candidates: int = 1
    run_backtest: bool = False
    backtest_windows: int | None = None
    threshold_criterion: Literal["f1", "recall", "precision", "balanced_accuracy", "weighted_composite"] = "f1"
    actor: str = ""


class GateConfig(BaseModel):
    primary_metric: Literal["f1", "recall", "precision", "auc", "pr_auc", "balanced_accuracy", "weighted_composite"] = "f1"
    min_primary_improvement_pct: float = 1.0
    protected_metrics: list[dict[str, Any]] = [{"metric": "recall", "max_regression_pct": 0.5}]
    max_historical_primary_regression_pct: float | None = 1.0
    require_backtest_pass: bool = True
    require_package_validation: bool = True
    segment_column: str | None = "SubTask"
    min_segment_rows: int = 100


class GateApprovalRequest(BaseModel):
    gate: GateConfig
    approver: str

    @field_validator("approver")
    @classmethod
    def _approver_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An approver name is required to save the promotion gate.")
        return value.strip()


class PromotionApprovalRequest(BaseModel):
    run_id: str
    candidate_id: str
    decision: Literal["APPROVED", "REJECTED"]
    approver: str
    typed_confirmation: str = Field(
        ..., description="Must exactly equal the candidate id being approved."
    )
    notes: str = ""

    @field_validator("approver")
    @classmethod
    def _approver_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An approver name is required to record a promotion decision.")
        return value.strip()


class MlTagApprovalRequest(BaseModel):
    column_name: str = "ml_tag"
    voice_value: Any = 1
    non_voice_value: Any = 0
    approver: str
    notes: str = ""

    @field_validator("approver")
    @classmethod
    def _approver_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An approver name is required to confirm the ml_tag encoding.")
        return value.strip()

    @field_validator("column_name")
    @classmethod
    def _column_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("The appended column needs a name.")
        return value


class ExportRequestBody(BaseModel):
    run_id: str
    candidate_id: str
    actor: str = ""
