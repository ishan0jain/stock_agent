from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class StockInput(BaseModel):
    symbol: str = Field(..., description="Primary stock symbol used by the caller.")
    name: str = Field(..., description="Canonical company name.")
    aliases: list[str] = Field(default_factory=list)
    sector: str | None = None
    sector_keywords: list[str] = Field(default_factory=list)
    fundamental_symbol: str | None = Field(
        default=None,
        description=(
            "Optional provider-specific symbol for financial-statement and valuation lookups "
            "(for example if the data vendor uses a different ticker format)."
        ),
    )
    nse_symbol: str | None = Field(
        default=None,
        description="NSE symbol if different from the primary symbol.",
    )
    bse_scrip_code: str | None = Field(
        default=None,
        description="BSE scrip code for company-specific corporate announcement lookups.",
    )
    exchange: str | None = Field(default="NSE", description="Primary exchange for the stock.")

    @field_validator("symbol", "name")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("aliases", "sector_keywords")
    @classmethod
    def clean_string_list(cls, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = raw.strip()
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped


class AnalysisOptions(BaseModel):
    as_of: datetime | None = Field(
        default=None,
        description="Reference time in ISO format. Defaults to now in IST.",
    )
    since: datetime | None = Field(
        default=None,
        description="Explicit lower-bound for ingestion. Overrides the derived market window.",
    )
    hours_before_previous_close: int = Field(
        default=2,
        ge=0,
        le=12,
        description="How many hours before the previous market close the window starts.",
    )
    company_news_limit: int = Field(default=25, ge=1, le=100)
    sector_news_limit: int = Field(default=20, ge=1, le=100)
    macro_news_limit: int = Field(default=20, ge=1, le=100)
    exchange_announcement_limit: int = Field(default=20, ge=1, le=100)
    global_market_limit: int = Field(default=12, ge=1, le=50)
    macro_series_limit: int = Field(default=12, ge=1, le=50)
    timeout_seconds: float = Field(default=12.0, ge=1.0, le=60.0)
    include_site_search: bool = Field(default=True)
    include_broad_business_news: bool = Field(default=True)


class StockAnalysisRequest(BaseModel):
    stock: StockInput
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class WatchlistAnalysisRequest(BaseModel):
    stocks: list[StockInput] = Field(..., min_length=1, max_length=100)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class TimeWindowRequest(BaseModel):
    stocks: list[StockInput] = Field(default_factory=list)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class ModelInfoRequest(BaseModel):
    model_dir: str = Field(..., description="Directory containing the saved model artifacts.")

    @field_validator("model_dir")
    @classmethod
    def validate_model_dir(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_dir cannot be empty")
        return value


class OhlcvDataSourceRequest(BaseModel):
    json_path: str | None = Field(
        default=None,
        description="Path to a JSON file on disk containing OHLCV candles.",
    )
    ohlcv_data: Any | None = Field(
        default=None,
        description="Inline OHLCV JSON payload in the same structure as the training file.",
    )

    @model_validator(mode="after")
    def validate_data_source(self) -> "OhlcvDataSourceRequest":
        if bool(self.json_path) == bool(self.ohlcv_data is not None):
            raise ValueError("provide exactly one of json_path or ohlcv_data")
        if self.json_path is not None:
            self.json_path = self.json_path.strip()
            if not self.json_path:
                raise ValueError("json_path cannot be empty")
        return self


class ModelTrainRequest(OhlcvDataSourceRequest):
    model_dir: str = Field(..., description="Directory where the trained model will be saved.")
    stock_name: str = Field(default="unknown_stock")
    window_size: int = Field(default=30, ge=5, le=365)
    horizon: int = Field(default=5, ge=1, le=30)
    target_field: Literal["open", "high", "low", "close", "volume"] = Field(default="close")
    epochs: int = Field(default=50, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=1024)
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    validation_split: float = Field(default=0.1, ge=0.0, lt=1.0)

    @field_validator("model_dir", "stock_name")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


class ModelPredictRequest(OhlcvDataSourceRequest):
    model_dir: str = Field(..., description="Directory containing the trained model.")
    output_path: str | None = Field(
        default=None,
        description="Optional path where the forecast JSON should be written.",
    )

    @field_validator("model_dir", "output_path")
    @classmethod
    def validate_predict_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("path values cannot be empty")
        return value


class PredictionStoreRequest(BaseModel):
    stock: StockInput
    prediction_made_at: datetime | None = Field(
        default=None,
        description="Timestamp when the prediction was created. Defaults to now in IST.",
    )
    prediction_for_date: datetime | None = Field(
        default=None,
        description="The main date being predicted. Defaults to the first forecast date when available.",
    )
    reference_price: float | None = Field(
        default=None,
        gt=0.0,
        description="Latest known actual price at prediction time, used to convert predicted value into a percent move.",
    )
    target_field: Literal["open", "high", "low", "close", "volume"] = Field(default="close")
    forecast_dates: list[str] = Field(default_factory=list, max_length=30)
    predicted_values: list[float] = Field(default_factory=list, max_length=30)
    predicted_direction: Literal["up", "down", "flat"] | None = None
    predicted_change_pct: float | None = None
    prediction_source: str = Field(default="ohlcv_model")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model_dir: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_prediction_payload(self) -> "PredictionStoreRequest":
        if self.forecast_dates and len(self.forecast_dates) != len(self.predicted_values):
            raise ValueError("forecast_dates and predicted_values must have the same length")
        if not self.forecast_dates and not self.predicted_values and self.predicted_direction is None and self.predicted_change_pct is None:
            raise ValueError(
                "provide forecast values or an explicit predicted direction/change to store a prediction"
            )
        return self


class PredictionReviewRequest(BaseModel):
    prediction_id: str = Field(..., description="Identifier returned when the prediction was stored.")
    actual_for_date: datetime | None = Field(
        default=None,
        description="Date being evaluated. Defaults to the stored prediction target date.",
    )
    previous_close: float | None = Field(
        default=None,
        gt=0.0,
        description="Previous close used with actual_close to derive the daily percent move.",
    )
    actual_close: float | None = Field(
        default=None,
        gt=0.0,
        description="Actual closing price for the reviewed day.",
    )
    actual_change_pct: float | None = Field(
        default=None,
        description="Actual percent move for the reviewed day. If omitted, it is derived from actual_close and previous_close/reference_price.",
    )
    analyst_notes: str | None = None
    auto_fetch_context: bool = Field(
        default=True,
        description="When true, fetch stock/news context between the prediction time and the reviewed day.",
    )
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)

    @field_validator("prediction_id")
    @classmethod
    def validate_prediction_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prediction_id cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_actual_payload(self) -> "PredictionReviewRequest":
        if self.actual_change_pct is None and self.actual_close is None:
            raise ValueError("provide actual_change_pct or actual_close")
        return self


class AgentMemoryRequest(BaseModel):
    stock: StockInput
