from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trend_analyzer.ohlcv_lstm_cnn import (
    TrainConfig,
    load_metadata,
    predict_with_model,
    train_model,
)


def train_ohlcv_model(
    *,
    model_dir: str,
    stock_name: str,
    window_size: int,
    horizon: int,
    target_field: str,
    epochs: int,
    batch_size: int,
    train_ratio: float,
    validation_split: float,
    json_path: str | None = None,
    ohlcv_data: Any | None = None,
) -> dict[str, Any]:
    return train_model(
        model_dir=Path(model_dir),
        json_path=Path(json_path) if json_path else None,
        payload=ohlcv_data,
        config=TrainConfig(
            stock_name=stock_name,
            window_size=window_size,
            horizon=horizon,
            target_field=target_field,
            epochs=epochs,
            batch_size=batch_size,
            train_ratio=train_ratio,
            validation_split=validation_split,
        ),
    )


def predict_ohlcv_model(
    *,
    model_dir: str,
    json_path: str | None = None,
    ohlcv_data: Any | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    return predict_with_model(
        model_dir=Path(model_dir),
        json_path=Path(json_path) if json_path else None,
        payload=ohlcv_data,
        output_path=Path(output_path) if output_path else None,
    )


def get_ohlcv_model_info(model_dir: str) -> dict[str, Any]:
    model_path = Path(model_dir)
    metadata = load_metadata(model_path)
    return {
        "model_dir": str(model_path),
        "model_file": str(model_path / metadata.model_path),
        "metadata": asdict(metadata),
        "artifacts": {
            "training_history": str(model_path / "training_history.json"),
            "test_metrics": str(model_path / "test_metrics.json"),
            "test_predictions": str(model_path / "test_predictions.json"),
            "forecast": str(model_path / "forecast.json"),
        },
    }


def get_ohlcv_model_results(model_dir: str) -> dict[str, Any]:
    model_path = Path(model_dir)
    metadata = load_metadata(model_path)
    return {
        "model_dir": str(model_path),
        "metadata": asdict(metadata),
        "metrics": read_json_artifact(model_path / "test_metrics.json", required=True),
        "history": read_json_artifact(model_path / "training_history.json", required=True),
        "test_predictions": read_json_artifact(
            model_path / "test_predictions.json",
            required=True,
        ),
        "forecast": read_json_artifact(model_path / "forecast.json", required=False),
    }


def read_json_artifact(path: Path, *, required: bool) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"model artifact not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
