from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = ["open", "high", "low", "close", "volume"]
TARGET_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass
class TrainingArtifacts:
    stock_name: str
    window_size: int
    horizon: int
    target_field: str
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: float
    target_std: float
    train_ratio: float
    created_at: str
    train_samples: int
    test_samples: int
    last_training_date: str
    model_path: str


@dataclass
class TrainConfig:
    stock_name: str = "unknown_stock"
    window_size: int = 30
    horizon: int = 5
    target_field: str = "close"
    epochs: int = 50
    batch_size: int = 32
    train_ratio: float = 0.8
    validation_split: float = 0.1


def require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for this script. Install it with "
            "`python -m pip install tensorflow` and run the command again."
        ) from exc

    return {
        "tf": tf,
        "Sequential": Sequential,
        "EarlyStopping": EarlyStopping,
        "Conv1D": Conv1D,
        "Dense": Dense,
        "Dropout": Dropout,
        "Input": Input,
        "LSTM": LSTM,
        "MaxPooling1D": MaxPooling1D,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and use a CNN-LSTM model on OHLCV JSON data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model on OHLCV JSON data.")
    train_parser.add_argument("--json", required=True, help="Path to the OHLCV JSON file.")
    train_parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory where the trained model and metadata will be saved.",
    )
    train_parser.add_argument("--stock-name", default="unknown_stock", help="Label for the stock.")
    train_parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Number of past trading days used as input.",
    )
    train_parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Number of future trading days to predict.",
    )
    train_parser.add_argument(
        "--target-field",
        choices=FEATURE_NAMES,
        default="close",
        help="Which OHLCV field to predict.",
    )
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    train_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio. Default is 0.8.",
    )
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation split used inside the training data.",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Load a saved model and forecast the next 5 trading days."
    )
    predict_parser.add_argument("--json", required=True, help="Path to the OHLCV JSON file.")
    predict_parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing the trained model and metadata.",
    )
    predict_parser.add_argument(
        "--output",
        help="Optional output JSON path for the forecast. Defaults to model-dir/forecast.json.",
    )

    return parser.parse_args()


def load_ohlcv_data_from_source(
    *,
    path: Path | None = None,
    payload: Any | None = None,
) -> tuple[list[datetime], np.ndarray]:
    if path is None and payload is None:
        raise ValueError("Either `path` or `payload` must be provided.")
    if path is not None and payload is not None:
        raise ValueError("Provide either `path` or `payload`, not both.")

    if path is not None:
        return load_ohlcv_json(path)

    assert payload is not None
    return load_ohlcv_payload(payload)


def load_ohlcv_json(path: Path) -> tuple[list[datetime], np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return load_ohlcv_payload(payload)


def load_ohlcv_payload(payload: Any) -> tuple[list[datetime], np.ndarray]:
    candles = extract_candles(payload)
    if not candles:
        raise ValueError("No OHLCV candles found in payload.")

    parsed_dates: list[datetime] = []
    rows: list[list[float]] = []

    for candle in candles:
        if len(candle) < 6:
            raise ValueError("Each candle must include at least timestamp, OHLC, and volume.")

        parsed_dates.append(parse_datetime(candle[0]))
        rows.append(
            [
                float(candle[1]),
                float(candle[2]),
                float(candle[3]),
                float(candle[4]),
                float(candle[5]),
            ]
        )

    order = np.argsort(np.array(parsed_dates, dtype=object))
    ordered_dates = [parsed_dates[index] for index in order]
    ordered_rows = np.asarray(rows, dtype=np.float32)[order]
    return ordered_dates, ordered_rows


def extract_candles(payload: Any) -> list[list[Any]]:
    if isinstance(payload, dict):
        if "candles" in payload and isinstance(payload["candles"], list):
            return payload["candles"]

        if "data" in payload and isinstance(payload["data"], list):
            for item in payload["data"]:
                nested = item.get("data", {}) if isinstance(item, dict) else {}
                candles = nested.get("candles", [])
                if candles:
                    return candles

    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            return payload
        if payload and isinstance(payload[0], dict) and "candles" in payload[0]:
            return payload[0]["candles"]

    raise ValueError("Unsupported JSON format. Expected Upstox-style OHLCV JSON.")


def parse_datetime(raw_value: str) -> datetime:
    try:
        return datetime.strptime(raw_value, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return datetime.fromisoformat(raw_value)


def build_sequences(
    values: np.ndarray,
    dates: list[datetime],
    window_size: int,
    horizon: int,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray, list[list[str]], list[str]]:
    sample_count = len(values) - window_size - horizon + 1
    if sample_count <= 0:
        raise ValueError(
            "Not enough rows to build sequences. Increase the JSON history or reduce "
            "`--window-size` / `--horizon`."
        )

    feature_count = values.shape[1]
    inputs = np.zeros((sample_count, window_size, feature_count), dtype=np.float32)
    targets = np.zeros((sample_count, horizon), dtype=np.float32)
    forecast_dates: list[list[str]] = []
    context_dates: list[str] = []

    for start in range(sample_count):
        split_point = start + window_size
        target_end = split_point + horizon
        inputs[start] = values[start:split_point]
        targets[start] = values[split_point:target_end, target_index]
        forecast_dates.append([dates[index].date().isoformat() for index in range(split_point, target_end)])
        context_dates.append(dates[split_point - 1].date().isoformat())

    return inputs, targets, forecast_dates, context_dates


def split_train_test(
    inputs: np.ndarray,
    targets: np.ndarray,
    forecast_dates: list[list[str]],
    context_dates: list[str],
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[str]], list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("`--train-ratio` must be between 0 and 1.")

    split_index = int(len(inputs) * train_ratio)
    split_index = min(max(split_index, 1), len(inputs) - 1)

    return (
        inputs[:split_index],
        targets[:split_index],
        inputs[split_index:],
        targets[split_index:],
        forecast_dates[split_index:],
        context_dates[split_index:],
    )


def fit_scalers(train_inputs: np.ndarray, train_targets: np.ndarray) -> dict[str, np.ndarray | float]:
    feature_mean = train_inputs.reshape(-1, train_inputs.shape[-1]).mean(axis=0)
    feature_std = train_inputs.reshape(-1, train_inputs.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std())
    if target_std == 0:
        target_std = 1.0

    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
    }


def scale_inputs(values: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    return (values - feature_mean) / feature_std


def scale_targets(values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    return (values - target_mean) / target_std


def inverse_scale_targets(values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    return (values * target_std) + target_mean


def build_model(window_size: int, feature_count: int, horizon: int):
    keras_modules = require_tensorflow()

    model = keras_modules["Sequential"](
        [
            keras_modules["Input"](shape=(window_size, feature_count)),
            keras_modules["Conv1D"](filters=64, kernel_size=3, activation="relu", padding="causal"),
            keras_modules["Conv1D"](filters=64, kernel_size=3, activation="relu", padding="causal"),
            keras_modules["MaxPooling1D"](pool_size=2),
            keras_modules["LSTM"](64, return_sequences=True),
            keras_modules["Dropout"](0.2),
            keras_modules["LSTM"](32),
            keras_modules["Dense"](64, activation="relu"),
            keras_modules["Dense"](horizon),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = np.clip(np.abs(actual), 1e-8, None)
    mape = float(np.mean(np.abs(error) / denominator) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_model(
    *,
    model_dir: Path,
    config: TrainConfig,
    json_path: Path | None = None,
    payload: Any | None = None,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)

    dates, values = load_ohlcv_data_from_source(path=json_path, payload=payload)
    target_index = TARGET_INDEX[config.target_field]

    inputs, targets, forecast_dates, context_dates = build_sequences(
        values=values,
        dates=dates,
        window_size=config.window_size,
        horizon=config.horizon,
        target_index=target_index,
    )
    train_x, train_y, test_x, test_y, test_forecast_dates, test_context_dates = split_train_test(
        inputs=inputs,
        targets=targets,
        forecast_dates=forecast_dates,
        context_dates=context_dates,
        train_ratio=config.train_ratio,
    )

    scalers = fit_scalers(train_x, train_y)
    train_x_scaled = scale_inputs(train_x, scalers["feature_mean"], scalers["feature_std"])
    test_x_scaled = scale_inputs(test_x, scalers["feature_mean"], scalers["feature_std"])
    train_y_scaled = scale_targets(train_y, scalers["target_mean"], scalers["target_std"])
    test_y_scaled = scale_targets(test_y, scalers["target_mean"], scalers["target_std"])

    keras_modules = require_tensorflow()
    tf = keras_modules["tf"]
    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_model(
        window_size=config.window_size,
        feature_count=train_x.shape[-1],
        horizon=config.horizon,
    )

    early_stopping = keras_modules["EarlyStopping"](
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    history = model.fit(
        train_x_scaled,
        train_y_scaled,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        callbacks=[early_stopping],
        verbose=1,
    )

    predictions_scaled = model.predict(test_x_scaled, verbose=0)
    predictions = inverse_scale_targets(
        predictions_scaled,
        scalers["target_mean"],
        scalers["target_std"],
    )
    actual = inverse_scale_targets(
        test_y_scaled,
        scalers["target_mean"],
        scalers["target_std"],
    )
    metrics = compute_metrics(actual, predictions)

    model_path = model_dir / "model.keras"
    model.save(model_path)

    artifacts = TrainingArtifacts(
        stock_name=config.stock_name,
        window_size=config.window_size,
        horizon=config.horizon,
        target_field=config.target_field,
        feature_names=FEATURE_NAMES,
        feature_mean=[float(value) for value in scalers["feature_mean"]],
        feature_std=[float(value) for value in scalers["feature_std"]],
        target_mean=float(scalers["target_mean"]),
        target_std=float(scalers["target_std"]),
        train_ratio=float(config.train_ratio),
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        train_samples=int(len(train_x)),
        test_samples=int(len(test_x)),
        last_training_date=dates[-1].date().isoformat(),
        model_path=model_path.name,
    )

    history_payload = {
        "loss": [float(value) for value in history.history.get("loss", [])],
        "mae": [float(value) for value in history.history.get("mae", [])],
        "val_loss": [float(value) for value in history.history.get("val_loss", [])],
        "val_mae": [float(value) for value in history.history.get("val_mae", [])],
    }

    prediction_rows = []
    for index in range(len(predictions)):
        prediction_rows.append(
            {
                "context_end_date": test_context_dates[index],
                "forecast_dates": test_forecast_dates[index],
                "actual": [float(value) for value in actual[index]],
                "predicted": [float(value) for value in predictions[index]],
            }
        )

    save_json(model_dir / "metadata.json", asdict(artifacts))
    save_json(model_dir / "training_history.json", history_payload)
    save_json(model_dir / "test_metrics.json", metrics)
    save_json(model_dir / "test_predictions.json", prediction_rows)

    return {
        "stock_name": config.stock_name,
        "target_field": config.target_field,
        "train_samples": len(train_x),
        "test_samples": len(test_x),
        "metrics": metrics,
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "metadata_path": str(model_dir / "metadata.json"),
        "history_path": str(model_dir / "training_history.json"),
        "test_metrics_path": str(model_dir / "test_metrics.json"),
        "test_predictions_path": str(model_dir / "test_predictions.json"),
        "last_training_date": dates[-1].date().isoformat(),
        "history": history_payload,
        "test_predictions": prediction_rows,
    }


def train_command(args: argparse.Namespace) -> None:
    result = train_model(
        model_dir=Path(args.model_dir),
        json_path=Path(args.json),
        config=TrainConfig(
            stock_name=args.stock_name,
            window_size=args.window_size,
            horizon=args.horizon,
            target_field=args.target_field,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            validation_split=args.validation_split,
        ),
    )
    print(f"Training complete for {result['stock_name']}.")
    print(f"Train samples: {result['train_samples']}")
    print(f"Test samples: {result['test_samples']}")
    print(
        "Test metrics: "
        f"MAE={result['metrics']['mae']:.4f}, "
        f"RMSE={result['metrics']['rmse']:.4f}, "
        f"MAPE={result['metrics']['mape']:.2f}%"
    )
    print(f"Saved model to: {result['model_path']}")
    print(f"Saved evaluation output to: {result['test_predictions_path']}")


def load_metadata(model_dir: Path) -> TrainingArtifacts:
    metadata_path = model_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TrainingArtifacts(**payload)


def next_trading_days(last_date: datetime, count: int) -> list[str]:
    future_dates: list[str] = []
    current = last_date

    while len(future_dates) < count:
        current = current + timedelta(days=1)
        if current.weekday() < 5:
            future_dates.append(current.date().isoformat())

    return future_dates


def predict_with_model(
    *,
    model_dir: Path,
    json_path: Path | None = None,
    payload: Any | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    keras_modules = require_tensorflow()
    tf = keras_modules["tf"]
    metadata = load_metadata(model_dir)

    dates, values = load_ohlcv_data_from_source(path=json_path, payload=payload)
    if len(values) < metadata.window_size:
        raise ValueError(
            f"Prediction needs at least {metadata.window_size} rows, but only {len(values)} were found."
        )

    feature_mean = np.asarray(metadata.feature_mean, dtype=np.float32)
    feature_std = np.asarray(metadata.feature_std, dtype=np.float32)

    latest_window = values[-metadata.window_size :]
    latest_window_scaled = scale_inputs(latest_window, feature_mean, feature_std)
    latest_window_scaled = np.expand_dims(latest_window_scaled, axis=0)

    model = tf.keras.models.load_model(model_dir / metadata.model_path)
    forecast_scaled = model.predict(latest_window_scaled, verbose=0)[0]
    forecast = inverse_scale_targets(
        forecast_scaled,
        metadata.target_mean,
        metadata.target_std,
    )

    forecast_dates = next_trading_days(dates[-1], metadata.horizon)
    result = {
        "stock_name": metadata.stock_name,
        "target_field": metadata.target_field,
        "latest_input_date": dates[-1].date().isoformat(),
        "forecast_dates": forecast_dates,
        "predicted_values": [float(value) for value in forecast],
        "model_dir": str(model_dir),
    }

    destination = output_path if output_path else model_dir / "forecast.json"
    save_json(destination, result)
    result["forecast_path"] = str(destination)
    return result


def predict_command(args: argparse.Namespace) -> None:
    result = predict_with_model(
        model_dir=Path(args.model_dir),
        json_path=Path(args.json),
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Forecast for {result['stock_name']} ({result['target_field']}):")
    for forecast_date, forecast_value in zip(result["forecast_dates"], result["predicted_values"]):
        print(f"  {forecast_date}: {forecast_value:.4f}")
    print(f"Saved forecast to: {result['forecast_path']}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_command(args)
        return

    if args.command == "predict":
        predict_command(args)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
