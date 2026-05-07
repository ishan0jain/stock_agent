# OHLCV Model and API Guide

This project contains:

- a CNN-LSTM training script for OHLCV JSON data
- a saved-model prediction flow for new daily data
- a FastAPI application to train and predict through HTTP

## 1. Install

From the project root:

```powershell
python -m pip install -e .[ml]
```

This installs the API package and the TensorFlow dependency required for model training and prediction.

## 2. OHLCV JSON Format

The trainer expects OHLCV candles in the same format as `trend_analyzer/ohlv.json`.

Each candle should contain:

```json
[
  "2026-05-07T00:00:00+0530",
  100.0,
  105.0,
  98.0,
  103.5,
  250000,
  0
]
```

Fields used by the model:

- index `0`: timestamp
- index `1`: open
- index `2`: high
- index `3`: low
- index `4`: close
- index `5`: volume

## 3. Train the Model

Run training from the project root:

```powershell
python trend_analyzer/ohlcv_lstm_cnn.py train --json trend_analyzer/ohlv.json --model-dir trend_analyzer/models/reliance --stock-name RELIANCE
```

What this does:

- reads the OHLCV JSON file
- creates training samples from the candle history
- trains on 80% of the samples
- tests on the remaining 20%
- saves the trained model and evaluation files

Useful optional arguments:

```powershell
python trend_analyzer/ohlcv_lstm_cnn.py train `
  --json trend_analyzer/ohlv.json `
  --model-dir trend_analyzer/models/reliance `
  --stock-name RELIANCE `
  --window-size 30 `
  --horizon 5 `
  --target-field close `
  --epochs 50 `
  --batch-size 32 `
  --train-ratio 0.8 `
  --validation-split 0.1
```

## 4. Check Test Results

After training, the model directory will contain:

- `model.keras`: trained model
- `metadata.json`: scaler settings and model metadata
- `training_history.json`: train/validation loss history
- `test_metrics.json`: summary metrics for the 20% test set
- `test_predictions.json`: actual vs predicted values on the test set

Example:

```powershell
Get-Content trend_analyzer/models/reliance/test_metrics.json
Get-Content trend_analyzer/models/reliance/test_predictions.json
```

`test_metrics.json` is the main file to check the evaluation result after training.

## 5. Get New Results Later

When you have a new OHLCV JSON file the next morning, run prediction using the latest saved model:

```powershell
python trend_analyzer/ohlcv_lstm_cnn.py predict --json trend_analyzer/ohlv.json --model-dir trend_analyzer/models/reliance
```

What this does:

- loads the saved model from `model.keras`
- reads the latest OHLCV rows from the JSON file
- uses the most recent `window-size` rows as input
- predicts the next 5 trading days
- writes the result to `forecast.json`

Output file:

```powershell
Get-Content trend_analyzer/models/reliance/forecast.json
```

If you want to train again with updated history, run the `train` command again using the new JSON file.

## 6. Start the Application

Run the FastAPI app:

```powershell
uvicorn stock_agents.app:app --reload
```

App URLs after startup:

- API root health check: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI spec: `http://127.0.0.1:8000/openapi.json`

## 7. API Endpoints for the Model

Available endpoints:

- `POST /api/v1/model/train`
- `POST /api/v1/model/predict`
- `POST /api/v1/model/info`

## 8. Train Through API

Example request body:

```json
{
  "json_path": "trend_analyzer/ohlv.json",
  "model_dir": "trend_analyzer/models/reliance",
  "stock_name": "RELIANCE",
  "window_size": 30,
  "horizon": 5,
  "target_field": "close",
  "epochs": 50,
  "batch_size": 32,
  "train_ratio": 0.8,
  "validation_split": 0.1
}
```

Example PowerShell request:

```powershell
$body = @'
{
  "json_path": "trend_analyzer/ohlv.json",
  "model_dir": "trend_analyzer/models/reliance",
  "stock_name": "RELIANCE",
  "window_size": 30,
  "horizon": 5,
  "target_field": "close",
  "epochs": 50,
  "batch_size": 32,
  "train_ratio": 0.8,
  "validation_split": 0.1
}
'@

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/model/train" -Method Post -ContentType "application/json" -Body $body
```

## 9. Predict Through API

Example request body:

```json
{
  "json_path": "trend_analyzer/ohlv.json",
  "model_dir": "trend_analyzer/models/reliance"
}
```

Example PowerShell request:

```powershell
$body = @'
{
  "json_path": "trend_analyzer/ohlv.json",
  "model_dir": "trend_analyzer/models/reliance"
}
'@

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/model/predict" -Method Post -ContentType "application/json" -Body $body
```

## 10. Get Saved Model Info Through API

Example request body:

```json
{
  "model_dir": "trend_analyzer/models/reliance"
}
```

Example PowerShell request:

```powershell
$body = @'
{
  "model_dir": "trend_analyzer/models/reliance"
}
'@

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/model/info" -Method Post -ContentType "application/json" -Body $body
```

## 11. Typical Daily Workflow

1. Put the latest stock OHLCV data into a JSON file.
2. If you want to retrain the model with the updated history, run the `train` command again.
3. Check `test_metrics.json` to review the new 20% test performance.
4. Run the `predict` command to get the next 5 trading-day forecast.
5. If you want to integrate this into another system, call the FastAPI endpoints instead of the CLI commands.

## 12. Notes

- One model directory should be used per stock.
- Example: `trend_analyzer/models/reliance`, `trend_analyzer/models/tcs`, `trend_analyzer/models/infy`
- The model predicts one target field at a time. Default is `close`.
- The prediction horizon is 5 days by default.
- The application will not train or predict until TensorFlow is installed.
