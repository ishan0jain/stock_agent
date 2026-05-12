# Stock Agents Application and Trend Prediction Guide

This guide explains:

- how to start the application
- how to use the stock context APIs
- how to train the OHLCV forecasting model
- how to predict future stock trends
- how to interpret the outputs responsibly

## 1. What This Repository Does

This repo has two separate but related capabilities:

1. `stock_agents`
   A FastAPI application that collects stock-specific context such as:
   - company news
   - sector news
   - India macro news
   - NSE/BSE corporate announcements
   - global market performance
   - macro cues such as crude, FX, and yields

2. `trend_analyzer`
   A CNN-LSTM model pipeline that:
   - trains on historical OHLCV candle data
   - predicts the next few trading days for one target field
   - saves forecast and evaluation artifacts

Use the API side to understand market context.
Use the model side to estimate short-term directional trends from historical price/volume data.

## 2. Prerequisites

- Python `3.11` or newer
- `pip`
- Optional API keys for richer data coverage:
  - `GNEWS_API_KEY`
  - `NEWSDATA_API_KEY`
  - `ALPHAVANTAGE_API_KEY`

Without these keys, some providers are skipped, but the app can still use:

- GDELT
- RSS feeds
- exchange announcement scraping

## 3. Install the Project

From the project root:

```powershell
python -m pip install -e .
```

If you also want to train and run the OHLCV prediction model:

```powershell
python -m pip install -e .[ml]
```

## 4. Optional Environment Variables

In PowerShell, you can set API keys like this:

```powershell
$env:GNEWS_API_KEY="your_key_here"
$env:NEWSDATA_API_KEY="your_key_here"
$env:ALPHAVANTAGE_API_KEY="your_key_here"
```

These variables apply to the current shell session.

## 5. Start the Application

Run:

```powershell
uvicorn stock_agents.app:app --reload
```

After startup, open:

- Health check: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI spec: `http://127.0.0.1:8000/openapi.json`

## 6. Main API Endpoints

### Stock context

- `POST /api/v1/context/stock`
- `POST /api/v1/context/watchlist`

These are the highest-value endpoints if you want a combined view of sentiment, events, and market cues.

### Individual data sources

- `POST /api/v1/news/company`
- `POST /api/v1/news/sector`
- `POST /api/v1/news/india-macro`
- `POST /api/v1/announcements/exchange`
- `POST /api/v1/markets/global`
- `POST /api/v1/markets/macro-cues`
- `POST /api/v1/fundamentals/company`
- `POST /api/v1/calendar/time-window`
- `POST /api/v1/feedback/predictions/store`
- `POST /api/v1/feedback/predictions/review`
- `POST /api/v1/feedback/memory`

### Model endpoints

- `POST /api/v1/model/train`
- `POST /api/v1/model/predict`
- `POST /api/v1/model/info`

## 6B. Feedback Loop and Agent Memory

The application can now learn from past prediction mistakes.

### What gets stored

- prediction records in `feedback_data/predictions/`
- review records in `feedback_data/reviews/`
- aggregated agent memory in `feedback_data/agent_memory.json`

### Main feedback endpoints

- `POST /api/v1/feedback/predictions/store`
- `POST /api/v1/feedback/predictions/review`
- `POST /api/v1/feedback/memory`

### How the loop works

1. Store the original prediction.
2. After the market closes, review the prediction with the actual move.
3. The system fetches stock context for the review window.
4. It checks whether:
   - company news was against the prediction
   - exchange announcements were missed
   - macro or global cues were ignored
   - event risk was underweighted
   - the model missed direction or move size
5. It updates the agent memory with recurring failure patterns.
6. Future stock-context and fundamentals responses include `memory_guidance`.

### Example store request

```json
{
  "stock": {
    "symbol": "RELIANCE",
    "name": "Reliance Industries",
    "nse_symbol": "RELIANCE"
  },
  "reference_price": 2450.0,
  "forecast_dates": ["2026-05-13"],
  "predicted_values": [2490.0],
  "prediction_source": "ohlcv_model",
  "confidence": 0.64
}
```

### Example review request

```json
{
  "prediction_id": "replace-with-stored-id",
  "actual_for_date": "2026-05-13T15:30:00+05:30",
  "actual_close": 2418.0,
  "previous_close": 2450.0,
  "auto_fetch_context": true
}
```

## 6A. Company Fundamentals Endpoint

Use `POST /api/v1/fundamentals/company` when you want a longer-term view of the company instead of short-term news flow.

This endpoint checks:

- valuation metrics such as P/E, PEG, price-to-book, EV multiples, and dividend yield
- profitability metrics such as margin, ROE, and ROA
- growth metrics such as revenue growth and earnings growth
- balance-sheet strength such as current ratio and debt-to-equity
- cash-flow quality such as operating cash flow and free cash flow

If the data provider uses a different ticker format than your normal trading symbol, send `fundamental_symbol`.

Example request:

```json
{
  "stock": {
    "symbol": "RELIANCE",
    "name": "Reliance Industries",
    "fundamental_symbol": "RELIANCE.NSE",
    "nse_symbol": "RELIANCE"
  },
  "options": {
    "timeout_seconds": 12
  }
}
```

## 7. Example Stock Context Request

Use this request body in Swagger UI or with `Invoke-RestMethod`:

```json
{
  "stock": {
    "symbol": "RELIANCE",
    "name": "Reliance Industries",
    "aliases": ["RIL", "Reliance Industries"],
    "sector": "Oil & Gas",
    "sector_keywords": ["refining", "petrochemicals", "telecom", "retail"],
    "fundamental_symbol": "RELIANCE.NSE",
    "nse_symbol": "RELIANCE"
  },
  "options": {
    "hours_before_previous_close": 2
  }
}
```

Example PowerShell call:

```powershell
$body = @'
{
  "stock": {
    "symbol": "RELIANCE",
    "name": "Reliance Industries",
    "aliases": ["RIL", "Reliance Industries"],
    "sector": "Oil & Gas",
    "sector_keywords": ["refining", "petrochemicals", "telecom", "retail"],
    "nse_symbol": "RELIANCE"
  },
  "options": {
    "hours_before_previous_close": 2
  }
}
'@

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/context/stock" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 8. How to Understand the Stock Context Output

The stock context response includes:

- `provider_status`
  Shows which providers returned data and which were skipped or failed.

- `ingestion`
  Raw normalized items grouped into:
  - company news
  - sector news
  - macro news
  - corporate announcements
  - global market performance
  - macro series

- `processing`
  Derived analysis such as:
  - detected events
  - relevance counts
  - sentiment by source bucket
  - global cues

- `report`
  Final trading summary including:
  - `label`
  - `overall_score`
  - `confidence`
  - `headline_count`
  - `key_drivers`
  - `trading_context`

### How to read the report

- A stronger positive `overall_score` suggests the combined news and market context is supportive.
- A negative `overall_score` suggests headwinds from news, announcements, or macro/global conditions.
- `confidence` depends on how much relevant information was found.
- `key_drivers` usually explains why the score moved positive or negative.

This is a context signal, not a guaranteed price forecast.

## 8A. How to Understand the Fundamentals Output

The fundamentals response is organized into:

- `valuation`
- `profitability`
- `growth`
- `financial_strength`
- `cash_flow`
- `earnings_quality`
- `report`

### Key long-term metrics to watch

- `pe_ratio`
  A rough valuation multiple. Lower is usually cheaper, but only meaningful relative to growth and sector quality.

- `peg_ratio`
  Helps compare valuation to growth. Lower is usually better.

- `return_on_equity_pct`
  Shows how efficiently the company uses shareholder capital. Higher is generally better.

- `operating_margin_pct`
  Indicates pricing power and operating discipline.

- `annual_revenue_growth_pct`
  Shows whether the business is still expanding.

- `debt_to_equity`
  Higher leverage raises balance-sheet risk.

- `free_cash_flow`
  Positive free cash flow usually improves long-term durability.

### How to read the `report`

- `overall_score`
  A composite long-term quality and valuation score from `0` to `100`.

- `label`
  Quick classification such as `strong`, `constructive`, `mixed`, `cautious`, or `weak`.

- `insights`
  Positive long-term factors identified from the available financial data.

- `risks`
  Long-term concerns such as expensive valuation, weak growth, leverage, or negative free cash flow.

## 9. OHLCV Data Required for Prediction

The forecasting model needs your own OHLCV JSON file.
This repository does not currently include a sample `ohlv.json` file.

Each candle should look like:

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

Supported payload patterns include:

- a top-level candle list
- `{ "candles": [...] }`
- an Upstox-style nested structure

## 10. Train the Forecasting Model

Train one model per stock.

Example:

```powershell
python trend_analyzer/ohlcv_lstm_cnn.py train `
  --json path\to\reliance_ohlcv.json `
  --model-dir trend_analyzer/models/reliance `
  --stock-name RELIANCE
```

Useful optional arguments:

```powershell
python trend_analyzer/ohlcv_lstm_cnn.py train `
  --json path\to\reliance_ohlcv.json `
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

### What training produces

Inside the model directory:

- `model.keras`
- `metadata.json`
- `training_history.json`
- `test_metrics.json`
- `test_predictions.json`

## 11. Predict Future Trend

Once a model is trained, run:

```powershell
python trend_analyzer/ohlcv_lstm_cnn.py predict `
  --json path\to\reliance_latest_ohlcv.json `
  --model-dir trend_analyzer/models/reliance
```

This creates:

- `forecast.json`

The prediction uses:

- the latest `window-size` rows from your OHLCV file
- the saved model and normalization metadata
- the configured forecast horizon, which is `5` trading days by default

## 12. Predict Through the API

Start the FastAPI app, then call:

```json
{
  "json_path": "path/to/reliance_latest_ohlcv.json",
  "model_dir": "trend_analyzer/models/reliance"
}
```

Example PowerShell call:

```powershell
$body = @'
{
  "json_path": "path/to/reliance_latest_ohlcv.json",
  "model_dir": "trend_analyzer/models/reliance"
}
'@

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/model/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 13. How to Understand the Forecast Output

The prediction result contains fields like:

- `stock_name`
- `target_field`
- `latest_input_date`
- `forecast_dates`
- `predicted_values`
- `forecast_path`

Example meaning:

- `forecast_dates`
  The future trading days being predicted.

- `predicted_values`
  The predicted values for the selected target field, usually `close`.

### Interpreting trend direction

If the predicted values are generally increasing across forecast dates:

- the model is suggesting a short-term upward trend

If the predicted values are generally decreasing:

- the model is suggesting a short-term downward trend

If the values are flat or mixed:

- the model is suggesting weak momentum or uncertainty

### Example interpretation

If `predicted_values` are:

```json
[2450.0, 2475.0, 2492.0, 2510.0, 2528.0]
```

that implies the model expects a gradual upward move in the forecast window.

If they are:

```json
[2450.0, 2435.0, 2420.0, 2412.0, 2405.0]
```

that implies a weakening short-term trend.

## 14. How to Judge Whether the Model Is Trustworthy

Do not trust a forecast by itself.
Check these files after training:

- `test_metrics.json`
- `test_predictions.json`
- `training_history.json`

### Key metrics

- `MAE`
  Average absolute prediction error. Lower is better.

- `RMSE`
  Penalizes larger misses more heavily. Lower is better.

- `MAPE`
  Average percentage error. Lower is better.

### Practical reading

- If test errors are large compared with the stock's normal daily movement, the forecast is weak.
- If predictions on the test set consistently lag or overshoot trend changes, retraining or more history may be needed.
- If the stock is highly event-driven, a pure OHLCV model will often miss sudden moves.

## 15. Best Way to Understand Future Trend in This Repo

Use both systems together:

1. Run `/api/v1/context/stock`
   This tells you whether the current news and macro setup is positive or negative.

2. Run the OHLCV forecast model
   This tells you whether recent price-volume history points up, down, or sideways.

3. Compare both results
   - positive context + upward forecast is stronger confirmation
   - negative context + downward forecast is stronger confirmation
   - disagreement means the setup is less clear

## 16. Common Workflow

1. Install the project.
2. Start the FastAPI server.
3. Check a stock with `/api/v1/context/stock`.
4. Prepare an OHLCV JSON file for that stock.
5. Train a model in a dedicated model directory.
6. Run prediction using the latest candle file.
7. Review:
   - forecast direction
   - test metrics
   - context report
8. Decide whether news context and price-history trend agree.

## 17. Important Limitations

- The forecast is based on historical OHLCV patterns, not fundamental valuation.
- News sentiment and event scoring are heuristic, not institutional-grade research.
- The model predicts one field at a time, usually `close`.
- A 5-day forecast is short-term guidance, not long-term investing advice.
- Sudden events such as results, regulation, war, or management actions can invalidate the forecast quickly.

## 18. Recommended Usage

Use this repo as a decision-support tool:

- for short-term market context
- for comparing multiple stocks in a watchlist
- for estimating near-term direction from recent OHLCV behavior

Do not use it as a standalone buy/sell automation system without additional validation, risk controls, and live monitoring.
