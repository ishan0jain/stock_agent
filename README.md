# Stock Agents API

FastAPI service for a stock or watchlist that returns:

- company news
- sector news
- India macro/business news
- NSE/BSE corporate announcements
- global market performance
- commodity, currency, and bond-yield cues
- normalized, deduplicated, stock-wise trading context

## Endpoints

- `POST /api/v1/calendar/time-window`
- `POST /api/v1/news/company`
- `POST /api/v1/news/sector`
- `POST /api/v1/news/india-macro`
- `POST /api/v1/announcements/exchange`
- `POST /api/v1/markets/global`
- `POST /api/v1/markets/macro-cues`
- `POST /api/v1/context/stock`
- `POST /api/v1/context/watchlist`

## Providers

Broad India business news:

- `GNews` via `GNEWS_API_KEY`
- `NewsData.io` via `NEWSDATA_API_KEY`
- `GDELT` without an API key
- RSS feeds from Indian business publishers

Global markets, commodities, FX, and yields:

- `Alpha Vantage` via `ALPHAVANTAGE_API_KEY`

## Run

```powershell
python -m pip install -e .
uvicorn stock_agents.app:app --reload
```

## OHLCV Model Training

Install the optional ML dependency:

```powershell
python -m pip install -e .[ml]
```

Train a CNN-LSTM model on an OHLCV JSON file and save the model:

```powershell
python trend_analyzer/ohlcv_lstm_cnn.py train --json trend_analyzer/ohlv.json --model-dir trend_analyzer/models/reliance --stock-name RELIANCE
```

Load the saved model and forecast the next 5 trading days from the latest rows in the JSON file:

```powershell
python trend_analyzer/ohlcv_lstm_cnn.py predict --json trend_analyzer/ohlv.json --model-dir trend_analyzer/models/reliance
```

Model API endpoints:

- `POST /api/v1/model/train`
- `POST /api/v1/model/predict`
- `POST /api/v1/model/info`

Example train request:

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

Example predict request:

```json
{
  "json_path": "trend_analyzer/ohlv.json",
  "model_dir": "trend_analyzer/models/reliance"
}
```

## Example request

```json
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
```
