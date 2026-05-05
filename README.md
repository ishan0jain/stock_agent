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
