# Autonomous Stock Agents API

Agentic AI market-analysis and prediction service for Indian stocks. Its main
endpoint runs a complete prediction workflow in one request:

```text
OHLCV forecast + news/events + fundamentals + financial-document RAG + memory
                                  |
                                  v
                   calibrated ensemble prediction
```

The individual context, fundamentals, model, and feedback APIs remain available
for diagnostics and direct use.

## What the Application Does

The application brings several kinds of market information into one API:

- short-term company, sector, and macro news
- NSE and BSE corporate announcements
- global equity-market, commodity, currency, and bond-yield signals
- company valuation, profitability, growth, balance-sheet, and cash-flow analysis
- CNN-LSTM forecasts generated from historical OHLCV data
- prediction reviews and stock-specific feedback memory

It can analyze one stock or an entire watchlist. The output includes the source
items, provider status, detected events, sentiment scores, important market
cues, confidence, and a bullish, neutral, or bearish trading context.

The project uses an agentic modular-monolith design. FastAPI exposes the HTTP
interface, Pydantic validates requests, specialist agents operate the existing
analysis tools, TensorFlow handles OHLCV forecasting, and JSON files currently
store model artifacts, prediction feedback, agent memory, and the local
financial-document index.

The autonomous orchestrator creates and executes a prediction plan without the
caller coordinating separate APIs. Its decisions remain inspectable because
every agent reports its score, confidence, evidence, status, and contribution.
The current agents use explicit financial and statistical policies rather than
an external LLM, which keeps the prediction path reproducible.

### What "Autonomous" Means

For each request, the orchestrator independently:

1. Loads stock-specific prediction memory.
2. Runs the saved CNN-LSTM model.
3. Collects and analyzes current market context.
4. Runs fundamental analysis when enabled.
5. Retrieves relevant financial-document chunks with citations when enabled.
6. Recalibrates weights using historical accuracy and failure patterns.
7. Resolves disagreements between agents.
8. Produces one prediction with confidence, rationale, and document evidence.
9. Stores that prediction for later evaluation when automatic storage is enabled.

The service does not currently schedule itself or automatically obtain future
closing prices. After the predicted date, a caller must submit the actual
outcome to the review endpoint. That review updates memory used by subsequent
autonomous predictions.

## How It Works

### Autonomous Agentic Prediction Flow

Use `POST /api/v1/predictions/autonomous` to execute the complete workflow:

```text
Autonomous orchestrator
    |
    +--> Memory agent
    |      - loads reviewed outcomes and failure patterns
    |      - recalibrates weights and confidence
    |
    +--> Technical agent
    |      - runs the saved CNN-LSTM model
    |      - converts the forecast into direction and confidence
    |
    +--> Market-context agent
    |      - analyzes news, announcements, events, macro data, and global cues
    |
    +--> Fundamental agent
    |      - evaluates valuation, quality, growth, leverage, and cash flow
    |
    +--> Financial-document RAG agent
    |      - searches stock-scoped reports, transcripts, presentations, and filings
    |      - returns cited evidence and a document signal
    |
    v
Decision agent
    - normalizes available signals
    - applies memory-adjusted weights
    - measures agent agreement and data coverage
    - produces one direction, expected change, rationale, and confidence
    |
    v
Prediction action agent
    - stores the ensemble prediction automatically
    - makes it available for later outcome review and learning
```

The orchestrator degrades gracefully. If an optional provider, fundamental
agent, or document index is unavailable, it records the failure, renormalizes
the remaining usable signals, and reduces confidence based on missing coverage.
It fails the request only when no specialist agent produces a usable signal.

The initial ensemble policy is:

| Agent | Base weight | Main inputs |
| --- | ---: | --- |
| Technical | 42% | CNN-LSTM OHLCV forecast and saved test metrics |
| Market context | 28% | News sentiment, events, announcements, macro and global cues |
| Fundamentals | 15% | Valuation, growth, profitability, leverage and cash flow |
| Financial documents | 15% | Retrieved annual reports, results, transcripts, presentations, filings, and rating reports |

These are starting weights, not fixed final weights. Agent confidence and
feedback memory modify them before every decision. The decision agent also
reduces final confidence when agents disagree or planned signals are missing.

### Market Context Flow

```text
Client request
    |
    v
FastAPI endpoint and Pydantic validation
    |
    v
Indian market time-window calculation
    |
    v
Provider ingestion
  - company and sector news
  - India macro news
  - NSE/BSE announcements
  - global markets and macro series
    |
    v
Normalization and processing
  - stock-name and alias matching
  - relevance scoring
  - duplicate removal
  - keyword-based sentiment scoring
  - corporate-event detection
    |
    v
Weighted signal combination
    |
    v
Bullish, neutral, or bearish context response
```

Company news has the largest weight in the final context score. Sector news,
macro news, corporate announcements, detected events, and global cues provide
the remaining contributions. The response also reports the underlying evidence
so callers can inspect how the result was produced.

### OHLCV Forecasting Flow

```text
OHLCV JSON or inline data
    |
    v
Validation and chronological sorting
    |
    v
Rolling input windows and future targets
    |
    v
Chronological train/test split and normalization
    |
    v
Conv1D layers -> LSTM layers -> Dense forecast
    |
    v
Saved model, metrics, test predictions, and future forecast
```

The convolution layers identify local patterns in price and volume data. The
LSTM layers model their sequence over time, and the final dense layer predicts
the configured number of future values. By default, the model uses a 30-candle
window and predicts the next five trading days.

### Fundamental Analysis Flow

The fundamentals endpoint requests company overview, income statement, balance
sheet, cash-flow, and earnings data from Alpha Vantage. It derives valuation,
profitability, growth, liquidity, leverage, cash-flow, and earnings-quality
metrics. Explicit scoring rules then produce a long-term score, confidence,
strengths, and risks.

### Financial-Document RAG Flow

```text
PDF, TXT, MD, CSV, JSON, or HTML
    |
    v
Text extraction and document validation
    |
    v
Stock-scoped chunking and local persistence
    |
    v
Lexical relevance retrieval for the current stock question
    |
    v
Cited excerpts + document signal + confidence
    |
    v
Financial-document agent contribution to the decision
```

Documents are stored under `rag_data/<SYMBOL>/<document_id>/`. Each document
has metadata and chunk files. Duplicate content is detected by SHA-256 hash.
Retrieval results include document title, type, chunk ID, relevance score, and
an excerpt so the decision remains auditable.

The current RAG implementation is local and deterministic. It performs
stock-scoped lexical retrieval and rule-based financial evidence scoring. It
does not send uploaded documents to an external LLM or vector database.

### Feedback and Memory Flow

```text
Store prediction
    |
    v
Submit the actual market result
    |
    v
Compare predicted and actual direction/magnitude
    |
    v
Fetch the market context for the reviewed period
    |
    v
Diagnose recurring failure patterns
    |
    v
Update global and stock-specific memory
```

Memory guidance is attached to stock-context and fundamental-analysis
responses. In the autonomous prediction flow, memory is also operational:

- reviewed direction accuracy adjusts the technical-agent weight
- repeated direction misses reduce dependence on the OHLCV model
- underweighted news, announcement, macro, and event patterns increase the
  market-context weight
- underweighted risks or tailwinds found in retrieved financial documents
  increase the financial-document agent's future weight
- repeated magnitude errors lower final confidence
- prediction accuracy is tracked separately by prediction source

Memory recalibrates ensemble behavior but does not automatically retrain the
CNN-LSTM model. Model retraining remains an explicit operation.

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
- `POST /api/v1/fundamentals/company`
- `POST /api/v1/feedback/predictions/store`
- `POST /api/v1/feedback/predictions/review`
- `POST /api/v1/feedback/memory`
- `POST /api/v1/rag/documents/ingest`
- `POST /api/v1/rag/documents/query`
- `POST /api/v1/rag/documents/list`
- `POST /api/v1/predictions/autonomous`
- `POST /api/v1/model/train`
- `POST /api/v1/model/predict`
- `POST /api/v1/model/info`
- `POST /api/v1/model/results`

## Providers

Broad India business news:

- `GNews` via `GNEWS_API_KEY`
- `NewsData.io` via `NEWSDATA_API_KEY`
- `GDELT` without an API key
- RSS feeds from Indian business publishers

Global markets, commodities, FX, and yields:

- `Alpha Vantage` via `ALPHAVANTAGE_API_KEY`

Fundamental and valuation reports:

- `Alpha Vantage` via `ALPHAVANTAGE_API_KEY`

## Run

Create and activate a virtual environment before the first run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -e .
python -m uvicorn stock_agents.app:app --reload
```

Open:

- Web control room: `http://127.0.0.1:8000/ui`
- Health check: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI document: `http://127.0.0.1:8000/openapi.json`

Provider API keys are optional, but missing keys reduce signal coverage:

```powershell
$env:GNEWS_API_KEY = "..."
$env:NEWSDATA_API_KEY = "..."
$env:ALPHAVANTAGE_API_KEY = "..."
```

## Web Control Room

The built-in UI at `/ui` provides the complete workflow without manually
constructing API requests:

1. Read the architecture walkthrough and each agent's responsibility.
2. Follow the connected flow chart from data and documents through feedback memory.
3. Drag and drop an OHLCV JSON file and inspect its date range and latest close.
4. Upload financial documents and test stock-scoped retrieval with citations.
5. Configure and start CNN-LSTM training.
6. View MAE, RMSE, MAPE, training/validation loss, and test-window comparisons.
7. Load saved training artifacts later using the model directory.
8. Run the autonomous technical, context, fundamental, document-RAG, memory, and decision agents.
9. Inspect final weights, signal coverage, agreement, forecast, citations, and rationale.
10. Submit an actual close and compare the realized move with the prediction.

The browser parses the uploaded JSON locally and sends it to the API as
`ohlcv_data`. The raw upload is not separately copied into an uploads folder.
Training still writes the model and result artifacts to the configured
`model_dir`.

Financial documents are sent as base64 JSON payloads and indexed under
`rag_data/`. The UI accepts PDF, TXT, Markdown, CSV, JSON, and HTML documents up
to 25 MB.

## Financial Document Knowledge Base

Index a document:

```json
{
  "stock": {
    "symbol": "RELIANCE",
    "name": "Reliance Industries"
  },
  "filename": "reliance-annual-report.pdf",
  "document_type": "annual_report",
  "title": "Reliance FY 2025-26 Annual Report",
  "content_base64": "<base64 document bytes>"
}
```

Use `POST /api/v1/rag/documents/ingest` for ingestion. Supported
`document_type` values are:

- `annual_report`
- `quarterly_result`
- `earnings_transcript`
- `investor_presentation`
- `exchange_filing`
- `credit_rating`
- `other`

Retrieve evidence directly:

```json
{
  "stock": {
    "symbol": "RELIANCE",
    "name": "Reliance Industries"
  },
  "query": "What do the latest documents say about margins, cash flow, debt, guidance, and risks?",
  "top_k": 5
}
```

Use `POST /api/v1/rag/documents/query`. Use
`POST /api/v1/rag/documents/list` to inspect indexed document metadata.

## OHLCV Model Training

The base project installation includes NumPy and TensorFlow.

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
- `POST /api/v1/model/results`

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

## Autonomous Integrated Prediction

The autonomous endpoint runs the technical, market-context, fundamental,
financial-document RAG, memory, decision, and persistence agents in one
request:

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
  "json_path": "trend_analyzer/ohlv.json",
  "model_dir": "trend_analyzer/models/reliance",
  "include_fundamentals": true,
  "include_financial_documents": true,
  "rag_query": "Latest outlook, risks, margins, cash flow, debt and guidance",
  "rag_top_k": 5,
  "auto_store_prediction": true,
  "options": {
    "hours_before_previous_close": 2
  }
}
```

Provide exactly one OHLCV source:

- `json_path`: path to an OHLCV JSON file
- `ohlcv_data`: inline OHLCV payload

The saved model must target `open`, `high`, `low`, or `close`. The autonomous
price flow rejects models whose target is `volume`.

The response contains:

- the autonomous execution plan and per-agent trace
- complete technical, context, fundamental, and financial-document agent outputs
- retrieved document citations and excerpts
- memory-derived weight and confidence calibration
- final signal contributions and unavailable-agent diagnostics
- one predicted direction, expected percentage move, and confidence
- the stored prediction ID used by the feedback review endpoint

Example response structure:

```json
{
  "mode": "autonomous_agentic_ensemble",
  "autonomous_plan": [
    "memory_agent",
    "technical_agent",
    "market_context_agent",
    "fundamental_agent",
    "financial_document_agent",
    "decision_agent",
    "prediction_action_agent"
  ],
  "memory_calibration": {
    "review_count": 8,
    "smoothed_accuracy": 0.5833,
    "weight_multipliers": {
      "technical": 0.92,
      "market_context": 1.18,
      "fundamentals": 1.0,
      "financial_documents": 1.12
    },
    "confidence_multiplier": 0.9
  },
  "decision": {
    "predicted_direction": "up",
    "predicted_change_pct": 1.24,
    "decision_score": 0.31,
    "confidence": 0.68,
    "agreement": 0.74,
    "signal_coverage": 1.0
  },
  "autonomous_actions": {
    "store_prediction": {
      "status": "completed",
      "prediction_id": "generated-uuid",
      "review_status": "pending"
    }
  }
}
```

`predicted_change_pct` is the weighted ensemble estimate. It is not simply the
first raw CNN-LSTM forecast.

## Example request

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

## Company Fundamentals

Use `POST /api/v1/fundamentals/company` to fetch valuation and financial-statement context for a single company. The response includes:

- P/E, PEG, price-to-book, price-to-sales, EV multiples, dividend yield
- margins, ROA, ROE, revenue and earnings growth
- balance-sheet and cash-flow checks such as current ratio, debt-to-equity, and free cash flow
- a long-term summary with strengths, risks, and a score

If the fundamentals provider needs a different ticker format than your trading symbol, pass it in `stock.fundamental_symbol`.

## Feedback Loop and Agent Memory

Autonomous predictions are stored by default. Complete the learning loop after
the market result is known:

1. Run `POST /api/v1/predictions/autonomous`.
2. Keep the returned `prediction_id`.
3. Submit the actual result to `POST /api/v1/feedback/predictions/review`.
4. Inspect learned state with `POST /api/v1/feedback/memory`.

Example review:

```json
{
  "prediction_id": "generated-uuid",
  "previous_close": 1450.0,
  "actual_close": 1478.5,
  "auto_fetch_context": true
}
```

What it does:

- saves each prediction to `feedback_data/predictions/`
- saves each review to `feedback_data/reviews/`
- updates `feedback_data/agent_memory.json`
- analyzes whether company news, announcements, macro cues, or event risk explained a wrong call
- tracks reviewed accuracy separately for the OHLCV model and autonomous ensemble
- changes future technical, context, and financial-document weights and ensemble confidence
- returns memory guidance that is also attached to `/api/v1/context/stock`, `/api/v1/context/watchlist`, and `/api/v1/fundamentals/company`

Manual predictions can still be stored with
`POST /api/v1/feedback/predictions/store`.
