from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from stock_agents.agentic_prediction import run_autonomous_prediction
from stock_agents.feedback import get_agent_memory, get_agent_memory_guidance, review_prediction, store_prediction
from stock_agents.financial_documents import (
    ingest_financial_document,
    list_financial_documents,
    query_financial_documents,
)
from stock_agents.fundamentals import analyze_company_fundamentals
from stock_agents.model_service import (
    get_ohlcv_model_info,
    get_ohlcv_model_results,
    predict_ohlcv_model,
    train_ohlcv_model,
)
from stock_agents.models import (
    AgentMemoryRequest,
    AutonomousPredictionRequest,
    FinancialDocumentIngestRequest,
    FinancialDocumentListRequest,
    FinancialDocumentQueryRequest,
    ModelInfoRequest,
    ModelPredictRequest,
    ModelTrainRequest,
    PredictionReviewRequest,
    PredictionStoreRequest,
    StockAnalysisRequest,
    TimeWindowRequest,
    WatchlistAnalysisRequest,
)
from stock_agents.service import (
    IST,
    analyze_stock,
    analyze_watchlist,
    derive_time_window,
    fetch_company_news,
    fetch_exchange_announcements,
    fetch_global_market_performance,
    fetch_macro_news,
    fetch_macro_series,
    fetch_sector_news,
)


UI_DIR = Path(__file__).resolve().parent / "ui"

app = FastAPI(
    title="Autonomous Stock Agents API",
    version="0.1.0",
    description=(
        "Agentic stock analysis combining OHLCV forecasting, market context, fundamentals, "
        "and feedback-memory calibration."
    ),
)
app.mount("/ui/assets", StaticFiles(directory=UI_DIR), name="ui-assets")


def request_since(request: StockAnalysisRequest | TimeWindowRequest) -> tuple[dict, datetime]:
    window = derive_time_window(request.options)
    since = request.options.since or datetime.fromisoformat(window.window_start)
    if since.tzinfo is None:
        since = since.replace(tzinfo=IST)
    else:
        since = since.astimezone(IST)
    return window.__dict__, since


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def ui_console() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/calendar/time-window")
def time_window(request: TimeWindowRequest) -> dict:
    return {"time_window": derive_time_window(request.options).__dict__}


@app.post("/api/v1/news/company")
def company_news(request: StockAnalysisRequest) -> dict:
    window, since = request_since(request)
    items, statuses = fetch_company_news(stock=request.stock, since=since, options=request.options)
    return {
        "stock": request.stock.model_dump(),
        "time_window": window,
        "provider_status": [status.__dict__ for status in statuses],
        "items": [item.__dict__ for item in items],
    }


@app.post("/api/v1/news/sector")
def sector_news(request: StockAnalysisRequest) -> dict:
    window, since = request_since(request)
    items, statuses = fetch_sector_news(stock=request.stock, since=since, options=request.options)
    return {
        "stock": request.stock.model_dump(),
        "time_window": window,
        "provider_status": [status.__dict__ for status in statuses],
        "items": [item.__dict__ for item in items],
    }


@app.post("/api/v1/news/india-macro")
def macro_news(request: TimeWindowRequest) -> dict:
    window, since = request_since(request)
    items, statuses = fetch_macro_news(since=since, options=request.options)
    return {
        "time_window": window,
        "provider_status": [status.__dict__ for status in statuses],
        "items": [item.__dict__ for item in items],
    }


@app.post("/api/v1/announcements/exchange")
def exchange_announcements(request: StockAnalysisRequest) -> dict:
    window, since = request_since(request)
    items, statuses = fetch_exchange_announcements(stock=request.stock, since=since, options=request.options)
    return {
        "stock": request.stock.model_dump(),
        "time_window": window,
        "provider_status": [status.__dict__ for status in statuses],
        "items": [item.__dict__ for item in items],
    }


@app.post("/api/v1/markets/global")
def global_markets(request: TimeWindowRequest) -> dict:
    items, statuses = fetch_global_market_performance(request.options)
    return {
        "provider_status": [status.__dict__ for status in statuses],
        "items": [item.__dict__ for item in items],
    }


@app.post("/api/v1/markets/macro-cues")
def macro_cues(request: TimeWindowRequest) -> dict:
    items, statuses = fetch_macro_series(request.options)
    return {
        "provider_status": [status.__dict__ for status in statuses],
        "items": [item.__dict__ for item in items],
    }


@app.post("/api/v1/context/stock")
def stock_context(request: StockAnalysisRequest) -> dict:
    result = analyze_stock(request.stock, request.options)
    result["memory_guidance"] = get_agent_memory_guidance(request.stock)
    return result


@app.post("/api/v1/context/watchlist")
def watchlist_context(request: WatchlistAnalysisRequest) -> dict:
    result = analyze_watchlist(request.stocks, request.options)
    for input_stock, output_stock in zip(request.stocks, result.get("stocks", []), strict=False):
        output_stock["memory_guidance"] = get_agent_memory_guidance(input_stock)
    return result


@app.post("/api/v1/fundamentals/company")
def company_fundamentals(request: StockAnalysisRequest) -> dict:
    result = analyze_company_fundamentals(request.stock, request.options)
    result["memory_guidance"] = get_agent_memory_guidance(request.stock)
    return result


@app.post("/api/v1/feedback/predictions/store")
def feedback_store_prediction(request: PredictionStoreRequest) -> dict:
    return store_prediction(request)


@app.post("/api/v1/feedback/predictions/review")
def feedback_review_prediction(request: PredictionReviewRequest) -> dict:
    try:
        return review_prediction(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/feedback/memory")
def feedback_memory(request: AgentMemoryRequest) -> dict:
    return get_agent_memory(request)


@app.post("/api/v1/rag/documents/ingest")
def rag_document_ingest(request: FinancialDocumentIngestRequest) -> dict:
    try:
        return ingest_financial_document(request)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/rag/documents/query")
def rag_document_query(request: FinancialDocumentQueryRequest) -> dict:
    try:
        return query_financial_documents(request)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/rag/documents/list")
def rag_document_list(request: FinancialDocumentListRequest) -> dict:
    try:
        return list_financial_documents(request)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/predictions/autonomous")
def autonomous_prediction(request: AutonomousPredictionRequest) -> dict:
    try:
        return run_autonomous_prediction(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/model/train")
def model_train(request: ModelTrainRequest) -> dict:
    try:
        return train_ohlcv_model(
            model_dir=request.model_dir,
            stock_name=request.stock_name,
            window_size=request.window_size,
            horizon=request.horizon,
            target_field=request.target_field,
            epochs=request.epochs,
            batch_size=request.batch_size,
            train_ratio=request.train_ratio,
            validation_split=request.validation_split,
            json_path=request.json_path,
            ohlcv_data=request.ohlcv_data,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SystemExit as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/model/predict")
def model_predict(request: ModelPredictRequest) -> dict:
    try:
        return predict_ohlcv_model(
            model_dir=request.model_dir,
            json_path=request.json_path,
            ohlcv_data=request.ohlcv_data,
            output_path=request.output_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SystemExit as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/model/info")
def model_info(request: ModelInfoRequest) -> dict:
    try:
        return get_ohlcv_model_info(request.model_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/model/results")
def model_results(request: ModelInfoRequest) -> dict:
    try:
        return get_ohlcv_model_results(request.model_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
