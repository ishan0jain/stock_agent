from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from stock_agents.models import AgentMemoryRequest, AnalysisOptions, PredictionReviewRequest, PredictionStoreRequest, StockInput
from stock_agents.service import IST, analyze_stock


FEEDBACK_ROOT = Path(__file__).resolve().parent.parent / "feedback_data"
PREDICTIONS_DIR = FEEDBACK_ROOT / "predictions"
REVIEWS_DIR = FEEDBACK_ROOT / "reviews"
MEMORY_PATH = FEEDBACK_ROOT / "agent_memory.json"

PATTERN_MESSAGES: dict[str, str] = {
    "negative_company_news_underweighted": "Reduce bullish conviction when company-news sentiment is negative.",
    "positive_company_news_underweighted": "Reduce bearish conviction when company-news sentiment is positive.",
    "negative_announcements_underweighted": "Treat exchange announcements as first-class signals during the review window.",
    "positive_announcements_underweighted": "Positive company announcements can overpower a weak price-history setup.",
    "macro_headwind_underweighted": "Macro and global cues should trim long bias when they turn negative.",
    "macro_tailwind_underweighted": "Macro and global cues should trim short bias when they turn positive.",
    "event_risk_underweighted": "Major corporate events should widen the expected move band and lower confidence.",
    "context_disagreed_with_prediction": "When context and model direction disagree, lower size or wait for confirmation.",
    "move_magnitude_underestimated": "Recent logic is underestimating move size; widen expected ranges.",
    "move_magnitude_overestimated": "Recent logic is overestimating move size; trim expected ranges.",
    "direction_miss": "Direction misses need lower confidence and stronger confirmation rules.",
}


def store_prediction(request: PredictionStoreRequest) -> dict[str, Any]:
    ensure_feedback_dirs()
    prediction_id = str(uuid.uuid4())
    created_at = normalize_timestamp(request.prediction_made_at)
    target_date = derive_target_date(request)
    predicted_value = derive_predicted_value(request, target_date)
    predicted_change_pct = derive_predicted_change_pct(request, predicted_value)
    predicted_direction = request.predicted_direction or direction_from_change(predicted_change_pct)

    record = {
        "prediction_id": prediction_id,
        "created_at": created_at,
        "prediction_for_date": target_date,
        "stock": request.stock.model_dump(),
        "reference_price": request.reference_price,
        "target_field": request.target_field,
        "forecast_dates": request.forecast_dates,
        "predicted_values": request.predicted_values,
        "predicted_value_for_date": predicted_value,
        "predicted_direction": predicted_direction,
        "predicted_change_pct": predicted_change_pct,
        "prediction_source": request.prediction_source,
        "confidence": request.confidence,
        "model_dir": request.model_dir,
        "notes": request.notes,
        "review_status": "pending",
    }
    write_json(PREDICTIONS_DIR / f"{prediction_id}.json", record)
    return {
        "prediction": record,
        "memory_guidance": get_agent_memory_guidance(request.stock),
    }


def review_prediction(request: PredictionReviewRequest) -> dict[str, Any]:
    ensure_feedback_dirs()
    prediction = load_prediction(request.prediction_id)
    stock = StockInput(**prediction["stock"])
    reviewed_date = normalize_review_date(request, prediction)
    predicted_change_pct = to_float(prediction.get("predicted_change_pct"))
    predicted_direction = str(prediction.get("predicted_direction") or "flat")
    actual_change_pct = derive_actual_change_pct(request, prediction)
    if actual_change_pct is None:
        raise ValueError(
            "could not derive actual_change_pct; provide actual_change_pct explicitly or send actual_close with previous_close/reference_price"
        )
    actual_direction = direction_from_change(actual_change_pct)
    direction_correct = predicted_direction == actual_direction

    price_error_pct = None
    if predicted_change_pct is not None and actual_change_pct is not None:
        price_error_pct = round(actual_change_pct - predicted_change_pct, 3)

    context_report = None
    context_summary = None
    if request.auto_fetch_context:
        context_options = build_review_context_options(request.options, prediction, reviewed_date)
        context_report = analyze_stock(stock, context_options)
        context_summary = summarize_context_report(context_report)

    diagnosis = diagnose_prediction_miss(
        predicted_direction=predicted_direction,
        actual_direction=actual_direction,
        predicted_change_pct=predicted_change_pct,
        actual_change_pct=actual_change_pct,
        context_summary=context_summary,
    )

    review = {
        "prediction_id": request.prediction_id,
        "reviewed_at": datetime.now(IST).isoformat(),
        "actual_for_date": reviewed_date,
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "predicted_change_pct": predicted_change_pct,
        "actual_change_pct": actual_change_pct,
        "price_error_pct": price_error_pct,
        "direction_correct": direction_correct,
        "outcome": "correct" if direction_correct else "incorrect",
        "diagnosis": diagnosis,
        "analyst_notes": request.analyst_notes,
        "context_summary": context_summary,
    }
    write_json(REVIEWS_DIR / f"{request.prediction_id}.json", review)

    prediction["review_status"] = review["outcome"]
    prediction["last_reviewed_at"] = review["reviewed_at"]
    prediction["review_file"] = str((REVIEWS_DIR / f"{request.prediction_id}.json").as_posix())
    write_json(PREDICTIONS_DIR / f"{request.prediction_id}.json", prediction)

    memory = update_agent_memory(stock, review)
    return {
        "prediction": prediction,
        "review": review,
        "memory_guidance": build_memory_guidance(stock, memory),
        "context_report": context_report,
    }


def get_agent_memory(request: AgentMemoryRequest) -> dict[str, Any]:
    return get_agent_memory_guidance(request.stock)


def get_agent_memory_guidance(stock: StockInput) -> dict[str, Any]:
    memory = load_memory()
    return build_memory_guidance(stock, memory)


def build_memory_guidance(stock: StockInput, memory: dict[str, Any]) -> dict[str, Any]:
    stock_memory = memory.get("stocks", {}).get(stock.symbol.upper(), {})
    global_memory = memory.get("global", {})
    pattern_counts = stock_memory.get("failure_patterns", {})
    ranked_patterns = sorted(pattern_counts.items(), key=lambda row: row[1], reverse=True)
    lessons = [PATTERN_MESSAGES[key] for key, _count in ranked_patterns if key in PATTERN_MESSAGES][:5]

    accuracy = None
    if stock_memory.get("review_count"):
        accuracy = round(stock_memory.get("correct_count", 0) / stock_memory["review_count"], 3)

    return {
        "stock": stock.symbol,
        "review_count": stock_memory.get("review_count", 0),
        "correct_count": stock_memory.get("correct_count", 0),
        "accuracy": accuracy,
        "failure_patterns": pattern_counts,
        "lessons": lessons,
        "global_review_count": global_memory.get("review_count", 0),
        "updated_at": memory.get("updated_at"),
    }


def update_agent_memory(stock: StockInput, review: dict[str, Any]) -> dict[str, Any]:
    memory = load_memory()
    memory["updated_at"] = datetime.now(IST).isoformat()

    global_memory = memory.setdefault("global", {"review_count": 0, "correct_count": 0, "failure_patterns": {}})
    stock_memory = memory.setdefault("stocks", {}).setdefault(
        stock.symbol.upper(),
        {
            "symbol": stock.symbol.upper(),
            "name": stock.name,
            "review_count": 0,
            "correct_count": 0,
            "failure_patterns": {},
            "recent_reviews": [],
        },
    )

    global_memory["review_count"] += 1
    stock_memory["review_count"] += 1
    if review.get("direction_correct"):
        global_memory["correct_count"] += 1
        stock_memory["correct_count"] += 1

    for pattern_key in review.get("diagnosis", {}).get("patterns", []):
        increment_pattern(global_memory["failure_patterns"], pattern_key)
        increment_pattern(stock_memory["failure_patterns"], pattern_key)

    summary_row = {
        "prediction_id": review["prediction_id"],
        "actual_for_date": review["actual_for_date"],
        "outcome": review["outcome"],
        "primary_reason": review["diagnosis"]["reasons"][0] if review["diagnosis"]["reasons"] else None,
        "reviewed_at": review["reviewed_at"],
    }
    stock_memory["recent_reviews"] = [summary_row, *stock_memory.get("recent_reviews", [])][:10]

    write_json(MEMORY_PATH, memory)
    return memory


def diagnose_prediction_miss(
    *,
    predicted_direction: str,
    actual_direction: str,
    predicted_change_pct: float | None,
    actual_change_pct: float | None,
    context_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    patterns: list[str] = []
    reasons: list[str] = []
    has_direction_miss = predicted_direction != actual_direction
    has_magnitude_miss = False

    if has_direction_miss:
        patterns.append("direction_miss")
        reasons.append(
            f"Predicted direction was {predicted_direction} but the actual direction was {actual_direction}."
        )

    if predicted_change_pct is not None and actual_change_pct is not None:
        absolute_error = abs(actual_change_pct - predicted_change_pct)
        if absolute_error >= 2.0:
            has_magnitude_miss = True
            if abs(actual_change_pct) > abs(predicted_change_pct):
                patterns.append("move_magnitude_underestimated")
                reasons.append(
                    f"Actual move of {actual_change_pct:.2f}% was materially larger than the predicted {predicted_change_pct:.2f}%."
                )
            else:
                patterns.append("move_magnitude_overestimated")
                reasons.append(
                    f"Predicted move of {predicted_change_pct:.2f}% was materially larger than the actual {actual_change_pct:.2f}%."
                )

    if not has_direction_miss and not has_magnitude_miss:
        return {
            "patterns": [],
            "reasons": ["Prediction direction matched the realized move and no material error threshold was triggered."],
        }

    if context_summary:
        if predicted_direction == "up":
            if context_summary["company_score"] <= -0.2:
                patterns.append("negative_company_news_underweighted")
                reasons.append("Company-news sentiment was negative during the review window.")
            if context_summary["announcement_score"] <= -0.2:
                patterns.append("negative_announcements_underweighted")
                reasons.append("Exchange announcements were negative and likely overrode the bullish setup.")
            if context_summary["macro_score"] <= -0.2 or context_summary["global_cues_score"] <= -0.15:
                patterns.append("macro_headwind_underweighted")
                reasons.append("Macro or global market cues were negative during the move.")
        elif predicted_direction == "down":
            if context_summary["company_score"] >= 0.2:
                patterns.append("positive_company_news_underweighted")
                reasons.append("Company-news sentiment was positive during the review window.")
            if context_summary["announcement_score"] >= 0.2:
                patterns.append("positive_announcements_underweighted")
                reasons.append("Positive announcements likely overrode the bearish setup.")
            if context_summary["macro_score"] >= 0.2 or context_summary["global_cues_score"] >= 0.15:
                patterns.append("macro_tailwind_underweighted")
                reasons.append("Macro or global market cues were supportive during the move.")

        if context_summary["event_count"] > 0:
            patterns.append("event_risk_underweighted")
            reasons.append(
                f"{context_summary['event_count']} company-specific event(s) were detected during the review window."
            )

        if context_summary["overall_label"] not in ("neutral", None):
            if (predicted_direction == "up" and context_summary["overall_label"] == "bearish") or (
                predicted_direction == "down" and context_summary["overall_label"] == "bullish"
            ):
                patterns.append("context_disagreed_with_prediction")
                reasons.append("The aggregated news context disagreed with the prediction direction.")

    return {
        "patterns": dedupe_preserve_order(patterns),
        "reasons": dedupe_preserve_order(reasons),
    }


def summarize_context_report(report: dict[str, Any]) -> dict[str, Any]:
    processing = report.get("processing", {})
    sentiment = processing.get("sentiment", {})
    global_cues = processing.get("global_cues", {})
    events = processing.get("events", [])
    return {
        "overall_label": report.get("report", {}).get("label"),
        "company_score": to_float(sentiment.get("company_news", {}).get("score")) or 0.0,
        "sector_score": to_float(sentiment.get("sector_news", {}).get("score")) or 0.0,
        "macro_score": to_float(sentiment.get("india_macro_news", {}).get("score")) or 0.0,
        "announcement_score": to_float(sentiment.get("corporate_announcements", {}).get("score")) or 0.0,
        "global_cues_score": to_float(global_cues.get("score")) or 0.0,
        "event_count": len(events),
        "key_drivers": report.get("report", {}).get("key_drivers", []),
    }


def build_review_context_options(options: AnalysisOptions, prediction: dict[str, Any], reviewed_date: str) -> AnalysisOptions:
    prediction_timestamp = parse_datetime_or_none(prediction.get("created_at")) or datetime.now(IST)
    target_day = datetime.fromisoformat(f"{reviewed_date}T15:30:00+05:30")
    as_of = target_day if target_day > prediction_timestamp else prediction_timestamp + timedelta(hours=6)
    return options.model_copy(update={"since": prediction_timestamp, "as_of": as_of})


def derive_target_date(request: PredictionStoreRequest) -> str:
    if request.prediction_for_date is not None:
        prediction_for_date = request.prediction_for_date
        if prediction_for_date.tzinfo is None:
            prediction_for_date = prediction_for_date.replace(tzinfo=IST)
        return prediction_for_date.astimezone(IST).date().isoformat()
    if request.forecast_dates:
        return normalize_date_string(request.forecast_dates[0])
    return normalize_timestamp(request.prediction_made_at)[:10]


def derive_predicted_value(request: PredictionStoreRequest, target_date: str) -> float | None:
    if not request.predicted_values:
        return None
    if request.forecast_dates:
        for index, raw_date in enumerate(request.forecast_dates):
            if normalize_date_string(raw_date) == target_date:
                return float(request.predicted_values[index])
    return float(request.predicted_values[0])


def derive_predicted_change_pct(request: PredictionStoreRequest, predicted_value: float | None) -> float | None:
    if request.predicted_change_pct is not None:
        return round(request.predicted_change_pct, 3)
    if request.reference_price in (None, 0.0) or predicted_value is None:
        return None
    return round(((predicted_value - request.reference_price) / request.reference_price) * 100.0, 3)


def derive_actual_change_pct(request: PredictionReviewRequest, prediction: dict[str, Any]) -> float | None:
    if request.actual_change_pct is not None:
        return round(request.actual_change_pct, 3)
    reference_price = request.previous_close or to_float(prediction.get("reference_price"))
    if request.actual_close is None or reference_price in (None, 0.0):
        return None
    return round(((request.actual_close - reference_price) / reference_price) * 100.0, 3)


def normalize_review_date(request: PredictionReviewRequest, prediction: dict[str, Any]) -> str:
    if request.actual_for_date is not None:
        actual_for_date = request.actual_for_date
        if actual_for_date.tzinfo is None:
            actual_for_date = actual_for_date.replace(tzinfo=IST)
        return actual_for_date.astimezone(IST).date().isoformat()
    stored = str(prediction.get("prediction_for_date") or "").strip()
    if stored:
        return normalize_date_string(stored)
    return datetime.now(IST).date().isoformat()


def normalize_timestamp(value: datetime | None) -> str:
    if value is None:
        return datetime.now(IST).isoformat()
    if value.tzinfo is None:
        value = value.replace(tzinfo=IST)
    return value.astimezone(IST).isoformat()


def direction_from_change(change_pct: float | None, flat_band: float = 0.5) -> str:
    if change_pct is None:
        return "flat"
    if change_pct >= flat_band:
        return "up"
    if change_pct <= -flat_band:
        return "down"
    return "flat"


def parse_datetime_or_none(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST)


def normalize_date_string(value: str) -> str:
    value = value.strip()
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=IST)
        return parsed.astimezone(IST).date().isoformat()
    except ValueError:
        return value[:10]


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def ensure_feedback_dirs() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_PATH.exists():
        write_json(
            MEMORY_PATH,
            {
                "updated_at": datetime.now(IST).isoformat(),
                "global": {"review_count": 0, "correct_count": 0, "failure_patterns": {}},
                "stocks": {},
            },
        )


def load_prediction(prediction_id: str) -> dict[str, Any]:
    path = PREDICTIONS_DIR / f"{prediction_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"prediction_id not found: {prediction_id}")
    return read_json(path)


def load_memory() -> dict[str, Any]:
    ensure_feedback_dirs()
    return read_json(MEMORY_PATH)


def increment_pattern(counter: dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
