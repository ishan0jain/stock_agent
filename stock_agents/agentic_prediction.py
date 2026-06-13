from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from stock_agents.feedback import get_agent_memory_guidance, store_prediction
from stock_agents.financial_documents import retrieve_financial_context
from stock_agents.fundamentals import analyze_company_fundamentals
from stock_agents.model_service import predict_ohlcv_model
from stock_agents.models import AutonomousPredictionRequest, PredictionStoreRequest
from stock_agents.service import IST, analyze_stock
from trend_analyzer.ohlcv_lstm_cnn import FEATURE_NAMES, load_ohlcv_data_from_source


BASE_AGENT_WEIGHTS: dict[str, float] = {
    "technical": 0.42,
    "market_context": 0.28,
    "fundamentals": 0.15,
    "financial_documents": 0.15,
}

UNDERWEIGHTED_CONTEXT_PATTERNS: tuple[str, ...] = (
    "negative_company_news_underweighted",
    "positive_company_news_underweighted",
    "negative_announcements_underweighted",
    "positive_announcements_underweighted",
    "macro_headwind_underweighted",
    "macro_tailwind_underweighted",
    "event_risk_underweighted",
    "context_disagreed_with_prediction",
)

UNDERWEIGHTED_DOCUMENT_PATTERNS: tuple[str, ...] = (
    "document_risk_underweighted",
    "document_tailwind_underweighted",
)


@dataclass
class AgentSignal:
    agent: str
    role: str
    status: str
    score: float | None = None
    confidence: float = 0.0
    change_pct: float | None = None
    evidence: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "role": self.role,
            "status": self.status,
            "score": self.score,
            "confidence": self.confidence,
            "change_pct": self.change_pct,
            "evidence": self.evidence,
            "payload": self.payload,
            "error": self.error,
        }


class AutonomousPredictionOrchestrator:
    def __init__(self, request: AutonomousPredictionRequest) -> None:
        self.request = request
        self.trace: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        plan = [
            "memory_agent",
            "technical_agent",
            "market_context_agent",
        ]
        if self.request.include_fundamentals:
            plan.append("fundamental_agent")
        if self.request.include_financial_documents:
            plan.append("financial_document_agent")
        plan.extend(["decision_agent", "prediction_action_agent"])

        memory = self._run_memory_agent()
        calibration = build_memory_calibration(memory)

        signals = {
            "technical": self._execute_signal_agent(
                "technical_agent",
                "Forecast price movement from the saved CNN-LSTM model.",
                self._run_technical_agent,
            ),
            "market_context": self._execute_signal_agent(
                "market_context_agent",
                "Evaluate news, events, announcements, macro data, and global cues.",
                self._run_market_context_agent,
            ),
        }
        if self.request.include_fundamentals:
            signals["fundamentals"] = self._execute_signal_agent(
                "fundamental_agent",
                "Evaluate valuation, quality, growth, leverage, and cash flow.",
                self._run_fundamental_agent,
            )
        if self.request.include_financial_documents:
            signals["financial_documents"] = self._execute_signal_agent(
                "financial_document_agent",
                "Retrieve and evaluate stock-scoped financial-document evidence.",
                self._run_financial_document_agent,
            )

        decision_started = perf_counter()
        decision = fuse_agent_signals(
            signals=signals,
            calibration=calibration,
            expected_agents=set(signals),
        )
        self._record_trace(
            agent="decision_agent",
            role="Reconcile specialist signals into one calibrated prediction.",
            status="completed",
            started=decision_started,
            summary=(
                f"Produced a {decision['predicted_direction']} prediction "
                f"with confidence {decision['confidence']:.3f}."
            ),
        )

        action = self._run_prediction_action(decision, signals)

        return {
            "generated_at": datetime.now(IST).isoformat(),
            "mode": "autonomous_agentic_ensemble",
            "stock": self.request.stock.model_dump(),
            "autonomous_plan": plan,
            "agent_trace": self.trace,
            "memory_calibration": calibration,
            "agent_outputs": {
                name: signal.as_dict()
                for name, signal in signals.items()
            },
            "decision": decision,
            "autonomous_actions": action,
        }

    def _run_memory_agent(self) -> dict[str, Any]:
        started = perf_counter()
        try:
            memory = get_agent_memory_guidance(self.request.stock)
            self._record_trace(
                agent="memory_agent",
                role="Recall reviewed outcomes and recurring failure patterns.",
                status="completed",
                started=started,
                summary=f"Loaded {memory.get('review_count', 0)} reviewed prediction(s).",
            )
            return memory
        except (OSError, ValueError) as exc:
            self._record_trace(
                agent="memory_agent",
                role="Recall reviewed outcomes and recurring failure patterns.",
                status="degraded",
                started=started,
                summary="Memory was unavailable; neutral calibration was used.",
                error=str(exc),
            )
            return {
                "stock": self.request.stock.symbol,
                "review_count": 0,
                "correct_count": 0,
                "accuracy": None,
                "failure_patterns": {},
                "source_metrics": {},
                "lessons": [],
            }

    def _execute_signal_agent(
        self,
        agent: str,
        role: str,
        operation: Callable[[], AgentSignal],
    ) -> AgentSignal:
        started = perf_counter()
        try:
            signal = operation()
        except (Exception, SystemExit) as exc:
            signal = AgentSignal(
                agent=agent,
                role=role,
                status="failed",
                error=str(exc),
            )
        self._record_trace(
            agent=agent,
            role=role,
            status=signal.status,
            started=started,
            summary=self._signal_summary(signal),
            error=signal.error,
        )
        return signal

    def _run_technical_agent(self) -> AgentSignal:
        forecast = predict_ohlcv_model(
            model_dir=self.request.model_dir,
            json_path=self.request.json_path,
            ohlcv_data=self.request.ohlcv_data,
            output_path=self.request.output_path,
        )
        target_field = str(forecast.get("target_field", "close"))
        if target_field not in FEATURE_NAMES or target_field == "volume":
            raise ValueError(
                "The autonomous price prediction flow requires an OHLC price target, not volume."
            )

        dates, values = load_ohlcv_data_from_source(
            path=Path(self.request.json_path) if self.request.json_path else None,
            payload=self.request.ohlcv_data,
        )
        target_index = FEATURE_NAMES.index(target_field)
        reference_value = self.request.reference_price or float(values[-1, target_index])
        if reference_value <= 0.0:
            raise ValueError("The latest OHLC reference value must be greater than zero.")
        predicted_values = [float(value) for value in forecast.get("predicted_values", [])]
        if not predicted_values:
            raise ValueError("The technical model did not return predicted values.")

        first_prediction = predicted_values[0]
        change_pct = ((first_prediction - reference_value) / reference_value) * 100.0
        score = math.tanh(change_pct / 3.0)
        metrics = load_model_metrics(Path(self.request.model_dir))
        confidence = technical_confidence(metrics)

        forecast["reference_value"] = reference_value
        forecast["reference_date"] = dates[-1].date().isoformat()
        forecast["first_forecast_change_pct"] = round(change_pct, 3)
        forecast["test_metrics"] = metrics
        return AgentSignal(
            agent="technical_agent",
            role="Forecast price movement from the saved CNN-LSTM model.",
            status="completed",
            score=round(score, 4),
            confidence=confidence,
            change_pct=round(change_pct, 3),
            evidence=[
                f"{target_field} forecast changes {change_pct:.2f}% on the first forecast day.",
                model_metric_summary(metrics),
            ],
            payload=forecast,
        )

    def _run_market_context_agent(self) -> AgentSignal:
        report = analyze_stock(self.request.stock, self.request.options)
        report_summary = report.get("report", {})
        processing = report.get("processing", {})
        score = clamp(to_float(report_summary.get("overall_score")) or 0.0, -1.0, 1.0)
        confidence = clamp(to_float(report_summary.get("confidence")) or 0.0, 0.0, 1.0)
        item_count = int(processing.get("normalized_item_count", 0))
        cue_count = int(processing.get("global_cues", {}).get("cue_count", 0))
        status = "completed" if item_count or cue_count else "skipped"
        if status == "skipped":
            confidence = 0.0

        evidence = [
            str(driver)
            for driver in report_summary.get("key_drivers", [])
            if str(driver).strip()
        ][:5]
        if not evidence:
            evidence.append("No material market-context driver was available.")

        return AgentSignal(
            agent="market_context_agent",
            role="Evaluate news, events, announcements, macro data, and global cues.",
            status=status,
            score=round(score, 4),
            confidence=round(confidence, 4),
            change_pct=round(score * 2.5, 3),
            evidence=evidence,
            payload=report,
        )

    def _run_fundamental_agent(self) -> AgentSignal:
        report = analyze_company_fundamentals(self.request.stock, self.request.options)
        report_summary = report.get("report", {})
        overall_score = to_float(report_summary.get("overall_score"))
        confidence = clamp(to_float(report_summary.get("confidence")) or 0.0, 0.0, 1.0)
        if overall_score is None or confidence <= 0.0:
            return AgentSignal(
                agent="fundamental_agent",
                role="Evaluate valuation, quality, growth, leverage, and cash flow.",
                status="skipped",
                evidence=[str(report_summary.get("summary") or "Fundamental data was unavailable.")],
                payload=report,
            )

        normalized_score = clamp((overall_score - 50.0) / 50.0, -1.0, 1.0)
        evidence = [
            *[str(item) for item in report_summary.get("insights", [])[:3]],
            *[str(item) for item in report_summary.get("risks", [])[:2]],
        ]
        return AgentSignal(
            agent="fundamental_agent",
            role="Evaluate valuation, quality, growth, leverage, and cash flow.",
            status="completed",
            score=round(normalized_score, 4),
            confidence=round(confidence, 4),
            change_pct=round(normalized_score * 1.5, 3),
            evidence=evidence or [str(report_summary.get("summary", ""))],
            payload=report,
        )

    def _run_financial_document_agent(self) -> AgentSignal:
        report = retrieve_financial_context(
            stock=self.request.stock,
            query=self.request.rag_query,
            top_k=self.request.rag_top_k,
        )
        score = to_float(report.get("signal_score"))
        confidence = clamp(to_float(report.get("confidence")) or 0.0, 0.0, 1.0)
        if score is None or confidence <= 0.0:
            return AgentSignal(
                agent="financial_document_agent",
                role="Retrieve and evaluate stock-scoped financial-document evidence.",
                status="skipped",
                evidence=[str(report.get("summary") or "No document evidence was available.")],
                payload=report,
            )

        citations = report.get("citations", [])
        evidence = [
            f"{citation.get('title')}: {citation.get('excerpt')}"
            for citation in citations[:4]
        ]
        return AgentSignal(
            agent="financial_document_agent",
            role="Retrieve and evaluate stock-scoped financial-document evidence.",
            status="completed",
            score=round(score, 4),
            confidence=round(confidence, 4),
            change_pct=round(score * 1.8, 3),
            evidence=evidence or [str(report.get("summary", ""))],
            payload=report,
        )

    def _run_prediction_action(
        self,
        decision: dict[str, Any],
        signals: dict[str, AgentSignal],
    ) -> dict[str, Any]:
        started = perf_counter()
        if not self.request.auto_store_prediction:
            action = {
                "store_prediction": {
                    "status": "skipped",
                    "reason": "auto_store_prediction is false",
                }
            }
            self._record_trace(
                agent="prediction_action_agent",
                role="Persist the final prediction for later review and learning.",
                status="skipped",
                started=started,
                summary="Prediction storage was disabled by the request.",
            )
            return action

        technical = signals.get("technical")
        technical_payload = technical.payload if technical else {}
        financial_documents = signals.get("financial_documents")
        document_payload = financial_documents.payload if financial_documents else {}
        try:
            stored = store_prediction(
                PredictionStoreRequest(
                    stock=self.request.stock,
                    reference_price=technical_payload.get("reference_value") or self.request.reference_price,
                    target_field=technical_payload.get("target_field", "close"),
                    forecast_dates=technical_payload.get("forecast_dates", []),
                    predicted_values=technical_payload.get("predicted_values", []),
                    predicted_direction=decision["predicted_direction"],
                    predicted_change_pct=decision["predicted_change_pct"],
                    prediction_source="autonomous_agentic_ensemble",
                    confidence=decision["confidence"],
                    model_dir=self.request.model_dir,
                    notes=(
                        "Autonomously combined technical, market-context, fundamental, "
                        "financial-document RAG, and feedback-memory signals."
                    ),
                    evidence_context={
                        "financial_documents": {
                            "signal_score": document_payload.get("signal_score"),
                            "query": document_payload.get("query"),
                            "citations": document_payload.get("citations", []),
                        },
                        "contributions": decision.get("contributions", {}),
                    },
                )
            )
            prediction = stored.get("prediction", {})
            action = {
                "store_prediction": {
                    "status": "completed",
                    "prediction_id": prediction.get("prediction_id"),
                    "review_status": prediction.get("review_status"),
                }
            }
            self._record_trace(
                agent="prediction_action_agent",
                role="Persist the final prediction for later review and learning.",
                status="completed",
                started=started,
                summary=f"Stored prediction {prediction.get('prediction_id')}.",
            )
            return action
        except (OSError, ValueError) as exc:
            self._record_trace(
                agent="prediction_action_agent",
                role="Persist the final prediction for later review and learning.",
                status="failed",
                started=started,
                summary="The prediction was produced but could not be stored.",
                error=str(exc),
            )
            return {
                "store_prediction": {
                    "status": "failed",
                    "error": str(exc),
                }
            }

    def _record_trace(
        self,
        *,
        agent: str,
        role: str,
        status: str,
        started: float,
        summary: str,
        error: str | None = None,
    ) -> None:
        self.trace.append(
            {
                "agent": agent,
                "role": role,
                "status": status,
                "duration_ms": round((perf_counter() - started) * 1000.0, 2),
                "summary": summary,
                "error": error,
            }
        )

    @staticmethod
    def _signal_summary(signal: AgentSignal) -> str:
        if signal.score is None:
            return signal.error or "No usable signal was produced."
        return (
            f"Produced score {signal.score:.3f} "
            f"with confidence {signal.confidence:.3f}."
        )


def run_autonomous_prediction(request: AutonomousPredictionRequest) -> dict[str, Any]:
    return AutonomousPredictionOrchestrator(request).run()


def build_memory_calibration(memory: dict[str, Any]) -> dict[str, Any]:
    review_count = int(memory.get("review_count", 0))
    correct_count = int(memory.get("correct_count", 0))
    source_metrics = memory.get("source_metrics", {})
    technical_metric = source_metrics.get("ohlcv_model", {})
    ensemble_metric = source_metrics.get("autonomous_agentic_ensemble", {})

    accuracy_source = "stock_overall"
    accuracy_reviews = review_count
    accuracy_correct = correct_count
    if int(technical_metric.get("review_count", 0)) > 0:
        accuracy_source = "ohlcv_model"
        accuracy_reviews = int(technical_metric.get("review_count", 0))
        accuracy_correct = int(technical_metric.get("correct_count", 0))

    smoothed_accuracy = bayesian_accuracy(accuracy_correct, accuracy_reviews)
    sample_strength = min(1.0, accuracy_reviews / 12.0)
    technical_multiplier = 1.0 + ((smoothed_accuracy - 0.5) * 1.2 * sample_strength)
    context_multiplier = 1.0
    fundamental_multiplier = 1.0
    document_multiplier = 1.0
    confidence_multiplier = 1.0
    reasons: list[str] = []

    patterns = memory.get("failure_patterns", {})
    pattern_denominator = max(review_count, 1)
    direction_miss_rate = int(patterns.get("direction_miss", 0)) / pattern_denominator
    context_miss_count = sum(int(patterns.get(key, 0)) for key in UNDERWEIGHTED_CONTEXT_PATTERNS)
    context_miss_rate = min(1.0, context_miss_count / pattern_denominator)
    document_miss_count = sum(
        int(patterns.get(key, 0)) for key in UNDERWEIGHTED_DOCUMENT_PATTERNS
    )
    document_miss_rate = min(1.0, document_miss_count / pattern_denominator)
    magnitude_miss_count = int(patterns.get("move_magnitude_underestimated", 0)) + int(
        patterns.get("move_magnitude_overestimated", 0)
    )
    magnitude_miss_rate = min(1.0, magnitude_miss_count / pattern_denominator)

    if review_count:
        technical_multiplier *= 1.0 - min(0.40, direction_miss_rate * 0.50)
        context_multiplier *= 1.0 + min(0.55, context_miss_rate * 0.60)
        document_multiplier *= 1.0 + min(0.60, document_miss_rate * 0.70)
        confidence_multiplier *= 1.0 - min(0.30, magnitude_miss_rate * 0.35)
        reasons.append(
            f"Historical accuracy source {accuracy_source} was smoothed to {smoothed_accuracy:.3f}."
        )
        if direction_miss_rate:
            reasons.append(
                f"Direction misses reduced the technical-agent weight ({direction_miss_rate:.1%} rate)."
            )
        if context_miss_rate:
            reasons.append(
                f"Underweighted context patterns increased the market-context weight ({context_miss_rate:.1%} rate)."
            )
        if document_miss_rate:
            reasons.append(
                f"Underweighted document evidence increased the financial-document weight ({document_miss_rate:.1%} rate)."
            )
        if magnitude_miss_rate:
            reasons.append(
                f"Magnitude misses reduced ensemble confidence ({magnitude_miss_rate:.1%} rate)."
            )
    else:
        reasons.append("No reviewed history exists; neutral prior calibration was used.")

    ensemble_reviews = int(ensemble_metric.get("review_count", 0))
    if ensemble_reviews:
        ensemble_accuracy = bayesian_accuracy(
            int(ensemble_metric.get("correct_count", 0)),
            ensemble_reviews,
        )
        ensemble_strength = min(1.0, ensemble_reviews / 12.0)
        confidence_multiplier *= 1.0 + ((ensemble_accuracy - 0.5) * 0.8 * ensemble_strength)
        reasons.append(
            f"Autonomous-ensemble history adjusted confidence using {ensemble_reviews} review(s)."
        )

    return {
        "review_count": review_count,
        "accuracy_source": accuracy_source,
        "smoothed_accuracy": round(smoothed_accuracy, 4),
        "weight_multipliers": {
            "technical": round(clamp(technical_multiplier, 0.45, 1.60), 4),
            "market_context": round(clamp(context_multiplier, 0.60, 1.70), 4),
            "fundamentals": round(clamp(fundamental_multiplier, 0.75, 1.25), 4),
            "financial_documents": round(clamp(document_multiplier, 0.70, 1.75), 4),
        },
        "confidence_multiplier": round(clamp(confidence_multiplier, 0.55, 1.20), 4),
        "failure_patterns": patterns,
        "reasons": reasons,
    }


def fuse_agent_signals(
    *,
    signals: dict[str, AgentSignal],
    calibration: dict[str, Any],
    expected_agents: set[str],
) -> dict[str, Any]:
    available = {
        name: signal
        for name, signal in signals.items()
        if signal.status in {"completed", "degraded"}
        and signal.score is not None
        and signal.confidence > 0.0
    }
    if not available:
        failures = "; ".join(
            f"{name}: {signal.error or signal.status}"
            for name, signal in signals.items()
        )
        raise RuntimeError(f"No specialist agent produced a usable prediction signal. {failures}")

    multipliers = calibration.get("weight_multipliers", {})
    unnormalized: dict[str, float] = {}
    for name, signal in available.items():
        base_weight = BASE_AGENT_WEIGHTS[name]
        memory_multiplier = to_float(multipliers.get(name)) or 1.0
        confidence_factor = 0.35 + (0.65 * signal.confidence)
        unnormalized[name] = base_weight * memory_multiplier * confidence_factor

    total_weight = sum(unnormalized.values())
    weights = {
        name: value / total_weight
        for name, value in unnormalized.items()
    }
    combined_score = sum(weights[name] * float(available[name].score) for name in available)
    expected_change_pct = sum(
        weights[name] * float(available[name].change_pct or 0.0)
        for name in available
    )
    change_score = math.tanh(expected_change_pct / 2.5)
    decision_score = (combined_score * 0.70) + (change_score * 0.30)

    if decision_score >= 0.12:
        predicted_direction = "up"
    elif decision_score <= -0.12:
        predicted_direction = "down"
    else:
        predicted_direction = "flat"

    agreement = signal_agreement(available, weights, combined_score)
    weighted_confidence = sum(
        weights[name] * available[name].confidence
        for name in available
    )
    expected_base_weight = sum(BASE_AGENT_WEIGHTS[name] for name in expected_agents)
    available_base_weight = sum(BASE_AGENT_WEIGHTS[name] for name in available)
    coverage = clamp(available_base_weight / expected_base_weight, 0.0, 1.0)
    memory_confidence_multiplier = to_float(calibration.get("confidence_multiplier")) or 1.0
    confidence = (
        weighted_confidence
        * (0.55 + (0.45 * agreement))
        * (0.65 + (0.35 * coverage))
        * memory_confidence_multiplier
    )
    confidence = clamp(confidence, 0.05, 0.95)

    contributions = {
        name: {
            "score": signal.score,
            "confidence": signal.confidence,
            "base_weight": BASE_AGENT_WEIGHTS[name],
            "memory_multiplier": multipliers.get(name, 1.0),
            "final_weight": round(weights[name], 4),
            "weighted_score": round(weights[name] * float(signal.score), 4),
        }
        for name, signal in available.items()
    }
    unavailable = {
        name: {
            "status": signal.status,
            "error": signal.error,
        }
        for name, signal in signals.items()
        if name not in available
    }

    rationale = [
        (
            f"{name} contributed score {signal.score:.3f} "
            f"at {weights[name]:.1%} final weight."
        )
        for name, signal in available.items()
    ]
    rationale.extend(calibration.get("reasons", []))
    if agreement < 0.50:
        rationale.append("Specialist agents disagreed, so final confidence was reduced.")
    if coverage < 1.0:
        rationale.append("One or more planned specialist signals were unavailable.")

    return {
        "predicted_direction": predicted_direction,
        "predicted_change_pct": round(expected_change_pct, 3),
        "decision_score": round(decision_score, 4),
        "confidence": round(confidence, 4),
        "agreement": round(agreement, 4),
        "signal_coverage": round(coverage, 4),
        "contributions": contributions,
        "unavailable_agents": unavailable,
        "rationale": rationale,
    }


def signal_agreement(
    signals: dict[str, AgentSignal],
    weights: dict[str, float],
    combined_score: float,
) -> float:
    if len(signals) == 1:
        return 0.65

    directions = {
        name: direction_bucket(float(signal.score))
        for name, signal in signals.items()
    }
    non_neutral_directions = [value for value in directions.values() if value]
    if not non_neutral_directions:
        directional_agreement = 1.0
    else:
        directional_agreement = abs(
            sum(weights[name] * directions[name] for name in signals)
        )
    dispersion = sum(
        weights[name] * abs(float(signal.score) - combined_score)
        for name, signal in signals.items()
    )
    magnitude_agreement = 1.0 - clamp(dispersion, 0.0, 1.0)
    return clamp((directional_agreement * 0.65) + (magnitude_agreement * 0.35), 0.0, 1.0)


def direction_bucket(score: float, neutral_band: float = 0.08) -> int:
    if score >= neutral_band:
        return 1
    if score <= -neutral_band:
        return -1
    return 0


def bayesian_accuracy(correct_count: int, review_count: int) -> float:
    return (correct_count + 2.0) / (review_count + 4.0)


def load_model_metrics(model_dir: Path) -> dict[str, float]:
    path = model_dir / "test_metrics.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    return {
        key: float(value)
        for key in ("mae", "rmse", "mape")
        if (value := to_float(payload.get(key))) is not None
    }


def technical_confidence(metrics: dict[str, float]) -> float:
    mape = metrics.get("mape")
    if mape is None:
        return 0.55
    return round(clamp(1.0 - (mape / 50.0), 0.15, 0.90), 4)


def model_metric_summary(metrics: dict[str, float]) -> str:
    if not metrics:
        return "No saved test metrics were available; default technical confidence was used."
    return (
        f"Saved test metrics: MAE={metrics.get('mae', 0.0):.3f}, "
        f"RMSE={metrics.get('rmse', 0.0):.3f}, "
        f"MAPE={metrics.get('mape', 0.0):.2f}%."
    )


def to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
