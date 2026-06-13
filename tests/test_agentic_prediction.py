from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np

from stock_agents.agentic_prediction import (
    AgentSignal,
    build_memory_calibration,
    fuse_agent_signals,
    run_autonomous_prediction,
)
from stock_agents.models import AutonomousPredictionRequest


class MemoryCalibrationTests(unittest.TestCase):
    def test_failure_history_reduces_technical_weight_and_boosts_context(self) -> None:
        calibration = build_memory_calibration(
            {
                "review_count": 10,
                "correct_count": 2,
                "failure_patterns": {
                    "direction_miss": 5,
                    "negative_company_news_underweighted": 3,
                    "context_disagreed_with_prediction": 2,
                    "document_risk_underweighted": 2,
                    "move_magnitude_underestimated": 3,
                },
                "source_metrics": {
                    "ohlcv_model": {
                        "review_count": 8,
                        "correct_count": 2,
                    }
                },
            }
        )

        multipliers = calibration["weight_multipliers"]
        self.assertLess(multipliers["technical"], 1.0)
        self.assertGreater(multipliers["market_context"], 1.0)
        self.assertGreater(multipliers["financial_documents"], 1.0)
        self.assertLess(calibration["confidence_multiplier"], 1.0)

    def test_recalibrated_context_can_override_an_unreliable_model(self) -> None:
        signals = {
            "technical": AgentSignal(
                agent="technical_agent",
                role="technical",
                status="completed",
                score=0.7,
                confidence=0.8,
                change_pct=2.0,
            ),
            "market_context": AgentSignal(
                agent="market_context_agent",
                role="context",
                status="completed",
                score=-0.8,
                confidence=0.9,
                change_pct=-2.0,
            ),
            "fundamentals": AgentSignal(
                agent="fundamental_agent",
                role="fundamentals",
                status="completed",
                score=0.2,
                confidence=0.7,
                change_pct=0.3,
            ),
        }
        decision = fuse_agent_signals(
            signals=signals,
            calibration={
                "weight_multipliers": {
                    "technical": 0.5,
                    "market_context": 1.6,
                    "fundamentals": 1.0,
                },
                "confidence_multiplier": 0.9,
                "reasons": [],
            },
            expected_agents=set(signals),
        )

        self.assertEqual(decision["predicted_direction"], "down")
        self.assertGreater(
            decision["contributions"]["market_context"]["final_weight"],
            decision["contributions"]["technical"]["final_weight"],
        )


class AutonomousOrchestratorTests(unittest.TestCase):
    @patch("stock_agents.agentic_prediction.store_prediction")
    @patch("stock_agents.agentic_prediction.retrieve_financial_context")
    @patch("stock_agents.agentic_prediction.analyze_company_fundamentals")
    @patch("stock_agents.agentic_prediction.analyze_stock")
    @patch("stock_agents.agentic_prediction.load_ohlcv_data_from_source")
    @patch("stock_agents.agentic_prediction.predict_ohlcv_model")
    @patch("stock_agents.agentic_prediction.get_agent_memory_guidance")
    def test_orchestrator_runs_agents_and_stores_ensemble_prediction(
        self,
        memory_mock,
        predict_mock,
        load_data_mock,
        context_mock,
        fundamentals_mock,
        documents_mock,
        store_mock,
    ) -> None:
        memory_mock.return_value = {
            "review_count": 0,
            "correct_count": 0,
            "failure_patterns": {},
            "source_metrics": {},
            "lessons": [],
        }
        predict_mock.return_value = {
            "stock_name": "TEST",
            "target_field": "close",
            "forecast_dates": ["2026-06-15"],
            "predicted_values": [102.0],
            "model_dir": "unused",
        }
        load_data_mock.return_value = (
            [datetime.fromisoformat("2026-06-12T00:00:00+05:30")],
            np.asarray([[98.0, 101.0, 97.0, 100.0, 1000.0]], dtype=np.float32),
        )
        context_mock.return_value = {
            "processing": {
                "normalized_item_count": 3,
                "global_cues": {"cue_count": 1},
            },
            "report": {
                "overall_score": 0.4,
                "confidence": 0.8,
                "key_drivers": ["Positive company update"],
            },
        }
        fundamentals_mock.return_value = {
            "report": {
                "overall_score": 70.0,
                "confidence": 0.7,
                "summary": "Constructive fundamentals.",
                "insights": ["Positive free cash flow."],
                "risks": [],
            }
        }
        documents_mock.return_value = {
            "status": "completed",
            "query": "outlook and risks",
            "document_count": 1,
            "retrieved_count": 1,
            "signal_score": 0.3,
            "confidence": 0.65,
            "summary": "Retrieved supportive document evidence.",
            "citations": [
                {
                    "document_id": "document-1",
                    "chunk_id": "document-1:0",
                    "title": "Annual Report",
                    "document_type": "annual_report",
                    "excerpt": "Revenue growth and margin expansion remained strong.",
                }
            ],
        }
        store_mock.return_value = {
            "prediction": {
                "prediction_id": "prediction-123",
                "review_status": "pending",
            }
        }

        request = AutonomousPredictionRequest(
            stock={
                "symbol": "TEST",
                "name": "Test Company",
                "sector": "Technology",
            },
            model_dir="unused",
            ohlcv_data=[
                ["2026-06-12T00:00:00+0530", 98, 101, 97, 100, 1000, 0]
            ],
        )
        result = run_autonomous_prediction(request)

        self.assertEqual(result["mode"], "autonomous_agentic_ensemble")
        self.assertEqual(result["decision"]["predicted_direction"], "up")
        self.assertEqual(
            result["autonomous_actions"]["store_prediction"]["prediction_id"],
            "prediction-123",
        )
        self.assertEqual(
            [row["agent"] for row in result["agent_trace"]],
            [
                "memory_agent",
                "technical_agent",
                "market_context_agent",
                "fundamental_agent",
                "financial_document_agent",
                "decision_agent",
                "prediction_action_agent",
            ],
        )
        store_mock.assert_called_once()
        stored_request = store_mock.call_args.args[0]
        self.assertEqual(
            stored_request.evidence_context["financial_documents"]["signal_score"],
            0.3,
        )


if __name__ == "__main__":
    unittest.main()
