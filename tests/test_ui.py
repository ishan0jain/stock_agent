from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from stock_agents.app import UI_DIR, app
from stock_agents.model_service import get_ohlcv_model_results


class UiRouteTests(unittest.TestCase):
    def test_ui_assets_exist_and_routes_are_registered(self) -> None:
        self.assertTrue((UI_DIR / "index.html").is_file())
        self.assertTrue((UI_DIR / "styles.css").is_file())
        self.assertTrue((UI_DIR / "app.js").is_file())

        paths = {route.path for route in app.routes}
        self.assertIn("/", paths)
        self.assertIn("/ui", paths)
        self.assertIn("/ui/assets", paths)
        self.assertIn("/api/v1/model/results", paths)
        self.assertIn("/api/v1/rag/documents/ingest", paths)
        self.assertIn("/api/v1/rag/documents/query", paths)
        self.assertIn("/api/v1/rag/documents/list", paths)

    def test_javascript_references_existing_html_elements(self) -> None:
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        javascript = (UI_DIR / "app.js").read_text(encoding="utf-8")
        html_ids = set(re.findall(r'id="([^"]+)"', html))
        referenced_ids = set(re.findall(r'byId\("([^"]+)"\)', javascript))
        self.assertEqual(referenced_ids - html_ids, set())


class ModelResultsTests(unittest.TestCase):
    def test_saved_artifacts_are_returned_for_the_ui(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            artifacts = {
                "metadata.json": {
                    "stock_name": "TEST",
                    "window_size": 30,
                    "horizon": 5,
                    "target_field": "close",
                    "feature_names": ["open", "high", "low", "close", "volume"],
                    "feature_mean": [1, 1, 1, 1, 1],
                    "feature_std": [1, 1, 1, 1, 1],
                    "target_mean": 1,
                    "target_std": 1,
                    "train_ratio": 0.8,
                    "created_at": "2026-06-13T00:00:00Z",
                    "train_samples": 80,
                    "test_samples": 20,
                    "last_training_date": "2026-06-12",
                    "model_path": "model.keras",
                },
                "test_metrics.json": {"mae": 1.0, "rmse": 1.2, "mape": 2.5},
                "training_history.json": {"loss": [2.0, 1.0], "val_loss": [2.2, 1.1]},
                "test_predictions.json": [
                    {
                        "context_end_date": "2026-06-05",
                        "forecast_dates": ["2026-06-06"],
                        "actual": [100.0],
                        "predicted": [101.0],
                    }
                ],
                "forecast.json": {
                    "forecast_dates": ["2026-06-15"],
                    "predicted_values": [102.0],
                },
            }
            for name, payload in artifacts.items():
                (model_dir / name).write_text(json.dumps(payload), encoding="utf-8")

            result = get_ohlcv_model_results(str(model_dir))

        self.assertEqual(result["metadata"]["stock_name"], "TEST")
        self.assertEqual(result["metrics"]["mape"], 2.5)
        self.assertEqual(result["test_predictions"][0]["predicted"], [101.0])
        self.assertEqual(result["forecast"]["predicted_values"], [102.0])


if __name__ == "__main__":
    unittest.main()
