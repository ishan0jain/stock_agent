from __future__ import annotations

import base64
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from stock_agents.feedback import diagnose_prediction_miss
from stock_agents.financial_documents import (
    ingest_financial_document,
    list_financial_documents,
    query_financial_documents,
)
from stock_agents.models import (
    FinancialDocumentIngestRequest,
    FinancialDocumentListRequest,
    FinancialDocumentQueryRequest,
)


STOCK = {
    "symbol": "TEST",
    "name": "Test Industries",
}


class FinancialDocumentRagTests(unittest.TestCase):
    def test_document_is_indexed_retrieved_with_citation_and_deduplicated(self) -> None:
        document_text = (
            "Management reported revenue growth and margin expansion. "
            "Free cash flow improved and debt reduction remains a priority. "
            "The outlook improved because demand was strong. "
        ) * 12
        request = FinancialDocumentIngestRequest(
            stock=STOCK,
            filename="annual-report.txt",
            document_type="annual_report",
            title="FY 2026 Annual Report",
            content_base64=base64.b64encode(document_text.encode("utf-8")).decode("ascii"),
        )

        with TemporaryDirectory() as directory, patch(
            "stock_agents.financial_documents.RAG_ROOT",
            Path(directory),
        ):
            indexed = ingest_financial_document(request)
            duplicate = ingest_financial_document(request)
            result = query_financial_documents(
                FinancialDocumentQueryRequest(
                    stock=STOCK,
                    query="revenue margins free cash flow debt outlook demand",
                    top_k=4,
                )
            )
            documents = list_financial_documents(
                FinancialDocumentListRequest(stock=STOCK)
            )

        self.assertEqual(indexed["status"], "indexed")
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(documents["document_count"], 1)
        self.assertGreater(result["retrieved_count"], 0)
        self.assertGreater(result["signal_score"], 0)
        self.assertEqual(result["citations"][0]["title"], "FY 2026 Annual Report")
        self.assertIn("excerpt", result["citations"][0])

    def test_review_diagnoses_underweighted_document_risk(self) -> None:
        diagnosis = diagnose_prediction_miss(
            predicted_direction="up",
            actual_direction="down",
            predicted_change_pct=1.5,
            actual_change_pct=-2.0,
            context_summary=None,
            document_score=-0.7,
        )

        self.assertIn("document_risk_underweighted", diagnosis["patterns"])


if __name__ == "__main__":
    unittest.main()
