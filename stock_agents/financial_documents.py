from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from stock_agents.models import (
    FinancialDocumentIngestRequest,
    FinancialDocumentListRequest,
    FinancialDocumentQueryRequest,
    StockInput,
)
from stock_agents.service import IST


RAG_ROOT = Path(__file__).resolve().parent.parent / "rag_data"
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 220

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
WHITESPACE_PATTERN = re.compile(r"\s+")
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

POSITIVE_DOCUMENT_TERMS: dict[str, float] = {
    "revenue growth": 1.0,
    "profit growth": 1.1,
    "margin expansion": 1.1,
    "strong demand": 0.8,
    "order book": 0.6,
    "debt reduction": 0.9,
    "free cash flow": 0.7,
    "capacity expansion": 0.5,
    "market share gain": 0.8,
    "guidance raised": 1.0,
    "outlook improved": 0.9,
    "record revenue": 0.9,
    "operating leverage": 0.6,
}

NEGATIVE_DOCUMENT_TERMS: dict[str, float] = {
    "revenue decline": -1.0,
    "profit decline": -1.1,
    "margin pressure": -1.0,
    "weak demand": -0.8,
    "debt increased": -0.8,
    "cash flow negative": -1.0,
    "guidance reduced": -1.0,
    "outlook weakened": -0.9,
    "impairment": -0.8,
    "default": -1.4,
    "material weakness": -1.0,
    "regulatory risk": -0.7,
    "litigation": -0.6,
    "going concern": -1.5,
    "customer concentration": -0.5,
}


def ingest_financial_document(request: FinancialDocumentIngestRequest) -> dict[str, Any]:
    raw_bytes, text = extract_document_content(request)
    cleaned_text = normalize_document_text(text)
    if len(cleaned_text) < 40:
        raise ValueError("document did not contain enough readable text to index")

    symbol = normalize_symbol(request.stock.symbol)
    content_hash = hashlib.sha256(raw_bytes or cleaned_text.encode("utf-8")).hexdigest()
    duplicate = find_document_by_hash(symbol, content_hash)
    if duplicate:
        return {
            "status": "duplicate",
            "document": duplicate,
        }

    document_id = str(uuid.uuid4())
    document_dir = RAG_ROOT / symbol / document_id
    chunks = chunk_document(cleaned_text)
    ingested_at = datetime.now(IST).isoformat()
    metadata = {
        "document_id": document_id,
        "symbol": symbol,
        "stock_name": request.stock.name,
        "filename": safe_filename(request.filename),
        "title": request.title or Path(request.filename).stem,
        "document_type": request.document_type,
        "published_at": normalize_optional_datetime(request.published_at),
        "source_url": request.source_url,
        "ingested_at": ingested_at,
        "content_hash": content_hash,
        "character_count": len(cleaned_text),
        "chunk_count": len(chunks),
    }
    chunk_rows = [
        {
            "chunk_id": f"{document_id}:{index}",
            "document_id": document_id,
            "chunk_index": index,
            "text": chunk,
            "token_count": len(tokenize(chunk)),
        }
        for index, chunk in enumerate(chunks)
    ]
    write_json(document_dir / "metadata.json", metadata)
    write_json(document_dir / "chunks.json", chunk_rows)
    return {
        "status": "indexed",
        "document": metadata,
    }


def query_financial_documents(request: FinancialDocumentQueryRequest) -> dict[str, Any]:
    return retrieve_financial_context(
        stock=request.stock,
        query=request.query,
        top_k=request.top_k,
    )


def list_financial_documents(request: FinancialDocumentListRequest) -> dict[str, Any]:
    symbol = normalize_symbol(request.stock.symbol) if request.stock else None
    documents = load_document_metadata(symbol)
    documents.sort(
        key=lambda row: str(row.get("published_at") or row.get("ingested_at") or ""),
        reverse=True,
    )
    return {
        "symbol": symbol,
        "document_count": len(documents),
        "documents": documents,
    }


def retrieve_financial_context(
    *,
    stock: StockInput,
    query: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    symbol = normalize_symbol(stock.symbol)
    effective_query = query or default_financial_query(stock)
    chunks = load_stock_chunks(symbol)
    if not chunks:
        return {
            "status": "empty",
            "symbol": symbol,
            "query": effective_query,
            "document_count": 0,
            "retrieved_count": 0,
            "signal_score": None,
            "confidence": 0.0,
            "summary": "No indexed financial documents were found for this stock.",
            "citations": [],
            "retrieved_chunks": [],
        }

    query_tokens = tokenize(effective_query)
    document_frequency = build_document_frequency(chunks)
    ranked: list[dict[str, Any]] = []
    for chunk in chunks:
        relevance = lexical_relevance(
            query=effective_query,
            query_tokens=query_tokens,
            text=str(chunk.get("text", "")),
            document_frequency=document_frequency,
            corpus_size=len(chunks),
        )
        if relevance <= 0.0:
            continue
        sentiment_score, drivers = financial_document_sentiment(str(chunk.get("text", "")))
        ranked.append(
            {
                **chunk,
                "relevance_score": round(relevance, 4),
                "sentiment_score": sentiment_score,
                "sentiment_drivers": drivers,
            }
        )

    ranked.sort(
        key=lambda row: (
            float(row["relevance_score"]),
            abs(float(row["sentiment_score"])),
        ),
        reverse=True,
    )
    selected = ranked[:top_k]
    if not selected:
        return {
            "status": "no_match",
            "symbol": symbol,
            "query": effective_query,
            "document_count": len({row["document_id"] for row in chunks}),
            "retrieved_count": 0,
            "signal_score": None,
            "confidence": 0.0,
            "summary": "Documents are indexed, but no chunk matched the retrieval query.",
            "citations": [],
            "retrieved_chunks": [],
        }

    max_relevance = max(float(row["relevance_score"]) for row in selected) or 1.0
    weighted_total = 0.0
    relevance_total = 0.0
    for row in selected:
        normalized_relevance = float(row["relevance_score"]) / max_relevance
        weighted_total += float(row["sentiment_score"]) * normalized_relevance
        relevance_total += normalized_relevance
    signal_score = weighted_total / relevance_total if relevance_total else 0.0
    average_relevance = sum(
        float(row["relevance_score"]) / max_relevance for row in selected
    ) / len(selected)
    confidence = min(0.88, 0.25 + (len(selected) * 0.07) + (average_relevance * 0.28))

    citations = [
        {
            "document_id": row["document_id"],
            "chunk_id": row["chunk_id"],
            "title": row["title"],
            "document_type": row["document_type"],
            "published_at": row.get("published_at"),
            "relevance_score": row["relevance_score"],
            "excerpt": excerpt(str(row["text"])),
        }
        for row in selected
    ]
    return {
        "status": "completed",
        "symbol": symbol,
        "query": effective_query,
        "document_count": len({row["document_id"] for row in chunks}),
        "retrieved_count": len(selected),
        "signal_score": round(max(-1.0, min(1.0, signal_score)), 4),
        "confidence": round(confidence, 4),
        "summary": build_retrieval_summary(selected, signal_score),
        "citations": citations,
        "retrieved_chunks": [
            {
                "document_id": row["document_id"],
                "chunk_id": row["chunk_id"],
                "title": row["title"],
                "document_type": row["document_type"],
                "published_at": row.get("published_at"),
                "relevance_score": row["relevance_score"],
                "sentiment_score": row["sentiment_score"],
                "sentiment_drivers": row["sentiment_drivers"],
                "text": row["text"],
            }
            for row in selected
        ],
    }


def extract_document_content(
    request: FinancialDocumentIngestRequest,
) -> tuple[bytes | None, str]:
    if request.text_content is not None:
        encoded = request.text_content.encode("utf-8")
        if len(encoded) > MAX_DOCUMENT_BYTES:
            raise ValueError("document exceeds the 25 MB limit")
        return None, request.text_content

    try:
        raw_bytes = base64.b64decode(request.content_base64 or "", validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("content_base64 is not valid base64") from exc
    if not raw_bytes:
        raise ValueError("uploaded document is empty")
    if len(raw_bytes) > MAX_DOCUMENT_BYTES:
        raise ValueError("document exceeds the 25 MB limit")

    suffix = Path(request.filename).suffix.lower()
    if suffix == ".pdf":
        return raw_bytes, extract_pdf_text(raw_bytes)
    if suffix in {".txt", ".md", ".csv", ".json", ".html", ".htm"}:
        return raw_bytes, decode_text_document(raw_bytes, suffix)
    raise ValueError("supported document types are PDF, TXT, MD, CSV, JSON, and HTML")


def extract_pdf_text(raw_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError(
            "PDF extraction requires `pypdf`; reinstall the project dependencies"
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ValueError(f"could not extract text from PDF: {exc}") from exc
    return "\n\n".join(pages)


def decode_text_document(raw_bytes: bytes, suffix: str) -> str:
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
    if suffix == ".json":
        try:
            payload = json.loads(text)
            text = json.dumps(payload, indent=2, ensure_ascii=True)
        except json.JSONDecodeError:
            pass
    if suffix in {".html", ".htm"}:
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def chunk_document(text: str) -> list[str]:
    paragraphs = [
        clean_document_text(paragraph)
        for paragraph in re.split(r"\n\s*\n", text)
        if clean_document_text(paragraph)
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= CHUNK_SIZE:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + CHUNK_SIZE)
            chunks.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)
        current = ""
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def load_stock_chunks(symbol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    symbol_dir = RAG_ROOT / symbol
    if not symbol_dir.exists():
        return rows
    for document_dir in symbol_dir.iterdir():
        if not document_dir.is_dir():
            continue
        metadata = read_json(document_dir / "metadata.json", default={})
        chunks = read_json(document_dir / "chunks.json", default=[])
        if not isinstance(metadata, dict) or not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            rows.append(
                {
                    **chunk,
                    "title": metadata.get("title"),
                    "document_type": metadata.get("document_type"),
                    "published_at": metadata.get("published_at"),
                    "filename": metadata.get("filename"),
                }
            )
    return rows


def load_document_metadata(symbol: str | None = None) -> list[dict[str, Any]]:
    roots: Iterable[Path]
    if symbol:
        roots = [RAG_ROOT / symbol]
    elif RAG_ROOT.exists():
        roots = [path for path in RAG_ROOT.iterdir() if path.is_dir()]
    else:
        roots = []

    documents: list[dict[str, Any]] = []
    for symbol_dir in roots:
        if not symbol_dir.exists():
            continue
        for document_dir in symbol_dir.iterdir():
            metadata = read_json(document_dir / "metadata.json", default=None)
            if isinstance(metadata, dict):
                documents.append(metadata)
    return documents


def find_document_by_hash(symbol: str, content_hash: str) -> dict[str, Any] | None:
    for metadata in load_document_metadata(symbol):
        if metadata.get("content_hash") == content_hash:
            return metadata
    return None


def build_document_frequency(chunks: list[dict[str, Any]]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for chunk in chunks:
        frequency.update(set(tokenize(str(chunk.get("text", "")))))
    return frequency


def lexical_relevance(
    *,
    query: str,
    query_tokens: list[str],
    text: str,
    document_frequency: Counter[str],
    corpus_size: int,
) -> float:
    if not query_tokens:
        return 0.0
    tokens = tokenize(text)
    counts = Counter(tokens)
    score = 0.0
    for token in set(query_tokens):
        term_count = counts.get(token, 0)
        if not term_count:
            continue
        inverse_frequency = math.log((corpus_size + 1) / (document_frequency[token] + 1)) + 1.0
        score += (1.0 + math.log(term_count)) * inverse_frequency
    query_phrase = clean_document_text(query).lower()
    if query_phrase and query_phrase in text.lower():
        score += 3.0
    length_normalizer = 1.0 + math.log(max(len(tokens), 20) / 20.0)
    return score / length_normalizer


def financial_document_sentiment(text: str) -> tuple[float, list[str]]:
    lowered = text.lower()
    raw_score = 0.0
    drivers: list[str] = []
    for term, weight in {**POSITIVE_DOCUMENT_TERMS, **NEGATIVE_DOCUMENT_TERMS}.items():
        occurrences = lowered.count(term)
        if not occurrences:
            continue
        raw_score += weight * min(occurrences, 3)
        drivers.append(term)
    return round(math.tanh(raw_score / 3.0), 4), drivers[:8]


def build_retrieval_summary(rows: list[dict[str, Any]], signal_score: float) -> str:
    direction = "neutral"
    if signal_score >= 0.15:
        direction = "supportive"
    elif signal_score <= -0.15:
        direction = "cautious"
    titles = list(dict.fromkeys(str(row.get("title") or "Untitled") for row in rows))
    return (
        f"Retrieved {len(rows)} relevant chunk(s) from {len(titles)} document(s). "
        f"The document evidence is {direction} for the stock outlook."
    )


def default_financial_query(stock: StockInput) -> str:
    return (
        f"{stock.name} {stock.symbol} revenue profit margins cash flow debt guidance "
        "outlook demand capex orders risks management commentary"
    )


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(value)]


def clean_document_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def normalize_document_text(value: str) -> str:
    blocks = re.split(r"\n\s*\n", value.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n\n".join(
        cleaned
        for block in blocks
        if (cleaned := clean_document_text(block))
    )


def excerpt(value: str, limit: int = 360) -> str:
    cleaned = clean_document_text(value)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."


def safe_filename(value: str) -> str:
    filename = Path(value).name
    cleaned = SAFE_NAME_PATTERN.sub("_", filename).strip("._")
    return cleaned or "document"


def normalize_symbol(value: str) -> str:
    cleaned = SAFE_NAME_PATTERN.sub("_", value.upper()).strip("._")
    if not cleaned:
        raise ValueError("stock symbol could not be normalized")
    return cleaned


def normalize_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=IST)
    return value.astimezone(IST).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    temporary_path.replace(path)


def read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default
