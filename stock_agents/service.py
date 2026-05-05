from __future__ import annotations

import html
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Sequence

from stock_agents.models import AnalysisOptions, StockInput


IST = timezone(timedelta(hours=5, minutes=30), "IST")

USER_AGENT = (
    "StockAgentsAPI/0.1 "
    "(news and market context fetcher; stores title, summary, source, timestamp, link)"
)

INDIA_BUSINESS_RSS_SOURCES: tuple[dict[str, str | bool], ...] = (
    {
        "name": "Economic Times - Stocks",
        "url": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "domain": "economictimes.indiatimes.com",
        "search_enabled": True,
    },
    {
        "name": "Economic Times - Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "domain": "economictimes.indiatimes.com",
        "search_enabled": True,
    },
    {
        "name": "Moneycontrol - Stocks",
        "url": "https://www.moneycontrol.com/features/rss/news/business/stocks/",
        "domain": "moneycontrol.com",
        "search_enabled": True,
    },
    {
        "name": "Moneycontrol - Markets",
        "url": "https://www.moneycontrol.com/features/rss/news/business/markets/",
        "domain": "moneycontrol.com",
        "search_enabled": True,
    },
    {
        "name": "BusinessLine - Stock Markets",
        "url": "https://www.thehindubusinessline.com/markets/stock-markets/feeder/default.rss",
        "domain": "thehindubusinessline.com",
        "search_enabled": True,
    },
    {
        "name": "BusinessLine - Companies",
        "url": "https://www.thehindubusinessline.com/companies/feeder/default.rss",
        "domain": "thehindubusinessline.com",
        "search_enabled": True,
    },
    {
        "name": "RBI - Press Releases",
        "url": "https://www.rbi.org.in/pressreleases_rss.xml",
        "domain": "rbi.org.in",
        "search_enabled": False,
    },
)

POSITIVE_TERMS: dict[str, float] = {
    "beat estimates": 1.4,
    "beats estimates": 1.4,
    "profit rises": 1.3,
    "revenue rises": 1.0,
    "margin expands": 1.1,
    "wins order": 1.3,
    "bags order": 1.3,
    "approval": 0.7,
    "guidance": 0.4,
    "raises guidance": 1.0,
    "upgrade": 1.0,
    "buyback": 1.0,
    "dividend": 0.6,
    "record high": 1.0,
    "rallies": 0.9,
    "gains": 0.6,
    "expansion": 0.6,
    "rate cut": 0.5,
    "inflation eases": 0.5,
    "foreign inflows": 0.6,
}

NEGATIVE_TERMS: dict[str, float] = {
    "misses estimates": -1.4,
    "profit falls": -1.3,
    "loss widens": -1.5,
    "margin contracts": -1.1,
    "downgrade": -1.0,
    "cuts target": -1.0,
    "probe": -1.0,
    "fine": -0.9,
    "penalty": -1.0,
    "default": -1.4,
    "debt": -0.6,
    "shutdown": -1.1,
    "fraud": -1.5,
    "falls": -0.6,
    "drops": -0.6,
    "slumps": -1.0,
    "declines": -0.6,
    "rate hike": -0.5,
    "inflation rises": -0.5,
    "rupee falls": -0.3,
    "crude rises": -0.3,
    "tariff": -0.3,
}

GLOBAL_MARKET_PROXIES: tuple[dict[str, str], ...] = (
    {"key": "usa_broad", "label": "US Large Cap", "symbol": "SPY"},
    {"key": "usa_tech", "label": "US Tech", "symbol": "QQQ"},
    {"key": "europe_broad", "label": "Europe Large Cap", "symbol": "FEZ"},
    {"key": "japan_broad", "label": "Japan Large Cap", "symbol": "EWJ"},
    {"key": "china_broad", "label": "China Large Cap", "symbol": "MCHI"},
    {"key": "india_broad", "label": "India Large Cap", "symbol": "INDA"},
)

MACRO_SERIES_REQUESTS: tuple[dict[str, str], ...] = (
    {"key": "wti", "label": "WTI Crude", "function": "WTI", "interval": "daily"},
    {"key": "brent", "label": "Brent Crude", "function": "BRENT", "interval": "daily"},
    {"key": "natgas", "label": "Natural Gas", "function": "NATURAL_GAS", "interval": "daily"},
    {"key": "copper", "label": "Copper", "function": "COPPER", "interval": "daily"},
    {
        "key": "us10y",
        "label": "US Treasury 10Y",
        "function": "TREASURY_YIELD",
        "interval": "daily",
        "maturity": "10year",
    },
    {
        "key": "us2y",
        "label": "US Treasury 2Y",
        "function": "TREASURY_YIELD",
        "interval": "daily",
        "maturity": "2year",
    },
    {"key": "usd_inr", "label": "USD/INR", "function": "FX_DAILY", "from_symbol": "USD", "to_symbol": "INR"},
    {"key": "eur_usd", "label": "EUR/USD", "function": "FX_DAILY", "from_symbol": "EUR", "to_symbol": "USD"},
    {"key": "usd_jpy", "label": "USD/JPY", "function": "FX_DAILY", "from_symbol": "USD", "to_symbol": "JPY"},
)

MACRO_TERMS: tuple[str, ...] = (
    "rbi",
    "reserve bank",
    "repo rate",
    "monetary policy",
    "inflation",
    "cpi",
    "wpi",
    "gdp",
    "gst collections",
    "fiscal deficit",
    "bond yield",
    "us treasury",
    "rupee",
    "crude",
    "oil prices",
    "fii",
    "foreign portfolio investors",
    "foreign investors",
    "budget",
    "tariff",
)

CORPORATE_EVENT_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "earnings",
        "label": "Earnings or result event",
        "terms": (
            "quarterly results",
            "financial results",
            "q1 results",
            "q2 results",
            "q3 results",
            "q4 results",
            "board meeting",
            "result date",
            "to announce results",
            "earnings",
        ),
    },
    {
        "key": "guidance",
        "label": "Guidance or outlook",
        "terms": ("guidance", "outlook", "forecast", "margin outlook", "revenue outlook"),
    },
    {
        "key": "dividend",
        "label": "Dividend",
        "terms": ("dividend", "record date", "ex-dividend", "ex dividend", "interim dividend"),
    },
    {
        "key": "buyback",
        "label": "Buyback",
        "terms": ("buyback", "buy-back", "share repurchase", "tender offer"),
    },
    {
        "key": "split_bonus",
        "label": "Split or bonus issue",
        "terms": ("stock split", "share split", "bonus issue", "bonus shares", "sub-division"),
    },
    {
        "key": "analyst_revision",
        "label": "Analyst revision",
        "terms": ("upgrade", "downgrade", "target price", "price target", "raises target", "cuts target"),
    },
    {
        "key": "order_win",
        "label": "Order or contract",
        "terms": ("wins order", "bags order", "order win", "contract win", "large order"),
    },
)


@dataclass(frozen=True)
class TimeWindow:
    as_of: datetime
    previous_trading_day: str
    market_open: str
    market_close: str
    window_start: str
    window_end: str
    market_status: str
    is_trading_day: bool


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    ok: bool
    item_count: int
    warning: str | None = None
    skipped: bool = False


@dataclass(frozen=True)
class IngestedItem:
    bucket: str
    provider: str
    source: str
    title: str
    link: str
    published_at: str | None
    summary: str = ""
    source_domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeriesPoint:
    key: str
    label: str
    current_value: float | None
    previous_value: float | None
    delta: float | None
    change_percent: float | None
    as_of: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._current_href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_map = {key.lower(): value or "" for key, value in attrs}
        self._current_href = attrs_map.get("href", "")
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = clean_text(" ".join(self._text_parts))
        if text:
            self.links.append({"href": self._current_href, "text": text})
        self._current_href = None
        self._text_parts = []


def analyze_stock(stock: StockInput, options: AnalysisOptions) -> dict[str, Any]:
    return analyze_watchlist([stock], options)["stocks"][0]


def analyze_watchlist(stocks: Sequence[StockInput], options: AnalysisOptions) -> dict[str, Any]:
    time_window = derive_time_window(options)
    since = datetime.fromisoformat(time_window.window_start)
    macro_news, macro_status = fetch_macro_news(since=since, options=options)
    global_markets, market_status = fetch_global_market_performance(options)
    macro_series, series_status = fetch_macro_series(options)

    reports: list[dict[str, Any]] = []
    for stock in stocks:
        company_news, company_status = fetch_company_news(stock=stock, since=since, options=options)
        sector_news, sector_status = fetch_sector_news(stock=stock, since=since, options=options)
        announcements, announcement_status = fetch_exchange_announcements(
            stock=stock,
            since=since,
            options=options,
        )
        reports.append(
            build_stock_report(
                stock=stock,
                time_window=time_window,
                company_news=company_news,
                sector_news=sector_news,
                macro_news=macro_news,
                announcements=announcements,
                global_markets=global_markets,
                macro_series=macro_series,
                provider_statuses=[
                    *company_status,
                    *sector_status,
                    *macro_status,
                    *announcement_status,
                    *market_status,
                    *series_status,
                ],
            )
        )

    return {
        "generated_at": datetime.now(IST).isoformat(),
        "time_window": asdict(time_window),
        "stocks": reports,
    }


def derive_time_window(options: AnalysisOptions) -> TimeWindow:
    as_of = options.as_of.astimezone(IST) if options.as_of else datetime.now(IST)
    trading_day = as_of.date()
    is_trading_day = trading_day.weekday() < 5
    market_open = datetime.combine(trading_day, dt_time(9, 15), tzinfo=IST)
    market_close = datetime.combine(trading_day, dt_time(15, 30), tzinfo=IST)
    previous_day = previous_business_day(trading_day)
    previous_close = datetime.combine(previous_day, dt_time(15, 30), tzinfo=IST)
    if options.since:
        window_start = options.since.astimezone(IST) if options.since.tzinfo else options.since.replace(tzinfo=IST)
    else:
        window_start = previous_close - timedelta(hours=options.hours_before_previous_close)
    if trading_day.weekday() >= 5:
        market_status = "weekend"
    elif as_of < market_open:
        market_status = "pre-open"
    elif as_of > market_close:
        market_status = "post-close"
    else:
        market_status = "open"
    return TimeWindow(
        as_of=as_of,
        previous_trading_day=previous_day.isoformat(),
        market_open=market_open.isoformat(),
        market_close=market_close.isoformat(),
        window_start=window_start.isoformat(),
        window_end=as_of.isoformat(),
        market_status=market_status,
        is_trading_day=is_trading_day,
    )


def previous_business_day(today: date) -> date:
    current = today - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def build_stock_report(
    stock: StockInput,
    time_window: TimeWindow,
    company_news: Sequence[IngestedItem],
    sector_news: Sequence[IngestedItem],
    macro_news: Sequence[IngestedItem],
    announcements: Sequence[IngestedItem],
    global_markets: Sequence[SeriesPoint],
    macro_series: Sequence[SeriesPoint],
    provider_statuses: Sequence[ProviderStatus],
) -> dict[str, Any]:
    relevant_company = score_items_for_stock(stock, company_news, default_bucket="company_news")
    relevant_sector = score_items_for_stock(stock, sector_news, default_bucket="sector_news")
    relevant_macro = score_items_for_stock(stock, macro_news, default_bucket="macro_news")
    relevant_announcements = score_items_for_stock(stock, announcements, default_bucket="corporate_announcements")

    normalized = dedupe_ranked_items(
        [*relevant_company, *relevant_sector, *relevant_macro, *relevant_announcements]
    )
    events = extract_events(normalized)
    company_sentiment = summarize_sentiment(relevant_company)
    sector_sentiment = summarize_sentiment(relevant_sector)
    macro_sentiment = summarize_sentiment(relevant_macro)
    announcement_sentiment = summarize_sentiment(relevant_announcements)
    event_signal = summarize_events(events)
    global_cues = build_global_cues(stock, global_markets, macro_series, macro_sentiment["score"])

    overall_score = (
        (company_sentiment["score"] * 0.40)
        + (sector_sentiment["score"] * 0.10)
        + (macro_sentiment["score"] * 0.10)
        + (announcement_sentiment["score"] * 0.15)
        + (event_signal["score"] * 0.10)
        + (global_cues["score"] * 0.15)
    )
    overall_score = round(max(-1.0, min(1.0, overall_score)), 3)

    return {
        "symbol": stock.symbol,
        "name": stock.name,
        "sector": stock.sector,
        "time_window": asdict(time_window),
        "provider_status": [asdict(status) for status in provider_statuses],
        "ingestion": {
            "company_news": [ranked_item_to_dict(item) for item in relevant_company[:20]],
            "sector_news": [ranked_item_to_dict(item) for item in relevant_sector[:15]],
            "india_macro_news": [ranked_item_to_dict(item) for item in relevant_macro[:15]],
            "corporate_announcements": [ranked_item_to_dict(item) for item in relevant_announcements[:15]],
            "global_market_performance": [asdict(item) for item in global_markets],
            "macro_series": [asdict(item) for item in macro_series],
        },
        "processing": {
            "normalized_item_count": len(normalized),
            "entity_resolution": build_entity_resolution(stock, normalized[:20]),
            "relevance_summary": {
                "company_news_count": len(relevant_company),
                "sector_news_count": len(relevant_sector),
                "macro_news_count": len(relevant_macro),
                "announcement_count": len(relevant_announcements),
            },
            "events": [event_to_dict(event) for event in events[:20]],
            "sentiment": {
                "company_news": company_sentiment,
                "sector_news": sector_sentiment,
                "india_macro_news": macro_sentiment,
                "corporate_announcements": announcement_sentiment,
            },
            "global_cues": global_cues,
        },
        "report": {
            "label": label_for_score(overall_score),
            "overall_score": overall_score,
            "confidence": confidence_for_stock_report(
                relevant_company,
                relevant_sector,
                relevant_macro,
                relevant_announcements,
                events,
            ),
            "headline_count": len(normalized),
            "key_drivers": top_drivers(normalized, events, global_cues),
            "trading_context": build_trading_context(
                stock=stock,
                overall_score=overall_score,
                company_sentiment=company_sentiment,
                event_signal=event_signal,
                global_cues=global_cues,
            ),
        },
    }


def fetch_company_news(
    stock: StockInput,
    since: datetime,
    options: AnalysisOptions,
) -> tuple[list[IngestedItem], list[ProviderStatus]]:
    query = build_company_query(stock)
    items: list[IngestedItem] = []
    statuses: list[ProviderStatus] = []
    gnews_items, gnews_status = fetch_gnews(bucket="company_news", query=query, since=since, limit=options.company_news_limit)
    newsdata_items, newsdata_status = fetch_newsdata(
        bucket="company_news",
        query=query,
        since=since,
        limit=options.company_news_limit,
    )
    gdelt_items, gdelt_status = fetch_gdelt(
        bucket="company_news",
        query=query,
        since=since,
        limit=options.company_news_limit,
    )
    rss_items, rss_status = fetch_rss_news(
        bucket="company_news",
        since=since,
        filter_terms=stock_terms(stock),
        search_query=query,
        limit=options.company_news_limit,
        include_site_search=options.include_site_search,
    )
    items.extend(gnews_items)
    items.extend(newsdata_items)
    items.extend(gdelt_items)
    items.extend(rss_items)
    statuses.extend([gnews_status, newsdata_status, gdelt_status, rss_status])
    return items, statuses


def fetch_sector_news(
    stock: StockInput,
    since: datetime,
    options: AnalysisOptions,
) -> tuple[list[IngestedItem], list[ProviderStatus]]:
    query = build_sector_query(stock)
    if not query:
        return [], [ProviderStatus(provider="sector_query", ok=False, item_count=0, warning="sector terms missing", skipped=True)]
    items: list[IngestedItem] = []
    statuses: list[ProviderStatus] = []
    gnews_items, gnews_status = fetch_gnews(bucket="sector_news", query=query, since=since, limit=options.sector_news_limit)
    newsdata_items, newsdata_status = fetch_newsdata(
        bucket="sector_news",
        query=query,
        since=since,
        limit=options.sector_news_limit,
    )
    gdelt_items, gdelt_status = fetch_gdelt(
        bucket="sector_news",
        query=query,
        since=since,
        limit=options.sector_news_limit,
    )
    sector_terms = [stock.sector] if stock.sector else []
    sector_terms.extend(stock.sector_keywords)
    rss_items, rss_status = fetch_rss_news(
        bucket="sector_news",
        since=since,
        filter_terms=tuple(term for term in sector_terms if term),
        search_query=query,
        limit=options.sector_news_limit,
        include_site_search=options.include_site_search,
    )
    items.extend(gnews_items)
    items.extend(newsdata_items)
    items.extend(gdelt_items)
    items.extend(rss_items)
    statuses.extend([gnews_status, newsdata_status, gdelt_status, rss_status])
    return items, statuses


def fetch_macro_news(
    since: datetime,
    options: AnalysisOptions,
) -> tuple[list[IngestedItem], list[ProviderStatus]]:
    query = build_macro_query()
    items: list[IngestedItem] = []
    statuses: list[ProviderStatus] = []
    if options.include_broad_business_news:
        gnews_items, gnews_status = fetch_gnews(bucket="macro_news", query=query, since=since, limit=options.macro_news_limit)
        newsdata_items, newsdata_status = fetch_newsdata(
            bucket="macro_news",
            query=query,
            since=since,
            limit=options.macro_news_limit,
        )
        gdelt_items, gdelt_status = fetch_gdelt(
            bucket="macro_news",
            query=query,
            since=since,
            limit=options.macro_news_limit,
        )
        items.extend(gnews_items)
        items.extend(newsdata_items)
        items.extend(gdelt_items)
        statuses.extend([gnews_status, newsdata_status, gdelt_status])
    rss_items, rss_status = fetch_rss_news(
        bucket="macro_news",
        since=since,
        filter_terms=MACRO_TERMS,
        search_query=query,
        limit=options.macro_news_limit,
        include_site_search=False,
    )
    items.extend(rss_items)
    statuses.append(rss_status)
    return items, statuses


def fetch_exchange_announcements(
    stock: StockInput,
    since: datetime,
    options: AnalysisOptions,
) -> tuple[list[IngestedItem], list[ProviderStatus]]:
    nse_items, nse_status = fetch_nse_announcements(stock, since, options.exchange_announcement_limit, options.timeout_seconds)
    bse_items, bse_status = fetch_bse_announcements(stock, since, options.exchange_announcement_limit, options.timeout_seconds)
    return [*nse_items, *bse_items], [nse_status, bse_status]


def fetch_global_market_performance(
    options: AnalysisOptions,
) -> tuple[list[SeriesPoint], list[ProviderStatus]]:
    api_key = getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return [], [ProviderStatus(provider="alphavantage_global_markets", ok=False, item_count=0, warning="ALPHAVANTAGE_API_KEY not set", skipped=True)]
    points: list[SeriesPoint] = []
    failures = 0
    for proxy in GLOBAL_MARKET_PROXIES[: options.global_market_limit]:
        payload = alphavantage_request(
            {
                "function": "GLOBAL_QUOTE",
                "symbol": proxy["symbol"],
                "apikey": api_key,
            },
            timeout_seconds=options.timeout_seconds,
        )
        point = parse_global_quote(proxy["key"], proxy["label"], payload)
        if point:
            points.append(point)
        else:
            failures += 1
    status = ProviderStatus(
        provider="alphavantage_global_markets",
        ok=bool(points),
        item_count=len(points),
        warning=None if failures == 0 else f"{failures} quote(s) could not be parsed",
    )
    return points, [status]


def fetch_macro_series(options: AnalysisOptions) -> tuple[list[SeriesPoint], list[ProviderStatus]]:
    api_key = getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return [], [ProviderStatus(provider="alphavantage_macro_series", ok=False, item_count=0, warning="ALPHAVANTAGE_API_KEY not set", skipped=True)]
    points: list[SeriesPoint] = []
    failures = 0
    for request_cfg in MACRO_SERIES_REQUESTS[: options.macro_series_limit]:
        payload = alphavantage_request(
            {**request_cfg, "apikey": api_key},
            timeout_seconds=options.timeout_seconds,
        )
        point = parse_series_point(request_cfg, payload)
        if point:
            points.append(point)
        else:
            failures += 1
    status = ProviderStatus(
        provider="alphavantage_macro_series",
        ok=bool(points),
        item_count=len(points),
        warning=None if failures == 0 else f"{failures} series could not be parsed",
    )
    return points, [status]


def fetch_nse_announcements(
    stock: StockInput,
    since: datetime,
    limit: int,
    timeout_seconds: float,
) -> tuple[list[IngestedItem], ProviderStatus]:
    symbol = stock.nse_symbol or stock.symbol
    urls = [
        f"https://www.nseindia.com/companies-listing/corporate-filings-application?param={urllib.parse.quote(symbol)}",
        f"https://www.nseindia.com/companies-listing/corporate-filings-announcements?symbol={urllib.parse.quote(symbol)}",
    ]
    items: list[IngestedItem] = []
    warnings: list[str] = []
    for url in urls:
        body = fetch_text(url, timeout_seconds=timeout_seconds)
        if not body:
            warnings.append(f"failed {url}")
            continue
        parsed = parse_html_announcement_links(
            body=body,
            provider="nse_announcements",
            bucket="corporate_announcements",
            source="NSE",
            base_url="https://www.nseindia.com",
            terms=stock_terms(stock),
        )
        items.extend(item for item in parsed if is_recent(item.published_at, since))
    deduped = dedupe_ingested_items(items)[:limit]
    return deduped, ProviderStatus(
        provider="nse_announcements",
        ok=bool(deduped),
        item_count=len(deduped),
        warning=None if deduped or not warnings else "; ".join(warnings[:2]),
    )


def fetch_bse_announcements(
    stock: StockInput,
    since: datetime,
    limit: int,
    timeout_seconds: float,
) -> tuple[list[IngestedItem], ProviderStatus]:
    urls = ["https://m.bseindia.com/corporates.aspx"]
    if stock.bse_scrip_code:
        urls.append(f"https://m.bseindia.com/StockReach.aspx?scripcd={urllib.parse.quote(stock.bse_scrip_code)}")
    items: list[IngestedItem] = []
    warnings: list[str] = []
    for url in urls:
        body = fetch_text(url, timeout_seconds=timeout_seconds)
        if not body:
            warnings.append(f"failed {url}")
            continue
        parsed = parse_html_announcement_links(
            body=body,
            provider="bse_announcements",
            bucket="corporate_announcements",
            source="BSE",
            base_url="https://m.bseindia.com",
            terms=stock_terms(stock),
        )
        items.extend(item for item in parsed if is_recent(item.published_at, since))
    deduped = dedupe_ingested_items(items)[:limit]
    return deduped, ProviderStatus(
        provider="bse_announcements",
        ok=bool(deduped),
        item_count=len(deduped),
        warning=None if deduped or not warnings else "; ".join(warnings[:2]),
    )


def fetch_gnews(bucket: str, query: str, since: datetime, limit: int) -> tuple[list[IngestedItem], ProviderStatus]:
    api_key = getenv("GNEWS_API_KEY")
    if not api_key:
        return [], ProviderStatus(provider="gnews", ok=False, item_count=0, warning="GNEWS_API_KEY not set", skipped=True)
    payload = http_get_json(
        "https://gnews.io/api/v4/search",
        {
            "q": query,
            "lang": "en",
            "country": "in",
            "max": str(limit),
            "from": since.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "sortby": "publishedAt",
            "apikey": api_key,
        },
    )
    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    items = [
        IngestedItem(
            bucket=bucket,
            provider="gnews",
            source=str(article.get("source", {}).get("name", "GNews")),
            title=clean_text(str(article.get("title", ""))),
            link=str(article.get("url", "")).strip(),
            published_at=normalize_date(str(article.get("publishedAt", ""))),
            summary=clean_text(str(article.get("description", ""))),
            source_domain=urllib.parse.urlparse(str(article.get("url", ""))).netloc,
        )
        for article in articles
        if article.get("title")
    ]
    deduped = dedupe_ingested_items(items)
    return deduped, ProviderStatus(provider="gnews", ok=bool(deduped), item_count=len(deduped))


def fetch_newsdata(bucket: str, query: str, since: datetime, limit: int) -> tuple[list[IngestedItem], ProviderStatus]:
    api_key = getenv("NEWSDATA_API_KEY")
    if not api_key:
        return [], ProviderStatus(provider="newsdata", ok=False, item_count=0, warning="NEWSDATA_API_KEY not set", skipped=True)
    payload = http_get_json(
        "https://newsdata.io/api/1/latest",
        {
            "apikey": api_key,
            "q": query,
            "country": "in",
            "language": "en",
            "size": str(limit),
        },
    )
    results = payload.get("results", []) if isinstance(payload, dict) else []
    items = [
        IngestedItem(
            bucket=bucket,
            provider="newsdata",
            source=clean_text(str(article.get("source_name", "NewsData.io"))),
            title=clean_text(str(article.get("title", ""))),
            link=str(article.get("link", "")).strip(),
            published_at=normalize_date(str(article.get("pubDate", ""))),
            summary=clean_text(str(article.get("description", ""))),
            source_domain=urllib.parse.urlparse(str(article.get("link", ""))).netloc,
        )
        for article in results
        if article.get("title")
    ]
    filtered = [item for item in items if is_recent(item.published_at, since)]
    deduped = dedupe_ingested_items(filtered)
    return deduped, ProviderStatus(provider="newsdata", ok=bool(deduped), item_count=len(deduped))


def fetch_gdelt(
    bucket: str,
    query: str,
    since: datetime,
    limit: int,
) -> tuple[list[IngestedItem], ProviderStatus]:
    payload = http_get_json(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        {
            "query": query,
            "mode": "ArtList",
            "maxrecords": str(limit),
            "sort": "datedesc",
            "format": "json",
        },
    )
    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    items = [
        IngestedItem(
            bucket=bucket,
            provider="gdelt",
            source=clean_text(str(article.get("domain", "GDELT"))),
            title=clean_text(str(article.get("title", ""))),
            link=str(article.get("url", "")).strip(),
            published_at=normalize_date(str(article.get("seendate", ""))),
            summary=clean_text(str(article.get("excerpt", ""))),
            source_domain=clean_text(str(article.get("domain", ""))),
        )
        for article in articles
        if article.get("title")
    ]
    filtered = [item for item in items if is_recent(item.published_at, since)]
    deduped = dedupe_ingested_items(filtered)
    return deduped, ProviderStatus(provider="gdelt", ok=bool(deduped), item_count=len(deduped))


def fetch_rss_news(
    bucket: str,
    since: datetime,
    filter_terms: Sequence[str],
    search_query: str,
    limit: int,
    include_site_search: bool,
) -> tuple[list[IngestedItem], ProviderStatus]:
    urls = list(INDIA_BUSINESS_RSS_SOURCES)
    if include_site_search:
        domains = sorted(
            {
                str(source["domain"])
                for source in INDIA_BUSINESS_RSS_SOURCES
                if bool(source.get("search_enabled", False))
            }
        )
        for domain in domains:
            site_query = f"{search_query} site:{domain} when:2d"
            urls.append(
                {
                    "name": f"Google News RSS - {domain}",
                    "url": google_news_search_url(site_query),
                    "domain": domain,
                    "search_enabled": False,
                }
            )
    items: list[IngestedItem] = []
    warnings = 0
    for source in urls:
        body = fetch_bytes(str(source["url"]), timeout_seconds=12.0)
        if not body:
            warnings += 1
            continue
        feed_items = parse_feed(
            body=body,
            provider="rss",
            bucket=bucket,
            source_name=str(source["name"]),
            source_domain=str(source["domain"]),
        )
        for item in feed_items:
            if not is_recent(item.published_at, since):
                continue
            if filter_terms and not item_matches_terms(item, filter_terms):
                continue
            items.append(item)
    deduped = dedupe_ingested_items(items)[:limit]
    warning = None if warnings == 0 else f"{warnings} RSS source(s) failed"
    return deduped, ProviderStatus(provider="rss_feeds", ok=bool(deduped), item_count=len(deduped), warning=warning)


def build_company_query(stock: StockInput) -> str:
    quoted = " OR ".join(f'"{term}"' for term in stock_terms(stock)[:6])
    suffix = '("stock" OR "share price" OR "results" OR "order" OR "guidance")'
    return f"({quoted}) {suffix} India"


def build_sector_query(stock: StockInput) -> str:
    parts: list[str] = []
    if stock.sector:
        parts.append(stock.sector)
    parts.extend(stock.sector_keywords)
    deduped = dedupe_terms(parts)
    if not deduped:
        return ""
    quoted = " OR ".join(f'"{term}"' for term in deduped[:6])
    return f"({quoted}) India business market"


def build_macro_query() -> str:
    quoted = " OR ".join(f'"{term}"' for term in MACRO_TERMS[:10])
    return f"({quoted}) India business market"


def stock_terms(stock: StockInput) -> tuple[str, ...]:
    terms = [stock.name, *stock.aliases, stock.symbol]
    if stock.nse_symbol and stock.nse_symbol.upper() != stock.symbol.upper():
        terms.append(stock.nse_symbol)
    return tuple(dedupe_terms(terms))


def dedupe_terms(terms: Sequence[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term:
            continue
        cleaned = term.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def google_news_search_url(query: str) -> str:
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    )


def score_items_for_stock(
    stock: StockInput,
    items: Sequence[IngestedItem],
    default_bucket: str,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in items:
        relevance_kind, relevance_score, matched_terms = relevance_for_stock(stock, item, default_bucket)
        if relevance_score <= 0:
            continue
        sentiment_score, sentiment_drivers = sentiment_for_item(item)
        ranked.append(
            {
                "item": item,
                "relevance": relevance_kind,
                "relevance_score": relevance_score,
                "matched_terms": matched_terms,
                "sentiment_score": sentiment_score,
                "sentiment_label": label_for_score(sentiment_score),
                "sentiment_drivers": sentiment_drivers[:6],
            }
        )
    ranked.sort(
        key=lambda row: (
            row["relevance_score"],
            abs(row["sentiment_score"]),
            row["item"].published_at or "",
        ),
        reverse=True,
    )
    return ranked


def relevance_for_stock(
    stock: StockInput,
    item: IngestedItem,
    default_bucket: str,
) -> tuple[str, float, list[str]]:
    haystack = f"{item.title} {item.summary}".lower()
    alias_matches = [term for term in stock_terms(stock) if exact_phrase(term, haystack)]
    if alias_matches:
        score = 1.0 if default_bucket != "macro_news" else 0.9
        return "stock-specific", score, alias_matches
    sector_matches = [term for term in ([stock.sector] if stock.sector else []) + stock.sector_keywords if term and contains_term(term, haystack)]
    if sector_matches:
        return "sector", 0.65, sector_matches
    macro_matches = [term for term in MACRO_TERMS if contains_term(term, haystack)]
    if default_bucket == "macro_news" and macro_matches:
        return "macro", 0.4, macro_matches[:4]
    if item.bucket == "corporate_announcements" and alias_matches:
        return "announcement", 1.0, alias_matches
    return "none", 0.0, []


def dedupe_ranked_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in items:
        item = row["item"]
        link_key = normalized_link(item.link)
        title_key = dedupe_title_key(item.title)
        if link_key and link_key in seen_links:
            continue
        if title_key in seen_titles:
            continue
        if link_key:
            seen_links.add(link_key)
        seen_titles.add(title_key)
        deduped.append(row)
    deduped.sort(
        key=lambda row: (
            row["relevance_score"],
            abs(row["sentiment_score"]),
            row["item"].published_at or "",
        ),
        reverse=True,
    )
    return deduped


def extract_events(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in items:
        text = f"{row['item'].title} {row['item'].summary}".lower()
        for definition in CORPORATE_EVENT_DEFINITIONS:
            terms = definition["terms"]
            matched = [term for term in terms if contains_term(term, text)]
            if not matched:
                continue
            events.append(
                {
                    "event_type": definition["key"],
                    "event_label": definition["label"],
                    "title": row["item"].title,
                    "source": row["item"].source,
                    "link": row["item"].link,
                    "published_at": row["item"].published_at,
                    "direction_score": row["sentiment_score"],
                    "importance_score": round(min(1.0, 0.55 + (0.20 * row["relevance_score"]) + (0.10 * abs(row["sentiment_score"]))), 3),
                    "relevance": row["relevance"],
                    "drivers": matched[:6],
                    "summary": row["item"].summary,
                }
            )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for event in events:
        key = (event["event_type"], dedupe_title_key(event["title"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    unique.sort(
        key=lambda event: (event["importance_score"], abs(event["direction_score"]), event["published_at"] or ""),
        reverse=True,
    )
    return unique


def summarize_sentiment(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"label": "neutral", "score": 0.0, "confidence": 0.0, "item_count": 0}
    denominator = sum(row["relevance_score"] for row in items)
    weighted = sum(row["sentiment_score"] * row["relevance_score"] for row in items) / denominator if denominator else 0.0
    confidence = min(1.0, 0.25 + (len(items) / 10.0))
    return {
        "label": label_for_score(weighted),
        "score": round(weighted, 3),
        "confidence": round(confidence, 3),
        "item_count": len(items),
    }


def summarize_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"score": 0.0, "event_count": 0}
    denominator = sum(event["importance_score"] for event in events)
    score = sum(event["direction_score"] * event["importance_score"] for event in events) / denominator if denominator else 0.0
    return {"score": round(score, 3), "event_count": len(events)}


def build_global_cues(
    stock: StockInput,
    global_markets: Sequence[SeriesPoint],
    macro_series: Sequence[SeriesPoint],
    macro_news_score: float,
) -> dict[str, Any]:
    sector_tags = " ".join(
        term.lower() for term in dedupe_terms([stock.sector or "", *stock.sector_keywords, stock.name])
    )
    cues: list[dict[str, Any]] = []
    score = 0.0

    for point in global_markets:
        if point.change_percent is None:
            continue
        normalized = math.tanh(point.change_percent / 2.5)
        weight = 0.2
        if point.key == "india_broad":
            weight = 0.35
        elif point.key == "usa_tech" and any(tag in sector_tags for tag in ("it", "software", "technology", "services")):
            weight = 0.3
        elif point.key == "china_broad" and any(tag in sector_tags for tag in ("metal", "mining", "steel", "commodity")):
            weight = 0.25
        contribution = normalized * weight
        if abs(contribution) < 0.03:
            continue
        score += contribution
        cues.append(
            {
                "key": point.key,
                "label": point.label,
                "score": round(contribution, 3),
                "reason": f"{point.label} moved {point.change_percent:.2f}%",
            }
        )

    series_map = {point.key: point for point in macro_series}
    oil_point = series_map.get("wti") or series_map.get("brent")
    if oil_point and oil_point.delta is not None:
        if any(tag in sector_tags for tag in ("oil", "gas", "upstream", "exploration", "refining")):
            oil_score = math.tanh(oil_point.delta / 3.0) * 0.25
        elif any(tag in sector_tags for tag in ("aviation", "airline", "paint", "cement", "chemical", "logistics", "tyre")):
            oil_score = math.tanh(-oil_point.delta / 3.0) * 0.25
        else:
            oil_score = 0.0
        if abs(oil_score) >= 0.03:
            score += oil_score
            cues.append(
                {
                    "key": oil_point.key,
                    "label": oil_point.label,
                    "score": round(oil_score, 3),
                    "reason": f"{oil_point.label} delta {oil_point.delta:.2f}",
                }
            )

    copper_point = series_map.get("copper")
    if copper_point and copper_point.delta is not None and any(
        tag in sector_tags for tag in ("metal", "copper", "wire", "mining", "capital goods", "infrastructure")
    ):
        copper_score = math.tanh(copper_point.delta / 2.0) * 0.2
        if abs(copper_score) >= 0.03:
            score += copper_score
            cues.append(
                {
                    "key": copper_point.key,
                    "label": copper_point.label,
                    "score": round(copper_score, 3),
                    "reason": f"{copper_point.label} delta {copper_point.delta:.2f}",
                }
            )

    if any(tag in sector_tags for tag in ("bank", "financial", "nbfc", "insurance")):
        yield_point = series_map.get("us10y")
        if yield_point and yield_point.delta is not None:
            yield_score = math.tanh(-yield_point.delta) * 0.1
            if abs(yield_score) >= 0.02:
                score += yield_score
                cues.append(
                    {
                        "key": yield_point.key,
                        "label": yield_point.label,
                        "score": round(yield_score, 3),
                        "reason": f"{yield_point.label} delta {yield_point.delta:.2f}",
                    }
                )

    score += macro_news_score * 0.15
    return {
        "score": round(max(-1.0, min(1.0, score)), 3),
        "cue_count": len(cues),
        "items": cues[:10],
    }


def confidence_for_stock_report(
    company_news: Sequence[dict[str, Any]],
    sector_news: Sequence[dict[str, Any]],
    macro_news: Sequence[dict[str, Any]],
    announcements: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> float:
    total = len(company_news) + len(sector_news) + len(macro_news) + len(announcements)
    confidence = 0.20
    confidence += min(0.35, total / 30.0)
    confidence += min(0.20, len(events) / 12.0)
    confidence += min(0.25, len(announcements) / 10.0)
    return round(min(1.0, confidence), 3)


def build_trading_context(
    stock: StockInput,
    overall_score: float,
    company_sentiment: dict[str, Any],
    event_signal: dict[str, Any],
    global_cues: dict[str, Any],
) -> dict[str, Any]:
    posture = "neutral"
    if overall_score >= 0.2:
        posture = "constructive"
    elif overall_score <= -0.2:
        posture = "defensive"
    return {
        "stock": stock.symbol,
        "posture": posture,
        "summary": (
            f"{stock.name} screens {label_for_score(overall_score)} with company-news score "
            f"{company_sentiment['score']}, event score {event_signal['score']}, and global-cue score {global_cues['score']}."
        ),
    }


def build_entity_resolution(stock: StockInput, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    resolution = []
    for row in items:
        resolution.append(
            {
                "title": row["item"].title,
                "matched_terms": row["matched_terms"],
                "relevance": row["relevance"],
                "source": row["item"].source,
            }
        )
    return {
        "stock_terms": list(stock_terms(stock)),
        "top_matches": resolution[:10],
    }


def top_drivers(
    items: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    global_cues: dict[str, Any],
) -> list[str]:
    drivers: list[str] = []
    for row in items[:5]:
        if row["item"].title:
            drivers.append(row["item"].title)
    for event in events[:3]:
        drivers.append(f"{event['event_label']}: {event['title']}")
    for cue in global_cues["items"][:2]:
        drivers.append(cue["reason"])
    return drivers[:10]


def event_to_dict(event: dict[str, Any]) -> dict[str, Any]:
    return dict(event)


def ranked_item_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = row["item"]
    return {
        "bucket": item.bucket,
        "provider": item.provider,
        "source": item.source,
        "title": item.title,
        "link": item.link,
        "published_at": item.published_at,
        "summary": item.summary,
        "relevance": row["relevance"],
        "relevance_score": row["relevance_score"],
        "matched_terms": row["matched_terms"],
        "sentiment_label": row["sentiment_label"],
        "sentiment_score": row["sentiment_score"],
        "sentiment_drivers": row["sentiment_drivers"],
        "metadata": item.metadata,
    }


def parse_feed(
    body: bytes,
    provider: str,
    bucket: str,
    source_name: str,
    source_domain: str,
) -> list[IngestedItem]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    if strip_namespace(root.tag).lower() == "rss":
        nodes = root.findall(".//item")
    else:
        nodes = [node for node in root.iter() if strip_namespace(node.tag).lower() == "entry"]
    items: list[IngestedItem] = []
    for node in nodes:
        parsed = parse_feed_node(node)
        if not parsed or not parsed.get("title"):
            continue
        items.append(
            IngestedItem(
                bucket=bucket,
                provider=provider,
                source=source_name,
                title=clean_text(str(parsed.get("title", ""))),
                link=str(parsed.get("link", "")).strip(),
                published_at=normalize_date(str(parsed.get("published_at", ""))),
                summary=clean_text(str(parsed.get("summary", ""))),
                source_domain=source_domain,
            )
        )
    return items


def parse_feed_node(node: ET.Element) -> dict[str, str] | None:
    values = {
        strip_namespace(child.tag).lower(): child
        for child in list(node)
        if isinstance(child.tag, str)
    }
    title = text_of(values.get("title"))
    link = text_of(values.get("link"))
    if not link and values.get("link") is not None:
        link = values["link"].attrib.get("href", "")
    summary = text_of(values.get("description")) or text_of(values.get("summary"))
    published_at = (
        text_of(values.get("pubdate"))
        or text_of(values.get("published"))
        or text_of(values.get("updated"))
    )
    if not title:
        return None
    return {"title": title, "link": link, "summary": summary, "published_at": published_at}


def parse_html_announcement_links(
    body: str,
    provider: str,
    bucket: str,
    source: str,
    base_url: str,
    terms: Sequence[str],
) -> list[IngestedItem]:
    parser = LinkParser()
    parser.feed(body)
    items: list[IngestedItem] = []
    for link in parser.links:
        text = link["text"]
        href = urllib.parse.urljoin(base_url, link["href"])
        if len(text) < 16:
            continue
        if terms and not any(term.lower() in text.lower() for term in terms):
            continue
        items.append(
            IngestedItem(
                bucket=bucket,
                provider=provider,
                source=source,
                title=text,
                link=href,
                published_at=None,
                summary="",
                source_domain=urllib.parse.urlparse(base_url).netloc,
            )
        )
    return items


def dedupe_ingested_items(items: Sequence[IngestedItem]) -> list[IngestedItem]:
    deduped: list[IngestedItem] = []
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    for item in items:
        link_key = normalized_link(item.link)
        title_key = dedupe_title_key(item.title)
        if link_key and link_key in seen_links:
            continue
        if title_key in seen_titles:
            continue
        if link_key:
            seen_links.add(link_key)
        seen_titles.add(title_key)
        deduped.append(item)
    return deduped


def item_matches_terms(item: IngestedItem, terms: Sequence[str]) -> bool:
    haystack = f"{item.title} {item.summary}".lower()
    return any(contains_term(term, haystack) for term in terms if term)


def sentiment_for_item(item: IngestedItem) -> tuple[float, list[str]]:
    text = f"{item.title}. {item.summary}".lower()
    score = 0.0
    drivers: list[str] = []
    for term, weight in POSITIVE_TERMS.items():
        if contains_term(term, text):
            score += weight
            drivers.append(term)
    for term, weight in NEGATIVE_TERMS.items():
        if contains_term(term, text):
            score += weight
            drivers.append(term)
    if not drivers:
        return 0.0, []
    return round(math.tanh(score / 3.0), 3), drivers


def parse_global_quote(key: str, label: str, payload: dict[str, Any]) -> SeriesPoint | None:
    row = payload.get("Global Quote", {}) if isinstance(payload, dict) else {}
    if not row:
        return None
    price = parse_float(row.get("05. price"))
    change = parse_float(row.get("09. change"))
    change_percent = parse_percentage(row.get("10. change percent"))
    as_of = normalize_date(str(row.get("07. latest trading day", "")))
    previous_price = price - change if price is not None and change is not None else None
    return SeriesPoint(
        key=key,
        label=label,
        current_value=price,
        previous_value=previous_price,
        delta=change,
        change_percent=change_percent,
        as_of=as_of,
        metadata={"symbol": row.get("01. symbol")},
    )


def parse_series_point(request_cfg: dict[str, str], payload: dict[str, Any]) -> SeriesPoint | None:
    function_name = request_cfg["function"]
    if function_name == "FX_DAILY":
        return parse_fx_daily_point(request_cfg, payload)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        return None
    values = [row for row in data if parse_float(row.get("value")) is not None]
    if not values:
        return None
    current = values[0]
    previous = values[1] if len(values) > 1 else None
    current_value = parse_float(current.get("value"))
    previous_value = parse_float(previous.get("value")) if previous else None
    delta = None if current_value is None or previous_value is None else current_value - previous_value
    pct = None
    if delta is not None and previous_value not in (None, 0.0):
        pct = (delta / previous_value) * 100.0
    return SeriesPoint(
        key=request_cfg["key"],
        label=request_cfg["label"],
        current_value=current_value,
        previous_value=previous_value,
        delta=delta,
        change_percent=pct,
        as_of=normalize_date(str(current.get("date", ""))),
        metadata={"function": function_name, "maturity": request_cfg.get("maturity")},
    )


def parse_fx_daily_point(request_cfg: dict[str, str], payload: dict[str, Any]) -> SeriesPoint | None:
    series = payload.get("Time Series FX (Daily)", {}) if isinstance(payload, dict) else {}
    if not isinstance(series, dict) or not series:
        return None
    dates = sorted(series.keys(), reverse=True)
    current_row = series[dates[0]]
    previous_row = series[dates[1]] if len(dates) > 1 else None
    current_value = parse_float(current_row.get("4. close"))
    previous_value = parse_float(previous_row.get("4. close")) if previous_row else None
    delta = None if current_value is None or previous_value is None else current_value - previous_value
    pct = None
    if delta is not None and previous_value not in (None, 0.0):
        pct = (delta / previous_value) * 100.0
    return SeriesPoint(
        key=request_cfg["key"],
        label=request_cfg["label"],
        current_value=current_value,
        previous_value=previous_value,
        delta=delta,
        change_percent=pct,
        as_of=normalize_date(dates[0]),
        metadata={
            "from_symbol": request_cfg.get("from_symbol"),
            "to_symbol": request_cfg.get("to_symbol"),
        },
    )


def alphavantage_request(params: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
    filtered = {key: value for key, value in params.items() if key not in {"key", "label"}}
    return http_get_json("https://www.alphavantage.co/query", filtered, timeout_seconds=timeout_seconds)


def http_get_json(
    url: str,
    params: dict[str, str],
    timeout_seconds: float = 12.0,
) -> dict[str, Any]:
    try:
        request = urllib.request.Request(
            url + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(2_500_000)
        return json.loads(body.decode("utf-8", errors="ignore"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError):
        return {}


def fetch_bytes(url: str, timeout_seconds: float = 12.0) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read(2_500_000)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def fetch_text(url: str, timeout_seconds: float = 12.0) -> str | None:
    body = fetch_bytes(url, timeout_seconds=timeout_seconds)
    if body is None:
        return None
    return body.decode("utf-8", errors="ignore")


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def text_of(node: ET.Element | None) -> str:
    return "" if node is None or node.text is None else node.text.strip()


def contains_term(term: str, text: str) -> bool:
    normalized = re.escape(term.strip().lower())
    if not normalized:
        return False
    return re.search(rf"(?<![a-z0-9]){normalized}(?![a-z0-9])", text.lower()) is not None


def exact_phrase(phrase: str, text: str) -> bool:
    normalized = re.escape(phrase.strip().lower())
    if not normalized:
        return False
    return re.search(rf"(?<![a-z0-9]){normalized}(?![a-z0-9])", text.lower()) is not None


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_date(value: str) -> str | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            if re.fullmatch(r"\d{14}", value):
                parsed = datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            elif re.fullmatch(r"\d{8}T\d{6}Z", value):
                parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            else:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST).isoformat()


def is_recent(published_at: str | None, since: datetime) -> bool:
    if not published_at:
        return True
    try:
        parsed = datetime.fromisoformat(published_at)
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=IST)
    return parsed >= since


def parse_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def parse_percentage(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return None


def normalized_link(value: str) -> str:
    return value.split("?")[0].lower().rstrip("/") if value else ""


def dedupe_title_key(value: str) -> str:
    return re.sub(r"\W+", "", value.lower())


def label_for_score(score: float) -> str:
    if score >= 0.2:
        return "bullish"
    if score <= -0.2:
        return "bearish"
    return "neutral"


def getenv(key: str) -> str | None:
    import os

    value = os.getenv(key)
    if value and value.strip():
        return value.strip()
    return None
