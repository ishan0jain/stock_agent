from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from stock_agents.models import AnalysisOptions, StockInput
from stock_agents.service import IST, ProviderStatus, alphavantage_request, getenv, parse_float


FUNDAMENTAL_ENDPOINTS: tuple[str, ...] = (
    "OVERVIEW",
    "INCOME_STATEMENT",
    "BALANCE_SHEET",
    "CASH_FLOW",
    "EARNINGS",
)


def analyze_company_fundamentals(stock: StockInput, options: AnalysisOptions) -> dict[str, Any]:
    provider_symbol = stock.fundamental_symbol or stock.nse_symbol or stock.symbol
    api_key = getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {
            "generated_at": datetime.now(IST).isoformat(),
            "stock": stock.model_dump(),
            "provider_symbol": provider_symbol,
            "provider_status": [
                ProviderStatus(
                    provider="alphavantage_fundamentals",
                    ok=False,
                    item_count=0,
                    warning="ALPHAVANTAGE_API_KEY not set",
                    skipped=True,
                ).__dict__
            ],
            "report": {
                "label": "unavailable",
                "overall_score": None,
                "confidence": 0.0,
                "summary": "Financial report data is unavailable because ALPHAVANTAGE_API_KEY is not set.",
                "insights": [],
                "risks": [],
            },
        }

    payloads, statuses = fetch_fundamental_bundle(
        symbol=provider_symbol,
        api_key=api_key,
        timeout_seconds=options.timeout_seconds,
    )
    overview = payloads.get("OVERVIEW", {})
    income = payloads.get("INCOME_STATEMENT", {})
    balance = payloads.get("BALANCE_SHEET", {})
    cash_flow = payloads.get("CASH_FLOW", {})
    earnings = payloads.get("EARNINGS", {})

    latest_income = first_report(income.get("annualReports"))
    previous_income = nth_report(income.get("annualReports"), 1)
    older_income = nth_report(income.get("annualReports"), 3)

    latest_balance = first_report(balance.get("annualReports"))
    latest_cash_flow = first_report(cash_flow.get("annualReports"))
    previous_cash_flow = nth_report(cash_flow.get("annualReports"), 1)

    latest_quarterly_earnings = first_report(earnings.get("quarterlyEarnings"))
    previous_annual_earnings = nth_report(earnings.get("annualEarnings"), 1)
    latest_annual_earnings = first_report(earnings.get("annualEarnings"))

    valuation = build_valuation_section(overview)
    profitability = build_profitability_section(overview, latest_income)
    growth = build_growth_section(overview, latest_income, previous_income, older_income, latest_annual_earnings, previous_annual_earnings)
    financial_strength = build_financial_strength_section(overview, latest_balance)
    cash_flow_summary = build_cash_flow_section(latest_cash_flow, previous_cash_flow)
    earnings_quality = build_earnings_quality_section(latest_quarterly_earnings, latest_annual_earnings, previous_annual_earnings)

    report = build_long_term_report(
        stock=stock,
        valuation=valuation,
        profitability=profitability,
        growth=growth,
        financial_strength=financial_strength,
        cash_flow_summary=cash_flow_summary,
        earnings_quality=earnings_quality,
        provider_statuses=statuses,
    )

    return {
        "generated_at": datetime.now(IST).isoformat(),
        "stock": stock.model_dump(),
        "provider_symbol": provider_symbol,
        "provider_status": [status.__dict__ for status in statuses],
        "valuation": valuation,
        "profitability": profitability,
        "growth": growth,
        "financial_strength": financial_strength,
        "cash_flow": cash_flow_summary,
        "earnings_quality": earnings_quality,
        "raw_financials": {
            "overview": prune_overview_payload(overview),
            "latest_annual_income_statement": summarize_income_report(latest_income),
            "latest_annual_balance_sheet": summarize_balance_sheet(latest_balance),
            "latest_annual_cash_flow": summarize_cash_flow_report(latest_cash_flow),
            "latest_quarterly_earnings": summarize_earnings_report(latest_quarterly_earnings),
        },
        "report": report,
    }


def fetch_fundamental_bundle(
    *,
    symbol: str,
    api_key: str,
    timeout_seconds: float,
) -> tuple[dict[str, dict[str, Any]], list[ProviderStatus]]:
    payloads: dict[str, dict[str, Any]] = {}
    statuses: list[ProviderStatus] = []
    for function_name in FUNDAMENTAL_ENDPOINTS:
        payload = alphavantage_request(
            {
                "function": function_name,
                "symbol": symbol,
                "apikey": api_key,
            },
            timeout_seconds=timeout_seconds,
        )
        payloads[function_name] = payload
        statuses.append(provider_status_for_payload(function_name, payload))
    return payloads, statuses


def provider_status_for_payload(function_name: str, payload: dict[str, Any]) -> ProviderStatus:
    if not payload:
        return ProviderStatus(
            provider=f"alphavantage_{function_name.lower()}",
            ok=False,
            item_count=0,
            warning="request failed or returned no payload",
        )
    warning = first_text(payload.get("Note"), payload.get("Information"), payload.get("Error Message"))
    if warning:
        return ProviderStatus(
            provider=f"alphavantage_{function_name.lower()}",
            ok=False,
            item_count=0,
            warning=warning,
        )
    item_count = estimate_payload_items(function_name, payload)
    return ProviderStatus(
        provider=f"alphavantage_{function_name.lower()}",
        ok=item_count > 0,
        item_count=item_count,
        warning=None if item_count > 0 else "payload did not contain expected report data",
    )


def estimate_payload_items(function_name: str, payload: dict[str, Any]) -> int:
    if function_name == "OVERVIEW":
        return 1 if payload.get("Symbol") or payload.get("Name") else 0
    if function_name == "EARNINGS":
        return len(as_report_list(payload.get("annualEarnings"))) + len(as_report_list(payload.get("quarterlyEarnings")))
    return len(as_report_list(payload.get("annualReports"))) + len(as_report_list(payload.get("quarterlyReports")))


def build_valuation_section(overview: dict[str, Any]) -> dict[str, Any]:
    pe_ratio = first_number(overview.get("PERatio"), overview.get("TrailingPE"))
    peg_ratio = parse_float(overview.get("PEGRatio"))
    price_to_book = parse_float(overview.get("PriceToBookRatio"))
    price_to_sales = parse_float(overview.get("PriceToSalesRatioTTM"))
    ev_to_revenue = parse_float(overview.get("EVToRevenue"))
    ev_to_ebitda = parse_float(overview.get("EVToEBITDA"))
    dividend_yield_pct = scaled_percent(overview.get("DividendYield"))
    market_cap = parse_float(overview.get("MarketCapitalization"))

    stance = "unknown"
    if pe_ratio is not None:
        if pe_ratio <= 15:
            stance = "attractive"
        elif pe_ratio <= 25:
            stance = "reasonable"
        elif pe_ratio <= 35:
            stance = "full"
        else:
            stance = "expensive"

    return {
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "peg_ratio": peg_ratio,
        "price_to_book": price_to_book,
        "price_to_sales_ttm": price_to_sales,
        "ev_to_revenue": ev_to_revenue,
        "ev_to_ebitda": ev_to_ebitda,
        "dividend_yield_pct": dividend_yield_pct,
        "analyst_target_price": parse_float(overview.get("AnalystTargetPrice")),
        "stance": stance,
    }


def build_profitability_section(overview: dict[str, Any], latest_income: dict[str, Any]) -> dict[str, Any]:
    return {
        "revenue_ttm": parse_float(overview.get("RevenueTTM")),
        "gross_profit_ttm": parse_float(overview.get("GrossProfitTTM")),
        "ebitda": parse_float(overview.get("EBITDA")),
        "diluted_eps_ttm": parse_float(overview.get("DilutedEPSTTM")),
        "profit_margin_pct": scaled_percent(overview.get("ProfitMargin")),
        "operating_margin_pct": scaled_percent(overview.get("OperatingMarginTTM")),
        "return_on_assets_pct": scaled_percent(overview.get("ReturnOnAssetsTTM")),
        "return_on_equity_pct": scaled_percent(overview.get("ReturnOnEquityTTM")),
        "latest_annual_revenue": report_number(latest_income, "totalRevenue"),
        "latest_annual_operating_income": report_number(latest_income, "operatingIncome"),
        "latest_annual_net_income": report_number(latest_income, "netIncome"),
    }


def build_growth_section(
    overview: dict[str, Any],
    latest_income: dict[str, Any],
    previous_income: dict[str, Any],
    older_income: dict[str, Any],
    latest_annual_earnings: dict[str, Any],
    previous_annual_earnings: dict[str, Any],
) -> dict[str, Any]:
    latest_revenue = report_number(latest_income, "totalRevenue")
    previous_revenue = report_number(previous_income, "totalRevenue")
    older_revenue = report_number(older_income, "totalRevenue")
    latest_net_income = report_number(latest_income, "netIncome")
    previous_net_income = report_number(previous_income, "netIncome")
    latest_eps = first_number(latest_annual_earnings.get("reportedEPS"))
    previous_eps = first_number(previous_annual_earnings.get("reportedEPS"))

    return {
        "quarterly_revenue_growth_yoy_pct": scaled_percent(overview.get("QuarterlyRevenueGrowthYOY")),
        "quarterly_earnings_growth_yoy_pct": scaled_percent(overview.get("QuarterlyEarningsGrowthYOY")),
        "annual_revenue_growth_pct": growth_percent(latest_revenue, previous_revenue),
        "annual_net_income_growth_pct": growth_percent(latest_net_income, previous_net_income),
        "annual_eps_growth_pct": growth_percent(latest_eps, previous_eps),
        "revenue_cagr_3y_pct": cagr_percent(latest_revenue, older_revenue, 3),
    }


def build_financial_strength_section(overview: dict[str, Any], latest_balance: dict[str, Any]) -> dict[str, Any]:
    current_assets = report_number(latest_balance, "totalCurrentAssets")
    current_liabilities = report_number(latest_balance, "totalCurrentLiabilities")
    total_liabilities = report_number(latest_balance, "totalLiabilities")
    shareholder_equity = report_number(latest_balance, "totalShareholderEquity")
    debt = first_number(
        latest_balance.get("shortLongTermDebtTotal"),
        latest_balance.get("longTermDebt"),
        latest_balance.get("shortTermDebt"),
        latest_balance.get("currentDebt"),
    )
    debt_to_equity = parse_float(overview.get("DebtToEquity"))
    if debt_to_equity is None:
        debt_to_equity = safe_ratio(debt, shareholder_equity)
    if debt_to_equity is None:
        debt_to_equity = safe_ratio(total_liabilities, shareholder_equity)

    return {
        "current_ratio": safe_ratio(current_assets, current_liabilities),
        "cash_and_equivalents": report_number(latest_balance, "cashAndCashEquivalentsAtCarryingValue"),
        "debt": debt,
        "total_liabilities": total_liabilities,
        "shareholder_equity": shareholder_equity,
        "book_value_per_share": parse_float(overview.get("BookValue")),
        "debt_to_equity": debt_to_equity,
    }


def build_cash_flow_section(latest_cash_flow: dict[str, Any], previous_cash_flow: dict[str, Any]) -> dict[str, Any]:
    operating_cash_flow = report_number(latest_cash_flow, "operatingCashflow")
    previous_operating_cash_flow = report_number(previous_cash_flow, "operatingCashflow")
    capital_expenditure = report_number(latest_cash_flow, "capitalExpenditures")
    previous_capital_expenditure = report_number(previous_cash_flow, "capitalExpenditures")
    free_cash_flow = free_cash_flow_value(operating_cash_flow, capital_expenditure)
    previous_free_cash_flow = free_cash_flow_value(previous_operating_cash_flow, previous_capital_expenditure)

    return {
        "operating_cash_flow": operating_cash_flow,
        "capital_expenditures": capital_expenditure,
        "free_cash_flow": free_cash_flow,
        "operating_cash_flow_growth_pct": growth_percent(operating_cash_flow, previous_operating_cash_flow),
        "free_cash_flow_growth_pct": growth_percent(free_cash_flow, previous_free_cash_flow),
    }


def build_earnings_quality_section(
    latest_quarterly_earnings: dict[str, Any],
    latest_annual_earnings: dict[str, Any],
    previous_annual_earnings: dict[str, Any],
) -> dict[str, Any]:
    reported_eps = first_number(latest_quarterly_earnings.get("reportedEPS"))
    estimated_eps = first_number(latest_quarterly_earnings.get("estimatedEPS"))
    surprise_pct = first_number(latest_quarterly_earnings.get("surprisePercentage"))
    annual_eps = first_number(latest_annual_earnings.get("reportedEPS"))
    previous_annual_eps = first_number(previous_annual_earnings.get("reportedEPS"))

    return {
        "latest_quarter_end": first_text(latest_quarterly_earnings.get("fiscalDateEnding")),
        "latest_reported_eps": reported_eps,
        "latest_estimated_eps": estimated_eps,
        "latest_earnings_surprise_pct": surprise_pct,
        "annual_eps": annual_eps,
        "annual_eps_growth_pct": growth_percent(annual_eps, previous_annual_eps),
    }


def build_long_term_report(
    *,
    stock: StockInput,
    valuation: dict[str, Any],
    profitability: dict[str, Any],
    growth: dict[str, Any],
    financial_strength: dict[str, Any],
    cash_flow_summary: dict[str, Any],
    earnings_quality: dict[str, Any],
    provider_statuses: Sequence[ProviderStatus],
) -> dict[str, Any]:
    score = 50.0
    insights: list[str] = []
    risks: list[str] = []
    metrics_seen = 0

    pe_ratio = as_float(valuation.get("pe_ratio"))
    if pe_ratio is not None:
        metrics_seen += 1
        if pe_ratio <= 15:
            score += 8
            insights.append(f"P/E of {pe_ratio:.2f} suggests the stock is not richly priced.")
        elif pe_ratio <= 25:
            score += 4
            insights.append(f"P/E of {pe_ratio:.2f} looks reasonable for a long-term holder.")
        elif pe_ratio > 35:
            score -= 6
            risks.append(f"P/E of {pe_ratio:.2f} implies the valuation already prices in strong expectations.")

    peg_ratio = as_float(valuation.get("peg_ratio"))
    if peg_ratio is not None:
        metrics_seen += 1
        if 0 < peg_ratio <= 1.2:
            score += 5
            insights.append(f"PEG ratio of {peg_ratio:.2f} looks supportive relative to growth.")
        elif peg_ratio > 2.0:
            score -= 4
            risks.append(f"PEG ratio of {peg_ratio:.2f} suggests growth may not fully justify the valuation.")

    roe = as_float(profitability.get("return_on_equity_pct"))
    if roe is not None:
        metrics_seen += 1
        if roe >= 15:
            score += 8
            insights.append(f"ROE of {roe:.1f}% indicates strong capital efficiency.")
        elif roe < 8:
            score -= 5
            risks.append(f"ROE of {roe:.1f}% is weak for a durable long-term compounder.")

    roa = as_float(profitability.get("return_on_assets_pct"))
    if roa is not None:
        metrics_seen += 1
        if roa >= 5:
            score += 4
        elif roa < 2:
            score -= 3
            risks.append(f"ROA of {roa:.1f}% suggests the asset base is not generating strong returns.")

    operating_margin = as_float(profitability.get("operating_margin_pct"))
    if operating_margin is not None:
        metrics_seen += 1
        if operating_margin >= 15:
            score += 5
            insights.append(f"Operating margin of {operating_margin:.1f}% points to solid business quality.")
        elif operating_margin < 5:
            score -= 4
            risks.append(f"Operating margin of {operating_margin:.1f}% leaves limited room for execution mistakes.")

    profit_margin = as_float(profitability.get("profit_margin_pct"))
    if profit_margin is not None:
        metrics_seen += 1
        if profit_margin >= 10:
            score += 4
        elif profit_margin < 3:
            score -= 3
            risks.append(f"Net profit margin of {profit_margin:.1f}% is thin for long-term resilience.")

    revenue_growth = as_float(growth.get("annual_revenue_growth_pct"))
    if revenue_growth is not None:
        metrics_seen += 1
        if revenue_growth >= 10:
            score += 6
            insights.append(f"Annual revenue grew {revenue_growth:.1f}% in the latest fiscal year.")
        elif revenue_growth < 0:
            score -= 6
            risks.append(f"Annual revenue declined {abs(revenue_growth):.1f}% in the latest fiscal year.")

    earnings_growth = as_float(growth.get("annual_net_income_growth_pct"))
    if earnings_growth is not None:
        metrics_seen += 1
        if earnings_growth >= 10:
            score += 6
            insights.append(f"Net income growth of {earnings_growth:.1f}% supports long-term earnings power.")
        elif earnings_growth < 0:
            score -= 6
            risks.append(f"Net income fell {abs(earnings_growth):.1f}% year over year.")

    quarterly_growth = as_float(growth.get("quarterly_revenue_growth_yoy_pct"))
    if quarterly_growth is not None:
        metrics_seen += 1
        if quarterly_growth >= 10:
            score += 4
        elif quarterly_growth < 0:
            score -= 4
            risks.append(f"Quarterly revenue growth of {quarterly_growth:.1f}% shows current momentum is weak.")

    current_ratio = as_float(financial_strength.get("current_ratio"))
    if current_ratio is not None:
        metrics_seen += 1
        if current_ratio >= 1.5:
            score += 4
            insights.append(f"Current ratio of {current_ratio:.2f} indicates comfortable short-term liquidity.")
        elif current_ratio < 1.0:
            score -= 5
            risks.append(f"Current ratio of {current_ratio:.2f} points to tight short-term liquidity.")

    debt_to_equity = as_float(financial_strength.get("debt_to_equity"))
    if debt_to_equity is not None:
        metrics_seen += 1
        if debt_to_equity <= 0.7:
            score += 6
            insights.append(f"Debt-to-equity of {debt_to_equity:.2f} suggests a conservative balance sheet.")
        elif debt_to_equity > 2.0:
            score -= 7
            risks.append(f"Debt-to-equity of {debt_to_equity:.2f} indicates elevated leverage risk.")

    free_cash_flow = as_float(cash_flow_summary.get("free_cash_flow"))
    if free_cash_flow is not None:
        metrics_seen += 1
        if free_cash_flow > 0:
            score += 6
            insights.append("Free cash flow is positive, which supports reinvestment capacity and downside protection.")
        elif free_cash_flow < 0:
            score -= 7
            risks.append("Free cash flow is negative, which weakens long-term valuation support.")

    free_cash_flow_growth = as_float(cash_flow_summary.get("free_cash_flow_growth_pct"))
    if free_cash_flow_growth is not None:
        metrics_seen += 1
        if free_cash_flow_growth >= 10:
            score += 4
        elif free_cash_flow_growth < 0:
            score -= 3
            risks.append(f"Free cash flow contracted {abs(free_cash_flow_growth):.1f}% year over year.")

    earnings_surprise = as_float(earnings_quality.get("latest_earnings_surprise_pct"))
    if earnings_surprise is not None:
        metrics_seen += 1
        if earnings_surprise > 0:
            score += 2
        elif earnings_surprise < 0:
            score -= 2
            risks.append(f"The latest quarterly earnings surprise was {earnings_surprise:.1f}%, below expectations.")

    provider_successes = sum(1 for status in provider_statuses if status.ok)
    confidence = 0.2 + min(0.55, metrics_seen / 16.0) + min(0.25, provider_successes / len(provider_statuses or [1]))
    overall_score = round(max(0.0, min(100.0, score)), 1)

    label = "mixed"
    if overall_score >= 75:
        label = "strong"
    elif overall_score >= 62:
        label = "constructive"
    elif overall_score < 35:
        label = "weak"
    elif overall_score < 48:
        label = "cautious"

    if not insights and not risks:
        summary = (
            f"{stock.name} has insufficient fundamental coverage to form a strong long-term view. "
            "Check the provider symbol and API availability."
        )
    else:
        positive_phrase = insights[0] if insights else "The available data does not show a standout fundamental strength."
        risk_phrase = risks[0] if risks else "No major long-term balance-sheet or valuation warning was detected in the available fields."
        summary = f"{stock.name} screens {label} for long-term investors. {positive_phrase} {risk_phrase}"

    return {
        "label": label,
        "overall_score": overall_score,
        "confidence": round(min(1.0, confidence), 3),
        "summary": summary,
        "insights": insights[:8],
        "risks": risks[:8],
        "long_term_drivers": build_long_term_drivers(valuation, profitability, growth, financial_strength, cash_flow_summary),
    }


def build_long_term_drivers(
    valuation: dict[str, Any],
    profitability: dict[str, Any],
    growth: dict[str, Any],
    financial_strength: dict[str, Any],
    cash_flow_summary: dict[str, Any],
) -> list[str]:
    drivers: list[str] = []
    pe_ratio = as_float(valuation.get("pe_ratio"))
    if pe_ratio is not None:
        drivers.append(f"P/E: {pe_ratio:.2f}")
    roe = as_float(profitability.get("return_on_equity_pct"))
    if roe is not None:
        drivers.append(f"ROE: {roe:.1f}%")
    revenue_growth = as_float(growth.get("annual_revenue_growth_pct"))
    if revenue_growth is not None:
        drivers.append(f"Annual revenue growth: {revenue_growth:.1f}%")
    debt_to_equity = as_float(financial_strength.get("debt_to_equity"))
    if debt_to_equity is not None:
        drivers.append(f"Debt-to-equity: {debt_to_equity:.2f}")
    free_cash_flow = as_float(cash_flow_summary.get("free_cash_flow"))
    if free_cash_flow is not None:
        drivers.append(f"Free cash flow: {free_cash_flow:.0f}")
    return drivers[:8]


def summarize_income_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "fiscal_date_ending": first_text(report.get("fiscalDateEnding")),
        "reported_currency": first_text(report.get("reportedCurrency")),
        "total_revenue": report_number(report, "totalRevenue"),
        "gross_profit": report_number(report, "grossProfit"),
        "operating_income": report_number(report, "operatingIncome"),
        "net_income": report_number(report, "netIncome"),
    }


def summarize_balance_sheet(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "fiscal_date_ending": first_text(report.get("fiscalDateEnding")),
        "reported_currency": first_text(report.get("reportedCurrency")),
        "cash_and_equivalents": report_number(report, "cashAndCashEquivalentsAtCarryingValue"),
        "current_assets": report_number(report, "totalCurrentAssets"),
        "current_liabilities": report_number(report, "totalCurrentLiabilities"),
        "total_liabilities": report_number(report, "totalLiabilities"),
        "shareholder_equity": report_number(report, "totalShareholderEquity"),
        "debt": first_number(
            report.get("shortLongTermDebtTotal"),
            report.get("longTermDebt"),
            report.get("shortTermDebt"),
            report.get("currentDebt"),
        ),
    }


def summarize_cash_flow_report(report: dict[str, Any]) -> dict[str, Any]:
    operating_cash_flow = report_number(report, "operatingCashflow")
    capital_expenditures = report_number(report, "capitalExpenditures")
    return {
        "fiscal_date_ending": first_text(report.get("fiscalDateEnding")),
        "reported_currency": first_text(report.get("reportedCurrency")),
        "operating_cash_flow": operating_cash_flow,
        "capital_expenditures": capital_expenditures,
        "free_cash_flow": free_cash_flow_value(operating_cash_flow, capital_expenditures),
    }


def summarize_earnings_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "fiscal_date_ending": first_text(report.get("fiscalDateEnding")),
        "reported_date": first_text(report.get("reportedDate")),
        "reported_eps": first_number(report.get("reportedEPS")),
        "estimated_eps": first_number(report.get("estimatedEPS")),
        "surprise_percentage": first_number(report.get("surprisePercentage")),
    }


def prune_overview_payload(overview: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(overview, dict):
        return {}
    keys = (
        "Symbol",
        "Name",
        "Description",
        "Exchange",
        "Currency",
        "Country",
        "Sector",
        "Industry",
        "MarketCapitalization",
        "PERatio",
        "PEGRatio",
        "PriceToBookRatio",
        "PriceToSalesRatioTTM",
        "EVToRevenue",
        "EVToEBITDA",
        "EPS",
        "DividendYield",
        "ProfitMargin",
        "OperatingMarginTTM",
        "ReturnOnAssetsTTM",
        "ReturnOnEquityTTM",
        "QuarterlyRevenueGrowthYOY",
        "QuarterlyEarningsGrowthYOY",
        "AnalystTargetPrice",
    )
    return {key: overview.get(key) for key in keys if overview.get(key) not in (None, "")}


def as_report_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    reports = [row for row in value if isinstance(row, dict)]
    reports.sort(key=lambda row: str(row.get("fiscalDateEnding", "")), reverse=True)
    return reports


def first_report(value: Any) -> dict[str, Any]:
    reports = as_report_list(value)
    return reports[0] if reports else {}


def nth_report(value: Any, index: int) -> dict[str, Any]:
    reports = as_report_list(value)
    return reports[index] if index < len(reports) else {}


def first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def first_number(*values: Any) -> float | None:
    for value in values:
        parsed = parse_float(value)
        if parsed is not None:
            return parsed
    return None


def report_number(report: dict[str, Any], field: str) -> float | None:
    if not isinstance(report, dict):
        return None
    return parse_float(report.get(field))


def scaled_percent(value: Any) -> float | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return round(parsed * 100.0, 2)


def growth_percent(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0.0):
        return None
    return round(((current - previous) / abs(previous)) * 100.0, 2)


def cagr_percent(current: float | None, starting: float | None, periods: int) -> float | None:
    if current is None or starting in (None, 0.0) or periods <= 0 or current <= 0 or starting <= 0:
        return None
    return round((((current / starting) ** (1 / periods)) - 1) * 100.0, 2)


def free_cash_flow_value(operating_cash_flow: float | None, capital_expenditures: float | None) -> float | None:
    if operating_cash_flow is None:
        return None
    if capital_expenditures is None:
        return operating_cash_flow
    return operating_cash_flow - abs(capital_expenditures)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return round(numerator / denominator, 3)


def as_float(value: Any) -> float | None:
    return value if isinstance(value, float) else parse_float(value)
