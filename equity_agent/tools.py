"""
Equity research tools for fetching stock data, financials, and market info.
"""

import json
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf
import pandas as pd


def get_stock_price(ticker: str, period: str = "1mo") -> dict[str, Any]:
    """
    Fetch current stock price and recent price history.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT)
        period: Time period for history. Valid values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        hist = stock.history(period=period)
        if hist.empty:
            return {"error": f"No price data found for {ticker}"}

        current_price = hist["Close"].iloc[-1]
        prev_price = hist["Close"].iloc[-2] if len(hist) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        # Build price summary
        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "current_price": round(float(current_price), 2),
            "price_change": round(float(price_change), 2),
            "price_change_pct": round(float(price_change_pct), 2),
            "currency": info.get("currency", "USD"),
            "period": period,
            "period_high": round(float(hist["High"].max()), 2),
            "period_low": round(float(hist["Low"].min()), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "market_cap": info.get("marketCap"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
        }

        # Add recent price history (last 10 data points)
        recent = hist.tail(10)[["Close", "Volume"]].copy()
        recent.index = recent.index.strftime("%Y-%m-%d")
        result["recent_prices"] = {
            date: {"close": round(float(row["Close"]), 2), "volume": int(row["Volume"])}
            for date, row in recent.iterrows()
        }

        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def get_financial_statements(ticker: str) -> dict[str, Any]:
    """
    Fetch income statement, balance sheet, and cash flow statement.

    Args:
        ticker: Stock ticker symbol
    """
    try:
        stock = yf.Ticker(ticker)

        result = {"ticker": ticker.upper()}

        # Income statement (annual)
        income_stmt = stock.income_stmt
        if not income_stmt.empty:
            latest = income_stmt.iloc[:, 0]
            result["income_statement"] = {
                "period": str(income_stmt.columns[0].date()),
                "total_revenue": _safe_value(latest.get("Total Revenue")),
                "gross_profit": _safe_value(latest.get("Gross Profit")),
                "operating_income": _safe_value(latest.get("Operating Income")),
                "net_income": _safe_value(latest.get("Net Income")),
                "ebitda": _safe_value(latest.get("EBITDA")),
                "eps_diluted": _safe_value(latest.get("Diluted EPS")),
            }
            # Revenue trend (last 4 years)
            if "Total Revenue" in income_stmt.index:
                rev_row = income_stmt.loc["Total Revenue"]
                result["revenue_trend"] = {
                    str(col.date()): _safe_value(val)
                    for col, val in zip(income_stmt.columns[:4], rev_row.iloc[:4])
                }

        # Balance sheet
        balance_sheet = stock.balance_sheet
        if not balance_sheet.empty:
            latest = balance_sheet.iloc[:, 0]
            result["balance_sheet"] = {
                "period": str(balance_sheet.columns[0].date()),
                "total_assets": _safe_value(latest.get("Total Assets")),
                "total_liabilities": _safe_value(latest.get("Total Liabilities Net Minority Interest")),
                "stockholders_equity": _safe_value(latest.get("Stockholders Equity")),
                "cash_and_equivalents": _safe_value(latest.get("Cash And Cash Equivalents")),
                "total_debt": _safe_value(latest.get("Total Debt")),
                "net_debt": _safe_value(latest.get("Net Debt")),
            }

        # Cash flow
        cashflow = stock.cashflow
        if not cashflow.empty:
            latest = cashflow.iloc[:, 0]
            result["cash_flow"] = {
                "period": str(cashflow.columns[0].date()),
                "operating_cash_flow": _safe_value(latest.get("Operating Cash Flow")),
                "capital_expenditures": _safe_value(latest.get("Capital Expenditure")),
                "free_cash_flow": _safe_value(latest.get("Free Cash Flow")),
                "dividends_paid": _safe_value(latest.get("Common Stock Dividend Paid")),
            }

        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def get_key_metrics(ticker: str) -> dict[str, Any]:
    """
    Fetch key valuation and fundamental metrics for a stock.

    Args:
        ticker: Stock ticker symbol
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "valuation": {
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
            },
            "profitability": {
                "gross_margin": _pct(info.get("grossMargins")),
                "operating_margin": _pct(info.get("operatingMargins")),
                "net_margin": _pct(info.get("profitMargins")),
                "return_on_equity": _pct(info.get("returnOnEquity")),
                "return_on_assets": _pct(info.get("returnOnAssets")),
            },
            "growth": {
                "revenue_growth_yoy": _pct(info.get("revenueGrowth")),
                "earnings_growth_yoy": _pct(info.get("earningsGrowth")),
                "earnings_quarterly_growth": _pct(info.get("earningsQuarterlyGrowth")),
            },
            "dividends": {
                "dividend_yield": _pct(info.get("dividendYield")),
                "dividend_rate": info.get("dividendRate"),
                "payout_ratio": _pct(info.get("payoutRatio")),
                "ex_dividend_date": _format_timestamp(info.get("exDividendDate")),
            },
            "financial_health": {
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "debt_to_equity": info.get("debtToEquity"),
                "total_cash_per_share": info.get("totalCashPerShare"),
            },
            "analyst_coverage": {
                "recommendation": info.get("recommendationKey"),
                "target_price": info.get("targetMeanPrice"),
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "number_of_analysts": info.get("numberOfAnalystOpinions"),
            },
            "short_interest": {
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": _pct(info.get("shortPercentOfFloat")),
            },
        }

        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def get_analyst_estimates(ticker: str) -> dict[str, Any]:
    """
    Fetch analyst EPS and revenue estimates (forward-looking).

    Args:
        ticker: Stock ticker symbol
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
        }

        # Earnings estimates
        try:
            earnings_est = stock.earnings_estimate
            if earnings_est is not None and not earnings_est.empty:
                result["earnings_estimates"] = earnings_est.to_dict()
        except Exception:
            pass

        # Revenue estimates
        try:
            rev_est = stock.revenue_estimate
            if rev_est is not None and not rev_est.empty:
                result["revenue_estimates"] = rev_est.to_dict()
        except Exception:
            pass

        # Earnings history
        try:
            earnings_hist = stock.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                recent = earnings_hist.tail(8)
                result["earnings_history"] = recent.to_dict(orient="records")
        except Exception:
            pass

        # Upgrade/downgrade history
        try:
            upgrades = stock.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                recent_upgrades = upgrades.head(10)
                result["recent_analyst_actions"] = recent_upgrades.to_dict(orient="records")
        except Exception:
            pass

        # EPS trend
        try:
            eps_trend = stock.eps_trend
            if eps_trend is not None and not eps_trend.empty:
                result["eps_trend"] = eps_trend.to_dict()
        except Exception:
            pass

        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def get_company_news(ticker: str, max_articles: int = 10) -> dict[str, Any]:
    """
    Fetch recent news articles about a company.

    Args:
        ticker: Stock ticker symbol
        max_articles: Maximum number of articles to return (default 10)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        news = stock.news

        if not news:
            return {"ticker": ticker.upper(), "articles": [], "message": "No recent news found"}

        articles = []
        for item in news[:max_articles]:
            articles.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime(
                    "%Y-%m-%d %H:%M"
                ) if item.get("providerPublishTime") else "",
                "url": item.get("link", ""),
                "summary": item.get("summary", ""),
            })

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "article_count": len(articles),
            "articles": articles,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def compare_stocks(tickers: list[str]) -> dict[str, Any]:
    """
    Compare key metrics across multiple stocks side by side.

    Args:
        tickers: List of stock ticker symbols to compare (max 5)
    """
    if len(tickers) > 5:
        tickers = tickers[:5]

    results = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None

            results[ticker.upper()] = {
                "company_name": info.get("longName", ticker),
                "current_price": round(current_price, 2) if current_price else None,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "gross_margin": _pct(info.get("grossMargins")),
                "net_margin": _pct(info.get("profitMargins")),
                "return_on_equity": _pct(info.get("returnOnEquity")),
                "revenue_growth": _pct(info.get("revenueGrowth")),
                "debt_to_equity": info.get("debtToEquity"),
                "dividend_yield": _pct(info.get("dividendYield")),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "analyst_recommendation": info.get("recommendationKey"),
                "analyst_target": info.get("targetMeanPrice"),
            }
        except Exception as e:
            results[ticker.upper()] = {"error": str(e)}

    return {"comparison": results, "tickers": [t.upper() for t in tickers]}


def get_sector_info(ticker: str) -> dict[str, Any]:
    """
    Get sector and industry classification with peers.

    Args:
        ticker: Stock ticker symbol
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "business_summary": info.get("longBusinessSummary", ""),
            "website": info.get("website"),
            "headquarters": {
                "city": info.get("city"),
                "state": info.get("state"),
                "country": info.get("country"),
            },
            "key_executives": info.get("companyOfficers", [])[:5],
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# --- Helpers ---

def _safe_value(val: Any) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    return val


def _pct(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return f"{round(float(val) * 100, 2)}%"


def _format_timestamp(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except Exception:
        return None


# --- Tool definitions for Claude API ---

TOOL_DEFINITIONS = [
    {
        "name": "get_stock_price",
        "description": (
            "Fetch the current stock price and recent price history for a given ticker symbol. "
            "Returns current price, daily change, 52-week high/low, volume, and recent price history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL, TSLA)",
                },
                "period": {
                    "type": "string",
                    "description": "Time period for history. Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y. Default: 1mo",
                    "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_financial_statements",
        "description": (
            "Fetch annual financial statements for a stock: income statement (revenue, profit, EPS), "
            "balance sheet (assets, liabilities, equity, debt), and cash flow statement. "
            "Use this to analyze a company's financial performance and health."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_key_metrics",
        "description": (
            "Fetch key valuation ratios and fundamental metrics: P/E, P/B, EV/EBITDA, PEG, "
            "margins, return ratios, growth rates, dividend info, analyst price targets, "
            "and financial health metrics. Essential for stock valuation and screening."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_analyst_estimates",
        "description": (
            "Fetch analyst EPS and revenue estimates, earnings history (actual vs. estimated), "
            "recent analyst upgrades/downgrades, and EPS trend data. "
            "Use this to understand analyst sentiment and forward expectations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_company_news",
        "description": (
            "Fetch recent news articles about a company. "
            "Returns article titles, publishers, dates, and summaries. "
            "Use this for qualitative analysis and sentiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "max_articles": {
                    "type": "integer",
                    "description": "Maximum number of articles to return (default 10, max 20)",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compare_stocks",
        "description": (
            "Compare key valuation and fundamental metrics across multiple stocks side by side. "
            "Useful for peer comparison and relative valuation analysis. "
            "Provide up to 5 ticker symbols."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols to compare (2-5 tickers)",
                    "minItems": 2,
                    "maxItems": 5,
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_sector_info",
        "description": (
            "Get company overview including sector/industry classification, "
            "business description, headquarters, website, and key executives. "
            "Use this to understand what a company does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    },
]

TOOL_MAP = {
    "get_stock_price": get_stock_price,
    "get_financial_statements": get_financial_statements,
    "get_key_metrics": get_key_metrics,
    "get_analyst_estimates": get_analyst_estimates,
    "get_company_news": get_company_news,
    "compare_stocks": compare_stocks,
    "get_sector_info": get_sector_info,
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Execute a tool by name and return JSON result."""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOL_MAP[name](**tool_input)
        return json.dumps(result, default=str, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
