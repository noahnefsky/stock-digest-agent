"""
Pydantic models for the stock digest agent
"""

from datetime import datetime
from typing import Dict, List
from typing import Optional as OptionalType
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class StockFinanceData(BaseModel):
    ticker: str
    current_price: float
    previous_close: float
    change_percent: float
    volume: int
    market_cap: OptionalType[float] = None
    pe_ratio: OptionalType[float] = None
    company_name: str
    beta: OptionalType[float] = None


class StockResearch(BaseModel):
    ticker: str
    news_summary: str
    key_developments: List[str] = Field(default_factory=list)
    analyst_sentiment: str
    risk_factors: List[str] = Field(default_factory=list)
    price_targets: OptionalType[str] = None
    sources: List[Dict] = Field(default_factory=list)


class TargetedResearch(BaseModel):
    ticker: str
    earnings_news: List[Dict] = Field(default_factory=list)
    analyst_ratings: List[Dict] = Field(default_factory=list)
    insider_trading: List[Dict] = Field(default_factory=list)
    technical_analysis: List[Dict] = Field(default_factory=list)
    sector_news: List[Dict] = Field(default_factory=list)


class StockReport(BaseModel):
    ticker: str
    company_name: str
    summary: OptionalType[str] = None
    current_performance: str
    key_insights: List[str] = Field(default_factory=list)
    recommendation: str
    risk_assessment: str
    price_outlook: str
    sources: List[Dict] = Field(default_factory=list)
    finance_data: OptionalType[StockFinanceData] = None


class StockDigestOutput(BaseModel):
    reports: Dict[str, StockReport] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    market_overview: str


class PDFData(BaseModel):
    pdf_base64: str
    filename: str


class State(TypedDict):
    tickers: List[str]
    finance_data: Dict[str, StockFinanceData]
    research_data: Dict[str, StockResearch]
    targeted_research: Dict[str, TargetedResearch]
    all_news_stories: List[tuple]
    structured_reports: StockDigestOutput
    pdf_data: OptionalType[PDFData]
    date: str 