import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List
from typing import Optional as OptionalType
import time

from dotenv import load_dotenv
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import TypedDict
from polygon import RESTClient

from prompts import get_stock_analysis_prompt, get_market_overview_prompt

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for structured output
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


class StockReport(BaseModel):
    ticker: str
    company_name: str
    summary: str
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


class State(TypedDict):
    tickers: List[str]
    finance_data: Dict[str, StockFinanceData]
    research_data: Dict[str, StockResearch]
    all_news_stories: List[tuple]
    structured_reports: StockDigestOutput
    date: str


class StockDigestAgent:
    def __init__(self):
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.polygon_client = RESTClient("ERBRbQgWPfTPOXhnWPJbwrlCqb54Do9i")
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def polygon_finance_node(self, state: State) -> Dict:
        """Retrieve stock data from Polygon.io"""
        dispatch_custom_event("finance_status", "Fetching financial data from Polygon.io...")
        
        tickers = state["tickers"]
        finance_data = {}
        
        for ticker in tickers:
            dispatch_custom_event("finance_ticker", f"Processing {ticker}")
            
            # Get ticker details for company info
            details = self.polygon_client.get_ticker_details(ticker)
            
            # Get last two daily bars for price comparison
            to_date = datetime.now().date()
            from_date = to_date - timedelta(days=7)
            
            aggs = self.polygon_client.get_aggs(
                ticker, 1, "day", from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d")
            )
            
            # Initialize default values
            current_price = 0.0
            previous_close = 0.0
            change_percent = 0.0
            volume = 0
            market_cap = None
            pe_ratio = None
            company_name = ticker
            
            # Handle aggregates data
            if isinstance(aggs, list) and len(aggs) >= 2:
                latest_bar = aggs[-1]
                previous_bar = aggs[-2]
                
                close_price = getattr(latest_bar, 'close', None)
                if close_price is not None and isinstance(close_price, (int, float)):
                    current_price = float(close_price)
                
                prev_close = getattr(previous_bar, 'close', None)
                if prev_close is not None and isinstance(prev_close, (int, float)):
                    previous_close = float(prev_close)
                
                if previous_close > 0:
                    change_percent = ((current_price - previous_close) / previous_close) * 100
                
                volume_raw = getattr(latest_bar, 'volume', None)
                if volume_raw is not None and isinstance(volume_raw, (int, float)):
                    volume = int(volume_raw)
            else:
                # Fallback to previous close
                prev_day = self.polygon_client.get_previous_close_agg(ticker)
                if isinstance(prev_day, list) and len(prev_day) > 0:
                    prev_bar = prev_day[0]
                    
                    close_price = getattr(prev_bar, 'close', None)
                    if close_price is not None and isinstance(close_price, (int, float)):
                        current_price = float(close_price)
                        previous_close = float(close_price)
                    
                    volume_raw = getattr(prev_bar, 'volume', None)
                    if volume_raw is not None and isinstance(volume_raw, (int, float)):
                        volume = int(volume_raw)
            
            # Handle ticker details - use getattr to safely access attributes
            if hasattr(details, 'name') and getattr(details, 'name', None) is not None:
                company_name = str(getattr(details, 'name'))
            
            market_cap_raw = getattr(details, 'market_cap', None)
            if market_cap_raw is not None and isinstance(market_cap_raw, (int, float)):
                market_cap = float(market_cap_raw)
            
            pe_ratio_raw = getattr(details, 'pe_ratio', None)
            if pe_ratio_raw is not None and isinstance(pe_ratio_raw, (int, float)):
                pe_ratio = float(pe_ratio_raw)
            
            finance_data[ticker] = StockFinanceData(
                ticker=ticker,
                current_price=current_price,
                previous_close=previous_close,
                change_percent=change_percent,
                volume=volume,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                company_name=company_name,
                beta=None
            )
            
            logger.info(f"Retrieved data for {ticker} from Polygon.io")
            
            # Rate limit to stay within free tier limits (5 calls/min)
            time.sleep(15)
            
        return {"finance_data": finance_data}

    def research_node(self, state: State) -> Dict:
        """Research stocks using Tavily client"""
        dispatch_custom_event("research_status", "Researching stock news and analysis...")
        
        tickers = state["tickers"]
        research_data = {}
        all_news_stories = []
        
        for ticker in tickers:
            dispatch_custom_event("research_ticker", f"Researching {ticker}")
            
            search_query = f"{ticker} stock news analysis earnings financial performance market sentiment {self.current_date}"
            
            search_results = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                max_results=10,
                include_raw_content=True,
                include_answer=True,
                include_domains=["reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com", "yahoo.com", "seekingalpha.com"]
            )
            
            news_content = [result['content'] for result in search_results.get('results', []) if result.get('content')]
            news_stories = [{
                'title': r.get('title', ''), 'content': r.get('content', '')[:800],
                'url': r.get('url', ''), 'published_date': r.get('published_date', ''),
                'source': r.get('source', ''), 'score': r.get('score', 0),
                'domain': r.get('domain', '')
            } for r in search_results.get('results', [])]
            
            news_stories.sort(key=lambda x: (x.get('score', 0), x.get('published_date', '')), reverse=True)
            all_news_stories.extend([(ticker, story) for story in news_stories])
            
            combined_content = "\n\n".join(news_content[:5])
            
            research_data[ticker] = StockResearch(
                ticker=ticker,
                news_summary=combined_content[:2000],
                analyst_sentiment="Pending analysis"
            )
            logger.info(f"Research completed for {ticker} with {len(news_stories)} stories")
        
        return {"research_data": research_data, "all_news_stories": all_news_stories}

    def gemini_analysis_node(self, state: State) -> Dict:
        """Analyze data with Gemini and generate structured reports"""
        dispatch_custom_event("gemini_analysis_status", "Generating structured stock reports...")
        
        tickers = state["tickers"]
        research_data = state["research_data"]
        finance_data = state.get("finance_data", {})
        all_news_stories = state.get("all_news_stories", [])
        
        reports = {}
        for ticker in tickers:
            research = research_data.get(ticker, {})
            finance = finance_data.get(ticker)
            
            ticker_stories = [story for t, story in all_news_stories if t == ticker]
            
            structured_llm = self.gemini_llm.with_structured_output(StockReport)
            analysis_prompt = get_stock_analysis_prompt(ticker, research, ticker_stories, self.current_date)
            report = structured_llm.invoke(analysis_prompt)
            
            report_dict = dict(report)
            report_dict['sources'] = ticker_stories
            report_dict['finance_data'] = finance
            reports[ticker] = StockReport(**report_dict)
            
        market_overview_prompt = get_market_overview_prompt(tickers, all_news_stories, self.current_date)
        market_overview = self.gemini_llm.invoke(market_overview_prompt).content
        
        structured_reports = StockDigestOutput(
            reports=reports,
            market_overview=str(market_overview),
            generated_at=datetime.now().isoformat(),
        )
        
        dispatch_custom_event("analysis_complete", f"Generated reports for {len(reports)} stocks")
        return {"structured_reports": structured_reports}

    def build_graph(self):
        """Build and compile the stock digest graph"""
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("PolygonFinance", self.polygon_finance_node)
        graph_builder.add_node("Research", self.research_node)
        graph_builder.add_node("GeminiAnalysis", self.gemini_analysis_node)
        
        graph_builder.add_edge(START, "PolygonFinance")
        graph_builder.add_edge("PolygonFinance", "Research")
        graph_builder.add_edge("Research", "GeminiAnalysis")
        graph_builder.add_edge("GeminiAnalysis", END)
        
        return graph_builder.compile()

    async def run_digest(self, tickers: List[str]) -> StockDigestOutput:
        """Run the complete stock digest workflow"""
        logger.info(f"Starting stock digest for tickers: {tickers}")
        
        graph = self.build_graph()
        initial_state = {"tickers": tickers, "date": self.current_date}
        
        final_state = await graph.ainvoke(initial_state)
        return final_state["structured_reports"]