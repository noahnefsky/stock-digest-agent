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
from pdf_utils import generate_pdf
from models import (
    StockFinanceData, StockResearch, TargetedResearch, StockReport, 
    StockDigestOutput, PDFData, State
)

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDigestAgent:
    def __init__(self):
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.polygon_client = RESTClient(os.getenv("POLYGON_API_KEY"))
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
                
                current_price = float(getattr(latest_bar, 'close', 0) or 0)
                previous_close = float(getattr(previous_bar, 'close', 0) or 0)
                
                if previous_close > 0:
                    change_percent = ((current_price - previous_close) / previous_close) * 100
                
                volume = int(getattr(latest_bar, 'volume', 0) or 0)
            else:
                # Fallback to previous close
                prev_day = self.polygon_client.get_previous_close_agg(ticker)
                if isinstance(prev_day, list) and len(prev_day) > 0:
                    prev_bar = prev_day[0]
                    current_price = float(getattr(prev_bar, 'close', 0) or 0)
                    previous_close = current_price
                    volume = int(getattr(prev_bar, 'volume', 0) or 0)
            
            # Handle ticker details
            if hasattr(details, 'name') and getattr(details, 'name', None):
                company_name = str(getattr(details, 'name'))
            
            market_cap_raw = getattr(details, 'market_cap', None)
            market_cap = float(market_cap_raw) if market_cap_raw else None
            
            pe_ratio_raw = getattr(details, 'pe_ratio', None)
            pe_ratio = float(pe_ratio_raw) if pe_ratio_raw else None
            
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
            time.sleep(8)  # Rate limit for free tier
            
        return {"finance_data": finance_data}

    def targeted_research_node(self, state: State) -> Dict:
        """Perform targeted research for each ticker with one comprehensive search"""
        dispatch_custom_event("targeted_research_status", "Performing comprehensive research for each ticker...")
        
        tickers = state["tickers"]
        targeted_research = {}
        
        for ticker in tickers:
            dispatch_custom_event("targeted_research_ticker", f"Researching {ticker}")
            
            # One comprehensive search query covering all important categories
            search_query = f"{ticker} earnings analyst ratings insider trading technical analysis sector news {self.current_date}"
            
            search_results = self.tavily_client.search(
                query=search_query,
                search_depth="basic",
                max_results=10,
                include_raw_content=True,
                include_answer=True,
                include_domains=["reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com", "yahoo.com", "seekingalpha.com"]
            )
            
            stories = [{
                'title': r.get('title', ''),
                'content': r.get('content', '')[:400],
                'url': r.get('url', ''),
                'published_date': r.get('published_date', ''),
                'source': r.get('source', ''),
                'score': r.get('score', 0),
                'domain': r.get('domain', ''),
                'keyword': 'comprehensive'
            } for r in search_results.get('results', [])]
            
            # Categorize stories based on content keywords
            categorized_stories = {
                "earnings_news": [],
                "analyst_ratings": [],
                "insider_trading": [],
                "technical_analysis": [],
                "sector_news": []
            }
            
            earnings_keywords = ["earnings", "quarterly", "revenue", "profit", "guidance", "results"]
            analyst_keywords = ["analyst", "rating", "target", "upgrade", "downgrade", "recommendation"]
            insider_keywords = ["insider", "SEC", "filing", "executive", "Form 4"]
            technical_keywords = ["technical", "support", "resistance", "RSI", "MACD", "chart"]
            sector_keywords = ["sector", "industry", "competitor", "market share", "regulatory"]
            
            for story in stories:
                content_lower = story['content'].lower() + ' ' + story['title'].lower()
                
                # Categorize based on content
                if any(keyword in content_lower for keyword in earnings_keywords):
                    categorized_stories["earnings_news"].append(story)
                elif any(keyword in content_lower for keyword in analyst_keywords):
                    categorized_stories["analyst_ratings"].append(story)
                elif any(keyword in content_lower for keyword in insider_keywords):
                    categorized_stories["insider_trading"].append(story)
                elif any(keyword in content_lower for keyword in technical_keywords):
                    categorized_stories["technical_analysis"].append(story)
                else:
                    categorized_stories["sector_news"].append(story)
            
            ticker_research = {
                "ticker": ticker,
                **categorized_stories
            }
            
            targeted_research[ticker] = TargetedResearch(**ticker_research)
            logger.info(f"Comprehensive research completed for {ticker} with {len(stories)} stories")
            time.sleep(2)
        
        return {"targeted_research": targeted_research}

    def gemini_analysis_node(self, state: State) -> Dict:
        """Analyze data with Gemini and generate structured reports"""
        dispatch_custom_event("gemini_analysis_status", "Generating structured stock reports...")
        
        tickers = state["tickers"]
        targeted_research = state.get("targeted_research", {})
        finance_data = state.get("finance_data", {})
        
        # Collect all news stories from targeted research
        all_news_stories = []
        for ticker in tickers:
            if ticker in targeted_research:
                research = targeted_research[ticker]
                for category, stories in research.model_dump().items():
                    if category != 'ticker' and stories:
                        for story in stories:
                            story_with_ticker = story.copy()
                            story_with_ticker['ticker'] = ticker
                            all_news_stories.append((ticker, story_with_ticker))
        
        reports = {}
        for ticker in tickers:
            research = targeted_research.get(ticker, {})
            finance = finance_data.get(ticker)
            ticker_stories = [story for t, story in all_news_stories if t == ticker]
            
            structured_llm = self.gemini_llm.with_structured_output(StockReport)
            analysis_prompt = get_stock_analysis_prompt(ticker, research, ticker_stories, self.current_date)
            report = structured_llm.invoke(analysis_prompt)
            
            report_dict = dict(report)
            report_dict['sources'] = ticker_stories
            report_dict['finance_data'] = finance
            reports[ticker] = StockReport(**report_dict)
        
        # Generate market overview from all the targeted research data
        market_overview_prompt = get_market_overview_prompt(tickers, all_news_stories, self.current_date)
        market_overview = self.gemini_llm.invoke(market_overview_prompt).content
        
        structured_reports = StockDigestOutput(
            reports=reports,
            market_overview=str(market_overview),
            generated_at=datetime.now().isoformat(),
        )
        
        dispatch_custom_event("analysis_complete", f"Generated reports for {len(reports)} stocks")
        return {"structured_reports": structured_reports}

    def pdf_generation_node(self, state: State) -> Dict:
        """Generate PDF report from the structured data"""
        dispatch_custom_event("pdf_generation_status", "Generating PDF report...")
        
        structured_reports = state["structured_reports"]
        targeted_research = state.get("targeted_research", {})
        
        # Generate PDF using utils
        pdf_base64, filename = generate_pdf(structured_reports, targeted_research)
        
        pdf_data = PDFData(pdf_base64=pdf_base64, filename=filename)
        
        dispatch_custom_event("pdf_complete", f"PDF generated: {filename}")
        return {"pdf_data": pdf_data}

    def build_graph(self):
        """Build and compile the stock digest graph"""
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("PolygonFinance", self.polygon_finance_node)
        graph_builder.add_node("TargetedResearch", self.targeted_research_node)
        graph_builder.add_node("GeminiAnalysis", self.gemini_analysis_node)
        graph_builder.add_node("PDFGeneration", self.pdf_generation_node)
        
        graph_builder.add_edge(START, "PolygonFinance")
        graph_builder.add_edge("PolygonFinance", "TargetedResearch")
        graph_builder.add_edge("TargetedResearch", "GeminiAnalysis")
        graph_builder.add_edge("GeminiAnalysis", "PDFGeneration")
        graph_builder.add_edge("PDFGeneration", END)
        
        return graph_builder.compile()

    async def run_digest(self, tickers: List[str]) -> StockDigestOutput:
        """Run the complete stock digest workflow"""
        logger.info(f"Starting stock digest for tickers: {tickers}")
        
        graph = self.build_graph()
        initial_state = {"tickers": tickers, "date": self.current_date}
        
        final_state = await graph.ainvoke(initial_state)
        return final_state["structured_reports"]