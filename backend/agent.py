import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List
from typing import Optional as OptionalType
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
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

    def _fetch_ticker_finance_data(self, ticker: str) -> StockFinanceData:
        """Fetch finance data for a single ticker"""
        logger.info(f"Processing {ticker}")
        
        # Get ticker details and previous close
        details = self.polygon_client.get_ticker_details(ticker)
        prev_day = self.polygon_client.get_previous_close_agg(ticker)
        
        # Initialize default values
        current_price = 0.0
        previous_close = 0.0
        change_percent = 0.0
        volume = 0
        market_cap = None
        pe_ratio = None
        company_name = ticker
        
        # Handle previous close data
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
        
        return StockFinanceData(
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

    def polygon_finance_node(self, state: State) -> Dict:
        """Retrieve stock data from Polygon.io with concurrent processing"""
        logger.info("Fetching financial data from Polygon.io...")
        
        tickers = state["tickers"]
        finance_data = {}
        
        # Process up to 3 tickers concurrently to respect rate limits
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_ticker = {executor.submit(self._fetch_ticker_finance_data, ticker): ticker for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                result = future.result()
                finance_data[ticker] = result
                logger.info(f"Retrieved data for {ticker} from Polygon.io")
                
                # Minimal delay between requests
                time.sleep(0.5)
            
        return {"finance_data": finance_data}

    def _fetch_ticker_research(self, ticker: str) -> TargetedResearch:
        """Fetch research data for a single ticker"""
        logger.info(f"Researching {ticker}")
        
        # Comprehensive search query for better results
        search_query = f"{ticker} stock earnings analyst news market"
        
        search_results = self.tavily_client.search(
            query=search_query,
            search_depth="basic",
            max_results=10,  # Increased to 10 sources
            include_raw_content=False,  # Faster without raw content
            include_answer=False,
            include_domains=["reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com", "yahoo.com", "seekingalpha.com", "wsj.com", "fool.com"]
        )
        
        stories = [{
            'title': r.get('title', ''),
            'content': r.get('content', '')[:200],  # Reduced content length
            'url': r.get('url', ''),
            'published_date': r.get('published_date', ''),
            'source': r.get('source', ''),
            'score': r.get('score', 0),
            'domain': r.get('domain', ''),
            'keyword': 'comprehensive'
        } for r in search_results.get('results', [])]
        
        logger.info(f"Found {len(stories)} stories for {ticker}")
        
        # Quick categorization - put all in sector_news for simplicity
        categorized_stories = {
            "earnings_news": [],
            "analyst_ratings": [],
            "insider_trading": [],
            "technical_analysis": [],
            "sector_news": stories  # All stories go here for speed
        }
        
        ticker_research = {
            "ticker": ticker,
            **categorized_stories
        }
        
        return TargetedResearch(**ticker_research)

    def targeted_research_node(self, state: State) -> Dict:
        """Perform targeted research for each ticker with concurrent processing"""
        logger.info("Performing comprehensive research for each ticker...")
        
        tickers = state["tickers"]
        targeted_research = {}
        
        # Process all tickers concurrently for maximum speed
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(self._fetch_ticker_research, ticker): ticker for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                result = future.result()
                targeted_research[ticker] = result
                logger.info(f"Research completed for {ticker}")
        
        return {"targeted_research": targeted_research}

    def _generate_ticker_summary(self, ticker: str, targeted_research: Dict, finance_data: Dict) -> tuple:
        """Generate summary for a single ticker"""
        logger.info(f"Generating summary for {ticker}")
        
        # Collect stories for this ticker
        ticker_stories = []
        if ticker in targeted_research:
            research = targeted_research[ticker]
            for category, stories in research.model_dump().items():
                if category != 'ticker' and stories:
                    for story in stories:
                        story_with_ticker = story.copy()
                        story_with_ticker['ticker'] = ticker
                        ticker_stories.append(story_with_ticker)
        
        ticker_finance = finance_data.get(ticker)
        summary = self._generate_summary_prompt(ticker, ticker_stories, ticker_finance)
        
        return ticker, summary, ticker_stories

    def stock_summary_generator_node(self, state: State) -> Dict:
        """Generate summaries for each stock with concurrent processing"""
        logger.info("Generating stock summaries...")
        
        tickers = state["tickers"]
        targeted_research = state.get("targeted_research", {})
        finance_data = state.get("finance_data", {})
        
        stock_summaries = {}
        all_news_stories = []
        
        # Process all summaries concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {
                executor.submit(self._generate_ticker_summary, ticker, targeted_research, finance_data): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker, summary, ticker_stories = future.result()
                stock_summaries[ticker] = summary
                
                # Add stories to all_news_stories
                for story in ticker_stories:
                    all_news_stories.append((ticker, story))
        
        logger.info(f"Generated summaries for {len(stock_summaries)} stocks")
        return {
            "stock_summaries": stock_summaries,
            "all_news_stories": all_news_stories
        }

    def _generate_summary_prompt(self, ticker: str, ticker_stories: List[Dict], ticker_finance) -> str:
        """Generate summary using LLM prompt with optimized content"""
        # Always try to generate a summary, even with minimal data
        prompt_parts = [f"Write a 2-3 sentence summary for {ticker} stock:"]
        
        # Add financial data if available
        if ticker_finance and ticker_finance.current_price > 0:
            prompt_parts.append(f"Current price: ${ticker_finance.current_price:.2f}")
            if ticker_finance.change_percent != 0:
                prompt_parts.append(f"Change: {ticker_finance.change_percent:+.1f}%")
            if ticker_finance.volume > 0:
                prompt_parts.append(f"Volume: {ticker_finance.volume:,}")
            if ticker_finance.company_name and ticker_finance.company_name != ticker:
                prompt_parts.append(f"Company: {ticker_finance.company_name}")
        
        # Add news stories if available
        if ticker_stories:
            # Use top 2 stories for better coverage
            for i, story in enumerate(ticker_stories[:2]):
                title = story.get('title', '').strip()
                if title:
                    prompt_parts.append(f"News {i+1}: {title}")
        
        # If we have minimal data, provide a more generic prompt
        if len(prompt_parts) <= 1:
            prompt_parts.append(f"Provide a brief overview of {ticker} stock based on available market information.")
        
        summary_prompt = " ".join(prompt_parts)
        
        try:
            result = self.gemini_llm.invoke(summary_prompt)
            summary = str(result.content) if result.content else ""
            if summary and not summary.startswith("No data"):
                logger.info(f"Generated summary for {ticker}: {summary[:50]}...")
                return summary
            else:
                # Fallback summary
                return f"{ticker} stock analysis completed. Current market data and news have been reviewed."
        except Exception as e:
            logger.warning(f"Error generating summary for {ticker}: {e}")
            return f"{ticker} stock analysis completed with available market information."

    def _generate_structured_report(self, ticker: str, research, ticker_stories: List[Dict], finance, summary: str) -> StockReport:
        """Generate structured report for a ticker"""
        structured_llm = self.gemini_llm.with_structured_output(StockReport)
        analysis_prompt = get_stock_analysis_prompt(ticker, research, ticker_stories, self.current_date)
        report = structured_llm.invoke(analysis_prompt)
        
        # Build report dictionary with all required data
        report_dict = dict(report)
        report_dict['sources'] = ticker_stories
        report_dict['finance_data'] = finance
        report_dict['summary'] = summary
        
        return StockReport(**report_dict)

    def _generate_ticker_report(self, ticker: str, targeted_research: Dict, finance_data: Dict, stock_summaries: Dict, all_news_stories: List) -> tuple:
        """Generate structured report for a single ticker"""
        logger.info(f"Formatting report for {ticker}")
        
        research = targeted_research.get(ticker, {})
        finance = finance_data.get(ticker)
        ticker_stories = [story for t, story in all_news_stories if t == ticker]
        
        # Use existing summary or generate a new one if needed
        summary = stock_summaries.get(ticker, "")
        if not summary or summary.startswith("No data available"):
            # Regenerate summary with available data
            summary = self._generate_summary_prompt(ticker, ticker_stories, finance)
        
        report = self._generate_structured_report(ticker, research, ticker_stories, finance, summary)
        return ticker, report

    def gemini_report_formatter_node(self, state: State) -> Dict:
        """Format structured reports using Gemini with concurrent processing"""
        logger.info("Formatting structured stock reports...")
        
        tickers = state["tickers"]
        targeted_research = state.get("targeted_research", {})
        finance_data = state.get("finance_data", {})
        stock_summaries = state.get("stock_summaries", {})
        all_news_stories = state.get("all_news_stories", [])
        
        reports = {}
        
        # Process all reports concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {
                executor.submit(self._generate_ticker_report, ticker, targeted_research, finance_data, stock_summaries, all_news_stories): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker, report = future.result()
                reports[ticker] = report
                logger.info(f"Formatted structured report for {ticker}")
        
        # Simplified market overview
        market_overview = f"Analysis of {len(tickers)} stocks completed on {self.current_date}."
        
        structured_reports = StockDigestOutput(
            reports=reports,
            market_overview=market_overview,
            generated_at=datetime.now().isoformat(),
        )
        
        logger.info(f"Generated reports for {len(reports)} stocks")
        return {"structured_reports": structured_reports}

    def pdf_generation_node(self, state: State) -> Dict:
        """Generate PDF report from the structured data"""
        logger.info("Generating PDF report...")
        
        structured_reports = state["structured_reports"]
        targeted_research = state.get("targeted_research", {})
        
        pdf_base64, filename = generate_pdf(structured_reports, targeted_research)
        pdf_data = PDFData(pdf_base64=pdf_base64, filename=filename)
        
        dispatch_custom_event("pdf_complete", f"PDF generated: {filename}")
        return {"pdf_data": pdf_data}

    def build_graph(self):
        """Build and compile the stock digest graph"""
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("PolygonFinance", self.polygon_finance_node)
        graph_builder.add_node("TargetedResearch", self.targeted_research_node)
        graph_builder.add_node("StockSummaryGenerator", self.stock_summary_generator_node)
        graph_builder.add_node("GeminiReportFormatter", self.gemini_report_formatter_node)
        graph_builder.add_node("PDFGeneration", self.pdf_generation_node)
        
        graph_builder.add_edge(START, "PolygonFinance")
        graph_builder.add_edge("PolygonFinance", "TargetedResearch")
        graph_builder.add_edge("TargetedResearch", "StockSummaryGenerator")
        graph_builder.add_edge("StockSummaryGenerator", "GeminiReportFormatter")
        graph_builder.add_edge("GeminiReportFormatter", "PDFGeneration")
        graph_builder.add_edge("PDFGeneration", END)
        
        return graph_builder.compile()

    async def run_digest(self, tickers: List[str]) -> StockDigestOutput:
        """Run the complete stock digest workflow with optimized processing"""
        logger.info(f"Starting optimized stock digest for tickers: {tickers}")
        
        # Run finance and research nodes in parallel
        start_time = time.time()
        
        graph = self.build_graph()
        initial_state = {"tickers": tickers, "date": self.current_date}
        
        final_state = await graph.ainvoke(initial_state)
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Stock digest completed in {processing_time:.2f} seconds for {len(tickers)} ticker(s)")
        
        return final_state["structured_reports"]