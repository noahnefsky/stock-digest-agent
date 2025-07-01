import json
import logging
import os
from datetime import datetime
from typing import Dict, List
from typing import Optional as OptionalType

import yfinance as yf
from dotenv import load_dotenv
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import TypedDict

from prompts import get_stock_analysis_prompt, get_market_overview_prompt

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for structured output
class StockData(BaseModel):
    ticker: str
    current_price: float
    previous_close: float
    change_percent: float
    volume: int
    market_cap: OptionalType[float] = None
    pe_ratio: OptionalType[float] = None
    company_name: str


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
    summary: str  # Step 1 from prompt: summary of most important insights
    current_performance: str  # Step 2 from prompt
    key_insights: List[str] = Field(default_factory=list)
    recommendation: str  # Step 4 from prompt
    risk_assessment: str  # Step 5 from prompt
    price_outlook: str  # Step 6 from prompt
    sources: List[Dict] = Field(default_factory=list)


class StockDigestOutput(BaseModel):
    reports: Dict[str, StockReport] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    market_overview: str


class State(TypedDict):
    tickers: List[str]
    stock_data: Dict[str, StockData]
    research_data: Dict[str, StockResearch]
    all_news_stories: List[tuple]  # List of (ticker, story) tuples
    structured_reports: StockDigestOutput
    date: str


class StockDigestAgent:
    def __init__(self):
        # Initialize LLMs and clients
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Set current date
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def stock_retrieval_node(self, state: State) -> Dict:
        """Retrieve stock data for specified tickers using yfinance"""
        dispatch_custom_event("stock_retrieval_status", "Fetching stock data...")
        
        tickers = state["tickers"]
        stock_data = {}
        
        # Process only the first ticker
        if tickers:
            ticker = tickers[0]
            dispatch_custom_event("ticker_processing", f"Processing {ticker}")
            
            # Fetch stock data
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="2d")
            
            if len(hist) >= 2:
                current_price = float(hist['Close'].iloc[-1])
                previous_close = float(hist['Close'].iloc[-2])
                change_percent = ((current_price - previous_close) / previous_close) * 100
            else:
                current_price = info.get('currentPrice', 0)
                previous_close = info.get('previousClose', 0)
                change_percent = info.get('changePercent', 0)
            
            stock_data[ticker] = StockData(
                ticker=ticker,
                current_price=current_price,
                previous_close=previous_close,
                change_percent=change_percent,
                volume=info.get('volume', 0),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                company_name=info.get('longName', ticker)
            )
            
            logger.info(f"Retrieved data for {ticker}: {current_price}")
        
        return {"stock_data": stock_data}

    def research_node(self, state: State) -> Dict:
        """Research stocks using Tavily client for current news and analysis"""
        dispatch_custom_event("research_status", "Researching stock news and analysis...")
        
        tickers = state["tickers"]
        research_data = {}
        all_news_stories = []
        
        for ticker in tickers:
            dispatch_custom_event("research_ticker", f"Researching {ticker}")
            
            # Search for recent news and analysis with more comprehensive parameters
            search_query = f"{ticker} stock news analysis earnings financial performance market sentiment {self.current_date}"
            
            search_results = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                max_results=10,
                include_raw_content=True,
                include_answer=True,
                include_images=False,
                include_domains=["reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com", "yahoo.com", "seekingalpha.com"],
                exclude_domains=["twitter.com", "reddit.com"],
                days=3  # Get news from last 3 days for better context
            )
            
            # Extract detailed information from search results
            news_content = []
            news_stories = []
            
            for result in search_results.get('results', []):
                if result.get('content'):
                    news_content.append(result['content'])
                    
                    # Store individual news stories with metadata
                    news_stories.append({
                        'title': result.get('title', ''),
                        'content': result.get('content', '')[:800],  # Increased to 800 chars for better context
                        'url': result.get('url', ''),
                        'published_date': result.get('published_date', ''),
                        'source': result.get('source', ''),
                        'score': result.get('score', 0),  # Relevance score
                        'domain': result.get('domain', '')
                    })
            
            # Sort news stories by relevance score (if available) and recency
            news_stories.sort(key=lambda x: (x.get('score', 0), x.get('published_date', '')), reverse=True)
            
            # Add ticker-specific stories to overall collection
            all_news_stories.extend([(ticker, story) for story in news_stories])
            
            # Create research summary with more detailed content
            combined_content = "\n\n".join(news_content[:5])  # Use more content
            logger.info(f"Research content length for {ticker}: {len(combined_content)}")
            
            research_data[ticker] = StockResearch(
                ticker=ticker,
                news_summary=combined_content[:2000],  # Increased length limit
                key_developments=[],  # Will be filled by LLM analysis
                analyst_sentiment="Pending analysis",
                risk_factors=[],
                price_targets=None
            )
            
            logger.info(f"Research completed for {ticker} with {len(news_stories)} stories")
        
        # Store all news stories in state for market overview
        return {
            "research_data": research_data,
            "all_news_stories": all_news_stories
        }

    def gemini_analysis_node(self, state: State) -> Dict:
        """Use Gemini LLM to analyze stock data and research, returning structured reports"""
        dispatch_custom_event("gemini_analysis_status", "Generating structured stock reports...")
        
        tickers = state["tickers"]
        research_data = state["research_data"]
        all_news_stories = state.get("all_news_stories", [])
        
        # Create individual reports for each ticker
        reports = {}
        
        for ticker in tickers:
            research = research_data.get(ticker, {})
            
            # Get ticker-specific news stories
            ticker_stories = [story for ticker_symbol, story in all_news_stories if ticker_symbol == ticker]
            
            # Create structured parser for individual stock report
            structured_llm = self.gemini_llm.with_structured_output(StockReport)
            
            analysis_prompt = get_stock_analysis_prompt(ticker, research, ticker_stories, self.current_date)
            
            # Generate structured report for this ticker
            report = structured_llm.invoke(analysis_prompt)
            
            # Add sources to the report for this specific ticker
            ticker_sources = []
            for ticker_symbol, story in all_news_stories:
                if ticker_symbol == ticker:
                    source_dict = {
                        'ticker': ticker_symbol,
                        'title': story.get('title', ''),
                        'url': story.get('url', ''),
                        'source': story.get('source', ''),
                        'domain': story.get('domain', ''),
                        'published_date': story.get('published_date', ''),
                        'score': story.get('score', 0)
                    }
                    ticker_sources.append(source_dict)
            
            # Create a new report instance with the sources included
            report_dict = report.model_dump()
            report_dict['sources'] = ticker_sources
            reports[ticker] = StockReport(**report_dict)
        
        # Generate comprehensive market overview based on all news stories
        market_overview_prompt = get_market_overview_prompt(tickers, all_news_stories, self.current_date)
        
        market_overview_response = self.gemini_llm.invoke(market_overview_prompt)
        market_overview = market_overview_response.content
        
        # Ensure market_overview is a string
        if isinstance(market_overview, list):
            market_overview = " ".join(str(item) for item in market_overview)
        else:
            market_overview = str(market_overview)
        
        # Create the final structured output
        structured_reports = StockDigestOutput(
            reports=reports,
            market_overview=market_overview,
            generated_at=datetime.now().isoformat(),
        )
        
        dispatch_custom_event("analysis_complete", f"Generated reports for {len(reports)} stocks")
        
        return {"structured_reports": structured_reports}

    def build_graph(self):
        """Build and compile the stock digest graph"""
        graph_builder = StateGraph(State)
        
        # Add nodes
        # graph_builder.add_node("Stock Retrieval", self.stock_retrieval_node)
        graph_builder.add_node("Research", self.research_node)
        graph_builder.add_node("Gemini Analysis", self.gemini_analysis_node)
        
        # Define edges
        graph_builder.add_edge(START, "Research")
        # graph_builder.add_edge("Stock Retrieval", "Research")
        graph_builder.add_edge("Research", "Gemini Analysis")
        graph_builder.add_edge("Gemini Analysis", END)
        
        # Compile the graph
        compiled_graph = graph_builder.compile()
        
        return compiled_graph

    async def run_digest(self, tickers: List[str]) -> StockDigestOutput:
        """Run the complete stock digest workflow"""
        logger.info(f"Starting stock digest for tickers: {tickers}")
        
        # Build and run the graph
        graph = self.build_graph()
        
        initial_state = {
            "tickers": tickers,
            "date": self.current_date
        }
        
        # Execute the workflow
        final_state = await graph.ainvoke(initial_state)
        
        return final_state["structured_reports"]