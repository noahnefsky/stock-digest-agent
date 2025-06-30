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


class StockReport(BaseModel):
    ticker: str
    company_name: str
    executive_summary: str
    current_performance: str
    key_insights: List[str] = Field(default_factory=list)
    recommendation: str
    risk_assessment: str
    price_outlook: str


class StockDigestOutput(BaseModel):
    reports: Dict[str, StockReport] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    market_overview: str


class State(TypedDict):
    tickers: List[str]
    stock_data: Dict[str, StockData]
    research_data: Dict[str, StockResearch]
    structured_reports: StockDigestOutput
    date: str


class StockDigestAgent:
    def __init__(self):
        # Initialize LLMs and clients
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
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
        
        for ticker in tickers:
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
        
        stock_data = state["stock_data"]
        research_data = {}
        
        for ticker, data in stock_data.items():
            dispatch_custom_event("research_ticker", f"Researching {ticker}")
            
            # Search for recent news and analysis
            search_query = f"{ticker} {data.company_name} stock news analysis today {self.current_date}"
            
            search_results = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                max_results=5,
                include_raw_content=True,
                days=1  # Focus on recent news
            )
            
            # Extract key information from search results
            news_content = []
            for result in search_results.get('results', []):
                if result.get('content'):
                    news_content.append(result['content'])
            
            # Create research summary
            combined_content = "\n".join(news_content[:3])  # Limit content
            
            research_data[ticker] = StockResearch(
                ticker=ticker,
                news_summary=combined_content[:1000],  # Limit length
                key_developments=[],  # Will be filled by LLM analysis
                analyst_sentiment="Pending analysis",
                risk_factors=[],
                price_targets=None
            )
            
            logger.info(f"Research completed for {ticker}")
        
        return {"research_data": research_data}

    def gemini_analysis_node(self, state: State) -> Dict:
        """Use Gemini LLM to analyze stock data and research, returning structured reports"""
        dispatch_custom_event("gemini_analysis_status", "Generating structured stock reports...")
        
        stock_data = state["stock_data"]
        research_data = state["research_data"]
        
        # Create structured parser using Gemini with structured output
        structured_llm = self.gemini_llm.with_structured_output(StockDigestOutput)
        
        # Prepare combined data for analysis
        analysis_data = {}
        for ticker in stock_data.keys():
            analysis_data[ticker] = {
                "stock_data": stock_data[ticker].model_dump(),
                "research": research_data[ticker].model_dump() if ticker in research_data else {}
            }
        
        analysis_prompt = f"""
        You are a professional stock analyst creating a comprehensive stock digest report.
        
        Analyze the following stock data and research for {len(stock_data)} stocks:
        
        {json.dumps(analysis_data, indent=2)}
        
        Current date: {self.current_date}
        
        For each stock, create a comprehensive report including:
        1. Executive summary (2-3 sentences)
        2. Current performance analysis
        3. Key insights from recent news and data
        4. Investment recommendation (Buy/Hold/Sell with reasoning)
        5. Risk assessment
        6. Price outlook for the next 30 days
        
        Also provide an overall market overview considering all analyzed stocks.
        
        Focus on:
        - Recent price movements and volume
        - News sentiment and key developments
        - Financial metrics (P/E ratio, market cap)
        - Risk factors and opportunities
        - Clear, actionable insights
        
        Return the analysis in the structured format with individual reports for each ticker.
        """
        
        # Generate structured output
        structured_reports = structured_llm.invoke(analysis_prompt)
        
        dispatch_custom_event("analysis_complete", f"Generated reports for {len(structured_reports.reports)} stocks")
        
        return {"structured_reports": structured_reports}

    def build_graph(self):
        """Build and compile the stock digest graph"""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("Stock Retrieval", self.stock_retrieval_node)
        graph_builder.add_node("Research", self.research_node)
        graph_builder.add_node("Gemini Analysis", self.gemini_analysis_node)
        
        # Define edges
        graph_builder.add_edge(START, "Stock Retrieval")
        graph_builder.add_edge("Stock Retrieval", "Research")
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