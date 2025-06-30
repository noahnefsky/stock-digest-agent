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
    stock_market_overview: str
    current_performance: str
    key_insights: List[str] = Field(default_factory=list)
    recommendation: str
    risk_assessment: str
    price_outlook: str


class StockDigestOutput(BaseModel):
    reports: Dict[str, StockReport] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    market_overview: str
    sources: List[Dict] = Field(default_factory=list)  # Add sources field


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
            
            analysis_prompt = f"""
            You are a professional portfolio analyst providing concise updates on stock positions based on recent news and market developments.
            
            Analyze the following data for {ticker}:
            
            Research Data: {research if isinstance(research, dict) else research.model_dump()}
            
            Recent News Stories for {ticker}:
            {chr(10).join([f"Title: {story['title']}{chr(10)}Content: {story['content']}{chr(10)}Source: {story['source']}{chr(10)}Date: {story['published_date']}{chr(10)}Relevance Score: {story.get('score', 'N/A')}{chr(10)}" for story in ticker_stories[:5]])}
            
            Current date: {self.current_date}
            
            Create a portfolio-focused report including:
            1. Stock Market Overview (4-6 sentences covering recent market context and sector trends affecting this specific stock)
            2. Current Performance (2-3 sentences on recent price action and key metrics)
            3. Key Insights from recent news (focus on specific events, earnings, analyst actions)
            4. Investment recommendation (Buy/Hold/Sell with brief reasoning)
            5. Risk Assessment (2-3 sentences identifying key risks from news)
            6. Price Outlook (1-2 sentences on near-term expectations)
            
            For the Key Insights section, extract specific insights from the news stories above and format them as bullet points. Focus on:
            - Specific news events and their market impact
            - Analyst ratings, price targets, and recommendations
            - Earnings announcements, revenue figures, or financial metrics
            - Strategic moves, partnerships, or business developments
            - Regulatory news or legal developments
            - Market sentiment shifts or institutional actions
            
            Format the key insights as clear, concise bullet points that highlight the most important news-driven developments. Each bullet should be a specific, actionable insight from the news feed.
            
            IMPORTANT: For the key_insights field, provide each insight as a separate string in the list. Each insight should be:
            - Concise but informative (1-2 sentences max)
            - Based directly on the news stories provided
            - Include specific details like numbers, dates, or analyst names when available
            - Focus on actionable information that investors can use
            - Prioritize insights from high-relevance sources (higher score)
            
            Examples of good key insights:
            - "Analyst John Smith upgraded rating to Buy with $150 price target citing strong Q4 earnings"
            - "Company announced $2B acquisition of TechCorp, expected to close Q2 2024"
            - "Q3 revenue beat estimates by 15%, driven by 40% growth in cloud services"
            - "CEO announced new AI partnership with Microsoft, stock up 8% on the news"
            - "Federal Reserve decision impacts sector, company expects 5% revenue growth"
            
            Focus on providing portfolio-relevant updates that help investors understand:
            - How recent news affects their position
            - What to watch for in the coming days/weeks
            - Key risks and opportunities from current events
            - Actionable insights for portfolio management
            
            Provide the ticker symbol as: {ticker}
            Use a generic company name if not available in the data.
            """
            
            # Generate structured report for this ticker
            report = structured_llm.invoke(analysis_prompt)
            reports[ticker] = report
        
        # Generate comprehensive market overview based on all news stories
        market_overview_prompt = f"""
        You are a senior market analyst creating a concise market overview based on recent news and developments.
        
        Analyze the following news stories from the past few days for these stocks: {', '.join(tickers)}
        
        News Stories Summary:
        {chr(10).join([f"Ticker: {ticker}{chr(10)}Title: {story['title']}{chr(10)}Content: {story['content'][:300]}...{chr(10)}Source: {story['source']}{chr(10)}Date: {story['published_date']}{chr(10)}" for ticker, story in all_news_stories[:15]])}
        
        Current date: {self.current_date}
        
        Create a concise market overview (4-5 sentences) that covers:
        
        1. Overall market sentiment across these stocks
        2. Key themes or trends emerging from the news
        3. Notable developments and their market implications
        4. Brief risk assessment or opportunities
        
        Focus on:
        - Connecting individual stock news to broader market themes
        - Identifying patterns across multiple stocks
        - Providing clear, actionable insights
        - Maintaining professional, analytical tone
        
        Keep the overview concise but insightful, suitable for quick market assessment.
        """
        
        market_overview_response = self.gemini_llm.invoke(market_overview_prompt)
        market_overview = market_overview_response.content
        
        # Ensure market_overview is a string
        if isinstance(market_overview, list):
            market_overview = " ".join(str(item) for item in market_overview)
        else:
            market_overview = str(market_overview)
        
        # Convert all_news_stories to a list of dictionaries for the sources
        sources_list = []
        for ticker, story in all_news_stories:
            source_dict = {
                'ticker': ticker,
                'title': story.get('title', ''),
                'url': story.get('url', ''),
                'source': story.get('source', ''),
                'domain': story.get('domain', ''),
                'published_date': story.get('published_date', ''),
                'score': story.get('score', 0)
            }
            sources_list.append(source_dict)
        
        # Create the final structured output
        structured_reports = StockDigestOutput(
            reports=reports,
            market_overview=market_overview,
            generated_at=datetime.now().isoformat(),
            sources=sources_list
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