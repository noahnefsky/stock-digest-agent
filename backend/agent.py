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
from mcp_use import MCPAgent, MCPClient
from typing_extensions import TypedDict
from polygon import RESTClient
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate

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
        
        # Initialize React agent for Tavily search
        self.react_tools = [
            TavilySearch(
                max_results=10,
                include_raw_content=True,
                search_depth="advanced",
                api_key=os.getenv("TAVILY_API_KEY")
            )
        ]
        
        self.react_prompt = PromptTemplate.from_template("""You are a helpful AI assistant that searches for information using the available tools.

When given a search query, you should:
1. Use the available search tools to find relevant information
2. Return the search results in a structured format
3. Focus on recent and relevant information
4. Provide clear, concise summaries

Available tools: {tools}
Tool names: {tool_names}

Question: {input}
{agent_scratchpad}""")
        
        self.react_agent = create_react_agent(
            self.gemini_llm, self.react_tools, self.react_prompt
        )
        self.react_agent_executor = AgentExecutor(
            agent=self.react_agent, tools=self.react_tools, handle_parsing_errors=True
        )
        
        # Initialize MCP client for Tavily (keeping as backup)
        self.mcp_config = {
            "mcpServers": {
                "tavily-mcp": {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@0.1.3"],
                    "env": {
                        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")
                    }
                }
            }
        }
        
        self.mcp_client = MCPClient.from_dict(self.mcp_config)
        self.mcp_agent = MCPAgent(llm=self.gemini_llm, client=self.mcp_client, max_steps=30)
        
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
        """Perform targeted research for each ticker using React agent with Tavily search"""
        dispatch_custom_event("targeted_research_status", "Performing comprehensive research for each ticker using React agent...")
        
        tickers = state["tickers"]
        targeted_research = {}
        
        for ticker in tickers:
            dispatch_custom_event("targeted_research_ticker", f"Researching {ticker}")
            
            # Comprehensive search query covering all important categories with past week filter
            search_query = f"Find recent news about {ticker} stock from the past week including earnings reports, analyst ratings, insider trading, technical analysis, and sector news. Return the search results with titles, URLs, and content."
            
            # Use React agent to perform the search
            search_result = self.react_agent_executor.invoke({"input": search_query})
            
            # Parse the search results from the React agent response
            search_results = self._parse_react_search_results(search_result)
            
            all_stories = []
            for r in search_results.get('results', []):
                if isinstance(r, dict):
                    story = {
                        'title': r.get('title', r.get('name', '')),
                        'content': r.get('content', r.get('snippet', ''))[:400],
                        'url': r.get('url', r.get('link', '')),
                        'published_date': r.get('published_date', r.get('date', '')),
                        'source': r.get('source', r.get('domain', '')),
                        'score': r.get('score', 0),
                        'domain': r.get('domain', ''),
                        'keyword': 'comprehensive'
                    }
                    all_stories.append(story)
                elif isinstance(r, str):
                    # Handle string results
                    all_stories.append({
                        'title': 'Search Result',
                        'content': r[:400],
                        'url': '',
                        'published_date': '',
                        'source': '',
                        'score': 0,
                        'domain': '',
                        'keyword': 'comprehensive'
                    })
            
            # Limit to top 10 sources per ticker by score
            all_stories.sort(key=lambda x: x.get('score', 0), reverse=True)
            all_stories = all_stories[:10]
            
            # If no real stories found, try a simpler search
            if not all_stories or all(all_story.get('title') == 'Search Result' for all_story in all_stories):
                logger.warning(f"No real stories found for {ticker}, trying simpler search")
                simple_search_query = f"Find recent news about {ticker} stock from the past week"
                simple_result = self.react_agent_executor.invoke({"input": simple_search_query})
                simple_results = self._parse_react_search_results(simple_result)
                
                for r in simple_results.get('results', []):
                    if isinstance(r, dict) and r.get('title') and r.get('title') != 'Search Result':
                        story = {
                            'title': r.get('title', ''),
                            'content': r.get('content', r.get('snippet', ''))[:400],
                            'url': r.get('url', r.get('link', '')),
                            'published_date': r.get('published_date', r.get('date', '')),
                            'source': r.get('source', r.get('domain', '')),
                            'score': r.get('score', 0),
                            'domain': r.get('domain', ''),
                            'keyword': 'simple_fallback'
                        }
                        all_stories.append(story)
                
                # Re-sort and limit
                all_stories.sort(key=lambda x: x.get('score', 0), reverse=True)
                all_stories = all_stories[:10]
            
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
            
            # Efficient categorization
            for story in all_stories:
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
            logger.info(f"Research completed for {ticker}: {len(all_stories)} stories, {sum(len(cat) for cat in categorized_stories.values())} categorized")
            time.sleep(1)  # Reduced delay
        
        return {"targeted_research": targeted_research}

    def _parse_react_search_results(self, react_result):
        """Parse search results from React agent response"""
        logger.info(f"React result type: {type(react_result)}, content: {str(react_result)[:500]}...")
        
        # React agent returns a dict with 'output' key
        if isinstance(react_result, dict) and 'output' in react_result:
            output = react_result['output']
            
            # If output is a string, try to parse it
            if isinstance(output, str):
                # Try to extract structured data from the text
                lines = output.split('\n')
                results = []
                current_result = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Title:') or line.startswith('title:'):
                        if current_result:
                            results.append(current_result)
                        current_result = {'title': line.split(':', 1)[1].strip()}
                    elif line.startswith('URL:') or line.startswith('url:'):
                        current_result['url'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Content:') or line.startswith('content:'):
                        current_result['content'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Source:') or line.startswith('source:'):
                        current_result['source'] = line.split(':', 1)[1].strip()
                
                if current_result:
                    results.append(current_result)
                
                if results:
                    return {"results": results}
                
                # If no structured data found, treat as a single result
                return {"results": [{"content": output, "title": "Search Result"}]}
            
            # If output is already a list or dict, return as is
            elif isinstance(output, list):
                return {"results": output}
            elif isinstance(output, dict):
                return output if "results" in output else {"results": [output]}
        
        # Fallback to empty results
        return {"results": []}

    def _parse_mcp_search_results(self, mcp_result):
        """Parse search results from MCP agent response"""
        logger.info(f"MCP result type: {type(mcp_result)}, content: {str(mcp_result)[:500]}...")
        
        # Handle different response formats from MCP
        if isinstance(mcp_result, dict):
            # If it's already a dict with results, return as is
            if "results" in mcp_result:
                return mcp_result
            # If it's a dict but no results key, wrap it
            return {"results": [mcp_result]}
        elif isinstance(mcp_result, str):
            # Try to parse JSON if it's a string
            try:
                parsed = json.loads(mcp_result)
                if isinstance(parsed, list):
                    return {"results": parsed}
                elif isinstance(parsed, dict):
                    return parsed if "results" in parsed else {"results": [parsed]}
                else:
                    return {"results": []}
            except json.JSONDecodeError:
                # If it's not JSON, try to extract structured content from the text
                # Look for patterns that might indicate search results
                if "title" in mcp_result.lower() or "url" in mcp_result.lower():
                    # Try to extract structured data from the text
                    lines = mcp_result.split('\n')
                    results = []
                    current_result = {}
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('Title:') or line.startswith('title:'):
                            if current_result:
                                results.append(current_result)
                            current_result = {'title': line.split(':', 1)[1].strip()}
                        elif line.startswith('URL:') or line.startswith('url:'):
                            current_result['url'] = line.split(':', 1)[1].strip()
                        elif line.startswith('Content:') or line.startswith('content:'):
                            current_result['content'] = line.split(':', 1)[1].strip()
                        elif line.startswith('Source:') or line.startswith('source:'):
                            current_result['source'] = line.split(':', 1)[1].strip()
                    
                    if current_result:
                        results.append(current_result)
                    
                    if results:
                        return {"results": results}
                
                # Check if the response indicates no results or error
                if "unavailability" in mcp_result.lower() or "no recent" in mcp_result.lower() or "impossible" in mcp_result.lower():
                    logger.warning(f"MCP returned error response for search: {mcp_result[:200]}...")
                    return {"results": []}
                
                # If no structured data found, treat as a single result
                return {"results": [{"content": mcp_result, "title": "Search Result"}]}
        elif isinstance(mcp_result, list):
            return {"results": mcp_result}
        else:
            # Fallback to empty results
            return {"results": []}



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

    def run_digest(self, tickers: List[str]) -> StockDigestOutput:
        """Run the complete stock digest workflow"""
        logger.info(f"Starting stock digest for tickers: {tickers}")
        
        graph = self.build_graph()
        initial_state = {"tickers": tickers, "date": self.current_date}
        
        final_state = graph.invoke(initial_state)
        return final_state["structured_reports"]