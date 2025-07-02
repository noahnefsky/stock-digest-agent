import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient
from polygon import RESTClient
import json
import re

from prompts import get_stock_analysis_prompt, get_market_overview_summary_prompt, get_stock_recommendations_extraction_prompt
from agent_utils import generate_pdf
from models import (
    StockFinanceData, TargetedResearch, StockReport,
    StockDigestOutput, PDFData, State
)

load_dotenv()

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

    def _fetch_ticker_data(self, ticker: str) -> tuple[str, StockFinanceData]:
        try:
            details = self.polygon_client.get_ticker_details(ticker)
            prev_day = self.polygon_client.get_previous_close_agg(ticker)
            if isinstance(prev_day, list) and prev_day:
                prev_bar = prev_day[0]
                previous_close = float(getattr(prev_bar, 'close', 0) or 0)
            
            # For free tier, we'll use previous close as current price
            current_price = previous_close
            
        except Exception as e:
            logger.warning(f"Error fetching data for {ticker}: {e}")
            details = type('obj', (object,), {
                'name': ticker, 
                'market_cap': None, 
                'last_updated_utc': None,
                'share_class_shares_outstanding': None
            })()
            current_price = 0.0

        company_name = getattr(details, 'name', ticker)
        market_cap = float(getattr(details, 'market_cap', 0) or 0) or None

        finance_data = StockFinanceData(
            ticker=ticker,
            current_price=current_price,
            market_cap=market_cap,
            company_name=company_name,
        )
        logger.info(f"Retrieved data for {ticker} from Polygon.io - Current: ${current_price:.2f}")
        return ticker, finance_data

    def stock_metrics_node(self, state: State) -> Dict:
        dispatch_custom_event("finance_status", "Fetching financial data from Polygon.io...")
        tickers = state["tickers"]
        finance_data = {}

        for i, ticker in enumerate(tickers):
            ticker, data = self._fetch_ticker_data(ticker)
            finance_data[ticker] = data
            dispatch_custom_event("finance_ticker", f"Completed {ticker} ({i+1}/{len(tickers)})")
            time.sleep(2)

        return {"finance_data": finance_data}

    def _fetch_ticker_research(self, ticker: str) -> tuple[str, TargetedResearch]:
        query = f"{ticker} earnings analyst ratings insider trading technical analysis sector news {self.current_date}"
        search_results = {"results": []}
        
        try:
            search_results = self.tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_raw_content=True,
                include_answer=True,
                include_domains=["reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com", "yahoo.com", "seekingalpha.com"]
            )
        except Exception as e:
            logger.warning(f"Error fetching research for {ticker}: {e}")

        stories = [{
            'title': r.get('title', ''),
            'content': r.get('content', '')[:300],
            'url': r.get('url', ''),
            'published_date': r.get('published_date', ''),
            'source': r.get('source', ''),
            'score': r.get('score', 0),
            'domain': r.get('domain', ''),
            'keyword': 'comprehensive'
        } for r in search_results.get('results', [])]

        categorized = {
            "earnings_news": [],
            "analyst_ratings": [],
            "insider_trading": [],
            "technical_analysis": [],
            "sector_news": []
        }

        keywords = {
            "earnings_news": {"earnings", "quarterly", "revenue", "profit", "guidance", "results"},
            "analyst_ratings": {"analyst", "rating", "target", "upgrade", "downgrade", "recommendation"},
            "insider_trading": {"insider", "sec", "filing", "executive", "form 4"},
            "technical_analysis": {"technical", "support", "resistance", "rsi", "macd", "chart"}
        }

        for story in stories:
            content = (story['content'] + ' ' + story['title']).lower()
            assigned = False
            for category, keys in keywords.items():
                if any(k in content for k in keys):
                    categorized[category].append(story)
                    assigned = True
                    break
            if not assigned:
                categorized["sector_news"].append(story)

        research = TargetedResearch(ticker=ticker, **categorized)
        logger.info(f"Research completed for {ticker} with {len(stories)} stories")
        return ticker, research

    def targeted_research_node(self, state: State) -> Dict:
        dispatch_custom_event("targeted_research_status", "Performing comprehensive research...")
        tickers = state["tickers"]
        research_data = {}

        for i, ticker in enumerate(tickers):
            ticker, research = self._fetch_ticker_research(ticker)
            research_data[ticker] = research
            dispatch_custom_event("targeted_research_ticker", f"Completed {ticker} ({i+1}/{len(tickers)})")
            time.sleep(3)

        return {"targeted_research": research_data}

    def _analyze_ticker(self, ticker: str, targeted_research: Dict, finance_data: Dict, all_stories: List) -> tuple[str, StockReport]:
        research = targeted_research.get(ticker, {})
        finance = finance_data.get(ticker)
        ticker_stories = [story for t, story in all_stories if t == ticker]

        structured_llm = self.gemini_llm.with_structured_output(StockReport)
        prompt = get_stock_analysis_prompt(ticker, research, ticker_stories, self.current_date)
        report = structured_llm.invoke(prompt)

        report_dict = dict(report)
        report_dict['sources'] = ticker_stories
        report_dict['finance_data'] = finance
        return ticker, StockReport(**report_dict)

    def gemini_analysis_formatter_node(self, state: State) -> Dict:
        dispatch_custom_event("gemini_analysis_status", "Generating structured stock reports...")
        tickers = state["tickers"]
        targeted_research = state.get("targeted_research", {})
        finance_data = state.get("finance_data", {})

        all_stories = []
        for ticker in tickers:
            research = targeted_research.get(ticker, {})
            if isinstance(research, dict):
                research_dict = research
            elif hasattr(research, 'model_dump'):
                research_dict = research.model_dump()
            else:
                research_dict = {}
            
            for category, stories in research_dict.items():
                if category != 'ticker' and stories:
                    all_stories.extend((ticker, story.copy()) for story in stories)

        reports = {}
        for i, ticker in enumerate(tickers):
            ticker, report = self._analyze_ticker(ticker, targeted_research, finance_data, all_stories)
            reports[ticker] = report
            time.sleep(2)
            dispatch_custom_event("analysis_ticker", f"Completed {ticker} ({i+1}/{len(tickers)})")

        return {
            "structured_reports": StockDigestOutput(
                reports=reports,
                market_overview="",
                generated_at=datetime.now().isoformat(),
                ticker_suggestions={}
            )
        }

    def market_overview_summary_node(self, state: State) -> Dict:
        dispatch_custom_event("market_overview_summary_status", "Creating detailed market overview...")
        structured_reports = state["structured_reports"]
        finance_data = state.get("finance_data", {})

        comprehensive_texts = []
        for ticker, report in structured_reports.reports.items():
            finance = finance_data.get(ticker)
            company = finance.company_name if finance else ticker
            market_cap = f"${finance.market_cap/1e9:.2f}B" if finance and finance.market_cap else "N/A"
            text = (
                f"TICKER: {ticker}\n"
                f"COMPANY: {company}\n"
                f"CURRENT PRICE: ${finance.current_price if finance else 'N/A'}\n"
                f"MARKET CAP: {market_cap}\n"
                f"SUMMARY: {report.summary}\n"
                f"CURRENT PERFORMANCE: {report.current_performance}\n"
                f"KEY INSIGHTS: {report.key_insights}\n"
                f"RECOMMENDATION: {report.recommendation}\n"
                f"RISK ASSESSMENT: {report.risk_assessment}\n"
                f"PRICE OUTLOOK: {report.price_outlook}\n"
            )
            comprehensive_texts.append(text)

        concatenated = "\n\n".join(comprehensive_texts)
        refine_prompt = PromptTemplate(input_variables=["text"], template=get_market_overview_summary_prompt())
        chain = load_summarize_chain(self.gemini_llm, chain_type="refine", refine_prompt=refine_prompt)
        docs = [Document(page_content=concatenated)]
        overview_result = chain.invoke({"input_documents": docs})

        updated_reports = StockDigestOutput(
            reports=structured_reports.reports,
            market_overview=overview_result["output_text"],
            generated_at=structured_reports.generated_at,
            ticker_suggestions=structured_reports.ticker_suggestions
        )
        return {"structured_reports": updated_reports}

    def stock_recommendations_research_node(self, state: State) -> Dict:
        dispatch_custom_event("stock_recommendations_status", "Finding current stock recommendations...")
        
        query = "best stock picks analyst recommendations buy rating upgrades 2025"
        
        search_results = self.tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
            include_domains=["seekingalpha.com", "marketwatch.com", "yahoo.com", "cnbc.com", "bloomberg.com", "reuters.com"]
        )
        
        # Extract text from search results
        answer_text = search_results.get("answer", "")
        if not answer_text and "results" in search_results:
            results_content = []
            for result in search_results["results"][:3]:
                if "content" in result:
                    results_content.append(result["content"][:300])  # Limit each result for speed
            answer_text = " ".join(results_content)
        
        logger.info(f"Stock recommendations found: {answer_text[:200]}...")
        logger.info(f"Returning state with text length: {len(answer_text)}")
        
        return {"recommendations_raw_text": answer_text}

    def recommendation_formatting_node(self, state: State) -> Dict:
        dispatch_custom_event("recommendation_formatting_status", "Formatting stock recommendations...")
        
        raw_text = state.get("recommendations_raw_text", "")
        logger.info(f"Formatting node received text length: {len(raw_text)}")
        logger.info(f"Full state keys: {list(state.keys())}")
        
        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning(f"No sufficient recommendations text found. Text length: {len(raw_text)}")
            ticker_suggestions = {}
        else:
            extraction_prompt = get_stock_recommendations_extraction_prompt(raw_text)
            
            response = self.gemini_llm.invoke(extraction_prompt)
            
            response_text = str(response.content)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            ticker_suggestions = {}
            if json_match:
                extracted_data = json.loads(json_match.group())
                for ticker, reason in extracted_data.items():
                    if re.match(r'^[A-Z]{2,5}$', ticker) and len(reason.strip()) > 0:
                        ticker_suggestions[ticker] = reason.strip()
            
            logger.info(f"LLM extracted tickers: {ticker_suggestions}")
        
        structured_reports = state["structured_reports"]
        updated_reports = StockDigestOutput(
            reports=structured_reports.reports,
            market_overview=structured_reports.market_overview,
            generated_at=structured_reports.generated_at,
            ticker_suggestions=ticker_suggestions
        )
        
        return {"structured_reports": updated_reports}

    def pdf_generation_node(self, state: State) -> Dict:
        dispatch_custom_event("pdf_generation_status", "Generating PDF report...")
        structured_reports = state["structured_reports"]
        targeted_research = state.get("targeted_research", {})

        pdf_base64, filename = generate_pdf(structured_reports, targeted_research)
        pdf_data = PDFData(pdf_base64=pdf_base64, filename=filename)
        return {"pdf_data": pdf_data}

    def build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("StockMetrics", self.stock_metrics_node)
        graph_builder.add_node("TargetedResearch", self.targeted_research_node)
        graph_builder.add_node("GeminiAnalysisFormatter", self.gemini_analysis_formatter_node)
        graph_builder.add_node("MarketOverviewSummary", self.market_overview_summary_node)
        graph_builder.add_node("StockRecommendationsResearch", self.stock_recommendations_research_node)
        graph_builder.add_node("RecommendationFormatting", self.recommendation_formatting_node)
        graph_builder.add_node("PDFGeneration", self.pdf_generation_node)

        graph_builder.add_edge(START, "StockMetrics")
        graph_builder.add_edge("StockMetrics", "TargetedResearch")
        graph_builder.add_edge("TargetedResearch", "GeminiAnalysisFormatter")
        graph_builder.add_edge("GeminiAnalysisFormatter", "MarketOverviewSummary")
        graph_builder.add_edge("MarketOverviewSummary", "StockRecommendationsResearch")
        graph_builder.add_edge("StockRecommendationsResearch", "RecommendationFormatting")
        graph_builder.add_edge("RecommendationFormatting", "PDFGeneration")
        graph_builder.add_edge("PDFGeneration", END)
        return graph_builder.compile()

    async def run_digest(self, tickers: List[str]) -> StockDigestOutput:
        logger.info(f"Starting stock digest for tickers: {tickers}")
        graph = self.build_graph()
        initial_state = {"tickers": tickers, "date": self.current_date}
        final_state = await graph.ainvoke(initial_state)
        return final_state["structured_reports"]