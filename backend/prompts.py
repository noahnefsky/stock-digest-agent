"""
Prompt templates for the Stock Digest Agent
"""

def get_stock_analysis_prompt(ticker: str, research_data: dict | object, ticker_stories: list, current_date: str) -> str:
    """
    Generate the prompt for individual stock analysis
    
    Args:
        ticker: Stock ticker symbol
        research_data: Research data for the stock (dict or Pydantic model)
        ticker_stories: List of news stories for the ticker
        current_date: Current date string
    
    Returns:
        Formatted prompt string
    """
    # Handle both dict and Pydantic model types
    if isinstance(research_data, dict):
        research_str = str(research_data)
    elif hasattr(research_data, 'model_dump'):
        research_str = str(research_data.model_dump())  # type: ignore
    else:
        research_str = str(research_data)
    
    return f"""
    You are a professional portfolio analyst providing concise updates on stock positions based on recent news and market developments.
    
    Analyze the following data for {ticker}:
    
    Research Data: {research_str}
    
    Recent News Stories for {ticker}:
    {chr(10).join([f"Title: {story['title']}{chr(10)}Content: {story['content']}{chr(10)}Source: {story['source']}{chr(10)}Date: {story['published_date']}{chr(10)}Relevance Score: {story.get('score', 'N/A')}{chr(10)}" for story in ticker_stories[:5]])}
    
    Current date: {current_date}
    
    Create a portfolio-focused report with the following fields:
    1. summary: In 4-5 sentences, summarize the most important details from the research data for this stock along with other important market details.
    2. current_performance: 2-3 sentences on recent price action and key metrics
    3. key_insights: List of 4-6 actionable items from the latest news (past 24 h). Focus on specific events, earnings, analyst actions or significant changes with the market or business
    4. recommendation: Investment recommendation (Buy/Hold/Sell with brief reasoning)
    5. risk_assessment: 2-3 sentences identifying key risks from news
    6. price_outlook: 1-2 sentences on near-term expectations

    **Key-Insights formatting rules**  
    • Return as separate bullet strings.  
    • Be explicit: numbers, dates, names.  
    • Prioritise high-relevance stories.
    
    Provide the ticker symbol as: {ticker}
    Use a generic company name if not available in the data.
    """


def get_market_overview_prompt(tickers: list, all_news_stories: list, current_date: str) -> str:
    """
    Generate the prompt for market overview analysis
    
    Args:
        tickers: List of stock tickers
        all_news_stories: List of all news stories across all tickers
        current_date: Current date string
    
    Returns:
        Formatted prompt string
    """
    return f"""
    You are a senior market analyst creating a concise market overview based on recent news and developments.
    
    Analyze the following news stories from the past few days for these stocks: {', '.join(tickers)}
    
    News Stories Summary:
    {chr(10).join([f"Ticker: {ticker}{chr(10)}Title: {story['title']}{chr(10)}Content: {story['content'][:300]}...{chr(10)}Source: {story['source']}{chr(10)}Date: {story['published_date']}{chr(10)}" for ticker, story in all_news_stories[:15]])}
    
    Current date: {current_date}
    
    Create a concise market overview (4-5 sentences) that covers:
    
     1. Market Sentiment
        2. Emerging Themes
        3. Notable Developments + Impact
        4. Risks / Opportunities
        5. Portfolio Actions to Consider
    
    Focus on cross-ticker patterns, high-impact news, and clear, actionable insights.
    """
