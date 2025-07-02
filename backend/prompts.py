def get_stock_analysis_prompt(ticker: str, research_data: dict | object, ticker_stories: list, current_date: str) -> str:
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
    return f"""
    You are a senior market analyst creating a concise market overview that connects individual stock developments to broader market trends.
    
    Analyze the following news stories from the past few days for these stocks: {', '.join(tickers)}
    
    News Stories Summary:
    {chr(10).join([f"Ticker: {ticker}{chr(10)}Title: {story['title']}{chr(10)}Content: {story['content'][:300]}...{chr(10)}Source: {story['source']}{chr(10)}Date: {story['published_date']}{chr(10)}" for ticker, story in all_news_stories[:15]])}
    
    Current date: {current_date}
    
    Create a concise market overview (5-7 sentences) covering overall market environemnt, trends and sentiment.
    
    """


def get_market_overview_summary_prompt() -> str:
    return """
    You are a senior portfolio strategist writing a polished, professional market overview for an investor.

    Conclude with a brief summary highlighting overall portfolio opportunities, risks, and positioning.

    Use clear, professional language suitable for an investor newsletter. Format as a unified commentary, not bullet points or isolated paragraphs.

    Analyze and incorporate the following research data:
    {text}
    """


def get_stock_recommendations_extraction_prompt(raw_text: str) -> str:
    return f"""
    Extract stock ticker symbols and their detailed investment reasons from the following text.
    Only return actual stock ticker symbols (2-5 letter codes like AAPL, MSFT, NVDA) and specific, detailed reasons.
    
    Text: {raw_text}
    
    Return as a JSON object with ticker symbols as keys and detailed reasons as values.
    Example format: {{
        "AAPL": "Strong iPhone 15 sales momentum, growing services revenue, expanding into AI with Apple Intelligence, solid cash position for buybacks",
        "MSFT": "Azure cloud growth accelerating, AI integration across products, strong enterprise adoption, OpenAI partnership driving innovation"
    }}
    
    Make the reasons specific and detailed (2-3 sentences) rather than generic phrases like "Top pick" or "Key stock".
    Only include real stock tickers, not words like "AI", "Tech", etc.
    Focus on concrete business drivers, financial metrics, or strategic advantages.
    """