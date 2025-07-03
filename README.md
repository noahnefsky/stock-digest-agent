# Stock Digest Agent

A comprehensive stock analysis and digest generation system that provides detailed market insights, financial data, and research for your portfolio.

## Demo Video

https://user-images.githubusercontent.com/your-user-id/stock-digest-agent/main/Demo.mp4

## Setup and Installation

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 3000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables
Create a `.env` file in the backend directory:
```
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
VITE_BACKEND_URL=your_backend_url
POLYGON_API_KEY=your_polygon_api_key
BACKEND_PORT=3000
```

## Features

### Core Functionality
- **Real-time Financial Data**: Fetch current price and market cap. Due to free trial limits we can only do the day before's price.
- **Comprehensive Research**: Gather news and analysis from multiple financial sources using Tavily
- **AI-Powered Analysis**: Generate insights using Google's Gemini AI model
- **Targeted Research**: Perform comprehensive searches for earnings, analyst ratings, insider trading, technical analysis, and sector news
- **Market Overview**: Create comprehensive market summaries using LangChain's refine summarization chain
- **Stock Recommendations**: Research and extract current analyst picks and trending stocks
- **PDF Export**: Generate and download comprehensive PDF reports with professional formatting

### Targeted Research Categories
The system performs comprehensive searches for each ticker and categorizes results across these areas:

1. **Earnings News**: Quarterly results, revenue growth, profit margins, guidance
2. **Analyst Ratings**: Price targets, upgrades, downgrades, buy/sell recommendations
3. **Insider Trading**: SEC filings, executive stock transactions, Form 4 reports
4. **Technical Analysis**: Support/resistance levels, moving averages, RSI, MACD
5. **Sector News**: Industry trends, competitor analysis, regulatory updates

## Project Structure

```
stock-digest-agent/
├── backend/
│   ├── agent.py          # Main agent with LangGraph workflow
│   ├── app.py            # FastAPI server
│   ├── models.py         # Pydantic data models
│   ├── prompts.py        # AI prompt templates
│   ├── agent_utils.py    # PDF generation utilities
│   └── requirements.txt  # Python dependencies
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── DailyDigestReport.tsx  # Main report display
    │   │   └── TickerInput.tsx        # Stock input interface
    │   ├── types/
    │   │   └── stock-digest.ts        # TypeScript interfaces
    │   └── pages/
    │       └── Index.tsx              # Main landing page
    └── package.json
```

## Backend Architecture

### LangGraph Workflow
The system uses LangGraph to orchestrate a 7-step workflow with real-time progress tracking:

1. **StockMetrics Node**: 
   - Fetches financial data from Polygon.io (prices, volume, market cap, P/E ratios)
   - Implements rate limiting with 2-second delays between requests
   - Handles missing data gracefully with fallback values

2. **TargetedResearch Node**: 
   - Performs comprehensive keyword searches for each ticker using Tavily
   - Categorizes results into earnings, analyst ratings, insider trading, technical analysis, and sector news
   - Implements 3-second delays between searches for API compliance

3. **GeminiAnalysisFormatter Node**: 
   - Generates structured stock reports using Google's Gemini 1.5 Flash model
   - Creates comprehensive analysis with summary, performance, insights, recommendations, and risk assessment
   - Uses Pydantic models for consistent structured output

4. **MarketOverviewSummary Node**: 
   - Creates comprehensive market overview using LangChain's refine summarization chain
   - Aggregates all ticker data into a holistic market perspective
   - Combines financial metrics, analysis, and insights across all stocks

5. **StockRecommendationsResearch Node**: 
   - Searches for current stock recommendations and analyst picks using Tavily
   - Focuses on trending stocks, upgrades, and buy ratings
   - Targets financial news domains for quality recommendations

6. **RecommendationFormatting Node**: 
   - Uses Gemini to extract and format stock recommendations from research
   - Parses JSON responses to identify ticker symbols and reasoning
   - Validates ticker format and reasoning quality

7. **PDFGeneration Node**: 
   - Creates professional PDF reports using ReportLab
   - Includes financial tables, formatted text, and structured sections
   - Generates base64-encoded PDF for easy frontend integration

### Data Models
- `StockFinanceData`: Financial metrics and pricing data from Polygon.io
- `TargetedResearch`: Categorized research results from Tavily searches
- `StockReport`: Complete analysis with recommendations and insights
- `StockDigestOutput`: Complete digest with reports, market overview, and recommendations
- `PDFData`: Base64-encoded PDF with filename

## API Endpoint

### POST /api/stock-digest
Generate a complete stock digest for multiple tickers.
