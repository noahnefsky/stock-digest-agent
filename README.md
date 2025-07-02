# Stock Digest Agent

A comprehensive stock analysis and digest generation system that provides detailed market insights, financial data, and research for your portfolio.

## Features

### Core Functionality
- **Real-time Financial Data**: Fetch current prices, volume, market cap, and P/E ratios from Polygon.io
- **Comprehensive Research**: Gather news and analysis from multiple financial sources using Tavily
- **AI-Powered Analysis**: Generate insights using Google's Gemini AI model
- **Targeted Research**: Perform comprehensive searches for earnings, analyst ratings, insider trading, technical analysis, and sector news
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
│   ├── pdf_utils.py      # PDF generation utilities
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
The system uses LangGraph to orchestrate a multi-step workflow:

1. **StockMetrics Node**: Fetch financial data from Polygon.io with rate limiting (2-second delays)
2. **TargetedResearch Node**: Perform comprehensive keyword searches for each ticker using Tavily
3. **GeminiAnalysisFormatter Node**: AI-powered report generation with structured output using Gemini
4. **MarketOverviewSummary Node**: Create comprehensive market overview using LangChain's refine summarization chain
5. **PDFGeneration Node**: Create downloadable PDF reports using ReportLab

### Data Models
- `StockFinanceData`: Financial metrics and pricing data from Polygon.io
- `TargetedResearch`: Categorized research results from Tavily searches
- `StockReport`: Complete analysis with recommendations and insights
- `StockDigestOutput`: Complete digest with reports and market overview
- `PDFData`: Base64-encoded PDF with filename

### Key Features
- **Rate Limiting**: Optimized delays to stay within API limits (2-3 seconds between requests)
- **Error Handling**: Graceful fallbacks for missing data
- **Structured Output**: Uses Pydantic models for consistent data structure
- **Real-time Progress**: Custom event dispatching for frontend updates
- **Market Overview Generation**: Uses LangChain's refine summarization chain to create comprehensive market insights

## Frontend Features

### User Interface
- **Modern Design**: Clean, responsive interface with gradient backgrounds
- **Stock Selector**: Dropdown to switch between different stocks in your portfolio
- **Interactive Cards**: Hover effects and smooth transitions
- **Real-time Updates**: Progress indicators during report generation via custom events

### PDF Export
- **One-click Download**: Export comprehensive reports as PDF
- **Professional Formatting**: Tables, charts, and structured content using ReportLab
- **Complete Data**: Includes all financial metrics, insights, and research sources

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

## API Endpoints

### POST /api/stock-digest
Generate a complete stock digest for multiple tickers.

**Request:**
```json
{
  "tickers": ["AAPL", "GOOGL", "MSFT"]
}
```

**Response:**
```json
{
  "reports": {
    "AAPL": {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "summary": "...",
      "current_performance": "...",
      "key_insights": ["..."],
      "recommendation": "Buy",
      "risk_assessment": "...",
      "price_outlook": "...",
      "sources": [...],
      "finance_data": {
        "ticker": "AAPL",
        "current_price": 150.25,
        "previous_close": 148.50,
        "change_percent": 1.18,
        "volume": 50000000,
        "market_cap": 2500000000000,
        "pe_ratio": 25.5,
        "company_name": "Apple Inc.",
        "beta": null
      }
    }
  },
  "generated_at": "2024-01-15T10:30:00",
  "market_overview": "...",
  "pdf_data": {
    "pdf_base64": "base64_encoded_pdf_content",
    "filename": "stock_digest_20240115_103000.pdf"
  }
}
```

## Usage

1. **Enter Tickers**: Add stock symbols to analyze (up to 20 tickers)
2. **Generate Report**: Click "Get Daily Digest" to start analysis
3. **Review Results**: Browse through individual stock reports
4. **Export PDF**: Download a comprehensive PDF report
5. **View Sources**: Expand the sources section to see research references

## Technical Details

### Rate Limiting
- Polygon.io: 2-second delays between requests for free tier compliance
- Tavily: 3-second delays between searches
- Optimized to stay within API limits while maintaining performance

### PDF Generation
- Uses ReportLab library for professional PDF creation
- Includes financial tables, formatted text, and structured sections
- Base64 encoded for easy frontend integration
- Professional styling with custom fonts and colors

### Error Handling
- Graceful fallbacks for missing data from APIs
- Comprehensive logging for debugging
- User-friendly error messages
- Robust data validation with Pydantic models

### AI Analysis
- Uses Google's Gemini 1.5 Flash model for analysis
- Structured output using Pydantic models
- Portfolio-focused prompts for actionable insights
- Market overview generation using LangChain's refine summarization chain
- Comprehensive aggregation of all ticker data for holistic market perspective

## Contributing

1. Follow the existing code style and patterns
2. Add meaningful comments for complex logic
3. Update documentation for new features
4. Test thoroughly before submitting changes
5. Ensure proper error handling and rate limiting

## License

This project is licensed under the MIT License. 