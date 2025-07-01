# Stock Digest Agent

A comprehensive stock analysis and digest generation system that provides detailed market insights, financial data, and research for your portfolio.

## Features

### Core Functionality
- **Real-time Financial Data**: Fetch current prices, volume, market cap, and P/E ratios from Polygon.io
- **Comprehensive Research**: Gather news and analysis from multiple financial sources
- **AI-Powered Analysis**: Generate insights using Google's Gemini AI model
- **Targeted Research**: Perform specific searches for earnings, analyst ratings, insider trading, technical analysis, and sector news
- **PDF Export**: Generate and download comprehensive PDF reports

### Targeted Research Categories
The system performs targeted searches for each ticker across these categories:

1. **Earnings News**: Quarterly results, revenue growth, profit margins, guidance
2. **Analyst Ratings**: Price targets, upgrades, downgrades, buy/sell recommendations
3. **Insider Trading**: SEC filings, executive stock transactions, Form 4 reports
4. **Technical Analysis**: Support/resistance levels, moving averages, RSI, MACD
5. **Sector News**: Industry trends, competitor analysis, regulatory updates

## Project Structure

```
stock-digest-agent/
├── backend/
│   ├── agent.py          # Main agent with graph workflow
│   ├── app.py            # FastAPI server
│   ├── prompts.py        # AI prompt templates
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

### Graph Workflow
The system uses LangGraph to orchestrate a multi-step workflow:

1. **PolygonFinance Node**: Fetch financial data from Polygon.io
2. **TargetedResearch Node**: Perform specific keyword searches for each ticker
3. **Research Node**: General news and analysis gathering
4. **GeminiAnalysis Node**: AI-powered report generation
5. **PDFGeneration Node**: Create downloadable PDF reports

### Data Models
- `StockFinanceData`: Financial metrics and pricing data
- `TargetedResearch`: Categorized research results
- `StockReport`: Complete analysis with recommendations
- `PDFData`: Base64-encoded PDF with filename

## Frontend Features

### User Interface
- **Modern Design**: Clean, responsive interface with gradient backgrounds
- **Stock Selector**: Dropdown to switch between different stocks in your portfolio
- **Interactive Cards**: Hover effects and smooth transitions
- **Real-time Updates**: Progress indicators during report generation

### PDF Export
- **One-click Download**: Export comprehensive reports as PDF
- **Professional Formatting**: Tables, charts, and structured content
- **Complete Data**: Includes all financial metrics, insights, and research sources

## Setup and Installation

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 3000 or python app.py
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
      "finance_data": {...}
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
- Polygon.io: 15-second delays between requests
- Tavily: 2-second delays between searches, 5-second delays between tickers
- Optimized to stay within API limits

### PDF Generation
- Uses ReportLab library for professional PDF creation
- Includes financial tables, formatted text, and structured sections
- Base64 encoded for easy frontend integration

### Error Handling
- Graceful fallbacks for missing data
- Comprehensive logging for debugging
- User-friendly error messages

## Contributing

1. Follow the existing code style and patterns
2. Add meaningful comments for complex logic
3. Update documentation for new features
4. Test thoroughly before submitting changes

## License

This project is licensed under the MIT License. 