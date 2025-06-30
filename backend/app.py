import json
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import StockDigestAgent

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StockDigestRequest(BaseModel):
    tickers: List[str]


@app.post("/api/stock-digest")
async def analyze_stocks(request: StockDigestRequest):
    try:
        # Create and initialize the stock digest agent
        agent = StockDigestAgent()

        # Run the stock digest workflow
        result = await agent.run_digest(request.tickers)
        
        # Convert the result to a dictionary for JSON serialization
        response_data = {
            "reports": {},
            "generated_at": result.generated_at,
            "market_overview": result.market_overview
        }
        
        # Convert each stock report to a dictionary
        for ticker, report in result.reports.items():
            response_data["reports"][ticker] = {
                "ticker": report.ticker,
                "company_name": report.company_name,
                "executive_summary": report.executive_summary,
                "current_performance": report.current_performance,
                "key_insights": report.key_insights,
                "recommendation": report.recommendation,
                "risk_assessment": report.risk_assessment,
                "price_outlook": report.price_outlook
            }
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)