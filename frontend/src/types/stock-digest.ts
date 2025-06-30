export interface StockReport {
  ticker: string;
  company_name: string;
  stock_market_overview: string;
  current_performance: string;
  key_insights: string[];
  recommendation: string;
  risk_assessment: string;
  price_outlook: string;
}

export interface Source {
  ticker: string;
  title: string;
  url: string;
  source: string;
  domain: string;
  published_date: string;
  score: number;
}

export interface StockDigestResponse {
  reports: Record<string, StockReport>;
  generated_at: string;
  market_overview: string;
  sources: Source[];
}

export interface StockDigestRequest {
  tickers: string[];
} 