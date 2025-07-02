export interface StockFinanceData {
  ticker: string;
  current_price: number;
  market_cap?: number;
  company_name: string;
}

export interface StockReport {
  ticker: string;
  company_name: string;
  summary: string;  // Step 1 from prompt: summary of most important insights
  current_performance: string;  // Step 2 from prompt
  key_insights: string[];  // Step 3 from prompt
  recommendation: string;  // Step 4 from prompt
  risk_assessment: string;  // Step 5 from prompt
  price_outlook: string;  // Step 6 from prompt
  sources: Source[];
  finance_data?: StockFinanceData;
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

export interface PDFData {
  pdf_base64: string;
  filename: string;
}

export interface StockDigestResponse {
  reports: Record<string, StockReport>;
  generated_at: string;
  market_overview: string;
  ticker_suggestions?: Record<string, string>;
  pdf_data?: PDFData;
}

export interface StockDigestRequest {
  tickers: string[];
} 