
interface StockData {
  ticker: string;
  companyName: string;
  currentPrice: number;
  openPrice: number;
  dayHigh: number;
  dayLow: number;
  change: number;
  changePercent: number;
  volume: number;
  sentiment: 'Bullish' | 'Bearish' | 'Neutral';
  insight: string;
}

const companyNames: Record<string, string> = {
  'AAPL': 'Apple Inc.',
  'GOOGL': 'Alphabet Inc.',
  'MSFT': 'Microsoft Corporation',
  'AMZN': 'Amazon.com Inc.',
  'TSLA': 'Tesla Inc.',
  'META': 'Meta Platforms Inc.',
  'NVDA': 'NVIDIA Corporation',
  'NFLX': 'Netflix Inc.',
  'AMD': 'Advanced Micro Devices Inc.',
  'BABA': 'Alibaba Group Holding Limited',
  'CRM': 'Salesforce Inc.',
  'ORCL': 'Oracle Corporation',
  'IBM': 'International Business Machines Corporation',
  'INTC': 'Intel Corporation',
  'ADBE': 'Adobe Inc.',
};

const insights = [
  "Strong institutional buying detected in the last trading session, indicating positive momentum.",
  "Technical indicators suggest potential resistance at current levels, monitor for breakout patterns.",
  "Volume analysis shows increased retail interest, suggesting growing market confidence.",
  "Recent earnings beat expectations, driving optimistic analyst revisions.",
  "Sector rotation may be impacting price action, consider broader market trends.",
  "Options flow indicates bullish sentiment among professional traders.",
  "Support levels holding well despite market volatility, showing strong fundamental backing.",
  "Relative strength compared to sector peers suggests outperformance potential.",
  "Recent insider buying activity may signal confidence in future prospects.",
  "Elevated short interest could lead to squeeze potential on positive news.",
];

export const generateMockData = (tickers: string[]): StockData[] => {
  return tickers.map(ticker => {
    // Generate realistic base price (between $20 and $300)
    const basePrice = Math.random() * 280 + 20;
    
    // Generate daily change (-5% to +5%)
    const changePercent = (Math.random() - 0.5) * 10;
    const change = basePrice * (changePercent / 100);
    const currentPrice = basePrice + change;
    
    // Generate open price (within 2% of current)
    const openPrice = currentPrice + (Math.random() - 0.5) * currentPrice * 0.04;
    
    // Generate high and low
    const dayRange = currentPrice * 0.05; // 5% range
    const dayHigh = Math.max(currentPrice, openPrice) + Math.random() * dayRange;
    const dayLow = Math.min(currentPrice, openPrice) - Math.random() * dayRange;
    
    // Generate volume (1M to 50M)
    const volume = Math.floor(Math.random() * 49000000) + 1000000;
    
    // Determine sentiment based on change
    let sentiment: 'Bullish' | 'Bearish' | 'Neutral';
    if (changePercent > 2) sentiment = 'Bullish';
    else if (changePercent < -2) sentiment = 'Bearish';
    else sentiment = 'Neutral';
    
    // Random insight
    const insight = insights[Math.floor(Math.random() * insights.length)];
    
    // Get company name or create generic one
    const companyName = companyNames[ticker] || `${ticker} Corporation`;
    
    return {
      ticker,
      companyName,
      currentPrice,
      openPrice,
      dayHigh,
      dayLow,
      change,
      changePercent,
      volume,
      sentiment,
      insight,
    };
  });
};
