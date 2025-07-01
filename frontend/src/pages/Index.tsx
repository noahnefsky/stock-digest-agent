import React, { useState } from 'react';
import { TickerInput } from '@/components/TickerInput';
import { DailyDigestReport } from '@/components/DailyDigestReport';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, BarChart3, DollarSign } from 'lucide-react';
import { StockDigestResponse } from '@/types/stock-digest';

const Index = () => {
  const [tickers, setTickers] = useState<string[]>([]);
  const [stockDigest, setStockDigest] = useState<StockDigestResponse | null>(null);
  const [showReport, setShowReport] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerateReport = async () => {
    if (tickers.length === 0) return;

    setIsGenerating(true);

    const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/api/stock-digest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ tickers }),
    });
    setStockDigest(await response.json());
    setIsGenerating(false);
    setShowReport(true);
  };

  const handleReset = () => {
    setShowReport(false);
    setTickers([]);
  };

  if (showReport) {
    return <DailyDigestReport tickers={tickers} onReset={handleReset} stockDigest={stockDigest} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-3 rounded-full">
              <TrendingUp className="h-8 w-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Daily Digest Generator
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Get comprehensive market insights and analysis for your favorite stocks. 
            Enter ticker symbols below to generate your personalized daily digest.
          </p>
        </div>

        {/* Main Input Card */}
        <Card className="max-w-2xl mx-auto border-0 shadow-xl bg-white/90 backdrop-blur-sm mb-12">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">Enter Stock Tickers</CardTitle>
            <CardDescription className="text-base">
              Add the stock symbols you want to analyze (e.g., AAPL, GOOGL, MSFT)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <TickerInput 
              tickers={tickers} 
              onTickersChange={setTickers}
              onGenerateReport={handleGenerateReport}
              isGenerating={isGenerating}
            />
          </CardContent>
        </Card>

        {/* Features Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <Card className="text-center border-0 shadow-lg bg-white/80 backdrop-blur-sm">
            <CardHeader>
              <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2">
                <BarChart3 className="h-6 w-6 text-blue-600" />
              </div>
              <CardTitle className="text-lg">Market Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Get detailed price movements, volume analysis, and trend indicators
              </CardDescription>
            </CardContent>
          </Card>

          <Card className="text-center border-0 shadow-lg bg-white/80 backdrop-blur-sm">
            <CardHeader>
              <div className="bg-green-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2">
                <DollarSign className="h-6 w-6 text-green-600" />
              </div>
              <CardTitle className="text-lg">Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Track daily returns, volatility, and key performance indicators
              </CardDescription>
            </CardContent>
          </Card>

          <Card className="text-center border-0 shadow-lg bg-white/80 backdrop-blur-sm">
            <CardHeader>
              <div className="bg-purple-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2">
                <TrendingUp className="h-6 w-6 text-purple-600" />
              </div>
              <CardTitle className="text-lg">Trend Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Identify market trends and get actionable insights for your portfolio
              </CardDescription>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Index;
