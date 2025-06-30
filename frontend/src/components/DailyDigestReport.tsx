import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ArrowLeft, TrendingUp, TrendingDown, DollarSign, BarChart3, Clock, Download, Target, AlertTriangle, ChevronDown, ExternalLink } from 'lucide-react';
import { StockDigestResponse } from '@/types/stock-digest';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface DailyDigestReportProps {
  tickers: string[];
  stockDigest: StockDigestResponse;
  onReset: () => void;
}

export const DailyDigestReport: React.FC<DailyDigestReportProps> = ({
  tickers,
  onReset,
  stockDigest
}) => {
  const [selectedTicker, setSelectedTicker] = useState<string>(tickers[0] || '');
  const [showSources, setShowSources] = useState(false);

  const currentDate = new Date(stockDigest.generated_at).toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  const stockReports = Object.entries(stockDigest.reports).map(([ticker, report]) => ({
    ticker,
    ...report
  }));

  const selectedStock = stockReports.find(stock => stock.ticker === selectedTicker);

  // Calculate summary statistics
  const buyRecommendations = stockReports.filter(stock =>
    stock.recommendation.toLowerCase().includes('buy')
  ).length;
  const sellRecommendations = stockReports.filter(stock =>
    stock.recommendation.toLowerCase().includes('sell')
  ).length;
  const holdRecommendations = stockReports.filter(stock =>
    !stock.recommendation.toLowerCase().includes('buy') &&
    !stock.recommendation.toLowerCase().includes('sell')
  ).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Button
            onClick={onReset}
            variant="outline"
            className="flex items-center gap-2 hover:bg-white/80"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Input
          </Button>
          <Button variant="outline" className="flex items-center gap-2 hover:bg-white/80">
            <Download className="h-4 w-4" />
            Export Report
          </Button>
        </div>

        {/* Report Title */}
        <div className="text-center mb-2">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="p-3 bg-blue-100 rounded-full">
              <BarChart3 className="h-8 w-8 text-blue-600" />
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Portfolio Update Report
            </h1>
          </div>
          <p className="text-xl text-gray-600 flex items-center justify-center gap-2">
            <Clock className="h-6 w-6" />
            {currentDate}
          </p>
        </div>

        {/* Stock Selector */}
        <div className="mb-12">
          {/* <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Stock Analysis</h2>
            <p className="text-gray-600">Select a stock to view detailed analysis</p>
          </div> */}
          
          <div className="max-w-md mx-auto">
            <Card className="border-0 shadow-xl bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <BarChart3 className="h-5 w-5 text-blue-600" />
                  </div>
                  Select Stock
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Select value={selectedTicker} onValueChange={setSelectedTicker}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Choose a stock ticker" />
                  </SelectTrigger>
                  <SelectContent>
                    {tickers.map((ticker) => (
                      <SelectItem key={ticker} value={ticker}>
                        {ticker}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Market Overview Card */}
        {/* <div className="mb-12">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Market Overview</h2>
            <p className="text-gray-600">Current market conditions and trends</p>
          </div>
          
          <Card className="border-0 shadow-xl bg-white/95 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-2xl">
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <BarChart3 className="h-6 w-6 text-indigo-600" />
                </div>
                Overall Market Context
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-gradient-to-r from-indigo-50 to-blue-50 p-6 rounded-lg border border-indigo-100">
                <p className="text-gray-700 leading-relaxed text-lg">
                  {stockDigest.market_overview}
                </p>
              </div>
            </CardContent>
          </Card>
        </div> */}

        {/* Individual Stock Report */}
        {selectedStock && (
          <div className="mb-12">
            
            <div className="space-y-8">
              {/* Stock Header Card */}
              <Card className="border-0 shadow-xl bg-white/95 backdrop-blur-sm hover:shadow-2xl transition-all duration-300">
                <CardHeader className="bg-gradient-to-r from-gray-50 to-blue-50 border-b border-gray-100">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="p-3 bg-blue-100 rounded-full">
                        <TrendingUp className="h-6 w-6 text-blue-600" />
                      </div>
                      <div>
                        <CardTitle className="text-2xl font-bold text-gray-900">{selectedStock.ticker}</CardTitle>
                        <CardDescription className="text-lg text-gray-600">{selectedStock.company_name}</CardDescription>
                      </div>
                    </div>
                    <div className="text-right">
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="p-6">
                  {/* Stock Market Overview - Full Width at Top */}
                  <div className="bg-gradient-to-r from-indigo-50 to-blue-50 p-4 rounded-lg border border-indigo-100">
                    <h4 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-indigo-600" />
                      Market Context
                    </h4>
                    <p className="text-gray-700 leading-relaxed">{selectedStock.stock_market_overview}</p>
                  </div>
                </CardContent>
              </Card>

              {/* Key Insights - Not in a card */}
              <div className="bg-gradient-to-r from-purple-50 to-violet-50 p-6 rounded-lg border border-purple-100 shadow-lg">
                <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2 text-lg">
                  <BarChart3 className="h-5 w-5 text-purple-600" />
                  Key Insights
                </h4>
                {Array.isArray(selectedStock.key_insights) && selectedStock.key_insights.length > 0 ? (
                  <ul className="space-y-3">
                    {selectedStock.key_insights.map((insight, index) => (
                      <li key={index} className="text-gray-700 leading-relaxed flex items-start gap-3">
                        <span className="text-purple-600 font-bold mt-1 text-lg">•</span>
                        <span className="text-base">{insight}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-700 leading-relaxed text-base">{selectedStock.key_insights}</p>
                )}
              </div>

              {/* Analysis Grid */}
              <div className="grid lg:grid-cols-3 gap-6">
                {/* Current Performance */}
                <Card className="border-0 shadow-lg bg-white/95 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <div className="p-2 bg-green-100 rounded-lg">
                        <TrendingUp className="h-4 w-4 text-green-600" />
                      </div>
                      Current Performance
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg border border-green-100">
                      <p className="text-gray-700 leading-relaxed">{selectedStock.current_performance}</p>
                    </div>
                  </CardContent>
                </Card>

                {/* Risk Assessment */}
                <Card className="border-0 shadow-lg bg-white/95 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <div className="p-2 bg-red-100 rounded-lg">
                        <AlertTriangle className="h-4 w-4 text-red-600" />
                      </div>
                      Risk Assessment
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-gradient-to-r from-red-50 to-pink-50 p-4 rounded-lg border border-red-100">
                      <p className="text-gray-700 leading-relaxed">{selectedStock.risk_assessment}</p>
                    </div>
                  </CardContent>
                </Card>

                {/* Price Outlook */}
                <Card className="border-0 shadow-lg bg-white/95 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <div className="p-2 bg-yellow-100 rounded-lg">
                        <DollarSign className="h-4 w-4 text-yellow-600" />
                      </div>
                      Price Outlook
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-4 rounded-lg border border-yellow-100">
                      <p className="text-gray-700 leading-relaxed">{selectedStock.price_outlook}</p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Final Recommendation */}
              <Card className="border-0 shadow-lg bg-white/95 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
                <CardHeader>
                  <CardTitle className="flex items-center gap-3 text-xl">
                    <div className="p-2 bg-gray-100 rounded-lg">
                      <Target className="h-5 w-5 text-gray-600" />
                    </div>
                    Final Recommendation
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="bg-gradient-to-r from-gray-50 to-slate-50 p-6 rounded-lg border border-gray-200">
                    <div className="flex items-center gap-3">
                      <span className="text-lg font-semibold text-gray-900">{selectedStock.recommendation}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Sources Panel */}
        <div className="mt-16">
          <Card className="border-0 shadow-xl bg-white/95 backdrop-blur-sm">
            <CardHeader>
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center justify-between w-full text-left hover:bg-gray-50 p-2 rounded-lg transition-colors"
              >
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-gray-100 rounded-lg">
                    <BarChart3 className="h-5 w-5 text-gray-600" />
                  </div>
                  Research Sources ({stockDigest.sources?.length || 0})
                </CardTitle>
                <ChevronDown 
                  className={`h-5 w-5 text-gray-600 transition-transform duration-200 ${
                    showSources ? 'rotate-180' : ''
                  }`} 
                />
              </button>
            </CardHeader>
            {showSources && (
              <CardContent>
                <div className="space-y-4">
                  <p className="text-gray-600 text-sm">
                    Sources used for this analysis, sorted by relevance and recency:
                  </p>
                  <div className="grid gap-4">
                    {stockDigest.sources && stockDigest.sources.length > 0 ? (
                      stockDigest.sources
                        .sort((a, b) => b.score - a.score) // Sort by relevance score
                        .map((source, index) => (
                          <div
                            key={index}
                            className="bg-gradient-to-r from-gray-50 to-blue-50 p-4 rounded-lg border border-gray-200 hover:shadow-md transition-shadow"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <Badge variant="outline" className="text-xs">
                                    {source.ticker}
                                  </Badge>
                                  <Badge variant="secondary" className="text-xs">
                                    Score: {source.score.toFixed(1)}
                                  </Badge>
                                  <span className="text-xs text-gray-500">
                                    {source.published_date}
                                  </span>
                                </div>
                                <h4 className="font-semibold text-gray-900 mb-1 truncate">
                                  {source.title}
                                </h4>
                                <p className="text-sm text-gray-600 mb-2">
                                  {source.source} • {source.domain}
                                </p>
                              </div>
                              {source.url && (
                                <a
                                  href={source.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="ml-4 p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded-lg transition-colors"
                                  title="Open source"
                                >
                                  <ExternalLink className="h-4 w-4" />
                                </a>
                              )}
                            </div>
                          </div>
                        ))
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <p>No sources available for this analysis.</p>
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};
