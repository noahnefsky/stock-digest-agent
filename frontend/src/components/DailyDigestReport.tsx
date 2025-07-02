import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { ArrowLeft, TrendingUp, TrendingDown, DollarSign, BarChart3, Clock, Download, Target, AlertTriangle, ChevronDown, ExternalLink, Circle, Globe } from 'lucide-react';
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

  const downloadPDF = () => {
    if (stockDigest.pdf_data) {
      // Convert base64 to blob
      const byteCharacters = atob(stockDigest.pdf_data.pdf_base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'application/pdf' });

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = stockDigest.pdf_data.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 pb-12">
      <div className="container mx-auto px-4 pt-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6 relative">
          <Button
            onClick={onReset}
            variant="outline"
            className="flex items-center gap-2 hover:bg-white/80 transition-all duration-200 shadow-sm"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Input
          </Button>
          <div className="flex items-center gap-4 absolute left-1/2 transform -translate-x-1/2">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full shadow-lg">
              <BarChart3 className="h-7 w-7 text-white" />
            </div>
            <div className="text-center">
              <h1 className="text-2xl font-bold">
                <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Portfolio Digest for{' '}
                </span>
                <span className="text-blue-600">
                  {currentDate}
                </span>
              </h1>
            </div>
          </div>
          <Button
            variant="outline"
            className="flex items-center gap-2 hover:bg-white/80 transition-all duration-200 shadow-sm"
            onClick={downloadPDF}
            disabled={!stockDigest.pdf_data}
          >
            <Download className="h-4 w-4" />
            Export PDF
          </Button>
        </div>

        {/* Main Content with Tabs */}
        <Tabs defaultValue="your-stocks" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="your-stocks" className="text-base font-medium">
              <BarChart3 className="h-4 w-4 mr-2" />
              Your Stocks
            </TabsTrigger>
            <TabsTrigger value="market-overview" className="text-base font-medium">
              <Globe className="h-4 w-4 mr-2" />
              Market Overview
            </TabsTrigger>
          </TabsList>

          {/* Your Stocks Tab */}
          <TabsContent value="your-stocks" className="space-y-6">
            {/* Stock Selector */}
            <div className="w-1/3 mx-auto">
              <Card className="border-0 shadow-lg bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
                <CardContent className="p-3">
                  <Select value={selectedTicker} onValueChange={setSelectedTicker}>
                    <SelectTrigger className="w-full h-9 text-base bg-white/80 border-2 border-blue-200 hover:border-blue-300 focus:border-blue-500 transition-all duration-200 shadow-sm">
                      <SelectValue placeholder="Choose a ticker" />
                    </SelectTrigger>
                    <SelectContent className="bg-white/95 backdrop-blur-sm border border-blue-200 shadow-lg">
                      {tickers.map((ticker) => (
                        <SelectItem
                          key={ticker}
                          value={ticker}
                          className="hover:bg-blue-50 focus:bg-blue-50 transition-colors duration-200"
                        >
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full"></div>
                            <span className="font-medium">{ticker}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>
            </div>

            {/* Individual Stock Report */}
            {selectedStock && (
              <div className="space-y-8">
                {/* Stock Header Card */}
                <Card className="border-0 shadow-2xl bg-white/95 backdrop-blur-sm hover:shadow-3xl transition-all duration-300">
                  <CardHeader className="bg-gradient-to-r from-gray-50 via-blue-50 to-indigo-50 border-b border-gray-100">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div>
                          <CardTitle className="text-3xl font-bold text-gray-900">{selectedStock.ticker}</CardTitle>
                          <CardDescription className="text-xl text-gray-600">{selectedStock.company_name}</CardDescription>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-6">
                    {/* StockFinanceData Section */}
                    {selectedStock.finance_data && (
                      <div className="mb-8">
                        <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                          <DollarSign className="h-5 w-5 text-green-600" />
                          Financial Metrics
                        </h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                          {/* Price and Change */}
                          <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-xl border border-green-200 shadow-sm hover:shadow-md transition-all duration-200">
                            <div className="text-xs text-gray-600 font-medium mb-1">Current Price</div>
                            <div className='flex flex-row gap-2 items-center'>
                              <div className="text-xl font-bold text-gray-900">${selectedStock.finance_data.current_price?.toFixed(2)}</div>
                              <div className={`text-sm font-medium px-2 py-1 rounded-full ${selectedStock.finance_data.change_percent >= 0 ? 'text-green-700 bg-green-100' : 'text-red-700 bg-red-100'}`}>
                                {selectedStock.finance_data.change_percent >= 0 ? '+' : ''}{selectedStock.finance_data.change_percent?.toFixed(2)}%
                              </div>
                            </div>
                          </div>

                          {/* Market Cap */}
                          {selectedStock.finance_data.market_cap && (
                            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-200 shadow-sm hover:shadow-md transition-all duration-200">
                              <div className="text-xs text-gray-600 font-medium mb-1">Market Cap</div>
                              <div className="text-xl font-bold text-gray-900">
                                {selectedStock.finance_data.market_cap >= 1000000000000 ?
                                  `${(selectedStock.finance_data.market_cap / 1000000000000).toFixed(2)}T` :
                                  selectedStock.finance_data.market_cap >= 1000000000 ?
                                    `${(selectedStock.finance_data.market_cap / 1000000000).toFixed(2)}B` :
                                    `${(selectedStock.finance_data.market_cap / 1000000).toFixed(2)}M`
                                }
                              </div>
                            </div>
                          )}

                          {/* P/E Ratio */}
                          {selectedStock.finance_data.pe_ratio && (
                            <div className="bg-gradient-to-r from-purple-50 to-violet-50 p-4 rounded-xl border border-purple-200 shadow-sm hover:shadow-md transition-all duration-200">
                              <div className="text-xs text-gray-600 font-medium mb-1">P/E Ratio</div>
                              <div className="text-xl font-bold text-gray-900">{selectedStock.finance_data.pe_ratio.toFixed(2)}</div>
                            </div>
                          )}

                          {/* Volume */}
                          <div className="bg-gradient-to-r from-orange-50 to-amber-50 p-4 rounded-xl border border-orange-200 shadow-sm hover:shadow-md transition-all duration-200">
                            <div className="text-xs text-gray-600 font-medium mb-1">Volume</div>
                            <div className="text-xl font-bold text-gray-900">
                              {selectedStock.finance_data.volume >= 1000000 ?
                                `${(selectedStock.finance_data.volume / 1000000).toFixed(1)}M` :
                                selectedStock.finance_data.volume >= 1000 ?
                                  `${(selectedStock.finance_data.volume / 1000).toFixed(0)}K` :
                                  selectedStock.finance_data.volume.toLocaleString()
                              }
                            </div>
                          </div>

                          {/* Beta */}
                          {selectedStock.finance_data.beta && (
                            <div className="bg-gradient-to-r from-cyan-50 to-teal-50 p-4 rounded-xl border border-cyan-200 shadow-sm hover:shadow-md transition-all duration-200">
                              <div className="text-xs text-gray-600 font-medium mb-1">Beta</div>
                              <div className="text-xl font-bold text-gray-900">{selectedStock.finance_data.beta.toFixed(2)}</div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Stock Market Overview - Full Width at Top */}
                    <div className="bg-gradient-to-r from-indigo-50 via-blue-50 to-purple-50 p-6 rounded-xl border border-indigo-200 shadow-sm">
                      <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-3 text-lg">
                        <div className="p-2 bg-indigo-100 rounded-lg">
                          <BarChart3 className="h-5 w-5 text-indigo-600" />
                        </div>
                        Stock Overview
                      </h4>
                      <p className="text-gray-700 leading-relaxed text-base">{selectedStock.summary}</p>
                    </div>
                  </CardContent>
                </Card>

                {/* Key Insights - Not in a card */}
                <div className="bg-gradient-to-r from-purple-50 via-violet-50 to-fuchsia-50 p-6 rounded-xl border border-purple-200 shadow-lg">
                  <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-3 text-lg">
                    <div className="p-2 bg-purple-100 rounded-lg">
                      <BarChart3 className="h-5 w-5 text-purple-600" />
                    </div>
                    Key Insights
                  </h4>
                  {Array.isArray(selectedStock.key_insights) && selectedStock.key_insights.length > 0 ? (
                    <ul className="space-y-3">
                      {selectedStock.key_insights.map((insight, index) => (
                        <li key={index} className="text-gray-700 leading-relaxed flex items-center gap-3">
                          <Circle className="h-5 w-5 text-purple-600 flex-shrink-0" />
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
                  <Card className="border-0 shadow-xl bg-white/95 backdrop-blur-sm hover:shadow-2xl transition-all duration-300">
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-3 text-lg">
                        <div className="p-2 bg-green-100 rounded-lg">
                          <TrendingUp className="h-5 w-5 text-green-600" />
                        </div>
                        Current Performance
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-xl border border-green-200 shadow-sm">
                        <p className="text-gray-700 leading-relaxed">{selectedStock.current_performance}</p>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Risk Assessment */}
                  <Card className="border-0 shadow-xl bg-white/95 backdrop-blur-sm hover:shadow-2xl transition-all duration-300">
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-3 text-lg">
                        <div className="p-2 bg-red-100 rounded-lg">
                          <AlertTriangle className="h-5 w-5 text-red-600" />
                        </div>
                        Risk Assessment
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-gradient-to-r from-red-50 to-pink-50 p-4 rounded-xl border border-red-200 shadow-sm">
                        <p className="text-gray-700 leading-relaxed">{selectedStock.risk_assessment}</p>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Price Outlook */}
                  <Card className="border-0 shadow-xl bg-white/95 backdrop-blur-sm hover:shadow-2xl transition-all duration-300">
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-3 text-lg">
                        <div className="p-2 bg-yellow-100 rounded-lg">
                          <DollarSign className="h-5 w-5 text-yellow-600" />
                        </div>
                        Price Outlook
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-4 rounded-xl border border-yellow-200 shadow-sm">
                        <p className="text-gray-700 leading-relaxed">{selectedStock.price_outlook}</p>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Final Recommendation */}
                <div className="bg-gradient-to-r from-slate-100 via-gray-100 to-slate-200 p-6 rounded-xl border-2 border-dashed border-slate-300 shadow-lg">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-slate-200 rounded-lg">
                      <Target className="h-5 w-5 text-slate-600" />
                    </div>
                    <h4 className="text-xl font-semibold text-slate-800">Final Recommendation</h4>
                  </div>
                  <div className="bg-white/80 p-5 rounded-xl border border-slate-200 shadow-sm backdrop-blur-sm">
                    <p className="text-lg text-slate-800 font-medium leading-relaxed">{selectedStock.recommendation}</p>
                  </div>
                </div>

                {/* Sources Panel for this Stock */}
                <Card className="border-0 shadow-lg bg-white/95 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
                  <CardHeader>
                    <button
                      onClick={() => setShowSources(!showSources)}
                      className="flex items-center justify-between w-full text-left hover:bg-gray-50 p-2 rounded-lg transition-colors"
                    >
                      <CardTitle className="flex items-center gap-3 text-xl">
                        <div className="p-2 bg-blue-100 rounded-lg">
                          <ExternalLink className="h-5 w-5 text-blue-600" />
                        </div>
                        Research Sources ({selectedStock.sources?.length || 0})
                      </CardTitle>
                      <ChevronDown
                        className={`h-5 w-5 text-gray-600 transition-transform duration-200 ${showSources ? 'rotate-180' : ''
                          }`}
                      />
                    </button>
                  </CardHeader>
                  {showSources && (
                    <CardContent>
                      <div className="space-y-4">
                        <p className="text-gray-600 text-sm">
                          Sources used for {selectedStock.ticker} analysis, sorted by relevance and recency:
                        </p>
                        <div className="grid gap-4">
                          {selectedStock.sources && selectedStock.sources.length > 0 ? (
                            selectedStock.sources
                              .sort((a, b) => b.score - a.score) // Sort by relevance score
                              .map((source, index) => (
                                <div
                                  key={index}
                                  className="p-4 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors"
                                >
                                  <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                      <div className="flex items-center gap-2 mb-2">
                                        <span className="text-xs text-gray-500">
                                          {source.published_date}
                                        </span>
                                      </div>
                                      <h4 className="font-semibold text-gray-900 mb-1 line-clamp-2">
                                        {source.title}
                                      </h4>
                                      <p className="text-sm text-gray-600 mb-1">
                                        {source.source}
                                      </p>
                                      {source.url && (
                                        <p className="text-xs text-blue-600 mb-2 truncate">
                                          {source.url}
                                        </p>
                                      )}
                                    </div>
                                    {source.url && (
                                      <a
                                        href={source.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="ml-4 p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded-lg transition-colors flex-shrink-0"
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
                              <p>No sources available for {selectedStock.ticker} analysis.</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  )}
                </Card>
              </div>
            )}
          </TabsContent>

          {/* Market Overview Tab */}
          <TabsContent value="market-overview" className="space-y-6">
            {/* Market Overview */}
            {stockDigest.market_overview && (
              <Card className="border-0 shadow-lg bg-gradient-to-r from-blue-50 to-indigo-50 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Globe className="h-5 w-5 text-blue-600" />
                    </div>
                    <div>
                      <CardTitle className="text-lg font-semibold text-gray-900">Market Overview</CardTitle>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pb-4">
                  <div className="bg-white/80 p-4 rounded-lg border border-blue-200">
                    <p className="text-gray-800 leading-relaxed text-base">{stockDigest.market_overview}</p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Stock Recommendations */}
            {stockDigest.ticker_suggestions && Object.keys(stockDigest.ticker_suggestions).length > 0 && (
              <Card className="border-0 shadow-lg bg-gradient-to-r from-green-50 to-emerald-50 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Target className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <CardTitle className="text-lg font-semibold text-gray-900">Stock Recommendations</CardTitle>
                      <CardDescription className="text-sm text-gray-600">
                        Current analyst picks and trending stocks
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pb-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(stockDigest.ticker_suggestions).map(([ticker, reason]) => (
                      <div
                        key={ticker}
                        className="bg-white/80 p-4 rounded-lg border border-green-200 hover:bg-white/90 transition-all duration-200 shadow-sm"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-gradient-to-r from-green-500 to-emerald-600 rounded-full"></div>
                            <span className="font-bold text-lg text-gray-900">{ticker}</span>
                          </div>
                          <Badge variant="secondary" className="bg-green-100 text-green-800 text-xs">
                            Recommended
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-700 leading-relaxed">{reason}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>


      </div>
    </div>
  );
};
