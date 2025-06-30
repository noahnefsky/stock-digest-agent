
import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ArrowLeft, TrendingUp, TrendingDown, DollarSign, BarChart3, Clock, Download } from 'lucide-react';
import { generateMockData } from '@/utils/mockData';

interface DailyDigestReportProps {
  tickers: string[];
  onReset: () => void;
}

export const DailyDigestReport: React.FC<DailyDigestReportProps> = ({
  tickers,
  onReset
}) => {
  const reportData = generateMockData(tickers);
  const currentDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  const topGainer = reportData.reduce((prev, current) => 
    (prev.changePercent > current.changePercent) ? prev : current
  );

  const topLoser = reportData.reduce((prev, current) => 
    (prev.changePercent < current.changePercent) ? prev : current
  );

  const averageChange = reportData.reduce((sum, stock) => sum + stock.changePercent, 0) / reportData.length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Button 
            onClick={onReset} 
            variant="outline"
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Input
          </Button>
          <Button variant="outline" className="flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export Report
          </Button>
        </div>

        {/* Report Title */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Daily Digest Report</h1>
          <p className="text-lg text-gray-600 flex items-center justify-center gap-2">
            <Clock className="h-5 w-5" />
            {currentDate}
          </p>
        </div>

        {/* Summary Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Portfolio Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-900">{tickers.length} Stocks</div>
              <p className="text-sm text-gray-600">Average Change: 
                <span className={`ml-1 font-semibold ${averageChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {averageChange >= 0 ? '+' : ''}{averageChange.toFixed(2)}%
                </span>
              </p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Top Performer</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{topGainer.ticker}</div>
              <p className="text-sm text-gray-600">
                +{topGainer.changePercent.toFixed(2)}% 
                <span className="ml-1">(${topGainer.change.toFixed(2)})</span>
              </p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">Biggest Decline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{topLoser.ticker}</div>
              <p className="text-sm text-gray-600">
                {topLoser.changePercent.toFixed(2)}% 
                <span className="ml-1">(${topLoser.change.toFixed(2)})</span>
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Individual Stock Cards */}
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Individual Stock Analysis</h2>
          
          {reportData.map((stock) => (
            <Card key={stock.ticker} className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-xl">{stock.ticker}</CardTitle>
                    <CardDescription>{stock.companyName}</CardDescription>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold">${stock.currentPrice.toFixed(2)}</div>
                    <div className={`flex items-center gap-1 ${stock.changePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {stock.changePercent >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                      <span className="font-semibold">
                        {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                      </span>
                      <span className="text-sm">
                        (${stock.change.toFixed(2)})
                      </span>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Open</p>
                    <p className="font-semibold">${stock.openPrice.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">High</p>
                    <p className="font-semibold">${stock.dayHigh.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Low</p>
                    <p className="font-semibold">${stock.dayLow.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Volume</p>
                    <p className="font-semibold">{stock.volume.toLocaleString()}</p>
                  </div>
                </div>
                
                <Separator className="my-4" />
                
                <div>
                  <p className="text-sm text-gray-600 mb-2">Market Sentiment</p>
                  <Badge variant={stock.sentiment === 'Bullish' ? 'default' : stock.sentiment === 'Bearish' ? 'destructive' : 'secondary'}>
                    {stock.sentiment}
                  </Badge>
                </div>

                <div className="mt-4">
                  <p className="text-sm text-gray-600 mb-2">Key Insights</p>
                  <p className="text-sm text-gray-800">{stock.insight}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Market Summary */}
        <Card className="mt-8 border-0 shadow-lg bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Market Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-700 leading-relaxed">
              Today's trading session showed mixed results across your portfolio. 
              The technology sector demonstrated resilience with {reportData.filter(s => s.changePercent > 0).length} of your stocks posting gains. 
              Market volatility remains elevated, suggesting continued caution is warranted. 
              Consider monitoring upcoming earnings announcements and economic indicators that may impact your holdings.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
