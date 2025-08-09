'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Activity, TrendingUp, TrendingDown, BarChart3, Clock, Zap, DollarSign, Users } from 'lucide-react';

interface StockData {
  ticker: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: string;
  pe: number;
  high: number;
  low: number;
  open: number;
  prevClose: number;
}

interface OrderBook {
  bids: Array<{ price: number; volume: number }>;
  asks: Array<{ price: number; volume: number }>;
}

interface Tick {
  price: number;
  volume: number;
  time: Date;
  type: 'buy' | 'sell';
}

export default function RealTimeStockDashboard() {
  const chartCanvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [timeFrame, setTimeFrame] = useState('1D');
  const [chartData, setChartData] = useState<Array<{ time: number; price: number; volume: number }>>([]);
  const [recentTicks, setRecentTicks] = useState<Tick[]>([]);
  
  const [stockData] = useState<StockData>({
    ticker: 'AAPL',
    name: 'Apple Inc.',
    price: 178.45,
    change: 2.35,
    changePercent: 1.33,
    volume: 52847293,
    marketCap: '2.82T',
    pe: 29.5,
    high: 179.83,
    low: 176.54,
    open: 177.20,
    prevClose: 176.10
  });

  const [orderBook] = useState<OrderBook>({
    bids: [
      { price: 178.44, volume: 1200 },
      { price: 178.43, volume: 2500 },
      { price: 178.42, volume: 3200 },
      { price: 178.41, volume: 1800 },
      { price: 178.40, volume: 4100 }
    ],
    asks: [
      { price: 178.45, volume: 1500 },
      { price: 178.46, volume: 2200 },
      { price: 178.47, volume: 3500 },
      { price: 178.48, volume: 2800 },
      { price: 178.49, volume: 4500 }
    ]
  });

  const [watchlist] = useState([
    { ticker: 'AAPL', price: 178.45, change: 1.33 },
    { ticker: 'MSFT', price: 420.82, change: 0.85 },
    { ticker: 'GOOGL', price: 142.65, change: -0.42 },
    { ticker: 'AMZN', price: 170.29, change: 2.15 },
    { ticker: 'NVDA', price: 850.42, change: 3.24 }
  ]);

  const [aiPrediction] = useState({
    direction: 'up',
    confidence: 72,
    targetPrice: 182.50,
    signals: ['강한 매수 신호', '골든크로스 형성', '거래량 급증']
  });

  useEffect(() => {
    // Generate initial chart data
    const data = [];
    const basePrice = 176;
    const now = Date.now();
    
    for (let i = 0; i < 390; i++) { // 6.5 hours of trading
      const time = now - (390 - i) * 60000; // 1 minute intervals
      const randomWalk = (Math.random() - 0.5) * 0.5;
      const trend = i * 0.006; // Slight upward trend
      const price = basePrice + trend + randomWalk;
      const volume = Math.floor(50000 + Math.random() * 200000);
      
      data.push({ time, price, volume });
    }
    
    setChartData(data);
  }, []);

  useEffect(() => {
    drawChart();
  }, [chartData]);

  useEffect(() => {
    // Simulate real-time ticks
    const interval = setInterval(() => {
      const newTick: Tick = {
        price: stockData.price + (Math.random() - 0.5) * 0.1,
        volume: Math.floor(100 + Math.random() * 1000),
        time: new Date(),
        type: Math.random() > 0.5 ? 'buy' : 'sell'
      };
      
      setRecentTicks(prev => [newTick, ...prev.slice(0, 19)]);
    }, 1000);

    return () => clearInterval(interval);
  }, [stockData.price]);

  const drawChart = () => {
    const canvas = chartCanvasRef.current;
    if (!canvas || chartData.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    
    ctx.clearRect(0, 0, width, height);

    // Find price range
    const prices = chartData.map(d => d.price);
    const minPrice = Math.min(...prices) - 0.5;
    const maxPrice = Math.max(...prices) + 0.5;
    const priceRange = maxPrice - minPrice;

    // Draw price line
    ctx.strokeStyle = stockData.change >= 0 ? '#10b981' : '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    chartData.forEach((point, index) => {
      const x = padding + (index / (chartData.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((point.price - minPrice) / priceRange) * (height - 2 * padding);
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw volume bars
    const maxVolume = Math.max(...chartData.map(d => d.volume));
    ctx.fillStyle = 'rgba(59, 130, 246, 0.3)';
    
    chartData.forEach((point, index) => {
      const x = padding + (index / (chartData.length - 1)) * (width - 2 * padding);
      const barHeight = (point.volume / maxVolume) * (height * 0.2);
      const barWidth = (width - 2 * padding) / chartData.length;
      
      ctx.fillRect(x - barWidth / 2, height - padding - barHeight, barWidth, barHeight);
    });

    // Draw axes
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Draw price labels
    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 5; i++) {
      const price = minPrice + (i / 5) * priceRange;
      const y = height - padding - (i / 5) * (height - 2 * padding);
      ctx.fillText(`$${price.toFixed(2)}`, padding - 10, y + 4);
    }

    // Draw current price line
    const currentY = height - padding - ((stockData.price - minPrice) / priceRange) * (height - 2 * padding);
    ctx.strokeStyle = '#3b82f6';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(padding, currentY);
    ctx.lineTo(width - padding, currentY);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  const calculateTotalVolume = (orders: Array<{ price: number; volume: number }>) => {
    return orders.reduce((sum, order) => sum + order.volume, 0);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Activity className="w-8 h-8 text-blue-500" />
            실시간 시장 데이터 대시보드
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            KRX/NASDAQ 실시간 데이터 • AI 가격 예측 • 호가창 분석
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-gray-600 dark:text-gray-400">실시간 연결됨</span>
        </div>
      </div>

      {/* Stock Header */}
      <div className="grid grid-cols-6 gap-4 mb-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div>
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white">{stockData.ticker}</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">{stockData.name}</p>
        </div>
        <div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">${stockData.price}</div>
          <div className={`text-sm font-semibold ${stockData.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {stockData.change >= 0 ? '+' : ''}{stockData.change} ({stockData.changePercent}%)
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-600 dark:text-gray-400">거래량</div>
          <div className="font-semibold">{(stockData.volume / 1000000).toFixed(1)}M</div>
        </div>
        <div>
          <div className="text-xs text-gray-600 dark:text-gray-400">고가/저가</div>
          <div className="font-semibold">${stockData.high}/${stockData.low}</div>
        </div>
        <div>
          <div className="text-xs text-gray-600 dark:text-gray-400">시가총액</div>
          <div className="font-semibold">{stockData.marketCap}</div>
        </div>
        <div>
          <div className="text-xs text-gray-600 dark:text-gray-400">PER</div>
          <div className="font-semibold">{stockData.pe}</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Chart */}
        <div className="col-span-2">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">가격 차트</h3>
              <div className="flex gap-2">
                {['1D', '1W', '1M', '3M', '1Y'].map(tf => (
                  <button
                    key={tf}
                    onClick={() => setTimeFrame(tf)}
                    className={`px-3 py-1 text-sm rounded ${
                      timeFrame === tf
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>
            <canvas
              ref={chartCanvasRef}
              width={600}
              height={300}
              className="w-full"
            />
          </div>

          {/* AI Prediction */}
          <div className="mt-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5 text-purple-600" />
              AI 가격 예측
            </h3>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400">예측 방향</div>
                <div className={`text-xl font-bold ${aiPrediction.direction === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                  {aiPrediction.direction === 'up' ? '상승' : '하락'} 예상
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400">신뢰도</div>
                <div className="text-xl font-bold text-blue-600">{aiPrediction.confidence}%</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400">목표가</div>
                <div className="text-xl font-bold text-purple-600">${aiPrediction.targetPrice}</div>
              </div>
            </div>
            <div className="mt-3 space-y-1">
              {aiPrediction.signals.map((signal, index) => (
                <div key={index} className="text-sm text-gray-700 dark:text-gray-300">
                  • {signal}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Order Book & Recent Trades */}
        <div className="space-y-4">
          {/* Order Book */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3">호가창</h3>
            <div className="space-y-1">
              {/* Asks */}
              {[...orderBook.asks].reverse().map((ask, index) => (
                <div key={`ask-${index}`} className="flex items-center justify-between text-sm">
                  <div className="text-red-600 font-medium">${ask.price.toFixed(2)}</div>
                  <div className="flex-1 mx-2">
                    <div className="h-4 bg-red-100 dark:bg-red-900/30 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-red-500"
                        style={{ width: `${(ask.volume / 5000) * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="text-gray-600 dark:text-gray-400">{ask.volume.toLocaleString()}</div>
                </div>
              ))}
              
              {/* Spread */}
              <div className="border-y border-gray-300 dark:border-gray-600 py-1 my-1">
                <div className="text-center text-sm font-semibold text-gray-900 dark:text-white">
                  스프레드: ${(orderBook.asks[0].price - orderBook.bids[0].price).toFixed(2)}
                </div>
              </div>

              {/* Bids */}
              {orderBook.bids.map((bid, index) => (
                <div key={`bid-${index}`} className="flex items-center justify-between text-sm">
                  <div className="text-green-600 font-medium">${bid.price.toFixed(2)}</div>
                  <div className="flex-1 mx-2">
                    <div className="h-4 bg-green-100 dark:bg-green-900/30 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-green-500"
                        style={{ width: `${(bid.volume / 5000) * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="text-gray-600 dark:text-gray-400">{bid.volume.toLocaleString()}</div>
                </div>
              ))}
            </div>
            
            <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700 grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-600 dark:text-gray-400">매도 총량: </span>
                <span className="font-semibold text-red-600">
                  {calculateTotalVolume(orderBook.asks).toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">매수 총량: </span>
                <span className="font-semibold text-green-600">
                  {calculateTotalVolume(orderBook.bids).toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {/* Recent Trades */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3">실시간 체결</h3>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {recentTicks.map((tick, index) => (
                <div key={index} className="flex items-center justify-between text-sm py-1">
                  <div className={`font-medium ${tick.type === 'buy' ? 'text-green-600' : 'text-red-600'}`}>
                    ${tick.price.toFixed(2)}
                  </div>
                  <div className="text-gray-600 dark:text-gray-400">
                    {tick.volume.toLocaleString()}
                  </div>
                  <div className="text-xs text-gray-500">
                    {tick.time.toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Watchlist */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3">관심 종목</h3>
            <div className="space-y-2">
              {watchlist.map(stock => (
                <div 
                  key={stock.ticker}
                  className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                  onClick={() => setSelectedStock(stock.ticker)}
                >
                  <div className="font-semibold text-sm">{stock.ticker}</div>
                  <div className="text-right">
                    <div className="text-sm font-medium">${stock.price}</div>
                    <div className={`text-xs ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {stock.change >= 0 ? '+' : ''}{stock.change}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}