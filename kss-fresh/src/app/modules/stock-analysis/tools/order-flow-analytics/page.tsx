'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { ArrowLeft, Activity, TrendingUp, AlertCircle, Database, Clock, Users, BarChart3, Shield, Settings, Download, Maximize2, RefreshCw, Info, ChevronRight, DollarSign, Zap } from 'lucide-react';

interface OrderData {
  id: string;
  time: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  type: 'market' | 'limit' | 'iceberg' | 'dark';
  venue: string;
  trader: 'retail' | 'institutional' | 'hft' | 'mm';
}

interface OrderBookLevel {
  price: number;
  quantity: number;
  orders: number;
}

interface MarketMetrics {
  bidAskSpread: number;
  spreadBps: number;
  imbalance: number;
  toxicity: number;
  liquidityScore: number;
  darkPoolRatio: number;
}

export default function OrderFlowAnalyticsPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1m');
  const [isConnected, setIsConnected] = useState(false);
  const [orderFlow, setOrderFlow] = useState<OrderData[]>([]);
  const [orderBook, setOrderBook] = useState<{ bids: OrderBookLevel[], asks: OrderBookLevel[] }>({ bids: [], asks: [] });
  const [metrics, setMetrics] = useState<MarketMetrics>({
    bidAskSpread: 0.05,
    spreadBps: 2.5,
    imbalance: 0.35,
    toxicity: 0.22,
    liquidityScore: 85,
    darkPoolRatio: 0.18
  });
  const [alerts, setAlerts] = useState<any[]>([]);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  // Symbols list
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'BAC', 'GS'];

  // Generate realistic order flow data
  useEffect(() => {
    if (!isConnected) return;

    const generateOrder = (): OrderData => {
      const types: OrderData['type'][] = ['market', 'limit', 'iceberg', 'dark'];
      const traders: OrderData['trader'][] = ['retail', 'institutional', 'hft', 'mm'];
      const side = Math.random() > 0.5 ? 'buy' : 'sell';
      const basePrice = 150 + Math.random() * 2;
      
      return {
        id: Math.random().toString(36).substr(2, 9),
        time: new Date().toLocaleTimeString('en-US', { hour12: false }),
        symbol: selectedSymbol,
        side,
        price: Number((basePrice + (side === 'buy' ? -0.01 : 0.01) * Math.random()).toFixed(2)),
        quantity: Math.floor(Math.random() * 10000) + 100,
        type: types[Math.floor(Math.random() * types.length)],
        venue: ['NYSE', 'NASDAQ', 'BATS', 'IEX'][Math.floor(Math.random() * 4)],
        trader: traders[Math.floor(Math.random() * traders.length)]
      };
    };

    const interval = setInterval(() => {
      const newOrder = generateOrder();
      setOrderFlow(prev => [newOrder, ...prev].slice(0, 100));
      
      // Update metrics
      setMetrics(prev => ({
        bidAskSpread: Math.max(0.01, prev.bidAskSpread + (Math.random() - 0.5) * 0.02),
        spreadBps: Math.max(0.5, prev.spreadBps + (Math.random() - 0.5) * 0.5),
        imbalance: Math.max(-1, Math.min(1, prev.imbalance + (Math.random() - 0.5) * 0.1)),
        toxicity: Math.max(0, Math.min(1, prev.toxicity + (Math.random() - 0.5) * 0.05)),
        liquidityScore: Math.max(0, Math.min(100, prev.liquidityScore + (Math.random() - 0.5) * 5)),
        darkPoolRatio: Math.max(0, Math.min(0.5, prev.darkPoolRatio + (Math.random() - 0.5) * 0.02))
      }));
      
      // Detect anomalies
      if (newOrder.type === 'dark' && newOrder.quantity > 5000) {
        setAlerts(prev => [{
          id: Date.now(),
          type: 'dark_pool',
          message: `Large Dark Pool order detected: ${newOrder.quantity} shares @ $${newOrder.price}`,
          severity: 'high',
          time: new Date().toLocaleTimeString()
        }, ...prev].slice(0, 5));
      }
      
      if (newOrder.trader === 'institutional' && newOrder.quantity > 8000) {
        setAlerts(prev => [{
          id: Date.now(),
          type: 'institutional',
          message: `Institutional ${newOrder.side} order: ${newOrder.quantity} shares`,
          severity: 'medium',
          time: new Date().toLocaleTimeString()
        }, ...prev].slice(0, 5));
      }
    }, Math.random() * 500 + 200);

    return () => clearInterval(interval);
  }, [isConnected, selectedSymbol]);

  // Generate order book
  useEffect(() => {
    if (!isConnected) return;

    const generateOrderBook = () => {
      const midPrice = 150;
      const bids: OrderBookLevel[] = [];
      const asks: OrderBookLevel[] = [];
      
      for (let i = 0; i < 10; i++) {
        bids.push({
          price: Number((midPrice - 0.01 * (i + 1)).toFixed(2)),
          quantity: Math.floor(Math.random() * 10000) + 1000,
          orders: Math.floor(Math.random() * 20) + 1
        });
        
        asks.push({
          price: Number((midPrice + 0.01 * (i + 1)).toFixed(2)),
          quantity: Math.floor(Math.random() * 10000) + 1000,
          orders: Math.floor(Math.random() * 20) + 1
        });
      }
      
      setOrderBook({ bids, asks });
    };

    generateOrderBook();
    const interval = setInterval(generateOrderBook, 1000);
    return () => clearInterval(interval);
  }, [isConnected]);

  // Liquidity heatmap visualization
  useEffect(() => {
    if (!canvasRef.current || !isConnected) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw liquidity heatmap
      const cellWidth = canvas.width / 20;
      const cellHeight = canvas.height / 10;

      for (let x = 0; x < 20; x++) {
        for (let y = 0; y < 10; y++) {
          const intensity = Math.random();
          const hue = 120 - intensity * 120; // Green to red
          ctx.fillStyle = `hsla(${hue}, 70%, 50%, ${intensity})`;
          ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth - 1, cellHeight - 1);
        }
      }

      // Draw order flow lines
      orderFlow.slice(0, 20).forEach((order, i) => {
        const y = (i + 1) * (canvas.height / 20);
        ctx.strokeStyle = order.side === 'buy' ? '#10b981' : '#ef4444';
        ctx.lineWidth = Math.log(order.quantity) / 2;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width * (order.quantity / 10000), y);
        ctx.stroke();
      });

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isConnected, orderFlow]);

  const getTraderColor = (trader: string) => {
    switch (trader) {
      case 'institutional': return 'text-blue-500';
      case 'hft': return 'text-purple-500';
      case 'mm': return 'text-orange-500';
      default: return 'text-gray-500';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'dark': return 'bg-gray-800 text-gray-200';
      case 'iceberg': return 'bg-blue-800 text-blue-200';
      case 'limit': return 'bg-green-800 text-green-200';
      default: return 'bg-yellow-800 text-yellow-200';
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/stock-analysis/tools"
                className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>도구 목록</span>
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <h1 className="text-xl font-bold">Order Flow Analytics</h1>
              <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded text-xs font-medium">
                Institutional Grade
              </span>
            </div>
            
            <div className="flex items-center gap-4">
              <button
                onClick={() => setIsConnected(!isConnected)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                  isConnected 
                    ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30' 
                    : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                }`}
              >
                <Activity className="w-4 h-4" />
                {isConnected ? 'Connected' : 'Disconnected'}
              </button>
              <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
                <Settings className="w-5 h-5" />
              </button>
              <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
                <Maximize2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex h-[calc(100vh-64px)]">
        {/* Left Sidebar - Controls */}
        <div className="w-64 bg-gray-900 border-r border-gray-800 p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Symbol</label>
            <select
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
            >
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Timeframe</label>
            <div className="grid grid-cols-3 gap-2">
              {['1m', '5m', '15m'].map(tf => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  className={`px-3 py-2 rounded-lg font-medium transition-colors ${
                    timeframe === tf
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-400">Market Metrics</h3>
            
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">Bid-Ask Spread</span>
                <span className="text-sm font-mono">${metrics.bidAskSpread.toFixed(3)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">Spread (bps)</span>
                <span className="text-sm font-mono">{metrics.spreadBps.toFixed(1)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">Order Imbalance</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all ${metrics.imbalance > 0 ? 'bg-green-500' : 'bg-red-500'}`}
                      style={{ 
                        width: `${Math.abs(metrics.imbalance) * 50 + 50}%`,
                        marginLeft: metrics.imbalance < 0 ? `${50 - Math.abs(metrics.imbalance) * 50}%` : '0'
                      }}
                    />
                  </div>
                  <span className="text-sm font-mono">{(metrics.imbalance * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">Toxicity Score</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all ${
                        metrics.toxicity < 0.3 ? 'bg-green-500' : 
                        metrics.toxicity < 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${metrics.toxicity * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-mono">{(metrics.toxicity * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">Liquidity Score</span>
                <span className="text-sm font-mono">{metrics.liquidityScore.toFixed(0)}/100</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">Dark Pool %</span>
                <span className="text-sm font-mono">{(metrics.darkPoolRatio * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-gray-800">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Alerts</h3>
            <div className="space-y-2">
              {alerts.map(alert => (
                <div key={alert.id} className="p-2 bg-gray-800 rounded-lg">
                  <div className="flex items-start gap-2">
                    <AlertCircle className={`w-4 h-4 mt-0.5 ${
                      alert.severity === 'high' ? 'text-red-400' : 'text-yellow-400'
                    }`} />
                    <div className="flex-1">
                      <p className="text-xs text-gray-300">{alert.message}</p>
                      <p className="text-xs text-gray-500 mt-1">{alert.time}</p>
                    </div>
                  </div>
                </div>
              ))}
              {alerts.length === 0 && (
                <p className="text-xs text-gray-500 text-center py-4">No alerts</p>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Top Section - Liquidity Heatmap */}
          <div className="h-1/3 p-4">
            <div className="h-full bg-gray-900 rounded-lg border border-gray-800 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold">Liquidity Heatmap</h2>
                <div className="flex items-center gap-4 text-sm text-gray-400">
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-sm"></div>
                    High Liquidity
                  </span>
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-sm"></div>
                    Low Liquidity
                  </span>
                </div>
              </div>
              <canvas
                ref={canvasRef}
                width={800}
                height={200}
                className="w-full h-full rounded"
              />
            </div>
          </div>

          {/* Middle Section - Order Book & Order Flow */}
          <div className="flex-1 flex gap-4 p-4 pt-0">
            {/* Order Book */}
            <div className="w-1/3 bg-gray-900 rounded-lg border border-gray-800 p-4">
              <h2 className="text-lg font-semibold mb-3">Order Book</h2>
              <div className="space-y-1">
                <div className="grid grid-cols-3 gap-2 text-xs font-medium text-gray-500 pb-2 border-b border-gray-800">
                  <span>Price</span>
                  <span className="text-center">Size</span>
                  <span className="text-right">Orders</span>
                </div>
                
                {/* Asks */}
                {orderBook.asks.slice(0, 5).reverse().map((level, i) => (
                  <div key={`ask-${i}`} className="grid grid-cols-3 gap-2 text-xs hover:bg-gray-800/50 py-1">
                    <span className="text-red-400 font-mono">${level.price.toFixed(2)}</span>
                    <span className="text-center text-gray-300">{level.quantity.toLocaleString()}</span>
                    <span className="text-right text-gray-500">{level.orders}</span>
                  </div>
                ))}
                
                <div className="h-px bg-gray-700 my-1" />
                
                {/* Bids */}
                {orderBook.bids.slice(0, 5).map((level, i) => (
                  <div key={`bid-${i}`} className="grid grid-cols-3 gap-2 text-xs hover:bg-gray-800/50 py-1">
                    <span className="text-green-400 font-mono">${level.price.toFixed(2)}</span>
                    <span className="text-center text-gray-300">{level.quantity.toLocaleString()}</span>
                    <span className="text-right text-gray-500">{level.orders}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Order Flow */}
            <div className="flex-1 bg-gray-900 rounded-lg border border-gray-800 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold">Order Flow</h2>
                <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors flex items-center gap-2">
                  <Download className="w-4 h-4" />
                  Export
                </button>
              </div>
              
              <div className="overflow-auto h-[calc(100%-40px)]">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-gray-900">
                    <tr className="text-gray-500 border-b border-gray-800">
                      <th className="text-left py-2">Time</th>
                      <th className="text-left">Side</th>
                      <th className="text-right">Price</th>
                      <th className="text-right">Size</th>
                      <th className="text-center">Type</th>
                      <th className="text-center">Venue</th>
                      <th className="text-center">Trader</th>
                    </tr>
                  </thead>
                  <tbody>
                    {orderFlow.map((order) => (
                      <tr key={order.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                        <td className="py-2 text-gray-400">{order.time}</td>
                        <td className={order.side === 'buy' ? 'text-green-400' : 'text-red-400'}>
                          {order.side.toUpperCase()}
                        </td>
                        <td className="text-right font-mono">${order.price.toFixed(2)}</td>
                        <td className="text-right">{order.quantity.toLocaleString()}</td>
                        <td className="text-center">
                          <span className={`px-2 py-0.5 rounded text-xs ${getTypeColor(order.type)}`}>
                            {order.type}
                          </span>
                        </td>
                        <td className="text-center text-gray-400">{order.venue}</td>
                        <td className={`text-center ${getTraderColor(order.trader)}`}>
                          {order.trader}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Bottom Stats */}
          <div className="grid grid-cols-4 gap-4 p-4 pt-0">
            {[
              { label: 'Total Volume', value: '2.4M', icon: BarChart3, change: '+12%' },
              { label: 'Block Trades', value: '47', icon: Database, change: '+5' },
              { label: 'Dark Pool %', value: '18.2%', icon: Shield, change: '-2.1%' },
              { label: 'HFT Activity', value: 'High', icon: Zap, change: '↑' }
            ].map((stat) => {
              const Icon = stat.icon;
              return (
                <div key={stat.label} className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-400">{stat.label}</span>
                    <Icon className="w-4 h-4 text-gray-500" />
                  </div>
                  <div className="flex items-end justify-between">
                    <span className="text-2xl font-bold">{stat.value}</span>
                    <span className={`text-sm ${
                      stat.change.includes('+') || stat.change === '↑' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {stat.change}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}