'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { ArrowLeft, LineChart, TrendingUp, TrendingDown, Activity, AlertTriangle, BarChart3, Clock, Target, Shield, Brain, Settings, Download, Maximize2, Info, RefreshCcw } from 'lucide-react';

interface Pattern {
  id: string;
  type: string;
  name: string;
  startIndex: number;
  endIndex: number;
  confidence: number;
  direction: 'bullish' | 'bearish' | 'neutral';
  targetPrice: number;
  stopLoss: number;
  description: string;
  accuracy: number;
}

interface Signal {
  type: 'buy' | 'sell' | 'hold';
  confidence: number;
  price: number;
  timestamp: Date;
  patterns: string[];
  riskReward: number;
}

interface Candle {
  time: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TimeFrame {
  value: string;
  label: string;
  minutes: number;
}

export default function AIChartAnalyzerPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [currentSignal, setCurrentSignal] = useState<Signal | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('15m');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedPattern, setSelectedPattern] = useState<Pattern | null>(null);
  const [showGrid, setShowGrid] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const timeframes: TimeFrame[] = [
    { value: '1m', label: '1분', minutes: 1 },
    { value: '5m', label: '5분', minutes: 5 },
    { value: '15m', label: '15분', minutes: 15 },
    { value: '1h', label: '1시간', minutes: 60 },
    { value: '4h', label: '4시간', minutes: 240 },
    { value: '1d', label: '1일', minutes: 1440 },
  ];

  const patternTypes = [
    { id: 'head-shoulders', name: 'Head & Shoulders', accuracy: 0.73 },
    { id: 'inverse-head-shoulders', name: 'Inverse Head & Shoulders', accuracy: 0.71 },
    { id: 'double-top', name: 'Double Top', accuracy: 0.68 },
    { id: 'double-bottom', name: 'Double Bottom', accuracy: 0.72 },
    { id: 'ascending-triangle', name: 'Ascending Triangle', accuracy: 0.75 },
    { id: 'descending-triangle', name: 'Descending Triangle', accuracy: 0.73 },
    { id: 'symmetric-triangle', name: 'Symmetric Triangle', accuracy: 0.65 },
    { id: 'rising-wedge', name: 'Rising Wedge', accuracy: 0.70 },
    { id: 'falling-wedge', name: 'Falling Wedge', accuracy: 0.72 },
    { id: 'bull-flag', name: 'Bull Flag', accuracy: 0.76 },
    { id: 'bear-flag', name: 'Bear Flag', accuracy: 0.74 },
    { id: 'cup-handle', name: 'Cup & Handle', accuracy: 0.69 },
  ];

  // Generate realistic price data
  const generateCandles = useCallback((count: number, timeframe: number) => {
    const newCandles: Candle[] = [];
    const now = new Date();
    let price = 50000 + Math.random() * 5000;
    
    for (let i = count - 1; i >= 0; i--) {
      const time = new Date(now.getTime() - i * timeframe * 60000);
      const volatility = 0.002 + Math.random() * 0.003;
      const trend = Math.sin(i / 20) * 0.001;
      
      const open = price;
      const change = (Math.random() - 0.5) * volatility + trend;
      const close = open * (1 + change);
      const high = Math.max(open, close) * (1 + Math.random() * volatility * 0.5);
      const low = Math.min(open, close) * (1 - Math.random() * volatility * 0.5);
      const volume = 1000000 + Math.random() * 2000000;
      
      newCandles.push({ time, open, high, low, close, volume });
      price = close;
    }
    
    return newCandles;
  }, []);

  // Analyze patterns using AI simulation
  const analyzePatterns = useCallback((data: Candle[]) => {
    const detectedPatterns: Pattern[] = [];
    
    // Simulate pattern detection
    for (let i = 20; i < data.length - 10; i++) {
      if (Math.random() < 0.1) { // 10% chance to detect a pattern
        const patternType = patternTypes[Math.floor(Math.random() * patternTypes.length)];
        const patternLength = 10 + Math.floor(Math.random() * 20);
        const confidence = 0.65 + Math.random() * 0.35;
        
        const pattern: Pattern = {
          id: `pattern-${i}`,
          type: patternType.id,
          name: patternType.name,
          startIndex: Math.max(0, i - patternLength),
          endIndex: i,
          confidence,
          direction: patternType.id.includes('bear') || patternType.id.includes('descending') ? 'bearish' : 'bullish',
          targetPrice: data[i].close * (1 + (Math.random() * 0.05) * (patternType.id.includes('bear') ? -1 : 1)),
          stopLoss: data[i].close * (1 - (Math.random() * 0.02) * (patternType.id.includes('bear') ? -1 : 1)),
          description: `${patternType.name} pattern detected with ${(confidence * 100).toFixed(1)}% confidence`,
          accuracy: patternType.accuracy
        };
        
        detectedPatterns.push(pattern);
      }
    }
    
    return detectedPatterns;
  }, []);

  // Generate trading signal
  const generateSignal = useCallback((patterns: Pattern[], currentPrice: number) => {
    if (patterns.length === 0) return null;
    
    const recentPatterns = patterns.slice(-3);
    const bullishCount = recentPatterns.filter(p => p.direction === 'bullish').length;
    const bearishCount = recentPatterns.filter(p => p.direction === 'bearish').length;
    const avgConfidence = recentPatterns.reduce((sum, p) => sum + p.confidence, 0) / recentPatterns.length;
    
    let signal: Signal;
    
    if (bullishCount > bearishCount && avgConfidence > 0.7) {
      signal = {
        type: 'buy',
        confidence: avgConfidence,
        price: currentPrice,
        timestamp: new Date(),
        patterns: recentPatterns.map(p => p.name),
        riskReward: 2.5 + Math.random() * 1.5
      };
    } else if (bearishCount > bullishCount && avgConfidence > 0.7) {
      signal = {
        type: 'sell',
        confidence: avgConfidence,
        price: currentPrice,
        timestamp: new Date(),
        patterns: recentPatterns.map(p => p.name),
        riskReward: 2.5 + Math.random() * 1.5
      };
    } else {
      signal = {
        type: 'hold',
        confidence: avgConfidence,
        price: currentPrice,
        timestamp: new Date(),
        patterns: recentPatterns.map(p => p.name),
        riskReward: 1.0
      };
    }
    
    return signal;
  }, []);

  // Draw chart on canvas
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || candles.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate dimensions
    const padding = 60;
    const chartWidth = canvas.width - padding * 2;
    const chartHeight = canvas.height - padding * 2 - (showVolume ? 100 : 0);
    const volumeHeight = showVolume ? 80 : 0;
    
    // Find price range
    const prices = candles.flatMap(c => [c.high, c.low]);
    const minPrice = Math.min(...prices) * 0.995;
    const maxPrice = Math.max(...prices) * 1.005;
    const priceRange = maxPrice - minPrice;
    
    // Draw background
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      
      // Horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = padding + (chartHeight / 10) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();
        
        // Price labels
        const price = maxPrice - (priceRange / 10) * i;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '11px Inter';
        ctx.textAlign = 'right';
        ctx.fillText(price.toFixed(0), padding - 10, y + 4);
      }
      
      // Vertical grid lines
      const candleWidth = chartWidth / candles.length;
      for (let i = 0; i < candles.length; i += Math.floor(candles.length / 10)) {
        const x = padding + i * candleWidth + candleWidth / 2;
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, padding + chartHeight + volumeHeight);
        ctx.stroke();
      }
    }
    
    // Draw candles
    const candleWidth = chartWidth / candles.length;
    candles.forEach((candle, i) => {
      const x = padding + i * candleWidth;
      const candleBodyWidth = candleWidth * 0.8;
      const wickX = x + candleWidth / 2;
      
      // Calculate positions
      const openY = padding + ((maxPrice - candle.open) / priceRange) * chartHeight;
      const closeY = padding + ((maxPrice - candle.close) / priceRange) * chartHeight;
      const highY = padding + ((maxPrice - candle.high) / priceRange) * chartHeight;
      const lowY = padding + ((maxPrice - candle.low) / priceRange) * chartHeight;
      
      // Determine color
      const isBullish = candle.close >= candle.open;
      const color = isBullish ? '#10b981' : '#ef4444';
      
      // Draw wick
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(wickX, highY);
      ctx.lineTo(wickX, lowY);
      ctx.stroke();
      
      // Draw body
      ctx.fillStyle = color;
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.abs(closeY - openY) || 1;
      ctx.fillRect(x + (candleWidth - candleBodyWidth) / 2, bodyTop, candleBodyWidth, bodyHeight);
    });
    
    // Draw volume
    if (showVolume) {
      const maxVolume = Math.max(...candles.map(c => c.volume));
      const volumeY = padding + chartHeight + 20;
      
      candles.forEach((candle, i) => {
        const x = padding + i * candleWidth;
        const volumeBarHeight = (candle.volume / maxVolume) * volumeHeight;
        const isBullish = candle.close >= candle.open;
        
        ctx.fillStyle = isBullish ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)';
        ctx.fillRect(x, volumeY + volumeHeight - volumeBarHeight, candleWidth * 0.8, volumeBarHeight);
      });
    }
    
    // Draw patterns
    patterns.forEach(pattern => {
      if (pattern.startIndex >= 0 && pattern.endIndex < candles.length) {
        const startX = padding + pattern.startIndex * candleWidth + candleWidth / 2;
        const endX = padding + pattern.endIndex * candleWidth + candleWidth / 2;
        
        const patternCandles = candles.slice(pattern.startIndex, pattern.endIndex + 1);
        const patternPrices = patternCandles.flatMap(c => [c.high, c.low]);
        const patternMinPrice = Math.min(...patternPrices);
        const patternMaxPrice = Math.max(...patternPrices);
        
        const minY = padding + ((maxPrice - patternMinPrice) / priceRange) * chartHeight;
        const maxY = padding + ((maxPrice - patternMaxPrice) / priceRange) * chartHeight;
        
        // Draw pattern highlight
        ctx.strokeStyle = pattern.direction === 'bullish' ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(startX, maxY, endX - startX, minY - maxY);
        ctx.setLineDash([]);
        
        // Draw pattern label
        if (pattern === selectedPattern) {
          ctx.fillStyle = pattern.direction === 'bullish' ? '#10b981' : '#ef4444';
          ctx.font = 'bold 12px Inter';
          ctx.textAlign = 'center';
          ctx.fillText(pattern.name, (startX + endX) / 2, maxY - 10);
        }
      }
    });
    
    // Draw current price line
    if (candles.length > 0) {
      const currentPrice = candles[candles.length - 1].close;
      const priceY = padding + ((maxPrice - currentPrice) / priceRange) * chartHeight;
      
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 5]);
      ctx.beginPath();
      ctx.moveTo(padding, priceY);
      ctx.lineTo(canvas.width - padding, priceY);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Price label
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(canvas.width - padding + 5, priceY - 12, 80, 24);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px Inter';
      ctx.textAlign = 'left';
      ctx.fillText(currentPrice.toFixed(0), canvas.width - padding + 10, priceY + 3);
    }
  }, [candles, patterns, selectedPattern, showGrid, showVolume]);

  // Initialize chart
  useEffect(() => {
    const timeframe = timeframes.find(tf => tf.value === selectedTimeframe);
    if (timeframe) {
      const newCandles = generateCandles(100, timeframe.minutes);
      setCandles(newCandles);
      
      // Analyze patterns after a short delay
      setTimeout(() => {
        setIsAnalyzing(true);
        const detectedPatterns = analyzePatterns(newCandles);
        setPatterns(detectedPatterns);
        
        if (newCandles.length > 0) {
          const signal = generateSignal(detectedPatterns, newCandles[newCandles.length - 1].close);
          setCurrentSignal(signal);
        }
        setIsAnalyzing(false);
      }, 1500);
    }
  }, [selectedTimeframe, generateCandles, analyzePatterns, generateSignal]);

  // Redraw chart when data changes
  useEffect(() => {
    drawChart();
  }, [drawChart]);

  // Handle canvas resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }
        drawChart();
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [drawChart]);

  // Auto refresh
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      const timeframe = timeframes.find(tf => tf.value === selectedTimeframe);
      if (timeframe) {
        const newCandles = generateCandles(100, timeframe.minutes);
        setCandles(newCandles);
        
        const detectedPatterns = analyzePatterns(newCandles);
        setPatterns(detectedPatterns);
        
        if (newCandles.length > 0) {
          const signal = generateSignal(detectedPatterns, newCandles[newCandles.length - 1].close);
          setCurrentSignal(signal);
        }
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [autoRefresh, selectedTimeframe, generateCandles, analyzePatterns, generateSignal]);

  const handleRefresh = () => {
    const timeframe = timeframes.find(tf => tf.value === selectedTimeframe);
    if (timeframe) {
      setIsAnalyzing(true);
      const newCandles = generateCandles(100, timeframe.minutes);
      setCandles(newCandles);
      
      setTimeout(() => {
        const detectedPatterns = analyzePatterns(newCandles);
        setPatterns(detectedPatterns);
        
        if (newCandles.length > 0) {
          const signal = generateSignal(detectedPatterns, newCandles[newCandles.length - 1].close);
          setCurrentSignal(signal);
        }
        setIsAnalyzing(false);
      }, 1500);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="border-b border-gray-800">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/stock-analysis/tools"
                className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>도구 목록</span>
              </Link>
              <div className="w-px h-6 bg-gray-700" />
              <h1 className="text-xl font-bold">AI Chart Pattern Analyzer</h1>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={handleRefresh}
                disabled={isAnalyzing}
                className="px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <RefreshCcw className={`w-4 h-4 ${isAnalyzing ? 'animate-spin' : ''}`} />
                <span>새로고침</span>
              </button>
              
              <button className="p-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors">
                <Download className="w-5 h-5" />
              </button>
              
              <button className="p-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors">
                <Maximize2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Main Chart Area */}
          <div className="xl:col-span-3 space-y-4">
            {/* Timeframe Selector */}
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {timeframes.map((tf) => (
                    <button
                      key={tf.value}
                      onClick={() => setSelectedTimeframe(tf.value)}
                      className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                        selectedTimeframe === tf.value
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
                      }`}
                    >
                      {tf.label}
                    </button>
                  ))}
                </div>
                
                <div className="flex items-center gap-4">
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={showGrid}
                      onChange={(e) => setShowGrid(e.target.checked)}
                      className="rounded"
                    />
                    <span>그리드</span>
                  </label>
                  
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={showVolume}
                      onChange={(e) => setShowVolume(e.target.checked)}
                      className="rounded"
                    />
                    <span>거래량</span>
                  </label>
                  
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={autoRefresh}
                      onChange={(e) => setAutoRefresh(e.target.checked)}
                      className="rounded"
                    />
                    <span>자동 새로고침</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Chart Canvas */}
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="relative" style={{ height: '600px' }}>
                <canvas
                  ref={canvasRef}
                  className="w-full h-full"
                  style={{ width: '100%', height: '100%' }}
                />
                
                {isAnalyzing && (
                  <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                    <div className="text-center">
                      <Brain className="w-12 h-12 mx-auto mb-3 animate-pulse text-blue-500" />
                      <p className="text-lg font-medium">AI 패턴 분석 중...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Trading Signal */}
            {currentSignal && (
              <div className={`rounded-lg p-6 ${
                currentSignal.type === 'buy' 
                  ? 'bg-green-900/20 border border-green-800' 
                  : currentSignal.type === 'sell'
                  ? 'bg-red-900/20 border border-red-800'
                  : 'bg-gray-900 border border-gray-800'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    {currentSignal.type === 'buy' ? (
                      <TrendingUp className="w-8 h-8 text-green-500" />
                    ) : currentSignal.type === 'sell' ? (
                      <TrendingDown className="w-8 h-8 text-red-500" />
                    ) : (
                      <Activity className="w-8 h-8 text-gray-500" />
                    )}
                    <div>
                      <h3 className="text-xl font-bold">
                        {currentSignal.type === 'buy' ? '매수 신호' : currentSignal.type === 'sell' ? '매도 신호' : '관망 신호'}
                      </h3>
                      <p className="text-sm text-gray-400">
                        신뢰도: {(currentSignal.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <p className="text-2xl font-bold">{currentSignal.price.toFixed(0)}</p>
                    <p className="text-sm text-gray-400">진입가격</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Target className="w-4 h-4 text-blue-500" />
                      <span className="text-sm text-gray-400">Risk/Reward</span>
                    </div>
                    <p className="text-lg font-semibold">1:{currentSignal.riskReward.toFixed(1)}</p>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Clock className="w-4 h-4 text-yellow-500" />
                      <span className="text-sm text-gray-400">시간</span>
                    </div>
                    <p className="text-lg font-semibold">
                      {currentSignal.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <LineChart className="w-4 h-4 text-purple-500" />
                      <span className="text-sm text-gray-400">패턴 수</span>
                    </div>
                    <p className="text-lg font-semibold">{currentSignal.patterns.length}개</p>
                  </div>
                </div>
                
                {currentSignal.patterns.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-700">
                    <p className="text-sm text-gray-400 mb-2">감지된 패턴:</p>
                    <div className="flex flex-wrap gap-2">
                      {currentSignal.patterns.map((pattern, i) => (
                        <span key={i} className="px-3 py-1 bg-gray-800 rounded-full text-sm">
                          {pattern}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Sidebar - Pattern Analysis */}
          <div className="xl:col-span-1 space-y-4">
            {/* Pattern Detection Results */}
            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-500" />
                패턴 인식 결과
              </h3>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {patterns.length > 0 ? (
                  patterns.map((pattern) => (
                    <div
                      key={pattern.id}
                      onClick={() => setSelectedPattern(pattern)}
                      className={`p-3 rounded-lg cursor-pointer transition-colors ${
                        selectedPattern?.id === pattern.id
                          ? 'bg-blue-900/30 border border-blue-700'
                          : 'bg-gray-800 hover:bg-gray-700'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium">{pattern.name}</h4>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          pattern.direction === 'bullish'
                            ? 'bg-green-900/50 text-green-400'
                            : 'bg-red-900/50 text-red-400'
                        }`}>
                          {pattern.direction === 'bullish' ? '상승' : '하락'}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-gray-400">신뢰도:</span>
                          <span className="ml-1 font-medium">
                            {(pattern.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">정확도:</span>
                          <span className="ml-1 font-medium">
                            {(pattern.accuracy * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      
                      <div className="mt-2 pt-2 border-t border-gray-700 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">목표가:</span>
                          <span className="text-green-400">{pattern.targetPrice.toFixed(0)}</span>
                        </div>
                        <div className="flex justify-between mt-1">
                          <span className="text-gray-400">손절가:</span>
                          <span className="text-red-400">{pattern.stopLoss.toFixed(0)}</span>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>패턴을 분석하는 중...</p>
                  </div>
                )}
              </div>
            </div>

            {/* Pattern Accuracy Statistics */}
            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-green-500" />
                패턴 정확도 통계
              </h3>
              
              <div className="space-y-2">
                {patternTypes.slice(0, 6).map((type) => (
                  <div key={type.id} className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">{type.name}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-gray-800 rounded-full h-2 overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-blue-500 to-blue-600"
                          style={{ width: `${type.accuracy * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium w-12 text-right">
                        {(type.accuracy * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">평균 정확도</span>
                  <span className="font-semibold text-blue-400">71.3%</span>
                </div>
                <div className="flex items-center justify-between text-sm mt-2">
                  <span className="text-gray-400">총 백테스트</span>
                  <span className="font-semibold">15,420회</span>
                </div>
              </div>
            </div>

            {/* AI Model Info */}
            <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 rounded-lg p-4 border border-blue-800/50">
              <div className="flex items-center gap-3 mb-3">
                <Brain className="w-6 h-6 text-blue-400" />
                <h3 className="text-lg font-semibold">AI 모델 정보</h3>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">모델 버전</span>
                  <span className="font-medium">v2.4.1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">학습 데이터</span>
                  <span className="font-medium">10년 (2014-2024)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">패턴 타입</span>
                  <span className="font-medium">12종</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">업데이트</span>
                  <span className="font-medium">실시간</span>
                </div>
              </div>
              
              <button className="w-full mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors flex items-center justify-center gap-2">
                <Info className="w-4 h-4" />
                <span>모델 상세정보</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}