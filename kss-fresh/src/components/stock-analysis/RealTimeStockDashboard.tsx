'use client';

import React, { useState, useEffect, useRef } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertCircle, 
  BarChart3,
  Brain,
  Target,
  Shield,
  Zap,
  DollarSign,
  Activity
} from 'lucide-react';
import { 
  MarketData, 
  StockRecommendation, 
  AIAnalysis,
  OrderBook,
  VolumeMetrics 
} from '@/lib/services/market-data/types';
import { marketDataService } from '@/lib/services/market-data/RealTimeDataService';
import { recommendationEngine } from '@/lib/ai/StockRecommendationEngine';

interface RealTimeStockDashboardProps {
  ticker?: string;
  onTickerChange?: (ticker: string) => void;
}

export const RealTimeStockDashboard: React.FC<RealTimeStockDashboardProps> = ({
  ticker = '005930',
  onTickerChange
}) => {
  // State management
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null);
  const [orderBook, setOrderBook] = useState<OrderBook | null>(null);
  const [volumeMetrics, setVolumeMetrics] = useState<VolumeMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1D' | '1W' | '1M' | '3M' | '1Y'>('1D');
  
  // Canvas refs for charts
  const priceChartRef = useRef<HTMLCanvasElement>(null);
  const volumeChartRef = useRef<HTMLCanvasElement>(null);
  
  // Initialize real-time data connection
  useEffect(() => {
    const initializeData = async () => {
      try {
        setLoading(true);
        
        // Connect to market data service
        await marketDataService.connect();
        
        // Subscribe to ticker
        await marketDataService.subscribe(ticker);
        
        // Set up event listeners
        marketDataService.on('price', handlePriceUpdate);
        marketDataService.on('orderbook', handleOrderBookUpdate);
        
        // Fetch initial data
        await fetchInitialData();
        
        setLoading(false);
      } catch (error) {
        console.error('Error initializing dashboard:', error);
        setLoading(false);
      }
    };
    
    initializeData();
    
    // Cleanup
    return () => {
      marketDataService.off('price', handlePriceUpdate);
      marketDataService.off('orderbook', handleOrderBookUpdate);
      marketDataService.unsubscribe(ticker);
    };
  }, [ticker]);
  
  // Fetch initial data
  const fetchInitialData = async () => {
    try {
      // Get market data
      const data = await marketDataService.fetchMarketData(ticker);
      setMarketData(data);
      
      // Get AI recommendations
      const recs = await recommendationEngine.getRecommendations({
        riskProfile: 'moderate',
        investmentHorizon: 30,
        sectors: [],
        excludeTickers: []
      });
      setRecommendations(recs);
      
      // Get order book
      const book = await marketDataService.getOrderBook(ticker);
      setOrderBook(book);
      
      // Get volume metrics
      const volume = await marketDataService.getVolumeAnalysis(ticker);
      setVolumeMetrics(volume);
      
      // Generate AI analysis
      generateAIAnalysis(ticker);
    } catch (error) {
      console.error('Error fetching initial data:', error);
    }
  };
  
  // Handle real-time price updates
  const handlePriceUpdate = (data: any) => {
    if (data.ticker === ticker) {
      setMarketData(prev => ({
        ...prev!,
        price: data.price,
        timestamp: data.timestamp,
        volume: data.volume
      }));
      
      // Update chart
      updatePriceChart(data);
    }
  };
  
  // Handle order book updates
  const handleOrderBookUpdate = (data: OrderBook) => {
    if (data.ticker === ticker) {
      setOrderBook(data);
    }
  };
  
  // Generate AI analysis
  const generateAIAnalysis = async (ticker: string) => {
    try {
      const prediction = await recommendationEngine.predictPrice(ticker, 30);
      const risk = await recommendationEngine.calculateRiskScore(ticker);
      
      const analysis: AIAnalysis = {
        ticker,
        timestamp: new Date(),
        pricePrediction: {
          oneDay: prediction.predictions.oneDay.price,
          oneWeek: prediction.predictions.oneWeek.price,
          oneMonth: prediction.predictions.oneMonth.price,
          confidence: prediction.predictions.oneMonth.confidence
        },
        sentiment: {
          news: Math.random() * 2 - 1,
          social: Math.random() * 2 - 1,
          analyst: Math.random() * 2 - 1,
          overall: Math.random() * 2 - 1
        },
        riskMetrics: {
          volatility: risk.volatility,
          beta: risk.beta,
          sharpeRatio: risk.sharpeRatio,
          maxDrawdown: risk.maxDrawdown,
          VaR95: risk.VaR95
        },
        recommendation: getRecommendation(risk.overallRisk)
      };
      
      setAiAnalysis(analysis);
    } catch (error) {
      console.error('Error generating AI analysis:', error);
    }
  };
  
  // Get recommendation based on risk
  const getRecommendation = (risk: number): AIAnalysis['recommendation'] => {
    if (risk < 3) return 'STRONG_BUY';
    if (risk < 5) return 'BUY';
    if (risk < 7) return 'HOLD';
    if (risk < 9) return 'SELL';
    return 'STRONG_SELL';
  };
  
  // Update price chart
  const updatePriceChart = (data: any) => {
    const canvas = priceChartRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Simple line chart update logic
    // In production, use a proper charting library like Chart.js or D3.js
    // This is a placeholder for demonstration
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600 mx-auto"></div>
          <p className="mt-4 text-gray-400">AI 분석 엔진 초기화 중...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-gray-900 text-white p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">실시간 AI 주식 분석 대시보드</h1>
          <p className="text-gray-400 mt-1">B2B 금융기관용 엔터프라이즈 솔루션</p>
        </div>
        <div className="flex items-center gap-4">
          <select 
            value={ticker}
            onChange={(e) => onTickerChange?.(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2"
          >
            <option value="005930">삼성전자</option>
            <option value="000660">SK하이닉스</option>
            <option value="035420">NAVER</option>
            <option value="AAPL">Apple</option>
            <option value="NVDA">NVIDIA</option>
          </select>
          <button className="bg-red-600 hover:bg-red-700 px-6 py-2 rounded-lg font-semibold transition-colors">
            실시간 모드
          </button>
        </div>
      </div>
      
      {/* Main Price Display */}
      {marketData && (
        <div className="bg-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold">{marketData.ticker}</h2>
              <div className="flex items-center gap-4 mt-2">
                <span className="text-4xl font-bold">
                  {marketData.currency === 'KRW' ? '₩' : '$'}
                  {marketData.price.toLocaleString()}
                </span>
                <div className={`flex items-center gap-1 ${marketData.changePercent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {marketData.changePercent >= 0 ? <TrendingUp /> : <TrendingDown />}
                  <span className="text-2xl font-semibold">
                    {marketData.changePercent >= 0 ? '+' : ''}{marketData.changePercent.toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-400">거래량</p>
                <p className="font-semibold">{marketData.volume.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-gray-400">시가총액</p>
                <p className="font-semibold">
                  {marketData.currency === 'KRW' ? '₩' : '$'}
                  {((marketData.marketCap || 0) / 1e12).toFixed(1)}조
                </p>
              </div>
              <div>
                <p className="text-gray-400">PER</p>
                <p className="font-semibold">{marketData.pe?.toFixed(2) || 'N/A'}</p>
              </div>
              <div>
                <p className="text-gray-400">EPS</p>
                <p className="font-semibold">{marketData.eps?.toLocaleString() || 'N/A'}</p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* AI Analysis Panel */}
      {aiAnalysis && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* AI Prediction */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="text-purple-500" />
              <h3 className="text-xl font-bold">AI 가격 예측</h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">1일 후</span>
                <span className="text-lg font-semibold">
                  {marketData?.currency === 'KRW' ? '₩' : '$'}
                  {aiAnalysis.pricePrediction.oneDay.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">1주 후</span>
                <span className="text-lg font-semibold">
                  {marketData?.currency === 'KRW' ? '₩' : '$'}
                  {aiAnalysis.pricePrediction.oneWeek.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">1개월 후</span>
                <span className="text-lg font-semibold">
                  {marketData?.currency === 'KRW' ? '₩' : '$'}
                  {aiAnalysis.pricePrediction.oneMonth.toLocaleString()}
                </span>
              </div>
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">신뢰도</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-purple-500 h-2 rounded-full"
                        style={{ width: `${aiAnalysis.pricePrediction.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm">
                      {(aiAnalysis.pricePrediction.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Risk Metrics */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Shield className="text-orange-500" />
              <h3 className="text-xl font-bold">리스크 지표</h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">변동성</span>
                <span className="font-semibold">
                  {(aiAnalysis.riskMetrics.volatility * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">베타</span>
                <span className="font-semibold">
                  {aiAnalysis.riskMetrics.beta.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">샤프 비율</span>
                <span className="font-semibold">
                  {aiAnalysis.riskMetrics.sharpeRatio.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">최대 낙폭</span>
                <span className="font-semibold text-red-400">
                  -{(aiAnalysis.riskMetrics.maxDrawdown * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">VaR (95%)</span>
                <span className="font-semibold">
                  {(aiAnalysis.riskMetrics.VaR95 * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
          
          {/* AI Recommendation */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Target className="text-green-500" />
              <h3 className="text-xl font-bold">AI 투자 의견</h3>
            </div>
            <div className="text-center py-6">
              <div className={`text-4xl font-bold mb-2 ${
                aiAnalysis.recommendation.includes('BUY') ? 'text-green-500' :
                aiAnalysis.recommendation.includes('SELL') ? 'text-red-500' :
                'text-yellow-500'
              }`}>
                {aiAnalysis.recommendation.replace('_', ' ')}
              </div>
              <div className="grid grid-cols-2 gap-2 mt-6 text-sm">
                <div className="bg-gray-700 rounded-lg p-3">
                  <p className="text-gray-400">뉴스 센티먼트</p>
                  <p className={`font-semibold ${aiAnalysis.sentiment.news > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {aiAnalysis.sentiment.news > 0 ? '긍정' : '부정'} 
                    ({(Math.abs(aiAnalysis.sentiment.news) * 100).toFixed(0)}%)
                  </p>
                </div>
                <div className="bg-gray-700 rounded-lg p-3">
                  <p className="text-gray-400">소셜 센티먼트</p>
                  <p className={`font-semibold ${aiAnalysis.sentiment.social > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {aiAnalysis.sentiment.social > 0 ? '긍정' : '부정'}
                    ({(Math.abs(aiAnalysis.sentiment.social) * 100).toFixed(0)}%)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Order Book and Volume */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Order Book */}
        {orderBook && (
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="text-blue-500" />
              <h3 className="text-xl font-bold">실시간 호가창</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {/* Asks */}
              <div>
                <h4 className="text-sm text-gray-400 mb-2">매도 호가</h4>
                <div className="space-y-1">
                  {orderBook.asks.slice(0, 5).reverse().map((ask, i) => (
                    <div key={i} className="flex justify-between text-sm">
                      <span className="text-red-400">{ask.price.toLocaleString()}</span>
                      <span className="text-gray-400">{ask.volume.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
              {/* Bids */}
              <div>
                <h4 className="text-sm text-gray-400 mb-2">매수 호가</h4>
                <div className="space-y-1">
                  {orderBook.bids.slice(0, 5).map((bid, i) => (
                    <div key={i} className="flex justify-between text-sm">
                      <span className="text-green-400">{bid.price.toLocaleString()}</span>
                      <span className="text-gray-400">{bid.volume.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">스프레드</span>
                <span className="font-semibold">{orderBook.spread.toFixed(2)}</span>
              </div>
            </div>
          </div>
        )}
        
        {/* Volume Analysis */}
        {volumeMetrics && (
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="text-cyan-500" />
              <h3 className="text-xl font-bold">거래량 분석</h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">현재 거래량</span>
                <span className="font-semibold">{volumeMetrics.totalVolume.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">10일 평균 대비</span>
                <span className={`font-semibold ${volumeMetrics.volumeRatio > 1 ? 'text-green-400' : 'text-red-400'}`}>
                  {((volumeMetrics.volumeRatio - 1) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-4">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">매수/매도 비율</span>
                  <span>{((volumeMetrics.buyVolume / volumeMetrics.totalVolume) * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-4">
                  <div 
                    className="bg-green-500 h-4 rounded-full"
                    style={{ width: `${(volumeMetrics.buyVolume / volumeMetrics.totalVolume) * 100}%` }}
                  />
                </div>
              </div>
              {volumeMetrics.institutionalVolume && (
                <div className="grid grid-cols-2 gap-2 mt-4 text-sm">
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-gray-400">기관 거래량</p>
                    <p className="font-semibold">
                      {((volumeMetrics.institutionalVolume / volumeMetrics.totalVolume) * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-gray-400">개인 거래량</p>
                    <p className="font-semibold">
                      {((volumeMetrics.retailVolume! / volumeMetrics.totalVolume) * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      {/* AI Recommendations List */}
      {recommendations.length > 0 && (
        <div className="bg-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="text-yellow-500" />
            <h3 className="text-xl font-bold">AI 추천 종목</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {recommendations.slice(0, 6).map((rec) => (
              <div key={rec.ticker} className="bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors cursor-pointer">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-semibold">{rec.ticker}</h4>
                    <p className="text-sm text-gray-400">{rec.name}</p>
                  </div>
                  <span className={`text-sm px-2 py-1 rounded ${
                    rec.confidence > 0.8 ? 'bg-green-500/20 text-green-400' :
                    rec.confidence > 0.6 ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-gray-500/20 text-gray-400'
                  }`}>
                    {(rec.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-400">목표 수익률</span>
                  <span className="font-semibold text-green-400">
                    +{(rec.expectedReturn * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="mt-2">
                  <p className="text-xs text-gray-400 line-clamp-2">
                    {rec.reasons[0]}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-xl font-bold mb-4">가격 차트</h3>
          <canvas 
            ref={priceChartRef}
            width={600}
            height={300}
            className="w-full bg-gray-900 rounded"
          />
        </div>
        <div className="bg-gray-800 rounded-xl p-6">
          <h3 className="text-xl font-bold mb-4">거래량 차트</h3>
          <canvas 
            ref={volumeChartRef}
            width={600}
            height={300}
            className="w-full bg-gray-900 rounded"
          />
        </div>
      </div>
    </div>
  );
};