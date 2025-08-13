'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { 
  ArrowLeft, 
  ChevronRight, 
  CheckCircle, 
  Circle,
  PlayCircle,
  BookOpen,
  Target,
  Sparkles,
  Clock,
  Trophy,
  Calculator,
  BarChart3,
  TrendingUp,
  DollarSign,
  PieChart,
  Activity,
  Brain,
  AlertTriangle
} from 'lucide-react';

// 심플한 차트 시뮬레이터 컴포넌트
function SimpleChartSimulator() {
  const [currentPrice, setCurrentPrice] = useState(50000);
  const [priceHistory, setPriceHistory] = useState<number[]>([50000]);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPrice(prev => {
        const change = (Math.random() - 0.5) * 1000;
        const newPrice = Math.max(45000, Math.min(55000, prev + change));
        setPriceHistory(history => [...history.slice(-20), newPrice]);
        return newPrice;
      });
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);

  const isUp = priceHistory.length > 1 && currentPrice > priceHistory[priceHistory.length - 2];

  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
      <h3 className="font-semibold mb-4">실시간 차트 시뮬레이터</h3>
      <div className="mb-4">
        <div className="text-2xl font-bold mb-1">
          <span className={isUp ? 'text-red-500' : 'text-blue-500'}>
            ₩{currentPrice.toLocaleString()}
          </span>
        </div>
        <div className={`text-sm ${isUp ? 'text-red-500' : 'text-blue-500'}`}>
          {isUp ? '▲' : '▼'} {Math.abs(currentPrice - priceHistory[priceHistory.length - 2] || 0).toFixed(0)}
        </div>
      </div>
      <div className="h-32 flex items-end gap-1">
        {priceHistory.map((price, i) => {
          const height = ((price - 45000) / 10000) * 100;
          const isLastBar = i === priceHistory.length - 1;
          return (
            <div
              key={i}
              className={`flex-1 ${
                isLastBar ? (isUp ? 'bg-red-500' : 'bg-blue-500') : 'bg-gray-300 dark:bg-gray-600'
              } rounded-t transition-all duration-300`}
              style={{ height: `${height}%` }}
            />
          );
        })}
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
        💡 빨간색은 가격 상승, 파란색은 가격 하락을 의미해요!
      </p>
    </div>
  );
}

// 계좌 개설 시뮬레이터
function AccountOpeningSimulator() {
  const [step, setStep] = useState(0);
  const steps = [
    { title: '증권사 선택', done: false },
    { title: '본인 인증', done: false },
    { title: '정보 입력', done: false },
    { title: '계좌 생성 완료!', done: false }
  ];

  return (
    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">계좌 개설 체험하기</h3>
      <div className="space-y-3">
        {steps.map((s, i) => (
          <div
            key={i}
            className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
              i <= step ? 'bg-white dark:bg-gray-800' : 'opacity-50'
            }`}
          >
            {i < step ? (
              <CheckCircle className="w-5 h-5 text-green-500" />
            ) : i === step ? (
              <Circle className="w-5 h-5 text-blue-500" />
            ) : (
              <Circle className="w-5 h-5 text-gray-400" />
            )}
            <span className={i <= step ? 'font-medium' : ''}>{s.title}</span>
          </div>
        ))}
      </div>
      {step < 3 && (
        <button
          onClick={() => setStep(step + 1)}
          className="mt-4 w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          다음 단계
        </button>
      )}
      {step === 3 && (
        <div className="mt-4 p-3 bg-green-100 dark:bg-green-900/30 rounded-lg text-center">
          <Trophy className="w-8 h-8 text-green-600 mx-auto mb-2" />
          <p className="text-green-700 dark:text-green-300 font-medium">축하합니다! 계좌 개설 완료!</p>
        </div>
      )}
    </div>
  );
}

// 수익률 계산기
function ReturnCalculator() {
  const [investment, setInvestment] = useState(1000000);
  const [returnRate, setReturnRate] = useState(10);
  
  const profit = investment * (returnRate / 100);
  const total = investment + profit;

  return (
    <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">수익률 계산기</h3>
      <div className="space-y-4">
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">투자금액</label>
          <input
            type="number"
            value={investment}
            onChange={(e) => setInvestment(Number(e.target.value))}
            className="w-full mt-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
          />
        </div>
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">수익률 (%)</label>
          <input
            type="range"
            min="-50"
            max="100"
            value={returnRate}
            onChange={(e) => setReturnRate(Number(e.target.value))}
            className="w-full mt-1"
          />
          <div className="text-center mt-1">
            <span className={`font-bold ${returnRate >= 0 ? 'text-red-500' : 'text-blue-500'}`}>
              {returnRate > 0 ? '+' : ''}{returnRate}%
            </span>
          </div>
        </div>
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex justify-between mb-2">
            <span>투자금액:</span>
            <span>₩{investment.toLocaleString()}</span>
          </div>
          <div className="flex justify-between mb-2">
            <span>수익/손실:</span>
            <span className={profit >= 0 ? 'text-red-500' : 'text-blue-500'}>
              {profit >= 0 ? '+' : ''}₩{profit.toLocaleString()}
            </span>
          </div>
          <div className="flex justify-between font-bold text-lg">
            <span>총 평가금액:</span>
            <span>₩{total.toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// 캔들스틱 차트 읽기 시뮬레이터 (Basic Track)
function CandleChartSimulator() {
  const [showAnswer, setShowAnswer] = useState(false);
  const [currentPattern, setCurrentPattern] = useState(0);
  
  const patterns = [
    { 
      name: '양봉', 
      desc: '시가보다 종가가 높은 경우',
      color: 'bg-red-500',
      openPrice: 45000,
      closePrice: 50000,
      highPrice: 52000,
      lowPrice: 44000
    },
    { 
      name: '음봉', 
      desc: '시가보다 종가가 낮은 경우',
      color: 'bg-blue-500',
      openPrice: 50000,
      closePrice: 45000,
      highPrice: 51000,
      lowPrice: 43000
    }
  ];

  const current = patterns[currentPattern];
  const bodyHeight = Math.abs(current.closePrice - current.openPrice) / 100;
  const wickTop = (current.highPrice - Math.max(current.openPrice, current.closePrice)) / 100;
  const wickBottom = (Math.min(current.openPrice, current.closePrice) - current.lowPrice) / 100;

  return (
    <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">캔들스틱 차트 읽기 연습</h3>
      
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-4">
        <div className="flex justify-center mb-4">
          <div className="relative" style={{ height: '200px', width: '60px' }}>
            {/* 위 꼬리 */}
            <div 
              className="absolute left-1/2 transform -translate-x-1/2 w-0.5 bg-gray-600"
              style={{ 
                top: `${50 - wickTop - bodyHeight/2}%`,
                height: `${wickTop}%`
              }}
            />
            {/* 몸통 */}
            <div 
              className={`absolute left-0 right-0 ${current.color} rounded`}
              style={{ 
                top: `${50 - bodyHeight/2}%`,
                height: `${bodyHeight}%`
              }}
            />
            {/* 아래 꼬리 */}
            <div 
              className="absolute left-1/2 transform -translate-x-1/2 w-0.5 bg-gray-600"
              style={{ 
                top: `${50 + bodyHeight/2}%`,
                height: `${wickBottom}%`
              }}
            />
          </div>
        </div>
        
        <div className="text-center">
          <p className="text-lg font-medium mb-2">이 캔들은 무엇을 의미할까요?</p>
          {!showAnswer ? (
            <button
              onClick={() => setShowAnswer(true)}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              정답 확인
            </button>
          ) : (
            <div className="space-y-2">
              <p className="text-xl font-bold text-purple-600">{current.name}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">{current.desc}</p>
              <div className="grid grid-cols-2 gap-2 mt-4 text-sm">
                <div>시가: ₩{current.openPrice.toLocaleString()}</div>
                <div>종가: ₩{current.closePrice.toLocaleString()}</div>
                <div>고가: ₩{current.highPrice.toLocaleString()}</div>
                <div>저가: ₩{current.lowPrice.toLocaleString()}</div>
              </div>
              <button
                onClick={() => {
                  setCurrentPattern((currentPattern + 1) % patterns.length);
                  setShowAnswer(false);
                }}
                className="mt-4 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                다음 패턴
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// 이동평균선 시뮬레이터 (Basic Track)
function MovingAverageSimulator() {
  const [showMA, setShowMA] = useState({ ma5: false, ma20: false, ma60: false });
  const prices = [48000, 49000, 47500, 50000, 51000, 49500, 52000, 53000, 51500, 54000, 
                  55000, 53500, 56000, 54500, 57000, 58000, 56500, 59000, 60000, 58500];
  
  const calculateMA = (period: number) => {
    return prices.map((_, index) => {
      if (index < period - 1) return null;
      const sum = prices.slice(index - period + 1, index + 1).reduce((a, b) => a + b, 0);
      return sum / period;
    });
  };

  const ma5 = calculateMA(5);
  const ma20 = calculateMA(20);

  return (
    <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">이동평균선 이해하기</h3>
      
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
        <div className="h-48 relative mb-4">
          {/* 가격 차트 */}
          <svg className="w-full h-full">
            {/* 가격선 */}
            <polyline
              points={prices.map((price, i) => `${i * 100 / 19},${200 - (price - 45000) / 150}`).join(' ')}
              fill="none"
              stroke="rgb(59, 130, 246)"
              strokeWidth="2"
            />
            
            {/* 5일 이동평균선 */}
            {showMA.ma5 && (
              <polyline
                points={ma5.map((price, i) => price ? `${i * 100 / 19},${200 - (price - 45000) / 150}` : '').filter(p => p).join(' ')}
                fill="none"
                stroke="rgb(239, 68, 68)"
                strokeWidth="2"
                strokeDasharray="5,5"
              />
            )}
            
            {/* 20일 이동평균선 */}
            {showMA.ma20 && (
              <polyline
                points={ma20.map((price, i) => price ? `${i * 100 / 19},${200 - (price - 45000) / 150}` : '').filter(p => p).join(' ')}
                fill="none"
                stroke="rgb(34, 197, 94)"
                strokeWidth="2"
                strokeDasharray="5,5"
              />
            )}
          </svg>
        </div>
        
        <div className="space-y-2">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showMA.ma5}
              onChange={(e) => setShowMA({ ...showMA, ma5: e.target.checked })}
              className="rounded"
            />
            <span className="text-red-500">5일 이동평균선</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showMA.ma20}
              onChange={(e) => setShowMA({ ...showMA, ma20: e.target.checked })}
              className="rounded"
            />
            <span className="text-green-500">20일 이동평균선</span>
          </label>
        </div>
      </div>
      
      <div className="text-sm text-gray-600 dark:text-gray-400">
        💡 이동평균선은 일정 기간 동안의 평균 가격을 연결한 선입니다.
        단기 이동평균선이 장기 이동평균선을 상향 돌파하면 '골든크로스'라고 해요!
      </div>
    </div>
  );
}

// 캔들스틱 시뮬레이터 (Basic Track)
function CandlePatternSimulator() {
  const [candles, setCandles] = useState([
    { open: 50000, close: 52000, high: 53000, low: 49500, type: 'bullish' },
    { open: 52000, close: 51000, high: 52500, low: 50500, type: 'bearish' },
    { open: 51000, close: 53500, high: 54000, low: 50800, type: 'bullish' },
    { open: 53500, close: 52500, high: 54000, low: 52000, type: 'bearish' },
    { open: 52500, close: 55000, high: 55500, low: 52000, type: 'bullish' }
  ]);

  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
      <h3 className="font-semibold mb-4">캔들스틱 패턴 연습</h3>
      
      <div className="h-64 flex items-end gap-2 mb-4">
        {candles.map((candle, i) => {
          const maxPrice = Math.max(...candles.map(c => c.high));
          const minPrice = Math.min(...candles.map(c => c.low));
          const range = maxPrice - minPrice;
          
          const bodyHeight = Math.abs(candle.close - candle.open) / range * 100;
          const bodyBottom = (Math.min(candle.close, candle.open) - minPrice) / range * 100;
          const upperShadow = (candle.high - Math.max(candle.close, candle.open)) / range * 100;
          const lowerShadow = (Math.min(candle.close, candle.open) - candle.low) / range * 100;
          
          return (
            <div key={i} className="flex-1 relative h-full">
              {/* Upper shadow */}
              <div 
                className="absolute w-0.5 bg-gray-400 left-1/2 -translate-x-1/2"
                style={{ 
                  bottom: `${bodyBottom + bodyHeight}%`,
                  height: `${upperShadow}%`
                }}
              />
              
              {/* Body */}
              <div 
                className={`absolute w-full ${
                  candle.type === 'bullish' ? 'bg-red-500' : 'bg-blue-500'
                } rounded`}
                style={{ 
                  bottom: `${bodyBottom}%`,
                  height: `${bodyHeight}%`
                }}
              />
              
              {/* Lower shadow */}
              <div 
                className="absolute w-0.5 bg-gray-400 left-1/2 -translate-x-1/2"
                style={{ 
                  bottom: `${lowerShadow}%`,
                  height: `${bodyBottom - lowerShadow}%`
                }}
              />
            </div>
          );
        })}
      </div>
      
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded"></div>
          <span>양봉 (상승)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-500 rounded"></div>
          <span>음봉 (하락)</span>
        </div>
      </div>
      
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
        💡 긴 몸통은 강한 매수/매도세를 나타내고, 긴 꼬리는 가격 거부를 의미합니다.
      </p>
    </div>
  );
}

// PER/PBR 계산기 (Intermediate Track)
function ValuationCalculator() {
  const [stockPrice, setStockPrice] = useState(50000);
  const [eps, setEps] = useState(5000);
  const [bps, setBps] = useState(40000);
  
  const per = stockPrice / eps;
  const pbr = stockPrice / bps;

  return (
    <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">PER/PBR 계산기</h3>
      
      <div className="space-y-4">
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">주가</label>
          <input
            type="number"
            value={stockPrice}
            onChange={(e) => setStockPrice(Number(e.target.value))}
            className="w-full mt-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
          />
        </div>
        
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">주당순이익 (EPS)</label>
          <input
            type="number"
            value={eps}
            onChange={(e) => setEps(Number(e.target.value))}
            className="w-full mt-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
          />
        </div>
        
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">주당순자산 (BPS)</label>
          <input
            type="number"
            value={bps}
            onChange={(e) => setBps(Number(e.target.value))}
            className="w-full mt-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
          />
        </div>
        
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">PER (주가수익비율)</p>
              <p className="text-2xl font-bold">{per.toFixed(2)}배</p>
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                {per < 10 ? '저평가' : per < 20 ? '적정' : '고평가'} 수준
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">PBR (주가순자산비율)</p>
              <p className="text-2xl font-bold">{pbr.toFixed(2)}배</p>
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                {pbr < 1 ? '저평가' : pbr < 2 ? '적정' : '고평가'} 수준
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// 포트폴리오 구성 시뮬레이터 (Intermediate Track)
function PortfolioSimulator() {
  const [portfolio, setPortfolio] = useState([
    { name: '삼성전자', weight: 30, return: 15 },
    { name: 'SK하이닉스', weight: 20, return: 25 },
    { name: '네이버', weight: 20, return: -5 },
    { name: '카카오', weight: 20, return: 10 },
    { name: '현금', weight: 10, return: 3 }
  ]);
  
  const totalReturn = portfolio.reduce((sum, item) => sum + (item.weight * item.return / 100), 0);
  
  const updateWeight = (index: number, newWeight: number) => {
    const newPortfolio = [...portfolio];
    newPortfolio[index].weight = newWeight;
    
    // 가중치 합계를 100%로 조정
    const totalWeight = newPortfolio.reduce((sum, item) => sum + item.weight, 0);
    if (totalWeight !== 100) {
      const scale = 100 / totalWeight;
      newPortfolio.forEach(item => item.weight = Math.round(item.weight * scale));
    }
    
    setPortfolio(newPortfolio);
  };

  return (
    <div className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">포트폴리오 구성하기</h3>
      
      <div className="space-y-3 mb-4">
        {portfolio.map((item, index) => (
          <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <span className="font-medium">{item.name}</span>
              <span className={`text-sm ${item.return >= 0 ? 'text-red-500' : 'text-blue-500'}`}>
                {item.return > 0 ? '+' : ''}{item.return}%
              </span>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0"
                max="100"
                value={item.weight}
                onChange={(e) => updateWeight(index, Number(e.target.value))}
                className="flex-1"
              />
              <span className="w-12 text-right">{item.weight}%</span>
            </div>
          </div>
        ))}
      </div>
      
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">포트폴리오 예상 수익률</p>
        <p className={`text-2xl font-bold ${totalReturn >= 0 ? 'text-red-500' : 'text-blue-500'}`}>
          {totalReturn > 0 ? '+' : ''}{totalReturn.toFixed(2)}%
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
          💡 분산투자로 리스크를 줄일 수 있어요!
        </p>
      </div>
    </div>
  );
}

// RSI 지표 시뮬레이터 (Advanced Track)
function RSISimulator() {
  const [rsiValue, setRsiValue] = useState(50);
  const [showSignal, setShowSignal] = useState(false);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setRsiValue(prev => {
        const change = (Math.random() - 0.5) * 10;
        return Math.max(0, Math.min(100, prev + change));
      });
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  const getSignal = () => {
    if (rsiValue > 70) return { text: '과매수', color: 'text-red-500', action: '매도 신호' };
    if (rsiValue < 30) return { text: '과매도', color: 'text-blue-500', action: '매수 신호' };
    return { text: '중립', color: 'text-gray-500', action: '관망' };
  };
  
  const signal = getSignal();

  return (
    <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">RSI (상대강도지수) 실시간 모니터</h3>
      
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-2">
            <span>0 (과매도)</span>
            <span>50</span>
            <span>100 (과매수)</span>
          </div>
          <div className="relative h-8 bg-gradient-to-r from-blue-500 via-gray-300 to-red-500 rounded-full">
            <div 
              className="absolute top-1/2 transform -translate-y-1/2 w-4 h-4 bg-white border-2 border-gray-800 rounded-full transition-all duration-300"
              style={{ left: `${rsiValue}%`, marginLeft: '-8px' }}
            />
          </div>
          <div className="flex justify-between mt-2">
            <div className="w-0.5 h-4 bg-blue-600" />
            <div className="w-0.5 h-4 bg-gray-400" />
            <div className="w-0.5 h-4 bg-red-600" />
          </div>
        </div>
        
        <div className="text-center">
          <p className="text-3xl font-bold mb-2">{rsiValue.toFixed(1)}</p>
          <p className={`text-lg font-medium ${signal.color}`}>{signal.text}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{signal.action}</p>
        </div>
        
        <button
          onClick={() => setShowSignal(!showSignal)}
          className="w-full mt-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
        >
          {showSignal ? '설명 숨기기' : 'RSI란?'}
        </button>
        
        {showSignal && (
          <div className="mt-4 p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg text-sm">
            <p>RSI는 0~100 사이의 값으로 표시되며:</p>
            <ul className="mt-2 space-y-1">
              <li>• 70 이상: 과매수 (하락 가능성)</li>
              <li>• 30 이하: 과매도 (상승 가능성)</li>
              <li>• 50 근처: 중립</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

// 볼린저 밴드 시뮬레이터 (Advanced Track)
function BollingerBandsSimulator() {
  const [price, setPrice] = useState(50000);
  const [volatility, setVolatility] = useState(1000);
  const ma = 50000;
  const upperBand = ma + (volatility * 2);
  const lowerBand = ma - (volatility * 2);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setPrice(prev => {
        const change = (Math.random() - 0.5) * 2000;
        return Math.max(lowerBand - 1000, Math.min(upperBand + 1000, prev + change));
      });
    }, 500);
    
    return () => clearInterval(interval);
  }, [lowerBand, upperBand]);
  
  const getSignal = () => {
    if (price > upperBand) return { text: '상단 밴드 돌파', color: 'text-red-500', signal: '과열' };
    if (price < lowerBand) return { text: '하단 밴드 돌파', color: 'text-blue-500', signal: '과매도' };
    return { text: '밴드 내 움직임', color: 'text-green-500', signal: '정상' };
  };
  
  const signal = getSignal();

  return (
    <div className="bg-pink-50 dark:bg-pink-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">볼린저 밴드 시뮬레이터</h3>
      
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
        <div className="relative h-48 mb-4">
          <div className="absolute w-full h-0.5 bg-red-500" style={{ top: '20%' }} />
          <div className="absolute w-full h-0.5 bg-gray-500" style={{ top: '50%' }} />
          <div className="absolute w-full h-0.5 bg-blue-500" style={{ top: '80%' }} />
          
          <div 
            className="absolute w-4 h-4 bg-purple-600 rounded-full transform -translate-x-1/2"
            style={{ 
              left: '50%',
              top: `${100 - ((price - (lowerBand - 1000)) / ((upperBand + 1000) - (lowerBand - 1000)) * 100)}%`
            }}
          />
          
          <div className="absolute right-0 text-xs space-y-12">
            <span className="text-red-500">상단: ₩{upperBand.toLocaleString()}</span>
            <span className="text-gray-500">중심: ₩{ma.toLocaleString()}</span>
            <span className="text-blue-500">하단: ₩{lowerBand.toLocaleString()}</span>
          </div>
        </div>
        
        <div className="text-center">
          <p className="text-2xl font-bold mb-1">₩{price.toLocaleString()}</p>
          <p className={`font-medium ${signal.color}`}>{signal.text}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">{signal.signal}</p>
        </div>
        
        <div className="mt-4">
          <label className="text-sm text-gray-600 dark:text-gray-400">변동성 조절</label>
          <input
            type="range"
            min="500"
            max="2000"
            value={volatility}
            onChange={(e) => setVolatility(Number(e.target.value))}
            className="w-full mt-1"
          />
        </div>
      </div>
    </div>
  );
}

// DCF 가치평가 시뮬레이터 (Professional Track)
function DCFCalculator() {
  const [fcf, setFcf] = useState(1000); // 억원
  const [growthRate, setGrowthRate] = useState(5); // %
  const [discountRate, setDiscountRate] = useState(10); // %
  const [terminalGrowth, setTerminalGrowth] = useState(2); // %
  
  const calculateDCF = () => {
    let totalPV = 0;
    let currentFCF = fcf;
    
    // 5년간 FCF 현재가치
    for (let i = 1; i <= 5; i++) {
      currentFCF = currentFCF * (1 + growthRate / 100);
      const pv = currentFCF / Math.pow(1 + discountRate / 100, i);
      totalPV += pv;
    }
    
    // Terminal Value
    const terminalFCF = currentFCF * (1 + terminalGrowth / 100);
    const terminalValue = terminalFCF / ((discountRate - terminalGrowth) / 100);
    const terminalPV = terminalValue / Math.pow(1 + discountRate / 100, 5);
    
    return totalPV + terminalPV;
  };
  
  const enterpriseValue = calculateDCF();

  return (
    <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">DCF 가치평가 모델</h3>
      
      <div className="space-y-4">
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">연간 잉여현금흐름 (억원)</label>
          <input
            type="number"
            value={fcf}
            onChange={(e) => setFcf(Number(e.target.value))}
            className="w-full mt-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
          />
        </div>
        
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">예상 성장률 (5년, %)</label>
          <input
            type="range"
            min="0"
            max="20"
            step="0.5"
            value={growthRate}
            onChange={(e) => setGrowthRate(Number(e.target.value))}
            className="w-full mt-1"
          />
          <div className="text-center text-sm">{growthRate}%</div>
        </div>
        
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">할인율 (WACC, %)</label>
          <input
            type="range"
            min="5"
            max="20"
            step="0.5"
            value={discountRate}
            onChange={(e) => setDiscountRate(Number(e.target.value))}
            className="w-full mt-1"
          />
          <div className="text-center text-sm">{discountRate}%</div>
        </div>
        
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400">영구성장률 (%)</label>
          <input
            type="range"
            min="0"
            max="5"
            step="0.5"
            value={terminalGrowth}
            onChange={(e) => setTerminalGrowth(Number(e.target.value))}
            className="w-full mt-1"
          />
          <div className="text-center text-sm">{terminalGrowth}%</div>
        </div>
        
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">기업가치 (Enterprise Value)</p>
            <p className="text-3xl font-bold text-emerald-600">
              {enterpriseValue.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}억원
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
              * 순부채를 차감하면 주식가치가 됩니다
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// 리스크 관리 시뮬레이터 (Professional Track)
function RiskManagementSimulator() {
  const [portfolio, setPortfolio] = useState([
    { asset: '국내주식', weight: 40, risk: 20, return: 8 },
    { asset: '해외주식', weight: 30, risk: 25, return: 10 },
    { asset: '채권', weight: 20, risk: 5, return: 3 },
    { asset: '대체투자', weight: 10, risk: 15, return: 6 }
  ]);
  
  const [riskTolerance, setRiskTolerance] = useState(15);
  
  const portfolioRisk = Math.sqrt(
    portfolio.reduce((sum, item) => sum + Math.pow(item.weight * item.risk / 100, 2), 0)
  );
  
  const portfolioReturn = portfolio.reduce((sum, item) => sum + (item.weight * item.return / 100), 0);
  const sharpeRatio = (portfolioReturn - 2) / portfolioRisk; // 무위험수익률 2% 가정

  return (
    <div className="bg-sky-50 dark:bg-sky-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">포트폴리오 리스크 관리</h3>
      
      <div className="mb-4">
        <label className="text-sm text-gray-600 dark:text-gray-400">위험 허용도</label>
        <input
          type="range"
          min="5"
          max="30"
          value={riskTolerance}
          onChange={(e) => setRiskTolerance(Number(e.target.value))}
          className="w-full mt-1"
        />
        <div className="text-center text-sm">{riskTolerance}%</div>
      </div>
      
      <div className="space-y-2 mb-4">
        {portfolio.map((item, index) => (
          <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-3">
            <div className="flex justify-between items-center">
              <span className="font-medium">{item.asset}</span>
              <div className="text-sm space-x-4">
                <span>비중: {item.weight}%</span>
                <span className="text-orange-500">위험: {item.risk}%</span>
                <span className="text-green-500">수익: {item.return}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">포트폴리오 위험도</p>
          <p className={`text-2xl font-bold ${portfolioRisk > riskTolerance ? 'text-red-500' : 'text-green-500'}`}>
            {portfolioRisk.toFixed(1)}%
          </p>
          {portfolioRisk > riskTolerance && (
            <p className="text-xs text-red-500 mt-1">위험 허용도 초과!</p>
          )}
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">샤프 비율</p>
          <p className="text-2xl font-bold">{sharpeRatio.toFixed(2)}</p>
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
            {sharpeRatio > 1 ? '우수' : sharpeRatio > 0.5 ? '양호' : '개선 필요'}
          </p>
        </div>
      </div>
    </div>
  );
}

export default function LearningTrackPage() {
  const params = useParams();
  const trackId = params.trackId as string;
  const [currentSection, setCurrentSection] = useState(0);
  const [completedSections, setCompletedSections] = useState<Set<number>>(new Set());
  const contentRef = useRef<HTMLDivElement>(null);

  // 학습 트랙별 콘텐츠 정의
  const tracks = {
    beginner: {
      title: '주식 투자 첫걸음',
      sections: [
        {
          title: '주식이 도대체 뭔가요?',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">주식이 도대체 뭔가요? 🤔</h2>
              
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🍕 피자로 이해하는 주식</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  친구들과 피자 가게를 차리려고 한다고 상상해보세요. 
                  혼자서는 돈이 부족해서 친구 4명이 각자 돈을 모았습니다.
                </p>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">피자 가게 = 회사</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      여러분이 차린 피자 가게가 바로 "회사"예요
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">피자 조각 = 주식</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      가게를 5조각으로 나눈 것이 바로 "주식"이에요
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💰 주식을 가지면 뭐가 좋아요?</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-3">
                    <span className="text-2xl">1️⃣</span>
                    <div>
                      <strong>주인이 됩니다</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        주식을 가진 만큼 그 회사의 주인이 됩니다
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-2xl">2️⃣</span>
                    <div>
                      <strong>이익을 나눠 받을 수 있어요</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        회사가 돈을 벌면 배당금을 받을 수 있어요
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-2xl">3️⃣</span>
                    <div>
                      <strong>비싸게 팔 수 있어요</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        회사가 성장하면 주식 가격도 올라가요
                      </p>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          ),
          simulator: <ReturnCalculator />
        },
        {
          title: '증권 계좌 만들기',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">증권 계좌 만들기 A to Z 🏦</h2>
              
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📱 어떤 증권사를 선택할까?</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">대형 증권사</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• 삼성증권</li>
                      <li>• NH투자증권</li>
                      <li>• 미래에셋증권</li>
                    </ul>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">모바일 증권사</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• 토스증권</li>
                      <li>• 카카오페이증권</li>
                      <li>• 네이버증권</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💳 준비물</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">🪪</span>
                    <span>신분증</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">🏦</span>
                    <span>은행 계좌</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">📱</span>
                    <span>휴대폰</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">🎂</span>
                    <span>만 19세 이상</span>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <AccountOpeningSimulator />
        },
        {
          title: '주식 시장의 기초',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">주식시장은 어떻게 돌아갈까? 🏛️</h2>
              
              <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">⏰ 주식시장 운영 시간</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">정규 거래</span>
                      <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
                        오전 9시 ~ 오후 3시 30분
                      </span>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">시간외 거래</span>
                      <span className="text-sm">
                        오전 8:30~9:00, 오후 3:40~4:00
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📈 가격은 어떻게 결정될까?</h3>
                <div className="text-center mb-4">
                  <p className="text-lg font-medium">수요와 공급의 법칙</p>
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                    <div className="text-2xl mb-2">📈</div>
                    <p className="font-medium text-red-600 dark:text-red-400">가격 상승</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      사고 싶은 사람 {'>'} 팔고 싶은 사람
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                    <div className="text-2xl mb-2">📉</div>
                    <p className="font-medium text-blue-600 dark:text-blue-400">가격 하락</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      팔고 싶은 사람 {'>'} 사고 싶은 사람
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <SimpleChartSimulator />
        }
      ]
    },
    basic: {
      title: '차트 읽기 기초',
      sections: [
        {
          title: '캔들스틱 차트의 이해',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">캔들스틱 차트 마스터하기 🕯️</h2>
              
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 캔들스틱의 구성요소</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">몸통 (Body)</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      시가와 종가 사이의 가격을 나타냅니다.
                      빨간색은 상승, 파란색은 하락을 의미해요.
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">꼬리 (Shadow/Wick)</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      하루 중 최고가와 최저가를 보여줍니다.
                      가격의 변동폭을 알 수 있어요.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🎨 색상의 의미</h3>
                <div className="space-y-3">
                  <div className="flex items-center gap-4 bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="w-12 h-12 bg-red-500 rounded"></div>
                    <div>
                      <p className="font-medium">빨간색 (양봉)</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        종가가 시가보다 높아요. 가격이 올랐다는 뜻!
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="w-12 h-12 bg-blue-500 rounded"></div>
                    <div>
                      <p className="font-medium">파란색 (음봉)</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        종가가 시가보다 낮아요. 가격이 내렸다는 뜻!
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-pink-50 dark:bg-pink-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💡 주요 패턴들</h3>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2">
                    <span>•</span>
                    <div>
                      <strong>장대양봉:</strong> 큰 빨간 몸통, 강한 상승 신호
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <span>•</span>
                    <div>
                      <strong>장대음봉:</strong> 큰 파란 몸통, 강한 하락 신호
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <span>•</span>
                    <div>
                      <strong>도지(Doji):</strong> 시가와 종가가 비슷, 추세 전환 가능성
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          ),
          simulator: <CandleChartSimulator />
        },
        {
          title: '이동평균선과 거래량',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">이동평균선과 거래량 분석 📈</h2>
              
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📏 이동평균선이란?</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  일정 기간 동안의 평균 주가를 연결한 선으로, 주가의 추세를 파악하는 데 사용해요.
                </p>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-red-600 dark:text-red-400">5일선 (단기)</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      최근 5일간의 평균 가격. 단기 추세를 보여줍니다.
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-green-600 dark:text-green-400">20일선 (중기)</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      최근 20일간의 평균. 중기 추세를 나타냅니다.
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-purple-600 dark:text-purple-400">60일선 (장기)</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      최근 60일간의 평균. 장기 추세를 보여줍니다.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🔄 골든크로스와 데드크로스</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium text-yellow-600 dark:text-yellow-400 mb-2">🌟 골든크로스</p>
                    <p className="text-sm">
                      단기 이동평균선이 장기 이동평균선을 위로 돌파!
                      <span className="block mt-1 text-green-600">→ 매수 신호</span>
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium text-gray-600 dark:text-gray-400 mb-2">💀 데드크로스</p>
                    <p className="text-sm">
                      단기 이동평균선이 장기 이동평균선을 아래로 돌파!
                      <span className="block mt-1 text-red-600">→ 매도 신호</span>
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 거래량의 중요성</h3>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2">
                    <span className="text-green-500">✓</span>
                    <span>가격 상승 + 거래량 증가 = 강한 상승 신호</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500">✓</span>
                    <span>가격 하락 + 거래량 증가 = 강한 하락 신호</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-orange-500">⚠️</span>
                    <span>가격 변동 + 거래량 감소 = 추세 약화 가능성</span>
                  </li>
                </ul>
              </div>
            </div>
          ),
          simulator: <MovingAverageSimulator />
        },
        {
          title: '지지선과 저항선',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">지지선과 저항선 찾기 🚧</h2>
              
              <div className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🛡️ 지지선 (Support)</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  주가가 하락하다가 멈추는 가격대. 이 선에서 매수세가 강해져요.
                </p>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="text-sm">💡 특징:</p>
                  <ul className="mt-2 space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <li>• 과거에 여러 번 반등했던 가격대</li>
                    <li>• 많은 투자자들이 매수하고 싶어하는 가격</li>
                    <li>• 돌파되면 더 큰 하락 가능성</li>
                  </ul>
                </div>
              </div>

              <div className="bg-rose-50 dark:bg-rose-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">⛔ 저항선 (Resistance)</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  주가가 상승하다가 멈추는 가격대. 이 선에서 매도세가 강해져요.
                </p>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="text-sm">💡 특징:</p>
                  <ul className="mt-2 space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <li>• 과거에 여러 번 하락했던 가격대</li>
                    <li>• 많은 투자자들이 매도하고 싶어하는 가격</li>
                    <li>• 돌파되면 더 큰 상승 가능성</li>
                  </ul>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🔄 역할 전환</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  지지선이 뚫리면 저항선이 되고, 저항선이 뚫리면 지지선이 됩니다!
                </p>
                <div className="text-center">
                  <div className="inline-block bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="text-sm">저항선 → 돌파 → 지지선 ✨</p>
                    <p className="text-sm mt-2">지지선 → 하향 돌파 → 저항선 💥</p>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <SimpleChartSimulator />
        }
      ]
    },
    intermediate: {
      title: '똑똑한 투자자 되기',
      sections: [
        {
          title: '기업 분석의 기초',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">기업을 분석하는 방법 🔍</h2>
              
              <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 재무제표 읽기</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">손익계산서</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      회사가 얼마나 벌고 썼는지 보여주는 성적표
                    </p>
                    <div className="mt-2 text-sm">
                      <p>• 매출액: 회사가 번 총 돈</p>
                      <p>• 영업이익: 본업으로 번 돈</p>
                      <p>• 순이익: 세금까지 낸 후 남은 돈</p>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">재무상태표</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      회사가 가진 재산과 빚을 보여주는 장부
                    </p>
                    <div className="mt-2 text-sm">
                      <p>• 자산: 회사가 가진 모든 것</p>
                      <p>• 부채: 회사가 갚아야 할 돈</p>
                      <p>• 자본: 자산에서 부채를 뺀 순재산</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💼 사업 모델 이해하기</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">1️⃣</span>
                    <div>
                      <p className="font-medium">무엇을 파는 회사인가?</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        주력 상품이나 서비스가 무엇인지 파악하세요
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">2️⃣</span>
                    <div>
                      <p className="font-medium">누구에게 파는가?</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        주요 고객층과 시장을 이해하세요
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">3️⃣</span>
                    <div>
                      <p className="font-medium">경쟁력은 무엇인가?</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        다른 회사와 차별화되는 강점을 찾으세요
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <ValuationCalculator />
        },
        {
          title: 'PER과 PBR로 가치평가',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">주식 가치평가 지표 이해하기 📐</h2>
              
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📈 PER (주가수익비율)</h3>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-3">
                  <p className="font-medium text-center text-lg mb-2">
                    PER = 주가 ÷ 주당순이익(EPS)
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                    "이 주식을 사면 투자금을 몇 년 만에 회수할 수 있을까?"
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm">• PER 10배 = 10년이면 투자금 회수 가능</p>
                  <p className="text-sm">• 낮을수록 저평가, 높을수록 고평가</p>
                  <p className="text-sm">• 업종별로 평균 PER이 다름</p>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 PBR (주가순자산비율)</h3>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-3">
                  <p className="font-medium text-center text-lg mb-2">
                    PBR = 주가 ÷ 주당순자산(BPS)
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                    "회사가 망해도 얼마나 돌려받을 수 있을까?"
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm">• PBR 1배 = 회사 청산 시 투자금 전액 회수</p>
                  <p className="text-sm">• 1배 미만 = 청산가치보다 싸게 거래</p>
                  <p className="text-sm">• 성장주는 보통 PBR이 높음</p>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">⚖️ 적정 수준 판단하기</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">동종업계 비교</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      같은 업종 내 다른 기업들과 비교해보세요
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">과거 평균과 비교</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      해당 기업의 과거 5년 평균과 비교해보세요
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <ValuationCalculator />
        },
        {
          title: '포트폴리오 구성 전략',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">균형잡힌 포트폴리오 만들기 🎯</h2>
              
              <div className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🥚 계란을 한 바구니에 담지 마라</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  분산투자는 리스크를 줄이는 가장 기본적인 방법입니다.
                </p>
                <div className="grid md:grid-cols-3 gap-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3 text-center">
                    <p className="font-medium mb-1">업종 분산</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      IT, 제조, 금융 등
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3 text-center">
                    <p className="font-medium mb-1">지역 분산</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      국내주, 해외주
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3 text-center">
                    <p className="font-medium mb-1">자산 분산</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      주식, 채권, 현금
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🎯 투자 목적에 맞는 구성</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium text-blue-600 dark:text-blue-400 mb-2">안정형 (보수적)</p>
                    <div className="flex items-center gap-4">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
                        <div className="bg-blue-500 h-full" style={{ width: '30%' }}></div>
                      </div>
                      <span className="text-sm">주식 30%</span>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      나머지는 채권, 예금 등 안전자산
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium text-green-600 dark:text-green-400 mb-2">균형형</p>
                    <div className="flex items-center gap-4">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
                        <div className="bg-green-500 h-full" style={{ width: '60%' }}></div>
                      </div>
                      <span className="text-sm">주식 60%</span>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium text-red-600 dark:text-red-400 mb-2">공격형</p>
                    <div className="flex items-center gap-4">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
                        <div className="bg-red-500 h-full" style={{ width: '80%' }}></div>
                      </div>
                      <span className="text-sm">주식 80%+</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📅 리밸런싱</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-3">
                  정기적으로 포트폴리오 비중을 조정하세요
                </p>
                <ul className="space-y-2 text-sm">
                  <li>• 분기별 또는 반기별로 점검</li>
                  <li>• 목표 비중에서 10% 이상 벗어나면 조정</li>
                  <li>• 비싸진 자산은 팔고, 싸진 자산은 매수</li>
                </ul>
              </div>
            </div>
          ),
          simulator: (
            <div className="space-y-6">
              {/* 간단한 포트폴리오 시뮬레이터 (임베드) */}
              <PortfolioSimulator />
              
              {/* 전문 포트폴리오 도구 링크 */}
              <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <PieChart className="w-6 h-6 text-green-600" />
                  전문가급 포트폴리오 도구
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <Link
                    href="/modules/stock-analysis/simulators/portfolio-optimizer"
                    className="group bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-green-300 dark:hover:border-green-600 transition-all duration-200"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <PieChart className="w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 dark:text-white group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors">
                          마코위츠 포트폴리오 최적화기
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          효율적 프론티어 계산
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      • 최적 자산 배분<br/>
                      • 리스크-수익 분석<br/>
                      • 상관관계 매트릭스
                    </p>
                  </Link>

                  <Link
                    href="/modules/stock-analysis/simulators/backtesting-engine"
                    className="group bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-teal-300 dark:hover:border-teal-600 transition-all duration-200"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900/30 text-teal-600 dark:text-teal-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <Activity className="w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">
                          전문가급 백테스팅 엔진
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          전략 성과 검증
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      • 몬테카를로 시뮬레이션<br/>
                      • Walk-Forward 검증<br/>
                      • 슬리피지 반영
                    </p>
                  </Link>
                </div>
              </div>
            </div>
          )
        }
      ]
    },
    advanced: {
      title: '기술적 분석 마스터',
      sections: [
        {
          title: '보조지표 활용하기',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">주요 보조지표 완벽 정복 📊</h2>
              
              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📈 RSI (상대강도지수)</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  주가의 상승압력과 하락압력 간의 상대적인 강도를 나타내는 지표
                </p>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="font-medium mb-2">활용법:</p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <span className="text-red-500">▲</span>
                      <span>70 이상: 과매수 구간 → 조정 가능성 높음</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500">▼</span>
                      <span>30 이하: 과매도 구간 → 반등 가능성 높음</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500">●</span>
                      <span>다이버전스: 가격과 RSI가 반대로 움직일 때 추세 전환 신호</span>
                    </li>
                  </ul>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 MACD</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  단기 이동평균과 장기 이동평균의 차이를 이용한 추세 추종 지표
                </p>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">구성요소:</p>
                    <ul className="space-y-1 text-sm">
                      <li>• MACD선: 12일 EMA - 26일 EMA</li>
                      <li>• 시그널선: MACD의 9일 EMA</li>
                      <li>• 히스토그램: MACD선 - 시그널선</li>
                    </ul>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">매매 신호:</p>
                    <ul className="space-y-1 text-sm">
                      <li>• 골든크로스: MACD가 시그널선 상향 돌파 → 매수</li>
                      <li>• 데드크로스: MACD가 시그널선 하향 돌파 → 매도</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📉 스토캐스틱</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  일정 기간 동안의 최고가와 최저가 범위 내에서 현재가의 위치를 표시
                </p>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <p className="font-medium text-sm mb-1">%K선 (빠른선)</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        현재가의 상대적 위치
                      </p>
                    </div>
                    <div>
                      <p className="font-medium text-sm mb-1">%D선 (느린선)</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        %K의 3일 이동평균
                      </p>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-sm">
                      💡 20 이하에서 %K가 %D를 상향 돌파 → 매수 신호
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <RSISimulator />
        },
        {
          title: '차트 패턴 분석',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">차트 패턴으로 미래 예측하기 🔮</h2>
              
              <div className="bg-pink-50 dark:bg-pink-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📐 추세 전환 패턴</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">헤드앤숄더 (머리어깨형)</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      상승 추세의 끝에서 나타나는 하락 전환 패턴
                    </p>
                    <div className="mt-2 text-center">
                      <span className="text-2xl">👤</span>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">더블 탑/바텀</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      두 번의 고점/저점을 형성하며 추세가 전환
                    </p>
                    <div className="mt-2 text-center">
                      <span className="text-2xl">M / W</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🔄 지속 패턴</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">삼각형 패턴</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      가격 변동폭이 줄어들다가 한 방향으로 돌파
                    </p>
                    <div className="mt-2 text-center">
                      <span className="text-2xl">◀▶</span>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">깃발형 패턴</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      급등/급락 후 잠시 쉬어가는 패턴
                    </p>
                    <div className="mt-2 text-center">
                      <span className="text-2xl">🏳️</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💡 패턴 활용 팁</h3>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500">✓</span>
                    <span>패턴 완성 전에는 섣부른 판단 금물</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500">✓</span>
                    <span>거래량으로 패턴의 신뢰도 확인</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500">✓</span>
                    <span>다른 지표와 함께 종합적으로 판단</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-indigo-500">✓</span>
                    <span>손절선은 패턴 무효화 지점에 설정</span>
                  </li>
                </ul>
              </div>
            </div>
          ),
          simulator: <BollingerBandsSimulator />
        },
        {
          title: '매매 전략 수립',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">나만의 매매 전략 만들기 🎯</h2>
              
              <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📋 매매 원칙 세우기</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">1. 진입 조건</p>
                    <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                      <li>• RSI 30 이하 + MACD 골든크로스</li>
                      <li>• 20일 이동평균선 지지 확인</li>
                      <li>• 거래량 평균 대비 1.5배 이상</li>
                    </ul>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">2. 손절 기준</p>
                    <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                      <li>• 매수가 대비 -5% 도달 시</li>
                      <li>• 주요 지지선 하향 돌파 시</li>
                      <li>• 매매 논리가 깨졌을 때</li>
                    </ul>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">3. 익절 기준</p>
                    <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                      <li>• 목표 수익률 15% 도달</li>
                      <li>• RSI 70 이상 과매수 구간</li>
                      <li>• 주요 저항선 도달</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">⚖️ 자금 관리</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">분할 매수</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      한 번에 전량 매수하지 말고 3~4회 나누어 매수
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">포지션 사이징</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      한 종목당 전체 자금의 20% 이내로 제한
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📝 매매일지 작성</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  모든 매매를 기록하고 분석하여 전략을 개선하세요
                </p>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="font-medium mb-2">기록할 내용:</p>
                  <div className="grid md:grid-cols-2 gap-3 text-sm">
                    <ul className="space-y-1">
                      <li>• 매수/매도 일시</li>
                      <li>• 종목명과 수량</li>
                      <li>• 매매 가격</li>
                    </ul>
                    <ul className="space-y-1">
                      <li>• 매매 이유</li>
                      <li>• 손익 결과</li>
                      <li>• 개선점</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: (
            <div className="space-y-6">
              {/* 간단한 RSI 시뮬레이터 (임베드) */}
              <RSISimulator />
              
              {/* 전문 기술적분석 도구 링크 */}
              <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-6 h-6 text-orange-600" />
                  전문가급 기술적분석 도구
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <Link
                    href="/modules/stock-analysis/simulators/chart-analyzer"
                    className="group bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-orange-300 dark:hover:border-orange-600 transition-all duration-200"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <BarChart3 className="w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors">
                          AI 차트 패턴 분석기
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          50가지 패턴 자동 인식
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      • AI 패턴 매칭<br/>
                      • 매매 신호 생성<br/>
                      • 백테스팅 결과
                    </p>
                  </Link>

                  <Link
                    href="/modules/stock-analysis/simulators/factor-investing-lab"
                    className="group bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-red-300 dark:hover:border-red-600 transition-all duration-200"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-12 h-12 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <TrendingUp className="w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 dark:text-white group-hover:text-red-600 dark:group-hover:text-red-400 transition-colors">
                          팩터 투자 연구소
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          8가지 투자 팩터 분석
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      • 가치/모멘텀 팩터<br/>
                      • 멀티팩터 백테스팅<br/>
                      • 퍼포먼스 분석
                    </p>
                  </Link>

                  <Link
                    href="/modules/stock-analysis/simulators/sector-rotation-tracker"
                    className="group bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-purple-300 dark:hover:border-purple-600 transition-all duration-200"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <Target className="w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                          섹터 로테이션 추적기
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          경기 사이클 분석
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      • 섹터별 성과 분석<br/>
                      • 로테이션 타이밍<br/>
                      • 경기 지표 연동
                    </p>
                  </Link>

                  <Link
                    href="/modules/stock-analysis/simulators/options-strategy-analyzer"
                    className="group bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-indigo-300 dark:hover:border-indigo-600 transition-all duration-200"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <Calculator className="w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                          옵션 전략 분석기
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          50가지 옵션 전략
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      • 손익구조 시각화<br/>
                      • 그리스값 분석<br/>
                      • 시간가치 추적
                    </p>
                  </Link>
                </div>
              </div>
            </div>
          )
        }
      ]
    },
    professional: {
      title: '전문 투자자 과정',
      sections: [
        {
          title: '재무분석 심화',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">고급 재무분석 기법 💼</h2>
              
              <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💰 DCF (현금흐름할인법)</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  미래의 현금흐름을 현재가치로 환산하여 기업가치를 평가하는 방법
                </p>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="font-medium mb-2">핵심 요소:</p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-500">1.</span>
                      <div>
                        <strong>잉여현금흐름 (FCF)</strong>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          영업현금흐름 - 자본적 지출
                        </p>
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-500">2.</span>
                      <div>
                        <strong>할인율 (WACC)</strong>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          가중평균자본비용
                        </p>
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-emerald-500">3.</span>
                      <div>
                        <strong>영구성장률</strong>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          장기적인 성장률 (보통 2-3%)
                        </p>
                      </div>
                    </li>
                  </ul>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 재무비율 심화분석</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">수익성 지표</p>
                    <div className="grid md:grid-cols-2 gap-2 text-sm">
                      <div>• ROE = 순이익/자기자본</div>
                      <div>• ROA = 순이익/총자산</div>
                      <div>• ROIC = NOPAT/투하자본</div>
                      <div>• 영업이익률 = 영업이익/매출</div>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">안정성 지표</p>
                    <div className="grid md:grid-cols-2 gap-2 text-sm">
                      <div>• 부채비율 = 부채/자기자본</div>
                      <div>• 유동비율 = 유동자산/유동부채</div>
                      <div>• 이자보상배율 = EBIT/이자비용</div>
                      <div>• 순부채비율 = 순부채/자기자본</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🔍 질적 분석</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">🏢</span>
                    <div>
                      <p className="font-medium">경영진 평가</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        경영진의 실적, 비전, 주주친화 정책
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">🏭</span>
                    <div>
                      <p className="font-medium">산업 분석</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        산업 성장성, 진입장벽, 경쟁 강도
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">🛡️</span>
                    <div>
                      <p className="font-medium">경제적 해자</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        브랜드, 특허, 네트워크 효과 등
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <DCFCalculator />
        },
        {
          title: '포트폴리오 최적화',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">전문가의 포트폴리오 관리 🎯</h2>
              
              <div className="bg-sky-50 dark:bg-sky-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📐 현대 포트폴리오 이론</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  리스크 대비 수익을 최적화하는 자산배분 전략
                </p>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="font-medium mb-2">효율적 프론티어</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    동일한 리스크에서 최대 수익을 내는 포트폴리오의 집합
                  </p>
                  <div className="grid md:grid-cols-2 gap-3 text-sm">
                    <div className="bg-sky-100 dark:bg-sky-900/30 rounded p-2">
                      <p className="font-medium">상관관계 활용</p>
                      <p className="text-xs">음의 상관관계 자산으로 리스크 감소</p>
                    </div>
                    <div className="bg-sky-100 dark:bg-sky-900/30 rounded p-2">
                      <p className="font-medium">분산효과</p>
                      <p className="text-xs">개별 리스크의 합 {'>'} 포트폴리오 리스크</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">⚖️ 자산배분 전략</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">전략적 자산배분</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      장기 목표에 맞춘 고정 비중 유지
                    </p>
                    <div className="mt-2 text-sm">
                      예: 주식 60% / 채권 30% / 대체투자 10%
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">전술적 자산배분</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      시장 상황에 따라 비중을 조절
                    </p>
                    <div className="mt-2 text-sm">
                      예: 경기 호황 → 주식 비중 확대
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">다이나믹 자산배분</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      정량 모델에 따라 자동 조절
                    </p>
                    <div className="mt-2 text-sm">
                      예: 변동성 상승 → 안전자산 비중 자동 증가
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 성과 측정</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">샤프 비율</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      (수익률 - 무위험수익률) / 표준편차
                    </p>
                    <p className="text-xs mt-1">위험 단위당 초과수익</p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">정보 비율</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      초과수익률 / 추적오차
                    </p>
                    <p className="text-xs mt-1">액티브 운용 능력 평가</p>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <RiskManagementSimulator />
        },
        {
          title: '리스크 관리 전략',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">체계적인 리스크 관리 시스템 🛡️</h2>
              
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">⚠️ 리스크의 종류</h3>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium text-red-600 dark:text-red-400 mb-2">시장 리스크</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      전체 시장의 움직임에 따른 손실 가능성
                    </p>
                    <p className="text-xs mt-1">대응: 분산투자, 헤지 전략</p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium text-orange-600 dark:text-orange-400 mb-2">신용 리스크</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      기업 부도나 신용등급 하락 위험
                    </p>
                    <p className="text-xs mt-1">대응: 신용분석, 등급 모니터링</p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium text-yellow-600 dark:text-yellow-400 mb-2">유동성 리스크</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      필요시 자산을 현금화하기 어려운 위험
                    </p>
                    <p className="text-xs mt-1">대응: 유동성 버퍼, 거래량 확인</p>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 리스크 측정 지표</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">VaR (Value at Risk)</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      특정 신뢰수준에서의 최대 예상 손실
                    </p>
                    <p className="text-xs mt-2">
                      예: 95% VaR = 1,000만원<br/>
                      → 95% 확률로 손실이 1,000만원 이하
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">최대 낙폭 (MDD)</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      최고점 대비 최대 하락률
                    </p>
                    <p className="text-xs mt-2">
                      과거 MDD 분석으로<br/>
                      미래 리스크 예측
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🛡️ 리스크 관리 도구</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">📉</span>
                    <div>
                      <p className="font-medium">손절매 (Stop Loss)</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        일정 손실 도달시 자동 청산
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">🔄</span>
                    <div>
                      <p className="font-medium">헤징 (Hedging)</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        선물, 옵션으로 반대 포지션 구축
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">📏</span>
                    <div>
                      <p className="font-medium">포지션 사이징</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        켈리 공식 등으로 최적 투자 규모 결정
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <RiskManagementSimulator />
        },
        {
          title: 'AI & 퀀트 투자',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">AI와 퀀트 기법의 융합 🤖</h2>
              
              <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🧠 AI 기반 투자 시스템</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  머신러닝과 딥러닝 기술을 활용한 현대적 투자 전략
                </p>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">🔮 예측 모델</p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• LSTM, GRU를 활용한 시계열 예측</li>
                      <li>• 앙상블 기법으로 예측 정확도 향상</li>
                      <li>• 감정 분석을 통한 시장 심리 파악</li>
                    </ul>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">⚡ 고빈도 거래</p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• 마이크로초 단위 시장 기회 포착</li>
                      <li>• 알고리즘 트레이딩 전략 최적화</li>
                      <li>• 레이턴시 최소화 기술</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 퀀트 전략</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  수학적 모델과 통계 기법을 활용한 체계적 투자
                </p>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">📈 팩터 투자</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      밸류, 모멘텀, 퀄리티, 로우볼 등 검증된 팩터를 활용한 투자
                    </p>
                    <div className="mt-2 text-xs">
                      <span className="bg-blue-100 dark:bg-blue-900/30 px-2 py-1 rounded">Size Factor</span>
                      <span className="bg-green-100 dark:bg-green-900/30 px-2 py-1 rounded ml-1">Value Factor</span>
                      <span className="bg-purple-100 dark:bg-purple-900/30 px-2 py-1 rounded ml-1">Momentum Factor</span>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">🔄 페어 트레이딩</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      상관관계가 높은 종목들 간의 스프레드를 활용한 마켓뉴트럴 전략
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <p className="font-medium mb-2">📉 평균회귀</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      가격이 평균으로 돌아가는 성질을 이용한 단기 차익거래
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🌐 거시경제 모델링</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  거시경제 지표와 시장 움직임의 관계를 모델링
                </p>
                <div className="grid md:grid-cols-3 gap-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3 text-center">
                    <p className="font-medium text-sm">📈 GDP 영향도</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">경제성장과 주식시장</p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3 text-center">
                    <p className="font-medium text-sm">💰 금리 민감도</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">섹터별 금리 영향</p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3 text-center">
                    <p className="font-medium text-sm">🏭 인플레이션</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">물가와 자산 배분</p>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: (
            <div className="space-y-6">
              {/* AI/퀀트 전문 도구 링크 */}
              <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <Brain className="w-6 h-6 text-purple-600" />
                  AI 투자 전문 도구
                </h3>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <Link
                    href="/modules/stock-analysis/simulators/ai-mentor"
                    className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:border-purple-300 dark:hover:border-purple-600 transition-all duration-200 group"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <Brain className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                          AI 투자 멘토
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          AI가 제공하는 전문가급 투자 조언
                        </p>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      💡 머신러닝 기반 포트폴리오 추천 및 리스크 분석
                    </p>
                  </Link>

                  <Link
                    href="/modules/stock-analysis/simulators/risk-management-dashboard"
                    className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:border-indigo-300 dark:hover:border-indigo-600 transition-all duration-200 group"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <AlertTriangle className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                          리스크 관리 대시보드
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          실시간 리스크 모니터링 시스템
                        </p>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      📊 VaR, MDD, 스트레스 테스트 등 종합 리스크 분석
                    </p>
                  </Link>
                </div>
              </div>

              {/* 거시경제 & 알고리즘 도구 */}
              <div className="bg-gradient-to-r from-emerald-50 to-cyan-50 dark:from-emerald-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <TrendingUp className="w-6 h-6 text-emerald-600" />
                  거시경제 & 알고리즘 도구
                </h3>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <Link
                    href="/modules/stock-analysis/simulators/macro-economic-dashboard"
                    className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:border-emerald-300 dark:hover:border-emerald-600 transition-all duration-200 group"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <Activity className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                          거시경제 대시보드
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          글로벌 경제 지표 통합 분석
                        </p>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      🌍 GDP, 금리, 인플레이션 등 주요 지표와 시장 영향 분석
                    </p>
                  </Link>

                  <Link
                    href="/modules/stock-analysis/simulators/algo-trading-system"
                    className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:border-cyan-300 dark:hover:border-cyan-600 transition-all duration-200 group"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-cyan-100 dark:bg-cyan-900/30 text-cyan-600 dark:text-cyan-400 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                        <DollarSign className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors">
                          알고리즘 트레이딩 시스템
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          자동화된 투자 전략 실행
                        </p>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      ⚡고빈도 거래 전략 백테스팅 및 실시간 실행
                    </p>
                  </Link>
                </div>
              </div>
            </div>
          )
        }
      ]
    },
    candlesticks: {
      title: '캔들스틱 차트',
      sections: [
        {
          title: '캔들스틱 차트의 이해',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">캔들스틱 차트의 이해 🕯️</h2>
              
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 캔들스틱이 뭐예요?</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  캔들스틱은 주식의 하루 움직임을 하나의 양초 모양으로 표현한 것입니다.
                  빨간색은 가격이 올랐고, 파란색은 가격이 떨어졌다는 의미예요.
                </p>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-6 h-16 bg-red-500 rounded"></div>
                      <div>
                        <h4 className="font-semibold text-red-600 dark:text-red-400">빨간 캔들 (양봉)</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          종가가 시가보다 높아요
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-6 h-16 bg-blue-500 rounded"></div>
                      <div>
                        <h4 className="font-semibold text-blue-600 dark:text-blue-400">파란 캔들 (음봉)</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          종가가 시가보다 낮아요
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🎯 캔들의 구성 요소</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">1️⃣</span>
                    <div>
                      <strong>몸통 (Body)</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        시가와 종가 사이의 부분. 두꺼울수록 가격 변동이 컸어요
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">2️⃣</span>
                    <div>
                      <strong>위꼬리 (Upper Shadow)</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        그날의 최고가를 보여줘요
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">3️⃣</span>
                    <div>
                      <strong>아래꼬리 (Lower Shadow)</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        그날의 최저가를 보여줘요
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📈 캔들 패턴 읽기</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-3">
                  캔들의 모양으로 시장의 심리를 읽을 수 있어요:
                </p>
                <ul className="space-y-2">
                  <li>• <strong>긴 빨간 캔들</strong>: 강한 매수세, 가격이 크게 올랐어요</li>
                  <li>• <strong>긴 파란 캔들</strong>: 강한 매도세, 가격이 크게 떨어졌어요</li>
                  <li>• <strong>도지 (십자가)</strong>: 매수세와 매도세가 팽팽해요</li>
                  <li>• <strong>망치형</strong>: 하락 후 반등 신호일 수 있어요</li>
                </ul>
              </div>
            </div>
          ),
          simulator: <CandleChartSimulator />
        },
        {
          title: '이동평균선과 거래량',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">이동평균선과 거래량 📈</h2>
              
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 이동평균선이란?</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  이동평균선은 일정 기간 동안의 주가 평균을 연결한 선입니다.
                  주가의 전반적인 추세를 파악하는 데 도움이 됩니다.
                </p>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">5일선</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      단기 추세를 보여줍니다. 민감하게 반응해요.
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">20일선</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      중기 추세를 보여줍니다. 가장 많이 사용해요.
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">60일선</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      장기 추세를 보여줍니다. 큰 흐름을 봐요.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🎯 골든크로스와 데드크로스</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">
                      🌟 골든크로스
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      단기 이평선이 장기 이평선을 위로 돌파할 때.
                      상승 신호로 해석됩니다.
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-600 dark:text-gray-400 mb-2">
                      ☠️ 데드크로스
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      단기 이평선이 장기 이평선을 아래로 돌파할 때.
                      하락 신호로 해석됩니다.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 거래량의 중요성</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-3">
                  거래량은 주가의 신뢰도를 나타냅니다:
                </p>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2">
                    <span>📈</span>
                    <div>
                      <strong>가격 상승 + 거래량 증가</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        건강한 상승, 추세가 계속될 가능성 높음
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <span>📉</span>
                    <div>
                      <strong>가격 상승 + 거래량 감소</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        상승 동력 약화, 조정 가능성 있음
                      </p>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          ),
          simulator: <MovingAverageSimulator />
        },
        {
          title: '지지선과 저항선',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">지지선과 저항선 이해하기 🏗️</h2>
              
              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🛡️ 지지선과 저항선이란?</h3>
                <div className="grid md:grid-cols-2 gap-4 mb-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                      지지선 (Support)
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      주가가 떨어지다가 멈추는 가격대.
                      "바닥"이라고 생각하면 쉬워요.
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">
                      저항선 (Resistance)
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      주가가 오르다가 멈추는 가격대.
                      "천장"이라고 생각하면 쉬워요.
                    </p>
                  </div>
                </div>
                <p className="text-gray-700 dark:text-gray-300">
                  💡 <strong>핵심 원리</strong>: 많은 사람들이 특정 가격대에서 매수/매도하려고 해서 생기는 현상입니다.
                </p>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📐 지지선과 저항선 찾는 법</h3>
                <ol className="space-y-3">
                  <li className="flex items-start gap-3">
                    <span className="text-lg font-bold text-blue-600">1</span>
                    <div>
                      <strong>과거 고점과 저점 찾기</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        차트에서 주가가 여러 번 반등했던 가격대를 찾아보세요
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-lg font-bold text-blue-600">2</span>
                    <div>
                      <strong>라운드 넘버 주목</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        50,000원, 100,000원 같은 깔끔한 숫자는 심리적 지지/저항선
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-lg font-bold text-blue-600">3</span>
                    <div>
                      <strong>거래량 확인</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        거래량이 많았던 가격대는 강한 지지/저항선이 됩니다
                      </p>
                    </div>
                  </li>
                </ol>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🔄 역할 전환</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-3">
                  놀라운 사실! 지지선과 저항선은 서로 역할을 바꿉니다:
                </p>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="font-medium mb-2">저항선 → 지지선</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    주가가 저항선을 뚫고 올라가면, 그 저항선은 이제 지지선이 됩니다
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mt-3">
                  <p className="font-medium mb-2">지지선 → 저항선</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    주가가 지지선을 뚫고 내려가면, 그 지지선은 이제 저항선이 됩니다
                  </p>
                </div>
              </div>
            </div>
          ),
          simulator: <SimpleChartSimulator />
        }
      ]
    },
    analysis: {
      title: '기업 분석',
      sections: [
        {
          title: '기업 분석의 기초',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">기업 분석의 기초 🏢</h2>
              
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 재무제표 3종 세트</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  기업을 분석하려면 3가지 재무제표를 봐야 해요:
                </p>
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                      💰 손익계산서 (Income Statement)
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      회사가 얼마나 벌고 썼는지 보여주는 성적표
                    </p>
                    <ul className="text-sm space-y-1">
                      <li>• <strong>매출</strong>: 제품/서비스를 팔아서 번 돈</li>
                      <li>• <strong>영업이익</strong>: 매출에서 비용을 뺀 실제 이익</li>
                      <li>• <strong>순이익</strong>: 세금까지 낸 후 최종 이익</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                      📋 재무상태표 (Balance Sheet)
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      회사가 가진 것과 빚을 보여주는 재산 목록
                    </p>
                    <ul className="text-sm space-y-1">
                      <li>• <strong>자산</strong>: 회사가 가진 모든 것 (현금, 건물, 장비)</li>
                      <li>• <strong>부채</strong>: 회사가 갚아야 할 빚</li>
                      <li>• <strong>자본</strong>: 자산에서 부채를 뺀 순재산</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">
                      💸 현금흐름표 (Cash Flow)
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      실제 현금이 어떻게 들어오고 나가는지 보여주는 가계부
                    </p>
                    <ul className="text-sm space-y-1">
                      <li>• <strong>영업활동</strong>: 장사해서 번 현금</li>
                      <li>• <strong>투자활동</strong>: 미래를 위해 쓴 현금</li>
                      <li>• <strong>재무활동</strong>: 돈 빌리고 갚은 내역</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🎯 좋은 기업 고르는 체크리스트</h3>
                <div className="space-y-3">
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="w-5 h-5" />
                    <span>매출이 꾸준히 성장하고 있나요? (연 10% 이상)</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="w-5 h-5" />
                    <span>영업이익률이 10% 이상인가요?</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="w-5 h-5" />
                    <span>부채비율이 100% 이하인가요?</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="w-5 h-5" />
                    <span>현금흐름이 (+) 인가요?</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input type="checkbox" className="w-5 h-5" />
                    <span>ROE가 15% 이상인가요?</span>
                  </label>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💡 비즈니스 모델 이해하기</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-3">
                  숫자만큼 중요한 것이 회사가 어떻게 돈을 버는지 이해하는 거예요:
                </p>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                    <p className="font-medium text-sm mb-1">🍔 맥도날드</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      햄버거 판매 + 부동산 임대업
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                    <p className="font-medium text-sm mb-1">📱 애플</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      하드웨어 + 서비스 구독료
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                    <p className="font-medium text-sm mb-1">🛒 아마존</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      이커머스 + 클라우드(AWS)
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                    <p className="font-medium text-sm mb-1">🎮 넷플릭스</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      구독료 + 오리지널 콘텐츠
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <ValuationCalculator />
        },
        {
          title: 'PER과 PBR로 가치평가',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">PER과 PBR로 가치평가하기 💎</h2>
              
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 PER (주가수익비율)</h3>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                  <p className="text-lg font-mono text-center mb-2">
                    PER = 주가 ÷ 주당순이익(EPS)
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                    "이 회사가 1년에 버는 돈의 몇 배를 주고 살 건가요?"
                  </p>
                </div>
                
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-3">
                    <p className="font-semibold text-green-700 dark:text-green-400">PER 10 이하</p>
                    <p className="text-sm">저평가 가능성</p>
                  </div>
                  <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-3">
                    <p className="font-semibold text-yellow-700 dark:text-yellow-400">PER 10-20</p>
                    <p className="text-sm">적정 수준</p>
                  </div>
                  <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-3">
                    <p className="font-semibold text-red-700 dark:text-red-400">PER 20 이상</p>
                    <p className="text-sm">고평가 가능성</p>
                  </div>
                </div>
                
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
                  💡 <strong>주의</strong>: 업종별로 평균 PER이 다릅니다. IT는 높고, 금융은 낮은 편이에요.
                </p>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📈 PBR (주가순자산비율)</h3>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                  <p className="text-lg font-mono text-center mb-2">
                    PBR = 주가 ÷ 주당순자산(BPS)
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                    "이 회사의 장부상 가치의 몇 배를 주고 살 건가요?"
                  </p>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="w-16 h-2 bg-green-500 rounded"></div>
                    <span><strong>PBR {'<'} 1</strong>: 장부가치보다 싸게 거래 (청산가치 이하)</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-16 h-2 bg-yellow-500 rounded"></div>
                    <span><strong>PBR = 1</strong>: 장부가치와 동일한 가격</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-16 h-2 bg-red-500 rounded"></div>
                    <span><strong>PBR {'>'} 1</strong>: 장부가치보다 비싸게 거래 (성장성 반영)</span>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🎯 실전 활용법</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">좋은 조합 ✅</h4>
                    <ul className="text-sm space-y-1">
                      <li>• 낮은 PER + 낮은 PBR = 저평가 우량주</li>
                      <li>• 높은 PER + 높은 성장률 = 성장주</li>
                      <li>• 낮은 PBR + 안정적 수익 = 가치주</li>
                    </ul>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">주의할 조합 ⚠️</h4>
                    <ul className="text-sm space-y-1">
                      <li>• 높은 PER + 낮은 성장률 = 고평가</li>
                      <li>• 낮은 PBR + 적자 = 부실 위험</li>
                      <li>• 급격한 PER 상승 = 버블 가능성</li>
                    </ul>
                  </div>
                </div>
                
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
                  💡 <strong>꿀팁</strong>: PER과 PBR은 같은 업종 내에서 비교하세요!
                </p>
              </div>
            </div>
          ),
          simulator: <ValuationCalculator />
        },
        {
          title: '포트폴리오 구성 전략',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">포트폴리오 구성 전략 🎨</h2>
              
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🥚 계란을 한 바구니에 담지 마세요!</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  분산투자는 위험을 줄이는 가장 기본적인 방법입니다.
                  여러 종목, 여러 산업에 나눠서 투자하세요.
                </p>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                    <div className="text-3xl mb-2">🏭</div>
                    <h4 className="font-semibold">산업 분산</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      IT, 금융, 제조업 등
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                    <div className="text-3xl mb-2">🌍</div>
                    <h4 className="font-semibold">지역 분산</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      국내, 미국, 신흥국
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                    <div className="text-3xl mb-2">💰</div>
                    <h4 className="font-semibold">자산 분산</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      주식, 채권, 원자재
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 나이별 추천 포트폴리오</h3>
                <div className="space-y-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-semibold">20-30대</h4>
                      <span className="text-sm text-gray-500">공격적</span>
                    </div>
                    <div className="flex gap-2 items-center">
                      <div className="h-4 bg-red-500 rounded" style={{width: '80%'}}></div>
                      <div className="h-4 bg-blue-500 rounded" style={{width: '20%'}}></div>
                    </div>
                    <p className="text-sm mt-2">주식 80% : 채권 20%</p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-semibold">40-50대</h4>
                      <span className="text-sm text-gray-500">균형형</span>
                    </div>
                    <div className="flex gap-2 items-center">
                      <div className="h-4 bg-red-500 rounded" style={{width: '60%'}}></div>
                      <div className="h-4 bg-blue-500 rounded" style={{width: '40%'}}></div>
                    </div>
                    <p className="text-sm mt-2">주식 60% : 채권 40%</p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-semibold">60대 이상</h4>
                      <span className="text-sm text-gray-500">안정형</span>
                    </div>
                    <div className="flex gap-2 items-center">
                      <div className="h-4 bg-red-500 rounded" style={{width: '40%'}}></div>
                      <div className="h-4 bg-blue-500 rounded" style={{width: '60%'}}></div>
                    </div>
                    <p className="text-sm mt-2">주식 40% : 채권 60%</p>
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🔄 리밸런싱 전략</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-3">
                  정기적으로 포트폴리오를 재조정하세요:
                </p>
                <ol className="space-y-3">
                  <li className="flex items-start gap-3">
                    <span className="bg-purple-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm">1</span>
                    <div>
                      <strong>목표 비중 설정</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        예: A주식 30%, B주식 30%, 채권 40%
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="bg-purple-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm">2</span>
                    <div>
                      <strong>정기 점검 (분기/반기)</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        현재 비중이 목표와 5% 이상 차이나는지 확인
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="bg-purple-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm">3</span>
                    <div>
                      <strong>비중 조정</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        오른 것은 팔고, 떨어진 것은 사서 원래 비중으로
                      </p>
                    </div>
                  </li>
                </ol>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mt-4">
                  <p className="text-sm">
                    💡 <strong>리밸런싱의 효과</strong>: 자연스럽게 "비싸게 팔고 싸게 사는" 효과!
                  </p>
                </div>
              </div>
            </div>
          ),
          simulator: <PortfolioSimulator />
        }
      ]
    },
    indicators: {
      title: '기술적 지표',
      sections: [
        {
          title: '보조지표 활용하기',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">보조지표 활용하기 📊</h2>
              
              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📈 RSI (상대강도지수)</h3>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                  <p className="text-gray-700 dark:text-gray-300 mb-3">
                    RSI는 주가의 상승/하락 압력을 0-100 사이 숫자로 표현합니다.
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <div className="w-20 h-6 bg-red-500 rounded text-white text-xs flex items-center justify-center">70 이상</div>
                      <span>과매수 구간 - 조정 가능성 ⚠️</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-20 h-6 bg-gray-500 rounded text-white text-xs flex items-center justify-center">30-70</div>
                      <span>중립 구간 - 추세 관찰 👀</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-20 h-6 bg-blue-500 rounded text-white text-xs flex items-center justify-center">30 이하</div>
                      <span>과매도 구간 - 반등 가능성 📈</span>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  💡 <strong>활용팁</strong>: RSI 다이버전스(주가와 RSI가 반대로 움직임)는 추세 전환 신호!
                </p>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📉 MACD (이동평균수렴확산)</h3>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                  <p className="text-gray-700 dark:text-gray-300 mb-3">
                    MACD는 단기 이평선과 장기 이평선의 차이를 이용한 지표입니다.
                  </p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="font-semibold text-sm">구성 요소</h4>
                      <ul className="text-sm space-y-1">
                        <li>• <strong>MACD선</strong>: 12일 EMA - 26일 EMA</li>
                        <li>• <strong>신호선</strong>: MACD의 9일 EMA</li>
                        <li>• <strong>히스토그램</strong>: MACD선 - 신호선</li>
                      </ul>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-semibold text-sm">매매 신호</h4>
                      <ul className="text-sm space-y-1">
                        <li>• <span className="text-green-600">골든크로스</span>: MACD가 신호선 상향돌파</li>
                        <li>• <span className="text-red-600">데드크로스</span>: MACD가 신호선 하향돌파</li>
                        <li>• <span className="text-blue-600">0선 돌파</span>: 추세 전환 신호</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 스토캐스틱 (Stochastic)</h3>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                  <p className="text-gray-700 dark:text-gray-300 mb-3">
                    현재 가격이 일정 기간 중 최고가/최저가 대비 어느 위치인지 표시합니다.
                  </p>
                  <div className="space-y-3">
                    <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-3">
                      <p className="font-semibold text-red-700 dark:text-red-400">80% 이상</p>
                      <p className="text-sm">과매수 - 단기 고점 가능성</p>
                    </div>
                    <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-3">
                      <p className="font-semibold text-blue-700 dark:text-blue-400">20% 이하</p>
                      <p className="text-sm">과매도 - 단기 저점 가능성</p>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  💡 <strong>주의사항</strong>: 강한 추세장에서는 과매수/과매도 상태가 오래 지속될 수 있어요!
                </p>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🎯 지표 조합 전략</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2 text-green-600">추세 추종 전략</h4>
                    <ul className="text-sm space-y-1">
                      <li>✓ MACD 골든크로스</li>
                      <li>✓ RSI 50 이상 유지</li>
                      <li>✓ 주가 {'>'} 20일 이평선</li>
                    </ul>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2 text-orange-600">역추세 전략</h4>
                    <ul className="text-sm space-y-1">
                      <li>✓ RSI 30 이하 진입</li>
                      <li>✓ 스토캐스틱 20 이하</li>
                      <li>✓ 지지선 근처</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <RSISimulator />
        },
        {
          title: '차트 패턴 분석',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">차트 패턴 분석 🔍</h2>
              
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🔄 반전 패턴</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  추세가 바뀔 가능성을 알려주는 패턴들입니다:
                </p>
                
                <div className="space-y-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                      🗻 헤드앤숄더 (Head & Shoulders)
                    </h4>
                    <div className="flex items-center gap-4">
                      <div className="flex items-end gap-1">
                        <div className="w-8 h-12 bg-gray-400 rounded"></div>
                        <div className="w-8 h-16 bg-blue-500 rounded"></div>
                        <div className="w-8 h-12 bg-gray-400 rounded"></div>
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          상승 추세 → 하락 전환 신호<br/>
                          왼쪽 어깨, 머리, 오른쪽 어깨 형태
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">
                      🎯 더블탑 / 더블바텀
                    </h4>
                    <div className="flex items-center gap-4">
                      <div className="flex items-end gap-1">
                        <div className="w-8 h-14 bg-red-500 rounded"></div>
                        <div className="w-8 h-8 bg-gray-400 rounded"></div>
                        <div className="w-8 h-14 bg-red-500 rounded"></div>
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          M자 모양(더블탑) = 하락 전환<br/>
                          W자 모양(더블바텀) = 상승 전환
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">➡️ 지속 패턴</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  잠시 쉬었다가 원래 추세로 계속 갈 가능성이 높은 패턴들:
                </p>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                      📐 삼각형 패턴
                    </h4>
                    <ul className="text-sm space-y-1">
                      <li>• <strong>상승 삼각형</strong>: 저점은 높아지고 고점은 일정</li>
                      <li>• <strong>하락 삼각형</strong>: 고점은 낮아지고 저점은 일정</li>
                      <li>• <strong>대칭 삼각형</strong>: 변동폭이 점점 좁아짐</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">
                      🚩 깃발형 패턴
                    </h4>
                    <ul className="text-sm space-y-1">
                      <li>• 급등/급락 후 잠시 횡보</li>
                      <li>• 깃대(급등) + 깃발(횡보)</li>
                      <li>• 돌파 시 기존 추세 지속</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📏 패턴 활용 시 체크리스트</h3>
                <div className="space-y-3">
                  <label className="flex items-start gap-3">
                    <input type="checkbox" className="w-5 h-5 mt-0.5" />
                    <div>
                      <strong>패턴 완성도</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        패턴이 교과서적인 모양에 가까운가?
                      </p>
                    </div>
                  </label>
                  
                  <label className="flex items-start gap-3">
                    <input type="checkbox" className="w-5 h-5 mt-0.5" />
                    <div>
                      <strong>거래량 확인</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        패턴 돌파 시 거래량이 증가하는가?
                      </p>
                    </div>
                  </label>
                  
                  <label className="flex items-start gap-3">
                    <input type="checkbox" className="w-5 h-5 mt-0.5" />
                    <div>
                      <strong>시간 프레임</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        일봉 이상에서 나타난 패턴인가?
                      </p>
                    </div>
                  </label>
                  
                  <label className="flex items-start gap-3">
                    <input type="checkbox" className="w-5 h-5 mt-0.5" />
                    <div>
                      <strong>추세와의 일치</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        전체 추세와 패턴의 방향이 일치하는가?
                      </p>
                    </div>
                  </label>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mt-4">
                  <p className="text-sm">
                    ⚠️ <strong>주의</strong>: 패턴만 보고 매매하지 마세요! 다른 지표와 함께 종합적으로 판단하세요.
                  </p>
                </div>
              </div>
            </div>
          ),
          simulator: <BollingerBandsSimulator />
        },
        {
          title: '매매 전략 수립',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">매매 전략 수립하기 🎯</h2>
              
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📋 나만의 매매 규칙 만들기</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  감정에 휘둘리지 않고 일관된 매매를 위해 명확한 규칙이 필요합니다:
                </p>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 space-y-3">
                  <div className="border-b border-gray-200 dark:border-gray-700 pb-3">
                    <h4 className="font-semibold text-red-600 dark:text-red-400">1. 진입 조건</h4>
                    <ul className="text-sm mt-2 space-y-1">
                      <li>✓ RSI 30 이하 + MACD 골든크로스</li>
                      <li>✓ 20일 이평선 지지 확인</li>
                      <li>✓ 거래량 평균 대비 1.5배 이상</li>
                    </ul>
                  </div>
                  
                  <div className="border-b border-gray-200 dark:border-gray-700 pb-3">
                    <h4 className="font-semibold text-blue-600 dark:text-blue-400">2. 손절 조건</h4>
                    <ul className="text-sm mt-2 space-y-1">
                      <li>✓ 매수가 대비 -7% 도달 시</li>
                      <li>✓ 주요 지지선 하향 돌파 시</li>
                      <li>✓ 악재 발생 시 즉시 손절</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-green-600 dark:text-green-400">3. 익절 조건</h4>
                    <ul className="text-sm mt-2 space-y-1">
                      <li>✓ 목표 수익률 +15% 도달 시</li>
                      <li>✓ RSI 70 이상 + 거래량 급증</li>
                      <li>✓ 주요 저항선 도달 시</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💰 자금 관리 원칙</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">포지션 사이징</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>한 종목 최대 투자</span>
                        <span className="font-mono">20%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>동시 보유 종목</span>
                        <span className="font-mono">5-7개</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>예비 현금 비중</span>
                        <span className="font-mono">30%</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">리스크 관리</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>1회 최대 손실</span>
                        <span className="font-mono text-red-600">-2%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>일일 최대 손실</span>
                        <span className="font-mono text-red-600">-5%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>월 최대 손실</span>
                        <span className="font-mono text-red-600">-10%</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-3 mt-4">
                  <p className="text-sm">
                    💡 <strong>2% 룰</strong>: 한 번의 거래에서 전체 자금의 2% 이상 잃지 않도록 포지션 크기를 조절하세요.
                  </p>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📓 매매일지 작성하기</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  성공적인 트레이더가 되려면 매매일지는 필수입니다:
                </p>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">매매일지 템플릿</h4>
                  <div className="space-y-3 text-sm">
                    <div className="grid grid-cols-2 gap-2">
                      <div>📅 <strong>날짜</strong>: 2024.01.15</div>
                      <div>🏷️ <strong>종목</strong>: 삼성전자</div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>💵 <strong>매수가</strong>: 75,000원</div>
                      <div>📊 <strong>수량</strong>: 10주</div>
                    </div>
                    <div>
                      <strong>📝 매수 이유</strong>:
                      <p className="mt-1 p-2 bg-gray-100 dark:bg-gray-700 rounded">
                        RSI 28 과매도 + 20일선 지지 + 실적 개선 기대
                      </p>
                    </div>
                    <div>
                      <strong>🎯 목표/손절</strong>:
                      <p className="mt-1 p-2 bg-gray-100 dark:bg-gray-700 rounded">
                        목표가: 86,000원 (+15%) / 손절가: 70,000원 (-7%)
                      </p>
                    </div>
                    <div>
                      <strong>💭 회고</strong>:
                      <p className="mt-1 p-2 bg-gray-100 dark:bg-gray-700 rounded">
                        인내심을 갖고 기다린 것이 좋았음. 다음엔 분할 매수 고려.
                      </p>
                    </div>
                  </div>
                </div>
                
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
                  💪 매매일지를 꾸준히 작성하면 자신만의 성공 패턴을 발견할 수 있습니다!
                </p>
              </div>
            </div>
          ),
          simulator: <RSISimulator />
        }
      ]
    },
    finance: {
      title: '재무분석',
      sections: [
        {
          title: '재무분석 심화',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">재무분석 심화 💼</h2>
              
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">💎 DCF 가치평가 모델</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  DCF(Discounted Cash Flow)는 기업의 미래 현금흐름을 현재가치로 환산하여 
                  기업의 내재가치를 계산하는 가장 정교한 평가 방법입니다.
                </p>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold mb-3">DCF 계산 5단계</h4>
                  <ol className="space-y-3">
                    <li className="flex items-start gap-3">
                      <span className="bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm flex-shrink-0">1</span>
                      <div>
                        <strong>미래 현금흐름 예측</strong>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          향후 5-10년간 FCF(잉여현금흐름) 추정
                        </p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm flex-shrink-0">2</span>
                      <div>
                        <strong>터미널 밸류 계산</strong>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          영구성장률 적용한 잔존가치 산출
                        </p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm flex-shrink-0">3</span>
                      <div>
                        <strong>할인율(WACC) 결정</strong>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          자본비용과 부채비용의 가중평균
                        </p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm flex-shrink-0">4</span>
                      <div>
                        <strong>현재가치로 할인</strong>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          미래 현금흐름을 현재가치로 환산
                        </p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm flex-shrink-0">5</span>
                      <div>
                        <strong>주당 가치 계산</strong>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          기업가치에서 순부채를 빼고 주식수로 나눔
                        </p>
                      </div>
                    </li>
                  </ol>
                </div>
                
                <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-3">
                  <p className="text-sm">
                    ⚠️ <strong>주의</strong>: DCF는 가정에 민감합니다. 보수적으로 추정하고 민감도 분석을 수행하세요.
                  </p>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 고급 재무비율 분석</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">수익성 지표</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span><strong>ROE</strong> (자기자본수익률)</span>
                        <span className="font-mono">15% 이상 우수</span>
                      </div>
                      <div className="flex justify-between">
                        <span><strong>ROIC</strong> (투하자본수익률)</span>
                        <span className="font-mono">10% 이상 양호</span>
                      </div>
                      <div className="flex justify-between">
                        <span><strong>GPM</strong> (매출총이익률)</span>
                        <span className="font-mono">업종별 상이</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-green-600 dark:text-green-400 mb-3">효율성 지표</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span><strong>자산회전율</strong></span>
                        <span className="font-mono">높을수록 좋음</span>
                      </div>
                      <div className="flex justify-between">
                        <span><strong>재고회전율</strong></span>
                        <span className="font-mono">업종 평균 비교</span>
                      </div>
                      <div className="flex justify-between">
                        <span><strong>CCC</strong> (현금전환주기)</span>
                        <span className="font-mono">짧을수록 우수</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mt-4">
                  <h4 className="font-semibold mb-2">듀폰 분석 (ROE 분해)</h4>
                  <p className="text-sm font-mono text-center">
                    ROE = 순이익률 × 자산회전율 × 재무레버리지
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 text-center mt-2">
                    수익성 × 효율성 × 레버리지 효과를 종합 분석
                  </p>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🔍 질적 분석 체크리스트</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  숫자만으로는 알 수 없는 기업의 질적 요소들:
                </p>
                
                <div className="space-y-3">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">경영진 평가</h4>
                    <ul className="text-sm space-y-1">
                      <li>✓ CEO의 비전과 실행력</li>
                      <li>✓ 주주친화적 경영 정책</li>
                      <li>✓ 투명한 의사소통</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">경쟁우위 (Moat)</h4>
                    <ul className="text-sm space-y-1">
                      <li>✓ 브랜드 파워</li>
                      <li>✓ 네트워크 효과</li>
                      <li>✓ 규모의 경제</li>
                      <li>✓ 전환비용</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">산업 분석</h4>
                    <ul className="text-sm space-y-1">
                      <li>✓ 산업 성장성</li>
                      <li>✓ 진입장벽</li>
                      <li>✓ 규제 리스크</li>
                      <li>✓ 기술 변화 속도</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <DCFCalculator />
        },
        {
          title: '포트폴리오 최적화',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">포트폴리오 최적화 이론 📐</h2>
              
              <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📈 현대 포트폴리오 이론 (MPT)</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  해리 마코위츠의 현대 포트폴리오 이론은 주어진 위험 수준에서 
                  기대수익을 최대화하는 최적 포트폴리오를 구성하는 방법론입니다.
                </p>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold mb-3">효율적 프론티어</h4>
                  <div className="h-48 bg-gray-100 dark:bg-gray-700 rounded flex items-center justify-center">
                    <p className="text-gray-500 dark:text-gray-400">
                      [위험-수익 그래프: 효율적 프론티어 곡선]
                    </p>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                    효율적 프론티어 위의 포트폴리오들이 최적 조합입니다.
                  </p>
                </div>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-indigo-100 dark:bg-indigo-800/30 rounded-lg p-3">
                    <h5 className="font-semibold mb-2">핵심 원리</h5>
                    <ul className="text-sm space-y-1">
                      <li>• 상관관계가 낮은 자산 조합</li>
                      <li>• 분산투자로 비체계적 위험 제거</li>
                      <li>• 위험 대비 수익 최적화</li>
                    </ul>
                  </div>
                  <div className="bg-purple-100 dark:bg-purple-800/30 rounded-lg p-3">
                    <h5 className="font-semibold mb-2">실전 적용</h5>
                    <ul className="text-sm space-y-1">
                      <li>• 자산군별 목표 비중 설정</li>
                      <li>• 정기적 리밸런싱</li>
                      <li>• 거래비용 고려</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🎯 자산배분 전략</h3>
                <div className="space-y-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-green-600 dark:text-green-400 mb-3">
                      전략적 자산배분 (Strategic Asset Allocation)
                    </h4>
                    <div className="grid md:grid-cols-3 gap-3 text-sm">
                      <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                        <p className="font-semibold">보수적</p>
                        <p>주식 30% : 채권 60% : 대체 10%</p>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                        <p className="font-semibold">중립적</p>
                        <p>주식 50% : 채권 40% : 대체 10%</p>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                        <p className="font-semibold">공격적</p>
                        <p>주식 70% : 채권 20% : 대체 10%</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">
                      전술적 자산배분 (Tactical Asset Allocation)
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      시장 상황에 따라 단기적으로 비중을 조정하는 전략:
                    </p>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center gap-3">
                        <span className="text-green-600">↑</span>
                        <span><strong>오버웨이트</strong>: 저평가된 자산 비중 확대</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-gray-600">→</span>
                        <span><strong>중립</strong>: 목표 비중 유지</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-red-600">↓</span>
                        <span><strong>언더웨이트</strong>: 고평가된 자산 비중 축소</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📊 성과 측정 지표</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">수익률 지표</h4>
                    <dl className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <dt><strong>CAGR</strong></dt>
                        <dd>연평균 성장률</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt><strong>TWR</strong></dt>
                        <dd>시간가중수익률</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt><strong>MWR</strong></dt>
                        <dd>금액가중수익률</dd>
                      </div>
                    </dl>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">위험조정 수익률</h4>
                    <dl className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <dt><strong>샤프지수</strong></dt>
                        <dd>(수익률-무위험) ÷ 변동성</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt><strong>소티노지수</strong></dt>
                        <dd>하방리스크만 고려</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt><strong>정보비율</strong></dt>
                        <dd>초과수익 ÷ 추적오차</dd>
                      </div>
                    </dl>
                  </div>
                </div>
                
                <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-3 mt-4">
                  <p className="text-sm">
                    💡 <strong>팁</strong>: 샤프지수 1.0 이상이면 우수한 성과, 2.0 이상이면 매우 우수합니다.
                  </p>
                </div>
              </div>
            </div>
          ),
          simulator: <PortfolioSimulator />
        },
        {
          title: '리스크 관리 전략',
          content: (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">리스크 관리 전략 🛡️</h2>
              
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">⚠️ 리스크의 종류와 이해</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-red-600 dark:text-red-400 mb-3">체계적 리스크</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      분산투자로 제거 불가능한 시장 전체 리스크
                    </p>
                    <ul className="text-sm space-y-1">
                      <li>• 금리 리스크</li>
                      <li>• 환율 리스크</li>
                      <li>• 인플레이션 리스크</li>
                      <li>• 정치적 리스크</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">비체계적 리스크</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      분산투자로 제거 가능한 개별 종목 리스크
                    </p>
                    <ul className="text-sm space-y-1">
                      <li>• 경영 리스크</li>
                      <li>• 신용 리스크</li>
                      <li>• 유동성 리스크</li>
                      <li>• 산업 리스크</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">📏 리스크 측정 방법</h3>
                <div className="space-y-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">VaR (Value at Risk)</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      특정 신뢰수준에서 발생 가능한 최대 손실액
                    </p>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <p className="text-sm font-mono">
                        예) 95% VaR = 1,000만원<br/>
                        → 95% 확률로 하루 손실이 1,000만원 이하
                      </p>
                    </div>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">최대낙폭 (MDD)</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      고점 대비 최대 하락률
                    </p>
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div className="bg-green-100 dark:bg-green-900/30 rounded p-2 text-center">
                        <p className="font-semibold">-10% 이하</p>
                        <p className="text-xs">안정적</p>
                      </div>
                      <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded p-2 text-center">
                        <p className="font-semibold">-20% 내외</p>
                        <p className="text-xs">보통</p>
                      </div>
                      <div className="bg-red-100 dark:bg-red-900/30 rounded p-2 text-center">
                        <p className="font-semibold">-30% 이상</p>
                        <p className="text-xs">위험</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">베타 (Beta)</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      시장 대비 개별 종목의 변동성
                    </p>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between items-center">
                        <span>β {'<'} 1.0</span>
                        <span className="text-blue-600">시장보다 안정적</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>β = 1.0</span>
                        <span className="text-gray-600">시장과 동일</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>β {'>'} 1.0</span>
                        <span className="text-red-600">시장보다 변동성 큼</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
                <h3 className="font-semibold mb-3">🛡️ 리스크 관리 도구</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">헤지 전략</h4>
                    <ul className="text-sm space-y-2">
                      <li className="flex items-start gap-2">
                        <span>📌</span>
                        <div>
                          <strong>풋옵션 매수</strong>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            하락 시 손실 제한
                          </p>
                        </div>
                      </li>
                      <li className="flex items-start gap-2">
                        <span>📌</span>
                        <div>
                          <strong>인버스 ETF</strong>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            시장 하락 시 수익
                          </p>
                        </div>
                      </li>
                      <li className="flex items-start gap-2">
                        <span>📌</span>
                        <div>
                          <strong>통화 헤지</strong>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            환율 변동 리스크 제거
                          </p>
                        </div>
                      </li>
                    </ul>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">포지션 관리</h4>
                    <ul className="text-sm space-y-2">
                      <li className="flex items-start gap-2">
                        <span>🎯</span>
                        <div>
                          <strong>스톱로스 설정</strong>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            자동 손절매 주문
                          </p>
                        </div>
                      </li>
                      <li className="flex items-start gap-2">
                        <span>🎯</span>
                        <div>
                          <strong>분할 매수/매도</strong>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            평균 단가 관리
                          </p>
                        </div>
                      </li>
                      <li className="flex items-start gap-2">
                        <span>🎯</span>
                        <div>
                          <strong>켈리 공식</strong>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            최적 베팅 사이즈 계산
                          </p>
                        </div>
                      </li>
                    </ul>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mt-4">
                  <h4 className="font-semibold mb-2">리스크 관리 체크리스트</h4>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="w-4 h-4" />
                      <span>최대 손실 한도 설정</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="w-4 h-4" />
                      <span>분산투자 실행</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="w-4 h-4" />
                      <span>정기적 리밸런싱</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="w-4 h-4" />
                      <span>스트레스 테스트</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          ),
          simulator: <RiskManagementSimulator />
        }
      ]
    }
  };

  const currentTrack = tracks[trackId as keyof typeof tracks] || tracks.beginner;
  const progress = ((completedSections.size) / currentTrack.sections.length) * 100;

  const handleSectionComplete = () => {
    setCompletedSections(prev => new Set([...prev, currentSection]));
    if (currentSection < currentTrack.sections.length - 1) {
      setCurrentSection(currentSection + 1);
      contentRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  useEffect(() => {
    // 진도 저장
    localStorage.setItem(`track-${trackId}-progress`, JSON.stringify({
      currentSection,
      completedSections: Array.from(completedSections)
    }));
  }, [trackId, currentSection, completedSections]);

  useEffect(() => {
    // 진도 불러오기
    const saved = localStorage.getItem(`track-${trackId}-progress`);
    if (saved) {
      const { currentSection: savedSection, completedSections: savedCompleted } = JSON.parse(saved);
      setCurrentSection(savedSection);
      setCompletedSections(new Set(savedCompleted));
    }
  }, [trackId]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <Link 
              href="/modules/stock-analysis"
              className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>학습 선택으로 돌아가기</span>
            </Link>
            
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm text-gray-600 dark:text-gray-400">전체 진행률</p>
                <p className="text-lg font-bold text-gray-900 dark:text-white">{Math.round(progress)}%</p>
              </div>
              <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-red-500 to-orange-500 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar - 목차 */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 sticky top-24">
              <h3 className="font-bold text-lg mb-4">{currentTrack.title}</h3>
              <div className="space-y-2">
                {currentTrack.sections.map((section, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentSection(index)}
                    className={`w-full text-left p-3 rounded-lg transition-all flex items-center gap-3 ${
                      index === currentSection
                        ? 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 font-medium'
                        : completedSections.has(index)
                        ? 'hover:bg-gray-50 dark:hover:bg-gray-700'
                        : 'opacity-60 hover:opacity-100'
                    }`}
                  >
                    {completedSections.has(index) ? (
                      <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                    ) : index === currentSection ? (
                      <Circle className="w-5 h-5 text-red-500 flex-shrink-0" />
                    ) : (
                      <Circle className="w-5 h-5 text-gray-400 flex-shrink-0" />
                    )}
                    <span className="text-sm">{section.title}</span>
                  </button>
                ))}
              </div>

              {progress === 100 && (
                <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
                  <Trophy className="w-8 h-8 text-green-600 mx-auto mb-2" />
                  <p className="text-sm font-medium text-green-700 dark:text-green-300">
                    축하합니다! 🎉<br />
                    모든 학습을 완료했어요!
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3" ref={contentRef}>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-8 mb-6">
              {currentTrack.sections[currentSection].content}
            </div>

            {/* Simulator */}
            <div className="mb-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <PlayCircle className="w-6 h-6 text-red-500" />
                실습해보기
              </h3>
              {currentTrack.sections[currentSection].simulator}
            </div>

            {/* Navigation */}
            <div className="flex items-center justify-between">
              <button
                onClick={() => setCurrentSection(Math.max(0, currentSection - 1))}
                disabled={currentSection === 0}
                className="flex items-center gap-2 px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ArrowLeft className="w-5 h-5" />
                이전
              </button>

              {currentSection < currentTrack.sections.length - 1 ? (
                <button
                  onClick={handleSectionComplete}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-500 to-orange-500 text-white rounded-lg font-medium hover:shadow-lg transition-all"
                >
                  다음으로
                  <ChevronRight className="w-5 h-5" />
                </button>
              ) : !completedSections.has(currentSection) ? (
                <button
                  onClick={handleSectionComplete}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg font-medium hover:shadow-lg transition-all"
                >
                  완료하기
                  <CheckCircle className="w-5 h-5" />
                </button>
              ) : (
                <Link
                  href="/modules/stock-analysis"
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-medium hover:shadow-lg transition-all"
                >
                  다음 코스 선택하기
                  <Sparkles className="w-5 h-5" />
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}