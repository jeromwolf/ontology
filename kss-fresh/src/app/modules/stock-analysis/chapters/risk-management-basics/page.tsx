'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, Shield, AlertTriangle, TrendingDown, TrendingUp, BarChart3, PieChart, Target, Calculator, Zap, DollarSign, Clock, ChevronRight, RefreshCw, Layers } from 'lucide-react';

function PositionSizingCalculator() {
  const [accountSize, setAccountSize] = useState(10000000); // 1천만원
  const [riskPercentage, setRiskPercentage] = useState(2); // 2%
  const [entryPrice, setEntryPrice] = useState(50000);
  const [stopLoss, setStopLoss] = useState(45000);
  const [stockName, setStockName] = useState('삼성전자');

  const riskAmount = accountSize * (riskPercentage / 100);
  const riskPerShare = entryPrice - stopLoss;
  const maxShares = riskPerShare > 0 ? Math.floor(riskAmount / riskPerShare) : 0;
  const totalInvestment = maxShares * entryPrice;
  const investmentRatio = (totalInvestment / accountSize) * 100;

  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <h2 className="text-2xl font-bold mb-6">🧮 포지션 사이징 계산기</h2>
      
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">투자 정보 입력</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">종목명</label>
                <input
                  type="text"
                  value={stockName}
                  onChange={(e) => setStockName(e.target.value)}
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-600"
                  placeholder="예: 삼성전자"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">계좌 총액</label>
                <div className="relative">
                  <input
                    type="number"
                    value={accountSize}
                    onChange={(e) => setAccountSize(Number(e.target.value))}
                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-600"
                    step="1000000"
                  />
                  <span className="absolute right-3 top-3 text-gray-500">원</span>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  리스크 허용 비율 ({riskPercentage}%)
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={riskPercentage}
                  onChange={(e) => setRiskPercentage(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.5%</span>
                  <span className="font-medium">매우 보수적: 1-2% / 적극적: 3-5%</span>
                  <span>10%</span>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">목표 매수가</label>
                  <div className="relative">
                    <input
                      type="number"
                      value={entryPrice}
                      onChange={(e) => setEntryPrice(Number(e.target.value))}
                      className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-600"
                      step="100"
                    />
                    <span className="absolute right-3 top-3 text-gray-500">원</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">손절선</label>
                  <div className="relative">
                    <input
                      type="number"
                      value={stopLoss}
                      onChange={(e) => setStopLoss(Number(e.target.value))}
                      className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-600"
                      step="100"
                    />
                    <span className="absolute right-3 top-3 text-gray-500">원</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">계산 결과</h3>
            
            {riskPerShare <= 0 ? (
              <div className="text-center py-8">
                <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <p className="text-red-600 dark:text-red-400 font-medium">
                  손절선이 매수가보다 높습니다
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  손절선을 매수가보다 낮게 설정해주세요
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                      {maxShares.toLocaleString()}
                    </div>
                    <div className="text-sm text-blue-700 dark:text-blue-300">최대 매수 가능 주수</div>
                  </div>
                  
                  <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {riskAmount.toLocaleString()}
                    </div>
                    <div className="text-sm text-green-700 dark:text-green-300">리스크 금액 (원)</div>
                  </div>
                </div>
                
                <div className="border-t border-gray-200 dark:border-gray-600 pt-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">총 투자금액</span>
                    <span className="font-semibold">{totalInvestment.toLocaleString()}원</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">계좌 대비 비율</span>
                    <span className="font-semibold">{investmentRatio.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">주당 리스크</span>
                    <span className="font-semibold text-red-600 dark:text-red-400">
                      -{riskPerShare.toLocaleString()}원 ({((riskPerShare/entryPrice)*100).toFixed(1)}%)
                    </span>
                  </div>
                </div>
                
                <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg mt-4">
                  <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">
                    💡 투자 시나리오
                  </h4>
                  <div className="text-sm space-y-1">
                    <div className="text-yellow-700 dark:text-yellow-300">
                      • <strong>{stockName}</strong> {maxShares.toLocaleString()}주를 {entryPrice.toLocaleString()}원에 매수
                    </div>
                    <div className="text-yellow-700 dark:text-yellow-300">
                      • 손절선 {stopLoss.toLocaleString()}원 도달 시 최대 손실: <strong>{riskAmount.toLocaleString()}원</strong>
                    </div>
                    <div className="text-yellow-700 dark:text-yellow-300">
                      • 전체 계좌의 <strong>{riskPercentage}%</strong>만 리스크 노출
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
            <h4 className="font-semibold mb-3">리스크 수준별 가이드</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-green-600 dark:text-green-400">보수적 (1-2%)</span>
                <span>안정적 장기투자</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-600 dark:text-blue-400">균형적 (2-3%)</span>
                <span>일반적 권장 수준</span>
              </div>
              <div className="flex justify-between">
                <span className="text-orange-600 dark:text-orange-400">적극적 (3-5%)</span>
                <span>고수익 추구</span>
              </div>
              <div className="flex justify-between">
                <span className="text-red-600 dark:text-red-400">위험 (5%+)</span>
                <span>전문가 수준</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DiversificationSimulator() {
  const [portfolios, setPortfolios] = useState([
    { name: '집중투자 포트폴리오', weights: [70, 20, 10, 0, 0], risk: 28.5, return: 12.5 },
    { name: '균형 포트폴리오', weights: [30, 25, 20, 15, 10], risk: 15.2, return: 9.8 },
    { name: '분산 포트폴리오', weights: [20, 20, 20, 20, 20], risk: 12.1, return: 8.5 }
  ]);
  
  const [selectedPortfolio, setSelectedPortfolio] = useState(1);
  const [customWeights, setCustomWeights] = useState([20, 20, 20, 20, 20]);
  
  const assets = [
    { name: '대형주', color: 'bg-blue-500', risk: 15, return: 8 },
    { name: '중소형주', color: 'bg-green-500', risk: 25, return: 12 },
    { name: '채권', color: 'bg-gray-500', risk: 5, return: 4 },
    { name: '리츠', color: 'bg-yellow-500', risk: 18, return: 7 },
    { name: '원자재', color: 'bg-red-500', risk: 30, return: 10 }
  ];

  const calculatePortfolioMetrics = (weights: number[]) => {
    const totalWeight = weights.reduce((sum, w) => sum + w, 0);
    if (totalWeight === 0) return { risk: 0, return: 0 };
    
    const normalizedWeights = weights.map(w => w / totalWeight);
    const portfolioReturn = normalizedWeights.reduce((sum, weight, i) => sum + weight * assets[i].return, 0);
    const portfolioRisk = Math.sqrt(normalizedWeights.reduce((sum, weight, i) => sum + Math.pow(weight * assets[i].risk, 2), 0));
    
    return { risk: portfolioRisk, return: portfolioReturn };
  };

  const customMetrics = calculatePortfolioMetrics(customWeights);

  const updateCustomWeight = (index: number, value: number) => {
    const newWeights = [...customWeights];
    newWeights[index] = Math.max(0, Math.min(100, value));
    setCustomWeights(newWeights);
  };

  const normalizeWeights = () => {
    const total = customWeights.reduce((sum, w) => sum + w, 0);
    if (total > 0) {
      const normalized = customWeights.map(w => Math.round(w / total * 100));
      setCustomWeights(normalized);
    }
  };

  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <h2 className="text-2xl font-bold mb-6">📊 분산투자 효과 시뮬레이터</h2>
      
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">포트폴리오 비교</h3>
            
            <div className="space-y-4">
              {portfolios.map((portfolio, index) => (
                <div 
                  key={index}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                    selectedPortfolio === index 
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                      : 'border-gray-200 dark:border-gray-600'
                  }`}
                  onClick={() => setSelectedPortfolio(index)}
                >
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-semibold">{portfolio.name}</h4>
                    <div className="text-right">
                      <div className="text-sm text-green-600 dark:text-green-400">
                        수익률: {portfolio.return}%
                      </div>
                      <div className="text-sm text-red-600 dark:text-red-400">
                        위험도: {portfolio.risk}%
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex gap-1 h-4 rounded overflow-hidden">
                    {assets.map((asset, assetIndex) => (
                      <div 
                        key={assetIndex}
                        className={asset.color}
                        style={{ width: `${portfolio.weights[assetIndex]}%` }}
                        title={`${asset.name}: ${portfolio.weights[assetIndex]}%`}
                      ></div>
                    ))}
                  </div>
                  
                  <div className="flex flex-wrap gap-2 mt-2">
                    {assets.map((asset, assetIndex) => (
                      portfolio.weights[assetIndex] > 0 && (
                        <div key={assetIndex} className="flex items-center gap-1 text-xs">
                          <div className={`w-3 h-3 ${asset.color} rounded`}></div>
                          <span>{asset.name} {portfolio.weights[assetIndex]}%</span>
                        </div>
                      )
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">나만의 포트폴리오 만들기</h3>
            
            <div className="space-y-4">
              {assets.map((asset, index) => (
                <div key={index}>
                  <div className="flex justify-between items-center mb-2">
                    <label className="flex items-center gap-2 text-sm font-medium">
                      <div className={`w-4 h-4 ${asset.color} rounded`}></div>
                      {asset.name}
                    </label>
                    <div className="text-sm text-gray-500">
                      {customWeights[index]}%
                    </div>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={customWeights[index]}
                    onChange={(e) => updateCustomWeight(index, Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              ))}
              
              <div className="flex justify-between items-center pt-4 border-t border-gray-200 dark:border-gray-600">
                <span className="font-medium">
                  총합: {customWeights.reduce((sum, w) => sum + w, 0)}%
                </span>
                <button
                  onClick={normalizeWeights}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
                >
                  100%로 조정
                </button>
              </div>
            </div>
          </div>
        </div>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">위험-수익률 차트</h3>
            
            <div className="relative h-64 bg-gray-50 dark:bg-gray-600 rounded-lg overflow-hidden">
              {/* 차트 배경 격자 */}
              <div className="absolute inset-0">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="absolute border-t border-gray-300 dark:border-gray-500" 
                       style={{ top: `${i * 20}%`, width: '100%' }}></div>
                ))}
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="absolute border-l border-gray-300 dark:border-gray-500" 
                       style={{ left: `${i * 20}%`, height: '100%' }}></div>
                ))}
              </div>
              
              {/* 축 레이블 */}
              <div className="absolute -bottom-6 left-0 right-0 flex justify-between text-xs text-gray-500">
                <span>0%</span>
                <span>위험도 (변동성)</span>
                <span>35%</span>
              </div>
              <div className="absolute -left-6 top-0 bottom-0 flex flex-col justify-between text-xs text-gray-500">
                <span>15%</span>
                <span className="transform -rotate-90 origin-center">수익률</span>
                <span>0%</span>
              </div>
              
              {/* 포트폴리오 점들 */}
              {portfolios.map((portfolio, index) => (
                <div
                  key={index}
                  className={`absolute w-4 h-4 rounded-full border-2 border-white transform -translate-x-2 -translate-y-2 ${
                    index === 0 ? 'bg-red-500' : index === 1 ? 'bg-blue-500' : 'bg-green-500'
                  } ${selectedPortfolio === index ? 'ring-2 ring-yellow-400' : ''}`}
                  style={{
                    left: `${(portfolio.risk / 35) * 100}%`,
                    bottom: `${(portfolio.return / 15) * 100}%`
                  }}
                  title={`${portfolio.name}: 위험도 ${portfolio.risk}%, 수익률 ${portfolio.return}%`}
                ></div>
              ))}
              
              {/* 사용자 정의 포트폴리오 */}
              <div
                className="absolute w-4 h-4 bg-purple-500 rounded-full border-2 border-white transform -translate-x-2 -translate-y-2 ring-2 ring-purple-300"
                style={{
                  left: `${(customMetrics.risk / 35) * 100}%`,
                  bottom: `${(customMetrics.return / 15) * 100}%`
                }}
                title={`나의 포트폴리오: 위험도 ${customMetrics.risk.toFixed(1)}%, 수익률 ${customMetrics.return.toFixed(1)}%`}
              ></div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">포트폴리오 분석</h3>
            
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {customMetrics.return.toFixed(1)}%
                </div>
                <div className="text-sm text-green-700 dark:text-green-300">예상 수익률</div>
              </div>
              
              <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {customMetrics.risk.toFixed(1)}%
                </div>
                <div className="text-sm text-red-700 dark:text-red-300">위험도 (변동성)</div>
              </div>
            </div>
            
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">샤프비율</span>
                <span className="font-medium">
                  {((customMetrics.return - 3) / customMetrics.risk).toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">분산효과</span>
                <span className="font-medium">
                  {(() => {
                    const nonZeroWeights = customWeights.filter(w => w > 0).length;
                    const totalWeight = customWeights.reduce((sum, w) => sum + w, 0);
                    const maxWeight = Math.max(...customWeights) / totalWeight * 100;
                    
                    if (nonZeroWeights === 1) {
                      return <span className="text-red-600 dark:text-red-400">집중 위험</span>;
                    } else if (maxWeight > 70) {
                      return <span className="text-orange-600 dark:text-orange-400">편중됨</span>;
                    } else if (nonZeroWeights >= 4 && maxWeight < 40) {
                      return <span className="text-green-600 dark:text-green-400">우수</span>;
                    } else if (nonZeroWeights >= 3) {
                      return <span className="text-blue-600 dark:text-blue-400">양호</span>;
                    } else {
                      return <span className="text-orange-600 dark:text-orange-400">부족</span>;
                    }
                  })()}
                </span>
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded text-sm text-blue-700 dark:text-blue-300">
              💡 <strong>분산투자 팁:</strong> 상관관계가 낮은 자산들을 조합하면 
              전체 포트폴리오의 위험을 줄이면서도 수익률을 유지할 수 있습니다.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function QuizSection() {
  const [answers, setAnswers] = useState<{ q1: string; q2: string; q3: string }>({ q1: '', q2: '', q3: '' });
  const [showResults, setShowResults] = useState(false);
  
  const correctAnswers = {
    q1: 'q1-2', // 계좌의 1-2%만 위험에 노출시킨다
    q2: 'q2-3', // 상관관계가 낮은 서로 다른 자산에 투자한다
    q3: 'q3-1'  // 주가가 매수 가격 대비 5-10% 하락했을 때
  };
  
  const handleAnswerChange = (question: string, value: string) => {
    if (!showResults) {
      setAnswers(prev => ({ ...prev, [question]: value }));
    }
  };
  
  const checkAnswers = () => {
    if (answers.q1 && answers.q2 && answers.q3) {
      setShowResults(true);
    } else {
      alert('모든 문제에 답해주세요.');
    }
  };
  
  const resetQuiz = () => {
    setAnswers({ q1: '', q2: '', q3: '' });
    setShowResults(false);
  };
  
  const getResultStyle = (question: 'q1' | 'q2' | 'q3', optionValue: string) => {
    if (!showResults) return '';
    
    const userAnswer = answers[question];
    const correctAnswer = correctAnswers[question];
    
    if (optionValue === correctAnswer) {
      return 'text-green-600 dark:text-green-400 font-medium';
    } else if (optionValue === userAnswer && optionValue !== correctAnswer) {
      return 'text-red-600 dark:text-red-400';
    }
    return 'text-gray-400';
  };
  
  const getResultIcon = (question: 'q1' | 'q2' | 'q3', optionValue: string) => {
    if (!showResults) return '';
    
    const userAnswer = answers[question];
    const correctAnswer = correctAnswers[question];
    
    if (optionValue === correctAnswer) {
      return ' ✓';
    } else if (optionValue === userAnswer && optionValue !== correctAnswer) {
      return ' ✗';
    }
    return '';
  };
  
  const score = showResults 
    ? Object.keys(correctAnswers).filter(q => answers[q as keyof typeof answers] === correctAnswers[q as keyof typeof correctAnswers]).length
    : 0;
  
  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <h2 className="text-2xl font-bold mb-6">🧠 이해도 체크</h2>
      
      {showResults && (
        <div className={`mb-6 p-4 rounded-lg ${
          score === 3 ? 'bg-green-100 dark:bg-green-900/10 text-green-700 dark:text-green-300' 
          : score === 2 ? 'bg-yellow-100 dark:bg-yellow-900/10 text-yellow-700 dark:text-yellow-300'
          : score === 1 ? 'bg-orange-100 dark:bg-orange-900/10 text-orange-700 dark:text-orange-300'
          : 'bg-red-100 dark:bg-red-900/10 text-red-700 dark:text-red-300'
        }`}>
          <p className="font-semibold">
            {score === 3 ? '🎉 완벽합니다!' : score === 2 ? '😊 잘하셨어요!' : score === 1 ? '💪 조금 더 공부해보세요!' : '📚 다시 학습해보세요!'}
            {` ${score}/3 문제를 맞추셨습니다.`}
          </p>
        </div>
      )}
      
      <div className="space-y-6">
        <div>
          <h3 className="font-semibold mb-3">Q1. 올바른 포지션 사이징 방법은?</h3>
          <div className="space-y-2 ml-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q1" 
                value="q1-1"
                checked={answers.q1 === 'q1-1'}
                onChange={(e) => handleAnswerChange('q1', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q1', 'q1-1')}>
                계좌의 10-20%를 한 종목에 투자한다{getResultIcon('q1', 'q1-1')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q1" 
                value="q1-2"
                checked={answers.q1 === 'q1-2'}
                onChange={(e) => handleAnswerChange('q1', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q1', 'q1-2')}>
                계좌의 1-2%만 위험에 노출시킨다{getResultIcon('q1', 'q1-2')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q1" 
                value="q1-3"
                checked={answers.q1 === 'q1-3'}
                onChange={(e) => handleAnswerChange('q1', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q1', 'q1-3')}>
                직감에 따라 투자 비중을 정한다{getResultIcon('q1', 'q1-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q2. 효과적인 분산투자 방법은?</h3>
          <div className="space-y-2 ml-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q2" 
                value="q2-1"
                checked={answers.q2 === 'q2-1'}
                onChange={(e) => handleAnswerChange('q2', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q2', 'q2-1')}>
                같은 업종의 여러 종목에 투자한다{getResultIcon('q2', 'q2-1')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q2" 
                value="q2-2"
                checked={answers.q2 === 'q2-2'}
                onChange={(e) => handleAnswerChange('q2', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q2', 'q2-2')}>
                투자 종목 수를 많이 늘린다{getResultIcon('q2', 'q2-2')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q2" 
                value="q2-3"
                checked={answers.q2 === 'q2-3'}
                onChange={(e) => handleAnswerChange('q2', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q2', 'q2-3')}>
                상관관계가 낮은 서로 다른 자산에 투자한다{getResultIcon('q2', 'q2-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q3. 일반적으로 손절선을 설정하는 기준은?</h3>
          <div className="space-y-2 ml-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q3" 
                value="q3-1"
                checked={answers.q3 === 'q3-1'}
                onChange={(e) => handleAnswerChange('q3', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q3', 'q3-1')}>
                매수 가격 대비 5-10% 하락했을 때{getResultIcon('q3', 'q3-1')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q3" 
                value="q3-2"
                checked={answers.q3 === 'q3-2'}
                onChange={(e) => handleAnswerChange('q3', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q3', 'q3-2')}>
                매수 가격 대비 20-30% 하락했을 때{getResultIcon('q3', 'q3-2')}
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="radio" 
                name="q3" 
                value="q3-3"
                checked={answers.q3 === 'q3-3'}
                onChange={(e) => handleAnswerChange('q3', e.target.value)}
                disabled={showResults}
                className="w-4 h-4" 
              />
              <span className={getResultStyle('q3', 'q3-3')}>
                기업의 펀더멘털이 나빠질 때{getResultIcon('q3', 'q3-3')}
              </span>
            </label>
          </div>
        </div>
      </div>

      <div className="flex gap-3 mt-8">
        {!showResults ? (
          <button
            onClick={checkAnswers}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
          >
            정답 확인하기
          </button>
        ) : (
          <button
            onClick={resetQuiz}
            className="px-6 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
          >
            다시 풀기
          </button>
        )}
      </div>
    </div>
  );
}

export default function RiskManagementBasicsPage() {
  const riskTypes = [
    {
      title: '시장 리스크 (Market Risk)',
      icon: TrendingDown,
      color: 'red',
      description: '전체 시장이 하락하여 발생하는 위험',
      examples: ['금리 상승', '경제 침체', '정치적 불안정', '자연재해'],
      management: ['분산투자', '헤징 전략', '현금 보유 비율 조정', '시황 분석']
    },
    {
      title: '개별 종목 리스크 (Specific Risk)',
      icon: Target,
      color: 'orange',
      description: '특정 기업이나 업종에만 영향을 미치는 위험',
      examples: ['실적 악화', '경영진 교체', '제품 결함', '규제 변화'],
      management: ['종목 분산', '기업 분석', '정기적 리밸런싱', '정보 모니터링']
    },
    {
      title: '유동성 리스크 (Liquidity Risk)',
      icon: Zap,
      color: 'blue',
      description: '원하는 시점에 적정 가격으로 매매할 수 없는 위험',
      examples: ['거래량 부족', '시장 폐쇄', '매수세 급감', '호가 스프레드 확대'],
      management: ['우량주 위주 투자', '적절한 포지션 크기', '시장 시간 고려', '비상 자금 확보']
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <Link 
            href="/modules/stock-analysis"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Stock Analysis로 돌아가기</span>
          </Link>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Chapter Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-16 h-16 bg-yellow-100 dark:bg-yellow-900/30 rounded-full flex items-center justify-center">
              <Shield className="w-8 h-8 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div className="text-left">
              <div className="text-sm text-gray-500 mb-1">Baby Chick • Chapter 5</div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                리스크 관리 기초
              </h1>
            </div>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            투자에서 가장 중요한 것은 수익이 아닌 손실 방지입니다. 체계적인 리스크 관리 방법을 학습해보세요.
          </p>
        </div>

        {/* Learning Objectives */}
        <div className="bg-blue-50 dark:bg-blue-900/10 rounded-xl p-6 mb-8">
          <h2 className="text-xl font-bold text-blue-900 dark:text-blue-300 mb-4">
            📚 학습 목표
          </h2>
          <ul className="space-y-2 text-blue-800 dark:text-blue-300">
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>투자 리스크의 유형과 특성 이해</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>포지션 사이징과 손절선 설정 방법</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>분산투자의 원리와 실전 적용법</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>위험-수익률 관점에서의 포트폴리오 구성</span>
            </li>
          </ul>
        </div>

        {/* Main Content */}
        <div className="space-y-12">
          {/* Section 1: Risk Types */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              1️⃣ 투자 리스크의 이해
            </h2>
            
            <div className="bg-red-50 dark:bg-red-900/10 p-6 rounded-xl mb-8">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
                <div>
                  <h3 className="text-xl font-bold text-red-800 dark:text-red-300">투자의 첫 번째 원칙</h3>
                  <p className="text-red-600 dark:text-red-400">"첫 번째 룰: 돈을 잃지 마라. 두 번째 룰: 첫 번째 룰을 잊지 마라." - 워렌 버핏</p>
                </div>
              </div>
            </div>

            <div className="grid gap-6">
              {riskTypes.map((risk, index) => {
                const Icon = risk.icon;
                const colorClasses = {
                  red: 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/10',
                  orange: 'border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-900/10',
                  blue: 'border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/10'
                };
                const iconColorClasses = {
                  red: 'text-red-600 dark:text-red-400',
                  orange: 'text-orange-600 dark:text-orange-400',
                  blue: 'text-blue-600 dark:text-blue-400'
                };

                return (
                  <div key={index} className={`border-2 rounded-xl p-6 ${colorClasses[risk.color as keyof typeof colorClasses]}`}>
                    <div className="flex items-start gap-4 mb-4">
                      <div className="w-12 h-12 bg-white dark:bg-gray-700 rounded-full flex items-center justify-center shadow-md">
                        <Icon className={`w-6 h-6 ${iconColorClasses[risk.color as keyof typeof iconColorClasses]}`} />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                          {risk.title}
                        </h3>
                        <p className="text-gray-600 dark:text-gray-400">
                          {risk.description}
                        </p>
                      </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                        <h4 className="font-semibold text-red-700 dark:text-red-400 mb-3">주요 사례</h4>
                        <ul className="space-y-1 text-sm">
                          {risk.examples.map((example, idx) => (
                            <li key={idx} className="text-gray-600 dark:text-gray-400">
                              • {example}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="bg-white dark:bg-gray-700 p-4 rounded-lg">
                        <h4 className="font-semibold text-green-700 dark:text-green-400 mb-3">관리 방법</h4>
                        <ul className="space-y-1 text-sm">
                          {risk.management.map((method, idx) => (
                            <li key={idx} className="text-gray-600 dark:text-gray-400">
                              • {method}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Section 2: Position Sizing */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              2️⃣ 포지션 사이징 (Position Sizing)
            </h2>
            
            <div className="bg-red-50 dark:bg-red-900/10 p-6 rounded-xl mb-8">
              <h3 className="text-xl font-bold text-red-800 dark:text-red-300 mb-4">
                🚨 초보자가 가장 많이 하는 실수
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-gray-700 p-4 rounded-lg border-l-4 border-red-500">
                  <h4 className="font-bold text-red-700 dark:text-red-400 mb-2">❌ 틀린 생각</h4>
                  <p className="text-sm text-red-600 dark:text-red-300 mb-2">
                    "1천만원 계좌니까 삼성전자 500만원어치 사야지!"
                  </p>
                  <div className="text-xs text-red-500 dark:text-red-400">
                    → 10% 하락 시 50만원 손실 (계좌의 5%)<br/>
                    → 20% 하락 시 100만원 손실 (계좌의 10%)<br/>
                    → <strong>손실 규모 예측 불가능</strong>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-700 p-4 rounded-lg border-l-4 border-green-500">
                  <h4 className="font-bold text-green-700 dark:text-green-400 mb-2">✅ 올바른 생각</h4>
                  <p className="text-sm text-green-600 dark:text-green-300 mb-2">
                    "이 투자에서 최대 20만원까지만 잃을 수 있어"
                  </p>
                  <div className="text-xs text-green-500 dark:text-green-400">
                    → 손절까지 거리: 5천원 (50,000→45,000)<br/>
                    → 20만원 ÷ 5천원 = 40주만 매수<br/>
                    → <strong>손실이 정확히 통제됨</strong>
                  </div>
                </div>
              </div>
            </div>

            <div className="prose prose-lg dark:prose-invert max-w-none mb-8">
              <p>
                포지션 사이징은 <strong>"얼마를 투자할까?"가 아니라 "얼마까지 잃을까?"</strong>를 먼저 정하는 것입니다. 
                이는 투자에서 가장 중요한 개념이지만, 대부분의 개인투자자가 간과하는 부분입니다.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-8">
              <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-green-800 dark:text-green-300 mb-4">
                  <Calculator className="inline w-5 h-5 mr-2" />
                  1% 규칙 (1% Rule)
                </h3>
                
                <div className="space-y-3 text-sm text-green-700 dark:text-green-300">
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <p className="font-semibold mb-2">기본 원칙</p>
                    <p className="text-gray-600 dark:text-gray-400">
                      한 번의 거래에서 전체 계좌의 1-2%만 위험에 노출시킨다
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <p className="font-semibold mb-2">계산 공식</p>
                    <p className="text-gray-600 dark:text-gray-400">
                      투자 주수 = (계좌 크기 × 위험 비율) ÷ (매수가 - 손절가)
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <p className="font-semibold mb-2">실제 예시</p>
                    <p className="text-gray-600 dark:text-gray-400">
                      1천만원 계좌, 2% 위험: 20만원만 손실 허용
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 mb-4">
                  <Shield className="inline w-5 h-5 mr-2" />
                  손절선 설정
                </h3>
                
                <div className="space-y-3 text-sm text-blue-700 dark:text-blue-300">
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <p className="font-semibold mb-2">기술적 손절</p>
                    <p className="text-gray-600 dark:text-gray-400">
                      지지선 하향 돌파, 이동평균선 이탈 등
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <p className="font-semibold mb-2">비율 기준 손절</p>
                    <p className="text-gray-600 dark:text-gray-400">
                      매수가 대비 5-10% 하락 시점
                    </p>
                  </div>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded">
                    <p className="font-semibold mb-2">펀더멘털 손절</p>
                    <p className="text-gray-600 dark:text-gray-400">
                      투자 논리 훼손, 기업 가치 변화 시
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-xl mb-8">
              <h3 className="text-lg font-bold text-yellow-800 dark:text-yellow-300 mb-4">
                💡 왜 200만원만 투자해야 할까? - 실전 시나리오
              </h3>
              
              <div className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 mb-3">🔥 연속 손실 시나리오</h4>
                    <p className="text-sm text-yellow-600 dark:text-yellow-300 mb-3">
                      <strong>상황:</strong> 10번 투자해서 7번 실패, 3번 성공
                    </p>
                    
                    <div className="bg-white dark:bg-gray-700 p-3 rounded mb-3">
                      <p className="font-semibold text-red-600 dark:text-red-400 text-sm mb-1">❌ 잘못된 방식 (계좌의 10%씩 투자)</p>
                      <p className="text-xs text-red-500 dark:text-red-400">
                        • 7번 실패: -10% × 7 = <strong>-70%</strong><br/>
                        • 계좌의 30%만 남음<br/>
                        • 3번 성공해도 회복 불가능 💀
                      </p>
                    </div>
                    
                    <div className="bg-white dark:bg-gray-700 p-3 rounded">
                      <p className="font-semibold text-green-600 dark:text-green-400 text-sm mb-1">✅ 올바른 방식 (계좌의 2%씩 리스크)</p>
                      <p className="text-xs text-green-500 dark:text-green-400">
                        • 7번 실패: -2% × 7 = <strong>-14%</strong><br/>
                        • 계좌의 86% 유지<br/>
                        • 3번 성공으로 쉽게 회복 ✨
                      </p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 mb-3">😰 감정적 거래 방지</h4>
                    
                    <div className="space-y-3">
                      <div className="bg-white dark:bg-gray-700 p-3 rounded border-l-4 border-red-500">
                        <p className="font-semibold text-red-600 dark:text-red-400 text-sm">❌ 큰 포지션</p>
                        <p className="text-xs text-red-500 dark:text-red-400">
                          500만원 투자 → 100만원 손실<br/>
                          → "큰일났다!" → 패닉 매도<br/>
                          → 더 큰 손실
                        </p>
                      </div>
                      
                      <div className="bg-white dark:bg-gray-700 p-3 rounded border-l-4 border-green-500">
                        <p className="font-semibold text-green-600 dark:text-green-400 text-sm">✅ 작은 포지션</p>
                        <p className="text-xs text-green-500 dark:text-green-400">
                          200만원 투자 → 20만원 손실<br/>
                          → "예상 범위 내" → 냉정한 판단<br/>
                          → 계획대로 실행
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-blue-100 dark:bg-blue-900/20 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">📊 투자금액별 비교표</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead className="text-blue-700 dark:text-blue-300">
                        <tr className="border-b border-blue-200 dark:border-blue-800">
                          <th className="text-left py-1">투자금액</th>
                          <th className="text-center py-1">주수</th>
                          <th className="text-center py-1">손절시 손실</th>
                          <th className="text-center py-1">계좌 비중</th>
                          <th className="text-center py-1">평가</th>
                        </tr>
                      </thead>
                      <tbody className="text-gray-600 dark:text-gray-400">
                        <tr className="border-b border-blue-100 dark:border-blue-900">
                          <td className="py-1">1,000만원</td>
                          <td className="text-center">200주</td>
                          <td className="text-center font-bold text-red-600 dark:text-red-400">100만원</td>
                          <td className="text-center font-bold text-red-600 dark:text-red-400">10%</td>
                          <td className="text-center">❌ 위험</td>
                        </tr>
                        <tr className="border-b border-blue-100 dark:border-blue-900">
                          <td className="py-1">500만원</td>
                          <td className="text-center">100주</td>
                          <td className="text-center font-bold text-orange-600 dark:text-orange-400">50만원</td>
                          <td className="text-center font-bold text-orange-600 dark:text-orange-400">5%</td>
                          <td className="text-center">⚠️ 주의</td>
                        </tr>
                        <tr className="bg-green-50 dark:bg-green-900/10">
                          <td className="py-1 font-bold">200만원</td>
                          <td className="text-center font-bold">40주</td>
                          <td className="text-center font-bold text-green-600 dark:text-green-400">20만원</td>
                          <td className="text-center font-bold text-green-600 dark:text-green-400">2%</td>
                          <td className="text-center">✅ 안전</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            <PositionSizingCalculator />
          </section>

          {/* Section 3: Diversification */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              3️⃣ 분산투자의 과학
            </h2>
            
            <div className="prose prose-lg dark:prose-invert max-w-none mb-8">
              <p>
                분산투자는 "<strong>모든 계란을 한 바구니에 담지 마라</strong>"는 격언으로 유명합니다. 
                하지만 단순히 종목 수를 늘리는 것이 아닌, <strong>상관관계가 낮은 자산들을 조합</strong>하는 것이 핵심입니다.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <div className="bg-purple-50 dark:bg-purple-900/10 p-6 rounded-xl text-center">
                <div className="w-16 h-16 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Layers className="w-8 h-8 text-purple-600 dark:text-purple-400" />
                </div>
                <h3 className="text-lg font-bold text-purple-800 dark:text-purple-300 mb-2">
                  자산군 분산
                </h3>
                <p className="text-sm text-purple-600 dark:text-purple-300">
                  주식, 채권, 리츠, 원자재 등 서로 다른 성격의 자산 조합
                </p>
              </div>

              <div className="bg-teal-50 dark:bg-teal-900/10 p-6 rounded-xl text-center">
                <div className="w-16 h-16 bg-teal-100 dark:bg-teal-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-8 h-8 text-teal-600 dark:text-teal-400" />
                </div>
                <h3 className="text-lg font-bold text-teal-800 dark:text-teal-300 mb-2">
                  업종 분산
                </h3>
                <p className="text-sm text-teal-600 dark:text-teal-300">
                  IT, 금융, 헬스케어, 에너지 등 다양한 산업 섹터 투자
                </p>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/10 p-6 rounded-xl text-center">
                <div className="w-16 h-16 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                  <DollarSign className="w-8 h-8 text-orange-600 dark:text-orange-400" />
                </div>
                <h3 className="text-lg font-bold text-orange-800 dark:text-orange-300 mb-2">
                  지역 분산
                </h3>
                <p className="text-sm text-orange-600 dark:text-orange-300">
                  국내, 선진국, 신흥국 등 지역별 경제 성장 차이 활용
                </p>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-xl mb-8">
              <h3 className="text-lg font-bold text-yellow-800 dark:text-yellow-300 mb-4">
                📊 분산투자 효과의 수학적 증명
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 mb-2">포트폴리오 위험도 공식</h4>
                  <div className="bg-white dark:bg-gray-700 p-3 rounded font-mono text-sm">
                    σp = √(w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂)
                  </div>
                  <p className="text-xs text-yellow-600 dark:text-yellow-300 mt-2">
                    ρ₁₂(상관계수)가 낮을수록 포트폴리오 위험 감소
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 mb-2">상관계수별 분산 효과</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>ρ = 1.0 (완전 양의 상관)</span>
                      <span className="text-red-600 dark:text-red-400">분산 효과 없음</span>
                    </div>
                    <div className="flex justify-between">
                      <span>ρ = 0.5 (중간 상관)</span>
                      <span className="text-yellow-600 dark:text-yellow-400">부분적 효과</span>
                    </div>
                    <div className="flex justify-between">
                      <span>ρ = 0.0 (무상관)</span>
                      <span className="text-blue-600 dark:text-blue-400">좋은 분산 효과</span>
                    </div>
                    <div className="flex justify-between">
                      <span>ρ = -1.0 (완전 음의 상관)</span>
                      <span className="text-green-600 dark:text-green-400">최적 분산 효과</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <DiversificationSimulator />
          </section>

          {/* Section 4: Risk Management Tools */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              4️⃣ 실전 리스크 관리 도구
            </h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 mb-4">
                  <RefreshCw className="inline w-5 h-5 mr-2" />
                  정기적 리밸런싱
                </h3>
                
                <div className="space-y-3 text-sm text-blue-700 dark:text-blue-300">
                  <div>
                    <strong>목적:</strong> 목표 자산 배분 비율 유지
                  </div>
                  <div>
                    <strong>시기:</strong> 분기별 또는 비중 5%p 이상 변화 시
                  </div>
                  <div>
                    <strong>방법:</strong> 상승 자산 매도 → 하락 자산 매수
                  </div>
                  <div>
                    <strong>효과:</strong> 자동적인 고점 매도, 저점 매수
                  </div>
                </div>

                <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900/20 rounded text-xs">
                  <strong>리밸런싱 예시:</strong><br/>
                  목표 비율이 주식 60%, 채권 40%인데 주식이 70%가 되었다면 
                  주식 10%를 매도하여 채권으로 전환
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-green-800 dark:text-green-300 mb-4">
                  <PieChart className="inline w-5 h-5 mr-2" />
                  핵심-위성 전략
                </h3>
                
                <div className="space-y-3 text-sm text-green-700 dark:text-green-300">
                  <div>
                    <strong>핵심 (70-80%):</strong> 안정적 대형주, 인덱스 펀드
                  </div>
                  <div>
                    <strong>위성 (20-30%):</strong> 고성장주, 테마주, 해외주식
                  </div>
                  <div>
                    <strong>장점:</strong> 안정성과 수익성의 균형
                  </div>
                  <div>
                    <strong>관리:</strong> 위성 부분만 적극적 매매
                  </div>
                </div>

                <div className="mt-4 p-3 bg-green-100 dark:bg-green-900/20 rounded text-xs">
                  <strong>구성 예시:</strong><br/>
                  핵심: KODEX 200 ETF, 삼성전자, SK하이닉스<br/>
                  위성: 2차전지, 바이오, 미국 성장주
                </div>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl mt-6">
              <h3 className="text-lg font-bold text-gray-800 dark:text-gray-300 mb-4">
                🎯 단계별 리스크 관리 체크리스트
              </h3>
              
              <div className="grid md:grid-cols-3 gap-6">
                <div>
                  <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-3">투자 전</h4>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li>☐ 투자 목적과 기간 명확화</li>
                    <li>☐ 리스크 허용 수준 설정</li>
                    <li>☐ 자산 배분 계획 수립</li>
                    <li>☐ 손절선 미리 설정</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold text-green-700 dark:text-green-400 mb-3">투자 중</h4>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li>☐ 포지션 사이징 준수</li>
                    <li>☐ 정기적 포트폴리오 점검</li>
                    <li>☐ 리밸런싱 실시</li>
                    <li>☐ 감정적 결정 지양</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-3">투자 후</h4>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li>☐ 매매일지 작성</li>
                    <li>☐ 성과 분석 및 반성</li>
                    <li>☐ 전략 개선 사항 도출</li>
                    <li>☐ 다음 투자 계획 수립</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* Interactive Quiz */}
          <section>
            <QuizSection />
          </section>
        </div>

        {/* Next Steps */}
        <div className="mt-16 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/10 dark:to-orange-900/10 rounded-2xl p-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-white dark:bg-gray-700 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-2xl">🎮</span>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                다음 단계로 진행
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                리스크 관리를 배웠다면 이제 실전 시뮬레이션으로 경험을 쌓아보세요.
              </p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                🎮 Chapter 6: 투자 심리 시뮬레이션
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                실제 시장 상황을 재현한 시뮬레이션으로 심리적 함정을 체험하고 극복하는 훈련을 해보세요.
              </p>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Clock className="w-4 h-4" />
                  <span>예상 학습시간: 90분</span>
                </div>
                <Link
                  href="/modules/stock-analysis/chapters/psychology-simulation"
                  className="inline-flex items-center gap-1 text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
                >
                  <span>시작하기</span>
                  <ChevronRight className="w-4 h-4" />
                </Link>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                🏛️ 전체 커리큘럼 보기
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                Baby Chick 단계의 전체 학습 경로를 확인하고 나만의 학습 계획을 세워보세요.
              </p>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Target className="w-4 h-4" />
                  <span>총 9개 챕터</span>
                </div>
                <Link
                  href="/modules/stock-analysis/stages/foundation"
                  className="inline-flex items-center gap-1 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  <span>전체 보기</span>
                  <ChevronRight className="w-4 h-4" />
                </Link>
              </div>
            </div>
          </div>

          {/* Progress Indicator */}
          <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
              <span>Baby Chick 진행률</span>
              <span>5/9 완료</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full" style={{ width: '56%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}