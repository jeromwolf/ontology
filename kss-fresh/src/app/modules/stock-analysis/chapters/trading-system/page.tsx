'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Monitor, Activity, BarChart3, TrendingUp, AlertCircle, ChevronRight, Play, Clock, Target, Zap, DollarSign, ArrowUpDown, Layers } from 'lucide-react';

function QuizSection() {
  const [answers, setAnswers] = useState<{ q1: string; q2: string; q3: string }>({ q1: '', q2: '', q3: '' });
  const [showResults, setShowResults] = useState(false);
  
  const correctAnswers = {
    q1: 'q1-3', // 매도 호가가 매수 호가보다 낮아야 체결된다
    q2: 'q2-2', // 시장가 주문은 즉시 체결되지만 가격을 보장할 수 없다
    q3: 'q3-1'  // 호가창에서 매수량이 많을수록 지지가 강하다는 신호
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
          <h3 className="font-semibold mb-3">Q1. 주식 거래에서 체결이 이루어지는 조건은?</h3>
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
                매수 호가가 매도 호가보다 높아야 체결된다{getResultIcon('q1', 'q1-1')}
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
                매수 호가와 매도 호가가 같아야 체결된다{getResultIcon('q1', 'q1-2')}
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
                매도 호가가 매수 호가보다 낮거나 같아야 체결된다{getResultIcon('q1', 'q1-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q2. 시장가 주문과 지정가 주문의 차이점은?</h3>
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
                시장가 주문은 가격을 보장하고, 지정가 주문은 체결을 보장한다{getResultIcon('q2', 'q2-1')}
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
                시장가 주문은 즉시 체결되지만 가격을 보장할 수 없다{getResultIcon('q2', 'q2-2')}
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
                두 주문 유형은 동일한 체결 조건을 갖는다{getResultIcon('q2', 'q2-3')}
              </span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="font-semibold mb-3">Q3. 호가창에서 매수량이 매도량보다 많다는 것은 무엇을 의미하는가?</h3>
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
                매수 의지가 강해 주가 상승 압력이 있다는 신호{getResultIcon('q3', 'q3-1')}
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
                매도 의지가 강해 주가 하락 압력이 있다는 신호{getResultIcon('q3', 'q3-2')}
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
                시장의 균형 상태를 나타낸다{getResultIcon('q3', 'q3-3')}
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

function TradingSimulator() {
  const [currentPrice, setCurrentPrice] = useState(85000);
  const [orderType, setOrderType] = useState<'market' | 'limit'>('limit');
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [quantity, setQuantity] = useState(10);
  const [limitPrice, setLimitPrice] = useState(85000);
  const [orders, setOrders] = useState<Array<{
    id: string;
    type: 'market' | 'limit';
    side: 'buy' | 'sell';
    quantity: number;
    price?: number;
    status: 'pending' | 'filled' | 'cancelled';
    time: string;
  }>>([]);

  const bidData = [
    { price: 84950, quantity: 1200, total: 1200 },
    { price: 84900, quantity: 850, total: 2050 },
    { price: 84850, quantity: 2300, total: 4350 },
    { price: 84800, quantity: 1800, total: 6150 },
    { price: 84750, quantity: 950, total: 7100 }
  ];

  const askData = [
    { price: 85000, quantity: 800, total: 800 },
    { price: 85050, quantity: 1500, total: 2300 },
    { price: 85100, quantity: 900, total: 3200 },
    { price: 85150, quantity: 1200, total: 4400 },
    { price: 85200, quantity: 650, total: 5050 }
  ];

  const handleOrderSubmit = () => {
    const newOrder = {
      id: Date.now().toString(),
      type: orderType,
      side: orderSide,
      quantity,
      price: orderType === 'limit' ? limitPrice : undefined,
      status: 'pending' as const,
      time: new Date().toLocaleTimeString('ko-KR')
    };

    setOrders(prev => [newOrder, ...prev]);

    // 시뮬레이션: 시장가 주문은 즉시 체결
    if (orderType === 'market') {
      setTimeout(() => {
        setOrders(prev => prev.map(order => 
          order.id === newOrder.id 
            ? { ...order, status: 'filled' as const, price: currentPrice }
            : order
        ));
      }, 500);
    }
    // 지정가 주문은 조건부 체결 시뮬레이션
    else if (orderType === 'limit') {
      const canFill = orderSide === 'buy' 
        ? limitPrice >= currentPrice 
        : limitPrice <= currentPrice;
      
      if (canFill) {
        setTimeout(() => {
          setOrders(prev => prev.map(order => 
            order.id === newOrder.id 
              ? { ...order, status: 'filled' as const }
              : order
          ));
        }, 1000);
      }
    }
  };

  return (
    <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-8">
      <h2 className="text-2xl font-bold mb-6">📊 호가창 & 주문 시뮬레이터</h2>
      
      <div className="grid lg:grid-cols-3 gap-6">
        {/* 호가창 */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4 text-center">호가창</h3>
            
            {/* 매도 호가 */}
            <div className="space-y-1 mb-2">
              {askData.reverse().map((ask, index) => (
                <div key={ask.price} className="grid grid-cols-3 text-xs">
                  <span className="text-right text-blue-600 dark:text-blue-400">{ask.quantity.toLocaleString()}</span>
                  <span className="text-center text-red-600 font-medium">{ask.price.toLocaleString()}</span>
                  <span className="text-red-200 text-right">{ask.total.toLocaleString()}</span>
                </div>
              ))}
            </div>
            
            {/* 현재가 */}
            <div className="bg-gray-100 dark:bg-gray-600 py-2 px-3 my-2 rounded text-center">
              <div className="text-lg font-bold text-gray-900 dark:text-white">
                {currentPrice.toLocaleString()}원
              </div>
              <div className="text-xs text-gray-500">현재가</div>
            </div>
            
            {/* 매수 호가 */}
            <div className="space-y-1">
              {bidData.map((bid, index) => (
                <div key={bid.price} className="grid grid-cols-3 text-xs">
                  <span className="text-blue-200">{bid.total.toLocaleString()}</span>
                  <span className="text-center text-blue-600 font-medium">{bid.price.toLocaleString()}</span>
                  <span className="text-right text-blue-600 dark:text-blue-400">{bid.quantity.toLocaleString()}</span>
                </div>
              ))}
            </div>
            
            <div className="grid grid-cols-3 text-xs mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
              <span className="text-gray-500">누적수량</span>
              <span className="text-center text-gray-500">가격</span>
              <span className="text-right text-gray-500">잔량</span>
            </div>
          </div>
        </div>
        
        {/* 주문 패널 */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4">주문 입력</h3>
            
            <div className="space-y-4">
              {/* 매수/매도 선택 */}
              <div>
                <label className="block text-sm font-medium mb-2">주문 구분</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setOrderSide('buy')}
                    className={`py-2 px-4 rounded text-sm font-medium ${
                      orderSide === 'buy' 
                        ? 'bg-red-600 text-white' 
                        : 'bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    매수
                  </button>
                  <button
                    onClick={() => setOrderSide('sell')}
                    className={`py-2 px-4 rounded text-sm font-medium ${
                      orderSide === 'sell' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    매도
                  </button>
                </div>
              </div>
              
              {/* 주문 유형 */}
              <div>
                <label className="block text-sm font-medium mb-2">주문 유형</label>
                <select 
                  value={orderType}
                  onChange={(e) => setOrderType(e.target.value as 'market' | 'limit')}
                  className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-600"
                >
                  <option value="market">시장가</option>
                  <option value="limit">지정가</option>
                </select>
              </div>
              
              {/* 수량 */}
              <div>
                <label className="block text-sm font-medium mb-2">수량</label>
                <input
                  type="number"
                  value={quantity}
                  onChange={(e) => setQuantity(Number(e.target.value))}
                  className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-600"
                  min="1"
                />
              </div>
              
              {/* 지정가 */}
              {orderType === 'limit' && (
                <div>
                  <label className="block text-sm font-medium mb-2">지정가격</label>
                  <input
                    type="number"
                    value={limitPrice}
                    onChange={(e) => setLimitPrice(Number(e.target.value))}
                    className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-600"
                    step="50"
                  />
                </div>
              )}
              
              {/* 주문 금액 */}
              <div className="bg-gray-100 dark:bg-gray-600 p-3 rounded">
                <div className="text-sm text-gray-600 dark:text-gray-400">주문 금액</div>
                <div className="text-lg font-semibold">
                  {(quantity * (orderType === 'limit' ? limitPrice : currentPrice)).toLocaleString()}원
                </div>
              </div>
              
              <button
                onClick={handleOrderSubmit}
                className={`w-full py-3 rounded font-semibold ${
                  orderSide === 'buy'
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {orderSide === 'buy' ? '매수' : '매도'} 주문
              </button>
            </div>
          </div>
        </div>
        
        {/* 주문 내역 */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4">주문 내역</h3>
            
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {orders.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-4">주문 내역이 없습니다</p>
              ) : (
                orders.map((order) => (
                  <div key={order.id} className="border border-gray-200 dark:border-gray-600 rounded p-3">
                    <div className="flex justify-between items-start mb-1">
                      <span className={`text-sm font-medium ${
                        order.side === 'buy' ? 'text-red-600' : 'text-blue-600'
                      }`}>
                        {order.side === 'buy' ? '매수' : '매도'}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        order.status === 'filled' 
                          ? 'bg-green-100 text-green-700' 
                          : order.status === 'cancelled'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-yellow-100 text-yellow-700'
                      }`}>
                        {order.status === 'filled' ? '체결' : order.status === 'cancelled' ? '취소' : '대기'}
                      </span>
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      <div>{order.type === 'market' ? '시장가' : '지정가'} / {order.quantity}주</div>
                      {order.price && <div>{order.price.toLocaleString()}원</div>}
                      <div>{order.time}</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function TradingSystemPage() {
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
              <Monitor className="w-8 h-8 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div className="text-left">
              <div className="text-sm text-gray-500 mb-1">Baby Chick • Chapter 3</div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                매매 시스템 실습
              </h1>
            </div>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            호가창을 읽고 이해하며, 다양한 주문 유형을 활용한 실전 매매의 기초를 익혀보세요.
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
              <span>호가창의 구조와 정보 해석 방법 이해</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>시장가 주문과 지정가 주문의 차이점과 활용법 학습</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>주식 체결 원리와 매매 매커니즘 이해</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold mt-1">•</span>
              <span>실전 시뮬레이션을 통한 주문 경험 축적</span>
            </li>
          </ul>
        </div>

        {/* Main Content */}
        <div className="space-y-12">
          {/* Section 1: 호가창 이해하기 */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              1️⃣ 호가창 읽기의 기초
            </h2>
            
            <div className="prose prose-lg dark:prose-invert max-w-none">
              <p>
                호가창(Order Book)은 특정 주식에 대한 매수/매도 주문들이 가격별로 정렬되어 표시되는 창입니다. 
                이는 주식 거래의 핵심이며, 시장의 수급 상황을 한눈에 파악할 수 있는 중요한 도구입니다.
              </p>

              <div className="grid md:grid-cols-2 gap-6 my-8">
                <div className="bg-red-50 dark:bg-red-900/10 p-6 rounded-lg">
                  <h3 className="text-lg font-bold text-red-700 dark:text-red-400 mb-3">
                    <TrendingUp className="inline w-5 h-5 mr-2" />
                    매도 호가 (Ask)
                  </h3>
                  <ul className="space-y-2 text-sm text-red-600 dark:text-red-300">
                    <li>• 주식을 <strong>팔고자 하는</strong> 주문들</li>
                    <li>• <strong>낮은 가격부터</strong> 위로 정렬</li>
                    <li>• 가격이 낮을수록 우선 체결</li>
                    <li>• 빨간색으로 표시됨</li>
                  </ul>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-lg">
                  <h3 className="text-lg font-bold text-blue-700 dark:text-blue-400 mb-3">
                    <Activity className="inline w-5 h-5 mr-2" />
                    매수 호가 (Bid)
                  </h3>
                  <ul className="space-y-2 text-sm text-blue-600 dark:text-blue-300">
                    <li>• 주식을 <strong>사고자 하는</strong> 주문들</li>
                    <li>• <strong>높은 가격부터</strong> 아래로 정렬</li>
                    <li>• 가격이 높을수록 우선 체결</li>
                    <li>• 파란색으로 표시됨</li>
                  </ul>
                </div>
              </div>

              <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-lg my-6">
                <h3 className="text-lg font-bold text-yellow-800 dark:text-yellow-300 mb-3">
                  <AlertCircle className="inline w-5 h-5 mr-2" />
                  호가창에서 읽어야 할 핵심 정보
                </h3>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 mb-2">매수/매도 균형</h4>
                    <p className="text-yellow-600 dark:text-yellow-300">
                      매수량이 매도량보다 많으면 상승 압력이 있고, 
                      매도량이 매수량보다 많으면 하락 압력이 있다는 신호입니다.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-yellow-700 dark:text-yellow-400 mb-2">호가 스프레드</h4>
                    <p className="text-yellow-600 dark:text-yellow-300">
                      1호가 매수가격과 1호가 매도가격의 차이를 말합니다. 
                      스프레드가 클수록 거래가 활발하지 않다는 의미입니다.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Section 2: 주문 유형과 체결 원리 */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              2️⃣ 주문 유형과 체결 원리
            </h2>
            
            <div className="grid lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-lg">
                  <h3 className="text-lg font-bold text-green-700 dark:text-green-400 mb-4">
                    <Zap className="inline w-5 h-5 mr-2" />
                    시장가 주문 (Market Order)
                  </h3>
                  
                  <div className="space-y-3 text-sm text-green-600 dark:text-green-300">
                    <div>
                      <strong className="text-green-700 dark:text-green-400">정의:</strong>
                      <p>가격을 지정하지 않고 현재 시장가격으로 즉시 거래하는 주문</p>
                    </div>
                    <div>
                      <strong className="text-green-700 dark:text-green-400">특징:</strong>
                      <ul className="list-disc list-inside mt-1 space-y-1">
                        <li>빠른 체결 보장</li>
                        <li>체결 가격 예측 불가</li>
                        <li>긴급한 거래 시 활용</li>
                      </ul>
                    </div>
                    <div>
                      <strong className="text-green-700 dark:text-green-400">주의점:</strong>
                      <p>시장 변동성이 클 때 예상보다 불리한 가격에 체결될 수 있습니다.</p>
                    </div>
                  </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/10 p-6 rounded-lg">
                  <h3 className="text-lg font-bold text-purple-700 dark:text-purple-400 mb-4">
                    <Target className="inline w-5 h-5 mr-2" />
                    지정가 주문 (Limit Order)
                  </h3>
                  
                  <div className="space-y-3 text-sm text-purple-600 dark:text-purple-300">
                    <div>
                      <strong className="text-purple-700 dark:text-purple-400">정의:</strong>
                      <p>원하는 가격을 직접 지정해서 주문하는 방식</p>
                    </div>
                    <div>
                      <strong className="text-purple-700 dark:text-purple-400">특징:</strong>
                      <ul className="list-disc list-inside mt-1 space-y-1">
                        <li>가격 통제 가능</li>
                        <li>체결 불확실성 존재</li>
                        <li>전략적 거래에 적합</li>
                      </ul>
                    </div>
                    <div>
                      <strong className="text-purple-700 dark:text-purple-400">활용법:</strong>
                      <p>목표 가격에서만 거래하고 싶을 때, 시장이 불안정할 때 사용합니다.</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg">
                <h3 className="text-lg font-bold text-gray-700 dark:text-gray-300 mb-4">
                  <ArrowUpDown className="inline w-5 h-5 mr-2" />
                  체결 우선순위 원칙
                </h3>
                
                <div className="space-y-4">
                  <div className="bg-white dark:bg-gray-700 p-4 rounded border-l-4 border-blue-500">
                    <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">1. 가격 우선</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      매수는 높은 가격부터, 매도는 낮은 가격부터 우선 체결
                    </p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-700 p-4 rounded border-l-4 border-green-500">
                    <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">2. 시간 우선</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      같은 가격이면 먼저 주문한 것부터 체결
                    </p>
                  </div>
                  
                  <div className="bg-white dark:bg-gray-700 p-4 rounded border-l-4 border-orange-500">
                    <h4 className="font-semibold text-orange-700 dark:text-orange-400 mb-2">3. 체결 조건</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      매수 호가 ≥ 매도 호가일 때만 체결 성사
                    </p>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-yellow-100 dark:bg-yellow-900/20 rounded">
                  <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">
                    체결 예시
                  </h4>
                  <p className="text-sm text-yellow-700 dark:text-yellow-300">
                    매도 1호가가 85,000원이고 매수 1호가가 84,950원이면 <strong>체결되지 않습니다</strong>.<br/>
                    누군가 85,000원에 매수 주문을 넣거나, 84,950원에 매도 주문을 넣어야 체결됩니다.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Section 3: 실전 시뮬레이터 */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              3️⃣ 실전 매매 시뮬레이션
            </h2>
            
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/10 dark:to-purple-900/10 p-6 rounded-xl mb-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">삼성전자 가상 거래</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">실제와 동일한 환경에서 안전하게 연습해보세요</p>
                </div>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="text-center">
                  <div className="font-semibold text-gray-500">현재가</div>
                  <div className="text-lg font-bold">85,000원</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-gray-500">전일대비</div>
                  <div className="text-lg font-bold text-red-600">+1,200 (+1.43%)</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-gray-500">거래량</div>
                  <div className="text-lg font-bold">12.3M</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-gray-500">시가총액</div>
                  <div className="text-lg font-bold">507조원</div>
                </div>
              </div>
            </div>

            <TradingSimulator />
          </section>

          {/* Section 4: 고급 주문 기법 */}
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              4️⃣ 고급 주문 기법과 전략
            </h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-orange-50 dark:bg-orange-900/10 p-6 rounded-lg">
                <h3 className="text-lg font-bold text-orange-700 dark:text-orange-400 mb-4">
                  <Layers className="inline w-5 h-5 mr-2" />
                  분할 매수/매도
                </h3>
                
                <p className="text-sm text-orange-600 dark:text-orange-300 mb-3">
                  큰 물량을 한 번에 거래하지 않고 여러 번에 나누어 처리하는 방법입니다.
                </p>
                
                <div className="space-y-2 text-sm">
                  <div>
                    <strong className="text-orange-700 dark:text-orange-400">장점:</strong>
                    <ul className="list-disc list-inside text-orange-600 dark:text-orange-300 mt-1">
                      <li>시장 충격 최소화</li>
                      <li>평균 단가 개선</li>
                      <li>위험 분산</li>
                    </ul>
                  </div>
                  <div>
                    <strong className="text-orange-700 dark:text-orange-400">예시:</strong>
                    <p className="text-orange-600 dark:text-orange-300">
                      1,000주를 사고 싶다면 → 250주씩 4번에 나누어 매수
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-teal-50 dark:bg-teal-900/10 p-6 rounded-lg">
                <h3 className="text-lg font-bold text-teal-700 dark:text-teal-400 mb-4">
                  <DollarSign className="inline w-5 h-5 mr-2" />
                  손절/익절 전략
                </h3>
                
                <p className="text-sm text-teal-600 dark:text-teal-300 mb-3">
                  미리 정한 기준에 따라 자동으로 매도하는 리스크 관리 방법입니다.
                </p>
                
                <div className="space-y-2 text-sm">
                  <div>
                    <strong className="text-teal-700 dark:text-teal-400">손절 (Stop Loss):</strong>
                    <p className="text-teal-600 dark:text-teal-300">
                      매수가 대비 -5~10% 하락 시 자동 매도
                    </p>
                  </div>
                  <div>
                    <strong className="text-teal-700 dark:text-teal-400">익절 (Take Profit):</strong>
                    <p className="text-teal-600 dark:text-teal-300">
                      매수가 대비 +10~20% 상승 시 자동 매도
                    </p>
                  </div>
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
              <span className="text-2xl">🎯</span>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                다음 단계로 진행
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                매매 시스템을 이해했다면 이제 투자 심리학을 배워볼 시간입니다.
              </p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                🧠 Chapter 4: 투자자 심리의 함정
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                손실회피, 확증편향, 군중심리 등 투자 실패의 주요 심리적 원인들을 학습하고 극복 방법을 익혀보세요.
              </p>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Clock className="w-4 h-4" />
                  <span>예상 학습시간: 40분</span>
                </div>
                <Link
                  href="/modules/stock-analysis/chapters/investor-psychology"
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
                  href="/modules/stock-analysis/stages/baby-chick"
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
              <span>3/9 완료</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full" style={{ width: '33%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}