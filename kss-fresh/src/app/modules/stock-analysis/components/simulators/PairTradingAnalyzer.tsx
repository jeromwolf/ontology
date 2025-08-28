'use client';

import React, { useState, useMemo } from 'react';
import { TrendingUp, TrendingDown, BarChart3, AlertCircle } from 'lucide-react';

interface StockData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

export default function PairTradingAnalyzer() {
  const [selectedStock1, setSelectedStock1] = useState<string>('AAPL');
  const [selectedStock2, setSelectedStock2] = useState<string>('MSFT');

  // 샘플 주식 데이터
  const stocks: StockData[] = [
    { symbol: 'AAPL', name: 'Apple Inc.', price: 175.43, change: 2.15, changePercent: 1.24 },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 378.85, change: -1.34, changePercent: -0.35 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 138.21, change: 0.87, changePercent: 0.63 },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 145.86, change: 3.21, changePercent: 2.25 },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 248.42, change: -5.67, changePercent: -2.23 },
  ];

  // 페어 트레이딩 분석 결과
  const analysisResult = useMemo(() => {
    const stock1 = stocks.find(s => s.symbol === selectedStock1);
    const stock2 = stocks.find(s => s.symbol === selectedStock2);
    
    if (!stock1 || !stock2) return null;

    const spread = stock1.changePercent - stock2.changePercent;
    const correlation = 0.75; // 샘플 상관계수
    const zscore = spread / 2.5; // 샘플 Z-score

    let signal = 'neutral';
    if (zscore > 2) signal = 'short';
    if (zscore < -2) signal = 'long';

    return {
      stock1,
      stock2,
      spread,
      correlation,
      zscore,
      signal
    };
  }, [selectedStock1, selectedStock2]);

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'long': return 'text-green-600 dark:text-green-400';
      case 'short': return 'text-red-600 dark:text-red-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getSignalText = (signal: string) => {
    switch (signal) {
      case 'long': return '매수 신호 (스프레드 확대 예상)';
      case 'short': return '매도 신호 (스프레드 축소 예상)';
      default: return '중립 (진입 신호 없음)';
    }
  };

  return (
    <div className="space-y-6">
      {/* 주식 쌍 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">주식 쌍 선택</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">첫 번째 주식</label>
            <select
              value={selectedStock1}
              onChange={(e) => setSelectedStock1(e.target.value)}
              className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
              {stocks.map(stock => (
                <option key={stock.symbol} value={stock.symbol}>
                  {stock.symbol} - {stock.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">두 번째 주식</label>
            <select
              value={selectedStock2}
              onChange={(e) => setSelectedStock2(e.target.value)}
              className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
              {stocks.map(stock => (
                <option key={stock.symbol} value={stock.symbol}>
                  {stock.symbol} - {stock.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* 분석 결과 */}
      {analysisResult && (
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 size={20} className="text-blue-600 dark:text-blue-400" />
              <h4 className="font-medium">스프레드</h4>
            </div>
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {analysisResult.spread.toFixed(2)}%
            </p>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp size={20} className="text-emerald-600 dark:text-emerald-400" />
              <h4 className="font-medium">상관계수</h4>
            </div>
            <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
              {analysisResult.correlation.toFixed(3)}
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 size={20} className="text-purple-600 dark:text-purple-400" />
              <h4 className="font-medium">Z-Score</h4>
            </div>
            <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {analysisResult.zscore.toFixed(2)}
            </p>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle size={20} className="text-orange-600 dark:text-orange-400" />
              <h4 className="font-medium">신호</h4>
            </div>
            <p className={`text-sm font-medium ${getSignalColor(analysisResult.signal)}`}>
              {getSignalText(analysisResult.signal)}
            </p>
          </div>
        </div>
      )}

      {/* 상세 분석 */}
      {analysisResult && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">페어 트레이딩 전략 분석</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3">현재 포지션 정보</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>{analysisResult.stock1.symbol}:</span>
                  <span className={analysisResult.stock1.change >= 0 ? 'text-green-600' : 'text-red-600'}>
                    ${analysisResult.stock1.price} ({analysisResult.stock1.changePercent.toFixed(2)}%)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>{analysisResult.stock2.symbol}:</span>
                  <span className={analysisResult.stock2.change >= 0 ? 'text-green-600' : 'text-red-600'}>
                    ${analysisResult.stock2.price} ({analysisResult.stock2.changePercent.toFixed(2)}%)
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-3">거래 권장사항</h4>
              <div className="text-sm space-y-1">
                {analysisResult.signal === 'long' && (
                  <div className="text-green-600 dark:text-green-400">
                    • {analysisResult.stock1.symbol} 매수, {analysisResult.stock2.symbol} 매도
                  </div>
                )}
                {analysisResult.signal === 'short' && (
                  <div className="text-red-600 dark:text-red-400">
                    • {analysisResult.stock1.symbol} 매도, {analysisResult.stock2.symbol} 매수
                  </div>
                )}
                {analysisResult.signal === 'neutral' && (
                  <div className="text-gray-600 dark:text-gray-400">
                    • 진입 대기 (스프레드가 정상 범위)
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}