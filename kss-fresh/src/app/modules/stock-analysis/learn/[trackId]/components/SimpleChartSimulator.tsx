'use client';

import React, { useState, useEffect } from 'react';

export default function SimpleChartSimulator() {
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
        <div className={`text-2xl font-bold ${isUp ? 'text-red-500' : 'text-blue-500'}`}>
          ₩{currentPrice.toLocaleString()}
        </div>
        <div className="text-sm text-gray-600 dark:text-gray-400">
          {isUp ? '상승' : '하락'} 중
        </div>
      </div>
      <div className="h-32 bg-white dark:bg-gray-700 rounded relative overflow-hidden">
        <div className="absolute inset-0 flex items-end justify-between px-2 pb-2">
          {priceHistory.slice(-10).map((price, i) => {
            const height = ((price - 45000) / (55000 - 45000)) * 100;
            return (
              <div
                key={i}
                className="w-6 bg-blue-500 rounded-t"
                style={{ height: `${height}%` }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}