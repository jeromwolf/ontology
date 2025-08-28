'use client';

import React, { useState } from 'react';

export default function ReturnCalculator() {
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