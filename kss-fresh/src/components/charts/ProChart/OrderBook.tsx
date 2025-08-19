'use client';

import { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { OrderBookEntry, MarketDepth } from './types';

interface OrderBookProps {
  symbol: string;
  lastPrice: number;
  priceChange: number;
  priceChangePercent: number;
  volume24h: number;
  onPriceClick?: (price: number, type: 'bid' | 'ask') => void;
  depth?: number;
  realtime?: boolean;
}

export default function OrderBook({
  symbol,
  lastPrice,
  priceChange,
  priceChangePercent,
  volume24h,
  onPriceClick,
  depth = 10,
  realtime = true
}: OrderBookProps) {
  const [marketDepth, setMarketDepth] = useState<MarketDepth>({
    bids: [],
    asks: [],
    spread: 0,
    midPrice: lastPrice
  });
  
  const [recentTrades, setRecentTrades] = useState<Array<{
    price: number;
    quantity: number;
    type: 'buy' | 'sell';
    time: Date;
  }>>([]);

  // 호가 데이터 시뮬레이션
  useEffect(() => {
    const generateOrderBook = () => {
      const bids: OrderBookEntry[] = [];
      const asks: OrderBookEntry[] = [];
      
      // 매수 호가
      let bidTotal = 0;
      for (let i = 0; i < depth; i++) {
        const price = lastPrice - (i + 1) * 10;
        const quantity = Math.floor(Math.random() * 10000) + 1000;
        bidTotal += quantity;
        bids.push({
          price,
          quantity,
          total: bidTotal,
          percentage: Math.random() * 100
        });
      }
      
      // 매도 호가
      let askTotal = 0;
      for (let i = 0; i < depth; i++) {
        const price = lastPrice + (depth - i) * 10;
        const quantity = Math.floor(Math.random() * 10000) + 1000;
        askTotal += quantity;
        asks.unshift({
          price,
          quantity,
          total: askTotal,
          percentage: Math.random() * 100
        });
      }
      
      setMarketDepth({
        bids,
        asks,
        spread: asks[asks.length - 1].price - bids[0].price,
        midPrice: (asks[asks.length - 1].price + bids[0].price) / 2
      });
    };

    generateOrderBook();
    
    if (realtime) {
      const interval = setInterval(generateOrderBook, 500);
      return () => clearInterval(interval);
    }
  }, [lastPrice, depth, realtime]);

  // 최근 체결 시뮬레이션
  useEffect(() => {
    if (!realtime) return;

    const interval = setInterval(() => {
      const newTrade = {
        price: lastPrice + (Math.random() - 0.5) * 20,
        quantity: Math.floor(Math.random() * 1000) + 100,
        type: Math.random() > 0.5 ? 'buy' as const : 'sell' as const,
        time: new Date()
      };
      
      setRecentTrades(prev => [newTrade, ...prev.slice(0, 19)]);
    }, Math.random() * 2000 + 500);

    return () => clearInterval(interval);
  }, [lastPrice, realtime]);

  return (
    <div className="h-full flex flex-col bg-gray-900/50">
      {/* 헤더 */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-semibold mb-2">호가창</h3>
        <div className="flex items-center gap-2 text-xs">
          <button className="flex-1 py-1 bg-gray-800 rounded hover:bg-gray-700">호가</button>
          <button className="flex-1 py-1 bg-gray-800 rounded hover:bg-gray-700">체결</button>
          <button className="flex-1 py-1 bg-gray-800 rounded hover:bg-gray-700">일별</button>
        </div>
      </div>
      
      {/* 호가 테이블 */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto">
          {/* 매도 호가 */}
          <div className="space-y-px">
            {marketDepth.asks.map((ask, i) => (
              <div
                key={`ask-${i}`}
                className="flex items-center bg-red-900/20 hover:bg-red-900/30 transition-colors cursor-pointer"
                onClick={() => onPriceClick?.(ask.price, 'ask')}
              >
                <div className="flex-1 px-2 py-1 text-right">
                  <div className="text-xs text-gray-400">{ask.quantity.toLocaleString()}</div>
                </div>
                <div className="px-3 py-1 text-red-400 font-medium text-sm">
                  ₩{ask.price.toLocaleString()}
                </div>
                <div className="w-20 px-2 py-1">
                  <div 
                    className="h-4 bg-red-500/30 rounded-sm transition-all duration-300"
                    style={{ width: `${ask.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          
          {/* 현재가 */}
          <div className="bg-gray-800 p-3 border-y border-gray-700 sticky top-0 z-10">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xl font-bold">
                  ₩{lastPrice.toLocaleString()}
                </div>
                <div className={`flex items-center gap-1 text-sm ${
                  priceChange >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {priceChange >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  <span>{Math.abs(priceChangePercent).toFixed(2)}%</span>
                </div>
              </div>
              <div className="text-right text-sm">
                <div className="text-gray-400">거래량</div>
                <div className="font-medium">{(volume24h / 1000000).toFixed(1)}M</div>
              </div>
            </div>
          </div>
          
          {/* 매수 호가 */}
          <div className="space-y-px">
            {marketDepth.bids.map((bid, i) => (
              <div
                key={`bid-${i}`}
                className="flex items-center bg-blue-900/20 hover:bg-blue-900/30 transition-colors cursor-pointer"
                onClick={() => onPriceClick?.(bid.price, 'bid')}
              >
                <div className="w-20 px-2 py-1">
                  <div 
                    className="h-4 bg-blue-500/30 rounded-sm ml-auto transition-all duration-300"
                    style={{ width: `${bid.percentage}%` }}
                  />
                </div>
                <div className="px-3 py-1 text-blue-400 font-medium text-sm">
                  ₩{bid.price.toLocaleString()}
                </div>
                <div className="flex-1 px-2 py-1">
                  <div className="text-xs text-gray-400">{bid.quantity.toLocaleString()}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* 주문 입력 */}
      <div className="p-4 border-t border-gray-700">
        <div className="grid grid-cols-2 gap-2 mb-3">
          <button className="py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors">
            매도
          </button>
          <button className="py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors">
            매수
          </button>
        </div>
        
        <div className="space-y-2">
          <div>
            <label className="text-xs text-gray-400">가격</label>
            <input
              type="text"
              value={lastPrice.toLocaleString()}
              className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-700 text-sm"
              readOnly
            />
          </div>
          <div>
            <label className="text-xs text-gray-400">수량</label>
            <input
              type="text"
              placeholder="0"
              className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-700 text-sm"
            />
          </div>
        </div>
      </div>
    </div>
  );
}