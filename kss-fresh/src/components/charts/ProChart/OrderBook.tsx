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
  market?: 'KR' | 'US';
}

export default function OrderBook({
  symbol,
  lastPrice,
  priceChange,
  priceChangePercent,
  volume24h,
  onPriceClick,
  depth = 10,
  realtime = true,
  market = 'KR'
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
      
      // 가격 단위 설정 (미국 주식은 0.01달러, 한국 주식은 10원)
      const priceStep = market === 'US' ? 0.01 : 10;
      const quantityRange = market === 'US' ? [10, 1000] : [1000, 10000];
      
      // 매수 호가
      let bidTotal = 0;
      for (let i = 0; i < depth; i++) {
        const price = lastPrice - (i + 1) * priceStep;
        const quantity = Math.floor(Math.random() * (quantityRange[1] - quantityRange[0])) + quantityRange[0];
        bidTotal += quantity;
        bids.push({
          price: Math.round(price * 100) / 100,
          quantity,
          total: bidTotal,
          percentage: Math.random() * 100
        });
      }
      
      // 매도 호가
      let askTotal = 0;
      for (let i = 0; i < depth; i++) {
        const price = lastPrice + (depth - i) * priceStep;
        const quantity = Math.floor(Math.random() * (quantityRange[1] - quantityRange[0])) + quantityRange[0];
        askTotal += quantity;
        asks.unshift({
          price: Math.round(price * 100) / 100,
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
  }, [lastPrice, depth, realtime, market]);

  // 최근 체결 시뮬레이션
  useEffect(() => {
    if (!realtime) return;

    const priceVariation = market === 'US' ? 0.5 : 20;
    const quantityRange = market === 'US' ? [10, 500] : [100, 1000];

    const interval = setInterval(() => {
      const newTrade = {
        price: lastPrice + (Math.random() - 0.5) * priceVariation,
        quantity: Math.floor(Math.random() * (quantityRange[1] - quantityRange[0])) + quantityRange[0],
        type: Math.random() > 0.5 ? 'buy' as const : 'sell' as const,
        time: new Date()
      };
      
      setRecentTrades(prev => [newTrade, ...prev.slice(0, 19)]);
    }, Math.random() * 2000 + 500);

    return () => clearInterval(interval);
  }, [lastPrice, realtime, market]);

  return (
    <div className="h-full flex flex-col bg-gray-900/50">
      {/* 헤더 */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-semibold mb-2">
          {market === 'US' ? '주식 정보' : '호가 시뮬레이션 (데모)'}
        </h3>
        <div className="text-xs text-gray-400">
          {market === 'US' ? 'API 제공 데이터' : '실제 호가가 아닌 시뮬레이션입니다'}
        </div>
      </div>
      
      {/* 콘텐츠 영역 */}
      <div className="flex-1 overflow-hidden">
        {market === 'US' ? (
          // 미국 주식 정보
          <div className="p-4 space-y-4">
            {/* 주요 지표 */}
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-gray-400 text-xs">52주 최고</div>
                <div className="font-medium">-</div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-gray-400 text-xs">52주 최저</div>
                <div className="font-medium">-</div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-gray-400 text-xs">시가총액</div>
                <div className="font-medium">-</div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-gray-400 text-xs">PER</div>
                <div className="font-medium">-</div>
              </div>
            </div>
            
            {/* 차트 업데이트 정보 */}
            <div className="bg-blue-900/20 border border-blue-500/30 rounded p-3">
              <div className="text-xs text-blue-300">
                <div className="font-medium mb-1">데이터 업데이트</div>
                <div>• 차트: 실시간 (2-3초 간격)</div>
                <div>• 가격: API 제공 데이터</div>
                <div>• 지연: 15분 (무료 플랜)</div>
              </div>
            </div>
          </div>
        ) : (
          // 한국 주식 호가창
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
                  {market === 'US' ? `$${lastPrice.toFixed(2)}` : `₩${lastPrice.toLocaleString()}`}
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
                <div className="font-medium">{
                  market === 'US' 
                    ? volume24h.toLocaleString() 
                    : `${(volume24h / 1000000).toFixed(1)}M`
                }</div>
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
                  {market === 'US' ? `$${bid.price.toFixed(2)}` : `₩${bid.price.toLocaleString()}`}
                </div>
                <div className="flex-1 px-2 py-1">
                  <div className="text-xs text-gray-400">{bid.quantity.toLocaleString()}</div>
                </div>
              </div>
            ))}
          </div>
          </div>
        )}
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