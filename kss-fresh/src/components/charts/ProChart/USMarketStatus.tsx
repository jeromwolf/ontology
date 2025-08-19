'use client';

import { useState, useEffect } from 'react';
import { Clock, TrendingUp, TrendingDown } from 'lucide-react';
import { usStockService } from '@/lib/services/us-stock-service';

export default function USMarketStatus() {
  const [marketSession, setMarketSession] = useState<string>('');
  const [nextUpdate, setNextUpdate] = useState<Date>(new Date());

  useEffect(() => {
    const updateMarketStatus = () => {
      const session = usStockService.getMarketSession();
      setMarketSession(session);
      setNextUpdate(new Date(Date.now() + 60000)); // 1분 후
    };

    updateMarketStatus();
    const interval = setInterval(updateMarketStatus, 60000); // 1분마다 업데이트

    return () => clearInterval(interval);
  }, []);

  const getSessionDisplay = () => {
    switch (marketSession) {
      case 'pre-market':
        return { text: '프리마켓', color: 'text-yellow-400', bgColor: 'bg-yellow-900/20', borderColor: 'border-yellow-500/30' };
      case 'regular':
        return { text: '정규장', color: 'text-green-400', bgColor: 'bg-green-900/20', borderColor: 'border-green-500/30' };
      case 'after-hours':
        return { text: '애프터마켓', color: 'text-orange-400', bgColor: 'bg-orange-900/20', borderColor: 'border-orange-500/30' };
      default:
        return { text: '장마감', color: 'text-red-400', bgColor: 'bg-red-900/20', borderColor: 'border-red-500/30' };
    }
  };

  const { text, color, bgColor, borderColor } = getSessionDisplay();

  return (
    <div className={`${bgColor} border ${borderColor} rounded-lg p-4 space-y-2`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className={`w-4 h-4 ${color}`} />
          <h3 className="text-sm font-semibold">미국 시장</h3>
        </div>
        <span className={`text-xs font-medium ${color}`}>{text}</span>
      </div>

      <div className="text-xs text-gray-400 space-y-1">
        <div>프리마켓: 04:00 - 09:30 ET</div>
        <div>정규장: 09:30 - 16:00 ET</div>
        <div>애프터마켓: 16:00 - 20:00 ET</div>
      </div>

      <div className="text-xs text-gray-500 pt-2 border-t border-gray-700">
        다음 업데이트: {nextUpdate.toLocaleTimeString()}
      </div>
    </div>
  );
}