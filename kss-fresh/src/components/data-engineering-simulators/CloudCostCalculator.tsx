'use client';

import { useState } from 'react';
import { DollarSign, TrendingDown, Calculator } from 'lucide-react';

export default function CloudCostCalculator() {
  const [platform, setPlatform] = useState<'snowflake' | 'bigquery' | 'redshift'>('snowflake');
  const [dataSize, setDataSize] = useState(100); // TB
  const [queries, setQueries] = useState(1000); // per day
  const [computeHours, setComputeHours] = useState(24);

  const calculateCost = () => {
    const costs = {
      snowflake: {
        storage: dataSize * 23, // $23/TB/month
        compute: computeHours * 30 * 2, // $2/credit/hour
        total: 0,
      },
      bigquery: {
        storage: dataSize * 20, // $20/TB/month (active)
        compute: (queries * 0.001 * 5), // $5/TB scanned
        total: 0,
      },
      redshift: {
        storage: dataSize * 24, // $24/TB (RA3 노드)
        compute: computeHours * 30 * 4.8, // ra3.4xlarge $4.8/hour
        total: 0,
      },
    };

    costs.snowflake.total = costs.snowflake.storage + costs.snowflake.compute;
    costs.bigquery.total = costs.bigquery.storage + costs.bigquery.compute;
    costs.redshift.total = costs.redshift.storage + costs.redshift.compute;

    return costs[platform];
  };

  const cost = calculateCost();

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <DollarSign size={32} />
          <h2 className="text-2xl font-bold">클라우드 DW 비용 계산기</h2>
        </div>
        <p className="text-emerald-100">Snowflake, BigQuery, Redshift 비용 추정</p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">☁️ 플랫폼 선택</h3>
        <div className="grid grid-cols-3 gap-4">
          {['snowflake', 'bigquery', 'redshift'].map((p) => (
            <button
              key={p}
              onClick={() => setPlatform(p as any)}
              className={`p-4 rounded-lg border-2 transition-all ${
                platform === p ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20' : 'border-gray-200 dark:border-gray-700'
              }`}
            >
              {p === 'snowflake' && '❄️ Snowflake'}
              {p === 'bigquery' && '🔷 BigQuery'}
              {p === 'redshift' && '🔴 Redshift'}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">⚙️ 사용량 입력</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-semibold mb-2">데이터 크기 (TB): {dataSize}</label>
            <input
              type="range"
              min="10"
              max="1000"
              value={dataSize}
              onChange={(e) => setDataSize(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-semibold mb-2">일일 쿼리 수: {queries}</label>
            <input
              type="range"
              min="100"
              max="10000"
              value={queries}
              onChange={(e) => setQueries(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-semibold mb-2">컴퓨팅 시간 (시간/일): {computeHours}</label>
            <input
              type="range"
              min="1"
              max="24"
              value={computeHours}
              onChange={(e) => setComputeHours(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Calculator className="text-emerald-600" />
          예상 월간 비용
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <div className="text-sm text-gray-600 dark:text-gray-400">스토리지</div>
            <div className="text-2xl font-bold text-blue-600">${cost.storage.toFixed(2)}</div>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <div className="text-sm text-gray-600 dark:text-gray-400">컴퓨팅</div>
            <div className="text-2xl font-bold text-purple-600">${cost.compute.toFixed(2)}</div>
          </div>
          <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg">
            <div className="text-sm text-gray-600 dark:text-gray-400">총 비용</div>
            <div className="text-3xl font-bold text-emerald-600">${cost.total.toFixed(2)}</div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-6 rounded-xl">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <TrendingDown className="text-orange-600" />
          비용 절감 팁
        </h3>
        <ul className="space-y-2 text-sm">
          <li>• 오래된 파티션 삭제 (스토리지 비용 절감)</li>
          <li>• 쿼리 결과 캐싱 활용 (컴퓨팅 비용 절감)</li>
          <li>• Parquet/ORC 컬럼나 포맷 사용 (압축률 3-5배)</li>
          <li>• 예약 슬롯/커밋 사용 (대규모 워크로드 20-30% 할인)</li>
          <li>• 불필요한 웨어하우스 자동 중지 설정</li>
        </ul>
      </div>
    </div>
  );
}
