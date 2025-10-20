'use client';

import { useState } from 'react';
import { Gauge, Zap, CheckCircle, TrendingUp } from 'lucide-react';

export default function SQLPerformanceTuner() {
  const [query, setQuery] = useState(`SELECT * FROM orders
WHERE order_date = '2024-01-01'
AND user_id IN (SELECT id FROM users WHERE country = 'US')`);
  const [explainPlan, setExplainPlan] = useState<any[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);

  const analyzeQuery = () => {
    // Simulated EXPLAIN ANALYZE
    setExplainPlan([
      { step: 'Seq Scan on orders', cost: '0.00..50000.00', rows: '10000', time: '250ms' },
      { step: '  Filter: order_date = 2024-01-01', cost: '', rows: '1000', time: '' },
      { step: '  SubPlan 1', cost: '', rows: '', time: '' },
      { step: '    Seq Scan on users', cost: '0.00..25000.00', rows: '5000', time: '120ms' },
    ]);

    setSuggestions([
      '✅ order_date에 B-Tree 인덱스 생성 추천',
      '✅ 서브쿼리를 JOIN으로 변경하여 성능 향상',
      '✅ SELECT * 대신 필요한 컬럼만 선택',
      '✅ country 컬럼에 인덱스 생성 고려',
    ]);
  };

  const optimizedQuery = `-- 최적화된 쿼리
SELECT o.order_id, o.total_amount, o.order_date
FROM orders o
INNER JOIN users u ON o.user_id = u.id
WHERE o.order_date = '2024-01-01'
  AND u.country = 'US';

-- 필요한 인덱스:
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_users_country ON users(country);
CREATE INDEX idx_orders_user_id ON orders(user_id);`;

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Gauge size={32} />
          <h2 className="text-2xl font-bold">SQL 성능 튜닝 도구</h2>
        </div>
        <p className="text-indigo-100">쿼리 실행 계획 분석 및 최적화 제안</p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">📝 SQL 쿼리 입력</h3>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full h-32 p-4 font-mono text-sm bg-gray-900 text-gray-100 rounded-lg"
        />
        <button
          onClick={analyzeQuery}
          className="mt-4 px-6 py-3 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg font-semibold"
        >
          <Zap className="inline mr-2" size={18} />
          EXPLAIN ANALYZE 실행
        </button>
      </div>

      {explainPlan.length > 0 && (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-lg font-bold mb-4">📊 실행 계획</h3>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm space-y-1">
              {explainPlan.map((step, idx) => (
                <div key={idx}>
                  {step.step} {step.cost && `(cost=${step.cost})`} {step.time && `[${step.time}]`}
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-lg font-bold mb-4 text-yellow-600">
              <TrendingUp className="inline mr-2" />
              최적화 제안
            </h3>
            <ul className="space-y-2">
              {suggestions.map((suggestion, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={18} />
                  <span>{suggestion}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-lg font-bold mb-4 text-green-600">
              <Zap className="inline mr-2" />
              최적화된 쿼리
            </h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
              {optimizedQuery}
            </pre>
          </div>
        </>
      )}

      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h3 className="text-xl font-bold mb-4">💡 SQL 튜닝 체크리스트</h3>
        <ul className="space-y-2 text-sm">
          <li><strong>SELECT *</strong> 사용하지 않기 → 필요한 컬럼만 선택</li>
          <li><strong>WHERE 절 인덱스</strong> 활용 → 자주 검색하는 컬럼에 인덱스</li>
          <li><strong>서브쿼리 최소화</strong> → JOIN으로 대체</li>
          <li><strong>함수 사용 주의</strong> → WHERE에 함수 사용 시 인덱스 무효화</li>
          <li><strong>파티션 프루닝</strong> → 파티션 키 필터 사용</li>
        </ul>
      </div>
    </div>
  );
}
