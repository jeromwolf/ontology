'use client';

import { useState } from 'react';
import { GitBranch, Database, ArrowRight } from 'lucide-react';

interface Node {
  id: string;
  name: string;
  type: 'source' | 'transform' | 'destination';
  level: number;
}

export default function DataLineageExplorer() {
  const [selectedTable, setSelectedTable] = useState('gold.daily_sales');

  const lineageGraph: Record<string, Node[]> = {
    'gold.daily_sales': [
      { id: 's1', name: 'postgres.orders', type: 'source', level: 0 },
      { id: 't1', name: 'bronze.raw_orders', type: 'transform', level: 1 },
      { id: 't2', name: 'silver.cleaned_orders', type: 'transform', level: 2 },
      { id: 'd1', name: 'gold.daily_sales', type: 'destination', level: 3 },
    ],
  };

  const nodes = lineageGraph[selectedTable] || [];

  const getNodeColor = (type: string) => {
    return type === 'source' ? 'bg-blue-100 border-blue-500' :
           type === 'transform' ? 'bg-green-100 border-green-500' :
           'bg-purple-100 border-purple-500';
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <GitBranch size={32} />
          <h2 className="text-2xl font-bold">데이터 계보 탐색기</h2>
        </div>
        <p className="text-purple-100">데이터의 흐름과 변환 과정을 시각화</p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">🎯 테이블 선택</h3>
        <select
          value={selectedTable}
          onChange={(e) => setSelectedTable(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
        >
          <option value="gold.daily_sales">gold.daily_sales</option>
        </select>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-6">📊 데이터 계보 그래프</h3>
        <div className="flex items-center gap-6 overflow-x-auto pb-4">
          {nodes.map((node, idx) => (
            <div key={node.id} className="flex items-center gap-6">
              <div className={`p-6 rounded-lg border-2 ${getNodeColor(node.type)} min-w-[200px]`}>
                <Database className="mb-2" size={24} />
                <div className="font-bold">{node.name}</div>
                <div className="text-xs text-gray-600 mt-1">{node.type}</div>
              </div>
              {idx < nodes.length - 1 && (
                <ArrowRight className="text-gray-400 flex-shrink-0" size={32} />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">💻 dbt 계보 추적 예제</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- models/gold/daily_sales.sql
{{
  config(
    materialized='table',
    tags=['gold', 'sales']
  )
}}

with cleaned_orders as (
  select * from {{ ref('silver__cleaned_orders') }}
),

aggregated as (
  select
    order_date,
    sum(total_amount) as total_sales,
    count(distinct order_id) as order_count
  from cleaned_orders
  group by order_date
)

select * from aggregated

-- dbt가 자동으로 계보 추적:
-- postgres.orders → bronze.raw_orders → silver.cleaned_orders → gold.daily_sales`}
        </pre>
      </div>
    </div>
  );
}
