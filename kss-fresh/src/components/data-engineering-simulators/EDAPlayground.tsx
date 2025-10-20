'use client';

import { useState, useEffect } from 'react';
import {
  BarChart3, PieChart, TrendingUp, Database,
  Download, Upload, RefreshCw, Search,
  CheckCircle, AlertCircle, Info
} from 'lucide-react';

interface DataStats {
  rows: number;
  columns: number;
  missing: number;
  duplicates: number;
  memory: string;
}

interface ColumnInfo {
  name: string;
  type: string;
  missing: number;
  unique: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
}

export default function EDAPlayground() {
  const [selectedDataset, setSelectedDataset] = useState('sales');
  const [stats, setStats] = useState<DataStats | null>(null);
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [correlationMatrix, setCorrelationMatrix] = useState<number[][]>([]);
  const [selectedChart, setSelectedChart] = useState<'histogram' | 'scatter' | 'box' | 'correlation'>('histogram');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const datasets = {
    sales: {
      name: '이커머스 판매 데이터',
      rows: 50000,
      columns: 12,
      description: '온라인 쇼핑몰 거래 기록',
    },
    customers: {
      name: '고객 행동 데이터',
      rows: 25000,
      columns: 15,
      description: '고객 인구통계 및 구매 패턴',
    },
    products: {
      name: '제품 카탈로그',
      rows: 5000,
      columns: 8,
      description: '제품 정보 및 재고 데이터',
    },
  };

  useEffect(() => {
    analyzeDataset();
  }, [selectedDataset]);

  const analyzeDataset = () => {
    setIsAnalyzing(true);

    setTimeout(() => {
      // Simulated stats
      const dataset = datasets[selectedDataset as keyof typeof datasets];
      setStats({
        rows: dataset.rows,
        columns: dataset.columns,
        missing: Math.floor(dataset.rows * 0.03),
        duplicates: Math.floor(dataset.rows * 0.01),
        memory: `${(dataset.rows * dataset.columns * 8 / 1024 / 1024).toFixed(2)} MB`,
      });

      // Simulated column info
      const mockColumns: ColumnInfo[] = selectedDataset === 'sales' ? [
        { name: 'order_id', type: 'int64', missing: 0, unique: dataset.rows },
        { name: 'customer_id', type: 'int64', missing: 150, unique: 18000, mean: 12450, std: 7200 },
        { name: 'product_id', type: 'int64', missing: 0, unique: 4500, mean: 2250, std: 1300 },
        { name: 'quantity', type: 'int64', missing: 80, unique: 20, mean: 2.5, std: 1.8, min: 1, max: 20 },
        { name: 'price', type: 'float64', missing: 0, unique: 3200, mean: 125.5, std: 98.3, min: 9.99, max: 999.99 },
        { name: 'discount', type: 'float64', missing: 200, unique: 11, mean: 0.15, std: 0.12, min: 0, max: 0.5 },
        { name: 'total_amount', type: 'float64', missing: 0, unique: 45000, mean: 280.3, std: 220.5 },
        { name: 'payment_method', type: 'object', missing: 10, unique: 4 },
        { name: 'shipping_country', type: 'object', missing: 50, unique: 35 },
        { name: 'order_date', type: 'datetime64', missing: 0, unique: 365 },
        { name: 'delivery_date', type: 'datetime64', missing: 1200, unique: 380 },
        { name: 'customer_satisfaction', type: 'int64', missing: 5000, unique: 5, mean: 4.2, std: 0.9, min: 1, max: 5 },
      ] : [];

      setColumns(mockColumns);

      // Simulated correlation matrix (only numeric columns)
      const numericCols = mockColumns.filter(col => col.type.includes('int') || col.type.includes('float'));
      const size = Math.min(numericCols.length, 6);
      const matrix = Array(size).fill(0).map(() =>
        Array(size).fill(0).map(() => Math.random() * 2 - 1)
      );
      // Make diagonal 1
      for (let i = 0; i < size; i++) {
        matrix[i][i] = 1;
        for (let j = i + 1; j < size; j++) {
          matrix[j][i] = matrix[i][j]; // Symmetric
        }
      }
      setCorrelationMatrix(matrix);
      setIsAnalyzing(false);
    }, 800);
  };

  const getCorrelationColor = (value: number) => {
    const absValue = Math.abs(value);
    if (absValue > 0.7) return value > 0 ? 'bg-green-600' : 'bg-red-600';
    if (absValue > 0.4) return value > 0 ? 'bg-green-400' : 'bg-red-400';
    if (absValue > 0.2) return value > 0 ? 'bg-green-200' : 'bg-red-200';
    return 'bg-gray-100 dark:bg-gray-700';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <BarChart3 size={32} />
          <h2 className="text-2xl font-bold">탐색적 데이터 분석 (EDA) Playground</h2>
        </div>
        <p className="text-blue-100">
          pandas, matplotlib 스타일의 인터랙티브 데이터 탐색 도구
        </p>
      </div>

      {/* Dataset Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Database className="text-blue-600" />
          데이터셋 선택
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          {Object.entries(datasets).map(([key, dataset]) => (
            <button
              key={key}
              onClick={() => setSelectedDataset(key)}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                selectedDataset === key
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
              }`}
            >
              <h4 className="font-bold mb-1">{dataset.name}</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                {dataset.description}
              </p>
              <div className="flex gap-3 text-xs text-gray-500">
                <span>{dataset.rows.toLocaleString()} rows</span>
                <span>{dataset.columns} cols</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Dataset Stats */}
      {stats && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Info className="text-green-600" />
            데이터셋 요약 통계
          </h3>
          <div className="grid md:grid-cols-5 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{stats.rows.toLocaleString()}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">총 레코드 수</div>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{stats.columns}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">컬럼 수</div>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <div className="text-2xl font-bold text-yellow-600">{stats.missing.toLocaleString()}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">결측치 (총)</div>
            </div>
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
              <div className="text-2xl font-bold text-red-600">{stats.duplicates.toLocaleString()}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">중복 레코드</div>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{stats.memory}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">메모리 사용량</div>
            </div>
          </div>
        </div>
      )}

      {/* Column Details */}
      {columns.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Search className="text-purple-600" />
            컬럼별 상세 정보
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="p-3 text-left">컬럼명</th>
                  <th className="p-3 text-left">타입</th>
                  <th className="p-3 text-right">결측치</th>
                  <th className="p-3 text-right">고유값</th>
                  <th className="p-3 text-right">평균</th>
                  <th className="p-3 text-right">표준편차</th>
                  <th className="p-3 text-right">최소값</th>
                  <th className="p-3 text-right">최대값</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {columns.map((col, idx) => (
                  <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="p-3 font-mono text-blue-600">{col.name}</td>
                    <td className="p-3">
                      <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                        {col.type}
                      </span>
                    </td>
                    <td className="p-3 text-right">
                      {col.missing > 0 ? (
                        <span className="text-yellow-600">{col.missing}</span>
                      ) : (
                        <CheckCircle className="inline text-green-500" size={16} />
                      )}
                    </td>
                    <td className="p-3 text-right">{col.unique.toLocaleString()}</td>
                    <td className="p-3 text-right">{col.mean?.toFixed(2) || '-'}</td>
                    <td className="p-3 text-right">{col.std?.toFixed(2) || '-'}</td>
                    <td className="p-3 text-right">{col.min?.toFixed(2) || '-'}</td>
                    <td className="p-3 text-right">{col.max?.toFixed(2) || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Correlation Matrix */}
      {correlationMatrix.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="text-indigo-600" />
            상관관계 히트맵
          </h3>
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              <table className="border-collapse">
                <thead>
                  <tr>
                    <th className="p-2"></th>
                    {correlationMatrix.map((_, idx) => (
                      <th key={idx} className="p-2 text-xs font-mono rotate-45 origin-bottom-left">
                        {columns[idx]?.name.slice(0, 10)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {correlationMatrix.map((row, i) => (
                    <tr key={i}>
                      <td className="p-2 text-xs font-mono font-bold">
                        {columns[i]?.name.slice(0, 10)}
                      </td>
                      {row.map((value, j) => (
                        <td key={j} className="p-0">
                          <div
                            className={`w-12 h-12 flex items-center justify-center text-xs font-bold ${getCorrelationColor(value)}`}
                            title={`${columns[i]?.name} vs ${columns[j]?.name}: ${value.toFixed(2)}`}
                          >
                            {value.toFixed(2)}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="mt-4 flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-600 rounded"></div>
              <span>양의 상관관계</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-600 rounded"></div>
              <span>음의 상관관계</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
              <span>약한 상관관계</span>
            </div>
          </div>
        </div>
      )}

      {/* Python Code Example */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">📝 Python 코드 예제</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('${datasets[selectedDataset as keyof typeof datasets].name}.csv')

# 기본 정보 확인
print(df.info())
print(df.describe())

# 결측치 확인
print(df.isnull().sum())

# 중복 레코드 확인
print(f"중복 레코드: {df.duplicated().sum()}")

# 상관관계 히트맵
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 컬럼별 분포 확인
df.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()`}
        </pre>
      </div>
    </div>
  );
}
