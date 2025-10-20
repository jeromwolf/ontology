'use client';

import { useState } from 'react';
import { Zap, TrendingDown, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

interface OptimizationIssue {
  type: 'critical' | 'warning' | 'info';
  title: string;
  description: string;
  fix: string;
}

export default function SparkOptimizer() {
  const [code, setCode] = useState(`df = spark.read.parquet("s3://bucket/data")
df.filter(col("date") == "2024-01-01").groupBy("user_id").count().show()`);
  const [issues, setIssues] = useState<OptimizationIssue[]>([]);
  const [optimizedCode, setOptimizedCode] = useState('');

  const analyzeCode = () => {
    const foundIssues: OptimizationIssue[] = [];

    if (code.includes('.show()') && !code.includes('.cache()')) {
      foundIssues.push({
        type: 'warning',
        title: '중복 연산 가능성',
        description: 'show() 호출 시 전체 DAG가 재실행됩니다',
        fix: '반복 사용 시 .cache() 또는 .persist() 추가',
      });
    }

    if (code.includes('==') && !code.includes('===')) {
      foundIssues.push({
        type: 'critical',
        title: '비효율적 필터 연산자',
        description: 'Python == 대신 Spark === 사용 권장',
        fix: 'col("date") === "2024-01-01"',
      });
    }

    if (!code.includes('.repartition') && !code.includes('.coalesce')) {
      foundIssues.push({
        type: 'info',
        title: '파티션 최적화 미적용',
        description: '파티션 수 조정으로 성능 향상 가능',
        fix: '.repartition(200) 또는 .coalesce(10) 추가',
      });
    }

    if (code.includes('.count()') && code.includes('.filter(')) {
      foundIssues.push({
        type: 'warning',
        title: '필터 후 count',
        description: 'count() 전에 필터링하면 스캔 데이터 감소',
        fix: '필터 조건을 최대한 앞으로 이동 (Predicate Pushdown)',
      });
    }

    setIssues(foundIssues);

    const optimized = code
      .replace('==', '===')
      .replace('.show()', '.cache().show()')
      .replace('.count()', '.repartition(200).count()');
    setOptimizedCode(optimized);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-yellow-500 to-orange-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Zap size={32} />
          <h2 className="text-2xl font-bold">Spark 쿼리 최적화 도구</h2>
        </div>
        <p className="text-yellow-100">PySpark 코드를 분석하고 최적화 제안을 받으세요</p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">📝 PySpark 코드 입력</h3>
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className="w-full h-48 p-4 font-mono text-sm bg-gray-900 text-gray-100 rounded-lg"
          placeholder="PySpark 코드를 입력하세요..."
        />
        <button
          onClick={analyzeCode}
          className="mt-4 px-6 py-3 bg-orange-500 hover:bg-orange-600 text-white rounded-lg font-semibold"
        >
          <Zap className="inline mr-2" size={18} />
          코드 분석
        </button>
      </div>

      {issues.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4">⚠️ 발견된 이슈</h3>
          <div className="space-y-3">
            {issues.map((issue, idx) => (
              <div key={idx} className={`p-4 rounded-lg border-l-4 ${
                issue.type === 'critical' ? 'bg-red-50 dark:bg-red-900/20 border-red-500' :
                issue.type === 'warning' ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500' :
                'bg-blue-50 dark:bg-blue-900/20 border-blue-500'
              }`}>
                <div className="flex items-start gap-3">
                  {issue.type === 'critical' ? <XCircle className="text-red-500 mt-1" /> :
                   issue.type === 'warning' ? <AlertTriangle className="text-yellow-500 mt-1" /> :
                   <CheckCircle className="text-blue-500 mt-1" />}
                  <div className="flex-1">
                    <h4 className="font-bold mb-1">{issue.title}</h4>
                    <p className="text-sm mb-2">{issue.description}</p>
                    <div className="text-sm bg-white dark:bg-gray-800 p-2 rounded font-mono">
                      💡 {issue.fix}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {optimizedCode && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 text-green-600">
            <CheckCircle className="inline mr-2" />
            최적화된 코드
          </h3>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
            {optimizedCode}
          </pre>
        </div>
      )}

      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h3 className="text-xl font-bold mb-4">💡 Spark 최적화 팁</h3>
        <ul className="space-y-2 text-sm">
          <li><strong>Catalyst Optimizer 활용:</strong> DataFrame API 사용 (RDD보다 최적화)</li>
          <li><strong>Predicate Pushdown:</strong> 필터를 최대한 데이터 소스에 가깝게</li>
          <li><strong>Broadcast Join:</strong> 작은 테이블은 broadcast() 사용</li>
          <li><strong>Partition 튜닝:</strong> repartition/coalesce로 파티션 수 조정</li>
          <li><strong>Caching:</strong> 반복 사용 데이터는 .cache() 또는 .persist()</li>
        </ul>
      </div>
    </div>
  );
}
