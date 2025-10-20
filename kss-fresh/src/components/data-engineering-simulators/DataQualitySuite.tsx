'use client';

import { useState } from 'react';
import { Shield, CheckCircle, XCircle, AlertTriangle, Play } from 'lucide-react';

interface QualityCheck {
  name: string;
  type: 'completeness' | 'accuracy' | 'consistency' | 'timeliness';
  status: 'pass' | 'fail' | 'warning' | 'pending';
  message: string;
}

export default function DataQualitySuite() {
  const [checks, setChecks] = useState<QualityCheck[]>([
    { name: 'NULL 값 체크', type: 'completeness', status: 'pending', message: '' },
    { name: '이메일 형식 검증', type: 'accuracy', status: 'pending', message: '' },
    { name: '날짜 범위 검증', type: 'consistency', status: 'pending', message: '' },
    { name: '최신 데이터 확인', type: 'timeliness', status: 'pending', message: '' },
  ]);

  const runChecks = () => {
    const results: QualityCheck[] = [
      { name: 'NULL 값 체크', type: 'completeness', status: 'pass', message: '99.8% 완전성' },
      { name: '이메일 형식 검증', type: 'accuracy', status: 'warning', message: '12건 형식 오류 발견' },
      { name: '날짜 범위 검증', type: 'consistency', status: 'fail', message: '2024년 이전 데이터 존재' },
      { name: '최신 데이터 확인', type: 'timeliness', status: 'pass', message: '최근 1시간 내 데이터' },
    ];
    setChecks(results);
  };

  const getStatusIcon = (status: QualityCheck['status']) => {
    switch (status) {
      case 'pass': return <CheckCircle className="text-green-500" />;
      case 'fail': return <XCircle className="text-red-500" />;
      case 'warning': return <AlertTriangle className="text-yellow-500" />;
      default: return <div className="w-5 h-5 rounded-full bg-gray-300" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Shield size={32} />
          <h2 className="text-2xl font-bold">데이터 품질 검증 Suite</h2>
        </div>
        <p className="text-green-100">Great Expectations 스타일 자동화 데이터 품질 체크</p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold">🔍 품질 검증 항목</h3>
          <button
            onClick={runChecks}
            className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-semibold"
          >
            <Play size={18} /> 검증 실행
          </button>
        </div>

        <div className="space-y-3">
          {checks.map((check, idx) => (
            <div key={idx} className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              {getStatusIcon(check.status)}
              <div className="flex-1">
                <div className="font-semibold">{check.name}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">{check.message || '대기 중...'}</div>
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                check.type === 'completeness' ? 'bg-blue-100 text-blue-700' :
                check.type === 'accuracy' ? 'bg-green-100 text-green-700' :
                check.type === 'consistency' ? 'bg-purple-100 text-purple-700' :
                'bg-orange-100 text-orange-700'
              }`}>
                {check.type}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">📝 Great Expectations 예제</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`import great_expectations as gx

context = gx.get_context()

# Expectation Suite 생성
suite = context.add_expectation_suite(suite_name="user_data_quality")

# Completeness
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
)

# Accuracy
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToMatchRegex(
        column="email",
        regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    )
)

# Consistency
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="age",
        min_value=0,
        max_value=120
    )
)

# Timeliness
suite.add_expectation(
    gx.expectations.ExpectColumnMaxToBeBetween(
        column="created_at",
        min_value=datetime.now() - timedelta(hours=24)
    )
)

# Checkpoint 실행
checkpoint = context.add_checkpoint(
    name="daily_quality_check",
    validations=[{"expectation_suite_name": "user_data_quality"}]
)

result = checkpoint.run()
print(result)`}
        </pre>
      </div>
    </div>
  );
}
