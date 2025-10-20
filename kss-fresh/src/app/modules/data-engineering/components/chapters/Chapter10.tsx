'use client';

import { useState } from 'react';
import {
  TrendingUp, DollarSign, Gauge, Zap,
  Database, HardDrive, CheckCircle, AlertTriangle
} from 'lucide-react';

export default function Chapter10() {
  return (
    <div className="space-y-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">성능 최적화와 비용 관리</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          쿼리 최적화, 파티셔닝, 인덱싱으로 성능을 높이고 클라우드 비용 절감하기
        </p>
      </div>

      {/* 쿼리 최적화 기법 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Gauge className="text-blue-600" />
          SQL 쿼리 최적화 기법
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-red-500 pl-4">
            <h3 className="font-bold mb-2 text-red-600">❌ 비효율적인 쿼리</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- SELECT * 사용 (필요 없는 컬럼도 로드)
SELECT * FROM orders WHERE order_date >= '2024-01-01';

-- 서브쿼리 중복 실행
SELECT user_id,
       (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) as order_count,
       (SELECT SUM(total) FROM orders WHERE orders.user_id = users.id) as total_spent
FROM users;

-- 함수를 WHERE 절에 사용 (인덱스 활용 불가)
SELECT * FROM events WHERE DATE(created_at) = '2024-01-15';`}
            </pre>
          </div>

          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="font-bold mb-2 text-green-600">✅ 최적화된 쿼리</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- 필요한 컬럼만 선택
SELECT user_id, order_date, total_amount
FROM orders
WHERE order_date >= '2024-01-01';

-- JOIN으로 한 번에 계산
SELECT u.user_id,
       COUNT(o.order_id) as order_count,
       COALESCE(SUM(o.total), 0) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.user_id;

-- 범위 조건으로 인덱스 활용
SELECT * FROM events
WHERE created_at >= '2024-01-15 00:00:00'
  AND created_at < '2024-01-16 00:00:00';`}
            </pre>
          </div>
        </div>
      </section>

      {/* 파티셔닝 전략 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <HardDrive className="text-purple-600" />
          파티셔닝 전략
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <h3 className="font-bold mb-2 text-blue-600">Range 파티셔닝</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              날짜/숫자 범위로 분할 (시계열 데이터에 적합)
            </p>
            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs">
{`-- PostgreSQL
CREATE TABLE events (
    id SERIAL,
    event_type VARCHAR(50),
    created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

CREATE TABLE events_2024_01
PARTITION OF events
FOR VALUES FROM ('2024-01-01')
           TO ('2024-02-01');`}
            </pre>
          </div>

          <div className="border border-green-200 dark:border-green-800 rounded-lg p-4">
            <h3 className="font-bold mb-2 text-green-600">Hash 파티셔닝</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              해시 함수로 균등 분산 (user_id 등)
            </p>
            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs">
{`-- BigQuery
CREATE TABLE users
PARTITION BY
  DATE_TRUNC(signup_date, MONTH)
CLUSTER BY country, user_id
AS
SELECT * FROM source_users;`}
            </pre>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
          <h3 className="font-bold mb-2">파티션 프루닝 (Partition Pruning)</h3>
          <p className="text-sm mb-2">
            WHERE 절에 파티션 키를 사용하면 필요한 파티션만 스캔하여 성능 향상
          </p>
          <pre className="bg-gray-900 text-gray-100 p-3 rounded text-sm">
{`-- 1개 파티션만 스캔 (1월 데이터만)
SELECT COUNT(*) FROM events
WHERE created_at >= '2024-01-01'
  AND created_at < '2024-02-01';

-- ❌ 전체 파티션 스캔
SELECT COUNT(*) FROM events
WHERE MONTH(created_at) = 1;  -- 함수 사용으로 프루닝 불가`}
          </pre>
        </div>
      </section>

      {/* 인덱싱 전략 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Database className="text-orange-600" />
          인덱싱 전략
        </h2>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-bold mb-2">B-Tree 인덱스</h3>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
              범위 검색, 정렬에 적합
            </p>
            <pre className="bg-gray-900 text-gray-100 p-2 rounded text-xs">
{`CREATE INDEX idx_order_date
ON orders(order_date);

-- 사용
WHERE order_date >= '2024-01-01'`}
            </pre>
          </div>

          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-bold mb-2">Hash 인덱스</h3>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
              등호(=) 검색에 최적화
            </p>
            <pre className="bg-gray-900 text-gray-100 p-2 rounded text-xs">
{`CREATE INDEX idx_user_email
ON users USING HASH(email);

-- 사용
WHERE email = 'user@example.com'`}
            </pre>
          </div>

          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-bold mb-2">복합 인덱스</h3>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
              여러 컬럼 조합 (순서 중요)
            </p>
            <pre className="bg-gray-900 text-gray-100 p-2 rounded text-xs">
{`CREATE INDEX idx_user_date
ON orders(user_id, order_date);

-- 사용 (좌측 우선)
WHERE user_id = 123
  AND order_date >= '2024-01-01'`}
            </pre>
          </div>
        </div>

        <div className="mt-4 bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
          <h3 className="font-bold mb-2">⚠️ 인덱스 주의사항</h3>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <AlertTriangle className="text-orange-500 mt-0.5 flex-shrink-0" size={16} />
              <span>너무 많은 인덱스는 INSERT/UPDATE 성능 저하</span>
            </li>
            <li className="flex items-start gap-2">
              <AlertTriangle className="text-orange-500 mt-0.5 flex-shrink-0" size={16} />
              <span>카디널리티가 낮은 컬럼(성별 등)은 인덱스 효과 적음</span>
            </li>
            <li className="flex items-start gap-2">
              <AlertTriangle className="text-orange-500 mt-0.5 flex-shrink-0" size={16} />
              <span>복합 인덱스는 컬럼 순서가 쿼리 패턴과 일치해야 함</span>
            </li>
          </ul>
        </div>
      </section>

      {/* 비용 최적화 */}
      <section className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <DollarSign className="text-green-600" />
          클라우드 비용 최적화 체크리스트
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-3 text-green-600">스토리지 최적화</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5" size={16} />
                <span>오래된 파티션 삭제 (GDPR 준수)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5" size={16} />
                <span>컬럼나 포맷(Parquet/ORC) 사용</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5" size={16} />
                <span>압축 알고리즘 적용 (Snappy, ZSTD)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5" size={16} />
                <span>사용하지 않는 테이블 아카이빙</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-3 text-blue-600">컴퓨팅 최적화</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5" size={16} />
                <span>쿼리 결과 캐싱 활용</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5" size={16} />
                <span>불필요한 웨어하우스 자동 중지</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5" size={16} />
                <span>예약 슬롯/커밋 사용 (대규모 워크로드)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5" size={16} />
                <span>스팟 인스턴스 활용 (비중요 작업)</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-4 bg-white dark:bg-gray-800 p-4 rounded-lg">
          <h3 className="font-bold mb-3 text-purple-600">비용 모니터링</h3>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- Snowflake 비용 쿼리
SELECT
    warehouse_name,
    DATE_TRUNC('day', start_time) as date,
    SUM(credits_used) as total_credits,
    SUM(credits_used) * 3.0 as estimated_cost_usd  -- $3/크레딧 가정
FROM snowflake.account_usage.warehouse_metering_history
WHERE start_time >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY warehouse_name, date
ORDER BY total_credits DESC;

-- BigQuery 비용 쿼리
SELECT
    user_email,
    SUM(total_bytes_processed) / POW(10, 12) as tb_processed,
    SUM(total_bytes_processed) / POW(10, 12) * 5 as cost_usd
FROM \`region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT\`
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY user_email
ORDER BY tb_processed DESC;`}
          </pre>
        </div>
      </section>

      {/* 성능 튜닝 프로세스 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <TrendingUp className="text-indigo-600" />
          성능 튜닝 프로세스
        </h2>

        <div className="space-y-3">
          {[
            { step: '1. 프로파일링', desc: 'EXPLAIN ANALYZE로 쿼리 실행 계획 분석', color: 'blue' },
            { step: '2. 병목 식별', desc: 'Seq Scan, Nested Loop 등 비효율적 연산 찾기', color: 'red' },
            { step: '3. 인덱스 추가', desc: 'WHERE/JOIN 조건에 사용되는 컬럼에 인덱스 생성', color: 'green' },
            { step: '4. 쿼리 재작성', desc: 'JOIN 순서 변경, 서브쿼리를 CTE로 전환', color: 'purple' },
            { step: '5. 파티셔닝', desc: '대용량 테이블을 시간/해시로 분할', color: 'orange' },
            { step: '6. 머티리얼라이즈드 뷰', desc: '자주 실행되는 복잡한 쿼리 결과를 미리 계산', color: 'teal' }
          ].map((item, idx) => (
            <div key={idx} className={`border-l-4 border-${item.color}-500 pl-4 py-2`}>
              <h3 className="font-bold">{item.step}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg">
        <h3 className="text-xl font-bold mb-2">다음 단계: MLOps를 위한 데이터 엔지니어링</h3>
        <p className="text-gray-700 dark:text-gray-300">
          최적화된 데이터 파이프라인 위에 Feature Store와 ML 파이프라인을 구축하는 방법을 학습합니다.
        </p>
      </div>
    </div>
  );
}
