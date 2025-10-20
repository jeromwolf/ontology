'use client';

import { useState } from 'react';
import {
  Cloud, Database, Zap, DollarSign,
  TrendingUp, Shield, Globe, Server,
  CheckCircle, Code, BarChart3, Lock
} from 'lucide-react';

export default function Chapter8() {
  const [selectedPlatform, setSelectedPlatform] = useState('snowflake');

  const platforms = {
    snowflake: {
      name: 'Snowflake',
      icon: '❄️',
      color: 'blue',
      strengths: ['분리된 스토리지/컴퓨팅', '제로 관리', '자동 스케일링', '타임 트래블'],
      pricing: '사용량 기반 (컴퓨팅 + 스토리지)',
      bestFor: '데이터 웨어하우징, 멀티 클라우드'
    },
    bigquery: {
      name: 'Google BigQuery',
      icon: '🔍',
      color: 'green',
      strengths: ['서버리스', '초고속 SQL', 'ML 통합', 'GCP 생태계'],
      pricing: '쿼리당 과금 ($5/TB) 또는 슬롯 예약',
      bestFor: '애드혹 분석, 빅데이터 처리'
    },
    databricks: {
      name: 'Databricks',
      icon: '🧱',
      color: 'orange',
      strengths: ['통합 분석', 'Delta Lake', 'MLOps', 'Spark 최적화'],
      pricing: 'DBU(Databricks Unit) + 클라우드 비용',
      bestFor: 'ML/AI 파이프라인, 레이크하우스'
    },
    redshift: {
      name: 'AWS Redshift',
      icon: '🚀',
      color: 'purple',
      strengths: ['AWS 통합', '컬럼나 스토리지', 'Redshift Spectrum', 'S3 연동'],
      pricing: '노드당 시간 과금',
      bestFor: 'AWS 중심 환경, 대규모 DW'
    }
  };

  return (
    <div className="space-y-8">
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">클라우드 데이터 플랫폼 실전</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          Snowflake, BigQuery, Databricks, Redshift를 비교하고 최적의 플랫폼 선택하기
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-sky-50 to-blue-50 dark:from-sky-900/20 dark:to-blue-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Zap className="text-sky-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          {[
            '주요 클라우드 데이터 플랫폼 비교 분석',
            '각 플랫폼의 아키텍처와 특징 이해',
            '비용 모델과 최적화 전략',
            '실제 쿼리 패턴과 성능 튜닝'
          ].map((goal, idx) => (
            <div key={idx} className="flex items-start gap-3">
              <CheckCircle className="text-sky-500 mt-1 flex-shrink-0" />
              <span>{goal}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 플랫폼 선택 탭 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Cloud className="text-blue-600" />
          클라우드 데이터 플랫폼 비교
        </h2>

        {/* 탭 버튼 */}
        <div className="flex flex-wrap gap-2 mb-6">
          {Object.keys(platforms).map((key) => (
            <button
              key={key}
              onClick={() => setSelectedPlatform(key)}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                selectedPlatform === key
                  ? `bg-${platforms[key].color}-500 text-white`
                  : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {platforms[key].icon} {platforms[key].name}
            </button>
          ))}
        </div>

        {/* 선택된 플랫폼 상세 */}
        <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
          <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <span className="text-3xl">{platforms[selectedPlatform].icon}</span>
            {platforms[selectedPlatform].name}
          </h3>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold mb-3 flex items-center gap-2">
                <TrendingUp className="text-green-500" />
                주요 강점
              </h4>
              <ul className="space-y-2">
                {platforms[selectedPlatform].strengths.map((strength, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={18} />
                    <span>{strength}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="font-bold mb-3 flex items-center gap-2">
                <DollarSign className="text-blue-500" />
                가격 모델
              </h4>
              <p className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-sm">
                {platforms[selectedPlatform].pricing}
              </p>

              <h4 className="font-bold mb-3 mt-4 flex items-center gap-2">
                <BarChart3 className="text-purple-500" />
                최적 사용 사례
              </h4>
              <p className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-sm">
                {platforms[selectedPlatform].bestFor}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Snowflake 심화 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          ❄️ Snowflake 심화
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="font-bold mb-2">분리된 스토리지와 컴퓨팅 아키텍처</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- Virtual Warehouse 생성 (컴퓨팅)
CREATE WAREHOUSE analytics_wh
  WITH WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 300           -- 5분 후 자동 중지
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- 데이터 로딩 (스토리지는 자동 관리)
COPY INTO customers
FROM @my_s3_stage/customers.csv
FILE_FORMAT = (TYPE = CSV);

-- 쿼리 실행 시 웨어하우스만 활성화
USE WAREHOUSE analytics_wh;
SELECT * FROM customers LIMIT 10;`}
            </pre>
          </div>

          <div className="border-l-4 border-purple-500 pl-4">
            <h3 className="font-bold mb-2">타임 트래블 & 제로 카피 클로닝</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- 1시간 전 데이터 조회
SELECT * FROM orders
AT(OFFSET => -3600);  -- 초 단위

-- 특정 시점으로 복구
CREATE TABLE orders_restored
CLONE orders AT(TIMESTAMP => '2024-01-15 12:00:00'::TIMESTAMP);

-- 제로 카피 클론 (즉시 복사, 스토리지 비용 없음)
CREATE TABLE dev_customers CLONE prod_customers;`}
            </pre>
          </div>
        </div>
      </section>

      {/* BigQuery 심화 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          🔍 Google BigQuery 심화
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="font-bold mb-2">파티셔닝 & 클러스터링</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- 날짜 파티션 테이블 생성
CREATE TABLE \`project.dataset.events\`
PARTITION BY DATE(event_timestamp)
CLUSTER BY user_id, event_type
AS
SELECT * FROM source_events;

-- 파티션 프루닝으로 비용 절감
SELECT user_id, COUNT(*) as event_count
FROM \`project.dataset.events\`
WHERE DATE(event_timestamp) = '2024-01-15'  -- 하루치만 스캔
GROUP BY user_id;

-- 비용 확인
SELECT
  FORMAT("%.2f", SUM(total_bytes_processed) / POW(10, 12) * 5) as estimated_cost_usd
FROM \`project.dataset.INFORMATION_SCHEMA.JOBS\`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY);`}
            </pre>
          </div>

          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="font-bold mb-2">BigQuery ML 통합</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- SQL로 머신러닝 모델 훈련
CREATE OR REPLACE MODEL \`project.dataset.churn_model\`
OPTIONS(
  model_type='logistic_reg',
  input_label_cols=['churned']
) AS
SELECT
  customer_lifetime_value,
  total_purchases,
  days_since_last_purchase,
  churned
FROM \`project.dataset.customer_features\`;

-- 모델로 예측
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(0)].prob as churn_probability
FROM ML.PREDICT(MODEL \`project.dataset.churn_model\`,
  (SELECT * FROM \`project.dataset.current_customers\`));`}
            </pre>
          </div>
        </div>
      </section>

      {/* Databricks 심화 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          🧱 Databricks & Delta Lake 심화
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-orange-500 pl-4">
            <h3 className="font-bold mb-2">Delta Lake ACID 트랜잭션</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`from delta.tables import DeltaTable
from pyspark.sql import SparkSession

# Delta 테이블 생성
df.write.format("delta") \\
  .mode("overwrite") \\
  .partitionBy("date") \\
  .save("/mnt/delta/events")

# MERGE 연산 (Upsert)
deltaTable = DeltaTable.forPath(spark, "/mnt/delta/customers")
deltaTable.alias("target") \\
  .merge(
    updates.alias("source"),
    "target.customer_id = source.customer_id"
  ) \\
  .whenMatchedUpdateAll() \\
  .whenNotMatchedInsertAll() \\
  .execute()

# 타임 트래블
df = spark.read.format("delta") \\
  .option("versionAsOf", 5) \\
  .load("/mnt/delta/events")

# VACUUM으로 오래된 파일 정리 (30일 이전)
deltaTable.vacuum(168)  # hours`}
            </pre>
          </div>

          <div className="border-l-4 border-red-500 pl-4">
            <h3 className="font-bold mb-2">Auto Loader로 스트리밍 수집</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`# S3에서 신규 파일 자동 감지 및 처리
df = spark.readStream.format("cloudFiles") \\
  .option("cloudFiles.format", "json") \\
  .option("cloudFiles.schemaLocation", "/mnt/schema") \\
  .load("s3://bucket/incoming/")

# Delta Lake에 스트리밍 쓰기
df.writeStream \\
  .format("delta") \\
  .outputMode("append") \\
  .option("checkpointLocation", "/mnt/checkpoints") \\
  .start("/mnt/delta/raw_events")`}
            </pre>
          </div>
        </div>
      </section>

      {/* 비용 최적화 전략 */}
      <section className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <DollarSign className="text-green-600" />
          클라우드 비용 최적화 전략
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-3 text-green-600">Snowflake 최적화</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5" size={16} />
                <span>웨어하우스 자동 중지 시간 최소화 (60초)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5" size={16} />
                <span>쿼리 결과 캐싱 활용 (24시간 유효)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5" size={16} />
                <span>클러스터링 키로 마이크로 파티션 최적화</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5" size={16} />
                <span>멀티 클러스터 웨어하우스로 동시성 처리</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-3 text-blue-600">BigQuery 최적화</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5" size={16} />
                <span>파티셔닝으로 스캔 데이터 최소화</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5" size={16} />
                <span>클러스터링으로 쿼리 성능 향상 (무료)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5" size={16} />
                <span>예약 슬롯으로 대규모 워크로드 비용 절감</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5" size={16} />
                <span>BI Engine으로 반복 쿼리 가속화</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-3 text-orange-600">Databricks 최적화</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-orange-500 mt-0.5" size={16} />
                <span>Delta Cache로 반복 쿼리 성능 향상</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-orange-500 mt-0.5" size={16} />
                <span>Photon 엔진으로 Spark 쿼리 2-3배 가속</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-orange-500 mt-0.5" size={16} />
                <span>Auto-scaling 클러스터로 유휴 리소스 제거</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-orange-500 mt-0.5" size={16} />
                <span>Z-ordering으로 데이터 스큐 최소화</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-3 text-purple-600">Redshift 최적화</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-purple-500 mt-0.5" size={16} />
                <span>Distribution Key로 조인 최적화</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-purple-500 mt-0.5" size={16} />
                <span>Sort Key로 쿼리 속도 향상</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-purple-500 mt-0.5" size={16} />
                <span>Spectrum으로 S3 데이터 직접 쿼리</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-purple-500 mt-0.5" size={16} />
                <span>RA3 인스턴스로 스토리지/컴퓨팅 분리</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 다음 단계 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg">
        <h3 className="text-xl font-bold mb-2">다음 단계: 데이터 오케스트레이션</h3>
        <p className="text-gray-700 dark:text-gray-300">
          클라우드 플랫폼을 선택한 후, Airflow/Dagster/Prefect로 데이터 파이프라인을
          스케줄링하고 모니터링하는 방법을 학습합니다.
        </p>
      </div>
    </div>
  );
}
