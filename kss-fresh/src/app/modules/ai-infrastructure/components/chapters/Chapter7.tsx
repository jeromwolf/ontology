'use client'

import React from 'react'
import { Database, Zap, GitBranch, Boxes, Clock, Layers, Server, Package } from 'lucide-react'

export default function Chapter7() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <Database className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                Feature Store와 데이터 파이프라인
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Feast, Tecton, DVC로 피처를 중앙 관리하고 데이터를 버전 관리하기
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Boxes className="w-6 h-6 text-slate-700" />
              Feature Store가 필요한 이유
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                머신러닝 프로젝트에서 <strong>피처 엔지니어링</strong>은 전체 시간의 60-80%를 차지합니다.
                하지만 각 팀이 동일한 피처를 중복 개발하고, 훈련과 서빙 환경에서 피처 계산 로직이 달라지는
                <strong> Training-Serving Skew</strong> 문제가 발생합니다.
                Feature Store는 이러한 문제를 해결하는 <strong>중앙 집중식 피처 저장소</strong>입니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">Feature Store 핵심 가치</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>재사용성</strong>: 한 번 정의한 피처를 모든 팀이 활용</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Clock className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>일관성</strong>: 훈련과 서빙에서 동일한 피처 제공</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <GitBranch className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span><strong>버전 관리</strong>: 피처 변경 이력 추적 및 롤백</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Server className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>성능</strong>: Online/Offline Feature 분리로 저지연 서빙</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Online vs Offline Features */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Layers className="w-6 h-6 text-slate-700" />
            Online vs Offline Features
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-green-100 dark:bg-green-900 rounded-lg">
                  <Zap className="w-6 h-6 text-green-600 dark:text-green-400" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 dark:text-white">Online Features</h3>
              </div>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                실시간 추론을 위한 저지연 피처 제공
              </p>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>저장소</strong>: Redis, DynamoDB, Cassandra</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>지연시간</strong>: 1-10ms</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>용도</strong>: 실시간 추론 (API 요청)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>데이터 크기</strong>: 소량 (최신 상태만)</span>
                </li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                  <Database className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 dark:text-white">Offline Features</h3>
              </div>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                모델 훈련을 위한 대규모 히스토리컬 데이터
              </p>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>저장소</strong>: S3, BigQuery, Snowflake</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>지연시간</strong>: 초~분 단위 (배치 처리)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>용도</strong>: 모델 훈련, 백필(Backfill)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>데이터 크기</strong>: 대용량 (수년치 이력)</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Feast Feature Store */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Package className="w-6 h-6 text-slate-700" />
            Feast: 오픈소스 Feature Store
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Feast 설치 및 초기 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# Feast 설치
pip install feast

# 새 프로젝트 초기화
feast init my_feature_repo
cd my_feature_repo

# 파일 구조
my_feature_repo/
├── feature_store.yaml    # 설정 파일
├── features.py           # Feature 정의
└── data/                 # 예제 데이터`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Feature 정의 예제
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# features.py
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from datetime import timedelta

# Entity 정의 (피처를 조인할 키)
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="사용자 고유 ID"
)

# Data Source 정의
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# FeatureView 정의
user_stats_fv = FeatureView(
    name="user_statistics",
    entities=["user_id"],
    ttl=timedelta(days=1),  # 피처 유효 기간
    features=[
        Feature(name="total_orders", dtype=ValueType.INT64),
        Feature(name="avg_order_value", dtype=ValueType.DOUBLE),
        Feature(name="days_since_last_order", dtype=ValueType.INT64),
    ],
    online=True,  # Online store에 materialize
    source=user_stats_source,
    tags={"team": "growth", "version": "v1"},
)

# Feature Service 정의 (여러 FeatureView 묶음)
from feast import FeatureService

recommendation_service = FeatureService(
    name="recommendation_features",
    features=[
        user_stats_fv[["total_orders", "avg_order_value"]],
    ],
)`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Feast Apply & Materialize
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# 1. Feature Store에 등록
feast apply

# 2. Offline → Online Store로 Materialize
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

# 또는 전체 기간 materialize
feast materialize \\
    2024-01-01T00:00:00 \\
    2024-12-31T23:59:59

# 3. 피처 조회 (Online)
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

# 엔티티 데이터프레임
entity_df = pd.DataFrame({
    "user_id": [1001, 1002, 1003],
    "event_timestamp": [
        datetime(2024, 1, 15),
        datetime(2024, 1, 15),
        datetime(2024, 1, 15),
    ]
})

# Online 피처 조회 (실시간 추론용)
online_features = store.get_online_features(
    features=[
        "user_statistics:total_orders",
        "user_statistics:avg_order_value",
    ],
    entity_rows=[
        {"user_id": 1001},
        {"user_id": 1002},
    ]
).to_dict()

# Offline 피처 조회 (훈련용)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_statistics:total_orders",
        "user_statistics:avg_order_value",
    ]
).to_df()

print(training_df.head())`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">Feast 아키텍처</h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Registry</p>
                  <p className="text-sm text-slate-200">피처 메타데이터 저장 (S3, GCS)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Offline Store</p>
                  <p className="text-sm text-slate-200">대용량 배치 피처 (BigQuery, Snowflake)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Online Store</p>
                  <p className="text-sm text-slate-200">저지연 실시간 피처 (Redis, DynamoDB)</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Tecton */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Tecton: 엔터프라이즈 Feature Store
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Tecton Feature Definition
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from tecton import Entity, BatchSource, FeatureView
from tecton.types import Field, String, Int64, Float64
from datetime import timedelta

# Entity 정의
user = Entity(
    name="user",
    join_keys=[Field("user_id", String)],
    description="사용자 엔티티"
)

# Batch Source
user_events = BatchSource(
    name="user_events",
    batch_config=SnowflakeBatchConfig(
        database="PROD",
        schema="ML_FEATURES",
        table="USER_EVENTS",
        timestamp_field="timestamp"
    )
)

# Real-time Feature (Stream Processing)
@stream_feature_view(
    source=KafkaSource(...),
    entities=[user],
    mode="spark_sql",
    online=True,
    offline=True,
    feature_start_time=datetime(2024, 1, 1),
    features=[
        Feature("click_count_1h", Int64),
        Feature("purchase_amount_1h", Float64),
    ],
    timestamp_field="timestamp"
)
def user_real_time_features(events):
    return f"""
        SELECT
            user_id,
            timestamp,
            COUNT(*) AS click_count_1h,
            SUM(amount) AS purchase_amount_1h
        FROM
            {events}
        WHERE
            event_type = 'click'
        GROUP BY
            user_id,
            TUMBLE(timestamp, INTERVAL 1 HOUR)
    """`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
                <h3 className="font-bold text-lg mb-3">Tecton 장점</h3>
                <ul className="space-y-2 text-sm text-purple-100">
                  <li>• 실시간 스트림 피처 지원 (Kafka, Kinesis)</li>
                  <li>• On-Demand 피처 변환 (요청 시 계산)</li>
                  <li>• 자동 Feature Monitoring</li>
                  <li>• SLA 기반 데이터 품질 보장</li>
                </ul>
              </div>
              <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-xl">
                <h3 className="font-bold text-lg mb-3">Hopsworks Feature Store</h3>
                <ul className="space-y-2 text-sm text-blue-100">
                  <li>• Python/Java/Scala API</li>
                  <li>• Time-travel 기능 (과거 시점 피처 조회)</li>
                  <li>• Feature Validation (Great Expectations 통합)</li>
                  <li>• HSFS (Hopsworks Feature Store) 오픈소스</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Data Versioning with DVC */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-slate-700" />
            DVC: 데이터 버전 관리
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                DVC 설치 및 초기화
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# DVC 설치
pip install dvc dvc-s3

# Git 저장소에서 초기화
git init
dvc init

# Remote storage 설정 (S3)
dvc remote add -d myremote s3://my-bucket/dvc-storage

# AWS credentials 설정
dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                데이터 추적 및 버전 관리
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# 1. 데이터 파일 추적
dvc add data/train.csv
# → data/train.csv.dvc 생성 (메타데이터)
# → data/train.csv를 .gitignore에 추가

git add data/train.csv.dvc .gitignore
git commit -m "Add training data v1"

# 2. Remote storage에 푸시
dvc push

# 3. 데이터 업데이트
# data/train.csv 수정 후
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "Update training data v2"
dvc push

# 4. 이전 버전으로 되돌리기
git checkout HEAD~1 data/train.csv.dvc
dvc checkout

# 5. Remote에서 데이터 받기
dvc pull`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                DVC Pipelines: 재현 가능한 ML 워크플로우
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# dvc.yaml - 파이프라인 정의
stages:
  prepare:
    cmd: python prepare.py
    deps:
      - data/raw.csv
    outs:
      - data/prepared.csv

  featurize:
    cmd: python featurize.py
    deps:
      - data/prepared.csv
    outs:
      - features/train.pkl
      - features/test.pkl

  train:
    cmd: python train.py
    deps:
      - features/train.pkl
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - models/model.pkl
    metrics:
      - metrics/train.json:
          cache: false

  evaluate:
    cmd: python evaluate.py
    deps:
      - models/model.pkl
      - features/test.pkl
    metrics:
      - metrics/eval.json:
          cache: false

# 파이프라인 실행
dvc repro

# 특정 스테이지만 실행
dvc repro train

# Metrics 비교
dvc metrics show
dvc metrics diff HEAD~1`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="font-bold text-slate-800 dark:text-white mb-2">LakeFS</p>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Data Lake를 Git처럼 관리 (Branch, Merge, Commit)
                </p>
              </div>
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <p className="font-bold text-slate-800 dark:text-white mb-2">Pachyderm</p>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Kubernetes 기반 데이터 파이프라인 버전 관리
                </p>
              </div>
              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <p className="font-bold text-slate-800 dark:text-white mb-2">Delta Lake</p>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  ACID 트랜잭션 지원 Data Lake (Time Travel)
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ETL Workflows */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            ETL 워크플로우 (Spark, Flink, Beam)
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Apache Spark: 배치 피처 처리
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, datediff, current_date

spark = SparkSession.builder \\
    .appName("FeatureEngineering") \\
    .getOrCreate()

# 원본 데이터 로드
orders = spark.read.parquet("s3://data/orders/")
users = spark.read.parquet("s3://data/users/")

# 사용자별 통계 피처 생성
user_features = orders.groupBy("user_id").agg(
    count("order_id").alias("total_orders"),
    avg("order_amount").alias("avg_order_value"),
    max("order_date").alias("last_order_date")
)

# 파생 피처 추가
user_features = user_features.withColumn(
    "days_since_last_order",
    datediff(current_date(), col("last_order_date"))
)

# 결과 저장
user_features.write.mode("overwrite") \\
    .parquet("s3://features/user_stats/")

spark.stop()`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Apache Flink: 실시간 피처 스트리밍
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Kafka Source 정의
t_env.execute_sql("""
    CREATE TABLE click_events (
        user_id STRING,
        product_id STRING,
        timestamp TIMESTAMP(3),
        WATERMARK FOR timestamp AS timestamp - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'user-clicks',
        'properties.bootstrap.servers' = 'localhost:9092',
        'format' = 'json'
    )
""")

# 1시간 윈도우 집계
t_env.execute_sql("""
    CREATE TABLE user_click_features AS
    SELECT
        user_id,
        TUMBLE_END(timestamp, INTERVAL '1' HOUR) as window_end,
        COUNT(*) as click_count_1h,
        COUNT(DISTINCT product_id) as unique_products_1h
    FROM click_events
    GROUP BY user_id, TUMBLE(timestamp, INTERVAL '1' HOUR)
""")

# Redis Sink (Online Store)
t_env.execute_sql("""
    CREATE TABLE redis_sink (
        user_id STRING,
        click_count_1h BIGINT,
        unique_products_1h BIGINT,
        PRIMARY KEY (user_id) NOT ENFORCED
    ) WITH (
        'connector' = 'redis',
        'host' = 'localhost',
        'port' = '6379'
    )
""")

# 스트림 실행
t_env.execute_sql("""
    INSERT INTO redis_sink
    SELECT user_id, click_count_1h, unique_products_1h
    FROM user_click_features
""")`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-orange-500 to-red-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">Apache Beam: 통합 배치/스트림 처리</h3>
              <div className="bg-white/10 rounded-lg p-6 font-mono text-sm backdrop-blur overflow-x-auto">
                <pre className="text-white">
{`import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Beam Pipeline (Dataflow에서 실행 가능)
with beam.Pipeline(options=PipelineOptions()) as p:
    (p
     | 'Read from BigQuery' >> beam.io.ReadFromBigQuery(
           query='SELECT * FROM \`project.dataset.orders\`')
     | 'Extract user_id' >> beam.Map(lambda x: (x['user_id'], x))
     | 'Group by user' >> beam.GroupByKey()
     | 'Compute features' >> beam.Map(compute_user_features)
     | 'Write to Feast' >> beam.ParDo(WriteFeastFeatures())
    )`}
                </pre>
              </div>
              <p className="text-sm text-orange-100 mt-4">
                Beam은 <strong>동일한 코드</strong>로 Spark, Flink, Dataflow 등 다양한 러너에서 실행 가능
              </p>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8 border border-blue-200 dark:border-blue-800">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4">
              핵심 요점
            </h2>
            <ul className="space-y-3 text-slate-700 dark:text-slate-300">
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">1.</span>
                <span><strong>Feature Store</strong>는 피처 재사용성과 훈련-서빙 일관성을 보장합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>Online Features</strong>는 Redis/DynamoDB에서 1-10ms 지연으로 제공됩니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Feast</strong>는 오픈소스 Feature Store로 AWS, GCP, Snowflake 통합을 지원합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>DVC</strong>는 데이터를 Git처럼 버전 관리하고, 파이프라인 재현성을 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>Spark</strong>는 배치 피처 처리, <strong>Flink</strong>는 실시간 스트림 피처에 최적화되어 있습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span><strong>Apache Beam</strong>은 배치와 스트림을 통합 API로 처리하며 다양한 러너를 지원합니다.</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Next Chapter Preview */}
        <section>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border-2 border-slate-300 dark:border-gray-600">
            <h3 className="text-lg font-bold text-slate-800 dark:text-white mb-2">
              다음 챕터 미리보기
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              <strong>Chapter 8: 실험 추적과 메타데이터 관리</strong>
              <br />
              MLflow, Weights & Biases, Neptune.ai로 실험을 체계적으로 관리하고,
              하이퍼파라미터 튜닝 결과를 비교하는 방법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
