'use client';

import { useState } from 'react';
import {
  Brain, Database, GitBranch, Zap,
  TrendingUp, CheckCircle, Code, Layers
} from 'lucide-react';

export default function Chapter11() {
  return (
    <div className="space-y-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">MLOps를 위한 데이터 엔지니어링</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          Feature Store, ML 파이프라인, 모델 서빙을 위한 데이터 인프라 구축
        </p>
      </div>

      {/* Feature Store */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Database className="text-blue-600" />
          Feature Store - ML을 위한 데이터 저장소
        </h2>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg mb-4">
          <p className="text-sm">
            <strong>Feature Store</strong>는 ML 모델 훈련과 서빙을 위한 피처를 중앙에서 관리하며,
            훈련/서빙 간의 데이터 불일치 문제를 해결합니다.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="border border-purple-200 dark:border-purple-800 rounded-lg p-4">
            <h3 className="font-bold mb-2 text-purple-600">Offline Store</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              배치 훈련용 (S3, Snowflake, BigQuery)
            </p>
            <ul className="text-xs space-y-1">
              <li>• 대규모 히스토리 데이터</li>
              <li>• Point-in-time 조인</li>
              <li>• 시간 여행 가능</li>
            </ul>
          </div>

          <div className="border border-green-200 dark:border-green-800 rounded-lg p-4">
            <h3 className="font-bold mb-2 text-green-600">Online Store</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              실시간 서빙용 (Redis, DynamoDB, Cassandra)
            </p>
            <ul className="text-xs space-y-1">
              <li>• 밀리초 단위 조회</li>
              <li>• Key-value 스토어</li>
              <li>• 최신 피처만 보관</li>
            </ul>
          </div>
        </div>

        <div className="border-l-4 border-blue-500 pl-4">
          <h3 className="font-bold mb-2">Feast로 Feature Store 구축</h3>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`# feature_repo/features.py
from feast import Entity, Feature, FeatureView, Field
from feast.types import Float32, Int64
from datetime import timedelta

# 엔티티 정의 (user_id)
user = Entity(name="user", join_keys=["user_id"])

# Feature View 정의
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_purchase_value", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int64),
    ],
    online=True,
    source=BatchSource(
        path="s3://bucket/user_features.parquet",
        timestamp_field="event_timestamp"
    )
)

# Feature Store 적용
from feast import FeatureStore
store = FeatureStore(repo_path=".")

# 오프라인 스토어에서 훈련 데이터 생성
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:total_purchases",
        "user_features:avg_purchase_value",
    ]
).to_df()

# 온라인 스토어에서 실시간 피처 조회
online_features = store.get_online_features(
    features=[
        "user_features:total_purchases",
        "user_features:avg_purchase_value",
    ],
    entity_rows=[{"user_id": 12345}]
).to_dict()`}
          </pre>
        </div>
      </section>

      {/* ML 데이터 파이프라인 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <GitBranch className="text-purple-600" />
          ML 데이터 파이프라인
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-purple-500 pl-4">
            <h3 className="font-bold mb-2">Feature Engineering 파이프라인</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`from airflow import DAG
from airflow.providers.google.cloud.operators.dataflow import DataflowTemplatedJobStartOperator
from airflow.operators.python import PythonOperator

def compute_features():
    """Spark로 피처 계산"""
    df = spark.read.parquet("s3://bucket/raw_events/")

    # Aggregation 피처
    user_features = df.groupBy("user_id").agg(
        count("*").alias("total_events"),
        sum("purchase_amount").alias("lifetime_value"),
        datediff(current_date(), max("event_date")).alias("days_inactive")
    )

    # Feature Store에 저장
    user_features.write.mode("overwrite").save(
        "s3://bucket/features/user_features/"
    )

with DAG('ml_feature_pipeline', ...) as dag:
    # 원시 데이터 추출
    extract = PythonOperator(
        task_id='extract_events',
        python_callable=extract_from_db
    )

    # 피처 계산
    transform = PythonOperator(
        task_id='compute_features',
        python_callable=compute_features
    )

    # Feature Store 업데이트
    materialize = PythonOperator(
        task_id='materialize_to_online_store',
        python_callable=feast_materialize
    )

    extract >> transform >> materialize`}
          </pre>
          </div>

          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="font-bold mb-2">모델 훈련 데이터 준비</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 레이블 데이터 로드
labels = pd.read_sql("""
    SELECT user_id, churned, event_timestamp
    FROM user_labels
    WHERE event_timestamp >= '2024-01-01'
""", engine)

# 2. Feature Store에서 피처 조회
from feast import FeatureStore
store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=labels,
    features=[
        "user_features:total_purchases",
        "user_features:avg_purchase_value",
        "user_features:days_since_last_purchase",
        "session_features:avg_session_duration",
    ]
).to_df()

# 3. 훈련/검증 분할
X = training_df.drop(columns=['churned', 'event_timestamp'])
y = training_df['churned']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)`}
            </pre>
          </div>
        </div>
      </section>

      {/* 데이터 버전 관리 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Layers className="text-orange-600" />
          데이터 버전 관리 (DVC)
        </h2>

        <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg mb-4">
          <p className="text-sm">
            <strong>DVC (Data Version Control)</strong>는 Git처럼 데이터와 모델의 버전을 관리하여
            실험 재현성을 보장합니다.
          </p>
        </div>

        <div className="border-l-4 border-orange-500 pl-4">
          <h3 className="font-bold mb-2">DVC로 데이터 추적</h3>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`# 초기 설정
dvc init
dvc remote add -d myremote s3://mybucket/dvcstore

# 데이터 추가 및 추적
dvc add data/raw_features.parquet
git add data/raw_features.parquet.dvc .gitignore
git commit -m "Add raw features v1.0"

# 원격 스토리지에 푸시
dvc push

# 다른 환경에서 데이터 다운로드
git pull
dvc pull

# 파이프라인 정의 (dvc.yaml)
stages:
  extract:
    cmd: python extract.py
    deps:
      - extract.py
    outs:
      - data/raw_events.parquet

  transform:
    cmd: python transform.py
    deps:
      - transform.py
      - data/raw_events.parquet
    outs:
      - data/features.parquet

  train:
    cmd: python train.py
    deps:
      - train.py
      - data/features.parquet
    outs:
      - models/model.pkl
    metrics:
      - metrics.json

# 파이프라인 실행
dvc repro`}
          </pre>
        </div>
      </section>

      {/* 모델 서빙 데이터 준비 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Brain className="text-indigo-600" />
          모델 서빙을 위한 데이터 준비
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-indigo-500 pl-4">
            <h3 className="font-bold mb-2">실시간 피처 엔지니어링</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`# FastAPI로 실시간 피처 생성 + 모델 추론
from fastapi import FastAPI
from feast import FeatureStore
import joblib

app = FastAPI()
store = FeatureStore(repo_path=".")
model = joblib.load("model.pkl")

@app.post("/predict")
async def predict_churn(user_id: int):
    # 1. Online Store에서 피처 조회 (밀리초 단위)
    features_dict = store.get_online_features(
        features=[
            "user_features:total_purchases",
            "user_features:avg_purchase_value",
            "user_features:days_since_last_purchase",
        ],
        entity_rows=[{"user_id": user_id}]
    ).to_dict()

    # 2. 실시간 계산 피처 추가
    recent_activity = get_redis_value(f"user:{user_id}:recent_activity")
    features_dict['recent_clicks'] = recent_activity.get('clicks', 0)

    # 3. 모델 추론
    prediction = model.predict_proba([
        features_dict.values()
    ])[0][1]

    return {"user_id": user_id, "churn_probability": prediction}`}
            </pre>
          </div>

          <div className="border-l-4 border-teal-500 pl-4">
            <h3 className="font-bold mb-2">배치 추론 파이프라인</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`# Spark로 대규모 배치 추론
from pyspark.ml import PipelineModel

# 모델 로드
model = PipelineModel.load("s3://bucket/models/churn_model/")

# Feature Store에서 피처 로드
features_df = spark.read.parquet("s3://bucket/features/user_features/")

# 배치 추론
predictions = model.transform(features_df)

# 결과 저장
predictions.select("user_id", "probability", "prediction") \\
    .write.mode("overwrite") \\
    .parquet("s3://bucket/predictions/daily_churn_scores/")`}
            </pre>
          </div>
        </div>
      </section>

      {/* MLOps 모범 사례 */}
      <section className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Zap className="text-purple-600" />
          MLOps 데이터 파이프라인 모범 사례
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {[
            { title: 'Point-in-Time Correctness', desc: '과거 특정 시점의 피처를 정확히 재현 (시간 여행)' },
            { title: 'Train-Serve 일관성', desc: '훈련과 서빙에서 동일한 피처 정의 사용 (Feature Store)' },
            { title: '데이터 검증', desc: 'Great Expectations로 피처 품질 자동 체크' },
            { title: '버전 관리', desc: 'DVC로 데이터/모델/코드 동기화' },
            { title: '모니터링', desc: '피처 드리프트, 데이터 스큐 실시간 감지' },
            { title: '백필', desc: '피처 정의 변경 시 히스토리 데이터 재계산' }
          ].map((practice, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-bold mb-2 flex items-center gap-2">
                <CheckCircle className="text-purple-500" size={18} />
                {practice.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">{practice.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg">
        <h3 className="text-xl font-bold mb-2">다음 단계: 실전 프로젝트와 케이스 스터디</h3>
        <p className="text-gray-700 dark:text-gray-300">
          Netflix, Uber, Airbnb의 실제 데이터 플랫폼 아키텍처를 분석하고
          학습한 모든 기술을 종합하는 프로젝트를 진행합니다.
        </p>
      </div>
    </div>
  );
}
