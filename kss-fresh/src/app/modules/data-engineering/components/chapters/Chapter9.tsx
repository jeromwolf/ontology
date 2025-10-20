'use client';

import { useState } from 'react';
import {
  GitBranch, Play, Pause, Calendar,
  AlertCircle, CheckCircle, Clock, Workflow,
  Code, Terminal, Activity, Zap
} from 'lucide-react';

export default function Chapter9() {
  const [selectedTool, setSelectedTool] = useState('airflow');

  return (
    <div className="space-y-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">데이터 오케스트레이션</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          Airflow, Dagster, Prefect로 데이터 파이프라인 스케줄링 및 모니터링 자동화
        </p>
      </div>

      {/* Apache Airflow */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Workflow className="text-blue-600" />
          Apache Airflow - 가장 인기 있는 워크플로우 엔진
        </h2>

        <div className="space-y-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg mb-4">
            <p className="text-sm">
              <strong>Airflow</strong>는 Python 코드로 DAG(Directed Acyclic Graph)를 정의하여
              복잡한 데이터 파이프라인을 스케줄링하고 모니터링합니다.
            </p>
          </div>

          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="font-bold mb-2">기본 DAG 작성</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_user_analytics',
    default_args=default_args,
    description='사용자 행동 분석 파이프라인',
    schedule_interval='0 2 * * *',  # 매일 오전 2시
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['analytics', 'daily'],
) as dag:

    # Task 1: S3에서 데이터 추출
    extract_data = PythonOperator(
        task_id='extract_from_s3',
        python_callable=extract_user_events,
    )

    # Task 2: Spark로 데이터 변환
    transform_data = BashOperator(
        task_id='transform_with_spark',
        bash_command='spark-submit /scripts/transform_users.py',
    )

    # Task 3: Snowflake에 로드
    load_data = PythonOperator(
        task_id='load_to_snowflake',
        python_callable=load_to_warehouse,
    )

    # Task 4: 데이터 품질 검증
    validate_data = PythonOperator(
        task_id='validate_data_quality',
        python_callable=run_great_expectations,
    )

    # DAG 의존성 정의
    extract_data >> transform_data >> load_data >> validate_data`}
            </pre>
          </div>

          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="font-bold mb-2">고급 기능: Dynamic Task 생성</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`from airflow.decorators import task

@task
def get_table_list():
    return ['users', 'orders', 'products', 'reviews']

@task
def process_table(table_name):
    print(f"Processing {table_name}")
    # ETL 로직

with DAG('dynamic_pipeline', ...) as dag:
    tables = get_table_list()

    # 동적으로 Task 생성
    process_tasks = process_table.expand(table_name=tables)

    # TaskFlow API로 의존성 자동 처리
    process_tasks`}
            </pre>
          </div>
        </div>
      </section>

      {/* Dagster */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Code className="text-purple-600" />
          Dagster - 데이터 중심 오케스트레이션
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg mb-4">
          <p className="text-sm">
            <strong>Dagster</strong>는 데이터 자산(Asset) 중심으로 파이프라인을 정의하며,
            타입 체크와 데이터 품질을 1급 시민으로 취급합니다.
          </p>
        </div>

        <div className="border-l-4 border-purple-500 pl-4">
          <h3 className="font-bold mb-2">Software-Defined Assets</h3>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`from dagster import asset, AssetExecutionContext
import pandas as pd

@asset
def raw_users(context: AssetExecutionContext) -> pd.DataFrame:
    """S3에서 원시 사용자 데이터를 추출"""
    df = pd.read_csv("s3://bucket/users.csv")
    context.log.info(f"Loaded {len(df)} users")
    return df

@asset
def cleaned_users(raw_users: pd.DataFrame) -> pd.DataFrame:
    """사용자 데이터 정제"""
    df = raw_users.dropna()
    df['email'] = df['email'].str.lower()
    return df

@asset(
    deps=[cleaned_users],
    metadata={"partition_expr": "date"}
)
def user_metrics():
    """정제된 데이터로 메트릭 계산"""
    # SQL 쿼리 실행
    return execute_dbt_model("user_daily_metrics")

# 자동 의존성 그래프 생성: raw_users → cleaned_users → user_metrics`}
          </pre>
        </div>
      </section>

      {/* Prefect */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Activity className="text-indigo-600" />
          Prefect - 현대적인 워크플로우 엔진
        </h2>

        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg mb-4">
          <p className="text-sm">
            <strong>Prefect</strong>는 네거티브 엔지니어링(설정보다 코드)을 강조하며,
            하이브리드 실행 모델과 강력한 에러 처리를 제공합니다.
          </p>
        </div>

        <div className="border-l-4 border-indigo-500 pl-4">
          <h3 className="font-bold mb-2">Prefect Flow 작성</h3>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def extract_data(source: str):
    """데이터 추출 (캐싱 1시간)"""
    return fetch_from_source(source)

@task
def transform_data(raw_data):
    """데이터 변환"""
    return clean_and_transform(raw_data)

@task
def load_data(transformed_data):
    """데이터 로드"""
    warehouse.insert(transformed_data)

@flow(name="ETL Pipeline")
def etl_pipeline(source: str = "s3://bucket/data"):
    raw = extract_data(source)
    transformed = transform_data(raw)
    load_data(transformed)

# Flow 실행
if __name__ == "__main__":
    etl_pipeline.serve(
        name="daily-etl",
        cron="0 2 * * *"  # 매일 오전 2시
    )`}
          </pre>
        </div>
      </section>

      {/* 비교 표 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6">도구 비교</h2>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-100 dark:bg-gray-700">
              <tr>
                <th className="p-3 text-left">기능</th>
                <th className="p-3 text-left">Airflow</th>
                <th className="p-3 text-left">Dagster</th>
                <th className="p-3 text-left">Prefect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              <tr>
                <td className="p-3 font-semibold">학습 곡선</td>
                <td className="p-3">중간 (DAG 개념 필요)</td>
                <td className="p-3">높음 (Asset 모델 이해)</td>
                <td className="p-3">낮음 (Python 데코레이터)</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">UI</td>
                <td className="p-3">✅ 강력한 웹 UI</td>
                <td className="p-3">✅ 현대적 UI</td>
                <td className="p-3">✅ 클라우드 UI</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">데이터 품질</td>
                <td className="p-3">외부 통합 필요</td>
                <td className="p-3">✅ 내장 (Type System)</td>
                <td className="p-3">외부 통합</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">스케일</td>
                <td className="p-3">✅ 대규모 운영 검증됨</td>
                <td className="p-3">중소 규모 적합</td>
                <td className="p-3">✅ 하이브리드 실행</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">커뮤니티</td>
                <td className="p-3">🥇 가장 큼</td>
                <td className="p-3">성장 중</td>
                <td className="p-3">성장 중</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 모범 사례 */}
      <section className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Zap className="text-blue-600" />
          오케스트레이션 모범 사례
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {[
            { title: '멱등성(Idempotency)', desc: '같은 Task를 여러 번 실행해도 결과가 동일하도록 설계' },
            { title: '작은 Task 단위', desc: '하나의 Task는 하나의 책임만 갖도록 분리 (재시도 최소화)' },
            { title: '실패 알림', desc: 'Slack, Email, PagerDuty로 즉시 알림 설정' },
            { title: 'SLA 모니터링', desc: 'Task 실행 시간 추적 및 SLA 위반 감지' },
            { title: '백필 전략', desc: '과거 데이터 재처리를 위한 catchup 설정' },
            { title: '환경 분리', desc: 'dev/staging/prod 환경 완전 분리' }
          ].map((practice, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-bold mb-2">{practice.title}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">{practice.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg">
        <h3 className="text-xl font-bold mb-2">다음 단계: 성능 최적화와 비용 관리</h3>
        <p className="text-gray-700 dark:text-gray-300">
          파이프라인 오케스트레이션 후, 쿼리 성능을 튜닝하고 클라우드 비용을 최적화하는 전략을 학습합니다.
        </p>
      </div>
    </div>
  );
}
