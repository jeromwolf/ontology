'use client';

import { useState } from 'react';
import { 
  Database, Server, Cloud, GitBranch, Layers,
  ArrowRight, CheckCircle, AlertCircle, Info,
  Code2, Terminal, Cpu, HardDrive,
  Network, Shield, Gauge, Users,
  ChevronRight, Play, FileText, Zap
} from 'lucide-react';

export default function Chapter1() {
  const [activeSection, setActiveSection] = useState('overview')
  const [selectedTool, setSelectedTool] = useState('spark')

  return (
    <div className="space-y-8">
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">데이터 엔지니어링 기초와 생태계</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          현대 데이터 인프라의 핵심을 이해하고 데이터 엔지니어의 역할 탐구
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Zap className="text-blue-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 엔지니어링의 정의와 중요성</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">현대 기업에서의 역할과 가치</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 파이프라인 기초 개념</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">ETL/ELT 프로세스와 아키텍처</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">주요 도구와 기술 스택 이해</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Apache Spark, Airflow, Kafka 등</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 엔지니어 커리어 패스</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">필요 스킬과 성장 전략</p>
            </div>
          </div>
        </div>
      </div>

      {/* 1. 데이터 엔지니어링이란? */}
      <section>
        <h2 className="text-3xl font-bold mb-6">1. 데이터 엔지니어링이란?</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Database className="text-blue-500" />
            정의와 핵심 역할
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <strong>데이터 엔지니어링</strong>은 원시 데이터를 비즈니스 가치로 전환하기 위한 
            인프라와 시스템을 설계, 구축, 유지하는 분야입니다. 데이터 엔지니어는 
            데이터 과학자와 분석가가 효율적으로 작업할 수 있는 기반을 제공합니다.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">수집 (Collect)</h4>
              <p className="text-sm">다양한 소스에서 데이터를 안정적으로 수집</p>
            </div>
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">변환 (Transform)</h4>
              <p className="text-sm">원시 데이터를 분석 가능한 형태로 가공</p>
            </div>
            <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">제공 (Serve)</h4>
              <p className="text-sm">사용자가 쉽게 접근할 수 있도록 데이터 제공</p>
            </div>
          </div>
        </div>

        {/* 데이터 엔지니어 vs 관련 직군 */}
        <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-900/50 dark:to-gray-800/50 p-6 rounded-xl mb-6">
          <h3 className="text-xl font-semibold mb-4">데이터 관련 직군 비교</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-300 dark:border-gray-600">
                  <th className="text-left py-3 px-4">직군</th>
                  <th className="text-left py-3 px-4">주요 역할</th>
                  <th className="text-left py-3 px-4">핵심 기술</th>
                  <th className="text-left py-3 px-4">결과물</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-3 px-4 font-semibold">데이터 엔지니어</td>
                  <td className="py-3 px-4">데이터 인프라 구축</td>
                  <td className="py-3 px-4">Python, SQL, Spark, Airflow</td>
                  <td className="py-3 px-4">데이터 파이프라인</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-3 px-4 font-semibold">데이터 과학자</td>
                  <td className="py-3 px-4">모델 개발 및 분석</td>
                  <td className="py-3 px-4">Python, R, ML, 통계</td>
                  <td className="py-3 px-4">예측 모델, 인사이트</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-3 px-4 font-semibold">데이터 분석가</td>
                  <td className="py-3 px-4">비즈니스 분석</td>
                  <td className="py-3 px-4">SQL, Excel, Tableau</td>
                  <td className="py-3 px-4">대시보드, 리포트</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-3 px-4 font-semibold">ML 엔지니어</td>
                  <td className="py-3 px-4">모델 배포 및 운영</td>
                  <td className="py-3 px-4">Python, Docker, K8s</td>
                  <td className="py-3 px-4">ML 서비스</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* 2. 데이터 파이프라인 기초 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">2. 데이터 파이프라인 기초</h2>
        
        {/* ETL vs ELT */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <ArrowRight className="text-orange-500" />
              ETL (Extract, Transform, Load)
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold text-orange-600">E</span>
                </div>
                <div>
                  <p className="font-medium">Extract (추출)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">소스에서 데이터 추출</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold text-orange-600">T</span>
                </div>
                <div>
                  <p className="font-medium">Transform (변환)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">중간 서버에서 데이터 가공</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold text-orange-600">L</span>
                </div>
                <div>
                  <p className="font-medium">Load (적재)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">타겟 시스템에 저장</p>
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
              <p className="text-sm"><strong>장점:</strong> 타겟 시스템 부하 감소, 보안성</p>
              <p className="text-sm"><strong>단점:</strong> 처리 시간, 확장성 제한</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <ArrowRight className="text-blue-500" />
              ELT (Extract, Load, Transform)
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold text-blue-600">E</span>
                </div>
                <div>
                  <p className="font-medium">Extract (추출)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">소스에서 데이터 추출</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold text-blue-600">L</span>
                </div>
                <div>
                  <p className="font-medium">Load (적재)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">원시 데이터를 즉시 적재</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold text-blue-600">T</span>
                </div>
                <div>
                  <p className="font-medium">Transform (변환)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">타겟 시스템에서 변환</p>
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm"><strong>장점:</strong> 확장성, 유연성, 빠른 로드</p>
              <p className="text-sm"><strong>단점:</strong> 강력한 타겟 시스템 필요</p>
            </div>
          </div>
        </div>

        {/* 파이프라인 아키텍처 예시 */}
        <div className="bg-gray-900 rounded-xl p-6 mb-6">
          <h3 className="text-white font-semibold mb-4">간단한 데이터 파이프라인 예제</h3>
          <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm text-gray-300">{`# Apache Airflow DAG 예제
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sales_data_pipeline',
    default_args=default_args,
    description='일일 판매 데이터 ETL 파이프라인',
    schedule_interval='@daily',
    catchup=False,
)

def extract_data(**context):
    """소스 시스템에서 데이터 추출"""
    # API 호출 또는 데이터베이스 쿼리
    raw_data = fetch_from_source()
    return raw_data

def transform_data(**context):
    """데이터 정제 및 변환"""
    raw_data = context['task_instance'].xcom_pull(task_ids='extract')
    
    # 데이터 정제
    cleaned_data = clean_nulls(raw_data)
    
    # 비즈니스 로직 적용
    transformed_data = apply_business_rules(cleaned_data)
    
    return transformed_data

# Task 정의
extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform_data,
    dag=dag,
)

load_task = PostgresOperator(
    task_id='load',
    postgres_conn_id='datawarehouse',
    sql="""
        INSERT INTO fact_sales (date, product_id, amount)
        VALUES ({{ ds }}, {{ ti.xcom_pull(task_ids='transform') }})
    """,
    dag=dag,
)

# Task 의존성 설정
extract_task >> transform_task >> load_task`}</code>
          </pre>
        </div>
      </section>

      {/* 3. 데이터 엔지니어링 기술 스택 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">3. 데이터 엔지니어링 기술 스택</h2>
        
        {/* 주요 도구 카테고리 */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
          <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Server className="text-red-500" />
              데이터 처리
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span>Apache Spark</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span>Apache Flink</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span>Apache Beam</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span>dbt (Data Build Tool)</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <GitBranch className="text-blue-500" />
              오케스트레이션
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span>Apache Airflow</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span>Dagster</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span>Prefect</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span>Luigi</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Database className="text-green-500" />
              스토리지
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span>Amazon S3</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span>HDFS</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span>Delta Lake</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span>Apache Iceberg</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Network className="text-purple-500" />
              스트리밍
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span>Apache Kafka</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span>Apache Pulsar</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span>Amazon Kinesis</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span>RabbitMQ</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-yellow-50 to-amber-50 dark:from-yellow-900/20 dark:to-amber-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Layers className="text-yellow-600" />
              데이터 웨어하우스
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-yellow-500" />
                <span>Snowflake</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-yellow-500" />
                <span>BigQuery</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-yellow-500" />
                <span>Redshift</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-yellow-500" />
                <span>Databricks</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Shield className="text-teal-500" />
              모니터링 & 품질
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span>Great Expectations</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span>Monte Carlo</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span>Datadog</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span>Prometheus</span>
              </li>
            </ul>
          </div>
        </div>

        {/* 도구 상세 설명 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold mb-4">핵심 도구 심화 학습</h3>
          <div className="flex gap-2 mb-4">
            {['spark', 'airflow', 'kafka'].map((tool) => (
              <button
                key={tool}
                onClick={() => setSelectedTool(tool)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedTool === tool
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                {tool === 'spark' && 'Apache Spark'}
                {tool === 'airflow' && 'Apache Airflow'}
                {tool === 'kafka' && 'Apache Kafka'}
              </button>
            ))}
          </div>

          {selectedTool === 'spark' && (
            <div className="space-y-4">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>Apache Spark</strong>는 대규모 데이터 처리를 위한 통합 분석 엔진입니다.
              </p>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">주요 특징:</h4>
                <ul className="space-y-1 text-sm">
                  <li>• In-memory 처리로 빠른 성능</li>
                  <li>• Batch & Stream 처리 통합</li>
                  <li>• SQL, DataFrame, ML 라이브러리 제공</li>
                  <li>• Python, Scala, Java, R 지원</li>
                </ul>
              </div>
              <pre className="bg-gray-900 text-gray-300 p-4 rounded-lg overflow-x-auto text-sm">
{`# PySpark 예제
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DataEngineering") \
    .getOrCreate()

# 데이터 읽기
df = spark.read.parquet("s3://bucket/data/")

# 변환 작업
result = df.filter(df.amount > 100) \
    .groupBy("category") \
    .agg({"amount": "sum"}) \
    .orderBy("sum(amount)", ascending=False)

# 결과 저장
result.write.mode("overwrite") \
    .parquet("s3://bucket/output/")`}</pre>
            </div>
          )}

          {selectedTool === 'airflow' && (
            <div className="space-y-4">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>Apache Airflow</strong>는 워크플로우를 프로그래밍 방식으로 작성, 스케줄링, 모니터링하는 플랫폼입니다.
              </p>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">주요 특징:</h4>
                <ul className="space-y-1 text-sm">
                  <li>• DAG 기반 워크플로우 정의</li>
                  <li>• 풍부한 웹 UI</li>
                  <li>• 확장 가능한 아키텍처</li>
                  <li>• 다양한 연결자(Operator) 제공</li>
                </ul>
              </div>
              <pre className="bg-gray-900 text-gray-300 p-4 rounded-lg overflow-x-auto text-sm">
{`# Airflow DAG 정의
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    'data_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily'
) as dag:
    
    extract = BashOperator(
        task_id='extract_data',
        bash_command='python extract.py'
    )
    
    transform = BashOperator(
        task_id='transform_data',
        bash_command='spark-submit transform.py'
    )
    
    extract >> transform`}</pre>
            </div>
          )}

          {selectedTool === 'kafka' && (
            <div className="space-y-4">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>Apache Kafka</strong>는 실시간 데이터 스트리밍을 위한 분산 이벤트 스트리밍 플랫폼입니다.
              </p>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">주요 특징:</h4>
                <ul className="space-y-1 text-sm">
                  <li>• 높은 처리량과 낮은 지연시간</li>
                  <li>• 내구성 있는 메시지 저장</li>
                  <li>• 수평적 확장 가능</li>
                  <li>• 이벤트 소싱 지원</li>
                </ul>
              </div>
              <pre className="bg-gray-900 text-gray-300 p-4 rounded-lg overflow-x-auto text-sm">
{`# Kafka Producer 예제
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode()
)

# 이벤트 전송
event = {
    'user_id': '12345',
    'action': 'purchase',
    'amount': 99.99,
    'timestamp': '2024-01-01T12:00:00'
}

producer.send('events', event)
producer.flush()`}</pre>
            </div>
          )}
        </div>
      </section>

      {/* 4. 데이터 엔지니어 커리어 패스 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">4. 데이터 엔지니어 커리어 패스</h2>
        
        {/* 스킬 매트릭스 */}
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl mb-6">
          <h3 className="text-xl font-semibold mb-4">필수 스킬 매트릭스</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3 text-indigo-700 dark:text-indigo-400">기술 스킬</h4>
              <div className="space-y-3">
                {[
                  { skill: "SQL & 데이터베이스", level: 90 },
                  { skill: "Python/Scala", level: 85 },
                  { skill: "분산 컴퓨팅 (Spark)", level: 80 },
                  { skill: "클라우드 플랫폼", level: 75 },
                  { skill: "데이터 모델링", level: 70 },
                ].map((item, idx) => (
                  <div key={idx}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">{item.skill}</span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">{item.level}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full"
                        style={{ width: `${item.level}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3 text-purple-700 dark:text-purple-400">소프트 스킬</h4>
              <div className="space-y-3">
                {[
                  { skill: "문제 해결 능력", level: 85 },
                  { skill: "커뮤니케이션", level: 80 },
                  { skill: "프로젝트 관리", level: 70 },
                  { skill: "비즈니스 이해", level: 75 },
                  { skill: "문서화", level: 65 },
                ].map((item, idx) => (
                  <div key={idx}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">{item.skill}</span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">{item.level}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                        style={{ width: `${item.level}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 경력 발전 경로 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold mb-4">경력 발전 단계</h3>
          
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-green-600 font-bold">Jr</span>
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">주니어 데이터 엔지니어 (0-2년)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  • SQL 마스터, Python 기초 • 간단한 ETL 파이프라인 구축 • 데이터 품질 체크 자동화
                </p>
                <p className="text-sm mt-2">
                  <strong>연봉:</strong> 4,000 - 6,000만원
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-blue-600 font-bold">Mid</span>
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">미드레벨 데이터 엔지니어 (2-5년)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  • 복잡한 파이프라인 설계 • 실시간 처리 구현 • 성능 최적화 및 모니터링
                </p>
                <p className="text-sm mt-2">
                  <strong>연봉:</strong> 6,000 - 9,000만원
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-purple-600 font-bold">Sr</span>
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">시니어 데이터 엔지니어 (5-10년)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  • 데이터 아키텍처 설계 • 팀 리딩 및 멘토링 • 전사 데이터 전략 수립
                </p>
                <p className="text-sm mt-2">
                  <strong>연봉:</strong> 9,000 - 13,000만원
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-orange-600 font-bold">Lead</span>
              </div>
              <div className="flex-1">
                <h4 className="font-semibold">리드/프린시플 엔지니어 (10년+)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  • 기술 의사결정 • 혁신적 솔루션 도입 • 조직 전체 기술 방향 설정
                </p>
                <p className="text-sm mt-2">
                  <strong>연봉:</strong> 13,000만원 이상
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 실습 프로젝트 */}
      <section className="mt-12">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">🚀 첫 번째 데이터 파이프라인 구축하기</h2>
          <p className="mb-6">
            이제 기초를 배웠으니, 실제로 간단한 데이터 파이프라인을 구축해봅시다.
            로컬 환경에서 Docker를 사용해 Airflow를 설치하고, 첫 DAG를 만들어보세요.
          </p>
          <div className="flex gap-4">
            <button className="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              실습 시작하기
            </button>
            <button className="bg-blue-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-400 transition-colors">
              환경 설정 가이드
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}