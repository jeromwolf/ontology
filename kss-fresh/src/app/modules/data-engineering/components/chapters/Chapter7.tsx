'use client';

import { useState } from 'react';
import {
  Shield, CheckCircle, AlertTriangle, XCircle,
  Database, GitBranch, FileCheck, Activity,
  TrendingUp, Settings, Users, Lock,
  Search, Filter, Award, Zap
} from 'lucide-react';

export default function Chapter7() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedFramework, setSelectedFramework] = useState('great-expectations');

  return (
    <div className="space-y-8">
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">데이터 품질과 거버넌스</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          데이터의 신뢰성을 보장하고 조직 전체의 데이터 관리 체계 구축하기
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Zap className="text-green-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 품질 프레임워크 이해</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Great Expectations, dbt tests 활용법</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 거버넌스 체계 수립</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">정책, 역할, 책임 정의</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 계보(Lineage) 추적</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">데이터 흐름과 의존성 파악</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 카탈로그 구축</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">메타데이터 관리와 검색</p>
            </div>
          </div>
        </div>
      </div>

      {/* 데이터 품질의 6가지 차원 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Award className="text-blue-600" />
          데이터 품질의 6가지 차원
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            {
              title: 'Accuracy (정확성)',
              icon: <CheckCircle className="text-green-500" />,
              desc: '데이터가 현실을 올바르게 반영',
              example: '고객 이메일 주소가 실제 존재하는 주소인지 검증'
            },
            {
              title: 'Completeness (완전성)',
              icon: <Database className="text-blue-500" />,
              desc: '필수 필드가 모두 채워져 있는지',
              example: 'NOT NULL 제약조건, 필수 컬럼 누락 체크'
            },
            {
              title: 'Consistency (일관성)',
              icon: <GitBranch className="text-purple-500" />,
              desc: '여러 소스 간 데이터 일치',
              example: 'CRM과 ERP 시스템의 고객 정보 동기화'
            },
            {
              title: 'Timeliness (적시성)',
              icon: <Activity className="text-orange-500" />,
              desc: '데이터가 최신 상태로 유지',
              example: '실시간 재고 데이터, SLA 기반 업데이트 주기'
            },
            {
              title: 'Validity (유효성)',
              icon: <Shield className="text-indigo-500" />,
              desc: '데이터 형식과 규칙 준수',
              example: '날짜 형식, 이메일 정규식, enum 값 검증'
            },
            {
              title: 'Uniqueness (고유성)',
              icon: <Lock className="text-red-500" />,
              desc: '중복 데이터 없음',
              example: 'Primary Key 제약, 중복 레코드 탐지'
            }
          ].map((dimension, idx) => (
            <div key={idx} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-2">
                {dimension.icon}
                <h3 className="font-bold">{dimension.title}</h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{dimension.desc}</p>
              <p className="text-xs bg-gray-100 dark:bg-gray-700 p-2 rounded">💡 {dimension.example}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Great Expectations 프레임워크 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <FileCheck className="text-purple-600" />
          Great Expectations - 데이터 품질 검증 프레임워크
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg mb-4">
          <p className="text-sm">
            <strong>Great Expectations</strong>는 Python 기반의 오픈소스 데이터 품질 프레임워크로,
            데이터 파이프라인에 자동화된 테스트를 추가할 수 있습니다.
          </p>
        </div>

        <div className="space-y-4">
          <div className="border-l-4 border-purple-500 pl-4">
            <h3 className="font-bold mb-2">1. Expectation 정의</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
{`import great_expectations as gx

# Expectation Suite 생성
context = gx.get_context()
suite = context.create_expectation_suite("user_data_suite")

# 컬럼 존재 확인
suite.add_expectation(
    gx.expectations.ExpectColumnToExist(column="email")
)

# 값이 NULL이 아님을 확인
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
)

# 값이 특정 범위 내에 있는지 확인
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="age",
        min_value=0,
        max_value=120
    )
)

# 이메일 형식 검증
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToMatchRegex(
        column="email",
        regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    )
)`}
            </pre>
          </div>

          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="font-bold mb-2">2. 데이터 검증 실행</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
{`# Batch로 데이터 로드
batch = context.get_batch(
    datasource_name="my_datasource",
    data_asset_name="users_table"
)

# Expectation Suite 실행
results = batch.validate(expectation_suite=suite)

# 결과 확인
if results["success"]:
    print("✅ 모든 데이터 품질 테스트 통과!")
else:
    print("❌ 데이터 품질 이슈 발견:")
    for result in results["results"]:
        if not result["success"]:
            print(f"  - {result['expectation_config']['expectation_type']}")
            print(f"    실패: {result['result']}")`}
            </pre>
          </div>

          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="font-bold mb-2">3. 데이터 문서 자동 생성</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
{`# HTML 문서 생성
context.build_data_docs()

# 생성된 문서는 브라우저에서 확인 가능
# - 모든 Expectation 목록
# - 검증 결과 시각화
# - 데이터 프로파일링 리포트`}
            </pre>
          </div>
        </div>
      </section>

      {/* dbt Tests */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Settings className="text-orange-600" />
          dbt Tests - 데이터 변환 품질 보장
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="border border-orange-200 dark:border-orange-800 rounded-lg p-4">
            <h3 className="font-bold mb-2 flex items-center gap-2">
              <CheckCircle className="text-orange-500" />
              Generic Tests (내장 테스트)
            </h3>
            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# schema.yml
models:
  - name: customers
    columns:
      - name: customer_id
        tests:
          - unique
          - not_null

      - name: email
        tests:
          - unique
          - not_null
          - email_format

      - name: created_at
        tests:
          - not_null
          - recent_date:
              interval: 7
              datepart: day`}
            </pre>
          </div>

          <div className="border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <h3 className="font-bold mb-2 flex items-center gap-2">
              <FileCheck className="text-blue-500" />
              Singular Tests (커스텀 SQL)
            </h3>
            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`-- tests/assert_revenue_positive.sql
SELECT
    order_id,
    total_amount
FROM {{ ref('orders') }}
WHERE total_amount <= 0

-- 이 쿼리가 결과를 반환하면 테스트 실패
-- (양수여야 할 금액이 0 이하인 경우)`}
            </pre>
          </div>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
          <h3 className="font-bold mb-2">dbt test 실행</h3>
          <pre className="bg-gray-900 text-gray-100 p-3 rounded text-sm">
{`# 모든 테스트 실행
dbt test

# 특정 모델만 테스트
dbt test --select customers

# 실패한 테스트만 재실행
dbt test --select result:fail`}
          </pre>
        </div>
      </section>

      {/* 데이터 계보 (Data Lineage) */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <GitBranch className="text-indigo-600" />
          데이터 계보 (Data Lineage) 추적
        </h2>

        <div className="mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            데이터 계보는 데이터의 출처부터 최종 목적지까지의 전체 여정을 시각화하여,
            데이터 흐름과 변환 과정을 이해할 수 있게 합니다.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h3 className="font-bold mb-2">📊 컬럼 레벨 계보</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              특정 컬럼이 어디서 왔고, 어떻게 변환되었는지 추적
            </p>
            <p className="text-xs mt-2 bg-white dark:bg-gray-700 p-2 rounded">
              예: users.email → transformed_users.email_normalized → analytics.user_metrics.email_domain
            </p>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h3 className="font-bold mb-2">🔄 테이블 레벨 계보</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              테이블 간의 의존성과 데이터 플로우 파악
            </p>
            <p className="text-xs mt-2 bg-white dark:bg-gray-700 p-2 rounded">
              예: raw_events → staging_events → fact_user_sessions → dashboard
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h3 className="font-bold mb-2">🎯 영향도 분석</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              스키마 변경 시 영향 받는 downstream 파악
            </p>
            <p className="text-xs mt-2 bg-white dark:bg-gray-700 p-2 rounded">
              예: orders 테이블 변경 → 15개 downstream 모델 영향 받음
            </p>
          </div>
        </div>

        <div className="border-l-4 border-indigo-500 pl-4">
          <h3 className="font-bold mb-2">주요 데이터 계보 도구</h3>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-indigo-600 font-bold">•</span>
              <span><strong>dbt</strong> - dbt docs로 자동 계보 시각화, DAG 그래프 제공</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-indigo-600 font-bold">•</span>
              <span><strong>Apache Atlas</strong> - 엔터프라이즈 메타데이터 관리 및 계보 추적</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-indigo-600 font-bold">•</span>
              <span><strong>OpenLineage</strong> - 오픈소스 계보 표준, Airflow/Spark 통합</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-indigo-600 font-bold">•</span>
              <span><strong>Marquez</strong> - OpenLineage 기반 계보 수집 및 시각화</span>
            </li>
          </ul>
        </div>
      </section>

      {/* 데이터 거버넌스 프레임워크 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Users className="text-teal-600" />
          데이터 거버넌스 프레임워크
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-bold mb-3 text-lg">핵심 구성 요소</h3>
            <div className="space-y-3">
              <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <h4 className="font-semibold mb-1">1. 데이터 정책 (Data Policies)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  데이터 접근, 사용, 보관, 삭제에 대한 규칙 정의
                </p>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <h4 className="font-semibold mb-1">2. 역할과 책임 (RACI Matrix)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  데이터 소유자, 관리자, 사용자의 역할 명확화
                </p>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <h4 className="font-semibold mb-1">3. 데이터 카탈로그</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  모든 데이터 자산의 메타데이터 중앙 관리
                </p>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <h4 className="font-semibold mb-1">4. 규정 준수 (Compliance)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  GDPR, CCPA, HIPAA 등 법규 준수 보장
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-bold mb-3 text-lg">데이터 카탈로그 도구</h3>
            <div className="space-y-3">
              <div className="border border-blue-200 dark:border-blue-800 rounded-lg p-3">
                <h4 className="font-semibold text-blue-600 mb-1">Amundsen (Lyft)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  오픈소스 데이터 검색 엔진, 테이블/컬럼 설명, 사용 통계
                </p>
                <p className="text-xs bg-blue-50 dark:bg-blue-900/20 p-2 rounded">
                  Python, React 기반 / Elasticsearch 검색
                </p>
              </div>

              <div className="border border-green-200 dark:border-green-800 rounded-lg p-3">
                <h4 className="font-semibold text-green-600 mb-1">DataHub (LinkedIn)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  메타데이터 플랫폼, 계보 추적, 데이터 품질 통합
                </p>
                <p className="text-xs bg-green-50 dark:bg-green-900/20 p-2 rounded">
                  Java, React / Kafka 기반 실시간 업데이트
                </p>
              </div>

              <div className="border border-purple-200 dark:border-purple-800 rounded-lg p-3">
                <h4 className="font-semibold text-purple-600 mb-1">Alation (Commercial)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  엔터프라이즈 데이터 카탈로그, AI 기반 추천
                </p>
                <p className="text-xs bg-purple-50 dark:bg-purple-900/20 p-2 rounded">
                  협업 기능, 비즈니스 용어집, 데이터 품질 점수
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 체크리스트 */}
      <section className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <CheckCircle className="text-teal-600" />
          데이터 품질 & 거버넌스 체크리스트
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-3">데이터 품질</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>모든 테이블에 primary key 정의</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>NULL 허용 여부 명시적 설정</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>외래 키로 참조 무결성 보장</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>dbt tests 또는 Great Expectations 통합</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>데이터 품질 메트릭 대시보드 구축</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-3">데이터 거버넌스</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5 flex-shrink-0" size={16} />
                <span>데이터 소유자 명확히 지정</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5 flex-shrink-0" size={16} />
                <span>민감 데이터 분류 및 암호화</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5 flex-shrink-0" size={16} />
                <span>접근 제어 정책 (RBAC) 구현</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5 flex-shrink-0" size={16} />
                <span>데이터 계보 자동 추적 시스템</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-blue-500 mt-0.5 flex-shrink-0" size={16} />
                <span>정기적인 데이터 품질 리뷰</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 다음 단계 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg">
        <h3 className="text-xl font-bold mb-2">다음 단계: 클라우드 데이터 플랫폼</h3>
        <p className="text-gray-700 dark:text-gray-300">
          데이터 품질과 거버넌스 체계를 갖춘 후, Snowflake, BigQuery, Databricks 같은
          클라우드 데이터 플랫폼을 활용하여 확장 가능한 데이터 인프라를 구축하는 방법을 학습합니다.
        </p>
      </div>
    </div>
  );
}
