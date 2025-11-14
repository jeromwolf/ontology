'use client';

import { useState } from 'react';
import {
  Building2, TrendingUp, Users, Zap,
  CheckCircle, Code, Database, Award
} from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter12() {
  const [selectedCompany, setSelectedCompany] = useState('netflix');

  const companies = {
    netflix: {
      name: 'Netflix',
      icon: '📺',
      color: 'red',
      scale: '2억+ 사용자, 페타바이트급 데이터',
      architecture: 'Data Mesh + Lakehouse',
      tools: ['Spark', 'Kafka', 'Druid', 'Trino', 'Iceberg']
    },
    uber: {
      name: 'Uber',
      icon: '🚗',
      color: 'black',
      scale: '하루 1억+ 트립, 실시간 데이터',
      architecture: 'Kappa + Real-time',
      tools: ['Hudi', 'Flink', 'Pinot', 'Presto', 'Kafka']
    },
    airbnb: {
      name: 'Airbnb',
      icon: '🏠',
      color: 'pink',
      scale: '400만+ 숙소, 복잡한 검색',
      architecture: 'Minerva (내부 플랫폼)',
      tools: ['Airflow', 'Hive', 'Spark', 'Druid', 'Superset']
    }
  };

  return (
    <div className="space-y-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">실전 프로젝트와 케이스 스터디</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          빅테크 기업의 데이터 플랫폼 아키텍처 분석과 실전 프로젝트
        </p>
      </div>

      {/* 회사 선택 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Building2 className="text-blue-600" />
          빅테크 데이터 플랫폼 아키텍처
        </h2>

        <div className="flex flex-wrap gap-2 mb-6">
          {Object.keys(companies).map((key) => (
            <button
              key={key}
              onClick={() => setSelectedCompany(key)}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                selectedCompany === key
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              {companies[key].icon} {companies[key].name}
            </button>
          ))}
        </div>

        <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
          <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
            {companies[selectedCompany].icon} {companies[selectedCompany].name}
          </h3>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold mb-2">규모</h4>
              <p className="text-sm bg-gray-50 dark:bg-gray-700 p-3 rounded">
                {companies[selectedCompany].scale}
              </p>

              <h4 className="font-bold mb-2 mt-4">아키텍처</h4>
              <p className="text-sm bg-gray-50 dark:bg-gray-700 p-3 rounded">
                {companies[selectedCompany].architecture}
              </p>
            </div>

            <div>
              <h4 className="font-bold mb-2">기술 스택</h4>
              <div className="flex flex-wrap gap-2">
                {companies[selectedCompany].tools.map((tool, idx) => (
                  <span key={idx} className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-full text-sm">
                    {tool}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Netflix 케이스 스터디 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          📺 Netflix - Data Mesh 아키텍처
        </h2>

        <div className="space-y-4">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
            <h3 className="font-bold mb-2">도메인별 데이터 소유권</h3>
            <p className="text-sm mb-3">
              Netflix는 중앙화된 데이터 팀 대신, 각 도메인 팀이 자신의 데이터를 소유하고 관리합니다.
            </p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-red-500 mt-0.5" size={16} />
                <span><strong>Streaming 팀</strong>: 시청 기록, 버퍼링 이벤트</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-red-500 mt-0.5" size={16} />
                <span><strong>Recommendation 팀</strong>: 추천 알고리즘, A/B 테스트</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-red-500 mt-0.5" size={16} />
                <span><strong>Content 팀</strong>: 메타데이터, 라이선스 정보</span>
              </li>
            </ul>
          </div>

          <div className="border-l-4 border-red-500 pl-4">
            <h3 className="font-bold mb-2">핵심 기술</h3>
            <ul className="space-y-2 text-sm">
              <li><strong>Keystone</strong> - 데이터 파이프라인 DSL (Airflow 대체)</li>
              <li><strong>Metacat</strong> - 통합 메타데이터 카탈로그</li>
              <li><strong>Iceberg</strong> - 대규모 테이블 포맷 (Apache 기부)</li>
              <li><strong>Druid</strong> - 실시간 OLAP 데이터베이스</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Uber 케이스 스터디 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          🚗 Uber - 실시간 데이터 플랫폼
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Apache Hudi 개발 배경</h3>
            <p className="text-sm mb-3">
              Uber는 하루 1억 건 이상의 트립 데이터를 거의 실시간으로 처리하기 위해
              Apache Hudi를 개발했습니다.
            </p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-gray-600 mt-0.5" size={16} />
                <span><strong>Upsert 지원</strong>: 레코드 업데이트/삭제 (GDPR 준수)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-gray-600 mt-0.5" size={16} />
                <span><strong>증분 처리</strong>: 변경된 데이터만 읽기 (효율성)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-gray-600 mt-0.5" size={16} />
                <span><strong>타임 트래블</strong>: 과거 스냅샷 조회</span>
              </li>
            </ul>
          </div>

          <div className="border-l-4 border-gray-500 pl-4">
            <h3 className="font-bold mb-2">실시간 파이프라인</h3>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`Kafka (트립 이벤트)
  ↓
Flink (실시간 집계)
  ↓
Hudi (Lakehouse 저장)
  ↓
Pinot (실시간 분석 쿼리)
  ↓
대시보드 (운전자/승객 메트릭)`}
            </pre>
          </div>
        </div>
      </section>

      {/* Airbnb 케이스 스터디 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          🏠 Airbnb - Minerva 데이터 플랫폼
        </h2>

        <div className="space-y-4">
          <div className="bg-pink-50 dark:bg-pink-900/20 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Minerva - 통합 데이터 플랫폼</h3>
            <p className="text-sm mb-3">
              Airbnb는 Hive, Presto, Spark를 통합하는 Minerva 플랫폼을 구축하여
              데이터 민주화를 실현했습니다.
            </p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-pink-500 mt-0.5" size={16} />
                <span><strong>Dataportal</strong>: 모든 테이블 검색 및 탐색</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-pink-500 mt-0.5" size={16} />
                <span><strong>Superset</strong>: 셀프 서비스 BI 도구 (오픈소스 기여)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-pink-500 mt-0.5" size={16} />
                <span><strong>Airflow</strong>: 10만+ DAG 관리 (최초 개발사)</span>
              </li>
            </ul>
          </div>

          <div className="border-l-4 border-pink-500 pl-4">
            <h3 className="font-bold mb-2">데이터 품질 프레임워크</h3>
            <p className="text-sm mb-2">
              Airbnb의 Midas 프레임워크는 모든 테이블에 자동으로 데이터 품질 체크를 적용합니다.
            </p>
            <ul className="space-y-1 text-sm">
              <li>• Freshness: 최신 파티션이 SLA 내에 도착했는지</li>
              <li>• Completeness: 예상 레코드 수와 일치하는지</li>
              <li>• Accuracy: 통계적 이상치 감지</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 실전 프로젝트 아이디어 */}
      <section className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Code className="text-blue-600" />
          실전 프로젝트 아이디어
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {[
            {
              title: '실시간 이커머스 추천 시스템',
              desc: 'Kafka + Flink + Feature Store로 실시간 상품 추천',
              stack: ['Kafka', 'Flink', 'Feast', 'Redis', 'FastAPI']
            },
            {
              title: '주식 시장 데이터 레이크하우스',
              desc: 'Delta Lake로 주식 데이터 수집/분석/백테스팅',
              stack: ['Databricks', 'Delta Lake', 'Airflow', 'Superset']
            },
            {
              title: '헬스케어 데이터 파이프라인',
              desc: '환자 데이터 ETL + 개인정보 보호 (HIPAA 준수)',
              stack: ['dbt', 'Great Expectations', 'Snowflake', 'Redash']
            },
            {
              title: '소셜 미디어 감성 분석',
              desc: 'Twitter API + NLP + 실시간 대시보드',
              stack: ['Kafka', 'Spark', 'Hugging Face', 'Grafana']
            }
          ].map((project, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-bold mb-2">{project.title}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{project.desc}</p>
              <div className="flex flex-wrap gap-1">
                {project.stack.map((tech, i) => (
                  <span key={i} className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded">
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 학습 완료 */}
      <section className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-8 rounded-xl text-center">
        <div className="flex justify-center mb-4">
          <Award className="text-green-600" size={64} />
        </div>
        <h2 className="text-3xl font-bold mb-4">🎉 데이터 엔지니어링 마스터!</h2>
        <p className="text-xl mb-6">
          12개 챕터를 모두 완료하셨습니다. 이제 현대적인 데이터 플랫폼을 설계하고 구축할 수 있습니다!
        </p>

        <div className="grid md:grid-cols-3 gap-4 mt-8">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-2">습득한 핵심 기술</h3>
            <ul className="text-sm space-y-1 text-left">
              <li>• EDA & 데이터 분석</li>
              <li>• ETL/ELT 파이프라인</li>
              <li>• 실시간 스트림 처리</li>
              <li>• 클라우드 데이터 플랫폼</li>
              <li>• 데이터 품질 & 거버넌스</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-2">숙련된 도구</h3>
            <ul className="text-sm space-y-1 text-left">
              <li>• Apache Spark & Airflow</li>
              <li>• Snowflake & BigQuery</li>
              <li>• dbt & Great Expectations</li>
              <li>• Kafka & Flink</li>
              <li>• Feast & DVC</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-bold mb-2">다음 단계</h3>
            <ul className="text-sm space-y-1 text-left">
              <li>• 실전 프로젝트 구축</li>
              <li>• 오픈소스 기여</li>
              <li>• AWS/GCP 자격증</li>
              <li>• 데이터 팀 리딩</li>
              <li>• 아키텍처 설계</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 데이터 플랫폼 & 도구',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Apache Airflow Documentation',
                url: 'https://airflow.apache.org/docs/',
                description: '워크플로우 오케스트레이션 플랫폼 공식 문서',
                year: 2024
              },
              {
                title: 'dbt (Data Build Tool)',
                url: 'https://docs.getdbt.com/',
                description: 'SQL 기반 데이터 변환 도구 - Analytics Engineering',
                year: 2024
              },
              {
                title: 'Apache Spark Documentation',
                url: 'https://spark.apache.org/docs/latest/',
                description: '대규모 데이터 처리 엔진 공식 가이드',
                year: 2024
              },
              {
                title: 'Apache Kafka Documentation',
                url: 'https://kafka.apache.org/documentation/',
                description: '분산 스트리밍 플랫폼 공식 문서',
                year: 2024
              },
              {
                title: 'Snowflake Documentation',
                url: 'https://docs.snowflake.com/',
                description: '클라우드 데이터 웨어하우스 플랫폼',
                year: 2024
              }
            ]
          },
          {
            title: '📖 핵심 리소스 & 서적',
            icon: 'research' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Designing Data-Intensive Applications',
                url: 'https://dataintensive.net/',
                description: 'Martin Kleppmann 저 - 데이터 시스템 설계 바이블',
                year: 2017
              },
              {
                title: 'Data Engineering Cookbook',
                url: 'https://github.com/andkret/Cookbook',
                description: 'Andreas Kretz 저 - 실전 데이터 엔지니어링 가이드',
                year: 2021
              },
              {
                title: 'Modern Data Stack',
                url: 'https://www.moderndatastack.xyz/',
                description: '최신 데이터 도구 및 아키텍처 트렌드',
                year: 2024
              },
              {
                title: 'The Data Warehouse Toolkit',
                url: 'https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/',
                description: 'Ralph Kimball 저 - 차원 모델링 고전',
                year: 2013
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 프레임워크',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Prefect',
                url: 'https://docs.prefect.io/',
                description: '현대적 워크플로우 오케스트레이션 플랫폼',
                year: 2024
              },
              {
                title: 'Dagster',
                url: 'https://docs.dagster.io/',
                description: '데이터 오케스트레이션 플랫폼 - Asset 기반 접근',
                year: 2024
              },
              {
                title: 'Great Expectations',
                url: 'https://docs.greatexpectations.io/',
                description: '데이터 품질 검증 프레임워크',
                year: 2024
              },
              {
                title: 'Fivetran',
                url: 'https://fivetran.com/docs',
                description: '자동화된 데이터 복제 플랫폼',
                year: 2024
              },
              {
                title: 'Airbyte',
                url: 'https://docs.airbyte.com/',
                description: '오픈소스 EL(T) 플랫폼 - 350+ 커넥터',
                year: 2024
              }
            ]
          }
        ]}
      />
    </div>
  );
}
