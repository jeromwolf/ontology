'use client';

import References from '@/components/common/References';

// Chapter 4: Google Cloud Platform
export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">Google Cloud Platform (GCP) 소개</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          GCP는 Google이 검색, YouTube, Gmail 등을 운영하는 데 사용하는 동일한 인프라를 
          클라우드 서비스로 제공합니다. 데이터 분석과 머신러닝 분야에서 강력한 서비스를 제공합니다.
        </p>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">GCP 핵심 강점</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-red-600 dark:text-red-400">•</span>
              <div><strong>데이터 분석 리더십</strong>: BigQuery, Dataflow, Looker 등 최고 수준</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-600 dark:text-red-400">•</span>
              <div><strong>AI/ML 혁신</strong>: TensorFlow, Vertex AI, AutoML</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-600 dark:text-red-400">•</span>
              <div><strong>네트워크 성능</strong>: Google의 글로벌 프라이빗 네트워크 활용</div>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">핵심 컴퓨팅 서비스</h2>
        <div className="grid gap-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border-l-4 border-red-500">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3 text-xl">
              Compute Engine
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              GCP의 IaaS 서비스로, AWS EC2 및 Azure VM과 동일한 개념입니다.
              고성능 가상 머신을 제공하며, 지속 사용 할인이 자동으로 적용됩니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">머신 패밀리:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• <strong>E2</strong>: 비용 최적화 (Spot VM 지원)</li>
                    <li>• <strong>N2, N2D</strong>: 범용 워크로드</li>
                    <li>• <strong>C2, C2D</strong>: 컴퓨팅 집약적</li>
                    <li>• <strong>M2</strong>: 메모리 최적화 (최대 12TB RAM)</li>
                    <li>• <strong>A2</strong>: GPU 가속 (NVIDIA A100)</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">고유 기능:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 지속 사용 할인: 자동 적용 (최대 30%)</li>
                    <li>• 커스텀 머신 타입: CPU/메모리 자유 조합</li>
                    <li>• 라이브 마이그레이션: 무중단 유지보수</li>
                    <li>• Sole-tenant 노드: 물리 서버 전용 사용</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3 text-xl">
              Cloud Run (서버리스 컨테이너)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              완전 관리형 서버리스 플랫폼으로, 컨테이너를 자동으로 확장하여 실행합니다.
              AWS Lambda와 Fargate의 장점을 결합한 서비스입니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <strong className="text-gray-900 dark:text-white block mb-2">주요 특징:</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 모든 컨테이너 지원 (Docker 이미지)</li>
                  <li>• 0에서 N까지 자동 확장</li>
                  <li>• 요청 기반 과금 (100ms 단위)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <strong className="text-gray-900 dark:text-white block mb-2">사용 사례:</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• API 백엔드, 웹 애플리케이션</li>
                  <li>• 이벤트 기반 처리</li>
                  <li>• 마이크로서비스</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3 text-xl">
              Google Kubernetes Engine (GKE)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Google이 Kubernetes를 만들었기 때문에 GKE는 가장 성숙한 관리형 Kubernetes 서비스입니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">Autopilot 모드:</strong>
                  <p className="text-gray-700 dark:text-gray-300">노드 관리 불필요, Pod 단위 과금</p>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">Standard 모드:</strong>
                  <p className="text-gray-700 dark:text-gray-300">완전한 클러스터 제어, 노드 관리</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">스토리지 서비스</h2>
        <div className="grid gap-6">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-3 text-xl">
              Cloud Storage
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              AWS S3 및 Azure Blob Storage와 동일한 객체 스토리지입니다.
              통합 버킷으로 모든 스토리지 클래스를 하나의 버킷에서 관리할 수 있습니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <strong className="text-gray-900 dark:text-white block mb-3">스토리지 클래스:</strong>
              <div className="grid md:grid-cols-4 gap-3 text-sm">
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <strong className="text-gray-900 dark:text-white">Standard</strong>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">자주 액세스</p>
                  <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">$0.020/GB</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <strong className="text-gray-900 dark:text-white">Nearline</strong>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">월 1회 액세스</p>
                  <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">$0.010/GB</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <strong className="text-gray-900 dark:text-white">Coldline</strong>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">분기 1회 액세스</p>
                  <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">$0.004/GB</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <strong className="text-gray-900 dark:text-white">Archive</strong>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">연 1회 액세스</p>
                  <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">$0.0012/GB</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Persistent Disk</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Compute Engine VM용 블록 스토리지 (AWS EBS, Azure Disk와 동일)
            </p>
            <div className="grid md:grid-cols-3 gap-3 text-sm">
              <div className="bg-gray-50 dark:bg-gray-700 rounded p-2">
                <strong className="text-gray-900 dark:text-white">Standard PD</strong>
                <p className="text-gray-600 dark:text-gray-400 text-xs">HDD, 저비용</p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded p-2">
                <strong className="text-gray-900 dark:text-white">Balanced PD</strong>
                <p className="text-gray-600 dark:text-gray-400 text-xs">SSD, 균형잡힌 성능</p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded p-2">
                <strong className="text-gray-900 dark:text-white">Extreme PD</strong>
                <p className="text-gray-600 dark:text-gray-400 text-xs">최고 성능</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">데이터베이스 & 데이터 분석</h2>
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6 border-l-4 border-blue-500">
            <h3 className="font-bold text-blue-900 dark:text-blue-100 mb-3 text-xl">
              BigQuery (데이터 웨어하우스)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              GCP의 대표 서비스로, 페타바이트급 데이터를 초고속으로 분석할 수 있는 서버리스 데이터 웨어하우스입니다.
              SQL 쿼리로 수 초 내에 테라바이트 데이터를 처리합니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded p-4 text-sm">
                <strong className="text-gray-900 dark:text-white block mb-2">주요 특징:</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 서버리스: 인프라 관리 불필요</li>
                  <li>• 실시간 분석: 스트리밍 데이터 지원</li>
                  <li>• ML 통합: BigQuery ML로 SQL로 ML 모델 학습</li>
                  <li>• BI Engine: 인메모리 분석 엔진</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4 text-sm">
                <strong className="text-gray-900 dark:text-white block mb-2">요금 모델:</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• On-Demand: 스캔한 데이터 양으로 과금</li>
                  <li>• Flat-Rate: 고정 슬롯 예약</li>
                  <li>• 스토리지: $0.020/GB (활성), $0.010/GB (장기)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-cyan-200 dark:border-cyan-700">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-cyan-100 dark:bg-cyan-900/30 rounded-lg flex items-center justify-center">
                  <span className="text-xl">🗄️</span>
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Cloud SQL</h3>
              </div>
              <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
                완전 관리형 관계형 데이터베이스 (MySQL, PostgreSQL, SQL Server)
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 자동 백업, 복제, 패치</li>
                <li>• 최대 64TB 스토리지</li>
                <li>• HA 구성: 99.95% SLA</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-orange-200 dark:border-orange-700">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                  <span className="text-xl">🔥</span>
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Firestore</h3>
              </div>
              <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
                NoSQL 문서 데이터베이스 (모바일/웹 앱에 최적화)
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 실시간 동기화</li>
                <li>• 오프라인 지원</li>
                <li>• 자동 확장, 글로벌 복제</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-purple-200 dark:border-purple-700">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                  <span className="text-xl">⚡</span>
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Cloud Spanner</h3>
              </div>
              <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
                글로벌 분산 관계형 데이터베이스 (NewSQL)
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 강력한 일관성 + 수평 확장</li>
                <li>• 99.999% SLA</li>
                <li>• 글로벌 트랜잭션 지원</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-red-200 dark:border-red-700">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center">
                  <span className="text-xl">💾</span>
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Memorystore</h3>
              </div>
              <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
                완전 관리형 인메모리 데이터 스토어 (Redis, Memcached)
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 마이크로초 단위 지연시간</li>
                <li>• 자동 장애 조치</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">AI & 머신러닝</h2>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          <h3 className="font-bold text-green-900 dark:text-green-100 mb-4 text-lg">Vertex AI</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            통합 AI 플랫폼으로, ML 모델의 전체 생명주기를 관리합니다.
            AutoML, 커스텀 학습, 모델 배포, MLOps를 하나의 플랫폼에서 제공합니다.
          </p>
          <div className="grid md:grid-cols-3 gap-3 text-sm">
            <div className="bg-white dark:bg-gray-800 rounded p-3">
              <strong className="text-gray-900 dark:text-white">AutoML</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">코드 없이 ML 모델 학습</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-3">
              <strong className="text-gray-900 dark:text-white">Pre-trained APIs</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">Vision, NLP, Translation</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-3">
              <strong className="text-gray-900 dark:text-white">Vertex AI Workbench</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">Jupyter 기반 ML 개발</p>
            </div>
          </div>
        </div>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-red-50 to-yellow-50 dark:from-red-900/20 dark:to-yellow-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-red-800 dark:text-red-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-red-600 dark:text-red-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>컴퓨팅:</strong> Compute Engine, Cloud Run, GKE (Kubernetes 본가)
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 dark:text-red-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>스토리지:</strong> Cloud Storage (4가지 클래스), Persistent Disk
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 dark:text-red-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>데이터:</strong> BigQuery (최강 DW), Cloud SQL, Firestore, Spanner
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-600 dark:text-red-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>AI/ML:</strong> Vertex AI, AutoML, Pre-trained APIs
            </span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 GCP 공식 문서',
            icon: 'web',
            color: 'border-red-500',
            items: [
              {
                title: 'Google Cloud Documentation',
                description: 'GCP 모든 서비스 공식 문서 (한국어 지원)',
                link: 'https://cloud.google.com/docs'
              },
              {
                title: 'BigQuery Documentation',
                description: 'BigQuery SQL 레퍼런스, 모범 사례, 최적화 가이드',
                link: 'https://cloud.google.com/bigquery/docs'
              },
              {
                title: 'Vertex AI Documentation',
                description: 'ML 모델 학습, 배포, MLOps 완벽 가이드',
                link: 'https://cloud.google.com/vertex-ai/docs'
              }
            ]
          },
          {
            title: '🛠️ 실습 가이드',
            icon: 'tools',
            color: 'border-blue-500',
            items: [
              {
                title: 'Google Cloud Skills Boost',
                description: 'Qwiklabs 실습 랩 및 학습 경로 (무료 크레딧 제공)',
                link: 'https://www.cloudskillsboost.google/'
              },
              {
                title: 'Google Codelabs',
                description: '단계별 실습 튜토리얼 (BigQuery, Kubernetes 등)',
                link: 'https://codelabs.developers.google.com/?cat=Cloud'
              },
              {
                title: 'GCP Solutions Library',
                description: '산업별, 사용 사례별 참조 아키텍처',
                link: 'https://cloud.google.com/solutions'
              }
            ]
          },
          {
            title: '📖 학습 리소스',
            icon: 'book',
            color: 'border-yellow-500',
            items: [
              {
                title: 'Cloud Engineer Certification',
                description: 'GCP Associate Cloud Engineer 자격증 준비',
                link: 'https://cloud.google.com/certification/cloud-engineer'
              },
              {
                title: 'Professional Cloud Architect',
                description: 'GCP 최고 수준의 아키텍트 자격증',
                link: 'https://cloud.google.com/certification/cloud-architect'
              },
              {
                title: 'Google Cloud Blog',
                description: 'GCP 최신 기능 및 모범 사례 블로그',
                link: 'https://cloud.google.com/blog'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
