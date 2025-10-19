'use client';

import References from '@/components/common/References';

// Chapter 3: Microsoft Azure 기초
export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">Microsoft Azure 소개</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          Microsoft Azure는 2010년 출시된 클라우드 플랫폼으로, 200개 이상의 서비스를 
          60개 이상의 리전에서 제공합니다. Windows 기반 엔터프라이즈 환경과의 뛰어난 통합성이 강점입니다.
        </p>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">Azure 핵심 강점</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 dark:text-blue-400">•</span>
              <div><strong>엔터프라이즈 통합</strong>: Active Directory, Office 365와 완벽 연동</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 dark:text-blue-400">•</span>
              <div><strong>하이브리드 클라우드</strong>: Azure Arc로 온프레미스 통합 관리</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 dark:text-blue-400">•</span>
              <div><strong>AI/ML 리더십</strong>: Azure OpenAI Service, Cognitive Services</div>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">핵심 컴퓨팅 서비스</h2>
        <div className="grid gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3 text-xl">
              Azure Virtual Machines
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Windows와 Linux 가상 머신을 제공합니다. AWS EC2와 유사하지만 
              Windows Server 라이선스 비용이 포함되어 있어 경제적입니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">VM 시리즈:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• <strong>B 시리즈</strong>: 버스트 가능, 저비용</li>
                    <li>• <strong>D 시리즈</strong>: 범용 컴퓨팅</li>
                    <li>• <strong>E 시리즈</strong>: 메모리 최적화</li>
                    <li>• <strong>F 시리즈</strong>: 컴퓨팅 최적화</li>
                    <li>• <strong>N 시리즈</strong>: GPU 가속</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">주요 기능:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• Azure Spot VM: 최대 90% 할인</li>
                    <li>• Reserved VM: 1-3년 약정 할인</li>
                    <li>• VM Scale Sets: 자동 확장</li>
                    <li>• Availability Zones: 99.99% SLA</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3 text-xl">
              Azure App Service
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              완전 관리형 PaaS 플랫폼으로, 웹 앱, 모바일 백엔드, RESTful API를 빠르게 구축하고 배포할 수 있습니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <strong className="text-gray-900 dark:text-white block mb-2">지원 언어:</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• .NET, .NET Core</li>
                  <li>• Java, Node.js, Python</li>
                  <li>• PHP, Ruby</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <strong className="text-gray-900 dark:text-white block mb-2">주요 기능:</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• CI/CD (GitHub, Azure DevOps)</li>
                  <li>• 슬롯을 통한 무중단 배포</li>
                  <li>• 내장 인증 (AAD, Google 등)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-cyan-50 dark:bg-cyan-900/20 rounded-lg p-6 border-l-4 border-cyan-500">
            <h3 className="font-semibold text-cyan-800 dark:text-cyan-200 mb-3 text-xl">
              Azure Functions (서버리스)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              이벤트 기반 서버리스 컴퓨팅으로, AWS Lambda와 동일한 개념입니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">트리거:</strong>
                  <p className="text-gray-700 dark:text-gray-300">HTTP, Timer, Blob Storage, Event Grid, Cosmos DB 등</p>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">Durable Functions:</strong>
                  <p className="text-gray-700 dark:text-gray-300">상태 유지 워크플로우 오케스트레이션</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">스토리지 서비스</h2>
        <div className="space-y-6">
          <div className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6 border-l-4 border-teal-500">
            <h3 className="font-semibold text-teal-800 dark:text-teal-200 mb-3 text-xl">
              Azure Blob Storage
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              AWS S3와 유사한 객체 스토리지 서비스입니다. 비정형 데이터(이미지, 비디오, 로그)를 저장합니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <strong className="text-gray-900 dark:text-white block mb-3">액세스 계층:</strong>
              <div className="grid md:grid-cols-4 gap-3 text-sm">
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <strong className="text-gray-900 dark:text-white">Hot</strong>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">자주 액세스</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <strong className="text-gray-900 dark:text-white">Cool</strong>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">30일 이상 보관</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <strong className="text-gray-900 dark:text-white">Archive</strong>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">180일 이상 보관</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <strong className="text-gray-900 dark:text-white">Premium</strong>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">낮은 지연시간</p>
                </div>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Azure Files</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                완전 관리형 파일 공유 (SMB/NFS 프로토콜)
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 클라우드 또는 온프레미스에서 마운트</li>
                <li>• Azure File Sync로 하이브리드 구성</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Azure Disk Storage</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                VM용 블록 레벨 스토리지 (AWS EBS와 동일)
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• Premium SSD, Standard SSD, Standard HDD</li>
                <li>• Ultra Disk: 최고 성능 (최대 160,000 IOPS)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">데이터베이스 서비스</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-indigo-200 dark:border-indigo-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">🗄️</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Azure SQL Database</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              완전 관리형 관계형 데이터베이스 (SQL Server 엔진 기반)
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• 자동 백업, 패치, 튜닝</li>
              <li>• Hyperscale: 최대 100TB</li>
              <li>• Serverless 옵션: 자동 일시 중지</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-green-200 dark:border-green-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">🌍</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Azure Cosmos DB</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              글로벌 분산 NoSQL 데이터베이스 (AWS DynamoDB와 유사)
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• 5가지 API: SQL, MongoDB, Cassandra, Gremlin, Table</li>
              <li>• 99.999% SLA, 한 자릿수 밀리초 지연시간</li>
              <li>• 자동 글로벌 복제</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-orange-200 dark:border-orange-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">🐘</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Azure Database for PostgreSQL/MySQL</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              완전 관리형 오픈소스 데이터베이스
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Single Server, Flexible Server</li>
              <li>• 자동 백업, 고가용성</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-red-200 dark:border-red-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">⚡</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Azure Cache for Redis</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              완전 관리형 인메모리 캐싱 서비스
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Enterprise 계층: Redis Enterprise 기능</li>
              <li>• 지리적 복제, 클러스터링</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">AI & 머신러닝</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
          <h3 className="font-bold text-purple-900 dark:text-purple-100 mb-4 text-lg">Azure OpenAI Service</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            OpenAI의 GPT-4, GPT-3.5, DALL-E, Whisper 모델을 Azure에서 사용할 수 있습니다.
            엔터프라이즈급 보안, 규정 준수, 지역 가용성을 제공합니다.
          </p>
          <div className="grid md:grid-cols-3 gap-3 text-sm">
            <div className="bg-white dark:bg-gray-800 rounded p-3">
              <strong className="text-gray-900 dark:text-white">Azure Cognitive Services</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">음성, 비전, 언어 AI</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-3">
              <strong className="text-gray-900 dark:text-white">Azure Machine Learning</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">ML 모델 학습 & 배포</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-3">
              <strong className="text-gray-900 dark:text-white">Azure Bot Service</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">챗봇 개발 플랫폼</p>
            </div>
          </div>
        </div>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-blue-800 dark:text-blue-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>컴퓨팅:</strong> Virtual Machines, App Service, Azure Functions
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>스토리지:</strong> Blob Storage, Files, Disk (4가지 액세스 계층)
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>데이터베이스:</strong> SQL Database, Cosmos DB, PostgreSQL/MySQL, Cache for Redis
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>AI:</strong> Azure OpenAI Service, Cognitive Services, Machine Learning
            </span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 Azure 공식 문서',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'Microsoft Azure Documentation',
                description: 'Azure 모든 서비스 공식 문서 (한국어 완벽 지원)',
                link: 'https://docs.microsoft.com/azure/'
              },
              {
                title: 'Azure Architecture Center',
                description: '참조 아키텍처, 모범 사례, 디자인 패턴',
                link: 'https://docs.microsoft.com/azure/architecture/'
              },
              {
                title: 'Azure OpenAI Service Documentation',
                description: 'GPT-4, DALL-E 사용 가이드 및 API 레퍼런스',
                link: 'https://learn.microsoft.com/azure/ai-services/openai/'
              }
            ]
          },
          {
            title: '🛠️ 실습 가이드',
            icon: 'tools',
            color: 'border-purple-500',
            items: [
              {
                title: 'Microsoft Learn',
                description: '무료 Azure 학습 경로 및 실습 랩 (한국어)',
                link: 'https://learn.microsoft.com/training/azure/'
              },
              {
                title: 'Azure Quickstart Templates',
                description: '1000+ Azure 리소스 배포 템플릿 (ARM, Bicep)',
                link: 'https://azure.microsoft.com/resources/templates/'
              },
              {
                title: 'Azure Samples',
                description: 'GitHub에 공개된 Azure 샘플 코드 모음',
                link: 'https://github.com/Azure-Samples'
              }
            ]
          },
          {
            title: '📖 학습 리소스',
            icon: 'book',
            color: 'border-indigo-500',
            items: [
              {
                title: 'Azure Fundamentals (AZ-900)',
                description: 'Azure 기초 자격증 - 무료 학습 경로',
                link: 'https://learn.microsoft.com/certifications/azure-fundamentals/'
              },
              {
                title: 'Azure Solutions Architect (AZ-305)',
                description: 'Azure 솔루션 설계 전문가 자격증',
                link: 'https://learn.microsoft.com/certifications/azure-solutions-architect/'
              },
              {
                title: 'Azure Friday',
                description: 'Azure 엔지니어와의 주간 비디오 시리즈',
                link: 'https://azure.microsoft.com/resources/videos/azure-friday/'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
