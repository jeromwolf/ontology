'use client';

import References from '@/components/common/References';

// Chapter 1: 클라우드 컴퓨팅 기초
export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">클라우드 컴퓨팅이란?</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          클라우드 컴퓨팅은 인터넷을 통해 컴퓨팅 리소스(서버, 스토리지, 데이터베이스, 네트워킹, 소프트웨어)를
          제공하는 서비스입니다. 물리적 데이터 센터나 서버를 직접 소유하고 관리하는 대신,
          필요한 만큼만 사용하고 비용을 지불하는 방식입니다.
        </p>
        <div className="bg-sky-50 dark:bg-sky-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-sky-800 dark:text-sky-200 mb-3">클라우드 컴퓨팅의 핵심 가치</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-sky-600 dark:text-sky-400">•</span>
              <div><strong>민첩성</strong>: 몇 분 안에 전 세계에 리소스 배포</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-sky-600 dark:text-sky-400">•</span>
              <div><strong>비용 절감</strong>: 초기 하드웨어 투자 불필요, 사용량 기반 과금</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-sky-600 dark:text-sky-400">•</span>
              <div><strong>확장성</strong>: 수요에 따라 자동으로 리소스 확장/축소</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-sky-600 dark:text-sky-400">•</span>
              <div><strong>신뢰성</strong>: 데이터 백업, 재해 복구, 비즈니스 연속성 보장</div>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">클라우드 서비스 모델</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6">
          클라우드 서비스는 제공하는 추상화 수준에 따라 3가지 주요 모델로 분류됩니다.
        </p>
        <div className="grid gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3 text-xl">
              IaaS (Infrastructure as a Service)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              가장 기본적인 클라우드 서비스 모델로, 가상화된 컴퓨팅 인프라를 제공합니다.
              서버, 스토리지, 네트워크를 온디맨드로 사용할 수 있습니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">주요 서비스:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• AWS EC2 (Elastic Compute Cloud)</li>
                    <li>• Azure Virtual Machines</li>
                    <li>• Google Compute Engine</li>
                    <li>• DigitalOcean Droplets</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">사용 사례:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 웹사이트 호스팅</li>
                    <li>• 테스트/개발 환경</li>
                    <li>• 고성능 컴퓨팅</li>
                    <li>• 빅데이터 분석</li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900/30 rounded">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>관리 범위:</strong> OS, 미들웨어, 런타임, 데이터, 애플리케이션은 사용자가 직접 관리
              </p>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3 text-xl">
              PaaS (Platform as a Service)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              애플리케이션 개발 및 배포를 위한 완전한 플랫폼을 제공합니다.
              인프라 관리 없이 코드 작성에만 집중할 수 있습니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">주요 서비스:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• AWS Elastic Beanstalk</li>
                    <li>• Azure App Service</li>
                    <li>• Google App Engine</li>
                    <li>• Heroku</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">사용 사례:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 웹 애플리케이션 개발</li>
                    <li>• API 개발</li>
                    <li>• 모바일 백엔드</li>
                    <li>• 마이크로서비스</li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-purple-100 dark:bg-purple-900/30 rounded">
              <p className="text-sm text-purple-800 dark:text-purple-200">
                <strong>관리 범위:</strong> 데이터와 애플리케이션만 사용자가 관리, 나머지는 플랫폼에서 자동 관리
              </p>
            </div>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 border-l-4 border-emerald-500">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3 text-xl">
              SaaS (Software as a Service)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              즉시 사용 가능한 완전한 소프트웨어 애플리케이션을 제공합니다.
              설치나 유지보수 없이 웹 브라우저로 바로 접근할 수 있습니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">주요 서비스:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• Google Workspace (Gmail, Docs)</li>
                    <li>• Microsoft 365</li>
                    <li>• Salesforce</li>
                    <li>• Slack, Zoom</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">사용 사례:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 이메일 & 협업 도구</li>
                    <li>• CRM 시스템</li>
                    <li>• 회계 소프트웨어</li>
                    <li>• 프로젝트 관리</li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-emerald-100 dark:bg-emerald-900/30 rounded">
              <p className="text-sm text-emerald-800 dark:text-emerald-200">
                <strong>관리 범위:</strong> 모든 것을 서비스 제공자가 관리, 사용자는 애플리케이션만 사용
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">클라우드 배포 모델</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6">
          클라우드 인프라를 어디에 배포하고 누가 접근할 수 있는지에 따라 4가지 배포 모델로 분류됩니다.
        </p>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">🌐</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Public Cloud</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              여러 고객이 공유하는 클라우드 인프라. AWS, Azure, GCP 등이 대표적입니다.
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">초기 비용 없음</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">무한 확장성</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">사용량 기반 과금</span>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">🏢</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Private Cloud</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              단일 조직 전용 클라우드 인프라. 온프레미스 또는 전용 데이터 센터에 구축됩니다.
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">완전한 통제권</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">높은 보안성</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">규정 준수 용이</span>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">🔄</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Hybrid Cloud</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              Public과 Private 클라우드를 조합한 모델. 데이터와 애플리케이션을 공유합니다.
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">유연성</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">비용 최적화</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">레거시 시스템 통합</span>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-cyan-100 dark:bg-cyan-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">☁️</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Multi-Cloud</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              여러 클라우드 제공업체를 동시에 사용하는 모델. AWS + Azure + GCP 등을 혼합 사용합니다.
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">벤더 종속 방지</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">최적 서비스 선택</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 dark:text-green-400">✓</span>
                <span className="text-gray-600 dark:text-gray-400">고가용성</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">클라우드 컴퓨팅의 핵심 특징 (NIST 정의)</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6">
          미국 표준기술연구소(NIST)는 클라우드 컴퓨팅의 5가지 핵심 특징을 정의했습니다.
        </p>
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-blue-500 dark:bg-blue-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-white text-xl font-bold">1</span>
              </div>
              <div>
                <h3 className="font-bold text-blue-900 dark:text-blue-100 mb-2">온디맨드 셀프 서비스 (On-demand Self-service)</h3>
                <p className="text-gray-700 dark:text-gray-300">
                  사용자가 필요할 때 언제든지 서비스 제공자와 상호작용 없이 자동으로
                  컴퓨팅 리소스(서버 시간, 네트워크 스토리지)를 프로비저닝할 수 있습니다.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-purple-500 dark:bg-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-white text-xl font-bold">2</span>
              </div>
              <div>
                <h3 className="font-bold text-purple-900 dark:text-purple-100 mb-2">광범위한 네트워크 접근 (Broad Network Access)</h3>
                <p className="text-gray-700 dark:text-gray-300">
                  표준 메커니즘을 통해 네트워크에서 기능을 사용할 수 있으며,
                  다양한 클라이언트 플랫폼(모바일, 태블릿, 노트북, 워크스테이션)에서 접근 가능합니다.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-lg p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-emerald-500 dark:bg-emerald-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-white text-xl font-bold">3</span>
              </div>
              <div>
                <h3 className="font-bold text-emerald-900 dark:text-emerald-100 mb-2">리소스 풀링 (Resource Pooling)</h3>
                <p className="text-gray-700 dark:text-gray-300">
                  멀티 테넌트 모델을 사용하여 여러 고객에게 동적으로 할당 및 재할당되는
                  물리적/가상 리소스를 풀링합니다. 위치 독립성을 제공합니다.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded-lg p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-orange-500 dark:bg-orange-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-white text-xl font-bold">4</span>
              </div>
              <div>
                <h3 className="font-bold text-orange-900 dark:text-orange-100 mb-2">신속한 탄력성 (Rapid Elasticity)</h3>
                <p className="text-gray-700 dark:text-gray-300">
                  수요에 따라 신속하고 탄력적으로 프로비저닝되고 해제되며,
                  경우에 따라 자동으로 확장 및 축소됩니다. 무제한 리소스처럼 보입니다.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-indigo-500 dark:bg-indigo-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-white text-xl font-bold">5</span>
              </div>
              <div>
                <h3 className="font-bold text-indigo-900 dark:text-indigo-100 mb-2">측정 가능한 서비스 (Measured Service)</h3>
                <p className="text-gray-700 dark:text-gray-300">
                  리소스 사용을 자동으로 제어하고 최적화합니다.
                  사용량을 모니터링, 제어, 보고하여 제공자와 소비자 모두에게 투명성을 제공합니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-sky-50 to-blue-50 dark:from-sky-900/20 dark:to-blue-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-sky-800 dark:text-sky-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-sky-600 dark:text-sky-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>서비스 모델:</strong> IaaS (인프라), PaaS (플랫폼), SaaS (소프트웨어)
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-sky-600 dark:text-sky-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>배포 모델:</strong> Public, Private, Hybrid, Multi-Cloud
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-sky-600 dark:text-sky-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>NIST 5대 특징:</strong> 온디맨드, 네트워크 접근, 리소스 풀링, 탄력성, 측정 가능
            </span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 공식 문서 & 표준',
            icon: 'web',
            color: 'border-sky-500',
            items: [
              {
                title: 'NIST Cloud Computing Definition (SP 800-145)',
                description: '미국 표준기술연구소의 클라우드 컴퓨팅 공식 정의 - 5대 핵심 특징',
                link: 'https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-145.pdf'
              },
              {
                title: 'AWS Cloud Computing Overview',
                description: 'AWS 클라우드 컴퓨팅 개요 및 기본 개념',
                link: 'https://aws.amazon.com/what-is-cloud-computing/'
              },
              {
                title: 'Microsoft Azure Fundamentals',
                description: 'Azure 클라우드 기초 개념 및 서비스 모델 (무료 학습 경로)',
                link: 'https://learn.microsoft.com/en-us/training/paths/azure-fundamentals/'
              },
              {
                title: 'Google Cloud Architecture Framework',
                description: 'GCP 클라우드 아키텍처 설계 원칙 및 모범 사례',
                link: 'https://cloud.google.com/architecture/framework'
              }
            ]
          },
          {
            title: '🛠️ 실전 가이드',
            icon: 'tools',
            color: 'border-blue-500',
            items: [
              {
                title: 'Cloud Computing Patterns',
                description: '클라우드 디자인 패턴 카탈로그 - 23가지 검증된 패턴',
                link: 'https://www.cloudcomputingpatterns.org/'
              },
              {
                title: 'AWS Well-Architected Framework',
                description: '5가지 기둥: 운영 우수성, 보안, 안정성, 성능 효율성, 비용 최적화',
                link: 'https://aws.amazon.com/architecture/well-architected/'
              },
              {
                title: 'Cloud Adoption Framework (CAF)',
                description: 'Microsoft의 클라우드 도입 전략 및 마이그레이션 가이드',
                link: 'https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/'
              },
              {
                title: 'The Twelve-Factor App',
                description: '클라우드 네이티브 애플리케이션 설계 12가지 원칙',
                link: 'https://12factor.net/'
              }
            ]
          },
          {
            title: '📖 핵심 논문 & 연구',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Above the Clouds: A Berkeley View of Cloud Computing',
                authors: 'Michael Armbrust, Armando Fox, et al. (UC Berkeley)',
                year: '2009',
                description: '클라우드 컴퓨팅의 기초를 정립한 고전 논문 - 10대 장애물과 기회 분석',
                link: 'https://www2.eecs.berkeley.edu/Pubs/TechRpts/2009/EECS-2009-28.pdf'
              },
              {
                title: 'A View of Cloud Computing',
                authors: 'Rajkumar Buyya, Chee Shin Yeo, et al.',
                year: '2009',
                description: '클라우드 컴퓨팅의 비즈니스 모델과 기술적 과제 종합 분석',
                link: 'https://arxiv.org/abs/0808.3558'
              }
            ]
          },
          {
            title: '⚡ 학습 리소스',
            icon: 'book',
            color: 'border-emerald-500',
            items: [
              {
                title: 'AWS Skill Builder (무료)',
                description: '500+ 무료 AWS 클라우드 학습 코스 및 실습 랩',
                link: 'https://skillbuilder.aws/'
              },
              {
                title: 'Microsoft Learn - Azure',
                description: 'Azure 인증 및 학습 경로 - AZ-900 Fundamentals부터 시작',
                link: 'https://learn.microsoft.com/en-us/training/azure/'
              },
              {
                title: 'Google Cloud Skills Boost',
                description: 'GCP 실습 및 퀘스트 플랫폼 - Qwiklabs 통합',
                link: 'https://www.cloudskillsboost.google/'
              },
              {
                title: 'Cloud Academy',
                description: '멀티 클라우드 학습 플랫폼 - AWS, Azure, GCP 통합 과정',
                link: 'https://cloudacademy.com/'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
