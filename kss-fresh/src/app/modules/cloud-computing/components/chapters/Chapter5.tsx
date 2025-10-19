'use client';

import React from 'react';
import References from '@/components/common/References';

// Chapter 5: 클라우드 아키텍처 패턴
export default function Chapter5() {
  return (
    <div className="space-y-8">
      {/* Introduction */}
      <section>
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
          클라우드 아키텍처 패턴
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed text-lg">
          확장 가능하고, 안정적이며, 비용 효율적인 클라우드 애플리케이션을 설계하기 위한 
          핵심 아키텍처 패턴을 학습합니다. 실제 프로덕션 환경에서 검증된 패턴들을 통해 
          대규모 시스템을 구축하는 방법을 배웁니다.
        </p>
      </section>

      {/* 1. 마이크로서비스 아키텍처 */}
      <section className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
        <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4 text-2xl">
          1. 마이크로서비스 아키텍처 (Microservices Architecture)
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 개념</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              애플리케이션을 작은, 독립적인 서비스들로 분해하여 각 서비스가 특정 비즈니스 기능을 담당하도록 하는 아키텍처 패턴입니다.
            </p>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>독립 배포</strong>: 각 서비스를 개별적으로 배포 및 스케일링</li>
              <li>• <strong>기술 다양성</strong>: 서비스마다 최적의 기술 스택 선택 가능</li>
              <li>• <strong>장애 격리</strong>: 한 서비스의 장애가 전체 시스템에 영향 최소화</li>
              <li>• <strong>팀 자율성</strong>: 작은 팀이 독립적으로 서비스 개발</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">클라우드 구현 패턴</h4>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-purple-200 dark:border-purple-700">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">AWS</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• ECS/EKS (컨테이너)</li>
                  <li>• API Gateway (라우팅)</li>
                  <li>• Lambda (서버리스)</li>
                  <li>• EventBridge (이벤트)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-purple-200 dark:border-purple-700">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">Azure</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• AKS (Kubernetes)</li>
                  <li>• API Management</li>
                  <li>• Service Bus (메시징)</li>
                  <li>• Event Grid (이벤트)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-purple-200 dark:border-purple-700">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">GCP</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• GKE (Kubernetes)</li>
                  <li>• Cloud Run (컨테이너)</li>
                  <li>• Pub/Sub (메시징)</li>
                  <li>• Apigee (API 관리)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">실전 고려사항</h4>
            <ul className="space-y-1 text-sm text-purple-800 dark:text-purple-200">
              <li>✓ <strong>서비스 경계</strong>: Domain-Driven Design (DDD)으로 서비스 분리</li>
              <li>✓ <strong>통신 패턴</strong>: REST API, gRPC, 메시지 큐 적절히 선택</li>
              <li>✓ <strong>데이터 관리</strong>: 각 서비스가 독립적인 데이터베이스 소유 (Database per Service)</li>
              <li>✓ <strong>분산 트랜잭션</strong>: Saga 패턴으로 데이터 일관성 보장</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. 이벤트 기반 아키텍처 */}
      <section className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6 border-l-4 border-indigo-500">
        <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4 text-2xl">
          2. 이벤트 기반 아키텍처 (Event-Driven Architecture)
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 개념</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              시스템 컴포넌트 간의 통신을 이벤트를 통해 수행하는 아키텍처 패턴입니다. 
              느슨한 결합(Loose Coupling)과 높은 확장성을 제공합니다.
            </p>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">주요 패턴</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-indigo-700 dark:text-indigo-300 block mb-2">Pub/Sub (발행/구독)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  이벤트 발행자(Publisher)가 이벤트를 발행하면 구독자(Subscriber)가 비동기로 수신
                </p>
                <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                  <li>• AWS SNS/SQS</li>
                  <li>• Azure Event Grid</li>
                  <li>• GCP Pub/Sub</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-indigo-700 dark:text-indigo-300 block mb-2">Event Streaming</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  연속적인 이벤트 스트림을 처리하고 저장 (Apache Kafka 스타일)
                </p>
                <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                  <li>• AWS Kinesis</li>
                  <li>• Azure Event Hubs</li>
                  <li>• GCP Dataflow</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-indigo-100 dark:bg-indigo-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-indigo-900 dark:text-indigo-100 mb-2">실전 사용 사례</h4>
            <ul className="space-y-2 text-sm text-indigo-800 dark:text-indigo-200">
              <li>🔹 <strong>주문 처리</strong>: 주문 생성 → 결제 → 재고 관리 → 배송 (각 단계가 이벤트로 연결)</li>
              <li>🔹 <strong>실시간 분석</strong>: 클릭스트림, 로그 데이터 실시간 수집 및 분석</li>
              <li>🔹 <strong>알림 시스템</strong>: 시스템 이벤트 발생 시 여러 채널(이메일, SMS, Slack)로 알림</li>
              <li>🔹 <strong>데이터 동기화</strong>: 여러 시스템 간 데이터 일관성 유지</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 3. CQRS 패턴 */}
      <section className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
        <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4 text-2xl">
          3. CQRS (Command Query Responsibility Segregation)
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 개념</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              읽기(Query)와 쓰기(Command) 작업을 별도의 모델로 분리하여 각각 최적화하는 패턴입니다.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-blue-300 dark:border-blue-700">
              <strong className="text-blue-700 dark:text-blue-300 block mb-3 text-lg">Command (쓰기)</strong>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• 데이터 변경 작업 (Create, Update, Delete)</li>
                <li>• 비즈니스 로직 실행</li>
                <li>• 정규화된 관계형 DB 사용</li>
                <li>• 트랜잭션 일관성 중시</li>
              </ul>
              <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/30 rounded text-xs">
                예시: RDS, SQL Database
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-green-300 dark:border-green-700">
              <strong className="text-green-700 dark:text-green-300 block mb-3 text-lg">Query (읽기)</strong>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• 데이터 조회 작업 (Read)</li>
                <li>• 비정규화된 뷰</li>
                <li>• 읽기 최적화 DB (NoSQL, 캐시)</li>
                <li>• 빠른 응답 속도 중시</li>
              </ul>
              <div className="mt-3 p-2 bg-green-50 dark:bg-green-900/30 rounded text-xs">
                예시: DynamoDB, Redis, Elasticsearch
              </div>
            </div>
          </div>

          <div className="bg-blue-100 dark:bg-blue-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">CQRS + Event Sourcing 결합</h4>
            <p className="text-sm text-blue-800 dark:text-blue-200 mb-2">
              모든 상태 변경을 이벤트로 저장하고, 읽기 모델은 이벤트를 재생(Replay)하여 구축
            </p>
            <ul className="space-y-1 text-sm text-blue-800 dark:text-blue-200">
              <li>✓ 완전한 감사(Audit) 추적</li>
              <li>✓ 시간 여행(Time Travel) - 과거 특정 시점의 상태 재구성</li>
              <li>✓ 읽기/쓰기 독립적 스케일링</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 4. Circuit Breaker & Resilience */}
      <section className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border-l-4 border-red-500">
        <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4 text-2xl">
          4. 회복성 패턴 (Resilience Patterns)
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Circuit Breaker (회로 차단기)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              실패하는 서비스 호출을 감지하고 자동으로 차단하여 전체 시스템 장애를 방지합니다.
            </p>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg">
                <strong className="text-green-800 dark:text-green-200 block mb-1">Closed (정상)</strong>
                <p className="text-xs text-green-700 dark:text-green-300">
                  모든 요청 정상 처리
                </p>
              </div>
              <div className="bg-yellow-100 dark:bg-yellow-900/30 p-3 rounded-lg">
                <strong className="text-yellow-800 dark:text-yellow-200 block mb-1">Open (차단)</strong>
                <p className="text-xs text-yellow-700 dark:text-yellow-300">
                  실패율 임계값 초과 시 모든 요청 즉시 실패 반환
                </p>
              </div>
              <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded-lg">
                <strong className="text-blue-800 dark:text-blue-200 block mb-1">Half-Open (반개방)</strong>
                <p className="text-xs text-blue-700 dark:text-blue-300">
                  일부 요청만 허용하여 복구 여부 확인
                </p>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">기타 회복성 패턴</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-red-700 dark:text-red-300">Retry (재시도)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                  일시적 장애 시 지수 백오프(Exponential Backoff)로 재시도
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-red-700 dark:text-red-300">Timeout (타임아웃)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                  응답 대기 시간 제한으로 리소스 고갈 방지
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-red-700 dark:text-red-300">Bulkhead (격벽)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                  리소스를 분리하여 한 컴포넌트 장애가 전체에 영향 방지
                </p>
              </div>
            </div>
          </div>

          <div className="bg-red-100 dark:bg-red-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-red-900 dark:text-red-100 mb-2">클라우드 구현</h4>
            <ul className="space-y-1 text-sm text-red-800 dark:text-red-200">
              <li>• <strong>AWS</strong>: AWS App Mesh, X-Ray (분산 추적)</li>
              <li>• <strong>Azure</strong>: Azure Service Fabric, Application Insights</li>
              <li>• <strong>라이브러리</strong>: Hystrix (Netflix), Polly (.NET), resilience4j (Java)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. API Gateway 패턴 */}
      <section className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6 border-l-4 border-teal-500">
        <h3 className="font-semibold text-teal-800 dark:text-teal-200 mb-4 text-2xl">
          5. API Gateway 패턴
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 개념</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              모든 클라이언트 요청의 단일 진입점 역할을 하여 라우팅, 인증, 속도 제한 등을 중앙에서 관리합니다.
            </p>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">주요 기능</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                  <strong className="text-teal-700 dark:text-teal-300 block mb-1">라우팅 및 프록시</strong>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    URL 경로에 따라 적절한 마이크로서비스로 요청 전달
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                  <strong className="text-teal-700 dark:text-teal-300 block mb-1">인증 & 권한</strong>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    OAuth 2.0, JWT 검증 중앙화
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                  <strong className="text-teal-700 dark:text-teal-300 block mb-1">속도 제한 (Rate Limiting)</strong>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    API 남용 방지 및 공정한 리소스 사용
                  </p>
                </div>
              </div>
              <div className="space-y-2">
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                  <strong className="text-teal-700 dark:text-teal-300 block mb-1">캐싱</strong>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    자주 사용되는 응답 캐싱으로 성능 향상
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                  <strong className="text-teal-700 dark:text-teal-300 block mb-1">프로토콜 변환</strong>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    REST ↔ gRPC, HTTP ↔ WebSocket 변환
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                  <strong className="text-teal-700 dark:text-teal-300 block mb-1">로깅 & 모니터링</strong>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    모든 API 호출 추적 및 분석
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-teal-100 dark:bg-teal-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-900 dark:text-teal-100 mb-2">클라우드 서비스</h4>
            <div className="grid md:grid-cols-3 gap-3 text-sm">
              <div>
                <strong className="text-teal-800 dark:text-teal-200 block mb-1">AWS</strong>
                <ul className="text-teal-700 dark:text-teal-300 space-y-1">
                  <li>• API Gateway (REST/WebSocket)</li>
                  <li>• Application Load Balancer</li>
                </ul>
              </div>
              <div>
                <strong className="text-teal-800 dark:text-teal-200 block mb-1">Azure</strong>
                <ul className="text-teal-700 dark:text-teal-300 space-y-1">
                  <li>• API Management</li>
                  <li>• Application Gateway</li>
                </ul>
              </div>
              <div>
                <strong className="text-teal-800 dark:text-teal-200 block mb-1">GCP</strong>
                <ul className="text-teal-700 dark:text-teal-300 space-y-1">
                  <li>• Apigee API Platform</li>
                  <li>• Cloud Endpoints</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. 캐싱 전략 */}
      <section className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
        <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4 text-2xl">
          6. 캐싱 전략 (Caching Strategies)
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">캐싱 레이어</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">1. CDN 캐싱 (엣지)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  정적 콘텐츠(이미지, CSS, JS)를 사용자에게 가장 가까운 위치에서 제공
                </p>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  CloudFront, Azure CDN, Cloud CDN
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">2. 애플리케이션 캐싱</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  자주 조회되는 데이터를 메모리에 저장 (Redis, Memcached)
                </p>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  ElastiCache, Azure Cache for Redis, Memorystore
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">3. 데이터베이스 캐싱</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  쿼리 결과 캐싱, 읽기 전용 복제본 활용
                </p>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  RDS Read Replica, DAX (DynamoDB Accelerator)
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">캐시 무효화 전략</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">TTL (Time To Live)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  특정 시간 후 자동 만료 (예: 5분, 1시간)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">Write-Through / Write-Behind</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  데이터 업데이트 시 캐시 동시/비동기 갱신
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Summary */}
      <section className="bg-gradient-to-r from-purple-100 to-indigo-100 dark:from-purple-900/30 dark:to-indigo-900/30 rounded-lg p-6">
        <h3 className="font-semibold text-purple-900 dark:text-purple-100 mb-4 text-xl">
          📚 학습 요약
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200">핵심 패턴</h4>
            <ul className="space-y-1 text-purple-700 dark:text-purple-300">
              <li>✓ 마이크로서비스: 독립 배포, 기술 다양성</li>
              <li>✓ 이벤트 기반: 느슨한 결합, 비동기 통신</li>
              <li>✓ CQRS: 읽기/쓰기 분리 최적화</li>
              <li>✓ Circuit Breaker: 장애 전파 방지</li>
            </ul>
          </div>
          <div className="space-y-2">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200">설계 원칙</h4>
            <ul className="space-y-1 text-purple-700 dark:text-purple-300">
              <li>✓ 확장성: 수평 확장 가능한 구조</li>
              <li>✓ 회복성: 장애에 강한 시스템</li>
              <li>✓ 관측성: 로깅, 모니터링, 추적</li>
              <li>✓ 자동화: IaC, CI/CD 파이프라인</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 아키텍처 패턴 문서',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'AWS Well-Architected Framework',
                url: 'https://aws.amazon.com/architecture/well-architected/',
                description: 'AWS 클라우드 아키텍처 설계 모범 사례'
              },
              {
                title: 'Azure Architecture Center',
                url: 'https://learn.microsoft.com/en-us/azure/architecture/',
                description: 'Azure 참조 아키텍처 및 디자인 패턴'
              },
              {
                title: 'GCP Architecture Framework',
                url: 'https://cloud.google.com/architecture/framework',
                description: 'Google Cloud 아키텍처 설계 프레임워크'
              },
              {
                title: 'Microservices.io',
                url: 'https://microservices.io/patterns/index.html',
                description: 'Chris Richardson의 마이크로서비스 패턴 카탈로그'
              }
            ]
          },
          {
            title: '📖 핵심 논문 & 서적',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Building Microservices (Sam Newman)',
                url: 'https://www.oreilly.com/library/view/building-microservices-2nd/9781492034018/',
                description: '마이크로서비스 설계 및 구현 바이블'
              },
              {
                title: 'Designing Data-Intensive Applications (Martin Kleppmann)',
                url: 'https://dataintensive.net/',
                description: '분산 시스템과 데이터 아키텍처 필독서'
              },
              {
                title: 'Release It! (Michael Nygard)',
                url: 'https://pragprog.com/titles/mnee2/release-it-second-edition/',
                description: '회복성 패턴 및 프로덕션 시스템 설계'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 프레임워크',
            icon: 'tools' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Spring Cloud',
                url: 'https://spring.io/projects/spring-cloud',
                description: '마이크로서비스 패턴 구현을 위한 Java 프레임워크'
              },
              {
                title: 'Istio Service Mesh',
                url: 'https://istio.io/',
                description: '마이크로서비스 트래픽 관리, 보안, 관측성'
              },
              {
                title: 'Netflix OSS',
                url: 'https://netflix.github.io/',
                description: 'Hystrix, Eureka 등 검증된 마이크로서비스 도구'
              },
              {
                title: 'Apache Kafka',
                url: 'https://kafka.apache.org/',
                description: '분산 이벤트 스트리밍 플랫폼'
              }
            ]
          },
          {
            title: '🎓 학습 리소스',
            icon: 'web' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'AWS Architecture Blog',
                url: 'https://aws.amazon.com/blogs/architecture/',
                description: 'AWS 솔루션 아키텍트의 실전 사례'
              },
              {
                title: 'Microsoft Cloud Design Patterns',
                url: 'https://learn.microsoft.com/en-us/azure/architecture/patterns/',
                description: '24가지 클라우드 디자인 패턴 상세 설명'
              },
              {
                title: 'The Twelve-Factor App',
                url: 'https://12factor.net/',
                description: 'SaaS 애플리케이션 구축 방법론'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
