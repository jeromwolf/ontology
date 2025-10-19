'use client';

import React from 'react';
import References from '@/components/common/References';

// Chapter 6: 서버리스 아키텍처
export default function Chapter6() {
  return (
    <div className="space-y-8">
      {/* Introduction */}
      <section>
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent">
          서버리스 아키텍처
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed text-lg">
          서버 관리 없이 코드만 작성하여 애플리케이션을 구축하고 실행하는 현대적인 클라우드 패러다임입니다. 
          사용한 만큼만 비용을 지불하며, 자동으로 확장되는 완전 관리형 서비스를 활용합니다.
        </p>
      </section>

      {/* 1. 서버리스 핵심 개념 */}
      <section className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
        <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4 text-2xl">
          1. 서버리스 핵심 개념
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">서버리스란?</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              서버가 없다는 의미가 아니라, 개발자가 서버를 <strong>관리하지 않아도 된다</strong>는 의미입니다.
            </p>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>✓ <strong>No Server Management</strong>: 서버 프로비저닝, 패치, 확장 자동화</li>
              <li>✓ <strong>Pay-per-Use</strong>: 실행 시간과 리소스 사용량에 대해서만 과금</li>
              <li>✓ <strong>Auto-Scaling</strong>: 트래픽에 따라 자동으로 0 ~ 무한대로 확장</li>
              <li>✓ <strong>Built-in HA</strong>: 고가용성 및 내결함성 내장</li>
              <li>✓ <strong>Event-Driven</strong>: 이벤트 발생 시에만 실행</li>
            </ul>
          </div>

          <div className="bg-green-100 dark:bg-green-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-green-900 dark:text-green-100 mb-2">전통적 vs 서버리스 비교</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <strong className="text-green-800 dark:text-green-200 block mb-2">전통적 방식 (EC2, VM)</strong>
                <ul className="space-y-1 text-green-700 dark:text-green-300">
                  <li>• 서버 항상 실행 → 유휴 비용 발생</li>
                  <li>• 수동 스케일링 필요</li>
                  <li>• OS 패치, 보안 관리 필요</li>
                  <li>• 용량 계획 필요</li>
                </ul>
              </div>
              <div>
                <strong className="text-green-800 dark:text-green-200 block mb-2">서버리스 (Lambda, Functions)</strong>
                <ul className="space-y-1 text-green-700 dark:text-green-300">
                  <li>• 사용 시간만큼만 과금</li>
                  <li>• 자동 스케일링</li>
                  <li>• 인프라 관리 불필요</li>
                  <li>• 무한한 확장성</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. AWS Lambda */}
      <section className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
        <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4 text-2xl">
          2. AWS Lambda - Function as a Service (FaaS)
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 특징</h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>지원 언어</strong>: Python, Node.js, Java, Go, Ruby, .NET Core, Custom Runtime</li>
              <li>• <strong>실행 시간</strong>: 최대 15분 (900초)</li>
              <li>• <strong>메모리</strong>: 128MB ~ 10,240MB (10GB) - CPU는 메모리에 비례 할당</li>
              <li>• <strong>동시 실행</strong>: 계정당 기본 1,000개 (증가 요청 가능)</li>
              <li>• <strong>콜드 스타트</strong>: 첫 요청 시 100ms ~ 수 초 (언어별 차이)</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">가격 구조</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-orange-700 dark:text-orange-300 block mb-2">요청 수 기반</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 매월 100만 건 무료</li>
                    <li>• 이후 100만 건당 <strong>$0.20</strong></li>
                  </ul>
                </div>
                <div>
                  <strong className="text-orange-700 dark:text-orange-300 block mb-2">실행 시간 기반 (GB-초)</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 매월 400,000 GB-초 무료</li>
                    <li>• 이후 GB-초당 <strong>$0.0000166667</strong></li>
                  </ul>
                </div>
              </div>
              <div className="mt-3 p-3 bg-orange-50 dark:bg-orange-900/30 rounded text-xs text-orange-800 dark:text-orange-200">
                예시: 1GB 메모리, 1초 실행, 300만 건/월 → 약 $6.67/월
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">트리거 (Trigger) 유형</h4>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2 text-sm">HTTP 요청</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  API Gateway, Application Load Balancer
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2 text-sm">데이터 변경</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  S3, DynamoDB Streams, Kinesis
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2 text-sm">스케줄</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  EventBridge (Cron 표현식)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2 text-sm">메시지 큐</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  SQS, SNS
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2 text-sm">IoT</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  AWS IoT Core
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2 text-sm">Alexa</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Alexa Skills
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Azure Functions */}
      <section className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
        <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4 text-2xl">
          3. Azure Functions
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">호스팅 플랜</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-2">Consumption Plan (소비 기반)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  Lambda와 유사한 완전 서버리스 모델
                </p>
                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 자동 스케일링</li>
                  <li>• 실행 시간만 과금</li>
                  <li>• 매월 100만 건 + 400,000 GB-초 무료</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-2">Premium Plan</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  콜드 스타트 방지, VNet 통합, 무제한 실행 시간
                </p>
                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• Pre-warmed 인스턴스 (콜드 스타트 0)</li>
                  <li>• 60분 이상 실행 가능</li>
                  <li>• 예약 인스턴스로 고정 비용</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-2">Dedicated Plan (App Service)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  기존 App Service 인프라 활용
                </p>
                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• VM 기반 (항상 실행)</li>
                  <li>• 예측 가능한 비용</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Durable Functions (상태 유지 함수)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              서버리스 환경에서 <strong>상태를 유지</strong>하며 복잡한 워크플로우를 오케스트레이션합니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">Function Chaining</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  여러 함수를 순서대로 실행 (A → B → C)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">Fan-out/Fan-in</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  병렬 실행 후 결과 집계
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">Human Interaction</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  승인 대기 등 사람의 입력 대기
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">Monitor Pattern</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  반복적인 폴링 및 상태 확인
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. GCP Cloud Functions & Cloud Run */}
      <section className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border-l-4 border-red-500">
        <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4 text-2xl">
          4. Google Cloud Platform 서버리스
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Cloud Functions (2nd Gen)</h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>지원 언어</strong>: Node.js, Python, Go, Java, Ruby, .NET, PHP</li>
              <li>• <strong>최대 실행 시간</strong>: 60분 (1세대는 9분)</li>
              <li>• <strong>메모리</strong>: 최대 16GB</li>
              <li>• <strong>동시 요청</strong>: 인스턴스당 최대 1,000개</li>
              <li>• <strong>무료 할당량</strong>: 월 200만 건 호출, 400,000 GB-초, 200,000 GHz-초</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Cloud Run (컨테이너 기반 서버리스)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              Docker 컨테이너를 서버리스로 실행하는 독특한 서비스 (AWS Fargate + Lambda 결합)
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <strong className="text-red-700 dark:text-red-300 block mb-3">핵심 장점</strong>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                    <li>✓ <strong>언어 제약 없음</strong>: 컨테이너화 가능한 모든 언어</li>
                    <li>✓ <strong>포터블</strong>: 로컬 Docker 환경과 동일</li>
                    <li>✓ <strong>빠른 스케일링</strong>: 0 → 수천 인스턴스 (초 단위)</li>
                  </ul>
                </div>
                <div>
                  <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                    <li>✓ <strong>HTTP/gRPC 지원</strong>: 웹 앱, API 호스팅 최적</li>
                    <li>✓ <strong>완전 관리형</strong>: Kubernetes 없이 컨테이너 실행</li>
                    <li>✓ <strong>비용 효율</strong>: 100ms 단위 과금</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-red-100 dark:bg-red-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-red-900 dark:text-red-100 mb-2">Cloud Functions vs Cloud Run 선택 가이드</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <strong className="text-red-800 dark:text-red-200 block mb-1">Cloud Functions 선택</strong>
                <ul className="text-red-700 dark:text-red-300 space-y-1">
                  <li>• 단순한 이벤트 핸들링</li>
                  <li>• 빠른 개발 필요</li>
                  <li>• 트리거 기반 실행</li>
                </ul>
              </div>
              <div>
                <strong className="text-red-800 dark:text-red-200 block mb-1">Cloud Run 선택</strong>
                <ul className="text-red-700 dark:text-red-300 space-y-1">
                  <li>• 복잡한 애플리케이션</li>
                  <li>• 기존 컨테이너 마이그레이션</li>
                  <li>• HTTP/gRPC 엔드포인트</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 서버리스 아키텍처 패턴 */}
      <section className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
        <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4 text-2xl">
          5. 서버리스 아키텍처 패턴
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">1. API 백엔드 패턴</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                <strong>구성</strong>: API Gateway + Lambda/Functions + DynamoDB/Cosmos DB
              </p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                <li>• RESTful API 호스팅</li>
                <li>• 인증/권한 (OAuth, JWT)</li>
                <li>• 자동 스케일링</li>
                <li>• 사용 사례: 모바일 앱 백엔드, SaaS API</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">2. 이벤트 프로세싱 패턴</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                <strong>구성</strong>: S3/Blob → Lambda → 데이터 처리 → 저장
              </p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                <li>• 이미지 리사이징, 썸네일 생성</li>
                <li>• 비디오 트랜스코딩</li>
                <li>• ETL 파이프라인</li>
                <li>• 사용 사례: 사용자 업로드 파일 처리</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">3. 스트림 프로세싱 패턴</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                <strong>구성</strong>: Kinesis/Event Hubs → Lambda → 실시간 분석
              </p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                <li>• 로그 분석</li>
                <li>• 실시간 모니터링</li>
                <li>• 사용 사례: 클릭스트림 분석, IoT 데이터 처리</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">4. 스케줄링 패턴</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                <strong>구성</strong>: EventBridge/Cloud Scheduler → Lambda → 정기 작업
              </p>
              <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                <li>• 데이터 백업 (매일 자정)</li>
                <li>• 보고서 생성 (주간/월간)</li>
                <li>• 사용 사례: 배치 작업, cron job 대체</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 6. 서버리스 모범 사례 */}
      <section className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6 border-l-4 border-teal-500">
        <h3 className="font-semibold text-teal-800 dark:text-teal-200 mb-4 text-2xl">
          6. 서버리스 모범 사례
        </h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-teal-700 dark:text-teal-300 block mb-2">1. 콜드 스타트 최적화</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>의존성 최소화</strong>: 필요한 라이브러리만 포함</li>
              <li>• <strong>프로비저닝된 동시성</strong>: AWS Lambda에서 인스턴스 미리 준비</li>
              <li>• <strong>컴파일된 언어 사용</strong>: Go, Java (GraalVM Native) > Python, Node.js</li>
              <li>• <strong>레이어 활용</strong>: 공통 의존성을 Lambda Layer로 분리</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-teal-700 dark:text-teal-300 block mb-2">2. 비용 최적화</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>메모리 튜닝</strong>: 실제 사용량에 맞춰 메모리 조정 (CPU도 함께 증가)</li>
              <li>• <strong>실행 시간 단축</strong>: 효율적인 코드 작성</li>
              <li>• <strong>Reserved Capacity</strong>: 예측 가능한 워크로드는 예약 용량 사용</li>
              <li>• <strong>아키텍처 최적화</strong>: 불필요한 함수 호출 제거</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-teal-700 dark:text-teal-300 block mb-2">3. 보안</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>최소 권한 원칙</strong>: IAM 역할 최소 권한만 부여</li>
              <li>• <strong>환경 변수 암호화</strong>: KMS로 민감 정보 암호화</li>
              <li>• <strong>VPC 내 실행</strong>: 프라이빗 리소스 접근 시 VPC 배치</li>
              <li>• <strong>Secrets Manager</strong>: DB 비밀번호 등 중앙 관리</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-teal-700 dark:text-teal-300 block mb-2">4. 모니터링 & 관측성</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>구조화된 로깅</strong>: JSON 형식 로그로 파싱 용이</li>
              <li>• <strong>분산 추적</strong>: X-Ray, Application Insights 활용</li>
              <li>• <strong>커스텀 메트릭</strong>: 비즈니스 메트릭 CloudWatch로 전송</li>
              <li>• <strong>알람 설정</strong>: 에러율, 실행 시간 임계값 모니터링</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Learning Summary */}
      <section className="bg-gradient-to-r from-green-100 to-teal-100 dark:from-green-900/30 dark:to-teal-900/30 rounded-lg p-6">
        <h3 className="font-semibold text-green-900 dark:text-green-100 mb-4 text-xl">
          📚 학습 요약
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <h4 className="font-semibold text-green-800 dark:text-green-200">핵심 개념</h4>
            <ul className="space-y-1 text-green-700 dark:text-green-300">
              <li>✓ FaaS: Lambda, Azure Functions, Cloud Functions</li>
              <li>✓ 컨테이너 서버리스: Cloud Run, Fargate</li>
              <li>✓ Pay-per-Use: 사용한 만큼만 과금</li>
              <li>✓ 자동 스케일링: 0 → 무한대</li>
            </ul>
          </div>
          <div className="space-y-2">
            <h4 className="font-semibold text-green-800 dark:text-green-200">사용 사례</h4>
            <ul className="space-y-1 text-green-700 dark:text-green-300">
              <li>✓ API 백엔드 (모바일, SaaS)</li>
              <li>✓ 이벤트 처리 (파일 업로드, ETL)</li>
              <li>✓ 실시간 스트림 프로세싱</li>
              <li>✓ 스케줄 작업 (cron job)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 문서',
            icon: 'web' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'AWS Lambda Developer Guide',
                url: 'https://docs.aws.amazon.com/lambda/',
                description: 'AWS Lambda 완전 가이드 및 모범 사례'
              },
              {
                title: 'Azure Functions Documentation',
                url: 'https://learn.microsoft.com/en-us/azure/azure-functions/',
                description: 'Azure Functions 및 Durable Functions 문서'
              },
              {
                title: 'Google Cloud Functions',
                url: 'https://cloud.google.com/functions/docs',
                description: 'Cloud Functions 2nd Gen 가이드'
              },
              {
                title: 'Cloud Run Documentation',
                url: 'https://cloud.google.com/run/docs',
                description: '컨테이너 기반 서버리스 완전 가이드'
              }
            ]
          },
          {
            title: '📖 핵심 서적 & 아티클',
            icon: 'research' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Serverless Architectures on AWS (Manning)',
                url: 'https://www.manning.com/books/serverless-architectures-on-aws-second-edition',
                description: 'AWS 서버리스 아키텍처 설계 실전 가이드'
              },
              {
                title: 'AWS Lambda Best Practices',
                url: 'https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html',
                description: 'AWS 공식 Lambda 모범 사례'
              },
              {
                title: 'Serverless Stack (SST)',
                url: 'https://sst.dev/',
                description: '풀스택 서버리스 프레임워크 및 가이드'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 프레임워크',
            icon: 'tools' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'Serverless Framework',
                url: 'https://www.serverless.com/',
                description: '멀티 클라우드 서버리스 배포 도구'
              },
              {
                title: 'AWS SAM (Serverless Application Model)',
                url: 'https://aws.amazon.com/serverless/sam/',
                description: 'AWS 공식 서버리스 배포 프레임워크'
              },
              {
                title: 'AWS CDK',
                url: 'https://aws.amazon.com/cdk/',
                description: '코드로 인프라 정의 (TypeScript, Python 등)'
              },
              {
                title: 'Terraform',
                url: 'https://www.terraform.io/',
                description: '멀티 클라우드 IaC 도구'
              }
            ]
          },
          {
            title: '🎓 학습 리소스',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'AWS Serverless Land',
                url: 'https://serverlessland.com/',
                description: 'AWS 서버리스 패턴 및 샘플 코드 모음'
              },
              {
                title: 'Serverless Guru Blog',
                url: 'https://www.serverlessguru.com/blog',
                description: '서버리스 아키텍처 실전 사례 및 팁'
              },
              {
                title: 'A Cloud Guru - Serverless',
                url: 'https://acloudguru.com/courses?q=serverless',
                description: '서버리스 인증 과정 및 핸즈온 랩'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
