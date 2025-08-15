'use client';

import React from 'react';
import { 
  Box, Network, Activity, Zap
} from 'lucide-react';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      {/* Microservices Overview */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Box className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          마이크로서비스 아키텍처
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            마이크로서비스는 작고 독립적으로 배포 가능한 서비스들로 구성된 아키텍처 스타일입니다.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-red-50 dark:bg-red-950/20 rounded-lg p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                모놀리스 아키텍처
              </h3>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>✅ 개발 초기 단순함</li>
                <li>✅ 디버깅 용이</li>
                <li>✅ 트랜잭션 관리 간단</li>
                <li>❌ 확장성 제한</li>
                <li>❌ 기술 스택 고정</li>
                <li>❌ 배포 리스크 높음</li>
              </ul>
            </div>
            
            <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                마이크로서비스 아키텍처
              </h3>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>✅ 독립적 확장</li>
                <li>✅ 기술 다양성</li>
                <li>✅ 장애 격리</li>
                <li>❌ 운영 복잡도</li>
                <li>❌ 네트워크 지연</li>
                <li>❌ 데이터 일관성</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* API Gateway Pattern */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Network className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          API Gateway 패턴
        </h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              API Gateway 역할
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 단일 진입점 제공</li>
                <li>• 인증/인가 처리</li>
                <li>• 요청 라우팅</li>
                <li>• 프로토콜 변환</li>
              </ul>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• Rate Limiting</li>
                <li>• 캐싱</li>
                <li>• 모니터링/분석</li>
                <li>• 응답 집계</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              BFF (Backend for Frontend) 패턴
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              각 클라이언트 타입별로 최적화된 API Gateway 제공
            </p>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs">
              Mobile App → Mobile BFF → Microservices<br/>
              Web App → Web BFF → Microservices<br/>
              Desktop App → Desktop BFF → Microservices
            </div>
          </div>
        </div>
      </section>

      {/* Service Discovery */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Activity className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          서비스 디스커버리
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              클라이언트 사이드 디스커버리
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              Client → Service Registry<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
              Client → Service Instance
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 클라이언트가 직접 서비스 위치 조회</li>
              <li>• Netflix Eureka, Consul</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              서버 사이드 디스커버리
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              Client → Load Balancer<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Service Instance
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 로드 밸런서가 서비스 라우팅</li>
              <li>• AWS ELB, Kubernetes Service</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Circuit Breaker Pattern */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Zap className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          Circuit Breaker 패턴
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            장애가 발생한 서비스로의 요청을 차단하여 연쇄 장애를 방지하는 패턴입니다.
          </p>
          
          <div className="bg-gradient-to-r from-green-50 to-yellow-50 to-red-50 dark:from-green-950/20 dark:via-yellow-950/20 dark:to-red-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              Circuit Breaker 상태
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <div className="text-green-600 dark:text-green-400 font-semibold mb-2">
                  Closed (정상)
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  모든 요청 통과<br/>
                  실패 카운트 모니터링
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <div className="text-yellow-600 dark:text-yellow-400 font-semibold mb-2">
                  Half-Open (확인)
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  일부 요청만 통과<br/>
                  복구 여부 확인
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <div className="text-red-600 dark:text-red-400 font-semibold mb-2">
                  Open (차단)
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  모든 요청 차단<br/>
                  즉시 에러 반환
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Retry 전략
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>Exponential Backoff:</strong> 2초 → 4초 → 8초 → 16초</li>
              <li>• <strong>Jitter:</strong> 랜덤 지연 추가로 동시 재시도 방지</li>
              <li>• <strong>Retry Budget:</strong> 전체 요청의 10% 이하로 재시도 제한</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}