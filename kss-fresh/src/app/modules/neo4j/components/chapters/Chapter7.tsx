'use client';

import React from 'react';
import { Zap, BarChart3, Shield, Gauge } from 'lucide-react';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">성능 최적화와 운영 ⚡</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          대규모 Neo4j 환경에서 최고 성능을 발휘하기 위한 
          전문가 수준의 최적화 기법을 마스터하세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🚀 쿼리 최적화 전략</h2>
        <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">고급 Cypher 최적화</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-medium mb-2">인덱스 전략</h4>
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm font-mono">
                CREATE INDEX FOR (n:Person) ON (n.name, n.age)
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                복합 인덱스로 검색 성능 10배 향상
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-medium mb-2">쿼리 힌트 활용</h4>
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm font-mono">
                MATCH (n:Person) USING INDEX n:Person(name) WHERE n.name = 'Alice'
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                명시적 인덱스 사용으로 실행 계획 최적화
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 모니터링 및 알림</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
            <BarChart3 className="text-blue-500 mb-3" size={28} />
            <h3 className="font-semibold mb-3">실시간 메트릭</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>쿼리 처리량</span>
                <span className="font-mono">1,234 QPS</span>
              </div>
              <div className="flex justify-between">
                <span>메모리 사용률</span>
                <span className="font-mono text-green-600">68%</span>
              </div>
              <div className="flex justify-between">
                <span>캐시 적중률</span>
                <span className="font-mono text-blue-600">94.2%</span>
              </div>
            </div>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6">
            <Shield className="text-emerald-500 mb-3" size={28} />
            <h3 className="font-semibold mb-3">자동 알림</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-sm">시스템 정상 운영</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
                <span className="text-sm">메모리 사용량 주의</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-gray-300 rounded-full"></div>
                <span className="text-sm">백업 완료</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚙️ 클러스터 관리</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">고가용성 아키텍처</h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
              <Gauge className="text-purple-500 mx-auto mb-2" size={24} />
              <h4 className="font-medium">리더 인스턴스</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">쓰기 작업 전담</p>
            </div>
            <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
              <BarChart3 className="text-blue-500 mx-auto mb-2" size={24} />
              <h4 className="font-medium">읽기 복제본</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">읽기 부하 분산</p>
            </div>
            <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
              <Shield className="text-emerald-500 mx-auto mb-2" size={24} />
              <h4 className="font-medium">백업 노드</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">장애 복구</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 성능 튜닝 체크리스트</h2>
        <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-3 text-emerald-600">✅ 필수 최적화</h3>
              <ul className="space-y-2 text-sm">
                <li>• JVM 힙 메모리 설정 (전체 메모리의 50%)</li>
                <li>• 페이지 캐시 최적화 (남은 메모리 활용)</li>
                <li>• 트랜잭션 로그 크기 조정</li>
                <li>• 커넥션 풀 크기 설정</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-3 text-blue-600">🔧 고급 설정</h3>
              <ul className="space-y-2 text-sm">
                <li>• 압축 알고리즘 선택</li>
                <li>• 병렬 처리 스레드 조정</li>
                <li>• 디스크 I/O 최적화</li>
                <li>• 네트워크 버퍼 튜닝</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}