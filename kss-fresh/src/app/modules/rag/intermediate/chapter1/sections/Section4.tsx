'use client'

import { Server } from 'lucide-react'

export default function Section4() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
          <Server className="text-green-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.4 벡터 DB 운영 및 유지보수</h2>
          <p className="text-gray-600 dark:text-gray-400">안정적인 벡터 데이터베이스 운영 방법</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">벡터 DB 클러스터 구성</h3>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <pre className="text-xs overflow-x-auto">
{`┌─────────────────┐     ┌─────────────────┐
│   Load Balancer │     │   API Gateway   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ├───────────┬───────────┤
         │           │           │
    ┌────▼────┐ ┌───▼────┐ ┌───▼────┐
    │ Primary │ │Replica 1│ │Replica 2│
    │  Node   │ │  Node   │ │  Node   │
    └─────────┘ └─────────┘ └─────────┘
         │           │           │
    ┌────▼────────────▼───────────▼────┐
    │         Persistent Storage        │
    │        (S3, GCS, NFS, etc)       │
    └──────────────────────────────────┘`}
            </pre>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">벡터 데이터 분산 전략</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>벡터 샤딩:</strong> 차원별 또는 클러스터별 분산</li>
              <li>• <strong>인덱스 복제:</strong> 검색 성능과 가용성 향상</li>
              <li>• <strong>쿼리 라우팅:</strong> 효율적인 검색 경로</li>
              <li>• <strong>핫 데이터 관리:</strong> 자주 검색되는 벡터 캐싱</li>
            </ul>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">벡터 DB 성능 최적화</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>GPU 가속:</strong> CUDA/OpenCL 기반 벡터 연산</li>
              <li>• <strong>메모리 최적화:</strong> 벡터 압축과 메모리 맵핑</li>
              <li>• <strong>인덱스 튜닝:</strong> 정확도-속도 트레이드오프</li>
              <li>• <strong>배치 처리:</strong> 벡터 임베딩 배치 업데이트</li>
            </ul>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">벡터 DB 모니터링 지표</h3>
          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded">
            <pre className="text-xs overflow-x-auto">
{`# 벡터 데이터베이스 모니터링 체크리스트
✅ 벡터 검색 레이턴시 (P50, P95, P99)
✅ 벡터 인덱싱 처리량 (Vector/s)
✅ 인덱스 크기와 압축률
✅ 검색 정확도 (Recall@K)
✅ 벡터 캐시 히트율
✅ 임베딩 모델 응답시간
✅ 벡터 데이터 일관성
✅ 인덱스 재구축 상태`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
