'use client'

import { BarChart2 } from 'lucide-react'

export default function Section4() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
          <BarChart2 className="text-green-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.4 실제 적용 사례와 성능 향상</h2>
          <p className="text-gray-600 dark:text-gray-400">기업들의 하이브리드 검색 도입 결과</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">🏢 이커머스 플랫폼 사례</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">문제 상황</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 상품명/SKU 검색 정확도 낮음</li>
                <li>• "빨간 운동화" → "레드 스니커즈" 매칭 안됨</li>
                <li>• 브랜드명 오타 처리 불가</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">개선 결과</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 검색 정확도 35% 향상</li>
                <li>• 클릭률(CTR) 28% 증가</li>
                <li>• 검색 포기율 40% 감소</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 p-4 bg-emerald-100 dark:bg-emerald-900/40 rounded-lg">
            <p className="text-sm text-emerald-800 dark:text-emerald-200">
              <strong>핵심 전략:</strong> SKU, 브랜드명은 BM25로, 상품 설명은 벡터 검색으로 처리.
              쿼리 타입에 따라 가중치 동적 조정.
            </p>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">📚 기술 문서 검색 시스템</h3>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">구현 상세</h4>
            <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
              <pre className="text-xs text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`# 문서 타입별 가중치 설정
WEIGHT_CONFIG = {
    "api_reference": {"bm25": 0.7, "vector": 0.3},  # 함수명, 파라미터 중요
    "tutorial": {"bm25": 0.3, "vector": 0.7},       # 개념 설명 중요
    "error_guide": {"bm25": 0.6, "vector": 0.4},    # 에러 코드 중요
    "conceptual": {"bm25": 0.2, "vector": 0.8}      # 의미 이해 중요
}

# 메타데이터 부스팅
if "error" in query and doc.type == "error_guide":
    score *= 1.5  # 에러 관련 쿼리는 에러 가이드 문서 우선`}
              </pre>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <p className="text-2xl font-bold text-blue-600">92%</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">정답 포함률 (Top 5)</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <p className="text-2xl font-bold text-blue-600">1.2초</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">평균 응답 시간</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <p className="text-2xl font-bold text-blue-600">4.7/5</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">사용자 만족도</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
