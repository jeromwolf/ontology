'use client'

import { Search } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
          <Search className="text-indigo-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.1 하이브리드 검색이 필요한 이유</h2>
          <p className="text-gray-600 dark:text-gray-400">각 검색 방식의 장단점과 보완 관계</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
            <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">🔤 키워드 검색 (BM25)</h3>
            <div className="space-y-3 text-sm text-indigo-700 dark:text-indigo-300">
              <p><strong>장점:</strong></p>
              <ul className="space-y-1 pl-4">
                <li>• 정확한 단어 매칭</li>
                <li>• 희귀 용어, 고유명사에 강함</li>
                <li>• 검색 결과 설명 가능</li>
                <li>• 빠른 속도</li>
              </ul>
              <p className="mt-3"><strong>단점:</strong></p>
              <ul className="space-y-1 pl-4">
                <li>• 동의어 처리 어려움</li>
                <li>• 문맥 이해 부족</li>
                <li>• 철자 오류에 취약</li>
              </ul>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">🧠 벡터 검색 (Semantic)</h3>
            <div className="space-y-3 text-sm text-purple-700 dark:text-purple-300">
              <p><strong>장점:</strong></p>
              <ul className="space-y-1 pl-4">
                <li>• 의미적 유사성 파악</li>
                <li>• 동의어, 유사어 처리</li>
                <li>• 문맥 기반 이해</li>
                <li>• 다국어 지원</li>
              </ul>
              <p className="mt-3"><strong>단점:</strong></p>
              <ul className="space-y-1 pl-4">
                <li>• 고유명사, ID에 약함</li>
                <li>• 계산 비용 높음</li>
                <li>• 블랙박스 성격</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">실제 사례로 보는 차이점</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <p className="font-medium text-gray-900 dark:text-white mb-2">쿼리: "SKU-12345의 재고 현황"</p>
              <div className="grid md:grid-cols-2 gap-4 mt-3">
                <div>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">키워드 검색 ✅</p>
                  <p className="text-sm text-green-600">정확히 SKU-12345를 포함한 문서 검색</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">벡터 검색 ❌</p>
                  <p className="text-sm text-red-600">유사한 제품 코드들을 반환할 수 있음</p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <p className="font-medium text-gray-900 dark:text-white mb-2">쿼리: "차가운 음료"</p>
              <div className="grid md:grid-cols-2 gap-4 mt-3">
                <div>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">키워드 검색 ❌</p>
                  <p className="text-sm text-red-600">"차가운"과 "음료"를 정확히 포함한 문서만</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">벡터 검색 ✅</p>
                  <p className="text-sm text-green-600">"아이스커피", "냉음료", "시원한 음료" 등도 검색</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
