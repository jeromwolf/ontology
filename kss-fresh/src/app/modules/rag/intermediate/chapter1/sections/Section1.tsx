'use client'

import { Database } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-teal-100 dark:bg-teal-900/20 flex items-center justify-center">
          <Database className="text-teal-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.1 주요 벡터 데이터베이스 비교</h2>
          <p className="text-gray-600 dark:text-gray-400">각 벡터 DB의 특징과 사용 사례</p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-gray-100 dark:bg-gray-700">
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">벡터 DB</th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">특징</th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">장점</th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">단점</th>
              <th className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left">적합한 사례</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            <tr>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">Pinecone</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">완전 관리형 SaaS</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">설치 불필요, 자동 스케일링</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">비용 높음, 종속성</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">빠른 MVP, 스타트업</td>
            </tr>
            <tr className="bg-gray-50 dark:bg-gray-700/50">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">Chroma</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">오픈소스, 임베디드</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">간단한 설치, 개발 친화적</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">스케일링 제한</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">프로토타입, 소규모</td>
            </tr>
            <tr>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">Weaviate</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">GraphQL API, 모듈형</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">다양한 모듈, 하이브리드 검색</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">복잡한 설정</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">엔터프라이즈, 복잡한 쿼리</td>
            </tr>
            <tr className="bg-gray-50 dark:bg-gray-700/50">
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">Qdrant</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">Rust 기반, 고성능</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">빠른 속도, 메모리 효율</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">상대적으로 새로움</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">고성능 요구사항</td>
            </tr>
            <tr>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3 font-medium">Milvus</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">클라우드 네이티브</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">대규모 확장성, GPU 지원</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">리소스 집약적</td>
              <td className="border border-gray-300 dark:border-gray-600 px-4 py-3">대규모 엔터프라이즈</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  )
}
