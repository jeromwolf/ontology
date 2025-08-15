'use client';

import VectorSearchDemo from '../VectorSearchDemo';

// Chapter 4: Vector Search
export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">벡터 검색의 원리</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          벡터 검색은 고차원 공간에서 가장 가까운 이웃을 찾는 과정입니다.
          쿼리 벡터와 문서 벡터 간의 거리를 계산하여 가장 유사한 문서를 찾습니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">벡터 검색 실습</h2>
        <VectorSearchDemo />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주요 벡터 데이터베이스 비교</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Pinecone</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 완전 관리형 서비스</li>
              <li>✓ 실시간 업데이트</li>
              <li>✓ 하이브리드 검색 지원</li>
              <li>✗ 클라우드 전용</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Weaviate</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 오픈소스</li>
              <li>✓ GraphQL API</li>
              <li>✓ 모듈식 아키텍처</li>
              <li>✓ 온프레미스 가능</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Chroma</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 경량 임베디드 DB</li>
              <li>✓ Python 네이티브</li>
              <li>✓ 개발자 친화적</li>
              <li>✗ 대규모 확장 제한</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Qdrant</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ Rust 기반 고성능</li>
              <li>✓ 필터링 기능 강력</li>
              <li>✓ 클라우드 & 온프레미스</li>
              <li>✓ gRPC API</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}