'use client';

import VectorSearchDemo from '../VectorSearchDemo';
import References from '@/components/common/References';

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

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-blue-800 dark:text-blue-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">벡터 검색의 원리와 거리 측정 방법 (코사인 유사도, 유클리드)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">주요 벡터 데이터베이스 비교 (Pinecone, Weaviate, Chroma, Qdrant)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">하이브리드 검색 (벡터 + 키워드) 전략</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">인덱싱 알고리즘과 검색 성능 최적화</span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 벡터 검색 핵심 논문 (Vector Search Papers)',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Efficient and Robust Approximate Nearest Neighbor Search Using HNSW',
                authors: 'Yu. A. Malkov, D. A. Yashunin',
                year: '2018',
                description: '현대 벡터 검색의 표준 - Hierarchical Navigable Small World 알고리즘',
                link: 'https://arxiv.org/abs/1603.09320'
              },
              {
                title: 'FAISS: A Library for Efficient Similarity Search',
                authors: 'Jeff Johnson, Matthijs Douze, Hervé Jégou (Meta AI)',
                year: '2019',
                description: 'Meta의 오픈소스 벡터 검색 라이브러리 - 10억개 벡터 검색',
                link: 'https://arxiv.org/abs/1702.08734'
              },
              {
                title: 'ScaNN: Efficient Vector Similarity Search',
                authors: 'Google Research',
                year: '2020',
                description: 'Google의 초고속 벡터 검색 알고리즘',
                link: 'https://arxiv.org/abs/1908.10396'
              },
              {
                title: 'Hybrid Search: Combining BM25 and Vector Search',
                authors: 'Various Contributors',
                year: '2023',
                description: '키워드 검색과 벡터 검색의 최적 결합 방법',
                link: 'https://www.pinecone.io/learn/hybrid-search-intro/'
              }
            ]
          },
          {
            title: '🛠️ 벡터 데이터베이스 (Vector Databases)',
            icon: 'tools',
            color: 'border-emerald-500',
            items: [
              {
                title: 'Pinecone',
                description: '완전 관리형 벡터 DB - 실시간 업데이트, 하이브리드 검색 (무료 티어 有)',
                link: 'https://www.pinecone.io/'
              },
              {
                title: 'Weaviate',
                description: '오픈소스 벡터 DB - GraphQL API, 모듈식 아키텍처',
                link: 'https://weaviate.io/'
              },
              {
                title: 'Qdrant',
                description: 'Rust 기반 고성능 벡터 DB - 강력한 필터링, gRPC API',
                link: 'https://qdrant.tech/'
              },
              {
                title: 'Chroma',
                description: '경량 임베디드 벡터 DB - Python 네이티브, 개발자 친화적',
                link: 'https://www.trychroma.com/'
              },
              {
                title: 'Milvus',
                description: '대규모 벡터 검색 - 수십억 벡터 지원, 클라우드 네이티브',
                link: 'https://milvus.io/'
              },
              {
                title: 'pgvector',
                description: 'PostgreSQL용 벡터 확장 - 기존 DB에 벡터 검색 추가',
                link: 'https://github.com/pgvector/pgvector'
              }
            ]
          },
          {
            title: '📖 벡터 DB 선택 가이드 (Selection Guides)',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'Vector Database Comparison',
                description: '12개 벡터 DB 성능/비용/기능 종합 비교',
                link: 'https://superlinked.com/vector-db-comparison/'
              },
              {
                title: 'Pinecone vs Weaviate vs Qdrant',
                description: '3대 벡터 DB 벤치마크 (속도, 정확도, 비용)',
                link: 'https://weaviate.io/blog/vector-database-benchmarks'
              },
              {
                title: 'Choosing a Vector Database',
                description: '사용 사례별 최적 벡터 DB 선택 전략',
                link: 'https://www.datastax.com/guides/what-is-a-vector-database'
              },
              {
                title: 'LangChain Vector Store Integration',
                description: '40+ 벡터 DB 통합 코드 예제',
                link: 'https://python.langchain.com/docs/integrations/vectorstores/'
              }
            ]
          },
          {
            title: '⚡ 성능 최적화 (Performance Optimization)',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'HNSW vs IVF vs Product Quantization',
                description: '인덱싱 알고리즘 성능 비교 및 튜닝 가이드',
                link: 'https://www.pinecone.io/learn/series/faiss/vector-indexes/'
              },
              {
                title: 'Hybrid Search Implementation',
                description: 'BM25 + 벡터 검색 결합으로 정확도 40% 향상',
                link: 'https://qdrant.tech/articles/hybrid-search/'
              },
              {
                title: 'Query Optimization Strategies',
                description: 'Top-K 검색, Re-ranking, 필터링 최적화',
                link: 'https://weaviate.io/developers/weaviate/search'
              }
            ]
          }
        ]}
      />
    </div>
  )
}