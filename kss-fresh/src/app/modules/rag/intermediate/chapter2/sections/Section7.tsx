'use client'

import References from '@/components/common/References'

export default function Section7() {
  return (
    <>
      {/* Practical Exercise */}
      <section className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">실습 과제</h2>

        <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
          <h3 className="font-bold mb-4">하이브리드 검색 시스템 구축</h3>

          <div className="space-y-4">
            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">📋 요구사항</h4>
              <ol className="space-y-2 text-sm">
                <li>1. Elasticsearch와 벡터 DB를 사용한 하이브리드 검색 구현</li>
                <li>2. 한국어 형태소 분석기 적용 (Nori, Komoran 등)</li>
                <li>3. 쿼리 타입 자동 분류 (키워드형, 자연어형, 혼합형)</li>
                <li>4. A/B 테스트를 통한 최적 가중치 찾기</li>
                <li>5. 검색 품질 메트릭 측정 (MRR, NDCG, Precision@K)</li>
              </ol>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🎯 평가 데이터셋</h4>
              <ul className="space-y-1 text-sm">
                <li>• 1000개의 문서 (뉴스, 제품 설명, FAQ 혼합)</li>
                <li>• 100개의 테스트 쿼리와 정답 셋</li>
                <li>• 키워드형 30%, 자연어형 50%, 혼합형 20%</li>
              </ul>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">💡 도전 과제</h4>
              <p className="text-sm">
                검색 로그를 분석하여 사용자의 검색 패턴을 학습하고,
                개인화된 가중치를 적용하는 시스템으로 확장해보세요.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 하이브리드 검색 공식 문서',
            icon: 'web' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Elasticsearch Hybrid Search',
                authors: 'Elastic',
                year: '2025',
                description: 'BM25 + kNN 벡터 검색 결합 - 프로덕션급 구현',
                link: 'https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html'
              },
              {
                title: 'Weaviate Hybrid Search',
                authors: 'Weaviate',
                year: '2025',
                description: 'BM25 + ANN 벡터 검색 - 실시간 융합 알고리즘',
                link: 'https://weaviate.io/developers/weaviate/search/hybrid'
              },
              {
                title: 'LangChain Ensemble Retriever',
                authors: 'LangChain',
                year: '2025',
                description: '다중 리트리버 결합 - RRF(Reciprocal Rank Fusion)',
                link: 'https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble'
              },
              {
                title: 'LlamaIndex Hybrid Retriever',
                authors: 'LlamaIndex',
                year: '2025',
                description: '벡터 + 키워드 검색 통합 - 자동 가중치 조정',
                link: 'https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/'
              },
              {
                title: 'Pinecone Sparse-Dense Search',
                authors: 'Pinecone',
                year: '2025',
                description: '단일 인덱스에서 하이브리드 검색 - 통합 API',
                link: 'https://docs.pinecone.io/docs/hybrid-search'
              }
            ]
          },
          {
            title: '📖 검색 알고리즘 연구 논문',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'BM25: The Probabilistic Relevance Framework',
                authors: 'Robertson & Zaragoza',
                year: '2009',
                description: 'BM25 알고리즘의 이론적 기반 - TF-IDF 확률론적 개선',
                link: 'https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf'
              },
              {
                title: 'Dense Passage Retrieval for Open-Domain QA',
                authors: 'Karpukhin et al., Meta AI',
                year: '2020',
                description: '벡터 검색 기반 QA - BM25 대비 9-19% 성능 향상',
                link: 'https://arxiv.org/abs/2004.04906'
              },
              {
                title: 'Reciprocal Rank Fusion (RRF)',
                authors: 'Cormack et al.',
                year: '2009',
                description: '다중 순위 결합 알고리즘 - 점수 정규화 불필요',
                link: 'https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf'
              },
              {
                title: 'BEIR: Heterogeneous Benchmark for IR',
                authors: 'Thakur et al.',
                year: '2021',
                description: '18개 데이터셋으로 하이브리드 검색 벤치마크',
                link: 'https://arxiv.org/abs/2104.08663'
              }
            ]
          },
          {
            title: '🛠️ 실전 구현 & 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'rank_bm25 (Python)',
                authors: 'dorianbrown',
                year: '2024',
                description: '순수 Python BM25 구현 - Gensim 기반, 한국어 지원',
                link: 'https://github.com/dorianbrown/rank_bm25'
              },
              {
                title: 'Haystack Hybrid Retrieval',
                authors: 'deepset',
                year: '2025',
                description: 'NLP 프레임워크 - BM25 + DPR 통합 파이프라인',
                link: 'https://haystack.deepset.ai/tutorials/08_preprocessing'
              },
              {
                title: 'Qdrant Hybrid Search',
                authors: 'Qdrant',
                year: '2025',
                description: 'Rust 기반 벡터 DB - 빠른 하이브리드 검색 API',
                link: 'https://qdrant.tech/documentation/concepts/hybrid-queries/'
              },
              {
                title: 'ColBERT: Efficient Passage Retrieval',
                authors: 'Stanford NLP',
                year: '2023',
                description: '지연 상호작용 모델 - 벡터+키워드 장점 결합',
                link: 'https://github.com/stanford-futuredata/ColBERT'
              },
              {
                title: 'Vespa Hybrid Search',
                authors: 'Vespa.ai',
                year: '2025',
                description: '대규모 검색 엔진 - BM25 + ANN + 신경망 순위화',
                link: 'https://docs.vespa.ai/en/reference/ranking-expressions.html'
              }
            ]
          }
        ]}
      />
    </>
  )
}
