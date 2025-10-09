'use client'

import References from '@/components/common/References'

export default function Section7() {
  return (
    <>
      {/* Practical Exercise */}
      <section className="bg-gradient-to-r from-teal-500 to-cyan-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">실습 과제</h2>

        <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
          <h3 className="font-bold mb-4">벡터 DB 성능 비교 실습</h3>
          <ol className="space-y-3 text-sm">
            <li>1. Chroma와 Qdrant를 로컬에 설치하고 동일한 데이터셋 적재</li>
            <li>2. 각각에서 1000개의 쿼리를 실행하고 성능 측정</li>
            <li>3. 인덱스 파라미터를 조정하여 성능 최적화</li>
            <li>4. 메타데이터 필터링을 추가하여 하이브리드 검색 구현</li>
            <li>5. 결과를 그래프로 시각화하고 비교 분석 리포트 작성</li>
          </ol>

          <div className="mt-6 p-4 bg-white/10 rounded-lg">
            <p className="text-xs">
              💡 <strong>힌트:</strong> locust나 k6 같은 부하 테스트 도구를 사용하면 더 정확한 벤치마크가 가능합니다.
            </p>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 벡터 DB 공식 문서',
            icon: 'web' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Pinecone Documentation',
                authors: 'Pinecone',
                year: '2025',
                description: '완전 관리형 벡터 DB - 설치 불필요',
                link: 'https://docs.pinecone.io/'
              },
              {
                title: 'Qdrant Documentation',
                authors: 'Qdrant',
                year: '2025',
                description: 'Rust 기반 고성능 벡터 검색 엔진',
                link: 'https://qdrant.tech/documentation/'
              },
              {
                title: 'Weaviate Documentation',
                authors: 'Weaviate',
                year: '2025',
                description: 'GraphQL API 기반 벡터 데이터베이스',
                link: 'https://weaviate.io/developers/weaviate'
              },
              {
                title: 'Chroma Documentation',
                authors: 'Chroma',
                year: '2025',
                description: '개발 친화적 임베디드 벡터 스토어',
                link: 'https://docs.trychroma.com/'
              },
              {
                title: 'Milvus Documentation',
                authors: 'Milvus',
                year: '2025',
                description: '클라우드 네이티브 대규모 벡터 DB',
                link: 'https://milvus.io/docs'
              }
            ]
          },
          {
            title: '📖 벡터 DB 벤치마크 & 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Vector Database Benchmark (VectorDBBench)',
                authors: 'Zilliz',
                year: '2024',
                description: '주요 벡터 DB 성능 비교 벤치마크',
                link: 'https://zilliz.com/vector-database-benchmark-tool'
              },
              {
                title: 'ANN Benchmarks',
                authors: 'Erik Bernhardsson',
                year: '2024',
                description: 'Approximate Nearest Neighbor 알고리즘 벤치마크',
                link: 'https://ann-benchmarks.com/'
              },
              {
                title: 'HNSW: Efficient and robust ANN search',
                authors: 'Malkov, Y., Yashunin, D.',
                year: '2018',
                description: 'HNSW 알고리즘 원조 논문',
                link: 'https://arxiv.org/abs/1603.09320'
              },
              {
                title: 'Product Quantization for Nearest Neighbor Search',
                authors: 'Jégou, H., et al.',
                year: '2011',
                description: 'PQ 압축 기법 논문 (메모리 효율)',
                link: 'https://ieeexplore.ieee.org/document/5432202'
              }
            ]
          },
          {
            title: '🛠️ 프로덕션 배포 & 최적화',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Qdrant Kubernetes Deployment',
                authors: 'Qdrant',
                year: '2024',
                description: 'K8s 환경에서 Qdrant 클러스터 구축',
                link: 'https://qdrant.tech/documentation/guides/distributed-deployment/'
              },
              {
                title: 'Pinecone Performance Optimization',
                authors: 'Pinecone',
                year: '2024',
                description: '쿼리 성능 튜닝 가이드',
                link: 'https://docs.pinecone.io/docs/performance-tuning'
              },
              {
                title: 'Weaviate Backup & Recovery',
                authors: 'Weaviate',
                year: '2024',
                description: '벡터 데이터 백업 및 복구 전략',
                link: 'https://weaviate.io/developers/weaviate/configuration/backups'
              },
              {
                title: 'Milvus GPU Acceleration',
                authors: 'Milvus',
                year: '2024',
                description: 'CUDA 기반 벡터 검색 가속화',
                link: 'https://milvus.io/docs/gpu_index.md'
              },
              {
                title: 'Vector DB Cost Optimization',
                authors: 'LangChain Blog',
                year: '2024',
                description: '벡터 DB 비용 최적화 전략',
                link: 'https://blog.langchain.dev/vector-database-cost-optimization/'
              }
            ]
          }
        ]}
      />
    </>
  )
}
