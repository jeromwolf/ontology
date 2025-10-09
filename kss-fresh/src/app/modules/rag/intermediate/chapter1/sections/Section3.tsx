'use client'

import { Cpu } from 'lucide-react'

export default function Section3() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
          <Cpu className="text-purple-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.3 인덱스 타입과 최적화</h2>
          <p className="text-gray-600 dark:text-gray-400">검색 성능과 정확도의 균형 맞추기</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">주요 인덱스 알고리즘</h3>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">1. HNSW (Hierarchical Navigable Small World)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                계층적 그래프 구조로 빠른 근사 최근접 이웃 검색
              </p>
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded">
                <pre className="text-xs overflow-x-auto">
{`# Qdrant에서 HNSW 인덱스 설정
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfig

client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    ),
    hnsw_config=HnswConfig(
        m=16,  # 각 노드의 연결 수
        ef_construct=200,  # 인덱스 구축 시 검색 깊이
        full_scan_threshold=10000
    )
)`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">2. IVF (Inverted File Index)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                벡터를 클러스터로 분할하여 검색 공간 축소
              </p>
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded">
                <pre className="text-xs overflow-x-auto">
{`# Milvus에서 IVF_FLAT 인덱스 설정
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {
        "nlist": 1024  # 클러스터 수
    }
}

collection.create_index(
    field_name="embeddings",
    index_params=index_params
)`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">3. Product Quantization (PQ)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                벡터를 압축하여 메모리 사용량 감소
              </p>
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded">
                <pre className="text-xs overflow-x-auto">
{`# Weaviate에서 PQ 설정
{
    "class": "Document",
    "vectorIndexConfig": {
        "pq": {
            "enabled": true,
            "trainingLimit": 100000,
            "segments": 768  # 차원을 나눌 세그먼트 수
        }
    }
}`}
                </pre>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-amber-50 dark:bg-amber-900/20 p-6 rounded-xl border border-amber-200 dark:border-amber-700">
          <h3 className="font-bold text-amber-800 dark:text-amber-200 mb-4">최적화 팁</h3>
          <ul className="space-y-3 text-sm text-amber-700 dark:text-amber-300">
            <li className="flex items-start gap-2">
              <span className="text-amber-600">•</span>
              <div>
                <strong>배치 처리:</strong> 개별 쿼리보다 배치로 처리하면 처리량 3-5배 향상
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600">•</span>
              <div>
                <strong>프리필터링:</strong> 메타데이터 필터를 먼저 적용하여 검색 공간 축소
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600">•</span>
              <div>
                <strong>인덱스 파라미터 튜닝:</strong> 데이터 크기와 쿼리 패턴에 맞춰 조정
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600">•</span>
              <div>
                <strong>캐싱 전략:</strong> 자주 사용되는 쿼리 결과는 Redis 등에 캐싱
              </div>
            </li>
          </ul>
        </div>
      </div>
    </section>
  )
}
