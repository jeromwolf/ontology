'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, BookOpen, Database, Cpu, TrendingUp, Server, BarChart3 } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter1Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/intermediate"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          중급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-teal-500 to-cyan-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Database size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 1: 고급 벡터 데이터베이스</h1>
              <p className="text-cyan-100 text-lg">프로덕션 환경을 위한 벡터 DB 선택과 최적화</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Vector DB Comparison */}
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

        {/* Section 2: Performance Benchmarks */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
              <BarChart3 className="text-blue-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.2 성능 벤치마크</h2>
              <p className="text-gray-600 dark:text-gray-400">실제 워크로드 기반 성능 비교</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">테스트 환경</h3>
              <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-300">
                <li>• <strong>데이터셋:</strong> 100만개 벡터 (768차원)</li>
                <li>• <strong>하드웨어:</strong> 16 vCPU, 64GB RAM, NVMe SSD</li>
                <li>• <strong>인덱스 타입:</strong> HNSW (Hierarchical Navigable Small World)</li>
                <li>• <strong>측정 지표:</strong> QPS, Latency, Recall@10</li>
              </ul>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
                <h4 className="font-bold text-gray-900 dark:text-white mb-4">쿼리 성능 (QPS)</h4>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Qdrant</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{width: '95%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">12,000</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Pinecone</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{width: '88%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">11,000</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Milvus</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{width: '80%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">10,000</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Weaviate</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-yellow-500 h-2 rounded-full" style={{width: '64%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">8,000</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Chroma</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-orange-500 h-2 rounded-full" style={{width: '40%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">5,000</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
                <h4 className="font-bold text-gray-900 dark:text-white mb-4">레이턴시 (P99, ms)</h4>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Qdrant</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{width: '20%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">2ms</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Pinecone</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{width: '25%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">2.5ms</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Milvus</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-yellow-500 h-2 rounded-full" style={{width: '35%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">3.5ms</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Weaviate</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-yellow-500 h-2 rounded-full" style={{width: '40%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">4ms</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Chroma</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div className="bg-orange-500 h-2 rounded-full" style={{width: '60%'}}></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">6ms</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Index Types and Optimization */}
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

        {/* Section 4: Production Deployment */}
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

        {/* Section 5: Practical Exercise */}
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
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/intermediate"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            중급 과정으로
          </Link>
          
          <Link
            href="/modules/rag/intermediate/chapter2"
            className="inline-flex items-center gap-2 bg-teal-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-teal-600 transition-colors"
          >
            다음: 하이브리드 검색 전략
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}