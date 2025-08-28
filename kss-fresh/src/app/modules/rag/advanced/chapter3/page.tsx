'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Server, Network, Database, Shield, Activity, Zap, Globe, HardDrive } from 'lucide-react'

export default function Chapter3Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/advanced"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          고급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-indigo-500 to-cyan-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Server size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 3: 분산 RAG 시스템</h1>
              <p className="text-indigo-100 text-lg">Netflix 규모의 RAG 시스템 구축 - 수십억 문서와 수만 QPS 처리</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: 분산 RAG 아키텍처의 필요성 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
              <Globe className="text-indigo-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.1 엔터프라이즈급 분산 RAG의 도전과제</h2>
              <p className="text-gray-600 dark:text-gray-400">단일 서버의 한계를 넘어서는 대규모 시스템</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
              <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">단일 노드 RAG의 물리적 한계</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>실제 Netflix의 콘텐츠 추천 시스템은 다음과 같은 규모를 처리합니다:</strong>
                </p>
                <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
                  <li><strong>2억+ 사용자</strong>: 실시간 개인화 추천 요구</li>
                  <li><strong>수십억 개의 콘텐츠 메타데이터</strong>: 영화, 시리즈, 자막, 리뷰</li>
                  <li><strong>초당 10만+ 쿼리</strong>: 피크 시간대 동시 접속</li>
                  <li><strong>99.99% 가용성 요구</strong>: 연간 52분 이하 다운타임</li>
                  <li><strong>&lt; 100ms 응답 시간</strong>: 사용자 경험을 위한 엄격한 레이턴시 요구</li>
                </ul>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-red-600 dark:text-red-400 mb-2">❌ 단일 노드 한계</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• RAM 용량 한계 (최대 수TB)</li>
                    <li>• CPU/GPU 처리 능력 제한</li>
                    <li>• 네트워크 대역폭 병목</li>
                    <li>• 장애 시 전체 시스템 마비</li>
                    <li>• 수직 확장의 비용 급증</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">✅ 분산 시스템 장점</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 무제한 수평 확장 가능</li>
                    <li>• 부하 분산으로 성능 향상</li>
                    <li>• 부분 장애 허용 (Fault Tolerance)</li>
                    <li>• 지역별 데이터 로컬리티</li>
                    <li>• 비용 효율적 확장</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">실제 사례: Uber의 분산 검색 시스템</h3>
              
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">Uber의 도전과제와 해결책</h4>
                <div className="grid md:grid-cols-3 gap-4 text-center">
                  <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded">
                    <p className="text-2xl font-bold text-blue-600">40억+</p>
                    <p className="text-xs text-blue-700 dark:text-blue-300">일일 검색 쿼리</p>
                  </div>
                  <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded">
                    <p className="text-2xl font-bold text-green-600">&lt;50ms</p>
                    <p className="text-xs text-green-700 dark:text-green-300">P99 레이턴시</p>
                  </div>
                  <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded">
                    <p className="text-2xl font-bold text-purple-600">99.95%</p>
                    <p className="text-xs text-purple-700 dark:text-purple-300">가용성 SLA</p>
                  </div>
                </div>
              </div>

              <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>아키텍처 핵심:</strong> Uber는 도시별로 분산된 검색 클러스터를 운영하며, 
                  각 클러스터는 해당 지역의 드라이버, 음식점, 경로 데이터를 처리합니다. 
                  글로벌 라우터가 사용자 위치에 따라 적절한 클러스터로 쿼리를 전달합니다.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: 분산 벡터 데이터베이스 설계 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Database className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.2 분산 벡터 데이터베이스 아키텍처</h2>
              <p className="text-gray-600 dark:text-gray-400">Faiss, Milvus, Qdrant를 활용한 대규모 벡터 검색</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">샤딩(Sharding) 전략</h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-3">1. 해시 기반 샤딩</h4>
                  <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded border border-slate-200 dark:border-slate-700 overflow-x-auto">
                    <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`# Consistent Hashing을 활용한 벡터 샤딩
import hashlib
import bisect
from typing import List, Tuple, Dict

class VectorShardRouter:
    def __init__(self, shards: List[str], virtual_nodes: int = 150):
        """
        분산 벡터 DB를 위한 Consistent Hashing 라우터
        virtual_nodes: 각 물리 노드당 가상 노드 수 (부하 균등화)
        """
        self.shards = shards
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        self._build_ring()
    
    def _hash(self, key: str) -> int:
        """MD5 해시를 사용한 32비트 정수 생성"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
    
    def _build_ring(self):
        """Consistent Hashing 링 구성"""
        for shard in self.shards:
            for i in range(self.virtual_nodes):
                virtual_key = f"{shard}:{i}"
                hash_value = self._hash(virtual_key)
                self.ring[hash_value] = shard
                bisect.insort(self.sorted_keys, hash_value)
    
    def get_shard(self, vector_id: str) -> str:
        """벡터 ID에 대한 샤드 결정"""
        if not self.ring:
            return None
            
        hash_value = self._hash(vector_id)
        idx = bisect.bisect(self.sorted_keys, hash_value)
        
        # 링의 끝에 도달하면 첫 번째 노드로
        if idx == len(self.sorted_keys):
            idx = 0
            
        return self.ring[self.sorted_keys[idx]]
    
    def add_shard(self, shard: str):
        """새로운 샤드 추가 (동적 확장)"""
        self.shards.append(shard)
        for i in range(self.virtual_nodes):
            virtual_key = f"{shard}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = shard
            bisect.insort(self.sorted_keys, hash_value)
    
    def remove_shard(self, shard: str):
        """샤드 제거 (장애 처리)"""
        self.shards.remove(shard)
        for i in range(self.virtual_nodes):
            virtual_key = f"{shard}:{i}"
            hash_value = self._hash(virtual_key)
            del self.ring[hash_value]
            self.sorted_keys.remove(hash_value)

# 사용 예제
router = VectorShardRouter([
    "vector-db-1.internal",
    "vector-db-2.internal", 
    "vector-db-3.internal"
])

# 벡터 저장 위치 결정
vector_id = "doc_12345_chunk_3"
target_shard = router.get_shard(vector_id)
print(f"Vector {vector_id} -> Shard: {target_shard}")`}
                    </pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-3">2. 의미 기반 샤딩 (Semantic Sharding)</h4>
                  <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded border border-slate-200 dark:border-slate-700 overflow-x-auto">
                    <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`# K-means 클러스터링을 활용한 의미 기반 샤딩
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple

class SemanticShardRouter:
    def __init__(self, n_shards: int, embedding_dim: int = 768):
        """
        의미적 유사성에 기반한 벡터 샤딩
        유사한 벡터들을 같은 샤드에 배치하여 검색 효율 향상
        """
        self.n_shards = n_shards
        self.embedding_dim = embedding_dim
        self.centroids = None
        self.shard_mapping = {}
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_shards,
            batch_size=10000,
            max_iter=100
        )
    
    def fit(self, sample_embeddings: np.ndarray):
        """샘플 임베딩으로 클러스터 중심점 학습"""
        print(f"Learning {self.n_shards} semantic clusters...")
        self.kmeans.fit(sample_embeddings)
        self.centroids = self.kmeans.cluster_centers_
        
        # 각 샤드의 통계 정보
        labels = self.kmeans.labels_
        for i in range(self.n_shards):
            count = np.sum(labels == i)
            print(f"Shard {i}: {count} vectors ({count/len(labels)*100:.1f}%)")
    
    def get_shard(self, embedding: np.ndarray) -> int:
        """벡터에 대한 최적 샤드 결정"""
        if self.centroids is None:
            raise ValueError("Router not fitted. Call fit() first.")
        
        # 가장 가까운 중심점 찾기
        distances = np.linalg.norm(self.centroids - embedding, axis=1)
        return int(np.argmin(distances))
    
    def get_relevant_shards(self, query_embedding: np.ndarray, 
                          top_k: int = 3) -> List[int]:
        """
        쿼리와 관련된 상위 k개 샤드 반환
        크로스 샤드 검색 시 사용
        """
        distances = np.linalg.norm(self.centroids - query_embedding, axis=1)
        return np.argsort(distances)[:top_k].tolist()

# Faiss를 활용한 분산 인덱스
class DistributedFaissIndex:
    def __init__(self, router: SemanticShardRouter, shard_urls: List[str]):
        self.router = router
        self.shard_urls = shard_urls
        self.shard_clients = self._init_clients()
    
    def _init_clients(self):
        """각 샤드에 대한 gRPC 클라이언트 초기화"""
        import grpc
        clients = {}
        for i, url in enumerate(self.shard_urls):
            channel = grpc.insecure_channel(url)
            clients[i] = FaissServiceStub(channel)
        return clients
    
    async def add_vectors(self, ids: List[str], embeddings: np.ndarray):
        """벡터를 적절한 샤드에 분산 저장"""
        shard_batches = defaultdict(list)
        
        # 각 벡터를 적절한 샤드로 라우팅
        for i, (id_, embedding) in enumerate(zip(ids, embeddings)):
            shard_id = self.router.get_shard(embedding)
            shard_batches[shard_id].append((id_, embedding))
        
        # 병렬로 각 샤드에 저장
        tasks = []
        for shard_id, batch in shard_batches.items():
            client = self.shard_clients[shard_id]
            task = client.add_vectors_async(batch)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def search(self, query_embedding: np.ndarray, 
                    k: int = 10, search_shards: int = 3) -> List[Tuple[str, float]]:
        """
        분산 검색 수행
        1. 관련성 높은 샤드 선택
        2. 병렬 검색 수행
        3. 결과 병합 및 재순위화
        """
        # 검색할 샤드 결정
        relevant_shards = self.router.get_relevant_shards(
            query_embedding, top_k=search_shards
        )
        
        # 병렬 검색
        tasks = []
        for shard_id in relevant_shards:
            client = self.shard_clients[shard_id]
            task = client.search_async(query_embedding, k=k*2)
            tasks.append(task)
        
        # 결과 수집 및 병합
        all_results = []
        shard_results = await asyncio.gather(*tasks)
        for results in shard_results:
            all_results.extend(results)
        
        # 점수 기준 재정렬
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]`}
                    </pre>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">Milvus 분산 아키텍처 실전 구성</h3>
              
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">프로덕션 Milvus 클러스터 구성</h4>
                <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded border border-slate-200 dark:border-slate-700 overflow-x-auto">
                  <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`# Kubernetes에서 Milvus 분산 클러스터 배포
apiVersion: v1
kind: ConfigMap
metadata:
  name: milvus-config
data:
  milvus.yaml: |
    etcd:
      endpoints:
        - etcd-0.etcd-headless.milvus.svc.cluster.local:2379
        - etcd-1.etcd-headless.milvus.svc.cluster.local:2379
        - etcd-2.etcd-headless.milvus.svc.cluster.local:2379
    
    minio:
      address: minio-service.milvus.svc.cluster.local
      port: 9000
      bucketName: milvus-data
    
    pulsar:
      address: pulsar-proxy.milvus.svc.cluster.local
      port: 6650
      maxMessageSize: 5242880  # 5MB
    
    # 성능 최적화 설정
    queryNode:
      gracefulTime: 5000  # 5초
      cache:
        enabled: true
        memoryLimit: 32GB  # 각 쿼리 노드 캐시
    
    indexNode:
      scheduler:
        buildParallel: 8  # 병렬 인덱스 빌드
    
    dataNode:
      segment:
        maxSize: 1024  # MB
        sealProportion: 0.75
    
    # 자동 압축 및 최적화
    dataCoord:
      compaction:
        enable: true
        globalInterval: 3600  # 1시간

---
# QueryNode 스테이트풀셋 (검색 처리)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: milvus-querynode
spec:
  serviceName: milvus-querynode
  replicas: 6  # 검색 부하에 따라 조정
  template:
    spec:
      containers:
      - name: querynode
        image: milvusdb/milvus:v2.3.0
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
          limits:
            memory: "64Gi"
            cpu: "16"
        env:
        - name: MILVUS_NODE_TYPE
          value: "querynode"
        volumeMounts:
        - name: cache
          mountPath: /var/lib/milvus/cache
  volumeClaimTemplates:
  - metadata:
      name: cache
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 500Gi  # NVMe SSD 권장`}
                    </pre>
                  </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">컴포넌트별 역할</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li><strong>QueryNode:</strong> 벡터 검색 수행</li>
                    <li><strong>DataNode:</strong> 데이터 삽입/삭제 처리</li>
                    <li><strong>IndexNode:</strong> 인덱스 구축</li>
                    <li><strong>Proxy:</strong> 클라이언트 요청 라우팅</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">확장 전략</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li><strong>읽기 확장:</strong> QueryNode 증설</li>
                    <li><strong>쓰기 확장:</strong> DataNode 증설</li>
                    <li><strong>저장 확장:</strong> MinIO 클러스터 확장</li>
                    <li><strong>메시징:</strong> Pulsar 파티션 증가</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: 로드 밸런싱과 캐싱 전략 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
              <Activity className="text-orange-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.3 지능형 로드 밸런싱과 캐싱</h2>
              <p className="text-gray-600 dark:text-gray-400">트래픽 분산과 응답 시간 최적화 전략</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
              <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">적응형 로드 밸런싱</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`# 지능형 로드 밸런서 구현
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import aiohttp

@dataclass
class NodeMetrics:
    """노드 성능 메트릭"""
    response_times: deque  # 최근 응답 시간
    error_count: int = 0
    success_count: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    last_health_check: float = 0.0

class AdaptiveLoadBalancer:
    def __init__(self, nodes: List[str], window_size: int = 100):
        """
        적응형 로드 밸런서
        - 응답 시간 기반 가중치 조정
        - 자동 장애 감지 및 복구
        - 리소스 사용률 고려
        """
        self.nodes = nodes
        self.window_size = window_size
        self.metrics: Dict[str, NodeMetrics] = {
            node: NodeMetrics(response_times=deque(maxlen=window_size))
            for node in nodes
        }
        self.weights = {node: 1.0 for node in nodes}
        self.circuit_breakers = {node: False for node in nodes}
        self._lock = asyncio.Lock()
    
    async def select_node(self) -> Optional[str]:
        """가중치 기반 노드 선택"""
        async with self._lock:
            available_nodes = [
                node for node in self.nodes
                if not self.circuit_breakers[node]
            ]
            
            if not available_nodes:
                return None
            
            # 가중치 정규화
            total_weight = sum(self.weights[node] for node in available_nodes)
            if total_weight == 0:
                return np.random.choice(available_nodes)
            
            # 가중치 기반 확률적 선택
            probs = [
                self.weights[node] / total_weight 
                for node in available_nodes
            ]
            return np.random.choice(available_nodes, p=probs)
    
    async def update_metrics(self, node: str, response_time: float, 
                           success: bool, resource_metrics: Dict = None):
        """노드 메트릭 업데이트 및 가중치 재계산"""
        async with self._lock:
            metrics = self.metrics[node]
            
            # 응답 시간 기록
            if success:
                metrics.response_times.append(response_time)
                metrics.success_count += 1
            else:
                metrics.error_count += 1
            
            # 리소스 메트릭 업데이트
            if resource_metrics:
                metrics.cpu_usage = resource_metrics.get('cpu', 0.0)
                metrics.memory_usage = resource_metrics.get('memory', 0.0)
                metrics.active_connections = resource_metrics.get('connections', 0)
            
            # 가중치 재계산
            self._recalculate_weight(node)
            
            # Circuit Breaker 체크
            self._check_circuit_breaker(node)
    
    def _recalculate_weight(self, node: str):
        """노드 가중치 재계산"""
        metrics = self.metrics[node]
        
        if not metrics.response_times:
            return
        
        # 기본 가중치 계산 요소
        avg_response_time = np.mean(metrics.response_times)
        p95_response_time = np.percentile(metrics.response_times, 95)
        error_rate = metrics.error_count / max(
            metrics.success_count + metrics.error_count, 1
        )
        
        # 가중치 계산 (낮은 응답시간, 낮은 에러율 = 높은 가중치)
        base_weight = 1.0 / (1.0 + avg_response_time / 100.0)  # 100ms 기준
        stability_factor = 1.0 / (1.0 + (p95_response_time - avg_response_time) / 50.0)
        reliability_factor = 1.0 - error_rate
        
        # 리소스 사용률 고려
        resource_factor = 1.0
        if metrics.cpu_usage > 0:
            resource_factor *= (1.0 - metrics.cpu_usage / 100.0)
        if metrics.memory_usage > 0:
            resource_factor *= (1.0 - metrics.memory_usage / 100.0)
        
        # 최종 가중치
        self.weights[node] = (
            base_weight * 
            stability_factor * 
            reliability_factor * 
            resource_factor
        )
    
    def _check_circuit_breaker(self, node: str):
        """Circuit Breaker 패턴 구현"""
        metrics = self.metrics[node]
        total_requests = metrics.success_count + metrics.error_count
        
        if total_requests < 10:  # 최소 요청 수
            return
        
        error_rate = metrics.error_count / total_requests
        
        # 에러율이 50% 이상이면 차단
        if error_rate > 0.5:
            self.circuit_breakers[node] = True
            print(f"Circuit breaker OPEN for {node} (error rate: {error_rate:.2%})")
            # 30초 후 자동 복구 시도
            asyncio.create_task(self._reset_circuit_breaker(node, delay=30))
    
    async def _reset_circuit_breaker(self, node: str, delay: int):
        """Circuit Breaker 재설정"""
        await asyncio.sleep(delay)
        async with self._lock:
            self.circuit_breakers[node] = False
            # 메트릭 초기화
            self.metrics[node].error_count = 0
            self.metrics[node].success_count = 0
            print(f"Circuit breaker CLOSED for {node}")

# 분산 캐싱 시스템
class DistributedRAGCache:
    def __init__(self, redis_cluster: List[str], ttl: int = 3600):
        """
        분산 RAG 캐시
        - 쿼리 결과 캐싱
        - 임베딩 캐싱
        - 자주 사용되는 문서 캐싱
        """
        self.redis_nodes = redis_cluster
        self.ttl = ttl
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
    async def get_cached_result(self, query: str, 
                               cache_embedding: bool = True) -> Optional[Dict]:
        """캐시된 검색 결과 조회"""
        # 쿼리 해시 생성
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        
        # Redis 클러스터에서 조회
        try:
            # 1. 쿼리 결과 캐시 확인
            result_key = f"rag:result:{query_hash}"
            cached_result = await self._redis_get(result_key)
            
            if cached_result:
                self.cache_stats['hits'] += 1
                return json.loads(cached_result)
            
            # 2. 임베딩 캐시 확인 (계산 비용 절감)
            if cache_embedding:
                embedding_key = f"rag:embedding:{query_hash}"
                cached_embedding = await self._redis_get(embedding_key)
                if cached_embedding:
                    return {'embedding': json.loads(cached_embedding)}
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            print(f"Cache error: {e}")
            return None
    
    async def cache_result(self, query: str, result: Dict, 
                          embedding: Optional[np.ndarray] = None):
        """검색 결과 캐싱"""
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        
        # 1. 결과 캐싱
        result_key = f"rag:result:{query_hash}"
        await self._redis_set(
            result_key, 
            json.dumps(result), 
            ttl=self.ttl
        )
        
        # 2. 임베딩 캐싱
        if embedding is not None:
            embedding_key = f"rag:embedding:{query_hash}"
            await self._redis_set(
                embedding_key,
                json.dumps(embedding.tolist()),
                ttl=self.ttl * 2  # 임베딩은 더 오래 보관
            )
        
        # 3. 자주 검색되는 쿼리 추적
        popularity_key = f"rag:popular:{query_hash}"
        await self._redis_incr(popularity_key)
    
    async def preload_popular_queries(self, threshold: int = 100):
        """인기 쿼리 사전 로딩"""
        # 자주 검색되는 쿼리 식별 및 사전 계산
        popular_queries = await self._get_popular_queries(threshold)
        
        for query in popular_queries:
            # 백그라운드에서 미리 계산
            asyncio.create_task(self._precompute_query(query))
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 반환"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total if total > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total,
            **self.cache_stats
        }`}
                    </pre>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">실전 캐싱 전략</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">L1 캐시 (로컬)</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li><strong>위치:</strong> 각 애플리케이션 서버</li>
                    <li><strong>저장:</strong> 자주 사용되는 임베딩</li>
                    <li><strong>크기:</strong> 1-2GB</li>
                    <li><strong>TTL:</strong> 5-10분</li>
                    <li><strong>히트율:</strong> 70-80%</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">L2 캐시 (Redis)</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li><strong>위치:</strong> Redis 클러스터</li>
                    <li><strong>저장:</strong> 쿼리 결과, 문서</li>
                    <li><strong>크기:</strong> 100-500GB</li>
                    <li><strong>TTL:</strong> 1-24시간</li>
                    <li><strong>히트율:</strong> 40-50%</li>
                  </ul>
                </div>
              </div>
              
              <div className="mt-4 bg-emerald-100 dark:bg-emerald-900/40 p-4 rounded-lg">
                <p className="text-sm text-emerald-800 dark:text-emerald-200">
                  <strong>💡 Pro Tip:</strong> Netflix는 Edge 캐시를 활용하여 지역별로 
                  인기 콘텐츠 임베딩을 미리 배포합니다. 이를 통해 글로벌 레이턴시를 
                  50ms 이하로 유지합니다.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: 장애 복구와 고가용성 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
              <Shield className="text-red-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.4 장애 복구와 99.99% 가용성 달성</h2>
              <p className="text-gray-600 dark:text-gray-400">Netflix 수준의 안정성을 위한 전략</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border border-red-200 dark:border-red-700">
              <h3 className="font-bold text-red-800 dark:text-red-200 mb-4">다층 방어 아키텍처</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`# 장애 복구 시스템 구현
import asyncio
from enum import Enum
from typing import List, Dict, Optional, Callable
import logging
from datetime import datetime, timedelta

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"

class ReplicaManager:
    def __init__(self, primary_nodes: List[str], replica_factor: int = 3):
        """
        복제본 관리자
        - 자동 복제본 승격
        - 데이터 일관성 보장
        - 복구 오케스트레이션
        """
        self.primary_nodes = primary_nodes
        self.replica_factor = replica_factor
        self.replicas: Dict[str, List[str]] = self._init_replicas()
        self.health_status: Dict[str, HealthStatus] = {}
        self.last_sync: Dict[str, datetime] = {}
        
    def _init_replicas(self) -> Dict[str, List[str]]:
        """각 프라이머리 노드의 복제본 초기화"""
        replicas = {}
        for i, primary in enumerate(self.primary_nodes):
            # 다른 노드들을 복제본으로 할당
            replica_nodes = []
            for j in range(1, self.replica_factor + 1):
                replica_idx = (i + j) % len(self.primary_nodes)
                replica_nodes.append(self.primary_nodes[replica_idx])
            replicas[primary] = replica_nodes
        return replicas
    
    async def monitor_health(self):
        """노드 상태 모니터링"""
        while True:
            tasks = []
            for node in self.primary_nodes:
                task = self._check_node_health(node)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(5)  # 5초마다 체크
    
    async def _check_node_health(self, node: str) -> HealthStatus:
        """개별 노드 상태 확인"""
        try:
            # 1. 기본 연결 체크
            response = await self._ping_node(node)
            if not response:
                self.health_status[node] = HealthStatus.DEAD
                await self._handle_node_failure(node)
                return HealthStatus.DEAD
            
            # 2. 응답 시간 체크
            if response['latency'] > 1000:  # 1초 이상
                self.health_status[node] = HealthStatus.DEGRADED
            elif response['latency'] > 500:  # 500ms 이상
                self.health_status[node] = HealthStatus.UNHEALTHY
            else:
                self.health_status[node] = HealthStatus.HEALTHY
            
            # 3. 데이터 동기화 체크
            last_sync = self.last_sync.get(node)
            if last_sync and datetime.now() - last_sync > timedelta(minutes=5):
                await self._trigger_sync(node)
            
            return self.health_status[node]
            
        except Exception as e:
            logging.error(f"Health check failed for {node}: {e}")
            self.health_status[node] = HealthStatus.UNHEALTHY
            return HealthStatus.UNHEALTHY
    
    async def _handle_node_failure(self, failed_node: str):
        """노드 장애 처리"""
        logging.critical(f"Node {failed_node} failed! Initiating failover...")
        
        # 1. 복제본 중 하나를 새 프라이머리로 승격
        replicas = self.replicas.get(failed_node, [])
        new_primary = None
        
        for replica in replicas:
            if self.health_status.get(replica) == HealthStatus.HEALTHY:
                new_primary = replica
                break
        
        if not new_primary:
            logging.error(f"No healthy replica found for {failed_node}")
            return
        
        # 2. 승격 프로세스
        await self._promote_replica(failed_node, new_primary)
        
        # 3. 클라이언트 라우팅 업데이트
        await self._update_routing_table(failed_node, new_primary)
        
        # 4. 새로운 복제본 생성
        await self._create_new_replicas(new_primary)
        
        logging.info(f"Failover complete: {failed_node} -> {new_primary}")

# 분산 트랜잭션 관리
class DistributedTransaction:
    def __init__(self, coordinator_url: str):
        """
        2단계 커밋을 사용한 분산 트랜잭션
        """
        self.coordinator = coordinator_url
        self.participants: List[str] = []
        self.transaction_id: str = None
        self.state = "INITIAL"
        
    async def begin(self):
        """트랜잭션 시작"""
        self.transaction_id = f"txn_{datetime.now().timestamp()}"
        self.state = "PREPARING"
        
        # 코디네이터에 트랜잭션 등록
        await self._register_transaction()
    
    async def add_operation(self, node: str, operation: Dict):
        """트랜잭션에 작업 추가"""
        self.participants.append(node)
        
        # 각 참여자에게 준비 요청
        prepare_result = await self._prepare_on_node(node, operation)
        if not prepare_result:
            await self.rollback()
            raise Exception(f"Prepare failed on {node}")
    
    async def commit(self):
        """트랜잭션 커밋"""
        if self.state != "PREPARING":
            raise Exception("Invalid transaction state")
        
        self.state = "COMMITTING"
        
        # 모든 참여자에게 커밋 요청
        commit_tasks = []
        for participant in self.participants:
            task = self._commit_on_node(participant)
            commit_tasks.append(task)
        
        results = await asyncio.gather(*commit_tasks, return_exceptions=True)
        
        # 하나라도 실패하면 롤백
        if any(isinstance(r, Exception) for r in results):
            await self.rollback()
            raise Exception("Commit failed")
        
        self.state = "COMMITTED"
    
    async def rollback(self):
        """트랜잭션 롤백"""
        self.state = "ROLLING_BACK"
        
        rollback_tasks = []
        for participant in self.participants:
            task = self._rollback_on_node(participant)
            rollback_tasks.append(task)
        
        await asyncio.gather(*rollback_tasks, return_exceptions=True)
        self.state = "ROLLED_BACK"

# 실전 예제: Chaos Engineering
class ChaosMonkey:
    """
    프로덕션 환경에서 장애 복구 테스트
    Netflix의 Chaos Monkey 구현
    """
    def __init__(self, cluster_manager, enabled: bool = False):
        self.cluster = cluster_manager
        self.enabled = enabled
        self.failure_probability = 0.001  # 0.1% 확률
        
    async def run(self):
        """주기적으로 무작위 장애 발생"""
        while self.enabled:
            await asyncio.sleep(60)  # 1분마다 실행
            
            if np.random.random() < self.failure_probability:
                # 무작위 노드 선택
                victim = np.random.choice(self.cluster.nodes)
                
                # 장애 유형 선택
                failure_type = np.random.choice([
                    'network_partition',
                    'high_latency',
                    'resource_exhaustion',
                    'process_crash'
                ])
                
                logging.warning(f"Chaos Monkey: Inducing {failure_type} on {victim}")
                await self._induce_failure(victim, failure_type)
    
    async def _induce_failure(self, node: str, failure_type: str):
        """실제 장애 유발"""
        if failure_type == 'network_partition':
            # 네트워크 격리
            await self._isolate_node(node, duration=30)
        elif failure_type == 'high_latency':
            # 높은 지연 유발
            await self._add_latency(node, latency_ms=5000, duration=60)
        elif failure_type == 'resource_exhaustion':
            # CPU/메모리 소진
            await self._exhaust_resources(node, duration=45)
        elif failure_type == 'process_crash':
            # 프로세스 강제 종료
            await self._kill_process(node)`}
                    </pre>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">실제 장애 시나리오와 복구 전략</h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">시나리오 1: 데이터센터 전체 장애</h4>
                  <div className="text-sm text-gray-700 dark:text-gray-300">
                    <p className="mb-2"><strong>상황:</strong> AWS us-east-1 리전 전체 다운</p>
                    <p className="mb-2"><strong>복구 전략:</strong></p>
                    <ul className="list-disc list-inside space-y-1">
                      <li>자동으로 us-west-2로 트래픽 전환</li>
                      <li>Cross-region 복제본 활성화</li>
                      <li>DNS 업데이트 (Route53 헬스체크)</li>
                      <li>캐시 워밍 시작</li>
                    </ul>
                    <p className="mt-2"><strong>RTO:</strong> 5분 이내</p>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">시나리오 2: 캐스케이딩 장애</h4>
                  <div className="text-sm text-gray-700 dark:text-gray-300">
                    <p className="mb-2"><strong>상황:</strong> 하나의 노드 장애가 연쇄적으로 확산</p>
                    <p className="mb-2"><strong>복구 전략:</strong></p>
                    <ul className="list-disc list-inside space-y-1">
                      <li>Circuit Breaker로 장애 노드 격리</li>
                      <li>백프레셔(Backpressure) 적용</li>
                      <li>Rate Limiting으로 과부하 방지</li>
                      <li>점진적 복구 (10% → 50% → 100%)</li>
                    </ul>
                    <p className="mt-2"><strong>RTO:</strong> 30초 이내</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: 성능 모니터링과 최적화 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
              <Activity className="text-green-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.5 실시간 성능 모니터링과 최적화</h2>
              <p className="text-gray-600 dark:text-gray-400">Grafana + Prometheus를 활용한 옵저버빌리티</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">종합 모니터링 대시보드</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`# Prometheus 메트릭 수집 설정
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-cluster'
    static_configs:
      - targets: ['rag-node-1:9090', 'rag-node-2:9090', 'rag-node-3:9090']
    
  - job_name: 'vector-db'
    static_configs:
      - targets: ['milvus-proxy:9091', 'milvus-query:9091']
    
  - job_name: 'cache-layer'
    static_configs:
      - targets: ['redis-1:9092', 'redis-2:9092', 'redis-3:9092']

# 커스텀 메트릭 정의
from prometheus_client import Counter, Histogram, Gauge
import time

# RAG 성능 메트릭
rag_query_total = Counter(
    'rag_query_total', 
    'Total number of RAG queries',
    ['status', 'query_type']
)

rag_query_duration = Histogram(
    'rag_query_duration_seconds',
    'RAG query duration in seconds',
    ['operation'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

rag_cache_hit_rate = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate percentage'
)

vector_db_active_connections = Gauge(
    'vector_db_active_connections',
    'Number of active vector DB connections',
    ['node']
)

# 메트릭 수집 데코레이터
def track_performance(operation: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                rag_query_total.labels(status='success', query_type=operation).inc()
                return result
            except Exception as e:
                rag_query_total.labels(status='error', query_type=operation).inc()
                raise e
            finally:
                duration = time.time() - start_time
                rag_query_duration.labels(operation=operation).observe(duration)
        
        return wrapper
    return decorator

# 실제 사용 예제
class MonitoredRAGEngine:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.cache = DistributedCache()
        
    @track_performance('semantic_search')
    async def search(self, query: str, top_k: int = 10):
        # 캐시 확인
        cached = await self.cache.get(query)
        if cached:
            rag_cache_hit_rate.set(
                self.cache.get_hit_rate() * 100
            )
            return cached
        
        # 벡터 검색
        with vector_db_active_connections.labels(
            node='primary'
        ).track_inprogress():
            results = await self.vector_db.search(query, top_k)
        
        # 결과 캐싱
        await self.cache.set(query, results)
        
        return results

# Grafana 알림 규칙
alerting_rules:
  - name: RAG Performance
    rules:
      - alert: HighQueryLatency
        expr: |
          histogram_quantile(0.95, 
            rate(rag_query_duration_seconds_bucket[5m])
          ) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 1s"
          
      - alert: LowCacheHitRate
        expr: rag_cache_hit_rate < 30
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below 30%"
          
      - alert: VectorDBOverload
        expr: vector_db_active_connections > 1000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Vector DB connection pool exhausted"`}
                    </pre>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">자동 성능 튜닝</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">쿼리 최적화</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 느린 쿼리 자동 감지</li>
                    <li>• 인덱스 추천 시스템</li>
                    <li>• 쿼리 플랜 분석</li>
                    <li>• 자동 쿼리 재작성</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">리소스 최적화</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 자동 스케일링</li>
                    <li>• 메모리 압축</li>
                    <li>• 배치 크기 조정</li>
                    <li>• 연결 풀 튜닝</li>
                  </ul>
                </div>
              </div>

              <div className="mt-4 bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">성능 벤치마크 결과</h4>
                <div className="grid grid-cols-4 gap-4 text-center">
                  <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded">
                    <p className="text-xl font-bold text-purple-600">100K</p>
                    <p className="text-xs text-purple-700 dark:text-purple-300">QPS</p>
                  </div>
                  <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded">
                    <p className="text-xl font-bold text-green-600">45ms</p>
                    <p className="text-xs text-green-700 dark:text-green-300">P50 Latency</p>
                  </div>
                  <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded">
                    <p className="text-xl font-bold text-blue-600">95ms</p>
                    <p className="text-xs text-blue-700 dark:text-blue-300">P99 Latency</p>
                  </div>
                  <div className="bg-orange-100 dark:bg-orange-900/30 p-3 rounded">
                    <p className="text-xl font-bold text-orange-600">99.99%</p>
                    <p className="text-xs text-orange-700 dark:text-orange-300">Uptime</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 6: 실전 구현 가이드 */}
        <section className="bg-gradient-to-r from-indigo-500 to-cyan-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">실전 구현 가이드</h2>
          
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">🚀 프로덕션 체크리스트</h3>
            
            <div className="space-y-4">
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">📋 인프라 준비</h4>
                <ol className="space-y-2 text-sm">
                  <li>✅ Kubernetes 클러스터 (최소 10 노드)</li>
                  <li>✅ 고성능 SSD 스토리지 (NVMe 권장)</li>
                  <li>✅ 100Gbps 네트워크 대역폭</li>
                  <li>✅ GPU 노드 풀 (임베딩 계산용)</li>
                  <li>✅ Multi-AZ 배포</li>
                </ol>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🔧 필수 구성 요소</h4>
                <ul className="space-y-2 text-sm">
                  <li>• <strong>Vector DB:</strong> Milvus 2.3+ 또는 Qdrant 1.7+</li>
                  <li>• <strong>Message Queue:</strong> Pulsar 또는 Kafka</li>
                  <li>• <strong>Cache:</strong> Redis Cluster 7.0+</li>
                  <li>• <strong>Monitoring:</strong> Prometheus + Grafana</li>
                  <li>• <strong>Service Mesh:</strong> Istio 또는 Linkerd</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">💡 최적화 팁</h4>
                <ul className="space-y-1 text-sm">
                  <li>• 임베딩 차원을 384로 줄여 메모리 30% 절약</li>
                  <li>• Product Quantization으로 인덱스 크기 75% 감소</li>
                  <li>• 배치 처리로 처리량 10배 향상</li>
                  <li>• Edge 캐싱으로 글로벌 레이턴시 50% 감소</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🚨 주의사항</h4>
                <p className="text-sm">
                  대규모 분산 시스템은 복잡도가 매우 높습니다. 
                  처음부터 완벽한 시스템을 구축하려 하지 말고, 
                  점진적으로 확장하며 각 단계에서 충분한 테스트를 수행하세요.
                  특히 Chaos Engineering은 충분한 준비 후에 도입하세요.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter2"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: Multi-Agent RAG Systems
          </Link>
          
          <Link
            href="/modules/rag/advanced"
            className="inline-flex items-center gap-2 bg-indigo-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-600 transition-colors"
          >
            고급 과정 완료
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}