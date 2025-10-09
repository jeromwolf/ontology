import { Database } from 'lucide-react'

export default function Section2() {
  return (
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
  )
}
