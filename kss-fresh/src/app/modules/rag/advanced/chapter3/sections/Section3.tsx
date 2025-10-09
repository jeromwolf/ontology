import { Activity } from 'lucide-react'

export default function Section3() {
  return (
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
  )
}
