import { Activity } from 'lucide-react'

export default function Section5() {
  return (
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
  )
}
