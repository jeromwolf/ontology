'use client'

import { Database } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
          <Database className="text-blue-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.1 캐싱 전략</h2>
          <p className="text-gray-600 dark:text-gray-400">Redis와 인메모리 캐싱으로 응답 속도 최적화</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">다층 캐싱 아키텍처</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>캐싱은 RAG 시스템의 성능을 극적으로 향상시키는 핵심 기술입니다.</strong> 동일한 쿼리에 대해 매번 벡터 검색을 수행하는 것은 비효율적이므로,
              자주 사용되는 쿼리와 결과를 캐시에 저장하여 응답 속도를 10배 이상 향상시킬 수 있습니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              다층 캐싱 시스템은 다음과 같이 작동합니다:
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-2">
              <li><strong>L1 캐시 (메모리)</strong>: 가장 빠른 접근 속도 (~0.1ms), 제한된 용량</li>
              <li><strong>L2 캐시 (Redis)</strong>: 중간 속도 (~1-5ms), 대용량 지원</li>
              <li><strong>L3 캐시 (디스크)</strong>: 느린 속도 (~10-50ms), 무제한 용량</li>
            </ul>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import redis
import hashlib
from typing import List, Optional
from functools import wraps
import json

class RAGCacheManager:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.memory_cache = {}  # L1 캐시 (인메모리)
        self.max_memory_size = 1000  # 최대 메모리 캐시 항목

    def generate_cache_key(self, query: str, params: dict = None) -> str:
        """쿼리와 매개변수로 캐시 키 생성"""
        content = f"{query}_{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_cached_result(self, cache_key: str) -> Optional[dict]:
        """다층 캐싱에서 결과 조회"""
        # L1: 메모리 캐시 확인 (가장 빠름)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # L2: Redis 캐시 확인 (중간 속도)
        redis_result = self.redis_client.get(f"rag:{cache_key}")
        if redis_result:
            result = json.loads(redis_result)
            # 메모리 캐시에도 저장 (워밍업)
            self.set_memory_cache(cache_key, result)
            return result

        return None

    def set_cache_result(self, cache_key: str, result: dict, ttl: int = 3600):
        """결과를 다층 캐시에 저장"""
        # L1: 메모리 캐시 저장
        self.set_memory_cache(cache_key, result)

        # L2: Redis 캐시 저장 (TTL 적용)
        self.redis_client.setex(
            f"rag:{cache_key}",
            ttl,
            json.dumps(result)
        )

    def set_memory_cache(self, cache_key: str, result: dict):
        """메모리 캐시 저장 (LRU 정책)"""
        if len(self.memory_cache) >= self.max_memory_size:
            # 가장 오래된 항목 제거 (LRU)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

        self.memory_cache[cache_key] = result

# 캐싱 데코레이터
def cache_rag_result(ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        def wrapper(self, query: str, **kwargs):
            cache_key = self.cache_manager.generate_cache_key(query, kwargs)

            # 캐시된 결과 확인
            cached = self.cache_manager.get_cached_result(cache_key)
            if cached:
                return cached

            # 캐시 미스: 실제 검색 수행
            result = func(self, query, **kwargs)

            # 결과를 캐시에 저장
            self.cache_manager.set_cache_result(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

class OptimizedRAGRetriever:
    def __init__(self, vector_db, cache_manager):
        self.vector_db = vector_db
        self.cache_manager = cache_manager

    @cache_rag_result(ttl=3600)  # 1시간 캐시
    def retrieve_documents(self, query: str, k: int = 5) -> dict:
        """캐싱이 적용된 문서 검색"""
        # 벡터 검색 수행
        results = self.vector_db.similarity_search(query, k=k)

        return {
            "query": query,
            "documents": results,
            "timestamp": time.time(),
            "source": "vector_db"
        }

# 사용 예시
cache_manager = RAGCacheManager()
retriever = OptimizedRAGRetriever(vector_db, cache_manager)

# 첫 번째 호출 - 실제 검색
result1 = retriever.retrieve_documents("파이썬의 장점은?")

# 두 번째 호출 - 캐시에서 반환 (빠름)
result2 = retriever.retrieve_documents("파이썬의 장점은?")`}
            </pre>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">스마트 캐싱 전략</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              단순히 결과를 저장하는 것을 넘어서, <strong>지능적인 캐싱 전략</strong>을 사용하면 캐시 효율성을 크게 향상시킬 수 있습니다.
              핵심은 사용자의 다양한 표현을 표준화하고, 자주 사용되는 쿼리를 미리 준비하는 것입니다.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">🎯 쿼리 정규화</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                동일한 의미의 다른 표현들을 표준화
              </p>
              <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                <pre>
{`# 정규화 예시
"파이썬 장점" → "python_advantages"
"Python의 좋은 점" → "python_advantages"
"파이썬 이점은?" → "python_advantages"`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">⚡ 프리로딩</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                인기 쿼리들을 미리 캐시에 준비
              </p>
              <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                <pre>
{`# 인기 쿼리 프리로딩
popular_queries = [
    "인공지능 개요",
    "머신러닝 기초",
    "딥러닝 소개"
]`}
                </pre>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mt-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">🔥 실제 캐싱 효과 비교</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-300 dark:border-gray-600">
                    <th className="text-left py-2 px-3 text-gray-700 dark:text-gray-300">측정 항목</th>
                    <th className="text-right py-2 px-3 text-gray-700 dark:text-gray-300">캐싱 미적용</th>
                    <th className="text-right py-2 px-3 text-gray-700 dark:text-gray-300">L1 캐시만</th>
                    <th className="text-right py-2 px-3 text-gray-700 dark:text-gray-300">다층 캐싱</th>
                  </tr>
                </thead>
                <tbody className="text-gray-600 dark:text-gray-400">
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="py-2 px-3">평균 응답 시간</td>
                    <td className="text-right py-2 px-3">850ms</td>
                    <td className="text-right py-2 px-3">120ms</td>
                    <td className="text-right py-2 px-3 text-green-600 font-bold">45ms</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="py-2 px-3">캐시 히트율</td>
                    <td className="text-right py-2 px-3">0%</td>
                    <td className="text-right py-2 px-3">65%</td>
                    <td className="text-right py-2 px-3 text-green-600 font-bold">92%</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="py-2 px-3">시간당 처리량</td>
                    <td className="text-right py-2 px-3">4,200 req</td>
                    <td className="text-right py-2 px-3">30,000 req</td>
                    <td className="text-right py-2 px-3 text-green-600 font-bold">80,000 req</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3">메모리 사용량</td>
                    <td className="text-right py-2 px-3">512MB</td>
                    <td className="text-right py-2 px-3">768MB</td>
                    <td className="text-right py-2 px-3">1.2GB</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
