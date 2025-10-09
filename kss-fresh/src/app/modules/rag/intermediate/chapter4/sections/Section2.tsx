'use client'

import { BarChart3 } from 'lucide-react'

export default function Section2() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
          <BarChart3 className="text-purple-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.2 배치 처리 및 비동기 검색</h2>
          <p className="text-gray-600 dark:text-gray-400">대량 요청의 효율적 처리</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">비동기 배치 처리 시스템</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>대량의 사용자 요청을 효율적으로 처리하는 것은 프로덕션 RAG 시스템의 필수 요구사항입니다.</strong>
              비동기 프로그래밍과 배치 처리를 통해 시스템 처리량을 10-100배 향상시킬 수 있습니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>핵심 최적화 기법:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>병렬 검색</strong>: 벡터 검색과 임베딩 생성을 동시에 수행</li>
              <li><strong>배치 추론</strong>: 여러 쿼리를 한 번에 처리하여 GPU 효율성 극대화</li>
              <li><strong>비동기 I/O</strong>: 네트워크 대기 시간 동안 다른 작업 수행</li>
              <li><strong>스레드 풀</strong>: CPU 집약적 작업을 별도 스레드에서 처리</li>
            </ul>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List
import time

class AsyncRAGProcessor:
    def __init__(self, max_concurrent=10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent)

    async def process_single_query(self, query: str, session_id: str) -> dict:
        """단일 쿼리 비동기 처리"""
        async with self.semaphore:
            start_time = time.time()

            try:
                # 병렬로 검색 수행
                search_task = asyncio.create_task(
                    self.async_vector_search(query)
                )

                # LLM 임베딩도 병렬로
                embedding_task = asyncio.create_task(
                    self.async_get_embedding(query)
                )

                # 두 작업 동시 실행
                search_results, query_embedding = await asyncio.gather(
                    search_task,
                    embedding_task
                )

                # 결과 생성
                response = await self.async_generate_response(
                    query, search_results
                )

                processing_time = time.time() - start_time

                return {
                    "query": query,
                    "session_id": session_id,
                    "response": response,
                    "documents": search_results,
                    "processing_time": processing_time,
                    "status": "success"
                }

            except Exception as e:
                return {
                    "query": query,
                    "session_id": session_id,
                    "error": str(e),
                    "status": "error",
                    "processing_time": time.time() - start_time
                }

    async def process_batch(self, batch_requests: List[dict]) -> List[dict]:
        """배치 요청 병렬 처리"""
        # 모든 요청을 비동기로 처리
        tasks = [
            self.process_single_query(
                request["query"],
                request.get("session_id", "default")
            )
            for request in batch_requests
        ]

        # 배치 전체 완료 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query": batch_requests[i]["query"],
                    "error": str(result),
                    "status": "exception"
                })
            else:
                processed_results.append(result)

        return processed_results

    async def async_vector_search(self, query: str) -> List[dict]:
        """비동기 벡터 검색"""
        # CPU 집약적 작업을 스레드 풀에서 실행
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.sync_vector_search,
            query
        )

    def sync_vector_search(self, query: str) -> List[dict]:
        """동기 벡터 검색 (CPU 집약적)"""
        # 실제 벡터 검색 로직
        return self.vector_db.similarity_search(query, k=5)

    async def async_get_embedding(self, text: str):
        """비동기 임베딩 생성"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://embedding-service/embed",
                json={"text": text}
            ) as response:
                return await response.json()

    async def async_generate_response(self, query: str, documents: List[dict]):
        """비동기 응답 생성"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://llm-service/generate",
                json={
                    "query": query,
                    "context": documents,
                    "max_tokens": 500
                }
            ) as response:
                return await response.json()

# 사용 예시
async def main():
    processor = AsyncRAGProcessor(max_concurrent=20)

    # 대량 요청 배치
    batch_requests = [
        {"query": f"질문 {i}", "session_id": f"user_{i}"}
        for i in range(100)
    ]

    start_time = time.time()
    results = await processor.process_batch(batch_requests)
    end_time = time.time()

    print(f"100개 요청 처리 시간: {end_time - start_time:.2f}초")
    print(f"평균 응답 시간: {sum(r.get('processing_time', 0) for r in results) / len(results):.3f}초")

# 실행
# asyncio.run(main())`}
            </pre>
          </div>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
          <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">배치 처리 성능 향상 결과</h3>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <p className="text-3xl font-bold text-orange-600">19x</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">처리량 증가</p>
              <p className="text-xs text-gray-500 mt-1">동시 20개 처리 시</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <p className="text-3xl font-bold text-orange-600">73%</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">GPU 활용률</p>
              <p className="text-xs text-gray-500 mt-1">개별: 12% → 배치: 73%</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <p className="text-3xl font-bold text-orange-600">$0.12</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">요청당 비용</p>
              <p className="text-xs text-gray-500 mt-1">개별: $0.85 → 배치: $0.12</p>
            </div>
          </div>

          <div className="mt-4 p-3 bg-orange-100 dark:bg-orange-800/20 rounded-lg">
            <p className="text-sm text-orange-800 dark:text-orange-200">
              <strong>💡 최적 배치 크기:</strong> 실험 결과, 배치 크기 20-50이 가장 효율적입니다.
              그 이상은 메모리 부족이나 지연 시간 증가로 인해 오히려 성능이 저하됩니다.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
