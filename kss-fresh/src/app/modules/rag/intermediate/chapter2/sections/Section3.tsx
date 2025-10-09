'use client'

import { Filter } from 'lucide-react'

export default function Section3() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
          <Filter className="text-orange-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.3 검색 결과 결합 및 재순위화</h2>
          <p className="text-gray-600 dark:text-gray-400">효과적인 하이브리드 검색 구현 방법</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
          <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">점수 정규화 기법</h3>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">1. Min-Max 정규화</h4>
              <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                <pre className="text-xs text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`def min_max_normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">2. Z-Score 정규화</h4>
              <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                <pre className="text-xs text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`def z_score_normalize(scores):
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    if std == 0:
        return [0.0] * len(scores)
    return [(s - mean) / std for s in scores]`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">3. Reciprocal Rank Fusion (RRF)</h4>
              <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                <pre className="text-xs text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`def reciprocal_rank_fusion(rankings, k=60):
    """여러 순위를 결합하는 RRF 알고리즘"""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)`}
                </pre>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">완전한 하이브리드 검색 파이프라인</h3>
          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`from typing import List, Tuple
import numpy as np

class HybridSearchEngine:
    def __init__(self, vector_db, bm25_engine, alpha=0.5):
        """
        alpha: 벡터 검색 가중치 (0~1)
        1-alpha: 키워드 검색 가중치
        """
        self.vector_db = vector_db
        self.bm25 = bm25_engine
        self.alpha = alpha

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        # 1. 벡터 검색 수행
        query_embedding = self.embed_query(query)
        vector_results = self.vector_db.similarity_search(
            query_embedding, k=top_k * 2
        )

        # 2. BM25 키워드 검색 수행
        query_tokens = self.tokenize(query)
        bm25_scores = []
        for i, doc in enumerate(self.bm25.corpus):
            score = self.bm25.score(query_tokens, i)
            bm25_scores.append((i, score))
        bm25_results = sorted(bm25_scores, key=lambda x: x[1], reverse=True)[:top_k * 2]

        # 3. 점수 정규화
        vector_scores = [r[1] for r in vector_results]
        vector_norm = self.min_max_normalize(vector_scores)

        bm25_scores = [r[1] for r in bm25_results]
        bm25_norm = self.min_max_normalize(bm25_scores)

        # 4. 결과 병합 및 재순위화
        combined_scores = {}

        for (doc_id, _), norm_score in zip(vector_results, vector_norm):
            combined_scores[doc_id] = self.alpha * norm_score

        for (doc_id, _), norm_score in zip(bm25_results, bm25_norm):
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - self.alpha) * norm_score
            else:
                combined_scores[doc_id] = (1 - self.alpha) * norm_score

        # 5. 최종 순위 결정
        final_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return final_results

    def adaptive_alpha(self, query: str) -> float:
        """쿼리 특성에 따라 alpha 값 동적 조정"""
        # 숫자, 코드, ID가 포함된 경우 키워드 검색 가중치 증가
        if any(c.isdigit() for c in query) or '-' in query:
            return 0.3

        # 일반적인 질문인 경우 벡터 검색 가중치 증가
        if any(word in query for word in ['무엇', '어떻게', '왜', '설명']):
            return 0.7

        return 0.5  # 기본값

# 사용 예제
hybrid_search = HybridSearchEngine(vector_db, bm25_engine, alpha=0.5)
results = hybrid_search.search("SKU-12345 제품의 재고 현황", top_k=5)
for doc_id, score in results:
    print(f"문서 {doc_id}: {score:.3f}")`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
