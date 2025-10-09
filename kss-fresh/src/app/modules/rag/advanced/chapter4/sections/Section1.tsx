'use client'

import { Brain } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
          <Brain className="text-orange-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.1 Cross-Encoder: Bi-Encoder의 한계를 넘어서</h2>
          <p className="text-gray-600 dark:text-gray-400">쿼리와 문서를 함께 인코딩하는 혁신적 접근</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
          <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">Bi-Encoder vs Cross-Encoder 심층 비교</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>기존 Bi-Encoder의 근본적 한계:</strong> 쿼리와 문서를 독립적으로 인코딩하기 때문에
              상호작용(interaction) 정보를 포착할 수 없습니다. Cross-Encoder는 이를 해결하여
              훨씬 정확한 관련성 평가를 가능하게 합니다.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">Bi-Encoder (기존)</h4>
              <div className="space-y-2 text-sm">
                <div className="bg-red-50 dark:bg-red-900/30 p-2 rounded">
                  <strong>독립 인코딩:</strong> query → embedding, doc → embedding
                </div>
                <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                  <strong>속도:</strong> 매우 빠름 (사전 계산 가능)
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/30 p-2 rounded">
                  <strong>확장성:</strong> 수백만 문서 처리 가능
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/30 p-2 rounded">
                  <strong>정확도:</strong> 상대적으로 낮음
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">Cross-Encoder (고급)</h4>
              <div className="space-y-2 text-sm">
                <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                  <strong>공동 인코딩:</strong> [query, doc] → score
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/30 p-2 rounded">
                  <strong>속도:</strong> 느림 (실시간 계산 필요)
                </div>
                <div className="bg-red-50 dark:bg-red-900/30 p-2 rounded">
                  <strong>확장성:</strong> Top-K 재순위화에만 사용
                </div>
                <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                  <strong>정확도:</strong> 매우 높음 (SOTA)
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">Cross-Encoder 구현: ms-marco-MiniLM 활용</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple, Dict
import torch
from dataclasses import dataclass
import time

@dataclass
class RerankingResult:
    """재순위화 결과"""
    doc_id: str
    original_score: float
    reranked_score: float
    original_rank: int
    new_rank: int
    content: str

class AdvancedCrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        """
        고급 Cross-Encoder 재순위화 시스템
        - MS MARCO로 학습된 최신 모델 사용
        - GPU 가속 지원
        - 배치 처리 최적화
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(model_name, device=self.device)

        # 성능 메트릭
        self.stats = {
            'total_reranked': 0,
            'avg_latency': 0,
            'score_improvements': []
        }

    def rerank(self, query: str, documents: List[Dict],
               top_k: int = 10, batch_size: int = 32) -> List[RerankingResult]:
        """
        문서 재순위화 수행

        Args:
            query: 검색 쿼리
            documents: [{'id': str, 'content': str, 'score': float}]
            top_k: 재순위화할 상위 문서 수
            batch_size: 배치 크기
        """
        start_time = time.time()

        # 1단계: 초기 순위 기록
        documents = sorted(documents, key=lambda x: x['score'], reverse=True)
        docs_to_rerank = documents[:top_k]

        # 2단계: Cross-Encoder 점수 계산
        pairs = [[query, doc['content']] for doc in docs_to_rerank]

        # 배치 처리로 효율성 향상
        cross_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            batch_scores = self.model.predict(batch)
            cross_scores.extend(batch_scores)

        # 3단계: 점수 정규화 및 결합
        results = []
        for i, (doc, cross_score) in enumerate(zip(docs_to_rerank, cross_scores)):
            # 원본 점수와 Cross-Encoder 점수 결합
            # α를 조정하여 가중치 조절 가능
            alpha = 0.7  # Cross-Encoder 가중치
            combined_score = alpha * cross_score + (1 - alpha) * doc['score']

            results.append(RerankingResult(
                doc_id=doc['id'],
                original_score=doc['score'],
                reranked_score=combined_score,
                original_rank=i + 1,
                new_rank=-1,  # 나중에 업데이트
                content=doc['content'][:200] + '...'
            ))

        # 4단계: 재순위화
        results.sort(key=lambda x: x.reranked_score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i + 1

        # 5단계: 나머지 문서 추가 (재순위화하지 않은 문서들)
        remaining_docs = documents[top_k:]
        for i, doc in enumerate(remaining_docs):
            results.append(RerankingResult(
                doc_id=doc['id'],
                original_score=doc['score'],
                reranked_score=doc['score'],
                original_rank=top_k + i + 1,
                new_rank=top_k + i + 1,
                content=doc['content'][:200] + '...'
            ))

        # 성능 통계 업데이트
        latency = time.time() - start_time
        self._update_stats(results, latency)

        return results

    def _update_stats(self, results: List[RerankingResult], latency: float):
        """성능 통계 업데이트"""
        self.stats['total_reranked'] += len(results)
        self.stats['avg_latency'] = (
            (self.stats['avg_latency'] * (self.stats['total_reranked'] - len(results)) +
             latency * len(results)) / self.stats['total_reranked']
        )

        # 점수 개선 추적
        for result in results:
            if result.original_rank <= 10:  # Top-10 내에서의 변화만 추적
                improvement = result.original_rank - result.new_rank
                self.stats['score_improvements'].append(improvement)

    def get_performance_report(self) -> Dict:
        """성능 리포트 생성"""
        if not self.stats['score_improvements']:
            return self.stats

        improvements = self.stats['score_improvements']
        return {
            **self.stats,
            'avg_rank_improvement': np.mean(improvements),
            'median_rank_improvement': np.median(improvements),
            'improvement_rate': len([x for x in improvements if x > 0]) / len(improvements)
        }

# 사용 예제
reranker = AdvancedCrossEncoderReranker()

# 검색 결과 (Bi-Encoder에서 나온 초기 결과)
search_results = [
    {'id': 'doc1', 'content': 'Python은 인터프리터 언어입니다...', 'score': 0.89},
    {'id': 'doc2', 'content': 'Python 프로그래밍의 기초...', 'score': 0.87},
    {'id': 'doc3', 'content': 'Python으로 웹 개발하기...', 'score': 0.85},
    # ... 더 많은 문서
]

# 재순위화 수행
query = "Python 인터프리터의 작동 원리"
reranked = reranker.rerank(query, search_results, top_k=20)

# 결과 분석
print("=== 재순위화 결과 ===")
for result in reranked[:10]:
    if result.original_rank != result.new_rank:
        change = result.original_rank - result.new_rank
        symbol = "↑" if change > 0 else "↓"
        print(f"{result.new_rank}. {result.doc_id} "
              f"(원래: {result.original_rank}위 {symbol}{abs(change)}) "
              f"점수: {result.reranked_score:.3f}")

# 성능 리포트
report = reranker.get_performance_report()
print(f"\\n평균 순위 개선: {report['avg_rank_improvement']:.2f}")
print(f"개선율: {report['improvement_rate']*100:.1f}%")`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
