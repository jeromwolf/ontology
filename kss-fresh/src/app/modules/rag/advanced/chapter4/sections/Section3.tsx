'use client'

import { Sparkles } from 'lucide-react'

export default function Section3() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
          <Sparkles className="text-green-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.3 다양성 인식 재순위화</h2>
          <p className="text-gray-600 dark:text-gray-400">관련성과 다양성의 최적 균형점 찾기</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">MMR (Maximal Marginal Relevance) 고급 구현</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>검색 결과의 다양성은 사용자 경험에 매우 중요합니다.</strong>
              특히 모호한 쿼리나 다면적 정보 요구에서는 단순히 관련성이 높은 문서만
              보여주는 것보다 다양한 관점을 제공하는 것이 효과적입니다.
            </p>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer

class DiversityAwareReranker:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        다양성 인식 재순위화 시스템
        - MMR (Maximal Marginal Relevance)
        - 주제 클러스터링 기반 다양성
        - 사용자 피드백 기반 학습
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.lambda_param = 0.7  # 관련성 vs 다양성 가중치

        # 주제 클러스터 정보 (사전 학습된 것으로 가정)
        self.topic_clusters = None
        self.aspect_keywords = self._load_aspect_keywords()

    def _load_aspect_keywords(self) -> Dict[str, List[str]]:
        """각 측면에 대한 키워드 로드"""
        return {
            'technical': ['algorithm', 'implementation', 'performance', 'code'],
            'conceptual': ['theory', 'concept', 'principle', 'fundamental'],
            'practical': ['example', 'use case', 'application', 'real-world'],
            'comparison': ['vs', 'compare', 'difference', 'better'],
            'tutorial': ['how to', 'guide', 'step by step', 'learn']
        }

    def mmr_rerank(self, query: str, documents: List[Dict],
                   lambda_param: float = None, top_k: int = 10) -> List[Dict]:
        """
        Maximal Marginal Relevance 기반 재순위화

        MMR = λ * Rel(d) - (1-λ) * max(Sim(d, d_i))
        """
        if lambda_param is None:
            lambda_param = self.lambda_param

        # 쿼리와 문서 임베딩
        query_emb = self.encoder.encode([query])
        doc_contents = [doc['content'] for doc in documents]
        doc_embs = self.encoder.encode(doc_contents)

        # 쿼리-문서 관련성
        relevance_scores = cosine_similarity(query_emb, doc_embs)[0]

        # MMR 알고리즘
        selected = []
        candidates = list(range(len(documents)))

        while len(selected) < top_k and candidates:
            mmr_scores = []

            for idx in candidates:
                # 관련성 점수
                rel_score = relevance_scores[idx]

                # 이미 선택된 문서들과의 최대 유사도
                if selected:
                    selected_embs = doc_embs[selected]
                    similarities = cosine_similarity(
                        [doc_embs[idx]], selected_embs
                    )[0]
                    max_sim = similarities.max()
                else:
                    max_sim = 0

                # MMR 점수 계산
                mmr = lambda_param * rel_score - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)

            # 최고 MMR 점수를 가진 문서 선택
            best_idx = candidates[np.argmax(mmr_scores)]
            selected.append(best_idx)
            candidates.remove(best_idx)

        # 선택된 문서들 반환
        reranked = []
        for rank, idx in enumerate(selected):
            doc = documents[idx].copy()
            doc['mmr_score'] = relevance_scores[idx]
            doc['diversity_rank'] = rank + 1
            reranked.append(doc)

        return reranked

    def aspect_aware_rerank(self, query: str, documents: List[Dict],
                           ensure_aspects: int = 3) -> List[Dict]:
        """
        측면 인식 재순위화
        다양한 관점/측면이 포함되도록 보장
        """
        # 각 문서의 측면 분류
        doc_aspects = []
        for doc in documents:
            aspects = self._classify_aspects(doc['content'])
            doc_aspects.append(aspects)

        # 측면별 최상위 문서 선택
        aspect_best = defaultdict(list)
        for i, (doc, aspects) in enumerate(zip(documents, doc_aspects)):
            for aspect in aspects:
                aspect_best[aspect].append((doc['score'], i))

        # 각 측면별로 정렬
        for aspect in aspect_best:
            aspect_best[aspect].sort(reverse=True)

        # 다양한 측면을 커버하도록 선택
        selected = []
        selected_indices = set()
        aspect_counts = defaultdict(int)

        # Round-robin 방식으로 각 측면에서 선택
        aspect_cycle = list(aspect_best.keys())
        aspect_idx = 0

        while len(selected) < min(len(documents), ensure_aspects * 3):
            aspect = aspect_cycle[aspect_idx % len(aspect_cycle)]

            # 해당 측면에서 아직 선택되지 않은 최상위 문서 찾기
            for score, doc_idx in aspect_best[aspect]:
                if doc_idx not in selected_indices:
                    selected.append(documents[doc_idx])
                    selected_indices.add(doc_idx)
                    aspect_counts[aspect] += 1
                    break

            aspect_idx += 1

            # 모든 측면이 최소 1개씩 포함되었는지 확인
            if len(selected) >= ensure_aspects and \
               all(count > 0 for count in aspect_counts.values()):
                break

        # 나머지 문서들은 점수순으로 추가
        for i, doc in enumerate(documents):
            if i not in selected_indices and len(selected) < 10:
                selected.append(doc)

        return selected

    def _classify_aspects(self, text: str) -> List[str]:
        """문서의 측면 분류"""
        text_lower = text.lower()
        aspects = []

        for aspect, keywords in self.aspect_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                aspects.append(aspect)

        # 측면이 없으면 기본값
        if not aspects:
            aspects = ['general']

        return aspects

    def learning_to_diversify(self, query: str, documents: List[Dict],
                            user_feedback: List[int] = None) -> List[Dict]:
        """
        사용자 피드백 기반 다양성 학습
        클릭률, 체류시간 등을 활용하여 최적 λ 값 학습
        """
        if user_feedback is None:
            # 피드백이 없으면 기본 MMR 사용
            return self.mmr_rerank(query, documents)

        # 피드백 기반 λ 조정
        # 상위 문서들의 유사도가 높은데 클릭률이 낮으면 λ 감소 (다양성 증가)
        top_5_similarity = self._calculate_diversity_score(documents[:5])
        click_rate = sum(user_feedback[:5]) / min(5, len(user_feedback))

        if top_5_similarity > 0.8 and click_rate < 0.4:
            # 너무 유사한 결과, 다양성 필요
            adjusted_lambda = max(0.3, self.lambda_param - 0.2)
        elif top_5_similarity < 0.5 and click_rate > 0.7:
            # 충분히 다양함, 관련성 증가 필요
            adjusted_lambda = min(0.9, self.lambda_param + 0.1)
        else:
            adjusted_lambda = self.lambda_param

        return self.mmr_rerank(query, documents, lambda_param=adjusted_lambda)

    def _calculate_diversity_score(self, documents: List[Dict]) -> float:
        """문서 집합의 다양성 점수 계산 (0-1, 낮을수록 다양)"""
        if len(documents) < 2:
            return 0.0

        contents = [doc['content'] for doc in documents]
        embeddings = self.encoder.encode(contents)

        # 모든 쌍의 유사도 평균
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

# 고급 재순위화 파이프라인
class AdvancedRerankingPipeline:
    def __init__(self):
        """
        다단계 재순위화 파이프라인
        1. Cross-Encoder로 관련성 재평가
        2. MMR로 다양성 확보
        3. 사용자 선호도 반영
        """
        self.cross_encoder = AdvancedCrossEncoderReranker()
        self.diversity_reranker = DiversityAwareReranker()
        self.user_preferences = {}

    def rerank(self, query: str, documents: List[Dict],
              user_id: str = None, ensure_diversity: bool = True) -> List[Dict]:
        """
        통합 재순위화 수행
        """
        # 1단계: Cross-Encoder 재순위화
        ce_results = self.cross_encoder.rerank(query, documents, top_k=30)

        # RerankingResult를 Dict로 변환
        reranked_docs = []
        for result in ce_results:
            reranked_docs.append({
                'id': result.doc_id,
                'content': result.content,
                'score': result.reranked_score,
                'original_rank': result.original_rank
            })

        # 2단계: 다양성 재순위화
        if ensure_diversity:
            # 사용자별 선호도 반영
            lambda_param = 0.7
            if user_id and user_id in self.user_preferences:
                lambda_param = self.user_preferences[user_id].get('lambda', 0.7)

            final_results = self.diversity_reranker.mmr_rerank(
                query, reranked_docs[:20],
                lambda_param=lambda_param,
                top_k=10
            )
        else:
            final_results = reranked_docs[:10]

        return final_results

    def update_user_preference(self, user_id: str, feedback: Dict):
        """사용자 선호도 업데이트"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {'lambda': 0.7}

        # 피드백 기반으로 lambda 조정
        if feedback.get('wanted_more_diversity'):
            self.user_preferences[user_id]['lambda'] = max(
                0.3, self.user_preferences[user_id]['lambda'] - 0.1
            )
        elif feedback.get('wanted_more_relevance'):
            self.user_preferences[user_id]['lambda'] = min(
                0.9, self.user_preferences[user_id]['lambda'] + 0.1
            )

# 사용 예제
pipeline = AdvancedRerankingPipeline()

# 초기 검색 결과
search_results = [
    {'id': '1', 'content': 'Python 프로그래밍 기초 튜토리얼...', 'score': 0.92},
    {'id': '2', 'content': 'Python 입문자를 위한 가이드...', 'score': 0.91},
    {'id': '3', 'content': 'Python vs Java 성능 비교...', 'score': 0.88},
    {'id': '4', 'content': 'Python 고급 기법과 최적화...', 'score': 0.87},
    {'id': '5', 'content': 'Python으로 웹 개발하기...', 'score': 0.86},
]

# 재순위화 수행
query = "Python 프로그래밍 배우기"
final_results = pipeline.rerank(query, search_results, ensure_diversity=True)

print("=== 다양성을 고려한 재순위화 결과 ===")
for i, doc in enumerate(final_results, 1):
    print(f"{i}. {doc['id']}: {doc['content'][:50]}...")`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
