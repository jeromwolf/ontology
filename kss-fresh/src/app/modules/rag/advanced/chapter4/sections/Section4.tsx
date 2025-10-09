'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Target } from 'lucide-react'
import References from '@/components/common/References'

export default function Section4() {
  return (
    <>
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
            <Target className="text-blue-600" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.4 Learning to Rank (LTR) for RAG</h2>
            <p className="text-gray-600 dark:text-gray-400">기계학습을 활용한 최적 순위 학습</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
            <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">LambdaMART를 활용한 RAG 최적화</h3>

            <div className="prose prose-sm dark:prose-invert mb-4">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>Learning to Rank는 검색 품질을 극대화하는 핵심 기술입니다.</strong>
                특히 RAG 시스템에서는 단순한 벡터 유사도를 넘어서 다양한 특징을 종합적으로
                고려하여 최적의 순위를 학습할 수 있습니다.
              </p>
            </div>

            <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
              <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class RankingFeatures:
    """랭킹을 위한 특징들"""
    # 텍스트 유사도 특징
    bm25_score: float
    vector_similarity: float
    cross_encoder_score: float

    # 문서 특징
    doc_length: int
    doc_freshness: float  # 최신성 (일 단위)
    doc_popularity: float  # 클릭률, 조회수 등

    # 쿼리-문서 매칭 특징
    exact_match_count: int
    synonym_match_count: int
    named_entity_overlap: float

    # 의미적 특징
    topic_similarity: float
    sentiment_alignment: float

    # 사용자 특징
    user_click_history: float  # 이 문서 타입에 대한 과거 클릭률
    session_dwell_time: float  # 세션 내 평균 체류시간

class LearningToRankRAG:
    def __init__(self, model_path: Optional[str] = None):
        """
        Learning to Rank 기반 RAG 재순위화
        - LambdaMART (XGBoost) 사용
        - 온라인 학습 지원
        - Feature importance 분석
        """
        self.scaler = StandardScaler()
        self.feature_names = [
            'bm25_score', 'vector_similarity', 'cross_encoder_score',
            'doc_length_norm', 'doc_freshness', 'doc_popularity',
            'exact_match_count', 'synonym_match_count', 'named_entity_overlap',
            'topic_similarity', 'sentiment_alignment',
            'user_click_history', 'session_dwell_time'
        ]

        if model_path:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        else:
            self.model = None

        # 온라인 학습을 위한 버퍼
        self.training_buffer = []
        self.buffer_size = 1000

    def extract_features(self, query: str, document: Dict,
                        user_context: Optional[Dict] = None) -> RankingFeatures:
        """
        쿼리-문서 쌍에서 랭킹 특징 추출
        """
        # 기본 점수들 (이미 계산되어 있다고 가정)
        features = RankingFeatures(
            bm25_score=document.get('bm25_score', 0.0),
            vector_similarity=document.get('vector_score', 0.0),
            cross_encoder_score=document.get('ce_score', 0.0),
            doc_length=len(document['content'].split()),
            doc_freshness=self._calculate_freshness(document),
            doc_popularity=document.get('popularity', 0.0),
            exact_match_count=self._count_exact_matches(query, document['content']),
            synonym_match_count=self._count_synonym_matches(query, document['content']),
            named_entity_overlap=self._calculate_entity_overlap(query, document['content']),
            topic_similarity=document.get('topic_similarity', 0.0),
            sentiment_alignment=self._calculate_sentiment_alignment(query, document),
            user_click_history=0.0,
            session_dwell_time=0.0
        )

        # 사용자 컨텍스트가 있으면 개인화 특징 추가
        if user_context:
            features.user_click_history = user_context.get('doc_type_ctr', {}).get(
                document.get('type', 'general'), 0.0
            )
            features.session_dwell_time = user_context.get('avg_dwell_time', 0.0)

        return features

    def _calculate_freshness(self, document: Dict) -> float:
        """문서 최신성 계산 (0-1)"""
        from datetime import datetime, timedelta

        if 'created_at' not in document:
            return 0.5

        created = datetime.fromisoformat(document['created_at'])
        age_days = (datetime.now() - created).days

        # 지수적 감쇠 (30일 반감기)
        return np.exp(-age_days / 30.0)

    def _count_exact_matches(self, query: str, content: str) -> int:
        """정확한 단어 매칭 수"""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        return len(query_terms.intersection(content_terms))

    def _count_synonym_matches(self, query: str, content: str) -> int:
        """동의어 매칭 수 (간단한 예제)"""
        # 실제로는 WordNet이나 사전 학습된 동의어 사전 사용
        synonym_dict = {
            'python': ['파이썬', 'python3', 'py'],
            'machine learning': ['ml', '머신러닝', '기계학습'],
            'deep learning': ['dl', '딥러닝', '심층학습']
        }

        count = 0
        query_lower = query.lower()
        content_lower = content.lower()

        for term, synonyms in synonym_dict.items():
            if term in query_lower:
                for syn in synonyms:
                    if syn in content_lower:
                        count += 1

        return count

    def _calculate_entity_overlap(self, query: str, content: str) -> float:
        """명명된 개체 중첩도 (간단한 버전)"""
        # 실제로는 NER 모델 사용
        # 여기서는 대문자로 시작하는 단어를 개체로 가정
        query_entities = set(w for w in query.split() if w[0].isupper())
        content_entities = set(w for w in content.split() if w[0].isupper())

        if not query_entities:
            return 0.0

        overlap = len(query_entities.intersection(content_entities))
        return overlap / len(query_entities)

    def _calculate_sentiment_alignment(self, query: str, document: Dict) -> float:
        """감성 정렬도 (질문과 답변의 톤 일치)"""
        # 간단한 규칙 기반 (실제로는 감성 분석 모델 사용)
        positive_words = {'good', 'best', 'excellent', 'great', '좋은', '최고'}
        negative_words = {'bad', 'worst', 'poor', 'terrible', '나쁜', '최악'}

        query_sentiment = sum(1 for w in query.split() if w in positive_words) - \
                         sum(1 for w in query.split() if w in negative_words)

        doc_sentiment = sum(1 for w in document['content'].split() if w in positive_words) - \
                       sum(1 for w in document['content'].split() if w in negative_words)

        # 같은 부호면 1, 다른 부호면 0
        if query_sentiment * doc_sentiment >= 0:
            return 1.0
        return 0.0

    def features_to_vector(self, features: RankingFeatures) -> np.ndarray:
        """특징을 벡터로 변환"""
        return np.array([
            features.bm25_score,
            features.vector_similarity,
            features.cross_encoder_score,
            features.doc_length / 1000.0,  # 정규화
            features.doc_freshness,
            features.doc_popularity,
            features.exact_match_count / 10.0,  # 정규화
            features.synonym_match_count / 5.0,  # 정규화
            features.named_entity_overlap,
            features.topic_similarity,
            features.sentiment_alignment,
            features.user_click_history,
            features.session_dwell_time / 60.0  # 분 단위로 정규화
        ])

    def train_model(self, training_data: List[Tuple[str, List[Dict], List[int]]],
                   validation_data: Optional[List] = None):
        """
        LambdaMART 모델 학습

        Args:
            training_data: [(query, documents, relevance_labels)]
            validation_data: 검증 데이터 (선택사항)
        """
        # 특징 추출
        X_train = []
        y_train = []
        qids = []

        for qid, (query, documents, labels) in enumerate(training_data):
            for doc, label in zip(documents, labels):
                features = self.extract_features(query, doc)
                X_train.append(self.features_to_vector(features))
                y_train.append(label)
                qids.append(qid)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        qids = np.array(qids)

        # 특징 스케일링
        X_train = self.scaler.fit_transform(X_train)

        # XGBoost DMatrix 생성
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(np.bincount(qids))

        # LambdaMART 파라미터
        params = {
            'objective': 'rank:ndcg',
            'eval_metric': ['ndcg@10', 'map@10'],
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.0
        }

        # 모델 학습
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            early_stopping_rounds=50,
            verbose_eval=50
        )

        # Feature importance 분석
        self._analyze_feature_importance()

    def _analyze_feature_importance(self):
        """특징 중요도 분석"""
        if self.model is None:
            return

        importance = self.model.get_score(importance_type='gain')

        # Feature name 매핑
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            key = f'f{i}'
            if key in importance:
                feature_importance[name] = importance[key]

        # 중요도 순으로 정렬
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("=== Feature Importance ===")
        for feature, score in sorted_importance[:10]:
            print(f"{feature}: {score:.2f}")

    def predict_ranking_scores(self, query: str, documents: List[Dict],
                             user_context: Optional[Dict] = None) -> List[float]:
        """
        문서들의 랭킹 점수 예측
        """
        if self.model is None:
            # 모델이 없으면 기본 점수 사용
            return [doc.get('score', 0.0) for doc in documents]

        # 특징 추출
        features = []
        for doc in documents:
            feat = self.extract_features(query, doc, user_context)
            features.append(self.features_to_vector(feat))

        # 스케일링
        X = self.scaler.transform(np.array(features))

        # 예측
        dtest = xgb.DMatrix(X)
        scores = self.model.predict(dtest)

        return scores.tolist()

    def rerank_with_ltr(self, query: str, documents: List[Dict],
                       user_context: Optional[Dict] = None) -> List[Dict]:
        """
        Learning to Rank를 사용한 재순위화
        """
        # 점수 예측
        scores = self.predict_ranking_scores(query, documents, user_context)

        # 문서와 점수 결합
        for doc, score in zip(documents, scores):
            doc['ltr_score'] = score

        # 점수순 정렬
        reranked = sorted(documents, key=lambda x: x['ltr_score'], reverse=True)

        return reranked

    def collect_feedback(self, query: str, documents: List[Dict],
                        clicks: List[int], dwell_times: List[float]):
        """
        사용자 피드백 수집 (온라인 학습용)
        """
        # 클릭과 체류시간을 기반으로 관련성 레이블 생성
        relevance_labels = []
        for click, dwell_time in zip(clicks, dwell_times):
            if not click:
                label = 0
            elif dwell_time < 10:  # 10초 미만
                label = 1
            elif dwell_time < 30:  # 30초 미만
                label = 2
            else:
                label = 3  # 높은 관련성

            relevance_labels.append(label)

        # 학습 버퍼에 추가
        self.training_buffer.append((query, documents, relevance_labels))

        # 버퍼가 가득 차면 재학습
        if len(self.training_buffer) >= self.buffer_size:
            self._retrain_online()

    def _retrain_online(self):
        """온라인 재학습"""
        print("Online retraining with {} examples...".format(len(self.training_buffer)))

        # 기존 모델을 베이스로 추가 학습
        self.train_model(self.training_buffer)

        # 버퍼 초기화
        self.training_buffer = []

# 통합 재순위화 시스템
class UnifiedRerankingSystem:
    def __init__(self):
        """
        모든 재순위화 기법을 통합한 시스템
        """
        self.cross_encoder = AdvancedCrossEncoderReranker()
        self.colbert = ColBERTv2(ColBERTConfig())
        self.diversity_reranker = DiversityAwareReranker()
        self.ltr_model = LearningToRankRAG()

    def rerank(self, query: str, initial_results: List[Dict],
              reranking_strategy: str = 'hybrid',
              user_context: Optional[Dict] = None) -> List[Dict]:
        """
        통합 재순위화 수행

        Strategies:
        - 'cross_encoder': Cross-Encoder만 사용
        - 'colbert': ColBERT만 사용
        - 'diversity': MMR 다양성 재순위화
        - 'ltr': Learning to Rank
        - 'hybrid': 모든 기법 조합 (기본값)
        """
        if reranking_strategy == 'cross_encoder':
            return self._rerank_cross_encoder(query, initial_results)

        elif reranking_strategy == 'colbert':
            return self._rerank_colbert(query, initial_results)

        elif reranking_strategy == 'diversity':
            return self.diversity_reranker.mmr_rerank(query, initial_results)

        elif reranking_strategy == 'ltr':
            return self.ltr_model.rerank_with_ltr(query, initial_results, user_context)

        elif reranking_strategy == 'hybrid':
            # 1단계: Cross-Encoder로 Top-30 재순위화
            ce_results = self._rerank_cross_encoder(query, initial_results[:50])[:30]

            # 2단계: LTR 모델로 점수 조정
            if self.ltr_model.model is not None:
                ltr_results = self.ltr_model.rerank_with_ltr(query, ce_results, user_context)
            else:
                ltr_results = ce_results

            # 3단계: MMR로 최종 다양성 확보
            final_results = self.diversity_reranker.mmr_rerank(
                query, ltr_results[:20], top_k=10
            )

            return final_results

        else:
            raise ValueError(f"Unknown strategy: {reranking_strategy}")

    def _rerank_cross_encoder(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Cross-Encoder 재순위화 래퍼"""
        results = self.cross_encoder.rerank(query, documents)
        return [
            {
                'id': r.doc_id,
                'content': r.content,
                'score': r.reranked_score
            }
            for r in results
        ]

    def _rerank_colbert(self, query: str, documents: List[Dict]) -> List[Dict]:
        """ColBERT 재순위화"""
        # ColBERT 인덱스 구축 (실제로는 사전 구축)
        doc_contents = [d['content'] for d in documents]
        index, boundaries = self.colbert.build_index(doc_contents)

        # 검색 수행
        results = self.colbert.retrieve(query, index, boundaries)

        # 결과 포맷팅
        reranked = []
        for doc_id, score in results:
            reranked.append({
                'id': documents[doc_id]['id'],
                'content': documents[doc_id]['content'],
                'score': score
            })

        return reranked

# 사용 예제
print("=== 고급 재순위화 시스템 데모 ===\\n")

# 초기 검색 결과
initial_results = [
    {'id': '1', 'content': 'Cross-Encoder는 쿼리와 문서를 함께 인코딩합니다.', 'score': 0.89},
    {'id': '2', 'content': 'ColBERT는 토큰 레벨 상호작용을 사용합니다.', 'score': 0.87},
    {'id': '3', 'content': 'MMR은 다양성을 위한 재순위화 알고리즘입니다.', 'score': 0.85},
    {'id': '4', 'content': 'Learning to Rank는 기계학습 기반 순위 최적화입니다.', 'score': 0.83},
    {'id': '5', 'content': 'BERT 기반 검색 모델들의 성능 비교.', 'score': 0.82},
]

# 통합 시스템 초기화
unified_system = UnifiedRerankingSystem()

# 다양한 전략으로 재순위화
strategies = ['cross_encoder', 'diversity', 'hybrid']

for strategy in strategies:
    print(f"\\n=== {strategy.upper()} 전략 결과 ===")
    results = unified_system.rerank(
        "재순위화 알고리즘의 종류",
        initial_results,
        reranking_strategy=strategy
    )

    for i, doc in enumerate(results[:5], 1):
        print(f"{i}. [{doc['id']}] {doc['content'][:50]}... (점수: {doc.get('score', 0):.3f})")`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Practical Exercise */}
      <section className="bg-gradient-to-r from-orange-500 to-red-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">실습 과제</h2>

        <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
          <h3 className="font-bold mb-4">고급 재순위화 시스템 구축</h3>

          <div className="space-y-4">
            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">📋 요구사항</h4>
              <ol className="space-y-2 text-sm">
                <li>1. Cross-Encoder와 ColBERT를 모두 활용한 하이브리드 시스템 구현</li>
                <li>2. 쿼리 유형별 최적 재순위화 전략 자동 선택</li>
                <li>3. A/B 테스트를 통한 λ 파라미터 최적화</li>
                <li>4. 실시간 사용자 피드백 기반 온라인 학습</li>
                <li>5. 재순위화 성능 모니터링 대시보드 구축</li>
              </ol>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🎯 성능 목표</h4>
              <ul className="space-y-1 text-sm">
                <li>• MRR@10: 0.4 이상</li>
                <li>• NDCG@10: 0.6 이상</li>
                <li>• 재순위화 레이턴시: &lt; 100ms (P95)</li>
                <li>• 다양성 점수: 0.7 이상</li>
              </ul>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">💡 도전 과제</h4>
              <p className="text-sm">
                다국어 재순위화 지원을 추가하여 한국어, 영어, 일본어 쿼리에 대해
                언어별 최적화된 재순위화를 수행하는 시스템으로 확장해보세요.
                특히 Cross-lingual 검색에서의 재순위화 전략을 고민해보세요.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 Cross-Encoder & Reranking 모델',
            icon: 'web' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'Sentence-Transformers Reranking',
                authors: 'UKPLab',
                year: '2024',
                description: 'Cross-Encoder 공식 라이브러리 - ms-marco-MiniLM, mxbai-rerank',
                link: 'https://www.sbert.net/docs/pretrained_cross-encoders.html'
              },
              {
                title: 'Cohere Rerank API',
                authors: 'Cohere',
                year: '2025',
                description: 'State-of-the-art 재순위화 - 다국어 지원, 실시간 API',
                link: 'https://docs.cohere.com/docs/rerank'
              },
              {
                title: 'Jina Reranker Models',
                authors: 'Jina AI',
                year: '2024',
                description: '경량 재순위화 모델 - jina-reranker-v1-base-en',
                link: 'https://jina.ai/reranker'
              },
              {
                title: 'RankGPT: LLM-based Reranking',
                authors: 'Microsoft Research',
                year: '2023',
                description: 'GPT를 활용한 재순위화 - Zero-shot Relevance 판단',
                link: 'https://github.com/sunnweiwei/RankGPT'
              },
              {
                title: 'BGE Reranker by BAAI',
                authors: 'Beijing Academy of AI',
                year: '2024',
                description: '고성능 중국어/영어 재순위화 - bge-reranker-large',
                link: 'https://huggingface.co/BAAI/bge-reranker-large'
              }
            ]
          },
          {
            title: '📖 Learning-to-Rank 연구',
            icon: 'research' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'Sentence-BERT: Cross-Encoders for Semantic Search',
                authors: 'Reimers & Gurevych',
                year: '2019',
                description: 'Cross-Encoder 기본 논문 - BERT 기반 문장 쌍 분류',
                link: 'https://arxiv.org/abs/1908.10084'
              },
              {
                title: 'ColBERT: Efficient and Effective Passage Search',
                authors: 'Khattab & Zaharia, Stanford',
                year: '2020',
                description: 'Late Interaction - Bi-Encoder + Cross-Encoder 장점 결합',
                link: 'https://arxiv.org/abs/2004.12832'
              },
              {
                title: 'RankNet to LambdaRank to LambdaMART',
                authors: 'Burges et al., Microsoft Research',
                year: '2010',
                description: 'Learning-to-Rank 알고리즘 발전사 - Pairwise to Listwise',
                link: 'https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/'
              },
              {
                title: 'MonoT5 & DuoT5: T5-based Reranking',
                authors: 'Nogueira et al., University of Waterloo',
                year: '2020',
                description: 'T5로 재순위화 - Text-to-Text Relevance 판단',
                link: 'https://arxiv.org/abs/2003.06713'
              }
            ]
          },
          {
            title: '🛠️ 실무 재순위화 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Haystack Ranker Component',
                authors: 'deepset',
                year: '2024',
                description: 'RAG 파이프라인 재순위화 - SentenceTransformers 통합',
                link: 'https://docs.haystack.deepset.ai/docs/ranker'
              },
              {
                title: 'LlamaIndex Reranking Postprocessor',
                authors: 'LlamaIndex',
                year: '2024',
                description: 'Query Engine 재순위화 - Cohere, SentenceTransformers 지원',
                link: 'https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/CohereRerank.html'
              },
              {
                title: 'Vespa Ranking Framework',
                authors: 'Vespa.ai',
                year: '2024',
                description: '대규모 재순위화 엔진 - ONNX 모델, LightGBM, XGBoost 지원',
                link: 'https://docs.vespa.ai/en/ranking.html'
              },
              {
                title: 'FlashRank: Fast Reranking',
                authors: 'PrithivirajDamodaran',
                year: '2024',
                description: '초경량 재순위화 - CPU에서 빠른 추론, 40MB 모델',
                link: 'https://github.com/PrithivirajDamodaran/FlashRank'
              },
              {
                title: 'Rank-BM25 Python Library',
                authors: 'dorianbrown',
                year: '2024',
                description: 'BM25 재순위화 - 경량 키워드 기반 재점수화',
                link: 'https://github.com/dorianbrown/rank_bm25'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter3"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 분산 RAG 시스템
          </Link>

          <Link
            href="/modules/rag/advanced/chapter5"
            className="inline-flex items-center gap-2 bg-orange-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-orange-600 transition-colors"
          >
            다음: RAG 평가와 모니터링
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </>
  )
}
