'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, BookOpen, Search, Zap, Filter, GitMerge, BarChart2 } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter2Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/intermediate"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          중급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <GitMerge size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 2: 하이브리드 검색 전략</h1>
              <p className="text-purple-100 text-lg">키워드와 벡터 검색의 시너지 활용하기</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Why Hybrid Search */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
              <Search className="text-indigo-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.1 하이브리드 검색이 필요한 이유</h2>
              <p className="text-gray-600 dark:text-gray-400">각 검색 방식의 장단점과 보완 관계</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
                <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">🔤 키워드 검색 (BM25)</h3>
                <div className="space-y-3 text-sm text-indigo-700 dark:text-indigo-300">
                  <p><strong>장점:</strong></p>
                  <ul className="space-y-1 pl-4">
                    <li>• 정확한 단어 매칭</li>
                    <li>• 희귀 용어, 고유명사에 강함</li>
                    <li>• 검색 결과 설명 가능</li>
                    <li>• 빠른 속도</li>
                  </ul>
                  <p className="mt-3"><strong>단점:</strong></p>
                  <ul className="space-y-1 pl-4">
                    <li>• 동의어 처리 어려움</li>
                    <li>• 문맥 이해 부족</li>
                    <li>• 철자 오류에 취약</li>
                  </ul>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
                <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">🧠 벡터 검색 (Semantic)</h3>
                <div className="space-y-3 text-sm text-purple-700 dark:text-purple-300">
                  <p><strong>장점:</strong></p>
                  <ul className="space-y-1 pl-4">
                    <li>• 의미적 유사성 파악</li>
                    <li>• 동의어, 유사어 처리</li>
                    <li>• 문맥 기반 이해</li>
                    <li>• 다국어 지원</li>
                  </ul>
                  <p className="mt-3"><strong>단점:</strong></p>
                  <ul className="space-y-1 pl-4">
                    <li>• 고유명사, ID에 약함</li>
                    <li>• 계산 비용 높음</li>
                    <li>• 블랙박스 성격</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">실제 사례로 보는 차이점</h3>
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="font-medium text-gray-900 dark:text-white mb-2">쿼리: "SKU-12345의 재고 현황"</p>
                  <div className="grid md:grid-cols-2 gap-4 mt-3">
                    <div>
                      <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">키워드 검색 ✅</p>
                      <p className="text-sm text-green-600">정확히 SKU-12345를 포함한 문서 검색</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">벡터 검색 ❌</p>
                      <p className="text-sm text-red-600">유사한 제품 코드들을 반환할 수 있음</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="font-medium text-gray-900 dark:text-white mb-2">쿼리: "차가운 음료"</p>
                  <div className="grid md:grid-cols-2 gap-4 mt-3">
                    <div>
                      <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">키워드 검색 ❌</p>
                      <p className="text-sm text-red-600">"차가운"과 "음료"를 정확히 포함한 문서만</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">벡터 검색 ✅</p>
                      <p className="text-sm text-green-600">"아이스커피", "냉음료", "시원한 음료" 등도 검색</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: BM25 Implementation */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
              <Zap className="text-blue-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.2 BM25 알고리즘 구현</h2>
              <p className="text-gray-600 dark:text-gray-400">Best Matching 25의 원리와 최적화</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">BM25 수식 이해하기</h3>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                  BM25는 TF-IDF의 확률론적 해석으로, 문서 길이를 정규화하여 더 정확한 점수를 계산합니다.
                </p>
                <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-xs font-mono">
                  score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1+1)) / (f(qi,D) + k1·(1-b+b·|D|/avgdl))
                </div>
                <ul className="mt-3 space-y-1 text-xs text-gray-600 dark:text-gray-400">
                  <li>• k1: 용어 빈도의 포화점 조절 (일반적으로 1.2)</li>
                  <li>• b: 문서 길이 정규화 강도 (일반적으로 0.75)</li>
                  <li>• avgdl: 평균 문서 길이</li>
                </ul>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">Python 구현 예제</h3>
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
                <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`from typing import List, Dict
import math
from collections import Counter

class BM25:
    def __init__(self, corpus: List[List[str]], k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.avgdl = sum(len(doc) for doc in corpus) / len(corpus)
        self.doc_freqs = self._calc_doc_freqs()
        self.idf = self._calc_idf()
        
    def _calc_doc_freqs(self) -> Dict[str, int]:
        """각 용어가 나타나는 문서 수 계산"""
        doc_freqs = {}
        for doc in self.corpus:
            for word in set(doc):
                doc_freqs[word] = doc_freqs.get(word, 0) + 1
        return doc_freqs
    
    def _calc_idf(self) -> Dict[str, float]:
        """역문서빈도(IDF) 계산"""
        idf = {}
        N = len(self.corpus)
        for word, freq in self.doc_freqs.items():
            idf[word] = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
        return idf
    
    def score(self, query: List[str], doc_idx: int) -> float:
        """쿼리와 문서의 BM25 점수 계산"""
        doc = self.corpus[doc_idx]
        doc_len = len(doc)
        scores = 0.0
        
        doc_freqs = Counter(doc)
        for term in query:
            if term not in self.idf:
                continue
                
            term_freq = doc_freqs[term]
            idf = self.idf[term]
            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            scores += numerator / denominator
            
        return scores

# 사용 예제
corpus = [
    ["신용카드", "결제", "시스템", "오류", "발생"],
    ["카드", "결제", "실패", "환불", "처리"],
    ["시스템", "점검", "안내", "공지사항"],
]

bm25 = BM25(corpus)
query = ["카드", "결제", "오류"]

for i, doc in enumerate(corpus):
    score = bm25.score(query, i)
    print(f"문서 {i}: {' '.join(doc)} - 점수: {score:.3f}")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Combining Search Methods */}
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

        {/* Section 4: Real World Case Studies */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
              <BarChart2 className="text-green-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.4 실제 적용 사례와 성능 향상</h2>
              <p className="text-gray-600 dark:text-gray-400">기업들의 하이브리드 검색 도입 결과</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">🏢 이커머스 플랫폼 사례</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">문제 상황</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 상품명/SKU 검색 정확도 낮음</li>
                    <li>• "빨간 운동화" → "레드 스니커즈" 매칭 안됨</li>
                    <li>• 브랜드명 오타 처리 불가</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">개선 결과</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 검색 정확도 35% 향상</li>
                    <li>• 클릭률(CTR) 28% 증가</li>
                    <li>• 검색 포기율 40% 감소</li>
                  </ul>
                </div>
              </div>
              
              <div className="mt-4 p-4 bg-emerald-100 dark:bg-emerald-900/40 rounded-lg">
                <p className="text-sm text-emerald-800 dark:text-emerald-200">
                  <strong>핵심 전략:</strong> SKU, 브랜드명은 BM25로, 상품 설명은 벡터 검색으로 처리. 
                  쿼리 타입에 따라 가중치 동적 조정.
                </p>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">📚 기술 문서 검색 시스템</h3>
              
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">구현 상세</h4>
                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                  <pre className="text-xs text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`# 문서 타입별 가중치 설정
WEIGHT_CONFIG = {
    "api_reference": {"bm25": 0.7, "vector": 0.3},  # 함수명, 파라미터 중요
    "tutorial": {"bm25": 0.3, "vector": 0.7},       # 개념 설명 중요
    "error_guide": {"bm25": 0.6, "vector": 0.4},    # 에러 코드 중요
    "conceptual": {"bm25": 0.2, "vector": 0.8}      # 의미 이해 중요
}

# 메타데이터 부스팅
if "error" in query and doc.type == "error_guide":
    score *= 1.5  # 에러 관련 쿼리는 에러 가이드 문서 우선`}
                  </pre>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="text-2xl font-bold text-blue-600">92%</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">정답 포함률 (Top 5)</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="text-2xl font-bold text-blue-600">1.2초</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">평균 응답 시간</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="text-2xl font-bold text-blue-600">4.7/5</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">사용자 만족도</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: Practical Exercise */}
        <section className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">실습 과제</h2>
          
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">하이브리드 검색 시스템 구축</h3>
            
            <div className="space-y-4">
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">📋 요구사항</h4>
                <ol className="space-y-2 text-sm">
                  <li>1. Elasticsearch와 벡터 DB를 사용한 하이브리드 검색 구현</li>
                  <li>2. 한국어 형태소 분석기 적용 (Nori, Komoran 등)</li>
                  <li>3. 쿼리 타입 자동 분류 (키워드형, 자연어형, 혼합형)</li>
                  <li>4. A/B 테스트를 통한 최적 가중치 찾기</li>
                  <li>5. 검색 품질 메트릭 측정 (MRR, NDCG, Precision@K)</li>
                </ol>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🎯 평가 데이터셋</h4>
                <ul className="space-y-1 text-sm">
                  <li>• 1000개의 문서 (뉴스, 제품 설명, FAQ 혼합)</li>
                  <li>• 100개의 테스트 쿼리와 정답 셋</li>
                  <li>• 키워드형 30%, 자연어형 50%, 혼합형 20%</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">💡 도전 과제</h4>
                <p className="text-sm">
                  검색 로그를 분석하여 사용자의 검색 패턴을 학습하고, 
                  개인화된 가중치를 적용하는 시스템으로 확장해보세요.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 하이브리드 검색 공식 문서',
              icon: 'web' as const,
              color: 'border-teal-500',
              items: [
                {
                  title: 'Elasticsearch Hybrid Search',
                  authors: 'Elastic',
                  year: '2025',
                  description: 'BM25 + kNN 벡터 검색 결합 - 프로덕션급 구현',
                  link: 'https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html'
                },
                {
                  title: 'Weaviate Hybrid Search',
                  authors: 'Weaviate',
                  year: '2025',
                  description: 'BM25 + ANN 벡터 검색 - 실시간 융합 알고리즘',
                  link: 'https://weaviate.io/developers/weaviate/search/hybrid'
                },
                {
                  title: 'LangChain Ensemble Retriever',
                  authors: 'LangChain',
                  year: '2025',
                  description: '다중 리트리버 결합 - RRF(Reciprocal Rank Fusion)',
                  link: 'https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble'
                },
                {
                  title: 'LlamaIndex Hybrid Retriever',
                  authors: 'LlamaIndex',
                  year: '2025',
                  description: '벡터 + 키워드 검색 통합 - 자동 가중치 조정',
                  link: 'https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/'
                },
                {
                  title: 'Pinecone Sparse-Dense Search',
                  authors: 'Pinecone',
                  year: '2025',
                  description: '단일 인덱스에서 하이브리드 검색 - 통합 API',
                  link: 'https://docs.pinecone.io/docs/hybrid-search'
                }
              ]
            },
            {
              title: '📖 검색 알고리즘 연구 논문',
              icon: 'research' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'BM25: The Probabilistic Relevance Framework',
                  authors: 'Robertson & Zaragoza',
                  year: '2009',
                  description: 'BM25 알고리즘의 이론적 기반 - TF-IDF 확률론적 개선',
                  link: 'https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf'
                },
                {
                  title: 'Dense Passage Retrieval for Open-Domain QA',
                  authors: 'Karpukhin et al., Meta AI',
                  year: '2020',
                  description: '벡터 검색 기반 QA - BM25 대비 9-19% 성능 향상',
                  link: 'https://arxiv.org/abs/2004.04906'
                },
                {
                  title: 'Reciprocal Rank Fusion (RRF)',
                  authors: 'Cormack et al.',
                  year: '2009',
                  description: '다중 순위 결합 알고리즘 - 점수 정규화 불필요',
                  link: 'https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf'
                },
                {
                  title: 'BEIR: Heterogeneous Benchmark for IR',
                  authors: 'Thakur et al.',
                  year: '2021',
                  description: '18개 데이터셋으로 하이브리드 검색 벤치마크',
                  link: 'https://arxiv.org/abs/2104.08663'
                }
              ]
            },
            {
              title: '🛠️ 실전 구현 & 도구',
              icon: 'tools' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'rank_bm25 (Python)',
                  authors: 'dorianbrown',
                  year: '2024',
                  description: '순수 Python BM25 구현 - Gensim 기반, 한국어 지원',
                  link: 'https://github.com/dorianbrown/rank_bm25'
                },
                {
                  title: 'Haystack Hybrid Retrieval',
                  authors: 'deepset',
                  year: '2025',
                  description: 'NLP 프레임워크 - BM25 + DPR 통합 파이프라인',
                  link: 'https://haystack.deepset.ai/tutorials/08_preprocessing'
                },
                {
                  title: 'Qdrant Hybrid Search',
                  authors: 'Qdrant',
                  year: '2025',
                  description: 'Rust 기반 벡터 DB - 빠른 하이브리드 검색 API',
                  link: 'https://qdrant.tech/documentation/concepts/hybrid-queries/'
                },
                {
                  title: 'ColBERT: Efficient Passage Retrieval',
                  authors: 'Stanford NLP',
                  year: '2023',
                  description: '지연 상호작용 모델 - 벡터+키워드 장점 결합',
                  link: 'https://github.com/stanford-futuredata/ColBERT'
                },
                {
                  title: 'Vespa Hybrid Search',
                  authors: 'Vespa.ai',
                  year: '2025',
                  description: '대규모 검색 엔진 - BM25 + ANN + 신경망 순위화',
                  link: 'https://docs.vespa.ai/en/reference/ranking-expressions.html'
                }
              ]
            }
          ]}
        />
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/intermediate/chapter1"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 고급 벡터 데이터베이스
          </Link>
          
          <Link
            href="/modules/rag/intermediate/chapter3"
            className="inline-flex items-center gap-2 bg-indigo-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-600 transition-colors"
          >
            다음: RAG를 위한 프롬프트 엔지니어링
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}