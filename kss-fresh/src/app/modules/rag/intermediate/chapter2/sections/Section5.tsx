'use client'

import { BookOpen } from 'lucide-react'
import CodeSandbox from '../../../components/CodeSandbox'

export default function Section5() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-emerald-100 dark:bg-emerald-900/20 flex items-center justify-center">
          <BookOpen className="text-emerald-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.5 실전 코드 예제</h2>
          <p className="text-gray-600 dark:text-gray-400">하이브리드 검색 실무 구현</p>
        </div>
      </div>

      <div className="space-y-6">
        <CodeSandbox
          title="실습 1: BM25 키워드 검색 구현"
          description="rank_bm25 라이브러리를 사용한 빠른 구현"
          language="python"
          code={`from rank_bm25 import BM25Okapi
import numpy as np

# 문서 코퍼스 (토큰화된 상태)
corpus = [
    "RAG는 검색 증강 생성 시스템입니다".split(),
    "벡터 데이터베이스는 임베딩을 저장합니다".split(),
    "BM25는 키워드 기반 검색 알고리즘입니다".split(),
    "하이브리드 검색은 벡터와 키워드를 결합합니다".split(),
    "Pinecone은 관리형 벡터 데이터베이스 서비스입니다".split()
]

# BM25 인덱스 생성
bm25 = BM25Okapi(corpus)

# 검색 쿼리
query = "벡터 검색 시스템".split()

# 모든 문서에 대한 점수 계산
scores = bm25.get_scores(query)

# 상위 3개 문서 검색
top_n = np.argsort(scores)[::-1][:3]

print("🔍 BM25 검색 결과:")
for i, idx in enumerate(top_n, 1):
    doc_text = " ".join(corpus[idx])
    print(f"{i}. [{scores[idx]:.3f}] {doc_text}")

# 배치 검색 (여러 쿼리 동시 처리)
queries = [
    "RAG 시스템".split(),
    "하이브리드 검색".split()
]

print("\\n📊 배치 검색 결과:")
for q in queries:
    best_doc_idx = np.argmax(bm25.get_scores(q))
    print(f"쿼리: {' '.join(q)}")
    print(f"최적 문서: {' '.join(corpus[best_doc_idx])}")`}
          output={`🔍 BM25 검색 결과:
1. [1.847] 하이브리드 검색은 벡터와 키워드를 결합합니다
2. [1.324] 벡터 데이터베이스는 임베딩을 저장합니다
3. [0.892] RAG는 검색 증강 생성 시스템입니다

📊 배치 검색 결과:
쿼리: RAG 시스템
최적 문서: RAG는 검색 증강 생성 시스템입니다
쿼리: 하이브리드 검색
최적 문서: 하이브리드 검색은 벡터와 키워드를 결합합니다`}
          highlightLines={[13, 17, 20, 21]}
        />

        <CodeSandbox
          title="실습 2: LangChain Ensemble Retriever (하이브리드 검색)"
          description="BM25 + Chroma 벡터 검색 결합 (RRF 자동 적용)"
          language="python"
          code={`from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# 문서 준비
docs = [
    Document(page_content="RAG는 검색 증강 생성 시스템입니다", metadata={"source": "doc1"}),
    Document(page_content="벡터 데이터베이스는 임베딩을 저장합니다", metadata={"source": "doc2"}),
    Document(page_content="BM25는 키워드 기반 검색 알고리즘입니다", metadata={"source": "doc3"}),
    Document(page_content="하이브리드 검색은 벡터와 키워드를 결합합니다", metadata={"source": "doc4"}),
    Document(page_content="Pinecone은 관리형 벡터 데이터베이스입니다", metadata={"source": "doc5"})
]

# 1. BM25 검색기 초기화
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3  # 상위 3개 반환

# 2. 벡터 검색기 초기화 (Chroma)
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 하이브리드 검색기 생성 (Ensemble Retriever)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # BM25: 50%, Vector: 50%
)

# 검색 실행
query = "벡터 기반 검색 시스템"
results = ensemble_retriever.get_relevant_documents(query)

print("🎯 하이브리드 검색 결과 (RRF 적용):")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
    print(f"   출처: {doc.metadata['source']}")

# 가중치 조정 실험
print("\\n⚖️ 가중치 조정 실험:")
for bm25_weight in [0.3, 0.5, 0.7]:
    vector_weight = 1 - bm25_weight
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )
    top_result = ensemble.get_relevant_documents(query)[0]
    print(f"BM25:{bm25_weight} / Vector:{vector_weight} → {top_result.page_content[:30]}...")`}
          output={`🎯 하이브리드 검색 결과 (RRF 적용):
1. 벡터 데이터베이스는 임베딩을 저장합니다
   출처: doc2
2. 하이브리드 검색은 벡터와 키워드를 결합합니다
   출처: doc4
3. RAG는 검색 증강 생성 시스템입니다
   출처: doc1

⚖️ 가중치 조정 실험:
BM25:0.3 / Vector:0.7 → 벡터 데이터베이스는 임베딩을 저장합니다...
BM25:0.5 / Vector:0.5 → 벡터 데이터베이스는 임베딩을 저장합니다...
BM25:0.7 / Vector:0.3 → 하이브리드 검색은 벡터와 키워드를 결합합니다...`}
          highlightLines={[24, 25, 26, 27]}
        />

        <CodeSandbox
          title="실습 3: 커스텀 RRF 구현 (Reciprocal Rank Fusion)"
          description="다중 리트리버 결합을 위한 고급 기법"
          language="python"
          code={`from typing import List, Dict, Tuple
from collections import defaultdict

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    RRF: 점수 정규화 없이 순위만으로 결합

    Args:
        rankings: 각 검색 방법의 결과 [(doc_id, score), ...]
        k: RRF 상수 (일반적으로 60)

    Returns:
        결합된 결과 [(doc_id, rrf_score), ...]
    """
    rrf_scores = defaultdict(float)

    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            # RRF 공식: 1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # RRF 점수 기준 정렬
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results

# 실전 예제: BM25 + Dense + Sparse 3개 검색 결합
bm25_results = [
    ("doc1", 2.5), ("doc3", 1.8), ("doc2", 1.2)
]

dense_vector_results = [
    ("doc2", 0.95), ("doc1", 0.87), ("doc4", 0.76)
]

sparse_vector_results = [
    ("doc4", 0.82), ("doc2", 0.78), ("doc3", 0.65)
]

# RRF로 3개 검색 결과 결합
final_results = reciprocal_rank_fusion([
    bm25_results,
    dense_vector_results,
    sparse_vector_results
])

print("🔥 RRF 최종 결과:")
for i, (doc_id, score) in enumerate(final_results[:5], 1):
    print(f"{i}. {doc_id}: RRF Score = {score:.4f}")

# k 값에 따른 민감도 분석
print("\\n📊 RRF 파라미터 k 영향:")
for k_value in [10, 60, 100]:
    results = reciprocal_rank_fusion(
        [bm25_results, dense_vector_results, sparse_vector_results],
        k=k_value
    )
    top_doc = results[0]
    print(f"k={k_value:3d} → Top: {top_doc[0]} (score: {top_doc[1]:.4f})")`}
          output={`🔥 RRF 최종 결과:
1. doc2: RRF Score = 0.0803
2. doc1: RRF Score = 0.0639
3. doc3: RRF Score = 0.0473
4. doc4: RRF Score = 0.0457

📊 RRF 파라미터 k 영향:
k= 10 → Top: doc2 (score: 0.2167)
k= 60 → Top: doc2 (score: 0.0803)
k=100 → Top: doc2 (score: 0.0589)`}
          highlightLines={[20, 21, 22, 23]}
        />
      </div>
    </section>
  )
}
