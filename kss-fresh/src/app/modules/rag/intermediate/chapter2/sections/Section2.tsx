'use client'

import { Zap } from 'lucide-react'

export default function Section2() {
  return (
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
  )
}
