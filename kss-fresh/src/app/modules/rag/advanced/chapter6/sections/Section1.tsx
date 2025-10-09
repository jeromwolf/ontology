'use client'

import { Brain } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-violet-100 dark:bg-violet-900/20 flex items-center justify-center">
          <Brain className="text-violet-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.1 Self-RAG: 자기 성찰하는 검색 증강 생성</h2>
          <p className="text-gray-600 dark:text-gray-400">Washington University의 혁신적 연구 (2023.10)</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-violet-50 dark:bg-violet-900/20 p-6 rounded-xl border border-violet-200 dark:border-violet-700">
          <h3 className="font-bold text-violet-800 dark:text-violet-200 mb-4">Self-RAG의 혁신적 접근법</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>Self-RAG는 기존 RAG의 한계를 극복하는 패러다임 전환입니다.</strong>
              모델이 스스로 검색 필요성을 판단하고, 검색된 정보의 관련성을 평가하며,
              생성된 답변의 품질을 자체적으로 검증합니다. 이는 인간의 비판적 사고 과정을
              모방한 것으로, RAG 시스템의 신뢰성과 효율성을 크게 향상시킵니다.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-violet-600 dark:text-violet-400 mb-2">🤔 Retrieval Decision</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                쿼리 분석을 통해 외부 지식이 필요한지 스스로 판단
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-2">✅ Relevance Check</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                검색된 각 문서의 관련성을 자체 평가하여 필터링
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">📊 Self-Reflection</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                생성된 답변의 품질과 정확성을 스스로 검증
              </p>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">Self-RAG 구현 및 학습</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
import numpy as np
from enum import Enum

class ReflectionToken(Enum):
    """Self-RAG 특수 토큰"""
    RETRIEVE = "[Retrieve]"
    NO_RETRIEVE = "[No Retrieve]"
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"
    SUPPORTED = "[Supported]"
    NOT_SUPPORTED = "[Not Supported]"
    USEFUL = "[Useful]"
    NOT_USEFUL = "[Not Useful]"

@dataclass
class SelfRAGOutput:
    """Self-RAG 출력 구조"""
    answer: str
    retrieve_decision: bool
    relevance_scores: List[float]
    support_scores: List[float]
    utility_score: float
    retrieved_docs: List[Dict[str, Any]]
    reflection_tokens: List[str]

class SelfRAG(nn.Module):
    def __init__(self, base_model: str = "meta-llama/Llama-2-7b-hf"):
        """
        Self-RAG 모델 구현
        - 적응형 검색 결정
        - 다중 평가 메커니즘
        - 자기 성찰 생성
        """
        super().__init__()

        # 기본 언어 모델
        self.base_model = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # 특수 토큰 추가
        self._add_special_tokens()

        # 평가 헤드들
        hidden_size = self.base_model.config.hidden_size
        self.retrieve_classifier = nn.Linear(hidden_size, 2)  # 검색 필요성
        self.relevance_classifier = nn.Linear(hidden_size, 2)  # 관련성
        self.support_classifier = nn.Linear(hidden_size, 2)  # 지원도
        self.utility_classifier = nn.Linear(hidden_size, 2)  # 유용성

        # 검색 엔진 (시뮬레이션)
        self.retriever = None  # 실제로는 Dense Retriever

    def _add_special_tokens(self):
        """특수 토큰 추가"""
        special_tokens = [token.value for token in ReflectionToken]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.base_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, query: str, context: Optional[str] = None) -> SelfRAGOutput:
        """
        Self-RAG 전체 파이프라인
        """
        # 1단계: 검색 필요성 판단
        retrieve_decision = self._decide_retrieval(query, context)

        retrieved_docs = []
        relevance_scores = []

        # 2단계: 조건부 검색
        if retrieve_decision:
            # 문서 검색
            retrieved_docs = self._retrieve_documents(query)

            # 각 문서의 관련성 평가
            for doc in retrieved_docs:
                relevance = self._evaluate_relevance(query, doc['content'])
                relevance_scores.append(relevance)
                doc['relevance_score'] = relevance

            # 관련성 높은 문서만 필터링
            threshold = 0.5
            filtered_docs = [doc for doc, score in zip(retrieved_docs, relevance_scores)
                           if score > threshold]
        else:
            filtered_docs = []

        # 3단계: 답변 생성
        answer, support_scores = self._generate_with_reflection(
            query, filtered_docs, context
        )

        # 4단계: 최종 유용성 평가
        utility_score = self._evaluate_utility(query, answer)

        # 5단계: 반영 토큰 생성
        reflection_tokens = self._generate_reflection_tokens(
            retrieve_decision, relevance_scores, support_scores, utility_score
        )

        return SelfRAGOutput(
            answer=answer,
            retrieve_decision=retrieve_decision,
            relevance_scores=relevance_scores,
            support_scores=support_scores,
            utility_score=utility_score,
            retrieved_docs=retrieved_docs,
            reflection_tokens=reflection_tokens
        )

    def _decide_retrieval(self, query: str, context: Optional[str] = None) -> bool:
        """검색 필요성 결정"""
        # 쿼리 인코딩
        prompt = f"Query: {query}"
        if context:
            prompt = f"Context: {context}\n{prompt}"

        inputs = self.tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.base_model(**inputs)
            hidden_states = outputs.last_hidden_state

            # 마지막 토큰의 hidden state 사용
            pooled = hidden_states[:, -1, :]

            # 검색 필요성 분류
            logits = self.retrieve_classifier(pooled)
            prob_retrieve = torch.softmax(logits, dim=-1)[0, 1].item()

        # 임계값 기반 결정
        return prob_retrieve > 0.5

    def _retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """문서 검색 (시뮬레이션)"""
        # 실제로는 Dense Retriever 사용
        sample_docs = [
            {
                'id': f'doc_{i}',
                'content': f'This is a sample document {i} related to {query}',
                'score': 0.9 - i * 0.1
            }
            for i in range(top_k)
        ]
        return sample_docs

    def _evaluate_relevance(self, query: str, document: str) -> float:
        """문서 관련성 평가"""
        prompt = f"Query: {query}\nDocument: {document}\nIs this document relevant?"

        inputs = self.tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.base_model(**inputs)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states[:, -1, :]

            logits = self.relevance_classifier(pooled)
            prob_relevant = torch.softmax(logits, dim=-1)[0, 1].item()

        return prob_relevant

    def _generate_with_reflection(self, query: str, documents: List[Dict[str, Any]],
                                 context: Optional[str] = None) -> Tuple[str, List[float]]:
        """반영을 포함한 답변 생성"""
        # 프롬프트 구성
        prompt_parts = []

        if context:
            prompt_parts.append(f"Context: {context}")

        prompt_parts.append(f"Query: {query}")

        if documents:
            prompt_parts.append("\nRetrieved Documents:")
            for i, doc in enumerate(documents):
                prompt_parts.append(f"[{i+1}] {doc['content']}")

        prompt_parts.append("\nGenerate answer with reflection:")
        prompt = "\n".join(prompt_parts)

        # 답변 생성
        inputs = self.tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=1024)

        with torch.no_grad():
            # 생성 시 특수 토큰 포함
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 답변과 지원도 점수 추출
        answer, support_scores = self._parse_generated_output(generated_text, documents)

        return answer, support_scores

    def _parse_generated_output(self, generated_text: str,
                               documents: List[Dict[str, Any]]) -> Tuple[str, List[float]]:
        """생성된 출력 파싱"""
        # 특수 토큰 기반으로 파싱
        answer_parts = []
        support_scores = []

        # 간단한 파싱 (실제로는 더 정교한 로직 필요)
        lines = generated_text.split('\n')
        for line in lines:
            if ReflectionToken.SUPPORTED.value in line:
                support_scores.append(1.0)
            elif ReflectionToken.NOT_SUPPORTED.value in line:
                support_scores.append(0.0)
            elif not any(token.value in line for token in ReflectionToken):
                answer_parts.append(line)

        # 부족한 support score는 0으로 채움
        while len(support_scores) < len(documents):
            support_scores.append(0.0)

        answer = ' '.join(answer_parts).strip()
        return answer, support_scores

    def _evaluate_utility(self, query: str, answer: str) -> float:
        """답변의 유용성 평가"""
        prompt = f"Query: {query}\nAnswer: {answer}\nIs this answer useful?"

        inputs = self.tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.base_model(**inputs)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states[:, -1, :]

            logits = self.utility_classifier(pooled)
            prob_useful = torch.softmax(logits, dim=-1)[0, 1].item()

        return prob_useful

    def _generate_reflection_tokens(self, retrieve_decision: bool,
                                   relevance_scores: List[float],
                                   support_scores: List[float],
                                   utility_score: float) -> List[str]:
        """반영 토큰 생성"""
        tokens = []

        # 검색 결정
        if retrieve_decision:
            tokens.append(ReflectionToken.RETRIEVE.value)
        else:
            tokens.append(ReflectionToken.NO_RETRIEVE.value)

        # 관련성 평가
        for score in relevance_scores:
            if score > 0.5:
                tokens.append(ReflectionToken.RELEVANT.value)
            else:
                tokens.append(ReflectionToken.IRRELEVANT.value)

        # 지원도 평가
        for score in support_scores:
            if score > 0.5:
                tokens.append(ReflectionToken.SUPPORTED.value)
            else:
                tokens.append(ReflectionToken.NOT_SUPPORTED.value)

        # 유용성 평가
        if utility_score > 0.5:
            tokens.append(ReflectionToken.USEFUL.value)
        else:
            tokens.append(ReflectionToken.NOT_USEFUL.value)

        return tokens

# Self-RAG 학습을 위한 데이터 생성
class SelfRAGDataGenerator:
    def __init__(self):
        """Self-RAG 학습 데이터 생성기"""
        self.critique_model = None  # GPT-4 등 사용

    def generate_training_data(self, qa_pairs: List[Dict[str, str]],
                             documents: List[str]) -> List[Dict[str, Any]]:
        """학습 데이터 생성"""
        training_data = []

        for qa in qa_pairs:
            query = qa['question']
            answer = qa['answer']

            # 1. 검색 필요성 레이블
            retrieve_needed = self._label_retrieval_need(query, answer)

            # 2. 관련 문서 샘플링
            if retrieve_needed:
                sampled_docs = self._sample_documents(query, documents)

                # 3. 관련성 레이블링
                relevance_labels = []
                for doc in sampled_docs:
                    relevance = self._label_relevance(query, doc)
                    relevance_labels.append(relevance)

                # 4. 지원도 레이블링
                support_labels = self._label_support(answer, sampled_docs)
            else:
                sampled_docs = []
                relevance_labels = []
                support_labels = []

            # 5. 유용성 레이블
            utility_label = self._label_utility(query, answer)

            training_data.append({
                'query': query,
                'answer': answer,
                'retrieve_needed': retrieve_needed,
                'documents': sampled_docs,
                'relevance_labels': relevance_labels,
                'support_labels': support_labels,
                'utility_label': utility_label
            })

        return training_data

    def _label_retrieval_need(self, query: str, answer: str) -> bool:
        """검색 필요성 레이블링"""
        # 휴리스틱: 팩트 기반 질문은 검색 필요
        fact_keywords = ['when', 'where', 'who', 'how many', 'what year']
        return any(kw in query.lower() for kw in fact_keywords)

    def _sample_documents(self, query: str, documents: List[str], k: int = 5) -> List[str]:
        """문서 샘플링 (BM25 등 사용)"""
        # 간단한 키워드 매칭
        query_words = set(query.lower().split())
        scored_docs = []

        for doc in documents:
            doc_words = set(doc.lower().split())
            score = len(query_words.intersection(doc_words))
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

    def _label_relevance(self, query: str, document: str) -> float:
        """관련성 레이블링"""
        # 실제로는 인간 평가 또는 GPT-4 사용
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        overlap = len(query_words.intersection(doc_words))
        return min(overlap / len(query_words), 1.0) if query_words else 0.0

    def _label_support(self, answer: str, documents: List[str]) -> List[float]:
        """지원도 레이블링"""
        support_scores = []
        answer_words = set(answer.lower().split())

        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(answer_words.intersection(doc_words))
            score = min(overlap / len(answer_words), 1.0) if answer_words else 0.0
            support_scores.append(score)

        return support_scores

    def _label_utility(self, query: str, answer: str) -> float:
        """유용성 레이블링"""
        # 간단한 휴리스틱: 답변 길이와 질문 단어 포함도
        if len(answer.split()) < 5:
            return 0.3

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        coverage = len(query_words.intersection(answer_words)) / len(query_words)

        return min(0.5 + coverage * 0.5, 1.0)

# 사용 예제
print("=== Self-RAG 데모 ===\n")

# Self-RAG 모델 초기화
self_rag = SelfRAG()

# 테스트 쿼리
queries = [
    "What is the capital of France?",  # 팩트 질문 - 검색 필요
    "Hello, how are you?",  # 일반 대화 - 검색 불필요
    "Explain the theory of relativity",  # 복잡한 설명 - 검색 도움
]

for query in queries:
    print(f"\nQuery: {query}")

    # Self-RAG 실행
    output = self_rag(query)

    print(f"Retrieve Decision: {output.retrieve_decision}")
    if output.retrieve_decision:
        print(f"Retrieved {len(output.retrieved_docs)} documents")
        print(f"Relevance Scores: {[f'{s:.2f}' for s in output.relevance_scores]}")
    print(f"Answer: {output.answer}")
    print(f"Utility Score: {output.utility_score:.2f}")
    print(f"Reflection Tokens: {' '.join(output.reflection_tokens[:5])}...")

# 학습 데이터 생성 예제
data_generator = SelfRAGDataGenerator()

qa_pairs = [
    {"question": "What is machine learning?",
     "answer": "Machine learning is a subset of AI that enables systems to learn from data."},
    {"question": "How's the weather?",
     "answer": "I don't have access to real-time weather data."}
]

documents = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "AI and machine learning are transforming industries.",
    "The weather forecast requires current atmospheric data."
]

training_data = data_generator.generate_training_data(qa_pairs, documents)

print("\n\n=== Generated Training Data ===")
for i, data in enumerate(training_data):
    print(f"\nExample {i+1}:")
    print(f"Query: {data['query']}")
    print(f"Retrieve Needed: {data['retrieve_needed']}")
    print(f"Utility Label: {data['utility_label']:.2f}")`}
            </pre>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">Self-RAG 성능 벤치마크</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">Open-domain QA 성능</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Self-RAG (7B)</span>
                  <span className="font-bold text-green-600">54.9</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>ChatGPT</span>
                  <span>44.0</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Llama2-chat (7B)</span>
                  <span>28.2</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Alpaca (7B)</span>
                  <span>24.5</span>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">효율성 개선</h4>
              <div className="space-y-2">
                <div className="bg-blue-50 dark:bg-blue-900/30 p-2 rounded">
                  <p className="text-sm"><strong>검색 횟수:</strong> 60% 감소</p>
                </div>
                <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                  <p className="text-sm"><strong>정확도:</strong> 15% 향상</p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/30 p-2 rounded">
                  <p className="text-sm"><strong>환각률:</strong> 70% 감소</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
