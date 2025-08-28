'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Sparkles, Brain, Rocket, TrendingUp, Layers, Eye } from 'lucide-react'

export default function Chapter6Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/advanced"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          고급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Sparkles size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 6: RAG의 최신 연구 동향</h1>
              <p className="text-violet-100 text-lg">2024년 최신 논문과 미래 기술 전망</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Self-RAG - 자기 성찰하는 RAG */}
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

        {/* Section 2: RAPTOR - 계층적 요약 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
              <Layers className="text-orange-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.2 RAPTOR: 재귀적 트리 구조의 검색</h2>
              <p className="text-gray-600 dark:text-gray-400">Stanford의 계층적 문서 구조화 연구 (2024.01)</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
              <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">RAPTOR의 혁신: 재귀적 요약 트리</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>RAPTOR(Recursive Abstractive Processing for Tree-Organized Retrieval)는
                  문서를 계층적으로 요약하여 다양한 추상화 수준에서 검색을 가능하게 합니다.</strong>
                  이는 긴 문서나 복잡한 주제에 대한 질문에 특히 효과적이며, 전체적인 맥락과
                  세부 정보를 모두 포착할 수 있습니다.
                </p>
              </div>

              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import torch
from transformers import AutoModel, AutoTokenizer

@dataclass
class RAPTORNode:
    """RAPTOR 트리의 노드"""
    level: int
    content: str
    summary: str
    children: List['RAPTORNode']
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None

class RAPTOR:
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",
                 summarization_model: str = "facebook/bart-large-cnn",
                 max_cluster_size: int = 10):
        """
        RAPTOR: 재귀적 트리 기반 검색
        - 계층적 문서 구조화
        - 다중 수준 요약
        - 적응적 검색
        """
        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.summarizer = self._init_summarizer(summarization_model)
        self.max_cluster_size = max_cluster_size
        self.tree_root = None
        
    def build_tree(self, documents: List[str]) -> RAPTORNode:
        """문서 집합으로부터 RAPTOR 트리 구축"""
        print("Building RAPTOR tree...")
        
        # 1단계: 리프 노드 생성 (원본 문서)
        leaf_nodes = []
        for i, doc in enumerate(documents):
            node = RAPTORNode(
                level=0,
                content=doc,
                summary=doc[:200] + "...",  # 초기에는 앞부분만
                children=[],
                embedding=self._get_embedding(doc)
            )
            leaf_nodes.append(node)
        
        # 2단계: 재귀적으로 상위 레벨 구축
        current_level_nodes = leaf_nodes
        level = 0
        
        while len(current_level_nodes) > 1:
            level += 1
            print(f"Building level {level} with {len(current_level_nodes)} nodes")
            
            # 클러스터링
            clusters = self._cluster_nodes(current_level_nodes)
            
            # 각 클러스터에 대해 부모 노드 생성
            parent_nodes = []
            for cluster_id, nodes in clusters.items():
                parent_node = self._create_parent_node(nodes, level)
                parent_nodes.append(parent_node)
            
            current_level_nodes = parent_nodes
        
        # 루트 노드
        self.tree_root = current_level_nodes[0] if current_level_nodes else None
        return self.tree_root
    
    def _cluster_nodes(self, nodes: List[RAPTORNode]) -> Dict[int, List[RAPTORNode]]:
        """노드 클러스터링"""
        if len(nodes) <= self.max_cluster_size:
            return {0: nodes}
        
        # 임베딩 추출
        embeddings = np.array([node.embedding for node in nodes])
        
        # 최적 클러스터 수 결정
        n_clusters = max(2, len(nodes) // self.max_cluster_size)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 클러스터별로 그룹화
        clusters = defaultdict(list)
        for node, label in zip(nodes, cluster_labels):
            node.cluster_id = label
            clusters[label].append(node)
        
        return clusters
    
    def _create_parent_node(self, children: List[RAPTORNode], level: int) -> RAPTORNode:
        """자식 노드들로부터 부모 노드 생성"""
        # 자식들의 내용 결합
        combined_content = "\n\n".join([child.summary for child in children])
        
        # 요약 생성
        summary = self._generate_summary(combined_content)
        
        # 부모 노드 생성
        parent = RAPTORNode(
            level=level,
            content=combined_content,
            summary=summary,
            children=children,
            embedding=self._get_embedding(summary)
        )
        
        return parent
    
    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """텍스트 요약 생성"""
        # 실제로는 BART 등의 요약 모델 사용
        # 여기서는 간단한 추출적 요약
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
        
        # TF-IDF 기반 중요 문장 선택 (시뮬레이션)
        important_sentences = sentences[:3]  # 처음 3문장
        return '. '.join(important_sentences) + '.'
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """텍스트 임베딩 생성"""
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()[0]
    
    def search(self, query: str, top_k: int = 5, 
              collapse_threshold: float = 0.5) -> List[Tuple[RAPTORNode, float]]:
        """
        RAPTOR 검색
        - 트리의 모든 레벨에서 검색
        - 관련성에 따라 노드 확장/축소
        """
        if not self.tree_root:
            return []
        
        query_embedding = self._get_embedding(query)
        
        # 모든 노드와 점수 계산
        all_nodes_scores = []
        self._collect_relevant_nodes(
            self.tree_root, query_embedding, 
            all_nodes_scores, collapse_threshold
        )
        
        # 점수순 정렬
        all_nodes_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 중복 제거 (자식이 선택되면 부모는 제외)
        selected_nodes = []
        selected_contents = set()
        
        for node, score in all_nodes_scores:
            # 이미 선택된 내용과 중복되지 않는지 확인
            if node.content not in selected_contents:
                selected_nodes.append((node, score))
                selected_contents.add(node.content)
                
                # 하위 노드들의 내용도 추가 (중복 방지)
                self._add_descendant_contents(node, selected_contents)
            
            if len(selected_nodes) >= top_k:
                break
        
        return selected_nodes
    
    def _collect_relevant_nodes(self, node: RAPTORNode, query_embedding: np.ndarray,
                               results: List[Tuple[RAPTORNode, float]], 
                               threshold: float):
        """관련 노드 수집 (재귀적)"""
        # 현재 노드와의 유사도 계산
        similarity = self._cosine_similarity(query_embedding, node.embedding)
        
        # 임계값 이상이면 결과에 추가
        if similarity >= threshold:
            results.append((node, similarity))
            
            # 높은 관련성이면 자식 노드도 탐색
            if similarity >= threshold + 0.2:  # 더 높은 임계값
                for child in node.children:
                    self._collect_relevant_nodes(
                        child, query_embedding, results, threshold
                    )
        else:
            # 낮은 관련성이어도 자식 중 일부는 관련될 수 있음
            # 샘플링하여 탐색
            if node.children and np.random.random() < 0.3:
                sample_size = min(3, len(node.children))
                sampled_children = np.random.choice(
                    node.children, size=sample_size, replace=False
                )
                for child in sampled_children:
                    self._collect_relevant_nodes(
                        child, query_embedding, results, threshold
                    )
    
    def _add_descendant_contents(self, node: RAPTORNode, contents_set: set):
        """노드의 모든 하위 내용 추가"""
        for child in node.children:
            contents_set.add(child.content)
            self._add_descendant_contents(child, contents_set)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _init_summarizer(self, model_name: str):
        """요약 모델 초기화"""
        # 실제로는 transformers pipeline 사용
        from transformers import pipeline
        return pipeline("summarization", model=model_name)
    
    def visualize_tree(self, max_depth: int = 3) -> str:
        """트리 구조 시각화"""
        if not self.tree_root:
            return "Tree not built"
        
        lines = []
        self._visualize_node(self.tree_root, lines, "", True, max_depth)
        return "\n".join(lines)
    
    def _visualize_node(self, node: RAPTORNode, lines: List[str], 
                       prefix: str, is_last: bool, max_depth: int):
        """노드 시각화 (재귀적)"""
        if max_depth <= 0:
            return
        
        # 현재 노드 출력
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}Level {node.level}: {node.summary[:50]}...")
        
        # 자식 노드들
        if node.children:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self._visualize_node(
                    child, lines, prefix + extension, 
                    is_last_child, max_depth - 1
                )

# RAPTOR 개선: 동적 트리 업데이트
class DynamicRAPTOR(RAPTOR):
    def __init__(self, *args, **kwargs):
        """동적 업데이트가 가능한 RAPTOR"""
        super().__init__(*args, **kwargs)
        self.update_threshold = 0.3  # 재구조화 임계값
        
    def add_documents(self, new_documents: List[str]):
        """새로운 문서 추가"""
        # 새 문서들을 리프 노드로 추가
        new_nodes = []
        for doc in new_documents:
            node = RAPTORNode(
                level=0,
                content=doc,
                summary=doc[:200] + "...",
                children=[],
                embedding=self._get_embedding(doc)
            )
            new_nodes.append(node)
        
        # 기존 트리와 병합
        self._merge_nodes(new_nodes)
    
    def _merge_nodes(self, new_nodes: List[RAPTORNode]):
        """새 노드들을 기존 트리에 병합"""
        # 가장 유사한 기존 클러스터 찾기
        for node in new_nodes:
            best_cluster = self._find_best_cluster(node)
            
            if best_cluster:
                # 기존 클러스터에 추가
                self._add_to_cluster(node, best_cluster)
            else:
                # 새 클러스터 생성
                self._create_new_cluster(node)
        
        # 트리 재균형화 체크
        if self._needs_rebalancing():
            self._rebalance_tree()
    
    def _find_best_cluster(self, node: RAPTORNode) -> Optional[RAPTORNode]:
        """가장 적합한 클러스터 찾기"""
        # 레벨 0의 모든 부모 노드들과 비교
        # 실제 구현은 더 복잡함
        return None
    
    def _needs_rebalancing(self) -> bool:
        """트리 재균형화 필요 여부 확인"""
        # 클러스터 크기 불균형 체크
        return False
    
    def _rebalance_tree(self):
        """트리 재균형화"""
        print("Rebalancing RAPTOR tree...")
        # 전체 트리 재구축 또는 부분 재구조화

# 사용 예제
print("=== RAPTOR 데모 ===\n")

# 샘플 문서
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Transfer learning reuses knowledge from one task for another.",
    "Supervised learning requires labeled training data.",
    "Unsupervised learning finds patterns without labels.",
    "Semi-supervised learning uses both labeled and unlabeled data.",
    "Active learning selects the most informative samples for labeling."
]

# RAPTOR 트리 구축
raptor = RAPTOR(max_cluster_size=3)
root = raptor.build_tree(documents)

# 트리 시각화
print("RAPTOR Tree Structure:")
print(raptor.visualize_tree(max_depth=3))

# 검색 테스트
queries = [
    "What are the types of machine learning?",
    "How does deep learning work?",
    "Explain learning without labels"
]

print("\n\n=== RAPTOR Search Results ===")
for query in queries:
    print(f"\nQuery: {query}")
    results = raptor.search(query, top_k=3)
    
    for i, (node, score) in enumerate(results, 1):
        print(f"\n{i}. Level {node.level} (Score: {score:.3f})")
        print(f"   Summary: {node.summary[:100]}...")
        if node.level == 0:
            print(f"   Type: Leaf node (original document)")
        else:
            print(f"   Type: Internal node (summary of {len(node.children)} children)")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Multimodal RAG */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
              <Eye className="text-green-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.3 Multimodal RAG: 텍스트를 넘어서</h2>
              <p className="text-gray-600 dark:text-gray-400">이미지, 비디오, 오디오를 통합한 차세대 검색</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">멀티모달 RAG의 최신 동향</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-3">🖼️ Visual RAG</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• CLIP 기반 이미지-텍스트 검색</li>
                    <li>• LayoutLM을 활용한 문서 이해</li>
                    <li>• Scene Graph 기반 추론</li>
                    <li>• OCR + RAG 통합</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-3">🎥 Video RAG</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 시간적 정보 인덱싱</li>
                    <li>• 키프레임 추출 및 검색</li>
                    <li>• 비디오 요약과 QA</li>
                    <li>• 실시간 스트리밍 RAG</li>
                  </ul>
                </div>
              </div>

              <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">최근 연구 하이라이트</h4>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                  <li><strong>• MM-RAG (Meta, 2024):</strong> 30B 파라미터 멀티모달 RAG, 이미지와 텍스트 동시 검색</li>
                  <li><strong>• VideoChat-RAG (2024):</strong> 비디오 대화를 위한 시간 인식 RAG</li>
                  <li><strong>• AudioRAG (Google, 2024):</strong> 음성/음악 검색과 생성 통합</li>
                </ul>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">통합 멀티모달 RAG 아키텍처</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`class UnifiedMultimodalRAG:
    """통합 멀티모달 RAG 시스템"""
    
    def __init__(self):
        # 모달리티별 인코더
        self.text_encoder = AutoModel.from_pretrained("bert-base")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2")
        
        # 통합 프로젝션 레이어
        self.projection = nn.Linear(768, 512)  # 공통 임베딩 공간
        
        # 크로스모달 어텐션
        self.cross_attention = nn.MultiheadAttention(512, 8)
        
    def encode_multimodal_query(self, query: Dict[str, Any]) -> torch.Tensor:
        """멀티모달 쿼리 인코딩"""
        embeddings = []
        
        if 'text' in query:
            text_emb = self.text_encoder(query['text'])
            embeddings.append(self.projection(text_emb))
            
        if 'image' in query:
            image_emb = self.image_encoder.get_image_features(query['image'])
            embeddings.append(self.projection(image_emb))
            
        if 'audio' in query:
            audio_emb = self.audio_encoder(query['audio']).last_hidden_state
            embeddings.append(self.projection(audio_emb.mean(dim=1)))
        
        # 크로스모달 융합
        if len(embeddings) > 1:
            fused = torch.stack(embeddings)
            attended, _ = self.cross_attention(fused, fused, fused)
            return attended.mean(dim=0)
        else:
            return embeddings[0]
    
    def retrieve_and_generate(self, query: Dict[str, Any]) -> str:
        """멀티모달 검색 및 생성"""
        # 1. 멀티모달 쿼리 인코딩
        query_embedding = self.encode_multimodal_query(query)
        
        # 2. 크로스모달 검색
        retrieved_items = self.cross_modal_search(query_embedding)
        
        # 3. 멀티모달 컨텍스트 구성
        context = self.build_multimodal_context(retrieved_items)
        
        # 4. 멀티모달 생성
        response = self.generate_with_multimodal_context(query, context)
        
        return response`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Future Directions */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Rocket className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.4 RAG의 미래: 2025년과 그 이후</h2>
              <p className="text-gray-600 dark:text-gray-400">차세대 RAG 기술의 발전 방향</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">🚀 2025년 RAG 기술 전망</h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-purple-600 dark:text-purple-400 mb-2">1. Agentic RAG</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    RAG 시스템이 단순 검색을 넘어 능동적으로 정보를 수집, 검증, 업데이트하는 
                    자율 에이전트로 진화. 필요시 외부 API 호출, 실시간 데이터 수집, 
                    정보 신뢰도 자동 평가 등을 수행.
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-2">2. Continual Learning RAG</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    사용자 피드백과 새로운 정보를 실시간으로 학습하여 지속적으로 개선되는 RAG. 
                    Catastrophic forgetting 없이 새로운 지식을 통합하고, 
                    오래된 정보를 자동으로 업데이트.
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">3. Personalized RAG</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    개인의 선호도, 전문성 수준, 문맥을 이해하여 맞춤형 정보를 제공하는 RAG. 
                    프라이버시를 보장하면서도 개인화된 지식 그래프를 구축하고 활용.
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-orange-600 dark:text-orange-400 mb-2">4. Quantum RAG</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    양자 컴퓨팅을 활용한 초고속 벡터 검색과 양자 중첩을 이용한 
                    다차원 의미 공간 탐색. 기존 RAG 대비 1000배 이상의 검색 속도 향상 예상.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">🎯 연구자를 위한 오픈 문제들</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">이론적 도전과제</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• RAG의 이론적 한계 증명</li>
                    <li>• 최적 검색-생성 균형점</li>
                    <li>• 정보 이론적 관점의 RAG</li>
                    <li>• 환각 현상의 수학적 모델링</li>
                  </ul>
                </div>
                
                <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">실용적 도전과제</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 실시간 지식 업데이트</li>
                    <li>• 다국어 크로스링구얼 RAG</li>
                    <li>• 에너지 효율적 RAG</li>
                    <li>• 엣지 디바이스용 경량 RAG</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Research Papers and Resources */}
        <section className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">추천 논문 및 리소스</h2>
          
          <div className="space-y-4">
            <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
              <h3 className="font-bold mb-4">📚 필독 논문 (2023-2024)</h3>
              
              <div className="space-y-3">
                <div className="bg-white/10 p-4 rounded-lg">
                  <h4 className="font-medium mb-1">Self-RAG: Learning to Retrieve, Generate, and Critique</h4>
                  <p className="text-sm opacity-90">Asai et al., 2023 - Washington University</p>
                  <a href="#" className="text-xs underline">arXiv:2310.11511</a>
                </div>
                
                <div className="bg-white/10 p-4 rounded-lg">
                  <h4 className="font-medium mb-1">RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval</h4>
                  <p className="text-sm opacity-90">Sarthi et al., 2024 - Stanford University</p>
                  <a href="#" className="text-xs underline">arXiv:2401.18059</a>
                </div>
                
                <div className="bg-white/10 p-4 rounded-lg">
                  <h4 className="font-medium mb-1">Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models</h4>
                  <p className="text-sm opacity-90">Jeong et al., 2024 - KAIST</p>
                  <a href="#" className="text-xs underline">arXiv:2403.14403</a>
                </div>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
              <h3 className="font-bold mb-4">🛠️ 오픈소스 프로젝트</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white/10 p-3 rounded">
                  <h4 className="font-medium text-sm">LlamaIndex</h4>
                  <p className="text-xs opacity-90">최신 RAG 기법 구현체</p>
                </div>
                <div className="bg-white/10 p-3 rounded">
                  <h4 className="font-medium text-sm">LangChain</h4>
                  <p className="text-xs opacity-90">프로덕션 RAG 파이프라인</p>
                </div>
                <div className="bg-white/10 p-3 rounded">
                  <h4 className="font-medium text-sm">RAGAS</h4>
                  <p className="text-xs opacity-90">RAG 평가 프레임워크</p>
                </div>
                <div className="bg-white/10 p-3 rounded">
                  <h4 className="font-medium text-sm">Haystack</h4>
                  <p className="text-xs opacity-90">엔터프라이즈 RAG 솔루션</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter5"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: RAG 평가와 모니터링
          </Link>
          
          <Link
            href="/modules/rag/advanced"
            className="inline-flex items-center gap-2 bg-violet-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-violet-600 transition-colors"
          >
            고급 과정 완료
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}