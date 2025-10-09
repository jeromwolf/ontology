'use client'

import { Gauge } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-emerald-100 dark:bg-emerald-900/20 flex items-center justify-center">
          <Gauge className="text-emerald-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.1 RAG 시스템 평가 메트릭 체계</h2>
          <p className="text-gray-600 dark:text-gray-400">검색과 생성을 모두 고려한 종합적 평가 프레임워크</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-emerald-50 dark:bg-emerald-900/20 p-6 rounded-xl border border-emerald-200 dark:border-emerald-700">
          <h3 className="font-bold text-emerald-800 dark:text-emerald-200 mb-4">RAG 평가의 3차원 접근법</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>RAG 시스템은 검색(Retrieval)과 생성(Generation) 두 가지 핵심 컴포넌트로 구성되며,
              각각을 독립적으로 평가하는 동시에 전체 시스템의 성능도 측정해야 합니다.</strong>
              Microsoft, Google, OpenAI의 실제 프로덕션 경험을 바탕으로 한 포괄적 평가 체계를 소개합니다.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-emerald-600 dark:text-emerald-400 mb-2">🔍 Retrieval Quality</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• Precision@K</li>
                <li>• Recall@K</li>
                <li>• MRR (Mean Reciprocal Rank)</li>
                <li>• NDCG (Normalized DCG)</li>
                <li>• Coverage & Diversity</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-2">📝 Generation Quality</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• Relevance Score</li>
                <li>• Faithfulness</li>
                <li>• Answer Correctness</li>
                <li>• Coherence & Fluency</li>
                <li>• Hallucination Rate</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-purple-600 dark:text-purple-400 mb-2">⚡ System Performance</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• End-to-end Latency</li>
                <li>• Throughput (QPS)</li>
                <li>• Resource Utilization</li>
                <li>• Cost per Query</li>
                <li>• Error Rate</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">RAGAS Framework 구현</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import ndcg_score
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import asyncio
from datetime import datetime
import json

@dataclass
class RAGEvaluationSample:
    """RAG 평가를 위한 데이터 샘플"""
    query: str
    true_answer: str
    retrieved_documents: List[Dict[str, Any]]
    generated_answer: str
    ground_truth_docs: List[str]  # 정답 문서 ID
    metadata: Dict[str, Any] = None

class ComprehensiveRAGEvaluator:
    def __init__(self,
                 embedding_model: str = 'intfloat/multilingual-e5-large',
                 cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 nli_model: str = 'microsoft/deberta-v3-base-nli'):
        """
        포괄적 RAG 평가 시스템
        - Retrieval, Generation, System 메트릭 통합
        - 다국어 지원
        - 실시간 모니터링 통합
        """
        # 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.nli_model = CrossEncoder(nli_model)

        # 평가 결과 저장
        self.evaluation_history = []

    def evaluate_sample(self, sample: RAGEvaluationSample) -> Dict[str, float]:
        """단일 샘플에 대한 전체 평가"""
        metrics = {}

        # 1. Retrieval 평가
        retrieval_metrics = self._evaluate_retrieval(sample)
        metrics.update(retrieval_metrics)

        # 2. Generation 평가
        generation_metrics = self._evaluate_generation(sample)
        metrics.update(generation_metrics)

        # 3. 종합 점수 계산
        metrics['overall_score'] = self._calculate_overall_score(metrics)

        return metrics

    def _evaluate_retrieval(self, sample: RAGEvaluationSample) -> Dict[str, float]:
        """검색 품질 평가"""
        retrieved_ids = [doc['id'] for doc in sample.retrieved_documents]
        ground_truth_ids = sample.ground_truth_docs

        # Precision@K
        k_values = [1, 3, 5, 10]
        precisions = {}
        for k in k_values:
            if len(retrieved_ids) >= k:
                relevant_at_k = sum(1 for doc_id in retrieved_ids[:k]
                                   if doc_id in ground_truth_ids)
                precisions[f'precision@{k}'] = relevant_at_k / k
            else:
                precisions[f'precision@{k}'] = 0.0

        # Recall@K
        recalls = {}
        for k in k_values:
            if len(ground_truth_ids) > 0:
                relevant_at_k = sum(1 for doc_id in retrieved_ids[:k]
                                   if doc_id in ground_truth_ids)
                recalls[f'recall@{k}'] = relevant_at_k / len(ground_truth_ids)
            else:
                recalls[f'recall@{k}'] = 0.0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_ids:
                mrr = 1.0 / (i + 1)
                break

        # NDCG
        relevance_scores = []
        for doc in sample.retrieved_documents:
            if doc['id'] in ground_truth_ids:
                relevance_scores.append(1.0)
            else:
                # Semantic similarity as soft relevance
                sim = self._calculate_semantic_similarity(
                    sample.query, doc['content']
                )
                relevance_scores.append(sim)

        ideal_scores = sorted(relevance_scores, reverse=True)
        if len(relevance_scores) > 0 and sum(ideal_scores) > 0:
            ndcg = ndcg_score([ideal_scores], [relevance_scores])
        else:
            ndcg = 0.0

        # Coverage (검색된 관련 문서의 비율)
        coverage = len(set(retrieved_ids) & set(ground_truth_ids)) / \
                  max(len(ground_truth_ids), 1)

        # Diversity (검색 결과의 다양성)
        diversity = self._calculate_diversity(sample.retrieved_documents)

        return {
            **precisions,
            **recalls,
            'mrr': mrr,
            'ndcg': ndcg,
            'coverage': coverage,
            'diversity': diversity
        }

    def _evaluate_generation(self, sample: RAGEvaluationSample) -> Dict[str, float]:
        """생성 품질 평가"""
        metrics = {}

        # 1. Answer Relevance (쿼리와 답변의 관련성)
        relevance = self._calculate_semantic_similarity(
            sample.query, sample.generated_answer
        )
        metrics['answer_relevance'] = relevance

        # 2. Faithfulness (검색된 문서에 대한 충실도)
        faithfulness = self._evaluate_faithfulness(
            sample.generated_answer,
            sample.retrieved_documents
        )
        metrics['faithfulness'] = faithfulness

        # 3. Answer Correctness (정답과의 일치도)
        correctness = self._evaluate_correctness(
            sample.generated_answer,
            sample.true_answer
        )
        metrics['answer_correctness'] = correctness

        # 4. Hallucination Detection
        hallucination_score = self._detect_hallucination(
            sample.generated_answer,
            sample.retrieved_documents
        )
        metrics['hallucination_score'] = hallucination_score

        # 5. Coherence and Fluency
        coherence = self._evaluate_coherence(sample.generated_answer)
        metrics['coherence'] = coherence

        return metrics

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산"""
        # 임베딩 생성
        inputs1 = self.tokenizer(text1, return_tensors='pt',
                                padding=True, truncation=True)
        inputs2 = self.tokenizer(text2, return_tensors='pt',
                                padding=True, truncation=True)

        with torch.no_grad():
            emb1 = self.embedding_model(**inputs1).last_hidden_state.mean(dim=1)
            emb2 = self.embedding_model(**inputs2).last_hidden_state.mean(dim=1)

        # Cosine similarity
        similarity = torch.cosine_similarity(emb1, emb2).item()
        return (similarity + 1) / 2  # Normalize to [0, 1]

    def _evaluate_faithfulness(self, answer: str, documents: List[Dict]) -> float:
        """답변이 검색된 문서에 충실한지 평가"""
        # 답변을 문장으로 분리
        answer_sentences = answer.split('. ')

        # 각 문장이 문서에서 지원되는지 확인
        supported_count = 0
        total_sentences = len(answer_sentences)

        for sentence in answer_sentences:
            if not sentence.strip():
                continue

            max_support = 0.0
            for doc in documents:
                # NLI 모델을 사용한 entailment 검사
                premise = doc['content']
                hypothesis = sentence

                # Cross-encoder로 entailment score 계산
                score = self.nli_model.predict([[premise, hypothesis]])[0]
                max_support = max(max_support, score)

            # Threshold 이상이면 지원됨
            if max_support > 0.5:
                supported_count += 1

        return supported_count / max(total_sentences, 1)

    def _detect_hallucination(self, answer: str, documents: List[Dict]) -> float:
        """환각(hallucination) 탐지"""
        # 답변에서 구체적인 사실/수치 추출 (간단한 규칙 기반)
        import re

        # 숫자, 날짜, 고유명사 패턴
        fact_patterns = [
            r'\d+(?:\.\d+)?%?',  # 숫자와 백분율
            r'\d{4}년',  # 연도
            r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*',  # 고유명사
        ]

        facts = []
        for pattern in fact_patterns:
            facts.extend(re.findall(pattern, answer))

        if not facts:
            return 0.0  # 구체적 사실이 없으면 환각 없음

        # 각 사실이 문서에서 언급되는지 확인
        hallucinated = 0
        for fact in facts:
            found = False
            for doc in documents:
                if fact in doc['content']:
                    found = True
                    break

            if not found:
                hallucinated += 1

        return hallucinated / len(facts) if facts else 0.0

    def _evaluate_correctness(self, generated: str, ground_truth: str) -> float:
        """정답과의 일치도 평가"""
        # 1. Semantic similarity
        semantic_score = self._calculate_semantic_similarity(generated, ground_truth)

        # 2. Token overlap (F1 score)
        generated_tokens = set(generated.lower().split())
        truth_tokens = set(ground_truth.lower().split())

        if not truth_tokens:
            return semantic_score

        overlap = generated_tokens.intersection(truth_tokens)
        precision = len(overlap) / len(generated_tokens) if generated_tokens else 0
        recall = len(overlap) / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 3. Combined score
        return 0.7 * semantic_score + 0.3 * f1

    def _evaluate_coherence(self, text: str) -> float:
        """텍스트 일관성 평가"""
        sentences = text.split('. ')
        if len(sentences) < 2:
            return 1.0

        # 연속된 문장 간 의미적 유사도
        coherence_scores = []
        for i in range(len(sentences) - 1):
            if sentences[i].strip() and sentences[i+1].strip():
                sim = self._calculate_semantic_similarity(
                    sentences[i], sentences[i+1]
                )
                coherence_scores.append(sim)

        return np.mean(coherence_scores) if coherence_scores else 1.0

    def _calculate_diversity(self, documents: List[Dict]) -> float:
        """문서 집합의 다양성 계산"""
        if len(documents) < 2:
            return 0.0

        # 모든 문서 쌍의 유사도 계산
        similarities = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                sim = self._calculate_semantic_similarity(
                    documents[i]['content'],
                    documents[j]['content']
                )
                similarities.append(sim)

        # 평균 유사도가 낮을수록 다양성이 높음
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity

        return diversity

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """종합 점수 계산"""
        # 가중치 설정
        weights = {
            'retrieval': 0.4,
            'generation': 0.5,
            'system': 0.1
        }

        # 카테고리별 점수 계산
        retrieval_score = np.mean([
            metrics.get('precision@10', 0),
            metrics.get('recall@10', 0),
            metrics.get('mrr', 0),
            metrics.get('ndcg', 0)
        ])

        generation_score = np.mean([
            metrics.get('answer_relevance', 0),
            metrics.get('faithfulness', 0),
            metrics.get('answer_correctness', 0),
            1 - metrics.get('hallucination_score', 0),  # 환각이 적을수록 좋음
            metrics.get('coherence', 0)
        ])

        # 종합 점수
        overall = (weights['retrieval'] * retrieval_score +
                  weights['generation'] * generation_score)

        return overall

    def evaluate_dataset(self, samples: List[RAGEvaluationSample],
                        batch_size: int = 32) -> Dict[str, Any]:
        """전체 데이터셋 평가"""
        all_metrics = []

        # 배치 처리
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]

            # 병렬 처리를 위한 비동기 실행
            batch_metrics = []
            for sample in batch:
                metrics = self.evaluate_sample(sample)
                batch_metrics.append(metrics)

            all_metrics.extend(batch_metrics)

        # 집계
        aggregated = self._aggregate_metrics(all_metrics)

        # 상세 분석
        analysis = self._analyze_results(all_metrics, samples)

        return {
            'aggregated_metrics': aggregated,
            'detailed_analysis': analysis,
            'sample_metrics': all_metrics
        }

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """메트릭 집계"""
        aggregated = {}

        # 모든 메트릭 수집
        all_metric_names = set()
        for metrics in metrics_list:
            all_metric_names.update(metrics.keys())

        # 각 메트릭에 대해 통계 계산
        for metric_name in all_metric_names:
            values = [m.get(metric_name, 0) for m in metrics_list]

            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'p95': np.percentile(values, 95)
            }

        return aggregated

    def _analyze_results(self, metrics_list: List[Dict[str, float]],
                        samples: List[RAGEvaluationSample]) -> Dict[str, Any]:
        """결과 상세 분석"""
        analysis = {}

        # 1. 성능 저하 샘플 식별
        overall_scores = [m['overall_score'] for m in metrics_list]
        threshold = np.percentile(overall_scores, 25)  # 하위 25%

        poor_performing = []
        for i, (sample, metrics) in enumerate(zip(samples, metrics_list)):
            if metrics['overall_score'] < threshold:
                poor_performing.append({
                    'index': i,
                    'query': sample.query,
                    'score': metrics['overall_score'],
                    'issues': self._identify_issues(metrics)
                })

        analysis['poor_performing_samples'] = poor_performing[:10]  # Top 10

        # 2. 카테고리별 강약점 분석
        retrieval_avg = np.mean([
            m.get('mrr', 0) for m in metrics_list
        ])
        generation_avg = np.mean([
            m.get('answer_correctness', 0) for m in metrics_list
        ])

        if retrieval_avg < 0.5:
            analysis['weaknesses'] = ['retrieval']
        if generation_avg < 0.7:
            analysis['weaknesses'] = analysis.get('weaknesses', []) + ['generation']

        # 3. 패턴 분석
        analysis['patterns'] = {
            'avg_hallucination_rate': np.mean([
                m.get('hallucination_score', 0) for m in metrics_list
            ]),
            'low_diversity_rate': sum(
                1 for m in metrics_list if m.get('diversity', 1) < 0.3
            ) / len(metrics_list)
        }

        return analysis

    def _identify_issues(self, metrics: Dict[str, float]) -> List[str]:
        """메트릭 기반 문제점 식별"""
        issues = []

        if metrics.get('precision@10', 1) < 0.3:
            issues.append('Low retrieval precision')
        if metrics.get('faithfulness', 1) < 0.5:
            issues.append('Low faithfulness to sources')
        if metrics.get('hallucination_score', 0) > 0.3:
            issues.append('High hallucination rate')
        if metrics.get('coherence', 1) < 0.7:
            issues.append('Poor answer coherence')

        return issues

# 실시간 모니터링 시스템
class RAGMonitoringSystem:
    def __init__(self, evaluator: ComprehensiveRAGEvaluator):
        """
        실시간 RAG 모니터링 시스템
        - 성능 메트릭 추적
        - 이상 탐지
        - 자동 알림
        """
        self.evaluator = evaluator
        self.metrics_buffer = []
        self.alert_thresholds = {
            'answer_correctness': 0.7,
            'hallucination_score': 0.2,
            'latency_p95': 1000,  # ms
            'error_rate': 0.01
        }
        self.alerts_triggered = []

    async def monitor_request(self, query: str, context: Dict[str, Any],
                            response: Dict[str, Any], latency: float):
        """실시간 요청 모니터링"""
        # RAG 평가 샘플 생성
        sample = RAGEvaluationSample(
            query=query,
            true_answer="",  # 실시간에서는 ground truth 없음
            retrieved_documents=response.get('documents', []),
            generated_answer=response.get('answer', ''),
            ground_truth_docs=[],
            metadata={'latency': latency}
        )

        # 평가 수행
        metrics = self.evaluator.evaluate_sample(sample)
        metrics['latency'] = latency
        metrics['timestamp'] = datetime.now()

        # 버퍼에 추가
        self.metrics_buffer.append(metrics)

        # 이상 탐지
        await self._check_anomalies(metrics)

        # 주기적 집계 (1000개마다)
        if len(self.metrics_buffer) >= 1000:
            await self._aggregate_and_report()

    async def _check_anomalies(self, metrics: Dict[str, float]):
        """이상 탐지 및 알림"""
        alerts = []

        # Threshold 체크
        if metrics.get('answer_correctness', 1) < self.alert_thresholds['answer_correctness']:
            alerts.append({
                'type': 'LOW_CORRECTNESS',
                'value': metrics['answer_correctness'],
                'threshold': self.alert_thresholds['answer_correctness']
            })

        if metrics.get('hallucination_score', 0) > self.alert_thresholds['hallucination_score']:
            alerts.append({
                'type': 'HIGH_HALLUCINATION',
                'value': metrics['hallucination_score'],
                'threshold': self.alert_thresholds['hallucination_score']
            })

        if metrics.get('latency', 0) > self.alert_thresholds['latency_p95']:
            alerts.append({
                'type': 'HIGH_LATENCY',
                'value': metrics['latency'],
                'threshold': self.alert_thresholds['latency_p95']
            })

        # 알림 발송
        for alert in alerts:
            await self._send_alert(alert)

    async def _send_alert(self, alert: Dict[str, Any]):
        """알림 발송"""
        self.alerts_triggered.append({
            'alert': alert,
            'timestamp': datetime.now()
        })

        # 실제 환경에서는 Slack, Email 등으로 발송
        print(f"⚠️ ALERT: {alert['type']} - Value: {alert['value']:.3f} "
              f"(Threshold: {alert['threshold']})")

    async def _aggregate_and_report(self):
        """주기적 집계 및 리포팅"""
        if not self.metrics_buffer:
            return

        # 시간대별 집계
        aggregated = self.evaluator._aggregate_metrics(self.metrics_buffer)

        # 리포트 생성
        report = {
            'period': {
                'start': self.metrics_buffer[0]['timestamp'],
                'end': self.metrics_buffer[-1]['timestamp']
            },
            'sample_count': len(self.metrics_buffer),
            'metrics': aggregated,
            'alerts_count': len(self.alerts_triggered),
            'health_score': self._calculate_health_score(aggregated)
        }

        # 저장 및 전송
        await self._save_report(report)

        # 버퍼 초기화
        self.metrics_buffer = []

    def _calculate_health_score(self, aggregated: Dict[str, Dict[str, float]]) -> float:
        """시스템 건강도 점수 계산"""
        score = 100.0

        # 각 메트릭별 감점
        if aggregated.get('answer_correctness', {}).get('mean', 1) < 0.8:
            score -= 20
        if aggregated.get('hallucination_score', {}).get('mean', 0) > 0.1:
            score -= 15
        if aggregated.get('latency', {}).get('p95', 0) > 800:
            score -= 10

        return max(0, score)

    async def _save_report(self, report: Dict[str, Any]):
        """리포트 저장"""
        filename = f"rag_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 실제로는 S3, BigQuery 등에 저장
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

# 사용 예제
print("=== RAG 평가 시스템 데모 ===\n")

# 평가 시스템 초기화
evaluator = ComprehensiveRAGEvaluator()

# 샘플 데이터
sample = RAGEvaluationSample(
    query="RAG 시스템의 핵심 구성요소는 무엇인가요?",
    true_answer="RAG 시스템의 핵심 구성요소는 검색기(Retriever)와 생성기(Generator)입니다.",
    retrieved_documents=[
        {
            'id': 'doc1',
            'content': 'RAG는 Retrieval-Augmented Generation의 약자로, 검색기와 생성기로 구성됩니다.'
        },
        {
            'id': 'doc2',
            'content': 'RAG 시스템은 외부 지식을 활용하여 더 정확한 답변을 생성합니다.'
        }
    ],
    generated_answer="RAG 시스템의 핵심 구성요소는 검색기(Retriever)와 생성기(Generator)입니다. "
                    "검색기는 관련 문서를 찾고, 생성기는 이를 바탕으로 답변을 생성합니다.",
    ground_truth_docs=['doc1']
)

# 평가 수행
metrics = evaluator.evaluate_sample(sample)

print("=== 평가 결과 ===")
print(f"Overall Score: {metrics['overall_score']:.3f}")
print(f"\nRetrieval Metrics:")
print(f"  - Precision@10: {metrics.get('precision@10', 0):.3f}")
print(f"  - MRR: {metrics['mrr']:.3f}")
print(f"  - NDCG: {metrics['ndcg']:.3f}")
print(f"\nGeneration Metrics:")
print(f"  - Answer Relevance: {metrics['answer_relevance']:.3f}")
print(f"  - Faithfulness: {metrics['faithfulness']:.3f}")
print(f"  - Hallucination Score: {metrics['hallucination_score']:.3f}")
print(f"  - Coherence: {metrics['coherence']:.3f}")`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
