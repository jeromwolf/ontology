'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, BarChart3, Activity, AlertTriangle, TrendingUp, Shield, Gauge } from 'lucide-react'

export default function Chapter5Page() {
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
        
        <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <BarChart3 size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 5: RAG 평가와 모니터링</h1>
              <p className="text-emerald-100 text-lg">프로덕션 RAG 시스템의 품질 측정과 지속적 개선</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: RAG 평가 메트릭 체계 */}
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

        {/* Section 2: A/B 테스팅 프레임워크 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
              <TrendingUp className="text-blue-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.2 RAG를 위한 A/B 테스팅</h2>
              <p className="text-gray-600 dark:text-gray-400">데이터 기반 의사결정을 위한 실험 프레임워크</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">프로덕션 A/B 테스팅 시스템</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`import hashlib
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

class ExperimentVariant(Enum):
    """실험 변형"""
    CONTROL = "control"
    TREATMENT_A = "treatment_a"
    TREATMENT_B = "treatment_b"

@dataclass
class ExperimentConfig:
    """실험 설정"""
    experiment_id: str
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    traffic_allocation: Dict[ExperimentVariant, float]
    success_metrics: List[str]
    guardrail_metrics: List[str]
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95

@dataclass
class ExperimentResult:
    """실험 결과"""
    variant: ExperimentVariant
    user_id: str
    metrics: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class RAGExperimentationFramework:
    def __init__(self):
        """
        RAG A/B 테스팅 프레임워크
        - 트래픽 분배
        - 메트릭 수집
        - 통계적 유의성 검정
        - 실시간 모니터링
        """
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self.variant_configs = self._init_variant_configs()
        
    def _init_variant_configs(self) -> Dict[ExperimentVariant, Dict[str, Any]]:
        """각 변형의 RAG 설정"""
        return {
            ExperimentVariant.CONTROL: {
                'retrieval': {
                    'model': 'bge-large-en',
                    'top_k': 5,
                    'reranking': False
                },
                'generation': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 500
                }
            },
            ExperimentVariant.TREATMENT_A: {
                'retrieval': {
                    'model': 'e5-large-v2',
                    'top_k': 10,
                    'reranking': True,
                    'reranker': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
                },
                'generation': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 500
                }
            },
            ExperimentVariant.TREATMENT_B: {
                'retrieval': {
                    'model': 'bge-large-en',
                    'top_k': 5,
                    'reranking': True,
                    'reranker': 'colbert-v2'
                },
                'generation': {
                    'model': 'gpt-4',
                    'temperature': 0.3,
                    'max_tokens': 800
                }
            }
        }
    
    def create_experiment(self, config: ExperimentConfig):
        """새 실험 생성"""
        # 트래픽 할당 합이 1인지 확인
        total_allocation = sum(config.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        self.experiments[config.experiment_id] = config
        print(f"Created experiment: {config.name}")
    
    def get_variant(self, experiment_id: str, user_id: str) -> Optional[ExperimentVariant]:
        """사용자에게 할당할 실험 변형 결정"""
        if experiment_id not in self.experiments:
            return None
        
        config = self.experiments[experiment_id]
        
        # 실험 기간 체크
        now = datetime.now()
        if now < config.start_date or now > config.end_date:
            return ExperimentVariant.CONTROL
        
        # Deterministic assignment based on user_id
        hash_input = f"{experiment_id}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0
        
        # 트래픽 할당에 따라 변형 선택
        cumulative = 0.0
        for variant, allocation in config.traffic_allocation.items():
            cumulative += allocation
            if bucket < cumulative:
                return variant
        
        return ExperimentVariant.CONTROL
    
    async def run_experiment_request(self, experiment_id: str, user_id: str,
                                   query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """실험 요청 실행"""
        # 변형 할당
        variant = self.get_variant(experiment_id, user_id)
        if variant is None:
            variant = ExperimentVariant.CONTROL
        
        # 변형별 설정 가져오기
        variant_config = self.variant_configs[variant]
        
        # RAG 실행 (시뮬레이션)
        start_time = datetime.now()
        response = await self._execute_rag_variant(query, context, variant_config)
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        # 메트릭 수집
        metrics = {
            'latency': latency,
            'relevance_score': response.get('relevance_score', 0),
            'click_through': 0,  # 나중에 업데이트
            'dwell_time': 0,  # 나중에 업데이트
            'thumbs_up': 0,  # 나중에 업데이트
        }
        
        # 결과 기록
        result = ExperimentResult(
            variant=variant,
            user_id=user_id,
            metrics=metrics,
            timestamp=datetime.now(),
            metadata={
                'query': query,
                'response': response
            }
        )
        self.results[experiment_id].append(result)
        
        # 응답에 실험 정보 추가
        response['experiment'] = {
            'id': experiment_id,
            'variant': variant.value
        }
        
        return response
    
    async def _execute_rag_variant(self, query: str, context: Dict[str, Any],
                                  variant_config: Dict[str, Any]) -> Dict[str, Any]:
        """변형별 RAG 실행 (시뮬레이션)"""
        # 실제로는 각 설정에 따라 다른 RAG 파이프라인 실행
        
        # 시뮬레이션을 위한 가짜 응답
        base_score = 0.8
        if variant_config['retrieval']['reranking']:
            base_score += 0.05
        if variant_config['generation']['model'] == 'gpt-4':
            base_score += 0.1
        
        return {
            'answer': f"Sample answer using {variant_config['generation']['model']}",
            'documents': [{'id': f'doc{i}', 'score': 0.9-i*0.1} 
                         for i in range(variant_config['retrieval']['top_k'])],
            'relevance_score': min(base_score + np.random.normal(0, 0.05), 1.0),
            'config': variant_config
        }
    
    def update_metrics(self, experiment_id: str, user_id: str, 
                      metric_updates: Dict[str, float]):
        """사용자 행동 메트릭 업데이트"""
        # 해당 사용자의 최신 결과 찾기
        experiment_results = self.results.get(experiment_id, [])
        
        for result in reversed(experiment_results):
            if result.user_id == user_id:
                result.metrics.update(metric_updates)
                break
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """실험 결과 분석"""
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        config = self.experiments[experiment_id]
        results = self.results.get(experiment_id, [])
        
        if len(results) < config.minimum_sample_size:
            return {
                'status': 'insufficient_data',
                'current_sample_size': len(results),
                'required_sample_size': config.minimum_sample_size
            }
        
        # 변형별 결과 분리
        variant_results = defaultdict(list)
        for result in results:
            variant_results[result.variant].append(result)
        
        # 각 메트릭에 대한 분석
        analysis = {
            'experiment_id': experiment_id,
            'name': config.name,
            'sample_sizes': {v.value: len(results) 
                           for v, results in variant_results.items()},
            'metric_analysis': {}
        }
        
        # Success metrics 분석
        for metric in config.success_metrics:
            metric_analysis = self._analyze_metric(
                variant_results, metric, config.confidence_level
            )
            analysis['metric_analysis'][metric] = metric_analysis
        
        # Guardrail metrics 체크
        guardrail_violations = []
        for metric in config.guardrail_metrics:
            violation = self._check_guardrail(variant_results, metric)
            if violation:
                guardrail_violations.append(violation)
        
        analysis['guardrail_violations'] = guardrail_violations
        
        # 승자 결정
        analysis['winner'] = self._determine_winner(
            analysis['metric_analysis'], config.success_metrics
        )
        
        return analysis
    
    def _analyze_metric(self, variant_results: Dict[ExperimentVariant, List[ExperimentResult]],
                       metric_name: str, confidence_level: float) -> Dict[str, Any]:
        """개별 메트릭 분석"""
        metric_values = {}
        for variant, results in variant_results.items():
            values = [r.metrics.get(metric_name, 0) for r in results]
            metric_values[variant] = values
        
        control_values = metric_values.get(ExperimentVariant.CONTROL, [])
        if not control_values:
            return {'error': 'No control data'}
        
        analysis = {
            'means': {},
            'confidence_intervals': {},
            'lifts': {},
            'p_values': {}
        }
        
        # 각 변형에 대한 분석
        for variant, values in metric_values.items():
            if not values:
                continue
            
            # 기본 통계
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            
            analysis['means'][variant.value] = mean
            
            # 신뢰구간
            ci = stats.t.interval(confidence_level, n-1, 
                                 loc=mean, scale=std/np.sqrt(n))
            analysis['confidence_intervals'][variant.value] = ci
            
            # Control 대비 상승률
            if variant != ExperimentVariant.CONTROL:
                control_mean = np.mean(control_values)
                lift = (mean - control_mean) / control_mean * 100
                analysis['lifts'][variant.value] = lift
                
                # T-test
                t_stat, p_value = stats.ttest_ind(values, control_values)
                analysis['p_values'][variant.value] = p_value
        
        return analysis
    
    def _check_guardrail(self, variant_results: Dict[ExperimentVariant, List[ExperimentResult]],
                        metric_name: str) -> Optional[Dict[str, Any]]:
        """Guardrail 메트릭 체크"""
        control_values = [r.metrics.get(metric_name, 0) 
                         for r in variant_results.get(ExperimentVariant.CONTROL, [])]
        
        if not control_values:
            return None
        
        control_mean = np.mean(control_values)
        
        violations = []
        for variant, results in variant_results.items():
            if variant == ExperimentVariant.CONTROL:
                continue
            
            values = [r.metrics.get(metric_name, 0) for r in results]
            if not values:
                continue
            
            variant_mean = np.mean(values)
            
            # 5% 이상 성능 저하 체크
            degradation = (control_mean - variant_mean) / control_mean * 100
            if degradation > 5:
                violations.append({
                    'variant': variant.value,
                    'metric': metric_name,
                    'degradation': degradation
                })
        
        return violations[0] if violations else None
    
    def _determine_winner(self, metric_analysis: Dict[str, Dict[str, Any]],
                         success_metrics: List[str]) -> Optional[str]:
        """승자 결정"""
        scores = defaultdict(float)
        
        for metric in success_metrics:
            analysis = metric_analysis.get(metric, {})
            p_values = analysis.get('p_values', {})
            lifts = analysis.get('lifts', {})
            
            for variant in [ExperimentVariant.TREATMENT_A, ExperimentVariant.TREATMENT_B]:
                variant_name = variant.value
                
                # 통계적으로 유의하고 개선이 있는 경우
                if (variant_name in p_values and 
                    p_values[variant_name] < 0.05 and
                    variant_name in lifts and 
                    lifts[variant_name] > 0):
                    scores[variant_name] += lifts[variant_name]
        
        # 가장 높은 점수의 변형 선택
        if scores:
            winner = max(scores.items(), key=lambda x: x[1])
            return winner[0]
        
        return None
    
    def generate_report(self, experiment_id: str) -> str:
        """실험 리포트 생성"""
        analysis = self.analyze_experiment(experiment_id)
        
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        report = []
        report.append(f"# Experiment Report: {analysis.get('name', experiment_id)}")
        report.append(f"\n## Sample Sizes")
        for variant, size in analysis['sample_sizes'].items():
            report.append(f"- {variant}: {size}")
        
        report.append(f"\n## Metric Analysis")
        for metric, metric_analysis in analysis['metric_analysis'].items():
            report.append(f"\n### {metric}")
            
            # 평균값
            report.append("**Means:**")
            for variant, mean in metric_analysis['means'].items():
                report.append(f"- {variant}: {mean:.4f}")
            
            # 상승률
            if metric_analysis.get('lifts'):
                report.append("\n**Lifts vs Control:**")
                for variant, lift in metric_analysis['lifts'].items():
                    p_value = metric_analysis['p_values'].get(variant, 1)
                    sig = "✅" if p_value < 0.05 else "❌"
                    report.append(f"- {variant}: {lift:+.2f}% (p={p_value:.4f}) {sig}")
        
        # Guardrail violations
        if analysis.get('guardrail_violations'):
            report.append("\n## ⚠️ Guardrail Violations")
            for violation in analysis['guardrail_violations']:
                report.append(f"- {violation['variant']}: {violation['metric']} "
                            f"degraded by {violation['degradation']:.2f}%")
        
        # Winner
        if analysis.get('winner'):
            report.append(f"\n## 🏆 Winner: {analysis['winner']}")
        else:
            report.append("\n## No clear winner yet")
        
        return "\n".join(report)

# 실시간 실험 대시보드
class ExperimentDashboard:
    def __init__(self, framework: RAGExperimentationFramework):
        """실험 대시보드"""
        self.framework = framework
        
    def get_live_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """실시간 메트릭 조회"""
        results = self.framework.results.get(experiment_id, [])
        if not results:
            return {}
        
        # 최근 1시간 데이터
        cutoff = datetime.now() - timedelta(hours=1)
        recent_results = [r for r in results if r.timestamp > cutoff]
        
        # 변형별 실시간 메트릭
        metrics = defaultdict(lambda: defaultdict(list))
        for result in recent_results:
            for metric, value in result.metrics.items():
                metrics[result.variant.value][metric].append(value)
        
        # 집계
        live_metrics = {}
        for variant, variant_metrics in metrics.items():
            live_metrics[variant] = {}
            for metric, values in variant_metrics.items():
                live_metrics[variant][metric] = {
                    'current': values[-1] if values else 0,
                    'mean': np.mean(values) if values else 0,
                    'trend': 'up' if len(values) > 1 and values[-1] > values[-2] else 'down'
                }
        
        return live_metrics

# 사용 예제
print("=== RAG A/B 테스팅 데모 ===\n")

# 프레임워크 초기화
framework = RAGExperimentationFramework()

# 실험 생성
experiment_config = ExperimentConfig(
    experiment_id="rag_reranking_test",
    name="Reranking Algorithm Comparison",
    description="Compare different reranking strategies",
    start_date=datetime.now(),
    end_date=datetime.now() + timedelta(days=7),
    traffic_allocation={
        ExperimentVariant.CONTROL: 0.33,
        ExperimentVariant.TREATMENT_A: 0.33,
        ExperimentVariant.TREATMENT_B: 0.34
    },
    success_metrics=['relevance_score', 'click_through'],
    guardrail_metrics=['latency'],
    minimum_sample_size=100
)

framework.create_experiment(experiment_config)

# 시뮬레이션: 여러 사용자 요청 실행
async def simulate_requests():
    for i in range(150):
        user_id = f"user_{i}"
        query = f"Test query {i % 10}"
        
        response = await framework.run_experiment_request(
            "rag_reranking_test", user_id, query, {}
        )
        
        # 시뮬레이션: 사용자 행동
        if response.get('relevance_score', 0) > 0.85:
            framework.update_metrics(
                "rag_reranking_test", user_id,
                {'click_through': 1, 'dwell_time': np.random.randint(10, 60)}
            )

# 실행
asyncio.run(simulate_requests())

# 분석
print("\n" + framework.generate_report("rag_reranking_test"))`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: 실시간 모니터링 대시보드 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Activity className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.3 실시간 RAG 모니터링 대시보드</h2>
              <p className="text-gray-600 dark:text-gray-400">Grafana + Prometheus 기반 종합 모니터링</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">프로덕션 모니터링 시스템 구축</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import pandas as pd

# Prometheus 메트릭 정의
rag_requests_total = Counter(
    'rag_requests_total',
    'Total number of RAG requests',
    ['endpoint', 'status']
)

rag_latency_seconds = Histogram(
    'rag_latency_seconds',
    'RAG request latency in seconds',
    ['operation'],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
)

rag_document_retrieval = Histogram(
    'rag_document_retrieval_count',
    'Number of documents retrieved per request',
    buckets=[1, 5, 10, 20, 50, 100]
)

rag_relevance_score = Summary(
    'rag_relevance_score',
    'Relevance scores of RAG responses'
)

rag_active_users = Gauge(
    'rag_active_users',
    'Number of active users in the last 5 minutes'
)

rag_cache_hit_rate = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate percentage'
)

rag_model_latency = Histogram(
    'rag_model_latency_seconds',
    'Model inference latency',
    ['model_name', 'operation'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

@dataclass
class RAGMetrics:
    """RAG 요청 메트릭"""
    request_id: str
    timestamp: datetime
    query: str
    latency_total: float
    latency_retrieval: float
    latency_generation: float
    documents_retrieved: int
    relevance_score: float
    cache_hit: bool
    error: Optional[str] = None
    user_id: Optional[str] = None
    
class RAGMonitoringService:
    def __init__(self):
        """
        실시간 RAG 모니터링 서비스
        - Prometheus 메트릭 수집
        - 실시간 이상 탐지
        - 대시보드 데이터 제공
        """
        self.metrics_buffer: List[RAGMetrics] = []
        self.active_users: set = set()
        self.alert_rules = self._init_alert_rules()
        
    def _init_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """알림 규칙 초기화"""
        return {
            'high_latency': {
                'condition': lambda m: m.latency_total > 2.0,
                'severity': 'warning',
                'message': 'High latency detected: {latency_total:.2f}s'
            },
            'low_relevance': {
                'condition': lambda m: m.relevance_score < 0.7,
                'severity': 'warning',
                'message': 'Low relevance score: {relevance_score:.2f}'
            },
            'error_rate': {
                'condition': lambda metrics: self._calculate_error_rate(metrics) > 0.05,
                'severity': 'critical',
                'message': 'Error rate exceeds 5%'
            },
            'cache_miss': {
                'condition': lambda metrics: self._calculate_cache_hit_rate(metrics) < 0.3,
                'severity': 'info',
                'message': 'Cache hit rate below 30%'
            }
        }
    
    async def record_request(self, metrics: RAGMetrics):
        """요청 메트릭 기록"""
        # Prometheus 메트릭 업데이트
        rag_requests_total.labels(
            endpoint='rag_query',
            status='success' if not metrics.error else 'error'
        ).inc()
        
        rag_latency_seconds.labels(operation='total').observe(metrics.latency_total)
        rag_latency_seconds.labels(operation='retrieval').observe(metrics.latency_retrieval)
        rag_latency_seconds.labels(operation='generation').observe(metrics.latency_generation)
        
        rag_document_retrieval.observe(metrics.documents_retrieved)
        rag_relevance_score.observe(metrics.relevance_score)
        
        # 활성 사용자 추적
        if metrics.user_id:
            self.active_users.add(metrics.user_id)
        
        # 버퍼에 추가
        self.metrics_buffer.append(metrics)
        
        # 이상 탐지
        await self._check_alerts(metrics)
        
        # 주기적 정리 (5분 이상 된 데이터)
        cutoff = datetime.now() - timedelta(minutes=5)
        self.metrics_buffer = [m for m in self.metrics_buffer if m.timestamp > cutoff]
    
    async def _check_alerts(self, metrics: RAGMetrics):
        """알림 규칙 체크"""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule_name in ['error_rate', 'cache_miss']:
                    # 집계 기반 규칙
                    if rule['condition'](self.metrics_buffer):
                        await self._send_alert(rule_name, rule)
                else:
                    # 개별 메트릭 기반 규칙
                    if rule['condition'](metrics):
                        await self._send_alert(rule_name, rule, metrics)
            except Exception as e:
                print(f"Alert check error: {e}")
    
    async def _send_alert(self, rule_name: str, rule: Dict[str, Any], 
                         metrics: Optional[RAGMetrics] = None):
        """알림 발송"""
        message = rule['message']
        if metrics:
            message = message.format(**metrics.__dict__)
        
        alert = {
            'rule': rule_name,
            'severity': rule['severity'],
            'message': message,
            'timestamp': datetime.now()
        }
        
        # 실제로는 Slack, PagerDuty 등으로 발송
        print(f"🚨 ALERT [{rule['severity'].upper()}]: {message}")
        
        # Webhook 호출 (예제)
        if rule['severity'] == 'critical':
            await self._send_webhook(alert)
    
    async def _send_webhook(self, alert: Dict[str, Any]):
        """Webhook 발송"""
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        
        payload = {
            'text': f"RAG System Alert",
            'attachments': [{
                'color': 'danger',
                'fields': [
                    {'title': 'Rule', 'value': alert['rule'], 'short': True},
                    {'title': 'Severity', 'value': alert['severity'], 'short': True},
                    {'title': 'Message', 'value': alert['message']},
                    {'title': 'Time', 'value': alert['timestamp'].isoformat()}
                ]
            }]
        }
        
        # async with aiohttp.ClientSession() as session:
        #     await session.post(webhook_url, json=payload)
    
    def _calculate_error_rate(self, metrics: List[RAGMetrics]) -> float:
        """에러율 계산"""
        if not metrics:
            return 0.0
        
        error_count = sum(1 for m in metrics if m.error is not None)
        return error_count / len(metrics)
    
    def _calculate_cache_hit_rate(self, metrics: List[RAGMetrics]) -> float:
        """캐시 히트율 계산"""
        if not metrics:
            return 0.0
        
        cache_hits = sum(1 for m in metrics if m.cache_hit)
        return cache_hits / len(metrics)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드용 데이터 조회"""
        if not self.metrics_buffer:
            return self._empty_dashboard_data()
        
        # 데이터프레임 변환
        df = pd.DataFrame([m.__dict__ for m in self.metrics_buffer])
        
        # 현재 메트릭
        current_metrics = {
            'qps': len(df[df['timestamp'] > datetime.now() - timedelta(seconds=60)]),
            'avg_latency': df['latency_total'].mean(),
            'p95_latency': df['latency_total'].quantile(0.95),
            'avg_relevance': df['relevance_score'].mean(),
            'error_rate': self._calculate_error_rate(self.metrics_buffer),
            'cache_hit_rate': self._calculate_cache_hit_rate(self.metrics_buffer),
            'active_users': len(self.active_users),
            'total_requests': len(df)
        }
        
        # 시계열 데이터 (1분 단위)
        df['timestamp_minute'] = df['timestamp'].dt.floor('T')
        time_series = {
            'latency': df.groupby('timestamp_minute')['latency_total'].agg(['mean', 'p95']),
            'throughput': df.groupby('timestamp_minute').size(),
            'relevance': df.groupby('timestamp_minute')['relevance_score'].mean()
        }
        
        # Top 느린 쿼리
        slow_queries = df.nlargest(10, 'latency_total')[
            ['query', 'latency_total', 'documents_retrieved']
        ].to_dict('records')
        
        # 모델별 레이턴시
        model_latency = {
            'retrieval': {
                'mean': df['latency_retrieval'].mean(),
                'p95': df['latency_retrieval'].quantile(0.95)
            },
            'generation': {
                'mean': df['latency_generation'].mean(),
                'p95': df['latency_generation'].quantile(0.95)
            }
        }
        
        return {
            'current_metrics': current_metrics,
            'time_series': time_series,
            'slow_queries': slow_queries,
            'model_latency': model_latency,
            'last_updated': datetime.now()
        }
    
    def _empty_dashboard_data(self) -> Dict[str, Any]:
        """빈 대시보드 데이터"""
        return {
            'current_metrics': {
                'qps': 0,
                'avg_latency': 0,
                'p95_latency': 0,
                'avg_relevance': 0,
                'error_rate': 0,
                'cache_hit_rate': 0,
                'active_users': 0,
                'total_requests': 0
            },
            'time_series': {},
            'slow_queries': [],
            'model_latency': {},
            'last_updated': datetime.now()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """시스템 헬스 체크"""
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now()
        }
        
        # 개별 컴포넌트 체크
        checks = {
            'metrics_collection': self._check_metrics_collection(),
            'alert_system': self._check_alert_system(),
            'data_freshness': self._check_data_freshness()
        }
        
        health_status['checks'] = checks
        
        # 전체 상태 결정
        if any(check['status'] == 'unhealthy' for check in checks.values()):
            health_status['status'] = 'unhealthy'
        elif any(check['status'] == 'degraded' for check in checks.values()):
            health_status['status'] = 'degraded'
        
        return health_status
    
    def _check_metrics_collection(self) -> Dict[str, str]:
        """메트릭 수집 상태 체크"""
        if not self.metrics_buffer:
            return {'status': 'unhealthy', 'message': 'No metrics collected'}
        
        latest_metric = max(self.metrics_buffer, key=lambda m: m.timestamp)
        age = (datetime.now() - latest_metric.timestamp).total_seconds()
        
        if age > 300:  # 5분 이상 된 데이터
            return {'status': 'unhealthy', 'message': f'Stale data: {age:.0f}s old'}
        elif age > 60:  # 1분 이상
            return {'status': 'degraded', 'message': f'Delayed data: {age:.0f}s old'}
        
        return {'status': 'healthy', 'message': 'Collecting metrics normally'}
    
    def _check_alert_system(self) -> Dict[str, str]:
        """알림 시스템 상태 체크"""
        # 실제로는 마지막 알림 발송 시간 등을 체크
        return {'status': 'healthy', 'message': 'Alert system operational'}
    
    def _check_data_freshness(self) -> Dict[str, str]:
        """데이터 신선도 체크"""
        if not self.metrics_buffer:
            return {'status': 'unhealthy', 'message': 'No data available'}
        
        # 최근 1분간 데이터 개수
        recent_count = sum(1 for m in self.metrics_buffer 
                          if m.timestamp > datetime.now() - timedelta(minutes=1))
        
        if recent_count < 10:
            return {'status': 'degraded', 'message': f'Low traffic: {recent_count} requests/min'}
        
        return {'status': 'healthy', 'message': f'Normal traffic: {recent_count} requests/min'}

# Grafana 대시보드 설정
GRAFANA_DASHBOARD_CONFIG = {
    "dashboard": {
        "title": "RAG System Monitoring",
        "panels": [
            {
                "title": "Request Rate",
                "targets": [{
                    "expr": "rate(rag_requests_total[5m])",
                    "legendFormat": "{{endpoint}} - {{status}}"
                }]
            },
            {
                "title": "Latency Percentiles",
                "targets": [{
                    "expr": "histogram_quantile(0.95, rate(rag_latency_seconds_bucket[5m]))",
                    "legendFormat": "P95 {{operation}}"
                }]
            },
            {
                "title": "Relevance Score Distribution",
                "targets": [{
                    "expr": "rag_relevance_score",
                    "legendFormat": "Relevance Score"
                }]
            },
            {
                "title": "Cache Hit Rate",
                "targets": [{
                    "expr": "rag_cache_hit_rate",
                    "legendFormat": "Cache Hit %"
                }]
            },
            {
                "title": "Active Users",
                "targets": [{
                    "expr": "rag_active_users",
                    "legendFormat": "Active Users"
                }]
            },
            {
                "title": "Error Rate",
                "targets": [{
                    "expr": "rate(rag_requests_total{status='error'}[5m]) / rate(rag_requests_total[5m])",
                    "legendFormat": "Error Rate"
                }]
            }
        ]
    }
}

# 사용 예제
async def simulate_rag_traffic():
    """RAG 트래픽 시뮬레이션"""
    monitoring = RAGMonitoringService()
    
    # 100개의 요청 시뮬레이션
    for i in range(100):
        # 메트릭 생성
        latency_retrieval = np.random.exponential(0.2)  # 평균 200ms
        latency_generation = np.random.exponential(0.5)  # 평균 500ms
        
        metrics = RAGMetrics(
            request_id=f"req_{i}",
            timestamp=datetime.now(),
            query=f"Query {i % 20}",
            latency_total=latency_retrieval + latency_generation + np.random.exponential(0.1),
            latency_retrieval=latency_retrieval,
            latency_generation=latency_generation,
            documents_retrieved=np.random.randint(5, 20),
            relevance_score=min(0.95, max(0.5, np.random.normal(0.85, 0.1))),
            cache_hit=np.random.random() > 0.6,
            error="timeout" if np.random.random() < 0.02 else None,
            user_id=f"user_{i % 30}"
        )
        
        await monitoring.record_request(metrics)
        await asyncio.sleep(0.1)  # 100ms 간격
    
    # 대시보드 데이터 조회
    dashboard_data = monitoring.get_dashboard_data()
    
    print("=== RAG Monitoring Dashboard ===")
    print(f"Current Metrics:")
    for metric, value in dashboard_data['current_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    print(f"\nTop Slow Queries:")
    for query in dashboard_data['slow_queries'][:5]:
        print(f"  - {query['query']}: {query['latency_total']:.2f}s")
    
    # 헬스 체크
    health = await monitoring.health_check()
    print(f"\nHealth Status: {health['status']}")
    for component, status in health['checks'].items():
        print(f"  {component}: {status['status']} - {status['message']}")

# 실행
print("=== RAG Monitoring System Demo ===\n")
asyncio.run(simulate_rag_traffic())`}
                </pre>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">실제 Grafana 대시보드 구성</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">🎯 핵심 메트릭</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• Request Rate (QPS)</li>
                    <li>• Latency (P50, P95, P99)</li>
                    <li>• Error Rate & Success Rate</li>
                    <li>• Active Users</li>
                    <li>• Resource Utilization</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">📊 상세 분석</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• Query Distribution</li>
                    <li>• Document Retrieval Stats</li>
                    <li>• Model Performance</li>
                    <li>• Cache Performance</li>
                    <li>• Cost Analysis</li>
                  </ul>
                </div>
              </div>

              <div className="mt-4 bg-emerald-100 dark:bg-emerald-900/40 p-4 rounded-lg">
                <p className="text-sm text-emerald-800 dark:text-emerald-200">
                  <strong>💡 Pro Tip:</strong> Golden Signals (Latency, Traffic, Errors, Saturation)를
                  기반으로 대시보드를 구성하면 시스템 상태를 효과적으로 파악할 수 있습니다.
                  특히 RAG 시스템에서는 Relevance Score를 추가 지표로 활용하세요.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Practical Exercise */}
        <section className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">실습 과제</h2>
          
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">종합 RAG 평가 및 모니터링 시스템 구축</h3>
            
            <div className="space-y-4">
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">📋 요구사항</h4>
                <ol className="space-y-2 text-sm">
                  <li>1. RAGAS 프레임워크를 활용한 오프라인 평가 시스템 구축</li>
                  <li>2. A/B 테스트 플랫폼 구현 (최소 3가지 변형 지원)</li>
                  <li>3. Prometheus + Grafana 실시간 모니터링 구축</li>
                  <li>4. 자동 알림 시스템 (Slack/Email 연동)</li>
                  <li>5. 주간 성능 리포트 자동 생성</li>
                </ol>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🎯 평가 기준</h4>
                <ul className="space-y-1 text-sm">
                  <li>• 평가 메트릭의 포괄성 (Retrieval + Generation + System)</li>
                  <li>• A/B 테스트의 통계적 엄밀성</li>
                  <li>• 모니터링 대시보드의 실용성</li>
                  <li>• 알림 시스템의 정확성 (False positive rate &lt; 5%)</li>
                  <li>• 시스템 확장성 (1000+ QPS 지원)</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">💡 도전 과제</h4>
                <p className="text-sm">
                  ML 기반 이상 탐지 시스템을 구축하여 단순 threshold 기반 알림을
                  넘어서는 지능형 모니터링을 구현해보세요. 특히 계절성과 트렌드를
                  고려한 동적 임계값 설정을 구현하면 더욱 효과적입니다.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter4"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 고급 Reranking 전략
          </Link>
          
          <Link
            href="/modules/rag/advanced/chapter6"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            다음: 최신 연구 동향
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}