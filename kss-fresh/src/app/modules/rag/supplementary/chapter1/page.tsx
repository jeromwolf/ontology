'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, BarChart3, CheckCircle2, AlertCircle, Code, FileText, TrendingUp } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter1Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/supplementary"
          className="inline-flex items-center gap-2 text-purple-600 hover:text-purple-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          보충 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <BarChart3 size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 1: RAGAS 평가 프레임워크</h1>
              <p className="text-purple-100 text-lg">Production RAG 시스템의 정량적 품질 측정</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: RAGAS Introduction */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <BarChart3 className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.1 RAGAS란 무엇인가?</h2>
              <p className="text-gray-600 dark:text-gray-400">Reference-Aware Grading And Scoring System</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-3">왜 RAGAS가 필요한가?</h3>
              <ul className="space-y-2 text-purple-700 dark:text-purple-300">
                <li>• RAG 시스템의 품질을 객관적으로 측정</li>
                <li>• 인간 평가 없이 자동화된 평가 가능</li>
                <li>• 모델 변경/업데이트 시 성능 추적</li>
                <li>• A/B 테스트 및 지속적 개선 가능</li>
              </ul>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
              <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">설치 및 초기 설정</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`# RAGAS 설치
pip install ragas langchain openai

# 필수 라이브러리 import
from ragas import evaluate
from ragas.metrics import (
    context_relevancy,
    answer_faithfulness,
    answer_relevancy,
    context_recall
)`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Section 2: Context Relevancy */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
              <FileText className="text-blue-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.2 Context Relevancy (문맥 관련성)</h2>
              <p className="text-gray-600 dark:text-gray-400">검색된 문서가 질문과 얼마나 관련이 있는가?</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">평가 원리</h3>
              <p className="text-blue-700 dark:text-blue-300 mb-4">
                Context Relevancy는 검색된 문서 중 실제로 질문에 답하는데 필요한 정보의 비율을 측정합니다.
              </p>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-blue-200 dark:border-blue-700">
                <p className="text-sm font-mono text-blue-600 dark:text-blue-400">
                  점수 = (관련 문장 수) / (전체 문장 수)
                </p>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
              <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">실제 구현 코드</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`from ragas.metrics import context_relevancy
from datasets import Dataset

# 평가 데이터 준비
data = {
    "question": [
        "한국의 수도는 어디인가요?",
        "Python에서 리스트를 정렬하는 방법은?"
    ],
    "contexts": [
        ["서울은 한국의 수도이며, 인구 약 950만명의 대도시입니다."],
        ["Python에서는 sort() 메서드나 sorted() 함수로 리스트를 정렬할 수 있습니다. sort()는 원본을 변경하고, sorted()는 새 리스트를 반환합니다."]
    ],
    "answer": [
        "한국의 수도는 서울입니다.",
        "sort() 메서드나 sorted() 함수를 사용합니다."
    ]
}

dataset = Dataset.from_dict(data)

# Context Relevancy 평가
result = evaluate(
    dataset,
    metrics=[context_relevancy],
)

print(f"Context Relevancy Score: {result['context_relevancy']:.3f}")`}</code>
              </pre>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-3">Production 체크리스트</h3>
              <ul className="space-y-2">
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="text-green-600 mt-1" size={16} />
                  <span className="text-green-700 dark:text-green-300">임계값 설정: 일반적으로 0.7 이상을 권장</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="text-green-600 mt-1" size={16} />
                  <span className="text-green-700 dark:text-green-300">모니터링: 시간에 따른 점수 추이 관찰</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="text-green-600 mt-1" size={16} />
                  <span className="text-green-700 dark:text-green-300">알림 설정: 점수가 임계값 이하로 떨어지면 즉시 알림</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Section 3: Answer Faithfulness */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
              <CheckCircle2 className="text-green-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.3 Answer Faithfulness (답변 충실도)</h2>
              <p className="text-gray-600 dark:text-gray-400">답변이 제공된 문맥에 얼마나 충실한가?</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-3">평가 원리</h3>
              <p className="text-green-700 dark:text-green-300 mb-4">
                답변의 각 주장이 검색된 문맥에서 직접 유추 가능한지 검증합니다. 환각(hallucination)을 방지하는 핵심 지표입니다.
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
              <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">실무 예제: 환각 감지</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`# 환각 감지 시스템 구현
class HallucinationDetector:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.metric = answer_faithfulness
        
    def check_answer(self, question, context, answer):
        data = {
            "question": [question],
            "contexts": [[context]],
            "answer": [answer]
        }
        
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[self.metric])
        
        score = result['answer_faithfulness']
        
        if score < self.threshold:
            return {
                "status": "hallucination_detected",
                "score": score,
                "message": "답변에 문맥에 없는 내용이 포함되어 있습니다."
            }
        
        return {
            "status": "faithful",
            "score": score,
            "message": "답변이 문맥에 충실합니다."
        }

# 사용 예제
detector = HallucinationDetector(threshold=0.8)

result = detector.check_answer(
    question="Python의 장점은?",
    context="Python은 읽기 쉬운 문법과 풍부한 라이브러리를 제공합니다.",
    answer="Python은 읽기 쉬운 문법, 풍부한 라이브러리, 그리고 빠른 실행 속도를 제공합니다."  # 환각: 빠른 실행 속도
)

print(result)`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Section 4: Answer Relevancy */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
              <TrendingUp className="text-orange-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.4 Answer Relevancy (답변 관련성)</h2>
              <p className="text-gray-600 dark:text-gray-400">답변이 질문에 얼마나 직접적으로 답하는가?</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-3">평가 원리</h3>
              <p className="text-orange-700 dark:text-orange-300 mb-4">
                답변에서 생성 가능한 질문들과 원래 질문의 유사도를 측정합니다. 불필요한 정보나 주제에서 벗어난 내용을 감지합니다.
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
              <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">자동화된 평가 파이프라인</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`# Production 환경을 위한 자동 평가 시스템
import pandas as pd
from datetime import datetime
import json

class RAGEvaluationPipeline:
    def __init__(self):
        self.metrics = [
            context_relevancy,
            answer_faithfulness,
            answer_relevancy,
            context_recall
        ]
        self.thresholds = {
            'context_relevancy': 0.7,
            'answer_faithfulness': 0.8,
            'answer_relevancy': 0.75,
            'context_recall': 0.65
        }
        
    def evaluate_batch(self, qa_pairs):
        """배치 평가 실행"""
        dataset = Dataset.from_dict(qa_pairs)
        results = evaluate(dataset, metrics=self.metrics)
        
        # 결과 분석
        analysis = self.analyze_results(results)
        
        # 로깅
        self.log_results(results, analysis)
        
        # 알림 확인
        self.check_alerts(analysis)
        
        return analysis
    
    def analyze_results(self, results):
        """결과 분석 및 인사이트 생성"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'warnings': [],
            'insights': []
        }
        
        for metric_name, score in results.items():
            analysis['scores'][metric_name] = score
            
            # 임계값 체크
            if score < self.thresholds.get(metric_name, 0.5):
                analysis['warnings'].append({
                    'metric': metric_name,
                    'score': score,
                    'threshold': self.thresholds[metric_name],
                    'severity': 'high' if score < 0.5 else 'medium'
                })
        
        # 인사이트 생성
        if results['context_relevancy'] < 0.7:
            analysis['insights'].append(
                "검색 품질 개선 필요: 임베딩 모델 재학습 고려"
            )
            
        if results['answer_faithfulness'] < 0.8:
            analysis['insights'].append(
                "환각 위험 감지: 프롬프트 엔지니어링 재검토 필요"
            )
            
        return analysis
    
    def log_results(self, results, analysis):
        """결과 로깅"""
        log_entry = {
            'timestamp': analysis['timestamp'],
            'raw_scores': results,
            'analysis': analysis
        }
        
        # JSON 파일로 저장 (실제로는 DB나 로깅 시스템 사용)
        with open(f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def check_alerts(self, analysis):
        """알림 조건 확인"""
        if analysis['warnings']:
            # 실제로는 Slack, Email 등으로 알림 발송
            print("⚠️ ALERT: RAG 품질 저하 감지!")
            for warning in analysis['warnings']:
                print(f"  - {warning['metric']}: {warning['score']:.3f} (임계값: {warning['threshold']})")

# 사용 예제
pipeline = RAGEvaluationPipeline()

# 실제 QA 데이터
qa_data = {
    "question": ["서울의 인구는?", "Python 리스트 정렬 방법은?"],
    "contexts": [
        ["서울의 인구는 약 950만명입니다."],
        ["Python에서 리스트를 정렬하려면 sort() 메서드를 사용하세요."]
    ],
    "answer": [
        "서울의 인구는 약 950만명입니다.",
        "sort() 메서드를 사용하면 됩니다."
    ],
    "ground_truths": [
        ["서울 인구는 950만명"],
        ["sort() 메서드 사용", "sorted() 함수 사용"]
    ]
}

# 평가 실행
analysis = pipeline.evaluate_batch(qa_data)
print(json.dumps(analysis, indent=2, ensure_ascii=False))`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Section 5: Custom Metrics */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
              <Code className="text-indigo-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.5 커스텀 메트릭 개발</h2>
              <p className="text-gray-600 dark:text-gray-400">비즈니스 요구사항에 맞춘 평가 지표 생성</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-3">실무 사례: 도메인 특화 메트릭</h3>
              <p className="text-indigo-700 dark:text-indigo-300 mb-4">
                의료, 법률, 금융 등 도메인별로 특화된 평가 기준이 필요할 수 있습니다.
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
              <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">의료 도메인 커스텀 메트릭 예제</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`from ragas.metrics import Metric
from langchain.chat_models import ChatOpenAI
import re

class MedicalAccuracyMetric(Metric):
    """의료 정보의 정확성을 평가하는 커스텀 메트릭"""
    
    name = "medical_accuracy"
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        # 의료 전문 용어 사전
        self.medical_terms = {
            "약물명": ["아스피린", "타이레놀", "부루펜"],
            "용량단위": ["mg", "ml", "IU"],
            "금기사항": ["임산부", "수유부", "알레르기"]
        }
    
    def score(self, question, answer, context):
        """의료 정보 정확성 점수 계산"""
        scores = []
        
        # 1. 약물 용량 정확성 체크
        dose_score = self._check_dosage_accuracy(answer, context)
        scores.append(dose_score)
        
        # 2. 금기사항 누락 체크
        contraindication_score = self._check_contraindications(answer, context)
        scores.append(contraindication_score)
        
        # 3. 의학 용어 일관성 체크
        terminology_score = self._check_terminology_consistency(answer, context)
        scores.append(terminology_score)
        
        # 전체 점수 계산 (가중평균)
        weights = [0.5, 0.3, 0.2]  # 용량 > 금기사항 > 용어
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        return final_score
    
    def _check_dosage_accuracy(self, answer, context):
        """약물 용량 정보의 정확성 확인"""
        # 숫자+단위 패턴 추출
        dose_pattern = r'\d+\s*(mg|ml|IU|g)'
        
        answer_doses = re.findall(dose_pattern, answer)
        context_doses = re.findall(dose_pattern, context)
        
        if not answer_doses:
            return 1.0  # 용량 정보 없으면 패스
            
        # 답변의 용량이 문맥에 있는지 확인
        correct_doses = sum(1 for dose in answer_doses if dose in context_doses)
        
        return correct_doses / len(answer_doses) if answer_doses else 1.0
    
    def _check_contraindications(self, answer, context):
        """중요 금기사항 누락 확인"""
        # 문맥에서 금기사항 추출
        context_warnings = []
        for warning in self.medical_terms["금기사항"]:
            if warning in context:
                context_warnings.append(warning)
        
        if not context_warnings:
            return 1.0  # 금기사항 없으면 패스
            
        # 답변에 포함되었는지 확인
        included_warnings = sum(1 for w in context_warnings if w in answer)
        
        return included_warnings / len(context_warnings)
    
    def _check_terminology_consistency(self, answer, context):
        """의학 용어 일관성 확인"""
        # 간단한 일관성 체크 (실제로는 더 복잡)
        return 0.9  # 예제를 위한 고정값

# 사용 예제
medical_metric = MedicalAccuracyMetric()

# 의료 QA 데이터
medical_qa = {
    "question": "감기에 타이레놀 복용법은?",
    "context": "타이레놀은 성인 기준 500mg을 4-6시간마다 복용합니다. 일일 최대 4000mg을 초과하지 마세요. 임산부는 의사와 상담 후 복용하세요.",
    "answer": "타이레놀은 500mg을 4시간마다 복용하면 됩니다."  # 임산부 주의사항 누락
}

score = medical_metric.score(
    medical_qa["question"],
    medical_qa["answer"],
    medical_qa["context"]
)

print(f"Medical Accuracy Score: {score:.3f}")
print("⚠️ 경고: 중요 금기사항이 누락되었습니다!" if score < 0.8 else "✅ 의료 정보 정확도 양호")`}</code>
              </pre>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-yellow-800 dark:text-yellow-200 mb-3">Production 고려사항</h3>
              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <AlertCircle className="text-yellow-600 mt-1" size={16} />
                  <div className="text-yellow-700 dark:text-yellow-300">
                    <p className="font-semibold">평가 비용 최적화</p>
                    <p className="text-sm">샘플링 전략: 전체의 10-20%만 평가하여 비용 절감</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <AlertCircle className="text-yellow-600 mt-1" size={16} />
                  <div className="text-yellow-700 dark:text-yellow-300">
                    <p className="font-semibold">실시간 vs 배치 평가</p>
                    <p className="text-sm">중요 쿼리는 실시간, 나머지는 일일 배치로 처리</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 6: Practical Implementation */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <TrendingUp className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.6 Production 구현 가이드</h2>
              <p className="text-gray-600 dark:text-gray-400">실제 서비스에서의 RAGAS 활용법</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">완전한 모니터링 대시보드</h3>
              <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
                <code>{`# Production 모니터링 시스템
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

class RAGMonitoringDashboard:
    def __init__(self):
        self.metrics_history = []
        
    def add_evaluation(self, eval_results):
        """평가 결과 추가"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'scores': eval_results,
            'alerts': self._check_alerts(eval_results)
        })
    
    def _check_alerts(self, scores):
        """알림 조건 확인"""
        alerts = []
        thresholds = {
            'context_relevancy': 0.7,
            'answer_faithfulness': 0.8,
            'answer_relevancy': 0.75
        }
        
        for metric, threshold in thresholds.items():
            if scores.get(metric, 1.0) < threshold:
                alerts.append({
                    'metric': metric,
                    'score': scores[metric],
                    'threshold': threshold,
                    'severity': 'high' if scores[metric] < threshold * 0.7 else 'medium'
                })
        
        return alerts
    
    def render_dashboard(self):
        """Streamlit 대시보드 렌더링"""
        st.title("🚀 RAG System Monitoring Dashboard")
        
        # 현재 상태 요약
        col1, col2, col3, col4 = st.columns(4)
        
        latest_scores = self.metrics_history[-1]['scores'] if self.metrics_history else {}
        
        with col1:
            st.metric(
                "Context Relevancy",
                f"{latest_scores.get('context_relevancy', 0):.2%}",
                delta=self._calculate_delta('context_relevancy')
            )
        
        with col2:
            st.metric(
                "Answer Faithfulness",
                f"{latest_scores.get('answer_faithfulness', 0):.2%}",
                delta=self._calculate_delta('answer_faithfulness')
            )
        
        with col3:
            st.metric(
                "Answer Relevancy",
                f"{latest_scores.get('answer_relevancy', 0):.2%}",
                delta=self._calculate_delta('answer_relevancy')
            )
        
        with col4:
            active_alerts = sum(len(h['alerts']) for h in self.metrics_history[-10:])
            st.metric("Active Alerts", active_alerts, delta=-2 if active_alerts > 0 else 0)
        
        # 시계열 그래프
        st.subheader("📊 Metrics Over Time")
        self._render_time_series()
        
        # 알림 섹션
        st.subheader("⚠️ Recent Alerts")
        self._render_alerts()
        
        # 권장 조치
        st.subheader("💡 Recommended Actions")
        self._render_recommendations()
    
    def _calculate_delta(self, metric_name):
        """변화량 계산"""
        if len(self.metrics_history) < 2:
            return 0
        
        current = self.metrics_history[-1]['scores'].get(metric_name, 0)
        previous = self.metrics_history[-2]['scores'].get(metric_name, 0)
        
        return f"{(current - previous):.1%}"
    
    def _render_time_series(self):
        """시계열 차트 렌더링"""
        if not self.metrics_history:
            st.info("No data available yet")
            return
        
        # 데이터 준비
        df = pd.DataFrame([
            {
                'timestamp': h['timestamp'],
                **h['scores']
            }
            for h in self.metrics_history
        ])
        
        # Plotly 차트
        fig = go.Figure()
        
        for metric in ['context_relevancy', 'answer_faithfulness', 'answer_relevancy']:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=2)
            ))
        
        # 임계값 라인 추가
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                      annotation_text="Critical Threshold")
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                      annotation_text="Warning Threshold")
        
        fig.update_layout(
            title="RAG Metrics Trend",
            xaxis_title="Time",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts(self):
        """최근 알림 표시"""
        recent_alerts = []
        for h in self.metrics_history[-10:]:
            for alert in h['alerts']:
                recent_alerts.append({
                    'Time': h['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'Metric': alert['metric'].replace('_', ' ').title(),
                    'Score': f"{alert['score']:.2%}",
                    'Threshold': f"{alert['threshold']:.2%}",
                    'Severity': alert['severity'].upper()
                })
        
        if recent_alerts:
            df = pd.DataFrame(recent_alerts)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No alerts in recent evaluations!")
    
    def _render_recommendations(self):
        """권장 조치 제안"""
        if not self.metrics_history:
            return
        
        latest = self.metrics_history[-1]['scores']
        
        recommendations = []
        
        if latest.get('context_relevancy', 1) < 0.7:
            recommendations.append({
                'issue': 'Low Context Relevancy',
                'action': '• 임베딩 모델 재학습\n• 청킹 전략 개선\n• 메타데이터 필터링 강화'
            })
        
        if latest.get('answer_faithfulness', 1) < 0.8:
            recommendations.append({
                'issue': 'Low Answer Faithfulness',
                'action': '• 프롬프트 엔지니어링 개선\n• 컨텍스트 윈도우 크기 조정\n• Few-shot 예제 추가'
            })
        
        if recommendations:
            for rec in recommendations:
                st.warning(f"**{rec['issue']}**")
                st.markdown(rec['action'])
        else:
            st.success("시스템이 최적 상태로 운영 중입니다!")

# 실제 사용 예제
if __name__ == "__main__":
    dashboard = RAGMonitoringDashboard()
    
    # 시뮬레이션 데이터 추가 (실제로는 실시간 평가 결과)
    import random
    for i in range(24):  # 24시간 데이터
        scores = {
            'context_relevancy': random.uniform(0.65, 0.85),
            'answer_faithfulness': random.uniform(0.75, 0.95),
            'answer_relevancy': random.uniform(0.70, 0.90),
            'context_recall': random.uniform(0.60, 0.80)
        }
        dashboard.add_evaluation(scores)
    
    # Streamlit 앱 실행
    dashboard.render_dashboard()`}</code>
              </pre>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">비용 분석</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-blue-200 dark:border-blue-700">
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">평가 전략</th>
                      <th className="text-right py-2 text-blue-800 dark:text-blue-200">월간 쿼리</th>
                      <th className="text-right py-2 text-blue-800 dark:text-blue-200">평가 비율</th>
                      <th className="text-right py-2 text-blue-800 dark:text-blue-200">예상 비용</th>
                    </tr>
                  </thead>
                  <tbody className="text-blue-700 dark:text-blue-300">
                    <tr>
                      <td className="py-2">전체 평가</td>
                      <td className="text-right">100,000</td>
                      <td className="text-right">100%</td>
                      <td className="text-right">$2,000</td>
                    </tr>
                    <tr>
                      <td className="py-2">샘플링 (권장)</td>
                      <td className="text-right">100,000</td>
                      <td className="text-right">15%</td>
                      <td className="text-right">$300</td>
                    </tr>
                    <tr>
                      <td className="py-2">중요 쿼리만</td>
                      <td className="text-right">100,000</td>
                      <td className="text-right">5%</td>
                      <td className="text-right">$100</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 RAGAS & 평가 프레임워크',
              icon: 'web' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'RAGAS Official Documentation',
                  authors: 'Explodinggradients',
                  year: '2024',
                  description: 'RAG 평가 프레임워크 - Context Relevancy, Answer Faithfulness, Answer Relevancy 공식 문서',
                  link: 'https://docs.ragas.io/'
                },
                {
                  title: 'TruLens: LLM Evaluation & Observability',
                  authors: 'TruEra',
                  year: '2024',
                  description: 'LLM 애플리케이션 평가 - Groundedness, Answer Relevance, Context Relevance 측정',
                  link: 'https://www.trulens.org/'
                },
                {
                  title: 'LangSmith Evaluation',
                  authors: 'LangChain',
                  year: '2024',
                  description: 'LangChain 공식 평가 도구 - 자동화된 테스트, 비교 분석, Production 모니터링',
                  link: 'https://docs.smith.langchain.com/evaluation'
                },
                {
                  title: 'DeepEval: Unit Testing for LLMs',
                  authors: 'Confident AI',
                  year: '2024',
                  description: 'LLM 단위 테스트 프레임워크 - 14개 평가 메트릭, Pytest 통합, CI/CD 지원',
                  link: 'https://docs.confident-ai.com/'
                },
                {
                  title: 'Evidently AI: ML Monitoring',
                  authors: 'Evidently AI',
                  year: '2024',
                  description: 'ML 시스템 모니터링 - 데이터 드리프트, 성능 저하 감지, 대시보드 생성',
                  link: 'https://www.evidentlyai.com/'
                }
              ]
            },
            {
              title: '📖 RAG 평가 연구 논문',
              icon: 'research' as const,
              color: 'border-pink-500',
              items: [
                {
                  title: 'ARES: An Automated Evaluation Framework for RAG',
                  authors: 'Saad-Falcon et al., Stanford',
                  year: '2024',
                  description: '자동화된 RAG 평가 - Synthetic 데이터 생성, LLM-as-judge, 인간 평가 대체',
                  link: 'https://arxiv.org/abs/2311.09476'
                },
                {
                  title: 'Benchmarking Large Language Models in RAG',
                  authors: 'Chen et al., Tsinghua University',
                  year: '2024',
                  description: 'RGB 벤치마크 - 4개 도메인, 다양한 RAG 시나리오, 종합 평가 프레임워크',
                  link: 'https://arxiv.org/abs/2309.01431'
                },
                {
                  title: 'RAGAS: Automated Evaluation of RAG',
                  authors: 'Es et al., Explodinggradients',
                  year: '2023',
                  description: 'RAGAS 논문 - Reference-free 평가, LLM 기반 메트릭, 자동화된 품질 측정',
                  link: 'https://arxiv.org/abs/2309.15217'
                },
                {
                  title: 'Evaluating RAG: A Survey',
                  authors: 'Liu et al., Microsoft Research',
                  year: '2024',
                  description: 'RAG 평가 서베이 - 기존 메트릭 분류, 한계점 분석, 미래 방향 제시',
                  link: 'https://arxiv.org/abs/2405.17009'
                }
              ]
            },
            {
              title: '🛠️ Production 모니터링 도구',
              icon: 'tools' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'Weights & Biases: ML Experiment Tracking',
                  authors: 'Weights & Biases',
                  year: '2024',
                  description: 'ML 실험 추적 - 메트릭 시각화, 하이퍼파라미터 최적화, 팀 협업',
                  link: 'https://wandb.ai/'
                },
                {
                  title: 'MLflow: Open Source ML Platform',
                  authors: 'Databricks',
                  year: '2024',
                  description: 'ML 라이프사이클 관리 - 실험 추적, 모델 레지스트리, 배포 자동화',
                  link: 'https://mlflow.org/'
                },
                {
                  title: 'Streamlit: Data App Framework',
                  authors: 'Snowflake',
                  year: '2024',
                  description: 'Python 대시보드 - 실시간 모니터링 UI, 빠른 프로토타이핑, 인터랙티브 차트',
                  link: 'https://streamlit.io/'
                },
                {
                  title: 'Grafana + Prometheus: Metrics Monitoring',
                  authors: 'Grafana Labs',
                  year: '2024',
                  description: '시계열 메트릭 모니터링 - 알림 설정, 실시간 대시보드, 다중 데이터소스 지원',
                  link: 'https://grafana.com/'
                },
                {
                  title: 'Arize AI: ML Observability Platform',
                  authors: 'Arize AI',
                  year: '2024',
                  description: 'ML 관측성 플랫폼 - 성능 모니터링, 드리프트 감지, 근본 원인 분석',
                  link: 'https://arize.com/'
                }
              ]
            }
          ]}
        />

        {/* Navigation */}
        <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
          <Link
            href="/modules/rag/supplementary"
            className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
          >
            <ArrowLeft size={20} />
            보충 과정으로 돌아가기
          </Link>
          
          <Link
            href="/modules/rag/supplementary/chapter2"
            className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
          >
            다음: Security & Privacy
            <ArrowRight size={20} />
          </Link>
        </div>
      </div>
    </div>
  )
}