'use client'

import { TrendingUp } from 'lucide-react'

export default function Section4() {
  return (
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
  )
}
