'use client'

import { Code, AlertCircle } from 'lucide-react'

export default function Section5() {
  return (
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
        dose_pattern = r'\\d+\\s*(mg|ml|IU|g)'

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
  )
}
