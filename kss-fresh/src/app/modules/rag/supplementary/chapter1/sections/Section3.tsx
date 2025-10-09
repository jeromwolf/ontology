'use client'

import { CheckCircle2 } from 'lucide-react'

export default function Section3() {
  return (
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
  )
}
