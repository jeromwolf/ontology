'use client'

import { FileText, CheckCircle2 } from 'lucide-react'

export default function Section2() {
  return (
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
  )
}
