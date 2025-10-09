'use client'

import { BarChart3 } from 'lucide-react'

export default function Section1() {
  return (
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
  )
}
