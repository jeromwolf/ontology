import { TestTube } from 'lucide-react'

export default function Section2ABTesting() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
          <TestTube className="text-purple-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.2 A/B 테스팅 프레임워크</h2>
          <p className="text-gray-600 dark:text-gray-400">데이터 기반 RAG 시스템 개선</p>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
        <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">RAG A/B 테스팅 시스템</h3>
        <p className="text-gray-700 dark:text-gray-300">
          A/B 테스팅은 데이터 기반으로 RAG 시스템의 성능을 지속적으로 개선하는 핵심 방법론입니다.
          사용자를 무작위로 분할하여 서로 다른 RAG 설정을 경험하게 한 후, 
          통계적으로 유의미한 차이를 측정하여 최적의 구성을 찾아냅니다.
        </p>
        
        <div className="mt-4">
          <h4 className="font-medium text-purple-800 dark:text-purple-200 mb-2">주요 테스트 영역:</h4>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
            <li>검색 알고리즘 비교 (Vector vs Hybrid vs BM25)</li>
            <li>LLM 모델 성능 평가 (GPT-4 vs Claude vs Llama)</li>
            <li>청킹 전략 최적화 (고정 크기 vs 의미 기반)</li>
            <li>프롬프트 엔지니어링 기법 비교</li>
          </ul>
        </div>
      </div>
    </section>
  )
}