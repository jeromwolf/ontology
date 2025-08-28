import { Scale } from 'lucide-react'

export default function Section4ScalingStrategies() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
          <Scale className="text-blue-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.4 확장성 전략</h2>
          <p className="text-gray-600 dark:text-gray-400">대규모 트래픽 대응 전략</p>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
        <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">대규모 RAG 시스템 확장</h3>
        <p className="text-gray-700 dark:text-gray-300">
          대규모 사용자 요청을 처리하기 위한 확장성 전략은 시스템 안정성과 비용 효율성을 동시에 고려해야 합니다.
          마이크로서비스 아키텍처와 효율적인 쫐싱 전략을 통해 수평적 확장을 달성할 수 있습니다.
        </p>
        
        <div className="mt-4">
          <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">핵심 확장 기법:</h4>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
            <li>로드 밸런싱 및 오토 스케일링</li>
            <li>백터 DB 샤딩 및 대조 전략</li>
            <li>엓지 쫐싱과 CDN 활용</li>
            <li>비동기 처리 및 대기열 관리</li>
            <li>글로벌 멀티 리전 배포</li>
          </ul>
        </div>
      </div>
    </section>
  )
}