import { Server } from 'lucide-react'

export default function Section5APIDeployment() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
          <Server className="text-indigo-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.5 API 디자인 및 배포</h2>
          <p className="text-gray-600 dark:text-gray-400">기업용 API 서비스 구축</p>
        </div>
      </div>

      <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
        <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">RESTful RAG API 디자인</h3>
        <p className="text-gray-700 dark:text-gray-300">
          기업용 RAG API는 직관적이고 안정적이며 확장 가능해야 합니다.
          RESTful 원칙을 따르고, 적절한 에러 처리와 레이트 리미팅을 구현하여
          외부 시스템과의 안정적인 통합을 지원해야 합니다.
        </p>
        
        <div className="mt-4">
          <h4 className="font-medium text-indigo-800 dark:text-indigo-200 mb-2">API 설계 원칙:</h4>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
            <li>RESTful 엔드포인트 설계</li>
            <li>비동기 처리 및 스트리밍 응답</li>
            <li>버전 관리 및 하위 호환성</li>
            <li>인증 및 인가 체계</li>
            <li>귀전 테스트 및 성능 리뿼스</li>
          </ul>
        </div>
      </div>
    </section>
  )
}