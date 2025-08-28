import { Shield } from 'lucide-react'

export default function Section3SecurityPrivacy() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
          <Shield className="text-red-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.3 보안 및 프라이버시</h2>
          <p className="text-gray-600 dark:text-gray-400">엔터프라이즈 수준의 RAG 시스템 보안</p>
        </div>
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border border-red-200 dark:border-red-700">
        <h3 className="font-bold text-red-800 dark:text-red-200 mb-4">보안 및 프라이버시 제어</h3>
        <p className="text-gray-700 dark:text-gray-300">
          Production RAG 시스템에서는 민감한 데이터를 안전하게 처리하고,
          개인정보 보호를 보장하며, 외부 위협으로부터 시스템을 보호하는 종합적인 보안 체계가 필요합니다.
        </p>
        
        <div className="mt-4">
          <h4 className="font-medium text-red-800 dark:text-red-200 mb-2">핵심 보안 영역:</h4>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
            <li>데이터 암호화 및 액세스 제어</li>
            <li>PII(개인식별정보) 탐지 및 마스킹</li>
            <li>쿨리 및 SQL 인젝션 방어</li>
            <li>실시간 위협 탐지 및 대응</li>
            <li>감사 추적 및 준수성 보고</li>
          </ul>
        </div>
      </div>
    </section>
  )
}