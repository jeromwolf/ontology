import { Eye } from 'lucide-react'

export default function Section6MonitoringDashboards() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-teal-100 dark:bg-teal-900/20 flex items-center justify-center">
          <Eye className="text-teal-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.6 모니터링 대시보드</h2>
          <p className="text-gray-600 dark:text-gray-400">실시간 시스템 상태 모니터링</p>
        </div>
      </div>

      <div className="bg-teal-50 dark:bg-teal-900/20 p-6 rounded-xl border border-teal-200 dark:border-teal-700">
        <h3 className="font-bold text-teal-800 dark:text-teal-200 mb-4">실시간 모니터링 대시보드</h3>
        <p className="text-gray-700 dark:text-gray-300">
          Grafana 및 Prometheus를 활용한 실시간 모니터링 대시보드를 통해
          시스템 상태를 실시간으로 추적하고, 비정상 상황을 조기에 발견하여
          신속한 대응이 가능한 모니터링 체계를 구축합니다.
        </p>
        
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">시스템 상태</h4>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
              <span className="text-sm text-gray-600 dark:text-gray-400">정상 운영 중</span>
            </div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">응답 시간</h4>
            <div className="text-lg font-bold text-green-600">1.2초</div>
          </div>
        </div>
      </div>
    </section>
  )
}