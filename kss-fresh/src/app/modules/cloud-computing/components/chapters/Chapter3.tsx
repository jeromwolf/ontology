'use client';

// Chapter 3: Azure 기초
export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">Microsoft Azure 소개</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          Microsoft의 클라우드 플랫폼으로, 200개 이상의 서비스를 제공합니다.
          Windows 기반 엔터프라이즈 환경과의 뛰어난 통합성이 장점입니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Azure 핵심 서비스</h2>
        <div className="grid gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Virtual Machines</h3>
            <p className="text-gray-700 dark:text-gray-300">
              Azure의 IaaS 컴퓨팅 서비스 (AWS EC2에 해당)
            </p>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">App Service</h3>
            <p className="text-gray-700 dark:text-gray-300">
              웹 앱, 모바일 백엔드를 위한 PaaS 플랫폼
            </p>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Azure Functions</h3>
            <p className="text-gray-700 dark:text-gray-300">
              이벤트 기반 서버리스 컴퓨팅 (AWS Lambda와 유사)
            </p>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-blue-800 dark:text-blue-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">Azure VM, App Service, Functions</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">엔터프라이즈 통합 강점</span>
          </li>
        </ul>
      </section>
    </div>
  )
}
