'use client';

import { Eye } from 'lucide-react';

export default function Chapter7() {
  return (
    <div className="p-8 space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Eye className="w-8 h-8 text-purple-600" />
          양자 컴퓨터 하드웨어 플랫폼
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-blue-700 dark:text-blue-400 mb-4">🧊 초전도 큐비트 (IBM, Google)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              조셉슨 접합을 이용한 초전도 회로로 큐비트를 구현하는 가장 성숙한 기술입니다.
            </p>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-green-600 dark:text-green-400 text-sm mb-1">장점</h4>
                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 빠른 게이트 연산 (10-100ns)</li>
                  <li>• 정확한 제어 가능</li>
                  <li>• 반도체 공정 활용</li>
                  <li>• 확장성 우수</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-red-600 dark:text-red-400 text-sm mb-1">단점</h4>
                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 극저온 필요 (10-20mK)</li>
                  <li>• 짧은 코히어런스 시간</li>
                  <li>• 복잡한 냉각 시스템</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-purple-700 dark:text-purple-400 mb-4">⚛️ 이온 트랩 (IonQ, Alpine)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              전기장으로 포획된 원자 이온을 큐비트로 사용하는 고정밀 플랫폼입니다.
            </p>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-green-600 dark:text-green-400 text-sm mb-1">장점</h4>
                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 높은 게이트 충실도 (99.9%+)</li>
                  <li>• 긴 코히어런스 시간</li>
                  <li>• 전 연결 토폴로지</li>
                  <li>• 정확한 상태 준비/측정</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-red-600 dark:text-red-400 text-sm mb-1">단점</h4>
                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 느린 게이트 속도 (μs)</li>
                  <li>• 복잡한 레이저 시스템</li>
                  <li>• 확장성 한계</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">☁️ 클라우드 양자 컴퓨팅 플랫폼</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🔧 주요 플랫폼 비교</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-300 dark:border-gray-600">
                  <th className="text-left p-3">플랫폼</th>
                  <th className="text-left p-3">하드웨어</th>
                  <th className="text-left p-3">언어/SDK</th>
                  <th className="text-left p-3">특징</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold text-blue-600">IBM Quantum</td>
                  <td className="p-3">초전도 큐비트</td>
                  <td className="p-3">Qiskit (Python)</td>
                  <td className="p-3">오픈소스, 교육 중심</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold text-green-600">Amazon Braket</td>
                  <td className="p-3">다중 벤더</td>
                  <td className="p-3">Python SDK</td>
                  <td className="p-3">AWS 통합, 시뮬레이터</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold text-purple-600">IonQ</td>
                  <td className="p-3">이온 트랩</td>
                  <td className="p-3">Cirq, Qiskit</td>
                  <td className="p-3">높은 충실도</td>
                </tr>
                <tr>
                  <td className="p-3 font-semibold text-indigo-600">Google Quantum AI</td>
                  <td className="p-3">초전도 큐비트</td>
                  <td className="p-3">Cirq (Python)</td>
                  <td className="p-3">연구 중심, 제한 접근</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  )
}