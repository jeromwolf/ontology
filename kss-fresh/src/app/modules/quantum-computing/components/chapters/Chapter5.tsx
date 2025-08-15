'use client';

import { Shield } from 'lucide-react';

export default function Chapter5() {
  return (
    <div className="p-8 space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Shield className="w-8 h-8 text-purple-600" />
          양자 오류의 본질
        </h2>
        
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-bold text-red-700 dark:text-red-400 mb-4">⚠️ 양자 컴퓨터의 최대 도전</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            양자 상태는 환경과의 상호작용으로 인해 매우 빠르게 손상됩니다. 
            현재 양자 컴퓨터의 오류율은 0.1~1%로, 실용적 계산을 위해서는 10⁻⁶ 이하로 낮춰야 합니다.
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-red-700 dark:text-red-400 mb-4">🌡️ 디코히어런스</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              환경과의 얽힘으로 양자 중첩이 파괴되는 현상
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• T₁: 이완 시간 (~100μs)</li>
              <li>• T₂: 위상 손실 시간 (~50μs)</li>
              <li>• 온도, 자기장 변화</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-yellow-50 to-amber-50 dark:from-yellow-900/20 dark:to-amber-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-yellow-700 dark:text-yellow-400 mb-4">⚡ 게이트 오류</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              양자 게이트 연산 중 발생하는 부정확성
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 제어 신호 부정확성</li>
              <li>• 큐비트 간 unwanted 상호작용</li>
              <li>• 교정 오류 (calibration drift)</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-700 dark:text-purple-400 mb-4">📐 측정 오류</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              큐비트 상태 판독 시 발생하는 오류
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• |0⟩ → |1⟩ 잘못 읽기</li>
              <li>• |1⟩ → |0⟩ 잘못 읽기</li>
              <li>• 전형적으로 1-5% 오류율</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🔧 3-큐비트 비트 플립 코드</h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">📊 기본 원리</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              하나의 논리 큐비트를 3개의 물리 큐비트로 인코딩하여 단일 비트 플립 오류를 정정합니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">인코딩:</h4>
              <code className="text-sm">
                |0⟩_L → |000⟩<br/>
                |1⟩_L → |111⟩<br/>
                |ψ⟩ = α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩
              </code>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🏗️ Surface Code</h2>
        
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🌐 실용적 양자 오류 정정</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Surface Code는 현재 가장 유망한 양자 오류 정정 코드로, 2D 격자에서 국소적 연산만을 사용합니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">장점</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 높은 오류 임계값 (~1%)</li>
                <li>• 2D 격자에서 국소적 연산</li>
                <li>• 확장 가능한 구조</li>
                <li>• 임의의 논리 게이트 가능</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">요구사항</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 거리 d 코드: d² 큐비트</li>
                <li>• 논리 오류율 ∝ p^(d+1)/2</li>
                <li>• 수천 개의 물리 큐비트</li>
                <li>• 실시간 신드롬 처리</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}