'use client';

import { Atom } from 'lucide-react';

export default function Chapter1() {
  return (
    <div className="p-8 space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Atom className="w-8 h-8 text-purple-600" />
          양자역학의 기본 원리
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-br from-purple-50 to-violet-50 dark:from-gray-700 dark:to-gray-700 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">🌊 양자 중첩 (Superposition)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              고전 비트는 0 또는 1의 상태만 가지지만, 큐비트는 0과 1의 상태를 동시에 가질 수 있습니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <code className="text-sm text-purple-600 dark:text-purple-400">
                |ψ⟩ = α|0⟩ + β|1⟩<br/>
                |α|² + |β|² = 1
              </code>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-700 dark:to-gray-700 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">🔗 양자 얽힘 (Entanglement)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              두 개 이상의 큐비트가 상관관계를 가져서, 하나의 측정이 다른 큐비트의 상태에 즉시 영향을 미칩니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <code className="text-sm text-blue-600 dark:text-blue-400">
                |Bell⟩ = (|00⟩ + |11⟩)/√2
              </code>
            </div>
          </div>
        </div>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 p-6 rounded-r-lg">
          <h4 className="font-bold text-yellow-800 dark:text-yellow-400 mb-2">💡 핵심 개념</h4>
          <p className="text-yellow-700 dark:text-yellow-300">
            양자역학의 "측정 문제": 큐비트를 측정하는 순간 중첩 상태가 붕괴되어 확정적인 0 또는 1 상태가 됩니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🎯 블로흐 구면 (Bloch Sphere)</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-6">
            블로흐 구면은 큐비트의 모든 가능한 상태를 3차원 구면 위의 점으로 표현하는 방법입니다.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gradient-to-b from-red-100 to-red-50 dark:from-red-900/30 dark:to-red-800/20 rounded-lg">
              <div className="text-2xl mb-2">🔴</div>
              <div className="font-bold text-red-700 dark:text-red-400">|0⟩ 상태</div>
              <div className="text-sm text-red-600 dark:text-red-300">북극 (Z=+1)</div>
            </div>
            
            <div className="text-center p-4 bg-gradient-to-b from-green-100 to-green-50 dark:from-green-900/30 dark:to-green-800/20 rounded-lg">
              <div className="text-2xl mb-2">🟢</div>
              <div className="font-bold text-green-700 dark:text-green-400">|+⟩ 상태</div>
              <div className="text-sm text-green-600 dark:text-green-300">적도 (X=+1)</div>
            </div>
            
            <div className="text-center p-4 bg-gradient-to-b from-blue-100 to-blue-50 dark:from-blue-900/30 dark:to-blue-800/20 rounded-lg">
              <div className="text-2xl mb-2">🔵</div>
              <div className="font-bold text-blue-700 dark:text-blue-400">|1⟩ 상태</div>
              <div className="text-sm text-blue-600 dark:text-blue-300">남극 (Z=-1)</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}