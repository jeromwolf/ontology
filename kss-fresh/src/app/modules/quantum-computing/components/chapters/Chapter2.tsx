'use client';

import { Calculator } from 'lucide-react';

export default function Chapter2() {
  return (
    <div className="p-8 space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Calculator className="w-8 h-8 text-purple-600" />
          기본 양자 게이트
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
              <h3 className="text-lg font-bold text-red-700 dark:text-red-400 mb-4">🎯 Pauli-X 게이트 (NOT)</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                큐비트의 상태를 뒤집는 게이트 (|0⟩ ↔ |1⟩)
              </p>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <code className="text-sm">
                  X = [0 1]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;[1 0]
                </code>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
              <h3 className="text-lg font-bold text-green-700 dark:text-green-400 mb-4">🌀 Pauli-Y 게이트</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                X와 Z의 조합, 복소수 위상 변화 포함
              </p>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <code className="text-sm">
                  Y = [0 -i]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;[i &nbsp;0]
                </code>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
              <h3 className="text-lg font-bold text-blue-700 dark:text-blue-400 mb-4">⚡ Pauli-Z 게이트</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                |1⟩ 상태에 -1 위상을 적용 (|0⟩는 불변)
              </p>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <code className="text-sm">
                  Z = [1 &nbsp;0]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;[0 -1]
                </code>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-xl p-6">
              <h3 className="text-lg font-bold text-purple-700 dark:text-purple-400 mb-4">🎲 Hadamard 게이트</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-4">
                중첩 상태를 생성하는 핵심 게이트
              </p>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <code className="text-sm">
                  H = 1/√2 [1 &nbsp;1]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1 -1]
                </code>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}