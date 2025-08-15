'use client';

import { Brain } from 'lucide-react';

export default function Chapter6() {
  return (
    <div className="p-8 space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Brain className="w-8 h-8 text-purple-600" />
          NISQ 시대의 양자 컴퓨팅
        </h2>
        
        <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🔬 NISQ (Noisy Intermediate-Scale Quantum)</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            현재와 가까운 미래의 양자 컴퓨터는 완벽한 오류 정정 없이 50-1000 큐비트 규모로 작동합니다.
            이 시대에 실용적 양자 우위를 달성하기 위한 알고리즘들이 개발되고 있습니다.
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">제한된 큐비트</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">50-1000 큐비트</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">높은 오류율</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">0.1-1% 게이트 오류</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">얕은 회로</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">100-1000 게이트 깊이</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">⚗️ Variational Quantum Eigensolver (VQE)</h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🎯 목표: 바닥 상태 에너지 계산</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              VQE는 분자의 바닥 상태 에너지를 찾는 하이브리드 양자-고전 알고리즘입니다.
              화학, 재료과학, 신약 개발에 혁명적 응용이 기대됩니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">알고리즘 구조:</h4>
              <code className="text-sm">
                E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩<br/>
                |ψ(θ)⟩ = U(θ)|0...0⟩<br/>
                θ* = argmin E(θ)
              </code>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🎯 QAOA (Quantum Approximate Optimization Algorithm)</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-green-700 dark:text-green-400 mb-4">🧩 조합 최적화 문제</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              QAOA는 MaxCut, 여행자 문제 등 NP-hard 조합 최적화 문제에 대한 근사해를 찾습니다.
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 포트폴리오 최적화</li>
              <li>• 교통 라우팅</li>
              <li>• 스케줄링 문제</li>
              <li>• 네트워크 분할</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-orange-700 dark:text-orange-400 mb-4">⚙️ 알고리즘 구조</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              p-layer QAOA는 비용 해밀토니안과 믹서 해밀토니안을 번갈아 적용합니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <code className="text-sm">
                |ψ(β,γ)⟩ = ∏ⱼ U_B(βⱼ)U_C(γⱼ)|+⟩⊗ⁿ
              </code>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🧠 Quantum Neural Networks</h2>
        
        <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🔗 PennyLane 프레임워크</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            PennyLane은 양자 머신러닝을 위한 파이썬 라이브러리로, 양자-고전 하이브리드 계산을 쉽게 구현할 수 있습니다.
          </p>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2">기본 QML 구조:</h4>
            <pre className="text-sm overflow-x-auto"><code>{`import pennylane as qml
import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(params, x):
    qml.AngleEmbedding(x, wires=range(2))
    qml.BasicEntanglerLayers(params, wires=range(2))
    return qml.expval(qml.PauliZ(0))

# 훈련 루프
for i in range(100):
    params = optimizer.step(cost, params)`}</code></pre>
          </div>
        </div>
      </section>
    </div>
  )
}