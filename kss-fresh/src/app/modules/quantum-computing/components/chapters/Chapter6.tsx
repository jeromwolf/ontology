'use client';

import { Brain } from 'lucide-react';
import References from '@/components/common/References';

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

      <References
        sections={[
          {
            title: 'NISQ Era Foundations',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Quantum Computing in the NISQ era and beyond',
                authors: 'John Preskill',
                year: '2018',
                description: 'NISQ 시대를 정의한 논문 - 양자우위 달성 전략 (Quantum)',
                link: 'https://arxiv.org/abs/1801.00862'
              },
              {
                title: 'Variational Quantum Eigensolver: A review',
                authors: 'Alberto Peruzzo, et al.',
                year: '2014',
                description: 'VQE 최초 제안 - 화학 시뮬레이션 응용 (Nature Communications)',
                link: 'https://www.nature.com/articles/ncomms5213'
              },
              {
                title: 'A variational eigenvalue solver on a photonic quantum processor',
                authors: 'Alberto Peruzzo, Jarrod McClean, Peter Shadbolt, et al.',
                year: '2014',
                description: 'VQE 실험 실증 - 광자 기반 (Nature Communications)',
                link: 'https://www.nature.com/articles/ncomms5213'
              }
            ]
          },
          {
            title: 'Variational Quantum Algorithms',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'The theory of variational hybrid quantum-classical algorithms',
                authors: 'Jarrod R. McClean, Jonathan Romero, Ryan Babbush, Alán Aspuru-Guzik',
                year: '2016',
                description: 'VQA 이론적 기초 확립 (New Journal of Physics)',
                link: 'https://arxiv.org/abs/1509.04279'
              },
              {
                title: 'Quantum Approximate Optimization Algorithm',
                authors: 'Edward Farhi, Jeffrey Goldstone, Sam Gutmann',
                year: '2014',
                description: 'QAOA 최초 제안 - 조합최적화 문제 (arXiv)',
                link: 'https://arxiv.org/abs/1411.4028'
              },
              {
                title: 'Variational quantum algorithms',
                authors: 'M. Cerezo, Andrew Arrasmith, Ryan Babbush, et al.',
                year: '2021',
                description: 'VQA 종합 리뷰 논문 (Nature Reviews Physics)',
                link: 'https://www.nature.com/articles/s42254-021-00348-9'
              }
            ]
          },
          {
            title: 'Quantum Machine Learning',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Quantum Machine Learning in Feature Hilbert Spaces',
                authors: 'Maria Schuld, Nathan Killoran',
                year: '2019',
                description: '양자 커널 방법론 - 머신러닝 응용 (Physical Review Letters)',
                link: 'https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504'
              },
              {
                title: 'Supervised learning with quantum-enhanced feature spaces',
                authors: 'Vojtech Havlicek, et al.',
                year: '2019',
                description: 'IBM의 양자 머신러닝 실험 (Nature)',
                link: 'https://www.nature.com/articles/s41586-019-0980-2'
              },
              {
                title: 'Quantum advantage in learning from experiments',
                authors: 'Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, et al.',
                year: '2022',
                description: '양자 머신러닝의 실험적 우위 증명 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.abn7293'
              }
            ]
          },
          {
            title: 'Practical Implementations & Tools',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'PennyLane: Automatic differentiation of hybrid quantum-classical computations',
                description: 'PennyLane - 변분 양자 알고리즘 프레임워크',
                link: 'https://pennylane.ai/'
              },
              {
                title: 'Qiskit: Quantum Machine Learning Tutorials',
                description: 'IBM Qiskit의 NISQ 알고리즘 실습 자료',
                link: 'https://qiskit.org/textbook/ch-applications/vqe-molecules.html'
              },
              {
                title: 'TensorFlow Quantum',
                description: 'Google의 양자-고전 하이브리드 머신러닝 라이브러리',
                link: 'https://www.tensorflow.org/quantum'
              },
              {
                title: 'Cirq: QAOA Tutorial',
                description: 'Google Cirq의 QAOA 구현 가이드',
                link: 'https://quantumai.google/cirq/experiments/qaoa'
              }
            ]
          }
        ]}
      />
    </div>
  )
}