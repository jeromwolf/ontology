'use client';

import { Calculator, GitBranch, Repeat, Layers, BookOpen } from 'lucide-react';
import References from '@/components/common/References';

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

      {/* Multi-Qubit Gates */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <GitBranch className="w-8 h-8 text-purple-600" />
          다중 큐비트 게이트
        </h2>

        <div className="space-y-6">
          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-cyan-700 dark:text-cyan-400 mb-4">🔗 CNOT (Controlled-NOT)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              양자 얽힘을 생성하는 가장 중요한 2큐비트 게이트입니다. 제어 큐비트가 |1⟩일 때만 타겟 큐비트에 X 게이트를 적용합니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">행렬 표현 (4×4)</h4>
                <code className="text-xs">
                  CNOT = [1 0 0 0]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0 1 0 0]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0 0 0 1]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0 0 1 0]
                </code>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Bell 상태 생성</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  H ⊗ I → CNOT → (|00⟩ + |11⟩)/√2
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                  최대 얽힘 상태 (maximal entanglement)
                </p>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
              <h3 className="text-lg font-bold text-indigo-700 dark:text-indigo-400 mb-4">⚙️ Toffoli 게이트 (CCNOT)</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                3큐비트 게이트로, 두 제어 큐비트가 모두 |1⟩일 때만 타겟에 X 적용. 고전 AND 게이트를 구현하며 양자 컴퓨팅의 보편성을 증명합니다.
              </p>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <code className="text-xs">
                  |a⟩|b⟩|c⟩ → |a⟩|b⟩|c ⊕ (a·b)⟩
                </code>
              </div>
            </div>

            <div className="bg-gradient-to-br from-pink-50 to-rose-50 dark:from-pink-900/20 dark:to-rose-900/20 rounded-xl p-6">
              <h3 className="text-lg font-bold text-pink-700 dark:text-pink-400 mb-4">🔄 SWAP 게이트</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                두 큐비트의 상태를 교환합니다. CNOT 게이트 3개의 조합으로 구현 가능합니다.
              </p>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <code className="text-xs">
                  SWAP = [1 0 0 0]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0 0 1 0]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0 1 0 0]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0 0 0 1]
                </code>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Rotation Gates */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Repeat className="w-8 h-8 text-purple-600" />
          회전 게이트 (Parametric Gates)
        </h2>

        <div className="bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-6">
            블로흐 구면의 각 축을 중심으로 임의의 각도 θ만큼 회전하는 게이트입니다.
            변분 양자 알고리즘(VQE, QAOA)에서 매개변수로 사용됩니다.
          </p>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-red-600 dark:text-red-400 mb-3">Rx(θ) - X축 회전</h4>
              <code className="text-xs block mb-2">
                Rx(θ) = [cos(θ/2) &nbsp;&nbsp;&nbsp;-i·sin(θ/2)]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-i·sin(θ/2) &nbsp;cos(θ/2)]
              </code>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Bloch sphere X-axis rotation
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-3">Ry(θ) - Y축 회전</h4>
              <code className="text-xs block mb-2">
                Ry(θ) = [cos(θ/2) &nbsp;-sin(θ/2)]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[sin(θ/2) &nbsp;&nbsp;cos(θ/2)]
              </code>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                실수 행렬, 주로 상태 준비에 사용
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">Rz(θ) - Z축 회전</h4>
              <code className="text-xs block mb-2">
                Rz(θ) = [e^(-iθ/2) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e^(iθ/2)]
              </code>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                위상 회전, 대각 행렬
              </p>
            </div>
          </div>

          <div className="mt-6 bg-blue-100 dark:bg-blue-900/30 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">💡 임의의 단일 큐비트 회전</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              임의의 SU(2) 회전은 3개의 회전 게이트 조합으로 표현 가능합니다:
            </p>
            <code className="text-sm block mt-2 text-blue-700 dark:text-blue-300">
              U(θ, φ, λ) = Rz(φ) · Ry(θ) · Rz(λ)
            </code>
          </div>
        </div>
      </section>

      {/* Phase Gates */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Layers className="w-8 h-8 text-purple-600" />
          위상 게이트 (Phase Gates)
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-violet-700 dark:text-violet-400 mb-4">🔸 S 게이트 (Phase gate)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              |1⟩ 상태에 +i 위상을 적용합니다. Z 게이트의 제곱근입니다 (S² = Z).
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-3 mb-3">
              <code className="text-xs">
                S = [1 &nbsp;0]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[0 &nbsp;i]
              </code>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              S = Rz(π/2) = √Z
            </p>
          </div>

          <div className="bg-gradient-to-br from-fuchsia-50 to-pink-50 dark:from-fuchsia-900/20 dark:to-pink-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-fuchsia-700 dark:text-fuchsia-400 mb-4">🔹 T 게이트 (π/8 gate)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              |1⟩ 상태에 e^(iπ/4) 위상을 적용합니다. S 게이트의 제곱근입니다 (T² = S).
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-3 mb-3">
              <code className="text-xs">
                T = [1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[0 &nbsp;e^(iπ/4)]
              </code>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              T = Rz(π/4) = √S = ∜Z
            </p>
          </div>
        </div>

        <div className="mt-6 bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
          <h3 className="text-lg font-bold text-purple-700 dark:text-purple-400 mb-4">🎯 범용 게이트 집합 (Universal Gate Sets)</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            다음 게이트 조합으로 임의의 양자 연산을 근사할 수 있습니다:
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Clifford + T</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {'{H, S, CNOT, T}'} - 오류 정정에 최적화
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Rotation + CNOT</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {'{Rx, Ry, Rz, CNOT}'} - NISQ 장치에서 사용
              </p>
            </div>
          </div>
          <div className="mt-4 p-4 bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 rounded-lg">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>Solovay-Kitaev 정리:</strong> 범용 게이트 집합으로 임의의 단일 큐비트 게이트를
              O(log^c(1/ε)) 개의 게이트로 ε-정확도로 근사할 수 있습니다 (c ≈ 2).
            </p>
          </div>
        </div>
      </section>

      {/* Circuit Composition */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="w-8 h-8 text-purple-600" />
          양자 회로 구성
        </h2>

        <div className="bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-xl p-6">
          <h3 className="text-lg font-bold text-teal-700 dark:text-teal-400 mb-4">🔧 회로 설계 원칙</h3>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">1. 상태 준비 (State Preparation)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                초기 |0⟩ 상태에서 원하는 중첩 상태를 생성합니다.
              </p>
              <code className="text-xs text-teal-600 dark:text-teal-400">
                Ry(θ) |0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
              </code>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">2. 얽힘 생성 (Entanglement)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                CNOT, CZ 등으로 다중 큐비트 상관관계를 생성합니다.
              </p>
              <code className="text-xs text-teal-600 dark:text-teal-400">
                H(q0) → CNOT(q0, q1) → (|00⟩ + |11⟩)/√2
              </code>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">3. 연산 적용 (Computation)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                문제에 맞는 단일/다중 큐비트 게이트를 적용합니다.
              </p>
              <code className="text-xs text-teal-600 dark:text-teal-400">
                Oracle: U_f |x⟩|y⟩ → |x⟩|y ⊕ f(x)⟩
              </code>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">4. 측정 (Measurement)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                계산 기저 또는 다른 기저에서 측정하여 결과를 얻습니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Quantum Gate Textbooks',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'Quantum Computation and Quantum Information',
                authors: 'Michael A. Nielsen, Isaac L. Chuang',
                year: '2010',
                description: 'Chapter 4: Quantum Circuits - 양자 게이트와 회로의 완벽한 설명',
                link: 'http://mmrc.amss.cas.cn/tlb/201702/W020170224608149940643.pdf'
              },
              {
                title: 'Quantum Computing: A Gentle Introduction',
                authors: 'Eleanor Rieffel, Wolfgang Polak',
                year: '2011',
                description: 'Chapter 3-4: 양자 게이트와 회로 구성의 단계별 설명',
                link: 'https://mitpress.mit.edu/9780262526678/'
              },
              {
                title: 'An Introduction to Quantum Computing',
                authors: 'Phillip Kaye, Raymond Laflamme, Michele Mosca',
                year: '2007',
                description: '양자 게이트와 알고리즘의 수학적 기초',
                link: 'https://www.amazon.com/Introduction-Quantum-Computing-Phillip-Kaye/dp/019857049X'
              }
            ]
          },
          {
            title: 'Universal Gate Sets & Theory',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Universal Quantum Gates',
                authors: 'Adriano Barenco, Charles H. Bennett, Richard Cleve, et al.',
                year: '1995',
                description: '범용 양자 게이트 집합의 이론적 기초 (Physical Review A)',
                link: 'https://journals.aps.org/pra/abstract/10.1103/PhysRevA.52.3457'
              },
              {
                title: 'Solovay-Kitaev Theorem',
                authors: 'Christopher M. Dawson, Michael A. Nielsen',
                year: '2006',
                description: '효율적인 양자 게이트 근사 컴파일 알고리즘',
                link: 'https://arxiv.org/abs/quant-ph/0505030'
              },
              {
                title: 'Elementary gates for quantum computation',
                authors: 'Adriano Barenco',
                year: '1995',
                description: 'CNOT와 단일 큐비트 게이트로 임의 연산 구현 (Physical Review A)',
                link: 'https://journals.aps.org/pra/abstract/10.1103/PhysRevA.52.3457'
              },
              {
                title: 'A fast quantum mechanical algorithm for database search',
                authors: 'Lov K. Grover',
                year: '1996',
                description: 'Grover 알고리즘과 오라클 게이트 (STOC 1996)',
                link: 'https://arxiv.org/abs/quant-ph/9605043'
              }
            ]
          },
          {
            title: 'Implementation & Hardware',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Superconducting Quantum Bits',
                authors: 'Jens Koch, Terri M. Yu, Jay Gambetta, et al.',
                year: '2007',
                description: 'Transmon 큐비트와 게이트 구현 (Physical Review A)',
                link: 'https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.042319'
              },
              {
                title: 'High-Fidelity Quantum Logic Gates Using Trapped-Ion Hyperfine Qubits',
                authors: 'J. P. Gaebler, T. R. Tan, Y. Lin, et al.',
                year: '2016',
                description: '이온 트랩 큐비트에서 99.9% 게이트 충실도 달성 (Physical Review Letters)',
                link: 'https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.060505'
              },
              {
                title: 'Silicon quantum electronics',
                authors: 'F. A. Zwanenburg, A. S. Dzurak, A. Morello, et al.',
                year: '2013',
                description: '실리콘 기반 양자 게이트 구현 리뷰 (Reviews of Modern Physics)',
                link: 'https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.85.961'
              }
            ]
          },
          {
            title: 'Online Learning & Tools',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Qiskit: Quantum Gates and Circuits',
                description: 'IBM Qiskit에서 양자 게이트 실습 튜토리얼',
                link: 'https://qiskit.org/textbook/ch-gates/introduction.html'
              },
              {
                title: 'Cirq: Google Quantum Gates',
                description: 'Google Cirq 프레임워크의 게이트 라이브러리',
                link: 'https://quantumai.google/cirq/gates'
              },
              {
                title: 'Microsoft Q# Quantum Gates',
                description: 'Q# 언어의 게이트 연산과 회로 구성',
                link: 'https://docs.microsoft.com/en-us/azure/quantum/user-guide/'
              },
              {
                title: 'Quirk: Quantum Circuit Simulator',
                description: '드래그 앤 드롭으로 양자 회로를 시각화하는 웹 도구',
                link: 'https://algassert.com/quirk'
              },
              {
                title: 'Q-CTRL: Quantum Control Insights',
                description: '양자 게이트 최적화와 오류 억제 기술',
                link: 'https://q-ctrl.com/resources'
              }
            ]
          }
        ]}
      />
    </div>
  )
}