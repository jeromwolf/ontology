'use client';

import { Atom, BookOpen, Activity, Waves } from 'lucide-react';
import References from '@/components/common/References';

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

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="w-8 h-8 text-purple-600" />
          큐비트 vs 고전 비트
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-700">
                <th className="p-4 text-left font-semibold">특성</th>
                <th className="p-4 text-left font-semibold text-red-600 dark:text-red-400">고전 비트</th>
                <th className="p-4 text-left font-semibold text-purple-600 dark:text-purple-400">큐비트</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-gray-200 dark:border-gray-700">
                <td className="p-4 font-medium">상태</td>
                <td className="p-4">0 또는 1 (확정적)</td>
                <td className="p-4">α|0⟩ + β|1⟩ (중첩)</td>
              </tr>
              <tr className="border-t border-gray-200 dark:border-gray-700">
                <td className="p-4 font-medium">정보량</td>
                <td className="p-4">1 bit</td>
                <td className="p-4">2개의 복소수 (α, β)</td>
              </tr>
              <tr className="border-t border-gray-200 dark:border-gray-700">
                <td className="p-4 font-medium">측정</td>
                <td className="p-4">항상 같은 값</td>
                <td className="p-4">확률적 (|α|², |β|²)</td>
              </tr>
              <tr className="border-t border-gray-200 dark:border-gray-700">
                <td className="p-4 font-medium">복사</td>
                <td className="p-4">가능 (임의 복사)</td>
                <td className="p-4">불가능 (No-cloning)</td>
              </tr>
              <tr className="border-t border-gray-200 dark:border-gray-700">
                <td className="p-4 font-medium">병렬성</td>
                <td className="p-4">순차 처리</td>
                <td className="p-4">양자 병렬성 (2ⁿ 상태)</td>
              </tr>
              <tr className="border-t border-gray-200 dark:border-gray-700">
                <td className="p-4 font-medium">얽힘</td>
                <td className="p-4">독립적</td>
                <td className="p-4">얽힘 가능 (비국소 상관)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="w-8 h-8 text-purple-600" />
          Dirac 표기법 (Bra-Ket Notation)
        </h2>

        <div className="space-y-6">
          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">📐 기본 표기</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Ket 벡터 |ψ⟩</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                  양자 상태를 나타내는 열벡터 (column vector)
                </p>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <code className="text-sm">
                    |0⟩ = [1]<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0]<br/><br/>
                    |1⟩ = [0]<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1]
                  </code>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">Bra 벡터 ⟨ψ|</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                  Ket의 켤레 전치 (conjugate transpose)
                </p>
                <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
                  <code className="text-sm">
                    ⟨0| = [1  0]<br/><br/>
                    ⟨1| = [0  1]
                  </code>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🔢 주요 연산</h3>
            <div className="space-y-4">
              <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
                <h4 className="font-semibold mb-2">내적 (Inner Product) ⟨φ|ψ⟩</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  두 상태의 겹침 정도를 나타내는 복소수
                </p>
                <code className="text-sm bg-white dark:bg-gray-800 px-3 py-1 rounded">
                  ⟨0|0⟩ = 1, ⟨0|1⟩ = 0 (직교 상태)
                </code>
              </div>

              <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
                <h4 className="font-semibold mb-2">외적 (Outer Product) |ψ⟩⟨φ|</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  연산자(행렬)를 생성
                </p>
                <code className="text-sm bg-white dark:bg-gray-800 px-3 py-1 rounded">
                  |0⟩⟨0| = [1 0; 0 0] (투사 연산자)
                </code>
              </div>

              <div className="p-4 bg-gradient-to-r from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-lg">
                <h4 className="font-semibold mb-2">노름 (Norm) ⟨ψ|ψ⟩</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  상태의 크기 (항상 1로 정규화)
                </p>
                <code className="text-sm bg-white dark:bg-gray-800 px-3 py-1 rounded">
                  ⟨ψ|ψ⟩ = |α|² + |β|² = 1
                </code>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Activity className="w-8 h-8 text-purple-600" />
          양자 측정과 상태 붕괴
        </h2>

        <div className="space-y-6">
          <div className="bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">📏 측정 공준 (Measurement Postulate)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              큐비트를 Z 기저로 측정하면:
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">결과: |0⟩</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">확률: |α|²</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">측정 후 상태: |0⟩</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">결과: |1⟩</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">확률: |β|²</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">측정 후 상태: |1⟩</p>
              </div>
            </div>
          </div>

          <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-400 p-6 rounded-r-lg">
            <h4 className="font-bold text-red-800 dark:text-red-400 mb-3">⚠️ 측정의 파괴적 특성</h4>
            <ul className="space-y-2 text-red-700 dark:text-red-300">
              <li>• <strong>비가역성:</strong> 측정 전 중첩 상태를 복원할 수 없음</li>
              <li>• <strong>정보 손실:</strong> α, β의 위상 정보가 사라짐</li>
              <li>• <strong>No-cloning:</strong> 측정 전 양자 상태를 복사할 수 없음</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🎲 다른 기저에서의 측정</h3>
            <div className="space-y-3">
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-purple-600 dark:text-purple-400">X 기저 측정:</strong>
                <code className="ml-2 text-sm">|+⟩ = (|0⟩+|1⟩)/√2, |-⟩ = (|0⟩-|1⟩)/√2</code>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-blue-600 dark:text-blue-400">Y 기저 측정:</strong>
                <code className="ml-2 text-sm">|+i⟩ = (|0⟩+i|1⟩)/√2, |-i⟩ = (|0⟩-i|1⟩)/√2</code>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Waves className="w-8 h-8 text-purple-600" />
          노이즈와 디코히어런스
        </h2>

        <div className="space-y-6">
          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">⚡ 양자 컴퓨팅의 최대 도전과제</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              큐비트는 외부 환경과의 상호작용으로 인해 양자 정보가 손실되는 <strong>디코히어런스(decoherence)</strong> 현상을 겪습니다.
            </p>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">T₁ 시간 (에너지 완화)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  |1⟩ → |0⟩ 자발적 붕괴 시간
                </p>
                <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                  <li>• 초전도 큐비트: ~100 μs</li>
                  <li>• 이온 트랩: ~1 s</li>
                  <li>• 광자: 무한대</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-red-600 dark:text-red-400 mb-3">T₂ 시간 (위상 완화)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  중첩 상태의 위상 정보 손실 시간
                </p>
                <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                  <li>• 항상 T₂ ≤ 2T₁</li>
                  <li>• 게이트 시간 &lt;&lt; T₂ 필요</li>
                  <li>• 오류 정정 필수</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🛡️ 노이즈 완화 전략</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-gradient-to-b from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
                <div className="text-3xl mb-2">❄️</div>
                <h4 className="font-semibold mb-2">극저온 냉각</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  ~10 mK로 열 잡음 최소화
                </p>
              </div>

              <div className="text-center p-4 bg-gradient-to-b from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
                <div className="text-3xl mb-2">🔒</div>
                <h4 className="font-semibold mb-2">차폐</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  전자기 잡음 차단
                </p>
              </div>

              <div className="text-center p-4 bg-gradient-to-b from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-lg">
                <div className="text-3xl mb-2">🔧</div>
                <h4 className="font-semibold mb-2">오류 정정</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  다중 물리 큐비트로 논리 큐비트 구성
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Foundational Textbooks',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'Quantum Computation and Quantum Information',
                authors: 'Michael A. Nielsen, Isaac L. Chuang',
                year: '2010',
                description: '양자 컴퓨팅의 바이블 - 큐비트와 양자 게이트의 완벽한 설명',
                link: 'http://mmrc.amss.cas.cn/tlb/201702/W020170224608149940643.pdf'
              },
              {
                title: 'Quantum Computing: A Gentle Introduction',
                authors: 'Eleanor Rieffel, Wolfgang Polak',
                year: '2011',
                description: '양자역학 기초부터 차근차근 설명하는 입문서 (MIT Press)',
                link: 'https://mitpress.mit.edu/9780262526678/'
              },
              {
                title: 'Quantum Computing for Computer Scientists',
                authors: 'Noson S. Yanofsky, Mirco A. Mannucci',
                year: '2008',
                description: '컴퓨터 과학자를 위한 양자 컴퓨팅 (Cambridge)',
                link: 'https://www.cambridge.org/core/books/quantum-computing-for-computer-scientists/8AEA723BEE5CC9F5C03FDD4BA850C711'
              }
            ]
          },
          {
            title: 'Quantum Mechanics Foundations',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'The Feynman Lectures on Physics Vol. III',
                authors: 'Richard P. Feynman',
                year: '1965',
                description: '양자역학의 고전 - Feynman의 명강의',
                link: 'https://www.feynmanlectures.caltech.edu/III_toc.html'
              },
              {
                title: 'Principles of Quantum Mechanics',
                authors: 'P.A.M. Dirac',
                year: '1930',
                description: 'Dirac 표기법을 창시한 역사적 교과서',
                link: 'https://en.wikipedia.org/wiki/The_Principles_of_Quantum_Mechanics'
              },
              {
                title: 'Quantum Mechanics and Path Integrals',
                authors: 'Richard P. Feynman, Albert R. Hibbs',
                year: '1965',
                description: 'Feynman 경로 적분 접근법',
                link: 'https://store.doverpublications.com/0486477223.html'
              }
            ]
          },
          {
            title: 'Online Learning Resources',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Qiskit Textbook: Introduction to Quantum Computing',
                description: 'IBM의 무료 양자 컴퓨팅 교재 - 큐비트부터 실습까지',
                link: 'https://qiskit.org/learn/intro-qc-qh'
              },
              {
                title: 'Quantum Computing for the Very Curious',
                authors: 'Andy Matuschak, Michael Nielsen',
                description: '인터랙티브 양자 컴퓨팅 에세이',
                link: 'https://quantum.country/qcvc'
              },
              {
                title: 'Microsoft Quantum Documentation',
                description: 'Microsoft Q# 양자 프로그래밍 문서',
                link: 'https://learn.microsoft.com/en-us/azure/quantum/'
              },
              {
                title: 'MIT OpenCourseWare: Quantum Information Science',
                authors: 'Peter Shor, Isaac Chuang',
                description: 'MIT의 양자 정보 이론 강의',
                link: 'https://ocw.mit.edu/courses/8-370x-quantum-information-science-i-spring-2018/'
              }
            ]
          },
          {
            title: 'Research Papers',
            icon: 'paper',
            color: 'border-orange-500',
            items: [
              {
                title: 'Decoherence and the Transition from Quantum to Classical',
                authors: 'Wojciech H. Zurek',
                year: '2003',
                description: '디코히어런스 현상의 종합 리뷰 (Physics Today)',
                link: 'https://arxiv.org/abs/quant-ph/0306072'
              },
              {
                title: 'Quantum Information Processing with Superconducting Qubits',
                authors: 'John Clarke, Frank K. Wilhelm',
                year: '2008',
                description: '초전도 큐비트의 물리적 구현 (Nature)',
                link: 'https://www.nature.com/articles/nature07128'
              },
              {
                title: 'The Physical Implementation of Quantum Computation',
                authors: 'David P. DiVincenzo',
                year: '2000',
                description: 'DiVincenzo 기준 - 양자 컴퓨터 구현 조건',
                link: 'https://arxiv.org/abs/quant-ph/0002077'
              }
            ]
          }
        ]}
      />
    </div>
  )
}