'use client';

import { Zap } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter3() {
  return (
    <div className="p-8 space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Zap className="w-8 h-8 text-purple-600" />
          Deutsch-Jozsa 알고리즘
        </h2>
        
        <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🎯 문제 정의</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            블랙박스 함수 f: {'{'}0,1{'}'} ⁿ → {'{'}0,1{'}'}이 주어졌을 때, 이 함수가 상수 함수인지 균형 함수인지 판별하는 문제입니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">상수 함수</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">모든 입력에 대해 같은 값 출력 (0 또는 1)</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">균형 함수</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">절반은 0, 절반은 1을 출력</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">📊 성능 비교</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-300 dark:border-gray-600">
                  <th className="text-left p-3">알고리즘</th>
                  <th className="text-left p-3">함수 호출 횟수</th>
                  <th className="text-left p-3">복잡도</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 text-red-600 dark:text-red-400">고전 최악의 경우</td>
                  <td className="p-3">2ⁿ⁻¹ + 1</td>
                  <td className="p-3">O(2ⁿ)</td>
                </tr>
                <tr>
                  <td className="p-3 text-green-600 dark:text-green-400">양자 알고리즘</td>
                  <td className="p-3">1</td>
                  <td className="p-3">O(1)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🔍 Grover 탐색 알고리즘</h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🎯 탐색 문제</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              N개의 항목 중에서 특정 조건을 만족하는 M개의 항목을 찾는 문제입니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">고전 탐색</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 평균: N/2 번의 검사</li>
                  <li>• 최악: N 번의 검사</li>
                  <li>• 복잡도: O(N)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">양자 탐색</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• π√(N/M)/4 번의 반복</li>
                  <li>• 확률적 성공 (99%+ 가능)</li>
                  <li>• 복잡도: O(√N)</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🔄 Grover 반복 과정</h3>
            <div className="space-y-4">
              <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-purple-100 to-indigo-100 dark:from-purple-900/30 dark:to-indigo-900/30 rounded-lg">
                <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</div>
                <div>
                  <h4 className="font-semibold">초기화</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">모든 상태의 균등한 중첩 생성 (Hadamard 게이트)</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 rounded-lg">
                <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</div>
                <div>
                  <h4 className="font-semibold">오라클 적용</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">정답 상태의 위상을 반전 (|x⟩ → -|x⟩)</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-blue-100 to-cyan-100 dark:from-blue-900/30 dark:to-cyan-900/30 rounded-lg">
                <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</div>
                <div>
                  <h4 className="font-semibold">확산 연산</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">평균에 대한 반사로 진폭 증폭</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-yellow-100 to-orange-100 dark:from-yellow-900/30 dark:to-orange-900/30 rounded-lg">
                <div className="w-8 h-8 bg-yellow-500 text-white rounded-full flex items-center justify-center text-sm font-bold">4</div>
                <div>
                  <h4 className="font-semibold">반복</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">최적 횟수만큼 2-3단계 반복 후 측정</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">⚡ 양자 우위 (Quantum Advantage)</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-green-700 dark:text-green-400 mb-4">📈 성능 향상</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>Deutsch-Jozsa:</strong> 지수적 가속</li>
              <li>• <strong>Grover:</strong> 이차적 가속</li>
              <li>• <strong>Shor:</strong> 지수적 가속 (특정 문제)</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-orange-700 dark:text-orange-400 mb-4">⚠️ 한계점</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 모든 문제에 적용되지 않음</li>
              <li>• 확률적 결과 (반복 필요)</li>
              <li>• 노이즈와 오류에 민감</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Original Algorithm Papers',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Rapid solution of problems by quantum computation',
                authors: 'David Deutsch, Richard Jozsa',
                year: '1992',
                description: 'Deutsch-Jozsa 알고리즘 원본 논문 - 첫 양자 우위 증명 (Proc. R. Soc. Lond. A)',
                link: 'https://royalsocietypublishing.org/doi/10.1098/rspa.1992.0167'
              },
              {
                title: 'A fast quantum mechanical algorithm for database search',
                authors: 'Lov K. Grover',
                year: '1996',
                description: 'Grover 알고리즘 원본 논문 - O(√N) 탐색 알고리즘 (STOC 1996)',
                link: 'https://arxiv.org/abs/quant-ph/9605043'
              },
              {
                title: 'Quantum Algorithms for Quantum Field Theories',
                authors: 'Stephen P. Jordan, Keith S. M. Lee, John Preskill',
                year: '2012',
                description: '양자 알고리즘의 물리학 응용 (Science)',
                link: 'https://arxiv.org/abs/1111.3633'
              }
            ]
          },
          {
            title: 'Grover Algorithm Analysis',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Tight bounds on quantum searching',
                authors: 'Michel Boyer, Gilles Brassard, Peter Høyer, Alain Tapp',
                year: '1998',
                description: 'Grover 알고리즘의 최적성 증명 (Fortschritte der Physik)',
                link: 'https://arxiv.org/abs/quant-ph/9605034'
              },
              {
                title: 'Quantum amplitude amplification and estimation',
                authors: 'Gilles Brassard, Peter Høyer, Michele Mosca, Alain Tapp',
                year: '2002',
                description: '진폭 증폭의 일반화 이론 (Contemporary Mathematics)',
                link: 'https://arxiv.org/abs/quant-ph/0005055'
              },
              {
                title: 'Optimal Quantum Adversary Lower Bounds for Ordered Search',
                authors: 'Andrew M. Childs, Troy Lee',
                year: '2008',
                description: '양자 탐색의 하한선 증명 (ICALP)',
                link: 'https://arxiv.org/abs/0708.3396'
              }
            ]
          },
          {
            title: 'Quantum Advantage & Supremacy',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Quantum supremacy using a programmable superconducting processor',
                authors: 'Frank Arute, et al. (Google AI Quantum)',
                year: '2019',
                description: 'Google의 양자 우위 달성 (Nature)',
                link: 'https://www.nature.com/articles/s41586-019-1666-5'
              },
              {
                title: 'Quantum computational advantage using photons',
                authors: 'Han-Sen Zhong, et al.',
                year: '2020',
                description: '중국의 양자 우위 달성 - 광자 기반 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.abe8770'
              },
              {
                title: 'Quantum advantage with shallow circuits',
                authors: 'Sergey Bravyi, David Gosset, Robert König',
                year: '2018',
                description: '얕은 회로로도 양자 우위 가능 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.aar3106'
              }
            ]
          },
          {
            title: 'Learning Resources',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Qiskit: Deutsch-Jozsa Tutorial',
                description: 'IBM Qiskit에서 Deutsch-Jozsa 알고리즘 실습',
                link: 'https://qiskit.org/textbook/ch-algorithms/deutsch-jozsa.html'
              },
              {
                title: 'Qiskit: Grover\'s Algorithm',
                description: 'Grover 알고리즘 단계별 구현 가이드',
                link: 'https://qiskit.org/textbook/ch-algorithms/grover.html'
              },
              {
                title: 'Quantum Algorithm Zoo',
                description: 'Stephen Jordan의 양자 알고리즘 종합 데이터베이스',
                link: 'https://quantumalgorithmzoo.org/'
              },
              {
                title: 'Microsoft Q#: Grover Implementation',
                description: 'Q#로 작성된 Grover 알고리즘 샘플',
                link: 'https://github.com/microsoft/QuantumKatas/tree/main/GroversAlgorithm'
              }
            ]
          }
        ]}
      />
    </div>
  )
}