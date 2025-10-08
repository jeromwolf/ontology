'use client';

import { Eye } from 'lucide-react';
import References from '@/components/common/References';

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

      <References
        sections={[
          {
            title: 'Superconducting Qubits',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Charge-insensitive qubit design derived from the Cooper pair box',
                authors: 'Jens Koch, Terri M. Yu, Jay Gambetta, et al.',
                year: '2007',
                description: 'Transmon 큐비트 개발 - IBM/Google의 기반 기술 (Physical Review A)',
                link: 'https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.042319'
              },
              {
                title: 'Superconducting quantum bits',
                authors: 'John Clarke, Frank K. Wilhelm',
                year: '2008',
                description: '초전도 큐비트 종합 리뷰 (Nature)',
                link: 'https://www.nature.com/articles/nature07128'
              },
              {
                title: 'Building logical qubits in a superconducting quantum computing system',
                authors: 'J. M. Gambetta, et al. (IBM Quantum)',
                year: '2017',
                description: 'IBM의 논리 큐비트 구축 전략 (npj Quantum Information)',
                link: 'https://www.nature.com/articles/s41534-016-0004-0'
              }
            ]
          },
          {
            title: 'Trapped Ion Systems',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Experimental Issues in Coherent Quantum-State Manipulation of Trapped Atomic Ions',
                authors: 'D. J. Wineland, et al.',
                year: '1998',
                description: '이온 트랩 양자 컴퓨팅의 실험적 기초 (Journal of Research of NIST)',
                link: 'https://nvlpubs.nist.gov/nistpubs/jres/103/jresv103n3p259_A1b.pdf'
              },
              {
                title: 'High-Fidelity Quantum Logic Gates Using Trapped-Ion Hyperfine Qubits',
                authors: 'J. P. Gaebler, et al.',
                year: '2016',
                description: '99.9% 게이트 충실도 달성 (Physical Review Letters)',
                link: 'https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.060505'
              },
              {
                title: 'Quantum supremacy using a programmable superconducting processor',
                authors: 'Honeywell Quantum Solutions',
                year: '2020',
                description: 'IonQ/Honeywell의 이온 트랩 양자 컴퓨터',
                link: 'https://www.honeywell.com/us/en/company/quantum'
              }
            ]
          },
          {
            title: 'Photonic & Silicon Qubits',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Quantum computational advantage using photons',
                authors: 'Han-Sen Zhong, et al.',
                year: '2020',
                description: '중국의 광자 기반 양자 우위 달성 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.abe8770'
              },
              {
                title: 'Silicon quantum electronics',
                authors: 'F. A. Zwanenburg, et al.',
                year: '2013',
                description: '실리콘 기반 양자 컴퓨팅 리뷰 (Reviews of Modern Physics)',
                link: 'https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.85.961'
              }
            ]
          },
          {
            title: 'Platform Comparisons & Future',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'IBM Quantum Roadmap',
                description: 'IBM의 1000+ 큐비트 시스템 로드맵',
                link: 'https://www.ibm.com/quantum/roadmap'
              },
              {
                title: 'Google Quantum AI',
                description: 'Google의 양자 하드웨어 및 알고리즘 연구',
                link: 'https://quantumai.google/'
              },
              {
                title: 'IonQ Technology',
                description: '상업용 이온 트랩 양자 컴퓨터',
                link: 'https://ionq.com/technology'
              }
            ]
          }
        ]}
      />
    </div>
  )
}