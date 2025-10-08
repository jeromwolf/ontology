'use client';

import { Beaker } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter8() {
  return (
    <div className="p-8 space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Beaker className="w-8 h-8 text-purple-600" />
          양자 시뮬레이션과 분자 모델링
        </h2>
        
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">🧬 신약 개발 혁명</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            양자 컴퓨터는 분자의 양자 특성을 자연스럽게 시뮬레이션할 수 있어, 
            신약 개발과 화학 반응 예측에서 고전 컴퓨터를 뛰어넘는 성능을 보일 것으로 예상됩니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">응용 분야</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 단백질 접힘 예측</li>
                <li>• 효소 촉매 반응</li>
                <li>• 광합성 메커니즘</li>
                <li>• 신약 분자 설계</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">예상 영향</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 신약 개발 기간 단축 (10년→3년)</li>
                <li>• 개발 비용 대폭 절감</li>
                <li>• 개인맞춤형 치료제</li>
                <li>• 희귀질환 치료법 발견</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">💰 양자 금융과 리스크 분석</h2>
        
        <div className="bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">📈 포트폴리오 최적화</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            양자 컴퓨터는 고차원 최적화 문제인 포트폴리오 최적화를 기존보다 빠르고 정확하게 해결할 수 있습니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">양자 알고리즘</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• QAOA 포트폴리오 선택</li>
                <li>• 양자 몬테카를로</li>
                <li>• VQE 리스크 모델링</li>
                <li>• 양자 머신러닝 예측</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">기대 효과</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 실시간 리스크 계산</li>
                <li>• 더 정확한 가격 모델</li>
                <li>• 고주파 거래 최적화</li>
                <li>• 사기 탐지 향상</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🔐 양자 암호학과 양자 인터넷</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-red-700 dark:text-red-400 mb-4">🚨 포스트 양자 암호학</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Shor 알고리즘의 위협에 대비한 새로운 암호 체계 개발이 진행 중입니다.
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 격자 기반 암호학</li>
              <li>• 코드 기반 암호학</li>
              <li>• 다변수 암호학</li>
              <li>• 등원급수 암호학</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-blue-700 dark:text-blue-400 mb-4">🌐 양자 키 분배 (QKD)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              양자역학 법칙에 기반한 이론적으로 완벽한 보안 통신 시스템입니다.
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• BB84 프로토콜</li>
              <li>• 광섬유 기반 QKD</li>
              <li>• 위성 QKD 네트워크</li>
              <li>• 양자 중계기 개발</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">📊 투자 동향과 시장 전망</h2>
        
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">💰 글로벌 투자 현황</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-3">정부 투자 (2024)</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>• 미국: 국가 양자 이니셔티브 ($18억)</li>
                <li>• 중국: 양자 정보 과학 ($150억)</li>
                <li>• EU: Quantum Flagship ($10억)</li>
                <li>• 한국: K-양자 뉴딜 (5천억원)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">민간 투자</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>• 2024년 벤처 투자: $24억</li>
                <li>• IBM, Google, Microsoft 등 빅테크</li>
                <li>• 양자 스타트업 1000+ 개</li>
                <li>• IPO 준비 기업들 다수</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-white dark:bg-gray-800 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">📈 시장 규모 전망</h4>
            <div className="grid md:grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">$13억</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">2024년 현재</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">$50억</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">2030년 예상</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">$1000억</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">2040년 목표</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Quantum Simulation & Chemistry',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Simulated Quantum Computation of Molecular Energies',
                authors: 'Peter J. J. O\'Malley, et al.',
                year: '2016',
                description: 'VQE를 이용한 분자 에너지 시뮬레이션 실증 (Physical Review X)',
                link: 'https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.031007'
              },
              {
                title: 'Quantum algorithms for quantum chemistry and quantum materials',
                authors: 'Bela Bauer, et al.',
                year: '2020',
                description: '양자 화학 및 재료과학을 위한 양자 알고리즘 종합 리뷰 (Chemical Reviews)',
                link: 'https://pubs.acs.org/doi/10.1021/acs.chemrev.9b00829'
              },
              {
                title: 'Quantum computational chemistry',
                authors: 'Sam McArdle, Suguru Endo, Alan Aspuru-Guzik, et al.',
                year: '2020',
                description: '양자 컴퓨팅 화학 응용 종합 리뷰 (Reviews of Modern Physics)',
                link: 'https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.92.015003'
              }
            ]
          },
          {
            title: 'Quantum Finance',
            icon: 'paper',
            color: 'border-yellow-500',
            items: [
              {
                title: 'Quantum computational finance: Monte Carlo pricing of financial derivatives',
                authors: 'Nikitas Stamatopoulos, et al.',
                year: '2020',
                description: '양자 몬테카를로 기반 금융상품 가격 결정 (Quantum)',
                link: 'https://quantum-journal.org/papers/q-2020-07-06-291/'
              },
              {
                title: 'Quantum optimization for finance',
                authors: 'Stefan Woerner, Daniel J. Egger',
                year: '2019',
                description: 'QAOA를 이용한 포트폴리오 최적화 (IEEE QCE)',
                link: 'https://arxiv.org/abs/1907.04769'
              },
              {
                title: 'Quantum Machine Learning in Finance: Credit Risk Analysis',
                authors: 'IBM Quantum Research',
                year: '2021',
                description: 'IBM의 양자 머신러닝 금융 응용 (arXiv)',
                link: 'https://arxiv.org/abs/2103.15192'
              }
            ]
          },
          {
            title: 'Quantum Cryptography & Security',
            icon: 'paper',
            color: 'border-red-500',
            items: [
              {
                title: 'Quantum cryptography: Public key distribution and coin tossing',
                authors: 'Charles H. Bennett, Gilles Brassard',
                year: '1984',
                description: 'BB84 프로토콜 - 양자 키 분배의 시작 (IEEE Conference)',
                link: 'https://arxiv.org/abs/2003.06557'
              },
              {
                title: 'Post-Quantum Cryptography',
                authors: 'Daniel J. Bernstein, Johannes Buchmann, Erik Dahmen',
                year: '2009',
                description: '포스트 양자 암호학 종합서 (Springer)',
                link: 'https://pqcrypto.org/'
              },
              {
                title: 'Satellite-based entanglement distribution over 1200 kilometers',
                authors: 'Juan Yin, et al.',
                year: '2017',
                description: '중국 묵자(Micius) 위성 기반 QKD 실증 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.aan3211'
              }
            ]
          },
          {
            title: 'Industry & Investment',
            icon: 'web',
            color: 'border-purple-500',
            items: [
              {
                title: 'Boston Consulting Group: Quantum Computing Market Report',
                description: '양자 컴퓨팅 시장 규모 및 투자 동향 분석',
                link: 'https://www.bcg.com/publications/2023/quantum-computing-just-might-save-the-planet'
              },
              {
                title: 'McKinsey: Quantum Technology Monitor',
                description: '글로벌 양자 기술 투자 및 산업 현황',
                link: 'https://www.mckinsey.com/industries/life-sciences/our-insights/quantum-computing-just-might-save-the-planet'
              },
              {
                title: 'EU Quantum Flagship',
                description: '유럽연합 양자 기술 10년 프로젝트 (€10억)',
                link: 'https://qt.eu/'
              },
              {
                title: 'US National Quantum Initiative',
                description: '미국 국가 양자 이니셔티브 현황',
                link: 'https://www.quantum.gov/'
              }
            ]
          }
        ]}
      />
    </div>
  )
}