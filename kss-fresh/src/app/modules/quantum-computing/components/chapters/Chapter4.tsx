'use client';

import { Shield } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter4() {
  return (
    <div className="p-8 space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Shield className="w-8 h-8 text-purple-600" />
          암호학의 양자 위협
        </h2>
        
        <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-400 rounded-r-xl p-6 mb-6">
          <h3 className="text-xl font-bold text-red-700 dark:text-red-400 mb-4">🚨 RSA 암호화의 위기</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            현재 인터넷 보안의 기반인 RSA 암호화는 큰 수의 소인수분해가 어렵다는 가정에 기반합니다.
            Shor 알고리즘은 이 문제를 양자 컴퓨터로 효율적으로 해결할 수 있습니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">고전 컴퓨터</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                2048비트 RSA: 현재 기술로 수백만 년 소요
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">양자 컴퓨터</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                충분한 큐비트 수: 몇 시간 내 해결 가능
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🔢 주기 찾기 문제</h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">📐 수학적 기초</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              정수 N = p × q (p, q는 소수)를 인수분해하기 위해, 다음 함수의 주기를 찾습니다:
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
              <code className="text-lg text-purple-600 dark:text-purple-400">
                f(x) = aˣ mod N
              </code>
            </div>
            <p className="text-gray-700 dark:text-gray-300">
              여기서 a는 N과 서로소인 임의의 수이고, r은 aʳ ≡ 1 (mod N)을 만족하는 최소 양의 정수입니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🌊 양자 푸리에 변환 (QFT)</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-700 dark:text-purple-400 mb-4">📊 QFT의 역할</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              QFT는 주기적 함수의 주기를 찾기 위한 핵심 도구입니다.
            </p>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• 시간 도메인 → 주파수 도메인 변환</li>
              <li>• 주기적 패턴의 주파수 성분 추출</li>
              <li>• O(n²) 게이트로 구현 (n = 큐비트 수)</li>
              <li>• 고전 FFT의 양자 버전</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-cyan-700 dark:text-cyan-400 mb-4">⚙️ 구현 특징</h3>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• Hadamard 게이트와 제어 회전 게이트 조합</li>
              <li>• 비트 순서 역전 (bit reversal) 필요</li>
              <li>• 근사 QFT로 게이트 수 최적화 가능</li>
              <li>• 병렬 구현으로 깊이 O(n²) → O(n)</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: '원본 논문 (Original Papers)',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer',
                authors: 'Peter W. Shor',
                year: '1997',
                description: '양자 컴퓨팅의 킬러 앱, Shor 알고리즘을 최초로 제안한 역사적 논문',
                link: 'https://arxiv.org/abs/quant-ph/9508027'
              },
              {
                title: 'Quantum Computation and Shor\'s Factoring Algorithm',
                authors: 'Richard Jozsa',
                year: '1998',
                description: 'Shor 알고리즘의 수학적 원리를 명확하게 설명한 리뷰 논문',
                link: 'https://arxiv.org/abs/quant-ph/9707033'
              },
              {
                title: 'Realization of a scalable Shor algorithm',
                authors: 'Thomas Monz et al.',
                year: '2016',
                description: 'Shor 알고리즘의 실제 양자 컴퓨터 구현 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.aad9480'
              },
              {
                title: 'Quantum Fourier Transform and Its Applications',
                authors: 'Michele Mosca',
                year: '1999',
                description: '양자 푸리에 변환(QFT)의 수학적 기초와 응용',
                link: 'https://arxiv.org/abs/quant-ph/9903071'
              }
            ]
          },
          {
            title: 'RSA 암호화와 양자 위협 (RSA & Quantum Threat)',
            icon: 'paper',
            color: 'border-red-500',
            items: [
              {
                title: 'A Method for Obtaining Digital Signatures and Public-Key Cryptosystems',
                authors: 'Ronald L. Rivest, Adi Shamir, Leonard Adleman',
                year: '1978',
                description: 'RSA 암호화를 최초로 제안한 역사적 논문',
                link: 'https://people.csail.mit.edu/rivest/Rsapaper.pdf'
              },
              {
                title: 'Post-Quantum Cryptography',
                authors: 'Daniel J. Bernstein, Johannes Buchmann, Erik Dahmen',
                year: '2009',
                description: '양자 컴퓨터에 안전한 암호화 알고리즘 (포스트 양자 암호학)',
                link: 'https://pqcrypto.org/'
              },
              {
                title: 'NIST Post-Quantum Cryptography Standardization',
                authors: 'NIST',
                year: '2022',
                description: 'NIST의 양자 내성 암호 표준화 프로젝트',
                link: 'https://csrc.nist.gov/projects/post-quantum-cryptography'
              }
            ]
          },
          {
            title: '양자 알고리즘 교재 (Quantum Algorithm Textbooks)',
            icon: 'book',
            color: 'border-blue-500',
            items: [
              {
                title: 'Quantum Computation and Quantum Information',
                authors: 'Michael A. Nielsen, Isaac L. Chuang',
                year: '2010',
                description: '양자 컴퓨팅의 바이블 - Shor 알고리즘 상세 설명 (Chapter 5)',
                link: 'http://mmrc.amss.cas.cn/tlb/201702/W020170224608149940643.pdf'
              },
              {
                title: 'Quantum Computing: A Gentle Introduction',
                authors: 'Eleanor Rieffel, Wolfgang Polak',
                year: '2011',
                description: '양자 컴퓨팅 입문서 - Shor 알고리즘 단계별 설명',
                link: 'https://mitpress.mit.edu/9780262526678/'
              },
              {
                title: 'Quantum Algorithms via Linear Algebra',
                authors: 'Richard J. Lipton, Kenneth W. Regan',
                year: '2014',
                description: '선형대수 관점에서 본 양자 알고리즘 (MIT Press)',
                link: 'https://mitpress.mit.edu/9780262028394/'
              }
            ]
          },
          {
            title: '실습 및 구현 자료 (Implementation Resources)',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Qiskit: Shor\'s Algorithm Tutorial',
                description: 'IBM Qiskit에서 Shor 알고리즘 구현 튜토리얼',
                link: 'https://qiskit.org/textbook/ch-algorithms/shor.html'
              },
              {
                title: 'Cirq: Implementing Shor\'s Algorithm',
                description: 'Google Cirq로 Shor 알고리즘 실습',
                link: 'https://quantumai.google/cirq/experiments/shor'
              },
              {
                title: 'Microsoft Quantum: Q# Shor Implementation',
                description: 'Microsoft Q#로 작성된 Shor 알고리즘 샘플',
                link: 'https://github.com/microsoft/QuantumKatas'
              },
              {
                title: 'Quantum Algorithm Zoo',
                description: 'Stephen Jordan의 양자 알고리즘 종합 데이터베이스',
                link: 'https://quantumalgorithmzoo.org/'
              }
            ]
          }
        ]}
      />
    </div>
  )
}