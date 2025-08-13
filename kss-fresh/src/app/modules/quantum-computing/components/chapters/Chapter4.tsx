'use client'

import { Shield } from 'lucide-react'

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
    </div>
  )
}