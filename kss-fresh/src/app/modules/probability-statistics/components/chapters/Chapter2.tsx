'use client';

import { useState } from 'react';
import { Activity } from 'lucide-react';

export default function Chapter2() {
  const [normalMean, setNormalMean] = useState(0)
  const [normalStd, setNormalStd] = useState(1)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">확률 분포</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          확률 분포는 확률 변수가 가질 수 있는 값들과 그 값들이 나타날 확률을 나타냅니다.
          연속형과 이산형으로 나뉘며, 각각 다양한 특성을 가집니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6">주요 확률 분포</h3>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Activity className="text-blue-500" />
              정규 분포 (Normal Distribution)
            </h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  가장 중요한 연속 확률 분포로, 자연과 사회의 많은 현상이 정규 분포를 따릅니다.
                </p>
                <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
                  <p className="font-mono text-sm">f(x) = (1/σ√(2π)) × e^(-½((x-μ)/σ)²)</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                    μ: 평균, σ: 표준편차
                  </p>
                </div>
              </div>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-semibold mb-1">평균 (μ): {normalMean}</label>
                  <input 
                    type="range" 
                    min="-5" 
                    max="5" 
                    value={normalMean}
                    onChange={(e) => setNormalMean(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold mb-1">표준편차 (σ): {normalStd}</label>
                  <input 
                    type="range" 
                    min="0.5" 
                    max="3" 
                    step="0.1"
                    value={normalStd}
                    onChange={(e) => setNormalStd(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
              <div className="bg-blue-100 dark:bg-blue-900/30 p-2 rounded text-center">
                <p className="font-semibold">68%</p>
                <p className="text-xs">μ ± σ</p>
              </div>
              <div className="bg-blue-200 dark:bg-blue-800/30 p-2 rounded text-center">
                <p className="font-semibold">95%</p>
                <p className="text-xs">μ ± 2σ</p>
              </div>
              <div className="bg-blue-300 dark:bg-blue-700/30 p-2 rounded text-center">
                <p className="font-semibold">99.7%</p>
                <p className="text-xs">μ ± 3σ</p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-4">이항 분포 (Binomial Distribution)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              n번의 독립적인 베르누이 시행에서 성공 횟수의 분포입니다.
            </p>
            <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg mb-4">
              <p className="font-mono text-sm">P(X = k) = C(n,k) × p^k × (1-p)^(n-k)</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                n: 시행 횟수, p: 성공 확률, k: 성공 횟수
              </p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2">예제: 동전 던지기</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                공정한 동전을 10번 던질 때 앞면이 정확히 5번 나올 확률:
              </p>
              <p className="text-sm font-mono mt-2">P(X = 5) = C(10,5) × 0.5^5 × 0.5^5 ≈ 0.246</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-4">포아송 분포 (Poisson Distribution)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              일정 시간 또는 공간에서 발생하는 사건의 횟수를 모델링합니다.
            </p>
            <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg mb-4">
              <p className="font-mono text-sm">P(X = k) = (λ^k × e^(-λ)) / k!</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                λ: 평균 발생률, k: 발생 횟수
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
                <h5 className="font-semibold text-sm mb-1">활용 예시</h5>
                <ul className="text-xs space-y-1">
                  <li>• 콜센터 전화 수</li>
                  <li>• 웹사이트 방문자 수</li>
                  <li>• 교통사고 발생 건수</li>
                </ul>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
                <h5 className="font-semibold text-sm mb-1">특징</h5>
                <ul className="text-xs space-y-1">
                  <li>• 평균 = 분산 = λ</li>
                  <li>• 희귀 사건 모델링</li>
                  <li>• n→∞일 때 이항분포 근사</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">기타 중요 분포</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">지수 분포</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              사건 간 대기 시간 모델링 (예: 고장 시간)
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">감마 분포</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              대기 시간의 합, 신뢰성 분석
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">베타 분포</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              확률의 확률, 베이지안 분석
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">카이제곱 분포</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              적합도 검정, 분산 분석
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}