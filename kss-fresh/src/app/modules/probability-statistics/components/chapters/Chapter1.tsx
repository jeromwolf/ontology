'use client'

import { useState } from 'react'
import { Dice1, Play, Lightbulb } from 'lucide-react'

export default function Chapter1() {
  const [diceResult, setDiceResult] = useState<number | null>(null)
  const [coinFlips, setCoinFlips] = useState<string[]>([])

  const rollDice = () => {
    const result = Math.floor(Math.random() * 6) + 1
    setDiceResult(result)
  }

  const flipCoins = () => {
    const flips = Array.from({ length: 10 }, () => 
      Math.random() < 0.5 ? 'H' : 'T'
    )
    setCoinFlips(flips)
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">확률의 기초</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          확률론은 불확실성을 수학적으로 다루는 분야입니다. 
          일상생활의 많은 현상들을 확률적으로 모델링할 수 있습니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Dice1 className="text-rose-500" />
          핵심 개념
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-rose-600 dark:text-rose-400 mb-2">표본공간 (Sample Space)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              모든 가능한 결과들의 집합. 주사위의 경우 S = { 1, 2, 3, 4, 5, 6 }
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-rose-600 dark:text-rose-400 mb-2">사건 (Event)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              표본공간의 부분집합. 예: "짝수가 나오는 사건" = { 2, 4, 6 }
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-rose-600 dark:text-rose-400 mb-2">확률 (Probability)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              사건이 발생할 가능성을 0과 1 사이의 수로 표현
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-rose-600 dark:text-rose-400 mb-2">조건부 확률</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              P(A|B) = 사건 B가 일어났을 때 사건 A가 일어날 확률
            </p>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">확률의 공리</h3>
        <ol className="list-decimal list-inside space-y-3">
          <li className="text-gray-700 dark:text-gray-300">
            <strong>비음성:</strong> 모든 사건 A에 대해 P(A) ≥ 0
          </li>
          <li className="text-gray-700 dark:text-gray-300">
            <strong>정규화:</strong> 전체 표본공간의 확률은 1, P(S) = 1
          </li>
          <li className="text-gray-700 dark:text-gray-300">
            <strong>가산가법성:</strong> 서로소인 사건들의 합사건의 확률은 각 사건의 확률의 합
          </li>
        </ol>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Play className="text-green-500" />
          인터랙티브 실험
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-3">주사위 던지기</h4>
            <button 
              onClick={rollDice}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors mb-3"
            >
              주사위 던지기
            </button>
            {diceResult && (
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600 dark:text-green-400">
                  {diceResult}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  P(X = {diceResult}) = 1/6 ≈ 0.167
                </p>
              </div>
            )}
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-3">동전 던지기 (10회)</h4>
            <button 
              onClick={flipCoins}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors mb-3"
            >
              동전 10개 던지기
            </button>
            {coinFlips.length > 0 && (
              <div>
                <div className="flex gap-1 flex-wrap mb-2">
                  {coinFlips.map((flip, idx) => (
                    <span key={idx} className={`w-8 h-8 flex items-center justify-center rounded ${
                      flip === 'H' ? 'bg-blue-500 text-white' : 'bg-gray-500 text-white'
                    }`}>
                      {flip}
                    </span>
                  ))}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  앞면: {coinFlips.filter(f => f === 'H').length}개, 
                  뒷면: {coinFlips.filter(f => f === 'T').length}개
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">베이즈 정리</h3>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
          <p className="text-lg font-mono text-center mb-4">
            P(A|B) = P(B|A) × P(A) / P(B)
          </p>
          <p className="text-gray-700 dark:text-gray-300">
            베이즈 정리는 사전 확률을 이용해 사후 확률을 계산하는 강력한 도구입니다.
            머신러닝, 의료 진단, 스팸 필터링 등 다양한 분야에서 활용됩니다.
          </p>
        </div>
        
        <div className="bg-purple-100 dark:bg-purple-800/30 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">예제: 질병 진단</h4>
          <ul className="list-disc list-inside space-y-1 text-sm text-gray-700 dark:text-gray-300">
            <li>질병 유병률: 1% (P(질병) = 0.01)</li>
            <li>검사 정확도: 99% (P(양성|질병) = 0.99)</li>
            <li>오진율: 5% (P(양성|건강) = 0.05)</li>
          </ul>
          <p className="mt-3 text-sm font-semibold text-purple-700 dark:text-purple-300">
            양성 판정시 실제 질병일 확률 ≈ 16.7%
          </p>
        </div>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Lightbulb className="text-yellow-500" />
          실생활 응용
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">보험</h4>
            <p className="text-sm">사고 발생 확률을 계산하여 보험료 책정</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">투자</h4>
            <p className="text-sm">포트폴리오 리스크 관리와 수익률 예측</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">의료</h4>
            <p className="text-sm">검사 결과 해석과 치료 효과 예측</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">AI/ML</h4>
            <p className="text-sm">불확실성 정량화와 예측 모델링</p>
          </div>
        </div>
      </div>
    </div>
  )
}