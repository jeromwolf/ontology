'use client'

import { useState } from 'react'
import { Brain } from 'lucide-react'

export default function Chapter5() {
  const [prior, setPrior] = useState(0.1)
  const [likelihood, setLikelihood] = useState(0.8)
  
  const evidence = prior * likelihood + (1 - prior) * 0.1
  const posterior = (likelihood * prior) / evidence

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">베이지안 통계</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          베이지안 통계는 사전 지식을 활용하여 불확실성을 정량화하고 
          새로운 증거를 통해 믿음을 업데이트하는 방법론입니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Brain className="text-purple-500" />
          베이즈 정리 시뮬레이터
        </h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-semibold mb-2">
                사전 확률 P(H): {(prior * 100).toFixed(1)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={prior}
                onChange={(e) => setPrior(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-semibold mb-2">
                우도 P(E|H): {(likelihood * 100).toFixed(1)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={likelihood}
                onChange={(e) => setLikelihood(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div className="pt-4 border-t">
              <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">계산 결과</h4>
                <p className="text-sm mb-1">
                  증거 P(E) = {(evidence * 100).toFixed(2)}%
                </p>
                <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                  사후 확률 P(H|E) = {(posterior * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
            <p className="text-sm font-mono text-center">
              P(H|E) = P(E|H) × P(H) / P(E)
            </p>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">빈도주의 vs 베이지안</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">빈도주의 접근</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>확률은 장기적 빈도</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>모수는 고정된 값</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>p-값과 신뢰구간 사용</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>객관적이고 반복 가능</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">베이지안 접근</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <span>확률은 믿음의 정도</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <span>모수는 확률 분포</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <span>사후 분포와 신용구간</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <span>사전 지식 활용 가능</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">베이지안 추론 과정</h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex items-start gap-4">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">1</div>
            <div>
              <h4 className="font-semibold mb-1">사전 분포 선택</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                과거 경험이나 전문가 의견을 반영한 초기 믿음
              </p>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex items-start gap-4">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">2</div>
            <div>
              <h4 className="font-semibold mb-1">데이터 수집</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                실험이나 관찰을 통한 새로운 증거 확보
              </p>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex items-start gap-4">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">3</div>
            <div>
              <h4 className="font-semibold mb-1">우도 계산</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                주어진 모수 값에서 데이터가 관측될 확률
              </p>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex items-start gap-4">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">4</div>
            <div>
              <h4 className="font-semibold mb-1">사후 분포 도출</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                베이즈 정리를 통해 업데이트된 믿음 계산
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">베이지안 방법의 장점</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              불확실성 정량화
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              모든 모수에 대한 전체 확률 분포 제공
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              작은 표본 크기
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              사전 정보 활용으로 소규모 데이터에서도 유용
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              순차적 업데이트
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              새 데이터가 들어올 때마다 점진적 업데이트
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              의사결정 지원
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              손실 함수와 결합하여 최적 의사결정
            </p>
          </div>
        </div>
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">실제 응용 사례</h3>
        
        <div className="space-y-3">
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">의료 진단</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              증상과 검사 결과를 종합한 질병 확률 계산
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">A/B 테스팅</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              전환율 차이의 확률적 평가와 조기 종료 결정
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">추천 시스템</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              사용자 선호도의 불확실성을 고려한 개인화
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}