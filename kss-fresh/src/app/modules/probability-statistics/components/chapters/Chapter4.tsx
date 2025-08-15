'use client';

import { useState } from 'react';
import { Target, AlertCircle, CheckCircle } from 'lucide-react';

export default function Chapter4() {
  const [sampleSize, setSampleSize] = useState(30)
  const [confidenceLevel, setConfidenceLevel] = useState(95)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">추론 통계</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          추론 통계는 표본 데이터를 사용하여 모집단에 대한 결론을 도출하는 방법입니다.
          가설 검정, 신뢰구간, p-값 등의 개념을 다룹니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Target className="text-indigo-500" />
          가설 검정
        </h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg mb-6">
          <h4 className="text-xl font-semibold mb-4">가설 검정의 단계</h4>
          <ol className="list-decimal list-inside space-y-3">
            <li className="text-gray-700 dark:text-gray-300">
              <strong>가설 설정:</strong> 귀무가설(H₀)과 대립가설(H₁) 설정
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              <strong>유의수준 결정:</strong> 일반적으로 α = 0.05 또는 0.01
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              <strong>검정통계량 계산:</strong> t, z, χ² 등 적절한 통계량 선택
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              <strong>p-값 계산:</strong> 관측된 결과가 나올 확률
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              <strong>결론:</strong> p &lt; α면 귀무가설 기각
            </li>
          </ol>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-indigo-100 dark:bg-indigo-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-indigo-700 dark:text-indigo-300 mb-2">제1종 오류 (α)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              귀무가설이 참인데 기각하는 오류
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              "거짓 양성" - 없는 효과를 있다고 판단
            </p>
          </div>
          <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">제2종 오류 (β)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              귀무가설이 거짓인데 기각하지 못하는 오류
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              "거짓 음성" - 있는 효과를 없다고 판단
            </p>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6">신뢰구간</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg mb-4">
          <h4 className="font-semibold mb-4">신뢰구간 계산기</h4>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-semibold mb-2">
                표본 크기 (n): {sampleSize}
              </label>
              <input
                type="range"
                min="10"
                max="100"
                value={sampleSize}
                onChange={(e) => setSampleSize(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-semibold mb-2">
                신뢰수준: {confidenceLevel}%
              </label>
              <div className="flex gap-2">
                {[90, 95, 99].map(level => (
                  <button
                    key={level}
                    onClick={() => setConfidenceLevel(level)}
                    className={`px-3 py-1 rounded ${
                      confidenceLevel === level 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  >
                    {level}%
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <p className="text-sm font-semibold mb-2">해석</p>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              "{confidenceLevel}% 신뢰구간"은 동일한 방법으로 반복 표본추출시 
              {confidenceLevel}%의 구간이 모수를 포함한다는 의미입니다.
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">평균의 신뢰구간</h4>
            <p className="text-sm font-mono mb-2">x̄ ± t × (s/√n)</p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              모분산을 모를 때 t-분포 사용
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">비율의 신뢰구간</h4>
            <p className="text-sm font-mono mb-2">p̂ ± z × √(p̂(1-p̂)/n)</p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              대표본에서 정규 근사 사용
            </p>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">주요 통계 검정</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">t-검정</h4>
            <ul className="text-sm space-y-1">
              <li>• 단일표본 t-검정</li>
              <li>• 독립표본 t-검정</li>
              <li>• 대응표본 t-검정</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">분산분석 (ANOVA)</h4>
            <ul className="text-sm space-y-1">
              <li>• 일원 분산분석</li>
              <li>• 이원 분산분석</li>
              <li>• 반복측정 ANOVA</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">카이제곱 검정</h4>
            <ul className="text-sm space-y-1">
              <li>• 적합도 검정</li>
              <li>• 독립성 검정</li>
              <li>• 동질성 검정</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">비모수 검정</h4>
            <ul className="text-sm space-y-1">
              <li>• Mann-Whitney U</li>
              <li>• Wilcoxon 검정</li>
              <li>• Kruskal-Wallis</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <AlertCircle className="text-red-500" />
          p-값의 올바른 해석
        </h3>
        <div className="space-y-3">
          <div className="flex items-start gap-2">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
            <p className="text-sm text-gray-700 dark:text-gray-300">
              p-값은 귀무가설이 참일 때 관측된 결과 이상으로 극단적인 결과가 나올 확률
            </p>
          </div>
          <div className="flex items-start gap-2">
            <AlertCircle className="text-red-500 mt-1 flex-shrink-0" size={20} />
            <p className="text-sm text-gray-700 dark:text-gray-300">
              p-값은 귀무가설이 참일 확률이 아님
            </p>
          </div>
          <div className="flex items-start gap-2">
            <AlertCircle className="text-red-500 mt-1 flex-shrink-0" size={20} />
            <p className="text-sm text-gray-700 dark:text-gray-300">
              p-값이 작다고 효과가 크다는 의미는 아님
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}