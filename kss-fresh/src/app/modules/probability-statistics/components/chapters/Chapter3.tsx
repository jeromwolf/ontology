'use client'

import { useState } from 'react'
import { BarChart3, Gauge, PieChart } from 'lucide-react'

export default function Chapter3() {
  const [dataSet] = useState([23, 25, 27, 29, 31, 33, 35, 37, 39, 41])
  
  const mean = dataSet.reduce((a, b) => a + b) / dataSet.length
  const median = dataSet[Math.floor(dataSet.length / 2)]
  const variance = dataSet.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / dataSet.length
  const stdDev = Math.sqrt(variance)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">기술 통계</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          기술 통계는 데이터의 특성을 요약하고 시각화하는 방법을 다룹니다.
          중심 경향성, 산포도, 분포의 형태 등을 측정합니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <BarChart3 className="text-green-500" />
          중심 경향성 측정
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">평균 (Mean)</h4>
            <p className="text-2xl font-bold mb-2">{mean.toFixed(1)}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              모든 값의 합을 개수로 나눈 값
            </p>
            <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs font-mono">Σx / n</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">중앙값 (Median)</h4>
            <p className="text-2xl font-bold mb-2">{median}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              정렬된 데이터의 중간 위치 값
            </p>
            <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs">이상치에 강건함</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">최빈값 (Mode)</h4>
            <p className="text-2xl font-bold mb-2">-</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              가장 자주 나타나는 값
            </p>
            <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs">범주형 데이터에 유용</p>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-green-100 dark:bg-green-800/30 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">샘플 데이터</h4>
          <div className="flex gap-2 flex-wrap">
            {dataSet.map((val, idx) => (
              <span key={idx} className="px-3 py-1 bg-white dark:bg-gray-700 rounded">
                {val}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Gauge className="text-blue-500" />
          산포도 측정
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">분산 (Variance)</h4>
            <p className="text-2xl font-bold mb-2">{variance.toFixed(2)}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              평균으로부터의 편차 제곱의 평균
            </p>
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs font-mono">σ² = Σ(x - μ)² / n</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">표준편차 (Std Dev)</h4>
            <p className="text-2xl font-bold mb-2">{stdDev.toFixed(2)}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              분산의 제곱근, 원래 단위와 동일
            </p>
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs font-mono">σ = √(σ²)</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">범위 (Range)</h4>
            <p className="text-2xl font-bold mb-2">{Math.max(...dataSet) - Math.min(...dataSet)}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              최댓값과 최솟값의 차이
            </p>
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs">Max - Min</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">사분위수 범위 (IQR)</h4>
            <p className="text-2xl font-bold mb-2">12</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Q3 - Q1, 중간 50% 데이터의 범위
            </p>
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs">이상치에 강건함</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">분포의 형태</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">왜도 (Skewness)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              분포의 비대칭성을 측정
            </p>
            <ul className="text-xs space-y-1">
              <li>• 양의 왜도: 오른쪽 꼬리가 긴 분포</li>
              <li>• 음의 왜도: 왼쪽 꼬리가 긴 분포</li>
              <li>• 0에 가까움: 대칭 분포</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">첨도 (Kurtosis)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              분포의 뾰족함 정도를 측정
            </p>
            <ul className="text-xs space-y-1">
              <li>• 첨도 &gt; 0: 정규분포보다 뾰족함</li>
              <li>• 첨도 &lt; 0: 정규분포보다 평평함</li>
              <li>• 첨도 = 0: 정규분포와 유사</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <PieChart className="text-yellow-500" />
          데이터 시각화
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">히스토그램</h4>
            <p className="text-sm">연속형 데이터의 분포 표현</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">박스플롯</h4>
            <p className="text-sm">5수 요약과 이상치 표시</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">산점도</h4>
            <p className="text-sm">두 변수 간 관계 시각화</p>
          </div>
        </div>
      </div>
    </div>
  )
}