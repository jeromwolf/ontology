'use client'

import { useState } from 'react'
import { TrendingUp } from 'lucide-react'

export default function Chapter7() {
  const [trendType, setTrendType] = useState('linear')
  const [seasonalPeriod] = useState(12)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">시계열 분석</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          시계열 분석은 시간에 따라 순차적으로 관측된 데이터를 분석하고 
          미래 값을 예측하는 통계적 방법입니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <TrendingUp className="text-teal-500" />
          시계열의 구성 요소
        </h3>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
              추세 (Trend)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              장기적인 방향성
            </p>
            <select 
              value={trendType}
              onChange={(e) => setTrendType(e.target.value)}
              className="mt-2 text-xs w-full p-1 rounded border dark:bg-gray-700"
            >
              <option value="linear">선형</option>
              <option value="exponential">지수</option>
              <option value="polynomial">다항</option>
            </select>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
              계절성 (Seasonal)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              주기적 패턴
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              주기: {seasonalPeriod}개월
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
              순환 (Cyclic)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              불규칙 장기 변동
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              경기 순환 등
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
              불규칙 (Irregular)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              무작위 변동
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              예측 불가능
            </p>
          </div>
        </div>
        
        <div className="mt-6 bg-teal-100 dark:bg-teal-900/30 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">분해 모델</h4>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <p className="font-semibold">가법 모델</p>
              <p className="font-mono text-xs">Y = Trend + Seasonal + Irregular</p>
            </div>
            <div>
              <p className="font-semibold">승법 모델</p>
              <p className="font-mono text-xs">Y = Trend × Seasonal × Irregular</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">정상성 (Stationarity)</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg mb-4">
          <h4 className="font-semibold mb-3">정상 시계열의 조건</h4>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-blue-500">1.</span>
              <span className="text-sm">평균이 시간에 따라 일정</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500">2.</span>
              <span className="text-sm">분산이 시간에 따라 일정</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500">3.</span>
              <span className="text-sm">자기공분산이 시차에만 의존</span>
            </li>
          </ul>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
              정상성 검정
            </h4>
            <ul className="text-sm space-y-1">
              <li>• ADF 검정</li>
              <li>• KPSS 검정</li>
              <li>• Phillips-Perron 검정</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
              정상화 방법
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 차분 (Differencing)</li>
              <li>• 로그 변환</li>
              <li>• 추세 제거</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">ARIMA 모델</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <p className="text-lg font-mono text-center mb-4">
            ARIMA(p, d, q)
          </p>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400">AR(p)</h4>
              <p className="text-sm mt-1">자기회귀</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                과거 p개 시점의 값 사용
              </p>
            </div>
            
            <div className="text-center">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400">I(d)</h4>
              <p className="text-sm mt-1">차분</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                d번 차분으로 정상화
              </p>
            </div>
            
            <div className="text-center">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400">MA(q)</h4>
              <p className="text-sm mt-1">이동평균</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                과거 q개 오차항 사용
              </p>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
            <h4 className="font-semibold mb-2">모델 선택 과정</h4>
            <ol className="text-sm space-y-1 list-decimal list-inside">
              <li>ACF/PACF 플롯 확인</li>
              <li>Box-Jenkins 방법론 적용</li>
              <li>AIC/BIC 기준으로 모델 비교</li>
              <li>잔차 진단</li>
            </ol>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">예측 기법</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
              전통적 방법
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 단순 이동평균</li>
              <li>• 지수 평활법</li>
              <li>• Holt-Winters</li>
              <li>• X-13ARIMA-SEATS</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
              현대적 방법
            </h4>
            <ul className="text-sm space-y-1">
              <li>• Prophet (Facebook)</li>
              <li>• LSTM/GRU</li>
              <li>• Transformer</li>
              <li>• 앙상블 방법</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">예측 성능 평가</h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">MAE</h4>
            <p className="text-xs text-gray-600 dark:text-gray-400">Mean Absolute Error</p>
            <p className="text-sm font-mono mt-2">Σ|실제-예측|/n</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">RMSE</h4>
            <p className="text-xs text-gray-600 dark:text-gray-400">Root Mean Square Error</p>
            <p className="text-sm font-mono mt-2">√(Σ(실제-예측)²/n)</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">MAPE</h4>
            <p className="text-xs text-gray-600 dark:text-gray-400">Mean Absolute Percentage Error</p>
            <p className="text-sm font-mono mt-2">Σ|실제-예측|/실제 × 100</p>
          </div>
        </div>
      </div>
    </div>
  )
}