'use client';

import { useState } from 'react';
import { LineChart, Calculator } from 'lucide-react';

export default function Chapter6() {
  const [slope] = useState(2.5)
  const [intercept] = useState(10)
  const [rSquared] = useState(0.85)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">회귀 분석</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          회귀 분석은 변수들 간의 관계를 모델링하고 예측하는 통계적 방법입니다.
          독립변수와 종속변수 간의 함수 관계를 추정합니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <LineChart className="text-cyan-500" />
          단순 선형 회귀
        </h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <div className="mb-4">
            <p className="text-lg font-mono text-center mb-4">
              Y = β₀ + β₁X + ε
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">모델 파라미터</h4>
                <ul className="space-y-2 text-sm">
                  <li>• β₀ (절편): {intercept}</li>
                  <li>• β₁ (기울기): {slope}</li>
                  <li>• R² (결정계수): {rSquared}</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">해석</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  X가 1단위 증가할 때 Y는 평균적으로 {slope}단위 증가합니다.
                  모델이 전체 변동의 {(rSquared * 100).toFixed(0)}%를 설명합니다.
                </p>
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-cyan-100 dark:bg-cyan-900/30 rounded-lg">
            <h4 className="font-semibold mb-2">가정 사항</h4>
            <ul className="text-sm space-y-1">
              <li>• 선형성: X와 Y의 관계가 선형</li>
              <li>• 독립성: 오차항들이 서로 독립</li>
              <li>• 등분산성: 오차의 분산이 일정</li>
              <li>• 정규성: 오차가 정규분포를 따름</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">다중 회귀 분석</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg mb-4">
          <p className="text-lg font-mono text-center mb-4">
            Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
          </p>
          <p className="text-sm text-gray-700 dark:text-gray-300 text-center">
            여러 독립변수를 사용하여 종속변수를 예측
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
              다중공선성
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              독립변수들 간의 높은 상관관계
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              VIF &gt; 10이면 문제 있음
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
              변수 선택
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              중요한 변수만 포함시키기
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              전진, 후진, 단계적 선택법
            </p>
          </div>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">회귀 진단</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">
              잔차 분석
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 잔차 플롯으로 패턴 확인</li>
              <li>• Q-Q 플롯으로 정규성 검정</li>
              <li>• 표준화 잔차 확인</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">
              영향력 진단
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 레버리지 (leverage)</li>
              <li>• Cook's distance</li>
              <li>• DFFITS, DFBETAS</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">고급 회귀 기법</h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              다항 회귀 (Polynomial Regression)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              비선형 관계를 모델링할 때 사용
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              로지스틱 회귀 (Logistic Regression)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              log(p/(1-p)) = β₀ + β₁X₁ + ... + βₚXₚ
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              이진 분류 문제에 사용
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              정규화 회귀 (Regularization)
            </h4>
            <div className="grid grid-cols-3 gap-2 mt-2">
              <div className="text-center">
                <p className="text-xs font-semibold">Ridge (L2)</p>
                <p className="text-xs">계수 축소</p>
              </div>
              <div className="text-center">
                <p className="text-xs font-semibold">Lasso (L1)</p>
                <p className="text-xs">변수 선택</p>
              </div>
              <div className="text-center">
                <p className="text-xs font-semibold">Elastic Net</p>
                <p className="text-xs">L1 + L2</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Calculator className="text-yellow-500" />
          회귀 분석 체크리스트
        </h3>
        
        <div className="space-y-2">
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">데이터 탐색 및 시각화</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">변수 변환 필요성 검토</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">모델 적합 및 계수 해석</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">회귀 가정 검토</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">이상치 및 영향점 확인</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">모델 검증 (교차 검증)</span>
          </label>
        </div>
      </div>
    </div>
  )
}