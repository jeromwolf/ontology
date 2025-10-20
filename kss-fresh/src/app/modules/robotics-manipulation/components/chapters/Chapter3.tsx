'use client'

import React from 'react'
import ChapterNavigation from '../ChapterNavigation'

export default function Chapter3() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Chapter 3: 역기구학 (Inverse Kinematics)</h1>
        <p className="text-xl text-white/90">
          목표 위치로부터 관절 각도 계산 - 로봇 제어의 핵심
        </p>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        {/* Section 1: 역기구학 개요 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3.1 역기구학 문제
          </h2>

          <div className="mb-6">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              역기구학(Inverse Kinematics, IK)은 <strong>원하는 엔드이펙터의 위치와 방향이 주어졌을 때,
              이를 달성하기 위한 관절 각도를 계산</strong>하는 과정입니다.
            </p>

            <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-6 rounded">
              <h4 className="text-lg font-bold text-red-900 dark:text-red-300 mb-3">
                역기구학의 어려움
              </h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>다중해 (Multiple Solutions)</strong>: 같은 목표에 도달하는 여러 관절 각도 조합 존재</li>
                <li><strong>해 없음 (No Solution)</strong>: 목표가 작업 공간 밖에 있으면 도달 불가능</li>
                <li><strong>비선형 방정식</strong>: 삼각함수가 포함된 복잡한 비선형 시스템</li>
                <li><strong>특이점</strong>: 특정 자세에서 야코비안 행렬의 역행렬이 존재하지 않음</li>
              </ul>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">역기구학 접근법</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg border border-green-200 dark:border-green-800">
                <h4 className="text-xl font-bold text-green-600 dark:text-green-400 mb-3">
                  해석적 해법 (Analytical)
                </h4>
                <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
                  <li>✓ <strong>장점</strong>: 빠르고 정확</li>
                  <li>✓ <strong>장점</strong>: 모든 해 찾기 가능</li>
                  <li>✗ <strong>단점</strong>: 복잡한 로봇에 적용 어려움</li>
                  <li>✗ <strong>단점</strong>: 수학적 유도 필요</li>
                </ul>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-200 dark:border-blue-800">
                <h4 className="text-xl font-bold text-blue-600 dark:text-blue-400 mb-3">
                  수치적 해법 (Numerical)
                </h4>
                <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
                  <li>✓ <strong>장점</strong>: 일반적 적용 가능</li>
                  <li>✓ <strong>장점</strong>: 구현 간단</li>
                  <li>✗ <strong>단점</strong>: 반복 계산 필요 (느림)</li>
                  <li>✗ <strong>단점</strong>: 하나의 해만 찾음</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: 해석적 해법 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3.2 해석적 역기구학
          </h2>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">2-DOF 평면 로봇 예제</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              2개의 회전 관절을 가진 평면 로봇의 역기구학을 기하학적 방법으로 풀어봅시다.
            </p>

            <div className="bg-gray-100 dark:bg-gray-900 p-6 rounded-lg font-mono text-sm overflow-x-auto mb-4">
              <pre className="text-gray-800 dark:text-gray-200">
{`주어진 값:
- 목표 위치: (x, y)
- 링크 길이: L₁, L₂

순기구학 방정식 (복습):
x = L₁cos(θ₁) + L₂cos(θ₁ + θ₂)
y = L₁sin(θ₁) + L₂sin(θ₁ + θ₂)

구하려는 값: θ₁, θ₂`}
              </pre>
            </div>

            <div className="space-y-6">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
                <h4 className="font-bold text-blue-900 dark:text-blue-300 mb-3">
                  Step 1: θ₂ 계산 (코사인 법칙)
                </h4>
                <div className="bg-white dark:bg-gray-800 p-4 rounded font-mono text-xs overflow-x-auto">
                  <pre className="text-gray-800 dark:text-gray-200">
{`거리: D = √(x² + y²)

코사인 법칙:
D² = L₁² + L₂² - 2·L₁·L₂·cos(180° - θ₂)
D² = L₁² + L₂² + 2·L₁·L₂·cos(θ₂)

cos(θ₂) = (D² - L₁² - L₂²) / (2·L₁·L₂)

θ₂ = ±arccos((D² - L₁² - L₂²) / (2·L₁·L₂))

⚠️ 두 개의 해가 존재 (팔꿈치 위/아래)`}
                  </pre>
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
                <h4 className="font-bold text-purple-900 dark:text-purple-300 mb-3">
                  Step 2: θ₁ 계산
                </h4>
                <div className="bg-white dark:bg-gray-800 p-4 rounded font-mono text-xs overflow-x-auto">
                  <pre className="text-gray-800 dark:text-gray-200">
{`보조 각도 정의:
α = arctan2(y, x)  // 목표점 각도
β = arctan2(L₂·sin(θ₂), L₁ + L₂·cos(θ₂))  // 내부 각도

θ₁ = α - β  (팔꿈치 아래)
θ₁ = α + β  (팔꿈치 위)

최종 해:
Solution 1 (Elbow Down): (θ₁, +θ₂)
Solution 2 (Elbow Up):   (θ₁', -θ₂)`}
                  </pre>
                </div>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">6-DOF 로봇의 해석적 해법</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              6-DOF 산업용 로봇은 <strong>기하학적 분리(Geometric Decoupling)</strong> 기법을 사용합니다:
            </p>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border border-orange-200 dark:border-orange-800">
                <h4 className="text-lg font-bold text-orange-600 dark:text-orange-400 mb-3">
                  1단계: 위치 (Position)
                </h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  처음 3개 관절(θ₁, θ₂, θ₃)로 손목 중심 위치 계산
                </p>
              </div>

              <div className="bg-teal-50 dark:bg-teal-900/20 p-6 rounded-lg border border-teal-200 dark:border-teal-800">
                <h4 className="text-lg font-bold text-teal-600 dark:text-teal-400 mb-3">
                  2단계: 방향 (Orientation)
                </h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  나머지 3개 관절(θ₄, θ₅, θ₆)로 엔드이펙터 방향 계산
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: 수치적 해법 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3.3 수치적 역기구학 (Jacobian 기반)
          </h2>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">야코비안 행렬 (Jacobian Matrix)</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              야코비안 J는 관절 속도와 엔드이펙터 속도 간의 관계를 나타내는 행렬입니다:
            </p>

            <div className="bg-gray-100 dark:bg-gray-900 p-6 rounded-lg font-mono text-sm overflow-x-auto mb-4">
              <pre className="text-gray-800 dark:text-gray-200">
{`v = J(θ) · θ̇

여기서:
- v: 엔드이펙터 선속도/각속도 (6×1)
- J(θ): 야코비안 행렬 (6×n)
- θ̇: 관절 속도 (n×1)

야코비안 요소:
     [ ∂x/∂θ₁  ∂x/∂θ₂  ...  ∂x/∂θₙ ]
     [ ∂y/∂θ₁  ∂y/∂θ₂  ...  ∂y/∂θₙ ]
J =  [ ∂z/∂θ₁  ∂z/∂θ₂  ...  ∂z/∂θₙ ]
     [   ωₓ       ωᵧ     ...    ωz   ]`}
              </pre>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">Newton-Raphson 반복법</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              목표 위치 x<sub>goal</sub>에 도달하기 위해 관절 각도를 반복적으로 업데이트합니다:
            </p>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded font-mono text-xs overflow-x-auto">
                <pre className="text-gray-800 dark:text-gray-200">
{`알고리즘:
1. 초기값 설정: θ = θ₀
2. 반복 (k = 0, 1, 2, ...):
   a. 현재 위치 계산: x = FK(θₖ)
   b. 오차 계산: Δx = x_goal - x
   c. 야코비안 계산: J = J(θₖ)
   d. 관절 변화량: Δθ = J⁺ · Δx
   e. 업데이트: θₖ₊₁ = θₖ + α·Δθ
   f. 수렴 조건 확인: ||Δx|| < ε

여기서:
- J⁺: 야코비안의 의사역행렬 (Pseudo-inverse)
- α: 학습률 (Step size)
- ε: 허용 오차`}
                </pre>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-6 rounded">
              <h4 className="font-bold text-yellow-900 dark:text-yellow-300 mb-2">
                ⚠️ 주의사항
              </h4>
              <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                <li>• 초기값에 따라 다른 해로 수렴</li>
                <li>• 특이점 근처에서 수치적 불안정</li>
                <li>• 수렴 보장 없음 (local minimum에 빠질 수 있음)</li>
                <li>• 실시간 제어에는 계산 비용 고려 필요</li>
              </ul>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">Damped Least Squares (DLS)</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              특이점 근처에서의 수치적 안정성을 개선한 방법입니다:
            </p>

            <div className="bg-gray-100 dark:bg-gray-900 p-6 rounded-lg font-mono text-sm overflow-x-auto">
              <pre className="text-gray-800 dark:text-gray-200">
{`일반 의사역행렬:
J⁺ = Jᵀ(J·Jᵀ)⁻¹

DLS (Levenberg-Marquardt):
J⁺ = Jᵀ(J·Jᵀ + λ²I)⁻¹

여기서:
- λ: Damping factor (감쇠 계수)
- I: 단위 행렬

장점:
✓ 특이점에서도 안정적 동작
✓ 행렬 역행렬 항상 존재
✓ λ 조정으로 성능/안정성 균형`}
              </pre>
            </div>
          </div>
        </section>

        {/* Section 4: 다중해 선택 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3.4 다중해 선택 전략
          </h2>

          <div className="mb-6">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              역기구학은 여러 개의 해를 가질 수 있습니다. 실제 로봇 제어에서는 최적의 해를 선택해야 합니다:
            </p>

            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-lg font-bold mb-3">1. 현재 자세와의 유사성</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  관절 변화량이 가장 작은 해 선택 (부드러운 동작)
                </p>
                <div className="bg-blue-100 dark:bg-blue-900/20 p-3 rounded font-mono text-xs">
                  cost = ||θ_new - θ_current||²
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-lg font-bold mb-3">2. 특이점 회피</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  야코비안 행렬식이 큰 해 선택 (조작성 지수 최대화)
                </p>
                <div className="bg-purple-100 dark:bg-purple-900/20 p-3 rounded font-mono text-xs">
                  cost = -|det(J)|
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-lg font-bold mb-3">3. 관절 한계 회피</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  관절 한계로부터 충분한 여유가 있는 해 선택
                </p>
                <div className="bg-green-100 dark:bg-green-900/20 p-3 rounded font-mono text-xs">
                  cost = Σ w_i · |(θ_i - θ_i_mid) / (θ_i_max - θ_i_min)|
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-lg font-bold mb-3">4. 충돌 회피</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  로봇 자체 또는 환경과의 충돌이 없는 해 선택
                </p>
                <div className="bg-red-100 dark:bg-red-900/20 p-3 rounded font-mono text-xs">
                  cost = min_distance_to_obstacles
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Summary */}
        <section className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-8 border border-orange-200 dark:border-orange-800">
          <h2 className="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-4">
            📌 Chapter 3 요약
          </h2>
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-orange-500 font-bold">✓</span>
              <span>역기구학은 목표 위치로부터 관절 각도를 계산하며, 다중해 또는 해가 없을 수 있습니다.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-orange-500 font-bold">✓</span>
              <span>해석적 해법은 빠르지만 복잡한 로봇에는 적용이 어렵고, 수치적 해법은 일반적이지만 느립니다.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-orange-500 font-bold">✓</span>
              <span>야코비안 기반 Newton-Raphson 방법은 반복적으로 해를 찾으며, DLS로 안정성을 향상시킵니다.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-orange-500 font-bold">✓</span>
              <span>다중해 선택은 현재 자세, 특이점 회피, 관절 한계, 충돌 회피 등을 고려합니다.</span>
            </li>
          </ul>
        </section>

        {/* Chapter Navigation */}
        <ChapterNavigation
          currentChapter={3}
          totalChapters={8}
          moduleSlug="robotics-manipulation"
        />
      </div>
    </div>
  )
}
