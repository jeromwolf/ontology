'use client'

import React from 'react'
import { Shield, Lock, Ban, Infinity, Zap, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react'

export default function Chapter7() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-emerald-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl shadow-lg">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                제약 최적화
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Lagrange Multipliers, Penalty Methods, Barrier Methods
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Lock className="w-6 h-6 text-emerald-600" />
              제약 최적화란?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                제약 최적화(Constrained Optimization)는 <strong>등식 또는 부등식 제약 조건</strong>이
                있는 최적화 문제입니다. 실제 문제는 거의 항상 제약이 존재하므로,
                제약을 효과적으로 다루는 것이 매우 중요합니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">제약 최적화의 도전 과제</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Ban className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
                    <span><strong>실행 가능성 유지</strong>: 모든 반복에서 제약 만족 필요</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" />
                    <span><strong>활성 제약 식별</strong>: 어떤 제약이 최적해에 영향을 주는지 판단</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>계산 효율성</strong>: 제약 처리로 인한 추가 계산 비용</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Problem Formulation */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            제약 최적화 문제 정식화
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
              일반적인 형태
            </h3>

            <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4 font-mono text-sm">
              <div className="space-y-2 text-slate-800 dark:text-slate-200">
                <p className="font-bold">minimize   f(x)</p>
                <p>subject to  g<sub>i</sub>(x) ≤ 0,  i = 1, ..., m  (부등식 제약)</p>
                <p className="ml-12">h<sub>j</sub>(x) = 0,  j = 1, ..., p  (등식 제약)</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <h4 className="font-bold text-slate-800 dark:text-white mb-2">등식 제약 (Equality)</h4>
                <p className="text-slate-600 dark:text-slate-400">
                  h(x) = 0 형태. 자유도를 감소시킴.
                </p>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                <h4 className="font-bold text-slate-800 dark:text-white mb-2">부등식 제약 (Inequality)</h4>
                <p className="text-slate-600 dark:text-slate-400">
                  g(x) ≤ 0 형태. 실행 가능 영역을 정의.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Lagrange Multipliers */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Lagrange 승수법 (Lagrange Multipliers)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">등식 제약 최적화의 고전적 방법</h3>
                <Infinity className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-blue-100">
                등식 제약이 있는 최적화 문제를 <strong>무제약 문제</strong>로 변환합니다.
                Lagrange 함수를 정의하고 그 정류점을 찾습니다.
              </p>
            </div>

            {/* Lagrangian */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Lagrangian 함수
              </h3>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4">
                <h4 className="font-bold text-slate-800 dark:text-white mb-3">문제 정의</h4>
                <div className="font-mono text-sm text-slate-800 dark:text-slate-200 mb-4">
                  <p>minimize   f(x)</p>
                  <p>subject to  h(x) = 0</p>
                </div>

                <h4 className="font-bold text-slate-800 dark:text-white mb-3 mt-4">Lagrangian</h4>
                <div className="font-mono text-center text-lg text-slate-800 dark:text-slate-200">
                  L(x, λ) = f(x) + λ<sup>T</sup>h(x)
                </div>
              </div>

              <div className="space-y-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">필요 조건 (Necessary Conditions)</h4>
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-300 space-y-1">
                    <p>∇<sub>x</sub>L = ∇f(x*) + λ*<sup>T</sup>∇h(x*) = 0</p>
                    <p>∇<sub>λ</sub>L = h(x*) = 0</p>
                  </div>
                </div>

                <div className="text-sm text-slate-700 dark:text-slate-300">
                  <p className="mb-2">
                    <strong>해석:</strong> 최적점에서 목적 함수의 기울기는 제약 함수의 기울기와
                    <strong>평행</strong>합니다.
                  </p>
                  <p>
                    <strong>λ (람다)</strong>는 제약 조건의 민감도를 나타냅니다.
                  </p>
                </div>
              </div>
            </div>

            {/* Example */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                간단한 예제
              </h3>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-4">
                  <p className="font-bold mb-2">문제:</p>
                  <p className="font-mono">minimize   f(x, y) = x² + y²</p>
                  <p className="font-mono">subject to  x + y = 1</p>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <p className="font-bold mb-2">Lagrangian:</p>
                  <p className="font-mono">L(x, y, λ) = x² + y² + λ(x + y - 1)</p>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <p className="font-bold mb-2">최적성 조건:</p>
                  <div className="font-mono space-y-1">
                    <p>∂L/∂x = 2x + λ = 0</p>
                    <p>∂L/∂y = 2y + λ = 0</p>
                    <p>∂L/∂λ = x + y - 1 = 0</p>
                  </div>
                </div>

                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <p className="font-bold mb-2">해:</p>
                  <p className="font-mono">x* = y* = 1/2, λ* = -1</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Penalty Methods */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            벌칙 함수법 (Penalty Methods)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">제약 위반에 벌칙 부과</h3>
                <AlertCircle className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-purple-100">
                제약을 위반하면 목적 함수에 <strong>큰 벌칙</strong>을 추가하여
                제약을 만족하도록 유도합니다. 무제약 최적화 기법을 사용할 수 있습니다.
              </p>
            </div>

            {/* Penalty Function */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                벌칙 함수 형태
              </h3>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-4">
                <div className="font-mono text-center text-lg text-slate-800 dark:text-slate-200 mb-4">
                  P(x, μ) = f(x) + μ · penalty(x)
                </div>
                <p className="text-sm text-slate-700 dark:text-slate-300 text-center">
                  μ → ∞ 로 점진적으로 증가
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">2차 벌칙 (Quadratic Penalty)</h4>
                  <div className="font-mono text-xs text-slate-700 dark:text-slate-300 mb-2">
                    <p>penalty = Σ[max(0, gᵢ)]² + Σhⱼ²</p>
                  </div>
                  <p className="text-slate-600 dark:text-slate-400">가장 일반적인 형태</p>
                </div>

                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">절대값 벌칙 (Absolute Penalty)</h4>
                  <div className="font-mono text-xs text-slate-700 dark:text-slate-300 mb-2">
                    <p>penalty = Σ|max(0, gᵢ)| + Σ|hⱼ|</p>
                  </div>
                  <p className="text-slate-600 dark:text-slate-400">비미분 가능</p>
                </div>
              </div>

              <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>⚠️ 주의:</strong> μ가 너무 크면 ill-conditioned 문제가 됩니다.
                  점진적으로 증가시켜야 합니다.
                </p>
              </div>
            </div>

            {/* Algorithm */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                벌칙 함수법 알고리즘
              </h3>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">1.</span>
                  <div>초기 μ₀ 설정 (작은 값)</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">2.</span>
                  <div>k = 0으로 시작</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">3.</span>
                  <div>무제약 최적화로 P(x, μₖ) 최소화 → x<sub>k</sub></div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">4.</span>
                  <div>수렴 검사: 제약 위반이 충분히 작으면 종료</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">5.</span>
                  <div>μ<sub>k+1</sub> = βμ<sub>k</sub> (β > 1, 보통 10)</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">6.</span>
                  <div>k ← k + 1, 3단계로 돌아감</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Barrier Methods */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            장벽 함수법 (Barrier Methods)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">실행 가능 영역 내부 유지</h3>
                <Shield className="w-8 h-8 opacity-80" />
              </div>
              <p className="text-emerald-100">
                제약 경계에 <strong>장벽(barrier)</strong>을 설치하여 해가 항상
                실행 가능 영역 <strong>내부</strong>에 머물도록 합니다.
                Interior Point Method의 핵심 개념입니다.
              </p>
            </div>

            {/* Barrier Function */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                장벽 함수 형태
              </h3>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4">
                <div className="font-mono text-center text-lg text-slate-800 dark:text-slate-200 mb-4">
                  B(x, μ) = f(x) + μ · barrier(x)
                </div>
                <p className="text-sm text-slate-700 dark:text-slate-300 text-center">
                  μ → 0 로 점진적으로 감소 (벌칙법과 반대!)
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">로그 장벽 (Log Barrier)</h4>
                  <div className="font-mono text-xs text-slate-700 dark:text-slate-300 mb-2">
                    <p>barrier = -Σ log(-gᵢ(x))</p>
                  </div>
                  <p className="text-slate-600 dark:text-slate-400">가장 일반적. gᵢ(x) {'<'} 0 필요</p>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">역수 장벽 (Inverse Barrier)</h4>
                  <div className="font-mono text-xs text-slate-700 dark:text-slate-300 mb-2">
                    <p>barrier = Σ 1/(-gᵢ(x))</p>
                  </div>
                  <p className="text-slate-600 dark:text-slate-400">더 강한 장벽 효과</p>
                </div>
              </div>

              <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>💡 핵심:</strong> 제약 경계에 가까워질수록 barrier 값이 급증하여
                  해가 경계를 넘지 못하게 막습니다.
                </p>
              </div>
            </div>

            {/* Comparison: Penalty vs Barrier */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Penalty vs Barrier 비교
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-emerald-50 dark:bg-emerald-900/20">
                    <tr>
                      <th className="px-4 py-3 text-left text-slate-800 dark:text-white">특성</th>
                      <th className="px-4 py-3 text-left text-slate-800 dark:text-white">Penalty Method</th>
                      <th className="px-4 py-3 text-left text-slate-800 dark:text-white">Barrier Method</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-gray-700">
                    <tr>
                      <td className="px-4 py-3 font-bold text-slate-700 dark:text-slate-300">시작점</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">어디서나 가능</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">내부여야 함</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-bold text-slate-700 dark:text-slate-300">경로</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">외부 → 경계</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">내부 → 경계</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-bold text-slate-700 dark:text-slate-300">파라미터</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">μ → ∞</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">μ → 0</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-bold text-slate-700 dark:text-slate-300">장점</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">시작이 쉬움</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">항상 실행 가능</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-bold text-slate-700 dark:text-slate-300">단점</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">Ill-conditioning</td>
                      <td className="px-4 py-3 text-slate-600 dark:text-slate-400">초기 실행 가능해 필요</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* Applications */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            실전 응용 사례
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🎯</span>
                제어 시스템
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• Model Predictive Control (MPC)</li>
                <li>• 상태 제약이 있는 최적 제어</li>
                <li>• 입력 제약 처리</li>
                <li>• 안전 제약 보장</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🏗️</span>
                구조 최적화
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 응력 제약</li>
                <li>• 변위 제약</li>
                <li>• 재료 사용량 제약</li>
                <li>• 안전 계수 보장</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">💼</span>
                금융 최적화
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 포트폴리오 제약 (예산, 섹터)</li>
                <li>• 위험 한도</li>
                <li>• 거래 제약</li>
                <li>• 규제 요구사항</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🤖</span>
                머신러닝
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• SVM (Support Vector Machines)</li>
                <li>• 제약이 있는 회귀</li>
                <li>• Fairness 제약</li>
                <li>• 스파시티 제약</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-blue-50 to-emerald-50 dark:from-blue-900/20 dark:to-emerald-900/20 rounded-xl p-8 border border-emerald-200 dark:border-emerald-800">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4">
              핵심 요점
            </h2>
            <ul className="space-y-3 text-slate-700 dark:text-slate-300">
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">1.</span>
                <span><strong>Lagrange 승수법</strong>은 등식 제약 최적화의 고전적 방법입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span><strong>Penalty Method</strong>는 외부에서 경계로, <strong>Barrier Method</strong>는 내부에서 경계로 접근합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span>Barrier Method는 <strong>Interior Point Method</strong>의 핵심 개념입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span>제약 처리는 실제 최적화 문제에서 <strong>필수적</strong>입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span>파라미터 조정이 <strong>수렴 성능</strong>에 큰 영향을 미칩니다.</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Next Chapter Preview */}
        <section>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border-2 border-emerald-300 dark:border-gray-600">
            <h3 className="text-lg font-bold text-slate-800 dark:text-white mb-2">
              다음 챕터 미리보기
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              <strong>Chapter 8: 다목적 최적화</strong>
              <br />
              Pareto Optimality, NSGA-II, Multi-criteria Decision Making 등
              여러 목표를 동시에 최적화하는 방법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
