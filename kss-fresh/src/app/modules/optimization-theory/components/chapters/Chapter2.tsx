'use client'

import React from 'react'
import { TrendingUp, GitBranch, Grid3x3, LineChart, Zap, CheckCircle, Calculator, Layers } from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-emerald-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl shadow-lg">
              <LineChart className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                선형 최적화
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Simplex 알고리즘과 Interior Point 방법
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-emerald-600" />
              선형 최적화란?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                선형 최적화(Linear Programming, LP)는 목적 함수와 모든 제약 조건이 <strong>선형</strong>인
                최적화 문제입니다. 산업 현장에서 가장 널리 사용되는 최적화 기법으로, 생산 계획, 물류,
                자원 배분 등 다양한 분야에 적용됩니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">선형 최적화의 특징</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
                    <span><strong>선형성</strong>: 목적 함수와 제약 조건이 모두 1차식</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>효율성</strong>: 다항 시간 내에 해를 구할 수 있음</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Grid3x3 className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>실용성</strong>: 대규모 실전 문제에 적용 가능</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Standard Form */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            선형 계획 문제의 표준형
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
            <div className="mb-6">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">
                표준형 (Standard Form)
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                모든 선형 계획 문제는 다음과 같은 표준형으로 변환할 수 있습니다:
              </p>
            </div>

            <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
              <div className="space-y-2 text-slate-800 dark:text-slate-200">
                <p className="font-bold">minimize   c<sup>T</sup>x</p>
                <p>subject to  Ax = b</p>
                <p className="ml-12">x ≥ 0</p>
                <br />
                <p className="text-emerald-700 dark:text-emerald-400">여기서:</p>
                <p className="ml-4">• c ∈ ℝ<sup>n</sup>: 비용 벡터</p>
                <p className="ml-4">• A ∈ ℝ<sup>m×n</sup>: 제약 행렬</p>
                <p className="ml-4">• b ∈ ℝ<sup>m</sup>: 제약 벡터</p>
                <p className="ml-4">• x ∈ ℝ<sup>n</sup>: 결정 변수</p>
              </div>
            </div>

            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <p className="text-sm text-slate-700 dark:text-slate-300">
                <strong>💡 Tip:</strong> 부등식 제약은 슬랙 변수(slack variable)를 추가하여
                등식 제약으로 변환할 수 있습니다.
              </p>
            </div>
          </div>
        </section>

        {/* Simplex Algorithm */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Simplex 알고리즘
          </h2>

          <div className="space-y-6">
            {/* Algorithm Overview */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">Simplex 알고리즘의 핵심 아이디어</h3>
                <Calculator className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-blue-100">
                George Dantzig이 1947년에 개발한 Simplex 알고리즘은 실행 가능 영역의
                <strong>꼭짓점(vertex)</strong>을 따라 이동하면서 최적해를 찾습니다.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Pivot Operation</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Basis 교체</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">다항 시간</span>
              </div>
            </div>

            {/* Simplex Steps */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Simplex 알고리즘 단계
              </h3>

              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    1
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">초기 기본 실행 가능 해 찾기</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      실행 가능 영역의 한 꼭짓점에서 시작합니다.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    2
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">최적성 검사</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      현재 해가 최적인지 reduced cost를 통해 확인합니다.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    3
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">진입 변수 선택</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      목적 함수 값을 개선할 수 있는 비기저 변수를 선택합니다.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    4
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">탈출 변수 선택</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      기저에서 제거할 변수를 minimum ratio test로 결정합니다.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    5
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">Pivot 연산</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      기저를 업데이트하고 다음 꼭짓점으로 이동합니다.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Interior Point Method */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Interior Point 방법
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">Interior Point Method</h3>
                <Layers className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-purple-100">
                실행 가능 영역의 <strong>내부</strong>를 통과하면서 최적해로 수렴하는 방법입니다.
                대규모 문제에서 Simplex보다 효율적일 수 있습니다.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Barrier Function</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Newton 방법</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">다항 시간 보장</span>
              </div>
            </div>

            {/* Barrier Method */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Barrier Method의 핵심
              </h3>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4">
                <p className="text-slate-700 dark:text-slate-300 mb-3">
                  원래 문제에 <strong>장벽 함수(barrier function)</strong>를 추가하여
                  제약 조건을 목적 함수에 통합합니다:
                </p>
                <div className="font-mono text-sm bg-white dark:bg-gray-800 rounded p-4">
                  <p className="text-slate-800 dark:text-slate-200">minimize   c<sup>T</sup>x - μ Σ log(x<sub>i</sub>)</p>
                  <p className="text-slate-800 dark:text-slate-200">subject to  Ax = b</p>
                </div>
              </div>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <p>
                  <strong>μ</strong>는 장벽 파라미터로, 점진적으로 0에 가까워지면서
                  원래 문제의 최적해에 수렴합니다.
                </p>
                <p>
                  각 반복에서 Newton 방법을 사용하여 중심 경로(central path)를 따라 이동합니다.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Comparison */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Simplex vs Interior Point 비교
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Simplex */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
                <span className="text-2xl">📐</span>
                Simplex 알고리즘
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-2">
                  <span className="text-green-600 dark:text-green-400">✓</span>
                  <span className="text-slate-700 dark:text-slate-300">직관적이고 이해하기 쉬움</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 dark:text-green-400">✓</span>
                  <span className="text-slate-700 dark:text-slate-300">실전에서 매우 빠름 (평균)</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 dark:text-green-400">✓</span>
                  <span className="text-slate-700 dark:text-slate-300">민감도 분석 용이</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-red-600 dark:text-red-400">✗</span>
                  <span className="text-slate-700 dark:text-slate-300">최악의 경우 지수 시간</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-red-600 dark:text-red-400">✗</span>
                  <span className="text-slate-700 dark:text-slate-300">대규모 문제에서 느릴 수 있음</span>
                </div>
              </div>
            </div>

            {/* Interior Point */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
                <span className="text-2xl">🎯</span>
                Interior Point Method
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-2">
                  <span className="text-green-600 dark:text-green-400">✓</span>
                  <span className="text-slate-700 dark:text-slate-300">다항 시간 복잡도 보장</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 dark:text-green-400">✓</span>
                  <span className="text-slate-700 dark:text-slate-300">대규모 문제에 효율적</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 dark:text-green-400">✓</span>
                  <span className="text-slate-700 dark:text-slate-300">예측 가능한 성능</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-red-600 dark:text-red-400">✗</span>
                  <span className="text-slate-700 dark:text-slate-300">구현이 복잡함</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-red-600 dark:text-red-400">✗</span>
                  <span className="text-slate-700 dark:text-slate-300">작은 문제에서 오버헤드</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Real-world Applications */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            실전 응용 사례
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Production Planning */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🏭</span>
                생산 계획
              </h3>
              <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                제한된 자원으로 최대 이익을 내는 생산량 결정
              </p>
              <div className="bg-emerald-50 dark:bg-gray-900 rounded p-3 font-mono text-xs">
                <p>maximize: Σ p<sub>i</sub>x<sub>i</sub></p>
                <p>subject to: Σ a<sub>ij</sub>x<sub>i</sub> ≤ b<sub>j</sub></p>
              </div>
            </div>

            {/* Transportation */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🚚</span>
                물류 최적화
              </h3>
              <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                공급지에서 수요지로의 최소 비용 운송 계획
              </p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 운송 비용 최소화</li>
                <li>• 수요 만족 보장</li>
                <li>• 공급 제약 고려</li>
              </ul>
            </div>

            {/* Portfolio */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">💼</span>
                포트폴리오 최적화
              </h3>
              <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                위험 대비 수익률 최대화
              </p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• Markowitz 평균-분산 모델</li>
                <li>• 자산 배분 제약</li>
                <li>• 위험 한도 설정</li>
              </ul>
            </div>

            {/* Network Flow */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🌐</span>
                네트워크 흐름
              </h3>
              <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                최대 유량, 최소 비용 흐름 문제
              </p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 통신 네트워크 최적화</li>
                <li>• 전력 그리드 관리</li>
                <li>• 교통 흐름 제어</li>
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
                <span>선형 최적화는 <strong>다항 시간</strong>에 해를 구할 수 있는 효율적인 방법입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span><strong>Simplex</strong>는 꼭짓점을 따라, <strong>Interior Point</strong>는 내부를 통과합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span>대규모 문제에서는 <strong>Interior Point</strong> 방법이 더 효율적일 수 있습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span>생산 계획, 물류, 포트폴리오 등 <strong>다양한 실전 응용</strong>이 가능합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span>모든 부등식 제약은 <strong>슬랙 변수</strong>를 통해 등식으로 변환할 수 있습니다.</span>
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
              <strong>Chapter 3: 비선형 최적화</strong>
              <br />
              Gradient Descent, Newton's Method 등 비선형 함수를 최적화하는 고급 기법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
