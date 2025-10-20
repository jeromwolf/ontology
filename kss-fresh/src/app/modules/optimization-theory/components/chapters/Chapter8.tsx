'use client'

import React from 'react'
import { Target, TrendingUp, Scale, GitBranch, Zap, CheckCircle2, Award, BarChart3 } from 'lucide-react'

export default function Chapter8() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-emerald-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl shadow-lg">
              <Target className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                다목적 최적화
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Pareto Optimality, NSGA-II, Multi-criteria Decision Making
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Scale className="w-6 h-6 text-emerald-600" />
              다목적 최적화란?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                다목적 최적화(Multi-objective Optimization)는 <strong>여러 개의 목적 함수</strong>를
                동시에 최적화하는 문제입니다. 실제 세계의 문제는 대부분 상충하는 여러 목표를 가지므로,
                단일 최적해 대신 <strong>Pareto 최적해 집합</strong>을 찾습니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">다목적 최적화의 특징</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Scale className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
                    <span><strong>목표 간 상충</strong>: 한 목표 개선 시 다른 목표 악화</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>해 집합</strong>: 단일 해가 아닌 Pareto Front</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Award className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>의사결정</strong>: 최종 해 선택에 선호도 필요</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Problem Formulation */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            문제 정식화
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
              일반적인 다목적 최적화 문제
            </h3>

            <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4 font-mono text-sm">
              <div className="space-y-2 text-slate-800 dark:text-slate-200">
                <p className="font-bold">minimize   F(x) = [f₁(x), f₂(x), ..., f<sub>k</sub>(x)]<sup>T</sup></p>
                <p>subject to  g<sub>i</sub>(x) ≤ 0,  i = 1, ..., m</p>
                <p className="ml-12">h<sub>j</sub>(x) = 0,  j = 1, ..., p</p>
                <p className="ml-12">x ∈ X</p>
              </div>
            </div>

            <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
              <p>
                <strong>F(x)</strong>: 목적 함수 벡터 (k개의 목적)
              </p>
              <p>
                <strong>핵심:</strong> 모든 목적을 동시에 최소화하는 단일 해는 일반적으로 존재하지 않습니다.
              </p>
            </div>
          </div>
        </section>

        {/* Pareto Optimality */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Pareto 최적성
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">지배(Dominance) 개념</h3>
                <Award className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-blue-100">
                해 x가 해 y를 <strong>지배(dominate)</strong>한다는 것은
                모든 목적에서 x가 y보다 나쁘지 않고, 적어도 하나의 목적에서 더 좋다는 의미입니다.
              </p>
            </div>

            {/* Definitions */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                핵심 개념 정의
              </h3>

              <div className="space-y-4">
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">지배 (Dominance)</h4>
                  <p className="text-sm text-slate-700 dark:text-slate-300 mb-2">
                    x가 y를 지배 (x ≺ y) ⟺
                  </p>
                  <div className="font-mono text-xs text-slate-700 dark:text-slate-300 space-y-1">
                    <p>1. ∀i: f<sub>i</sub>(x) ≤ f<sub>i</sub>(y) (모든 목적에서 나쁘지 않음)</p>
                    <p>2. ∃j: f<sub>j</sub>(x) {'<'} f<sub>j</sub>(y) (적어도 하나는 더 좋음)</p>
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">Pareto 최적해</h4>
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    해 x*가 Pareto 최적 ⟺ x*를 지배하는 다른 실행 가능한 해가 존재하지 않음
                  </p>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">Pareto Front (Pareto 경계)</h4>
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    모든 Pareto 최적해들의 목적 함수 값들이 이루는 곡면
                  </p>
                </div>

                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">Pareto Set</h4>
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    모든 Pareto 최적해들의 집합 (결정 변수 공간에서)
                  </p>
                </div>
              </div>
            </div>

            {/* Example */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                간단한 예제: 자동차 설계
              </h3>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">목적 1: 연비 최대화</h4>
                  <p className="text-slate-600 dark:text-slate-400">
                    minimize: -fuel_efficiency(x)
                  </p>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">목적 2: 가속 성능 최대화</h4>
                  <p className="text-slate-600 dark:text-slate-400">
                    minimize: -acceleration(x)
                  </p>
                </div>
              </div>

              <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>상충 관계:</strong> 연비를 높이려면 엔진을 작게 해야 하지만,
                  가속 성능은 떨어집니다. Pareto Front는 이 두 목표 간의 최선의 trade-off를 보여줍니다.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* NSGA-II */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            NSGA-II 알고리즘
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">Non-dominated Sorting Genetic Algorithm II</h3>
                <GitBranch className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-purple-100">
                다목적 최적화를 위한 가장 널리 사용되는 <strong>진화 알고리즘</strong>입니다.
                비지배 정렬과 밀집도 거리를 통해 다양하고 균일한 Pareto Front를 찾습니다.
              </p>
            </div>

            {/* Key Features */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                NSGA-II의 핵심 특징
              </h3>

              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    1
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">Fast Non-dominated Sorting</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      개체군을 비지배 수준(rank)으로 분류. Front 1, 2, 3, ... 순서로 정렬.
                      복잡도 O(MN²) (M: 목적 수, N: 개체 수)
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    2
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">Crowding Distance</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      같은 Front 내에서 밀집도를 계산하여 다양성 유지.
                      밀집도가 낮은(더 고립된) 해를 우선적으로 선택.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    3
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">Elitism</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      부모와 자손을 합쳐 최선의 N개를 선택. 좋은 해가 손실되지 않음.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* NSGA-II Algorithm */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                NSGA-II 알고리즘 단계
              </h3>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">1.</span>
                  <div>초기 개체군 P₀ 생성 (크기 N)</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">2.</span>
                  <div>비지배 정렬 및 crowding distance 계산</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">3.</span>
                  <div>선택, 교차, 돌연변이로 자손 Q<sub>t</sub> 생성 (크기 N)</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">4.</span>
                  <div>부모와 자손 결합: R<sub>t</sub> = P<sub>t</sub> ∪ Q<sub>t</sub> (크기 2N)</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">5.</span>
                  <div>R<sub>t</sub>를 비지배 정렬하여 Front F₁, F₂, ... 생성</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">6.</span>
                  <div>P<sub>t+1</sub> = ∅</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">7.</span>
                  <div>Front를 순서대로 추가하다가 |P<sub>t+1</sub>| + |F<sub>i</sub>| {'>'} N이면</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">8.</span>
                  <div>F<sub>i</sub>를 crowding distance로 정렬하여 필요한 만큼만 추가</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">9.</span>
                  <div>종료 조건까지 3-8 반복</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Scalarization Methods */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            스칼라화 기법
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">다목적을 단일 목적으로 변환</h3>
                <BarChart3 className="w-8 h-8 opacity-80" />
              </div>
              <p className="text-emerald-100">
                여러 목적 함수를 <strong>가중합 등의 방법</strong>으로 단일 함수로 변환하여
                기존 단일 목적 최적화 기법을 사용할 수 있게 합니다.
              </p>
            </div>

            {/* Methods */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                주요 스칼라화 방법
              </h3>

              <div className="space-y-4">
                <div className="border-l-4 border-emerald-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">가중합 (Weighted Sum)</h4>
                  <div className="bg-emerald-50 dark:bg-gray-900 rounded p-3 font-mono text-xs mb-2">
                    minimize: Σ wᵢfᵢ(x), where Σwᵢ = 1, wᵢ ≥ 0
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    간단하지만 비볼록 Pareto Front는 찾지 못함
                  </p>
                </div>

                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">ε-제약 (ε-Constraint)</h4>
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3 font-mono text-xs mb-2">
                    minimize: f₁(x), subject to: fᵢ(x) ≤ εᵢ, i=2,...,k
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    비볼록 Front도 찾을 수 있음. ε 값 변경으로 Front 탐색
                  </p>
                </div>

                <div className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Achievement Function</h4>
                  <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3 font-mono text-xs mb-2">
                    minimize: max<sub>i</sub> wᵢ(fᵢ(x) - zᵢ*)
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    목표점 z*에 가장 가까운 Pareto 최적해 찾기
                  </p>
                </div>
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
                <span className="text-2xl">🚗</span>
                제품 설계
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 성능 vs 비용</li>
                <li>• 무게 vs 강도</li>
                <li>• 효율 vs 출력</li>
                <li>• 크기 vs 용량</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🏭</span>
                생산 최적화
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 품질 vs 비용</li>
                <li>• 생산량 vs 재고</li>
                <li>• 처리량 vs 에너지</li>
                <li>• 속도 vs 정확도</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">💼</span>
                포트폴리오 관리
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 수익 vs 위험</li>
                <li>• 유동성 vs 수익률</li>
                <li>• 단기 vs 장기 수익</li>
                <li>• 다각화 vs 집중</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🤖</span>
                머신러닝
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 정확도 vs 복잡도</li>
                <li>• 성능 vs 해석 가능성</li>
                <li>• 정밀도 vs 재현율</li>
                <li>• Fairness vs Accuracy</li>
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
                <span>다목적 최적화는 <strong>Pareto 최적해 집합</strong>을 찾습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span><strong>지배 관계</strong>로 해의 우열을 비교합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span><strong>NSGA-II</strong>는 가장 널리 사용되는 진화 알고리즘입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span>스칼라화를 통해 <strong>단일 목적 기법</strong>을 활용할 수 있습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span>최종 해 선택에는 <strong>의사결정자의 선호도</strong>가 필요합니다.</span>
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
              <strong>Chapter 9: 하이퍼파라미터 튜닝</strong>
              <br />
              Grid Search, Random Search, Bayesian Optimization 등
              머신러닝 모델의 최적 파라미터를 찾는 방법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
