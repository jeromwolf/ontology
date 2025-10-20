'use client'

import React from 'react'
import { Binary, GitBranch, Scissors, Layers, Zap, TrendingUp, Box, Network } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-emerald-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl shadow-lg">
              <Binary className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                정수 계획법
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Branch and Bound, Cutting Plane, Dynamic Programming
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Binary className="w-6 h-6 text-emerald-600" />
              정수 계획법이란?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                정수 계획법(Integer Programming, IP)은 결정 변수가 <strong>정수 값</strong>만
                가질 수 있는 최적화 문제입니다. 선형 계획법보다 <strong>NP-hard</strong>로
                훨씬 어렵지만, 실제 문제를 더 정확히 모델링할 수 있습니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">정수 계획법의 종류</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Binary className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span><strong>순수 정수 계획(Pure IP)</strong>: 모든 변수가 정수</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Box className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>혼합 정수 계획(MIP)</strong>: 일부 변수만 정수</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>이진 정수 계획(BIP)</strong>: 변수가 0 또는 1</span>
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
              일반적인 정수 계획 문제
            </h3>

            <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4 font-mono text-sm">
              <div className="space-y-2 text-slate-800 dark:text-slate-200">
                <p className="font-bold">minimize   c<sup>T</sup>x</p>
                <p>subject to  Ax ≤ b</p>
                <p className="ml-12">x ≥ 0</p>
                <p className="ml-12 text-emerald-600 dark:text-emerald-400">x ∈ ℤ<sup>n</sup> (정수 제약)</p>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
              <p className="text-sm text-slate-700 dark:text-slate-300">
                <strong>⚠️ 주의:</strong> 정수 제약이 추가되면 문제의 복잡도가
                <strong>다항 시간에서 지수 시간</strong>으로 급증합니다.
              </p>
            </div>
          </div>
        </section>

        {/* Branch and Bound */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Branch and Bound
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">분기 한정법의 핵심 원리</h3>
                <GitBranch className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-blue-100">
                <strong>분기(Branch)</strong>를 통해 문제를 하위 문제로 나누고,
                <strong>한정(Bound)</strong>을 통해 불필요한 탐색을 제거합니다.
                가장 널리 사용되는 정수 계획 해법입니다.
              </p>
            </div>

            {/* Algorithm Steps */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Branch and Bound 알고리즘 단계
              </h3>

              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    1
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">LP Relaxation</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      정수 제약을 제거하고 선형 계획 문제로 풀어 하한(lower bound) 계산
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    2
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">정수 해 확인</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      LP 해가 정수면 최적해. 아니면 다음 단계로
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    3
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">분기 (Branching)</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      비정수 변수 x<sub>i</sub>를 선택하여 x<sub>i</sub> ≤ ⌊x<sub>i</sub>⌋와 x<sub>i</sub> ≥ ⌈x<sub>i</sub>⌉로 분기
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    4
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">한정 (Bounding)</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      현재 최선해보다 나쁜 하한을 가진 노드는 제거 (Pruning)
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    5
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">반복</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      모든 노드가 처리될 때까지 3-4 단계 반복
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Node Selection Strategies */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                노드 선택 전략
              </h3>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">깊이 우선 (DFS)</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    메모리 효율적, 빠른 정수해 발견
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">너비 우선 (BFS)</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    체계적 탐색, 메모리 많이 사용
                  </p>
                </div>
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">최선 우선 (Best-First)</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    가장 좋은 하한을 가진 노드 우선 탐색
                  </p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">하이브리드</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    여러 전략을 조합하여 사용
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Cutting Plane Method */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Cutting Plane 방법
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">절단 평면법</h3>
                <Scissors className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-purple-100">
                LP Relaxation의 실행 가능 영역을 <strong>절단(Cut)</strong>하여
                정수 해로 수렴시키는 방법입니다. 분기 없이 제약 조건만 추가합니다.
              </p>
            </div>

            {/* Types of Cuts */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                주요 절단(Cut) 종류
              </h3>

              <div className="space-y-4">
                <div className="border-l-4 border-emerald-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Gomory Cut</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Simplex tableau에서 유도되는 일반적인 절단. 모든 정수해를 보존.
                  </p>
                </div>

                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Mixed Integer Cut (MIR)</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    혼합 정수 계획 문제에 특화된 절단.
                  </p>
                </div>

                <div className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Cover Cut</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Knapsack 제약에서 유도되는 절단.
                  </p>
                </div>

                <div className="border-l-4 border-orange-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Clique Cut</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    그래프 문제에서 유도되는 절단.
                  </p>
                </div>
              </div>
            </div>

            {/* Branch and Cut */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Branch and Cut
              </h3>
              <p className="text-slate-700 dark:text-slate-300 mb-4">
                Branch and Bound와 Cutting Plane을 <strong>결합</strong>한 방법으로,
                현대 상용 솔버(CPLEX, Gurobi)의 핵심 알고리즘입니다.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-full text-sm">
                  Branching
                </span>
                <span className="text-emerald-600">+</span>
                <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-sm">
                  Cutting
                </span>
                <span className="text-emerald-600">=</span>
                <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-sm">
                  최고 성능
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* Dynamic Programming */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            동적 계획법 (Dynamic Programming)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">부분 문제로 분해</h3>
                <Layers className="w-8 h-8 opacity-80" />
              </div>
              <p className="text-emerald-100">
                문제를 <strong>중복되는 부분 문제</strong>로 나누고 결과를 저장하여
                재사용합니다. 특정 구조를 가진 정수 계획 문제에 효과적입니다.
              </p>
            </div>

            {/* Classic Problems */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                동적 계획법으로 풀 수 있는 정수 계획 문제
              </h3>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">배낭 문제 (Knapsack)</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    제한된 용량으로 최대 가치 선택
                  </p>
                  <div className="font-mono text-xs bg-white dark:bg-gray-900 rounded p-2">
                    DP[i][w] = max value with items 1..i and capacity w
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">최단 경로</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    그래프에서 최단 거리 찾기
                  </p>
                  <div className="font-mono text-xs bg-white dark:bg-gray-900 rounded p-2">
                    DP[v] = shortest distance to v
                  </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">자원 할당</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    여러 활동에 자원 최적 배분
                  </p>
                  <div className="font-mono text-xs bg-white dark:bg-gray-900 rounded p-2">
                    DP[i][r] = max profit with activities 1..i and resource r
                  </div>
                </div>

                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">Lot Sizing</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    생산 계획 및 재고 관리
                  </p>
                  <div className="font-mono text-xs bg-white dark:bg-gray-900 rounded p-2">
                    DP[t] = min cost for periods 1..t
                  </div>
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
                <span className="text-2xl">📦</span>
                물류 & 스케줄링
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 차량 경로 문제 (VRP)</li>
                <li>• 작업 스케줄링</li>
                <li>• 시설 입지 선정</li>
                <li>• 재고 관리</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🌐</span>
                네트워크 설계
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 통신 네트워크 최적화</li>
                <li>• 최대 흐름/최소 비용</li>
                <li>• 네트워크 확장 계획</li>
                <li>• TSP (외판원 문제)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">💰</span>
                금융 & 투자
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 포트폴리오 선택 (정수 제약)</li>
                <li>• 자본 예산 편성</li>
                <li>• 프로젝트 선택</li>
                <li>• 자산 배분</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🏭</span>
                생산 계획
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 생산 믹스 결정</li>
                <li>• 설비 배치</li>
                <li>• Lot Sizing</li>
                <li>• 공급망 최적화</li>
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
                <span>정수 계획은 <strong>NP-hard</strong>로 선형 계획보다 훨씬 어렵습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span><strong>Branch and Bound</strong>는 가장 널리 사용되는 해법입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span><strong>Cutting Plane</strong>은 제약 추가로 정수해에 접근합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span><strong>Branch and Cut</strong>이 현대 상용 솔버의 핵심입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span>특정 구조에서는 <strong>동적 계획법</strong>이 효율적입니다.</span>
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
              <strong>Chapter 6: 메타휴리스틱</strong>
              <br />
              Genetic Algorithm, Simulated Annealing, Particle Swarm 등
              복잡한 최적화 문제를 위한 근사 해법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
