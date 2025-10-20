'use client'

import React from 'react'
import { Flame, Dna, Bird, Shuffle, Zap, TrendingUp, Wind, Sparkles } from 'lucide-react'

export default function Chapter6() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-emerald-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl shadow-lg">
              <Dna className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                메타휴리스틱
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Genetic Algorithm, Simulated Annealing, Particle Swarm
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-emerald-600" />
              메타휴리스틱이란?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                메타휴리스틱(Metaheuristics)은 복잡한 최적화 문제에 대한 <strong>근사 해법</strong>입니다.
                정확한 최적해는 보장하지 않지만, <strong>합리적인 시간</strong> 내에
                <strong>좋은 해</strong>를 찾을 수 있습니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">메타휴리스틱이 필요한 이유</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>NP-hard 문제</strong>: 정확한 해법은 지수 시간 소요</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <TrendingUp className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>대규모 문제</strong>: 기존 방법으로는 풀 수 없음</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Wind className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
                    <span><strong>실시간 요구</strong>: 빠른 응답이 필요한 경우</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Genetic Algorithm */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            유전 알고리즘 (Genetic Algorithm)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">생물 진화의 원리 모방</h3>
                <Dna className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-blue-100">
                다윈의 <strong>자연선택</strong> 이론에 기반하여 해집단을 진화시킵니다.
                선택, 교차, 돌연변이를 반복하며 점진적으로 개선된 해를 찾습니다.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Selection</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Crossover</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Mutation</span>
              </div>
            </div>

            {/* GA Steps */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                유전 알고리즘 단계
              </h3>

              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    1
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">초기 개체군 생성</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      랜덤하게 초기 해(염색체) 집단 생성
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    2
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">적합도 평가 (Fitness Evaluation)</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      각 개체의 품질을 적합도 함수로 측정
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    3
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">선택 (Selection)</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      적합도가 높은 개체를 부모로 선택 (룰렛 휠, 토너먼트 등)
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    4
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">교차 (Crossover)</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      부모의 유전자를 조합하여 자손 생성 (1점, 2점, 균등 교차 등)
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    5
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">돌연변이 (Mutation)</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      낮은 확률로 유전자 일부 변경 (다양성 유지)
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">
                    6
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-800 dark:text-white mb-1">세대 교체 & 반복</h4>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      새로운 개체군으로 교체하고 2-5 단계 반복
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* GA Parameters */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                주요 파라미터
              </h3>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">개체군 크기 (Population Size)</p>
                  <p className="text-slate-600 dark:text-slate-400">
                    일반적으로 50-200. 클수록 다양성 증가하지만 느림.
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">교차율 (Crossover Rate)</p>
                  <p className="text-slate-600 dark:text-slate-400">
                    일반적으로 0.6-0.9. 높을수록 탐색 범위 넓음.
                  </p>
                </div>
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">돌연변이율 (Mutation Rate)</p>
                  <p className="text-slate-600 dark:text-slate-400">
                    일반적으로 0.001-0.1. 너무 높으면 랜덤 탐색이 됨.
                  </p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">세대 수 (Generations)</p>
                  <p className="text-slate-600 dark:text-slate-400">
                    종료 조건. 개선이 없으면 조기 종료 가능.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Simulated Annealing */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            담금질 기법 (Simulated Annealing)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">금속 담금질 과정 모방</h3>
                <Flame className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-purple-100">
                금속을 천천히 냉각시켜 최적 구조를 얻는 과정에서 영감을 받았습니다.
                초기에는 <strong>나쁜 해도 수용</strong>하여 지역 최적해를 탈출하고,
                점진적으로 수용 확률을 낮춰 수렴시킵니다.
              </p>
            </div>

            {/* SA Algorithm */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Simulated Annealing 알고리즘
              </h3>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4">
                <h4 className="font-bold text-slate-800 dark:text-white mb-3">수용 확률 (Acceptance Probability)</h4>
                <div className="font-mono text-center text-lg text-slate-800 dark:text-slate-200 mb-2">
                  P(accept) = exp(-ΔE / T)
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                  <p>• ΔE = E(new) - E(current): 에너지 차이</p>
                  <p>• T: 현재 온도 (Temperature)</p>
                  <p>• ΔE {'<'} 0 (개선): 항상 수용</p>
                  <p>• ΔE {'>'} 0 (악화): exp(-ΔE/T) 확률로 수용</p>
                </div>
              </div>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">1.</span>
                  <div>
                    <strong>초기 온도 설정</strong>: T₀ (높게 설정)
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">2.</span>
                  <div>
                    <strong>이웃 해 생성</strong>: 현재 해의 작은 변형
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">3.</span>
                  <div>
                    <strong>수용 판단</strong>: 확률에 따라 수용/거절
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">4.</span>
                  <div>
                    <strong>온도 감소</strong>: T ← αT (α ≈ 0.95)
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">5.</span>
                  <div>
                    <strong>종료 조건</strong>: T {'<'} T_min 또는 충분한 반복
                  </div>
                </div>
              </div>
            </div>

            {/* Cooling Schedules */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                냉각 스케줄 (Cooling Schedules)
              </h3>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">기하학적 (Geometric)</p>
                  <p className="text-slate-600 dark:text-slate-400 font-mono text-xs mb-2">T<sub>k+1</sub> = α × T<sub>k</sub></p>
                  <p className="text-slate-600 dark:text-slate-400">가장 일반적. α ≈ 0.9-0.99</p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">선형 (Linear)</p>
                  <p className="text-slate-600 dark:text-slate-400 font-mono text-xs mb-2">T<sub>k+1</sub> = T<sub>k</sub> - β</p>
                  <p className="text-slate-600 dark:text-slate-400">일정한 속도로 냉각</p>
                </div>
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">로그 (Logarithmic)</p>
                  <p className="text-slate-600 dark:text-slate-400 font-mono text-xs mb-2">T<sub>k</sub> = T₀ / log(k+1)</p>
                  <p className="text-slate-600 dark:text-slate-400">이론적 최적해 보장</p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">적응형 (Adaptive)</p>
                  <p className="text-slate-600 dark:text-slate-400 font-mono text-xs mb-2">수용률 기반 조정</p>
                  <p className="text-slate-600 dark:text-slate-400">문제에 따라 자동 조절</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Particle Swarm Optimization */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            입자 군집 최적화 (Particle Swarm Optimization)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">새 떼의 집단 행동 모방</h3>
                <Bird className="w-8 h-8 opacity-80" />
              </div>
              <p className="text-emerald-100">
                새 떼나 물고기 떼의 <strong>집단 지능</strong>에서 영감을 받았습니다.
                각 입자가 자신의 최선 경험과 군집의 최선 경험을 따라 이동합니다.
              </p>
            </div>

            {/* PSO Update */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                PSO 업데이트 공식
              </h3>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4">
                <div className="space-y-3 font-mono text-sm text-slate-800 dark:text-slate-200">
                  <p><strong>속도 업데이트:</strong></p>
                  <p>v<sub>i</sub><sup>(t+1)</sup> = ωv<sub>i</sub><sup>(t)</sup> + c₁r₁(p<sub>i</sub> - x<sub>i</sub><sup>(t)</sup>) + c₂r₂(g - x<sub>i</sub><sup>(t)</sup>)</p>
                  <br />
                  <p><strong>위치 업데이트:</strong></p>
                  <p>x<sub>i</sub><sup>(t+1)</sup> = x<sub>i</sub><sup>(t)</sup> + v<sub>i</sub><sup>(t+1)</sup></p>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-1">ω (inertia weight)</p>
                  <p className="text-slate-600 dark:text-slate-400">관성: 이전 속도의 영향</p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-1">c₁ (cognitive)</p>
                  <p className="text-slate-600 dark:text-slate-400">개인 최선으로의 끌림</p>
                </div>
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-1">c₂ (social)</p>
                  <p className="text-slate-600 dark:text-slate-400">군집 최선으로의 끌림</p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-1">r₁, r₂</p>
                  <p className="text-slate-600 dark:text-slate-400">랜덤 수 [0, 1]</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Comparison */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            메타휴리스틱 비교
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-lg">
              <thead className="bg-gradient-to-r from-emerald-600 to-teal-700 text-white">
                <tr>
                  <th className="px-6 py-4 text-left">알고리즘</th>
                  <th className="px-6 py-4 text-left">영감</th>
                  <th className="px-6 py-4 text-left">탐색 방식</th>
                  <th className="px-6 py-4 text-left">장점</th>
                  <th className="px-6 py-4 text-left">단점</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-gray-700">
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">GA</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">생물 진화</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">개체군 기반</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">다양성 유지</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">느린 수렴</td>
                </tr>
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">SA</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">금속 담금질</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">단일 해</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">단순, 효과적</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">파라미터 민감</td>
                </tr>
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">PSO</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">새 떼 행동</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">군집 기반</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">빠른 수렴</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">조기 수렴 위험</td>
                </tr>
              </tbody>
            </table>
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
                경로 최적화
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• TSP (외판원 문제)</li>
                <li>• 차량 경로 문제 (VRP)</li>
                <li>• 배송 스케줄링</li>
                <li>• 네트워크 라우팅</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🤖</span>
                머신러닝
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• Neural Architecture Search</li>
                <li>• 하이퍼파라미터 최적화</li>
                <li>• Feature Selection</li>
                <li>• 모델 앙상블</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">📅</span>
                스케줄링
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• Job Shop Scheduling</li>
                <li>• Timetabling</li>
                <li>• 인력 배치</li>
                <li>• 자원 할당</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">⚙️</span>
                공학 설계
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 구조 최적화</li>
                <li>• 회로 설계</li>
                <li>• 안테나 설계</li>
                <li>• 파라미터 튜닝</li>
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
                <span>메타휴리스틱은 <strong>복잡한 NP-hard 문제</strong>에 효과적입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span><strong>GA</strong>는 다양성 유지, <strong>SA</strong>는 단순성, <strong>PSO</strong>는 빠른 수렴이 장점입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span>최적해를 <strong>보장하지 않지만</strong> 실용적으로 좋은 해를 찾습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span>파라미터 튜닝이 성능에 <strong>큰 영향</strong>을 미칩니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span>문제 특성에 맞는 <strong>적절한 알고리즘 선택</strong>이 중요합니다.</span>
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
              <strong>Chapter 7: 제약 최적화</strong>
              <br />
              Lagrange Multipliers, Penalty Methods, Barrier Methods 등
              제약 조건을 다루는 고급 기법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
