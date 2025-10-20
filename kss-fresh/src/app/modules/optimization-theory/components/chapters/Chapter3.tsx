'use client'

import React from 'react'
import { TrendingDown, Minimize2, Mountain, Activity, Zap, CheckCircle, ArrowRight, Code } from 'lucide-react'

export default function Chapter3() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-emerald-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl shadow-lg">
              <Mountain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                비선형 최적화
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Gradient Descent, Newton's Method, Quasi-Newton
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <TrendingDown className="w-6 h-6 text-emerald-600" />
              비선형 최적화란?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                비선형 최적화(Nonlinear Optimization)는 목적 함수나 제약 조건에 <strong>비선형 항</strong>이
                포함된 최적화 문제입니다. 머신러닝의 손실 함수 최소화, 신경망 학습 등
                현대 AI의 핵심 기술입니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">비선형 최적화의 도전 과제</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Mountain className="w-5 h-5 text-orange-600 mt-0.5 flex-shrink-0" />
                    <span><strong>지역 최적해</strong>: 전역 최적해를 보장할 수 없음</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Activity className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>복잡한 지형</strong>: 목적 함수가 매우 복잡한 형태</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>계산 비용</strong>: 반복 알고리즘의 높은 연산량</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Gradient Descent */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            경사하강법 (Gradient Descent)
          </h2>

          <div className="space-y-6">
            {/* Basic Concept */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">Gradient Descent의 핵심 원리</h3>
                <TrendingDown className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-blue-100">
                현재 위치에서 <strong>기울기(gradient)의 반대 방향</strong>으로 이동하여
                함수의 최솟값을 찾아갑니다. 가장 기본적이면서도 강력한 최적화 알고리즘입니다.
              </p>
            </div>

            {/* Update Rule */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                업데이트 공식
              </h3>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4">
                <div className="font-mono text-center text-lg text-slate-800 dark:text-slate-200">
                  x<sub>k+1</sub> = x<sub>k</sub> - α ∇f(x<sub>k</sub>)
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">x<sub>k</sub></p>
                  <p className="text-slate-600 dark:text-slate-400">현재 위치</p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">α (알파)</p>
                  <p className="text-slate-600 dark:text-slate-400">학습률 (step size)</p>
                </div>
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">∇f(x<sub>k</sub>)</p>
                  <p className="text-slate-600 dark:text-slate-400">기울기 벡터</p>
                </div>
              </div>
            </div>

            {/* Variants */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Gradient Descent의 변형
              </h3>

              <div className="space-y-4">
                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Batch Gradient Descent</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    전체 데이터셋을 사용하여 기울기 계산. 정확하지만 느림.
                  </p>
                </div>

                <div className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Stochastic Gradient Descent (SGD)</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    한 번에 하나의 샘플만 사용. 빠르지만 노이즈가 많음.
                  </p>
                </div>

                <div className="border-l-4 border-emerald-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Mini-batch Gradient Descent</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    작은 배치 단위로 기울기 계산. 속도와 안정성의 균형.
                  </p>
                </div>

                <div className="border-l-4 border-orange-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">Momentum</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    과거 기울기의 관성을 활용하여 수렴 속도 개선.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Newton's Method */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Newton's Method
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">2차 미분 정보 활용</h3>
                <Minimize2 className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-purple-100">
                Newton's Method는 <strong>Hessian 행렬</strong>(2차 미분)을 사용하여
                곡률 정보를 활용합니다. Gradient Descent보다 빠른 수렴을 보이지만
                계산 비용이 높습니다.
              </p>
            </div>

            {/* Update Rule */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Newton's Method 업데이트 공식
              </h3>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-4">
                <div className="font-mono text-center text-lg text-slate-800 dark:text-slate-200">
                  x<sub>k+1</sub> = x<sub>k</sub> - [∇<sup>2</sup>f(x<sub>k</sub>)]<sup>-1</sup> ∇f(x<sub>k</sub>)
                </div>
              </div>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <p>
                  <strong>∇<sup>2</sup>f(x)</strong>는 Hessian 행렬로, 2차 편미분으로 구성됩니다.
                </p>
                <p>
                  각 반복마다 2차 Taylor 전개를 통한 근사로 최적해에 빠르게 접근합니다.
                </p>
              </div>

              <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>⚠️ 주의:</strong> Hessian 행렬의 역행렬 계산이 필요하여
                  O(n³)의 계산 복잡도를 가집니다.
                </p>
              </div>
            </div>

            {/* Advantages and Disadvantages */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
                <h4 className="font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  장점
                </h4>
                <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                  <li>• 2차 수렴 속도 (quadratic convergence)</li>
                  <li>• 최적해 근처에서 매우 빠름</li>
                  <li>• 학습률 선택 불필요</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
                <h4 className="font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-orange-600" />
                  단점
                </h4>
                <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                  <li>• 높은 계산 비용 (O(n³))</li>
                  <li>• 메모리 요구량 많음</li>
                  <li>• Hessian이 singular일 경우 실패</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Quasi-Newton Methods */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Quasi-Newton 방법
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">Hessian 근사를 통한 효율성</h3>
                <Code className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-emerald-100">
                Quasi-Newton 방법은 Hessian 행렬을 <strong>근사</strong>하여 Newton's Method의
                효율성을 유지하면서 계산 비용을 크게 줄입니다.
              </p>
            </div>

            {/* BFGS */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                BFGS 알고리즘
              </h3>

              <p className="text-slate-700 dark:text-slate-300 mb-4">
                <strong>Broyden-Fletcher-Goldfarb-Shanno (BFGS)</strong>는 가장 널리 사용되는
                Quasi-Newton 방법입니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 mb-4">
                <h4 className="font-bold text-slate-800 dark:text-white mb-3">BFGS 업데이트 규칙</h4>
                <div className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                  <p>1. 탐색 방향: p<sub>k</sub> = -B<sub>k</sub><sup>-1</sup> ∇f(x<sub>k</sub>)</p>
                  <p>2. 선 탐색: α<sub>k</sub> = argmin f(x<sub>k</sub> + αp<sub>k</sub>)</p>
                  <p>3. 위치 업데이트: x<sub>k+1</sub> = x<sub>k</sub> + α<sub>k</sub>p<sub>k</sub></p>
                  <p>4. Hessian 근사 업데이트: B<sub>k+1</sub> (BFGS 공식 적용)</p>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">L-BFGS</p>
                  <p className="text-slate-600 dark:text-slate-400">
                    Limited-memory BFGS. 대규모 문제에 적합한 메모리 효율적 버전.
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">DFP</p>
                  <p className="text-slate-600 dark:text-slate-400">
                    Davidon-Fletcher-Powell. BFGS의 dual 형태.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Comparison */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            알고리즘 비교
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-lg">
              <thead className="bg-gradient-to-r from-emerald-600 to-teal-700 text-white">
                <tr>
                  <th className="px-6 py-4 text-left">알고리즘</th>
                  <th className="px-6 py-4 text-left">수렴 속도</th>
                  <th className="px-6 py-4 text-left">계산 비용</th>
                  <th className="px-6 py-4 text-left">메모리</th>
                  <th className="px-6 py-4 text-left">적용 분야</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-gray-700">
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">Gradient Descent</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">선형</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">낮음</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">O(n)</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">딥러닝</td>
                </tr>
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">Newton's Method</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">2차</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">매우 높음</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">O(n²)</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">작은 문제</td>
                </tr>
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">BFGS</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">Super-linear</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">중간</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">O(n²)</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">중규모</td>
                </tr>
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">L-BFGS</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">Super-linear</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">낮음</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">O(n)</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">대규모</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Real-world Applications */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            실전 응용 사례
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🧠</span>
                딥러닝 학습
              </h3>
              <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                신경망의 가중치를 최적화하여 손실 함수 최소화
              </p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• SGD + Momentum</li>
                <li>• Adam, RMSprop</li>
                <li>• 학습률 스케줄링</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">📊</span>
                통계 모델 적합
              </h3>
              <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                Maximum Likelihood Estimation (MLE)
              </p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• Logistic Regression</li>
                <li>• GLM (Generalized Linear Models)</li>
                <li>• Newton-Raphson</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🎯</span>
                하이퍼파라미터 튜닝
              </h3>
              <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                모델의 최적 하이퍼파라미터 탐색
              </p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• Bayesian Optimization</li>
                <li>• L-BFGS 기반 탐색</li>
                <li>• Gaussian Process</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">⚙️</span>
                공학 설계 최적화
              </h3>
              <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                제품 설계 파라미터 최적화
              </p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 구조 최적화</li>
                <li>• 공기역학 설계</li>
                <li>• 재료 선택</li>
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
                <span><strong>Gradient Descent</strong>는 가장 기본적인 1차 최적화 방법입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span><strong>Newton's Method</strong>는 2차 정보를 활용하여 빠르게 수렴합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span><strong>Quasi-Newton</strong>은 효율성과 수렴 속도의 균형을 맞춥니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span>딥러닝에서는 <strong>SGD와 변형</strong>이 널리 사용됩니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span><strong>L-BFGS</strong>는 대규모 문제에 메모리 효율적입니다.</span>
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
              <strong>Chapter 4: 볼록 최적화</strong>
              <br />
              볼록 함수와 볼록 집합의 특성을 이해하고 KKT 조건을 통한 최적해 판별 방법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
