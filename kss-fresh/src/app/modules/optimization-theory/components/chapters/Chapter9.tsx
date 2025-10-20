'use client'

import React from 'react'
import { Settings, Grid3x3, Shuffle, TrendingUp, Zap, Brain, Target, Award } from 'lucide-react'

export default function Chapter9() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-gray-900 dark:to-emerald-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-600 to-teal-700 rounded-xl shadow-lg">
              <Settings className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-700 bg-clip-text text-transparent">
                하이퍼파라미터 튜닝
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Grid Search, Random Search, Bayesian Optimization
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Brain className="w-6 h-6 text-emerald-600" />
              하이퍼파라미터 튜닝이란?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                하이퍼파라미터 튜닝(Hyperparameter Tuning)은 머신러닝 모델의 <strong>하이퍼파라미터</strong>를
                최적화하여 모델 성능을 극대화하는 과정입니다. 학습 알고리즘이 직접 학습하지 않는 파라미터를
                <strong>최적화 기법</strong>으로 찾아야 합니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">하이퍼파라미터 vs 파라미터</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Settings className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span><strong>하이퍼파라미터</strong>: 학습 전에 설정 (학습률, 배치 크기, 트리 깊이 등)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>파라미터</strong>: 학습 중에 업데이트 (가중치, 편향 등)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>중요성</strong>: 하이퍼파라미터가 모델 성능에 큰 영향</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Common Hyperparameters */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            주요 하이퍼파라미터
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🧠</span>
                신경망
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 학습률 (Learning Rate)</li>
                <li>• 배치 크기 (Batch Size)</li>
                <li>• 레이어 수, 뉴런 수</li>
                <li>• Dropout 비율</li>
                <li>• 활성화 함수</li>
                <li>• 옵티마이저 (Adam, SGD 등)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🌳</span>
                트리 기반 모델
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 트리 깊이 (Max Depth)</li>
                <li>• 트리 개수 (n_estimators)</li>
                <li>• 최소 샘플 수 (min_samples_split)</li>
                <li>• 특성 개수 (max_features)</li>
                <li>• 학습률 (Boosting)</li>
                <li>• Subsample 비율</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🎯</span>
                SVM
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• C (정규화 파라미터)</li>
                <li>• Kernel (RBF, Linear 등)</li>
                <li>• Gamma (RBF kernel 폭)</li>
                <li>• Degree (Polynomial kernel)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🔍</span>
                K-Means
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 클러스터 개수 (k)</li>
                <li>• 초기화 방법 (k-means++)</li>
                <li>• 최대 반복 횟수</li>
                <li>• 거리 측정 방법</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Grid Search */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Grid Search (격자 탐색)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">체계적 전수 조사</h3>
                <Grid3x3 className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-blue-100">
                가능한 모든 하이퍼파라미터 조합을 <strong>체계적으로 시도</strong>하는 방법입니다.
                철저하지만 계산 비용이 매우 높습니다.
              </p>
            </div>

            {/* How it works */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Grid Search 작동 원리
              </h3>

              <div className="space-y-4">
                <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-2">예제: Random Forest 튜닝</h4>
                  <div className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                    <p>파라미터 그리드:</p>
                    <div className="bg-white dark:bg-gray-900 rounded p-3 font-mono text-xs">
                      <p>n_estimators: [100, 200, 300]</p>
                      <p>max_depth: [10, 20, 30]</p>
                      <p>min_samples_split: [2, 5, 10]</p>
                    </div>
                    <p className="mt-2">
                      <strong>총 조합 수:</strong> 3 × 3 × 3 = 27개
                    </p>
                    <p>
                      <strong>5-fold CV 사용 시:</strong> 27 × 5 = 135번 학습
                    </p>
                  </div>
                </div>

                <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                  <div className="flex items-start gap-3">
                    <span className="font-bold text-emerald-600">1.</span>
                    <div>파라미터 그리드 정의</div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="font-bold text-emerald-600">2.</span>
                    <div>모든 조합에 대해 교차 검증</div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="font-bold text-emerald-600">3.</span>
                    <div>각 조합의 평균 성능 기록</div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="font-bold text-emerald-600">4.</span>
                    <div>최고 성능의 조합 선택</div>
                  </div>
                </div>
              </div>

              <div className="mt-4 grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-1">✅ 장점</p>
                  <ul className="text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• 단순하고 이해하기 쉬움</li>
                    <li>• 모든 조합 탐색 (완전성)</li>
                    <li>• 병렬화 용이</li>
                  </ul>
                </div>
                <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-1">❌ 단점</p>
                  <ul className="text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• 지수적 계산 비용</li>
                    <li>• 차원의 저주</li>
                    <li>• 불필요한 영역도 탐색</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Random Search */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Random Search (무작위 탐색)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">효율적인 무작위 샘플링</h3>
                <Shuffle className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-4 text-purple-100">
                하이퍼파라미터 공간에서 <strong>무작위로 샘플링</strong>하여 평가합니다.
                Grid Search보다 효율적이며, 중요한 파라미터를 더 잘 탐색합니다.
              </p>
            </div>

            {/* Why Random Search Works */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Random Search의 효과
              </h3>

              <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4 mb-4">
                <h4 className="font-bold text-slate-800 dark:text-white mb-2">핵심 통찰력</h4>
                <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">
                  실제로 <strong>중요한 하이퍼파라미터는 소수</strong>입니다.
                  Random Search는 중요한 파라미터를 더 다양하게 탐색합니다.
                </p>
                <div className="text-sm text-slate-700 dark:text-slate-300">
                  <p className="mb-2"><strong>예:</strong> 2개 파라미터, 각각 9개 값</p>
                  <p>• Grid Search: 9 × 9 = 81개 조합, 각 파라미터당 9개 값 시도</p>
                  <p>• Random Search: 81번 샘플링, 각 파라미터당 ~81개 다른 값 시도!</p>
                </div>
              </div>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">1.</span>
                  <div>파라미터 분포 정의 (uniform, log-uniform 등)</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">2.</span>
                  <div>설정한 횟수(n_iter)만큼 무작위 샘플링</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">3.</span>
                  <div>각 샘플에 대해 교차 검증</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">4.</span>
                  <div>최고 성능의 조합 선택</div>
                </div>
              </div>

              <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>💡 Tip:</strong> 연속형 파라미터(학습률 등)는 log-uniform 분포 사용 권장.
                  예: 10<sup>-5</sup> ~ 10<sup>-1</sup> 범위
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Bayesian Optimization */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Bayesian Optimization (베이지안 최적화)
          </h2>

          <div className="space-y-6">
            {/* Overview */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">지능적 탐색 전략</h3>
                <Brain className="w-8 h-8 opacity-80" />
              </div>
              <p className="text-emerald-100">
                이전 평가 결과를 활용하여 <strong>다음 탐색 위치를 지능적으로 선택</strong>합니다.
                비싼 목적 함수 평가를 최소화하면서 최적해를 찾습니다.
              </p>
            </div>

            {/* How it works */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Bayesian Optimization 핵심 구성 요소
              </h3>

              <div className="space-y-4">
                <div className="border-l-4 border-emerald-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">1. Surrogate Model (대리 모델)</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    목적 함수의 확률적 모델. 일반적으로 <strong>Gaussian Process (GP)</strong> 사용.
                  </p>
                  <div className="bg-emerald-50 dark:bg-gray-900 rounded p-3 text-xs">
                    <p>평가한 점: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)</p>
                    <p>GP로 f(x)의 분포 모델링: f(x) ~ GP(μ(x), σ²(x))</p>
                  </div>
                </div>

                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-bold text-slate-800 dark:text-white mb-1">2. Acquisition Function (획득 함수)</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    다음 평가할 점을 선택하는 기준. <strong>Exploration</strong>과 <strong>Exploitation</strong> 균형.
                  </p>
                  <div className="grid md:grid-cols-2 gap-2 text-xs">
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-2">
                      <p className="font-bold">EI (Expected Improvement)</p>
                      <p className="text-slate-600 dark:text-slate-400">가장 일반적</p>
                    </div>
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-2">
                      <p className="font-bold">UCB (Upper Confidence Bound)</p>
                      <p className="text-slate-600 dark:text-slate-400">탐색 강조</p>
                    </div>
                    <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-2">
                      <p className="font-bold">PI (Probability of Improvement)</p>
                      <p className="text-slate-600 dark:text-slate-400">개선 확률</p>
                    </div>
                    <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded p-2">
                      <p className="font-bold">TS (Thompson Sampling)</p>
                      <p className="text-slate-600 dark:text-slate-400">확률적 샘플링</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Algorithm */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Bayesian Optimization 알고리즘
              </h3>

              <div className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">1.</span>
                  <div>초기 점 몇 개를 무작위 또는 Grid로 평가</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">2.</span>
                  <div>평가 결과로 Gaussian Process 학습</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">3.</span>
                  <div>Acquisition Function을 최적화하여 다음 평가 점 선택</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">4.</span>
                  <div>선택된 점에서 목적 함수 평가</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">5.</span>
                  <div>평가 결과를 데이터에 추가하고 2-4 반복</div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-bold text-emerald-600">6.</span>
                  <div>예산 소진 또는 수렴 시 종료</div>
                </div>
              </div>

              <div className="mt-4 grid md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-1">✅ 장점</p>
                  <ul className="text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• 적은 평가로 최적해 발견</li>
                    <li>• 비싼 목적 함수에 효과적</li>
                    <li>• 연속형 파라미터에 강점</li>
                    <li>• 과거 정보 활용</li>
                  </ul>
                </div>
                <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
                  <p className="font-bold text-slate-800 dark:text-white mb-1">❌ 단점</p>
                  <ul className="text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• 구현 복잡</li>
                    <li>• 고차원에서 성능 저하</li>
                    <li>• GP 학습 비용</li>
                    <li>• 범주형 변수 처리 어려움</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Comparison */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            방법론 비교
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-lg">
              <thead className="bg-gradient-to-r from-emerald-600 to-teal-700 text-white">
                <tr>
                  <th className="px-6 py-4 text-left">방법</th>
                  <th className="px-6 py-4 text-left">효율성</th>
                  <th className="px-6 py-4 text-left">구현 난이도</th>
                  <th className="px-6 py-4 text-left">병렬화</th>
                  <th className="px-6 py-4 text-left">권장 사용</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-gray-700">
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">Grid Search</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">낮음</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">매우 쉬움</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">완벽</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">소규모, 확실성 필요</td>
                </tr>
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">Random Search</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">중간</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">쉬움</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">완벽</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">중규모, 일반적</td>
                </tr>
                <tr className="hover:bg-emerald-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4 font-bold text-slate-800 dark:text-white">Bayesian Opt</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">높음</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">어려움</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">어려움</td>
                  <td className="px-6 py-4 text-slate-600 dark:text-slate-400">비싼 평가, 저차원</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Practical Tips */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            실전 팁
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🎯</span>
                탐색 범위 설정
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• 기본값 주변부터 시작</li>
                <li>• log-scale로 넓게 탐색</li>
                <li>• 초기에는 넓게, 점차 좁히기</li>
                <li>• 문헌/경험 참고</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">⚡</span>
                효율성 향상
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• Early stopping 활용</li>
                <li>• 작은 데이터로 사전 탐색</li>
                <li>• 병렬 처리 최대 활용</li>
                <li>• 중요 파라미터 우선</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">📊</span>
                검증 전략
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• K-fold Cross Validation</li>
                <li>• Stratified 샘플링</li>
                <li>• 별도 테스트 세트 유지</li>
                <li>• 과적합 모니터링</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🛠️</span>
                도구 활용
              </h3>
              <ul className="text-sm text-slate-700 dark:text-slate-300 space-y-2">
                <li>• scikit-learn: GridSearchCV</li>
                <li>• Optuna: Bayesian Opt</li>
                <li>• Ray Tune: 분산 튜닝</li>
                <li>• Hyperopt, SMAC</li>
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
                <span>하이퍼파라미터 튜닝은 <strong>모델 성능 극대화</strong>의 핵심입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span><strong>Grid Search</strong>는 완전하지만 비효율적, <strong>Random Search</strong>가 일반적으로 더 효율적입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span><strong>Bayesian Optimization</strong>은 비싼 평가에 최적이지만 구현이 복잡합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span><strong>교차 검증</strong>으로 과적합을 방지해야 합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span>문제 특성에 맞는 <strong>적절한 방법 선택</strong>이 중요합니다.</span>
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
              <strong>Chapter 10: 실전 응용</strong>
              <br />
              실제 프로젝트에서의 최적화 문제 정식화, 구현, 배포까지
              종합적인 실전 가이드를 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
