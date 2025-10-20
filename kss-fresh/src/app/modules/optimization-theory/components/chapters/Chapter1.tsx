'use client'

import React from 'react'
import { TrendingUp, Target, Minimize, Maximize, ArrowDown, ArrowUp, Zap, CheckCircle } from 'lucide-react'

export default function Chapter1() {
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
                최적화 기초
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                최적화 문제의 정의와 분류
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-emerald-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-emerald-600" />
              최적화란 무엇인가?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                최적화(Optimization)는 주어진 제약 조건 하에서 목적 함수를 최대화 또는 최소화하는
                최적의 해를 찾는 수학적 과정입니다. AI, 머신러닝, 공학, 경제학 등 거의 모든 분야에서
                <strong>핵심적인 역할</strong>을 합니다.
              </p>

              <div className="bg-emerald-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-emerald-600">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">최적화의 핵심 요소</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Target className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
                    <span><strong>목적 함수(Objective Function)</strong>: 최대화 또는 최소화할 대상</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>결정 변수(Decision Variables)</strong>: 최적화할 변수들</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>제약 조건(Constraints)</strong>: 변수가 만족해야 할 조건</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Optimization Problem Types */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            최적화 문제의 분류
          </h2>

          <div className="space-y-4">
            {/* Linear Optimization */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">선형 최적화 (Linear Optimization)</h3>
                <ArrowUp className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-blue-100">
                목적 함수와 제약 조건이 모두 선형인 최적화 문제
              </p>
              <div className="bg-white/10 rounded-lg p-4 font-mono text-sm">
                <p>minimize: c<sup>T</sup>x</p>
                <p>subject to: Ax ≤ b, x ≥ 0</p>
              </div>
            </div>

            {/* Nonlinear Optimization */}
            <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">비선형 최적화 (Nonlinear Optimization)</h3>
                <ArrowDown className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-purple-100">
                목적 함수 또는 제약 조건에 비선형 항이 포함된 문제
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">경사하강법</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Newton 방법</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Quasi-Newton</span>
              </div>
            </div>

            {/* Convex Optimization */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">볼록 최적화 (Convex Optimization)</h3>
                <Minimize className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-emerald-100">
                목적 함수가 볼록 함수이고 제약 조건이 볼록 집합을 형성하는 문제
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">전역 최적해 보장</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">효율적 해법</span>
              </div>
            </div>

            {/* Integer Optimization */}
            <div className="bg-gradient-to-r from-orange-500 to-red-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">정수 최적화 (Integer Optimization)</h3>
                <Maximize className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-orange-100">
                결정 변수가 정수 값만 가질 수 있는 최적화 문제
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Branch and Bound</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Cutting Plane</span>
              </div>
            </div>
          </div>
        </section>

        {/* Mathematical Formulation */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            수학적 정식화
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
            <div className="mb-6">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">
                일반적인 최적화 문제
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                최적화 문제는 다음과 같은 표준 형태로 표현됩니다:
              </p>
            </div>

            <div className="bg-emerald-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
              <div className="space-y-2 text-slate-800 dark:text-slate-200">
                <p className="font-bold">minimize   f(x)</p>
                <p>subject to  g<sub>i</sub>(x) ≤ 0,  i = 1, ..., m</p>
                <p className="ml-12">h<sub>j</sub>(x) = 0,  j = 1, ..., p</p>
                <p className="ml-12">x ∈ X</p>
                <br />
                <p className="text-emerald-700 dark:text-emerald-400">여기서:</p>
                <p className="ml-4">• f(x): 목적 함수</p>
                <p className="ml-4">• g<sub>i</sub>(x): 부등식 제약 조건</p>
                <p className="ml-4">• h<sub>j</sub>(x): 등식 제약 조건</p>
                <p className="ml-4">• X: 정의역 (feasible region)</p>
              </div>
            </div>
          </div>
        </section>

        {/* Real-world Examples */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            실전 응용 사례
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Machine Learning */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">🤖</span>
                머신러닝
              </h3>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li>• 손실 함수 최소화 (Loss Minimization)</li>
                <li>• 하이퍼파라미터 튜닝</li>
                <li>• Neural Architecture Search</li>
                <li>• 모델 압축 (Pruning, Quantization)</li>
              </ul>
            </div>

            {/* Operations Research */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">📦</span>
                산업 공학
              </h3>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li>• 생산 계획 최적화</li>
                <li>• 물류 네트워크 설계</li>
                <li>• 재고 관리</li>
                <li>• 스케줄링</li>
              </ul>
            </div>

            {/* Finance */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">💰</span>
                금융
              </h3>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li>• 포트폴리오 최적화</li>
                <li>• 위험 관리 (Risk Management)</li>
                <li>• 옵션 가격 결정</li>
                <li>• 자산 배분</li>
              </ul>
            </div>

            {/* Engineering */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-emerald-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
                <span className="text-2xl">⚙️</span>
                공학
              </h3>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li>• 구조 설계 최적화</li>
                <li>• 에너지 효율 최적화</li>
                <li>• 제어 시스템 설계</li>
                <li>• 신호 처리</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Key Concepts */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-emerald-600 to-teal-700 rounded-2xl p-8 text-white shadow-xl">
            <h2 className="text-2xl font-bold mb-6">
              핵심 개념
            </h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
                <h3 className="font-bold text-lg mb-3">1. 전역 최적해 vs 지역 최적해</h3>
                <p className="text-emerald-100 text-sm">
                  전역 최적해(Global Optimum)는 모든 가능한 해 중 최선의 해이며,
                  지역 최적해(Local Optimum)는 특정 영역 내에서만 최선인 해입니다.
                </p>
              </div>

              <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
                <h3 className="font-bold text-lg mb-3">2. 실행 가능 영역 (Feasible Region)</h3>
                <p className="text-emerald-100 text-sm">
                  모든 제약 조건을 만족하는 결정 변수 값들의 집합입니다.
                  최적해는 반드시 이 영역 내에 존재합니다.
                </p>
              </div>

              <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
                <h3 className="font-bold text-lg mb-3">3. KKT 조건</h3>
                <p className="text-emerald-100 text-sm">
                  Karush-Kuhn-Tucker 조건은 제약이 있는 최적화 문제의 최적해가
                  만족해야 하는 필요조건입니다.
                </p>
              </div>

              <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
                <h3 className="font-bold text-lg mb-3">4. 볼록성 (Convexity)</h3>
                <p className="text-emerald-100 text-sm">
                  볼록 최적화 문제는 지역 최적해가 곧 전역 최적해이므로
                  효율적으로 해를 찾을 수 있습니다.
                </p>
              </div>
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
                <span>최적화는 <strong>목적 함수, 결정 변수, 제약 조건</strong>으로 구성됩니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">2.</span>
                <span>주요 분류: <strong>선형, 비선형, 볼록, 정수 최적화</strong></span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">3.</span>
                <span><strong>볼록 최적화</strong>는 전역 최적해를 효율적으로 찾을 수 있습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">4.</span>
                <span>실전 응용: <strong>머신러닝, 금융, 공학, 산업 공학</strong></span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-emerald-600 text-xl font-bold">5.</span>
                <span>KKT 조건은 제약 최적화 문제의 최적해 판별에 사용됩니다.</span>
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
              <strong>Chapter 2: 선형 최적화</strong>
              <br />
              Simplex 알고리즘과 Interior Point 방법을 통한 선형 계획 문제의 효율적 해법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
