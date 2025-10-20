'use client'

import React from 'react'
import ChapterNavigation from '../ChapterNavigation'

export default function Chapter5() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Chapter 5: 궤적 생성 (Trajectory Generation)</h1>
        <p className="text-xl text-white/90">
          시간 변수를 포함한 부드럽고 효율적인 로봇 움직임 생성
        </p>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        {/* 1. 궤적 생성 개요 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1. 궤적 생성이란?
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-600 p-6 mb-6">
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300 mb-3">
              경로(Path) vs 궤적(Trajectory)
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full border border-blue-300 dark:border-blue-700 mt-3">
                <thead className="bg-blue-100 dark:bg-blue-900/50">
                  <tr>
                    <th className="px-4 py-2 border-b text-left">구분</th>
                    <th className="px-4 py-2 border-b text-left">경로 (Path)</th>
                    <th className="px-4 py-2 border-b text-left">궤적 (Trajectory)</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  <tr className="border-b">
                    <td className="px-4 py-2 font-semibold">정의</td>
                    <td className="px-4 py-2">공간상의 점들의 순서</td>
                    <td className="px-4 py-2">시간에 따른 위치/속도/가속도</td>
                  </tr>
                  <tr className="border-b">
                    <td className="px-4 py-2 font-semibold">시간 정보</td>
                    <td className="px-4 py-2">❌ 없음</td>
                    <td className="px-4 py-2">✅ 포함 (t → q(t))</td>
                  </tr>
                  <tr className="border-b">
                    <td className="px-4 py-2 font-semibold">동역학 고려</td>
                    <td className="px-4 py-2">❌ 불필요</td>
                    <td className="px-4 py-2">✅ 필수 (속도/가속도 제한)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2 font-semibold">예시</td>
                    <td className="px-4 py-2">A*로 찾은 웨이포인트 목록</td>
                    <td className="px-4 py-2">각 순간의 관절 각도/속도/가속도</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            궤적 생성의 목표
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold text-lg text-green-900 dark:text-green-300 mb-2">
                ✅ 부드러움 (Smoothness)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                급격한 속도 변화 없이 연속적인 가속도 프로파일 생성
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border border-purple-300 dark:border-purple-700">
              <h4 className="font-bold text-lg text-purple-900 dark:text-purple-300 mb-2">
                ⚡ 동역학 제약 준수
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                최대 속도, 가속도, 저크(Jerk) 한계 내에서 움직임
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border border-orange-300 dark:border-orange-700">
              <h4 className="font-bold text-lg text-orange-900 dark:text-orange-300 mb-2">
                🎯 정확성
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                지정된 웨이포인트를 정확히 통과
              </p>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border border-blue-300 dark:border-blue-700">
              <h4 className="font-bold text-lg text-blue-900 dark:text-blue-300 mb-2">
                ⏱️ 시간 최적화
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                제약 조건 내에서 가능한 빠른 이동
              </p>
            </div>
          </div>
        </section>

        {/* 2. 다항식 보간 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            2. 다항식 보간 (Polynomial Interpolation)
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            다항식 보간은 웨이포인트 사이를 부드러운 다항식 함수로 연결하는 가장 기본적인 궤적 생성 방법입니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            3차 다항식 (Cubic Polynomial)
          </h3>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-300 dark:border-blue-700 mb-6">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              시작점 q₀, 도착점 q₁을 시간 T 동안 연결하는 3차 다항식:
            </p>
            <div className="bg-white dark:bg-gray-900 p-4 rounded font-mono text-sm mb-4">
              <p className="mb-2">q(t) = a₀ + a₁·t + a₂·t² + a₃·t³</p>
              <p className="mb-2">q̇(t) = a₁ + 2a₂·t + 3a₃·t²</p>
              <p>q̈(t) = 2a₂ + 6a₃·t</p>
            </div>

            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300">경계 조건 (4개):</h4>
            <div className="space-y-2 text-sm font-mono bg-white dark:bg-gray-900 p-4 rounded">
              <p>1. q(0) = q₀ (시작 위치)</p>
              <p>2. q(T) = q₁ (도착 위치)</p>
              <p>3. q̇(0) = v₀ (시작 속도)</p>
              <p>4. q̇(T) = v₁ (도착 속도)</p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            계수 계산 예제
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <p className="font-semibold mb-3 text-gray-900 dark:text-white">
              예제: q₀ = 0, q₁ = 10, v₀ = 0, v₁ = 0, T = 2초
            </p>
            <div className="space-y-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <p className="font-semibold mb-2">행렬 방정식:</p>
                <pre className="text-xs font-mono overflow-x-auto">
{`┌           ┐   ┌    ┐   ┌    ┐
│ 1  0  0  0│   │ a₀ │   │ 0  │
│ 1  T  T² T³│ · │ a₁ │ = │ 10 │
│ 0  1  0  0│   │ a₂ │   │ 0  │
│ 0  1 2T 3T²│   │ a₃ │   │ 0  │
└           ┘   └    ┘   └    ┘`}
                </pre>
              </div>

              <div className="bg-green-100 dark:bg-green-900/30 p-4 rounded border border-green-400">
                <p className="font-semibold mb-2 text-green-900 dark:text-green-300">해:</p>
                <p className="font-mono text-xs">
                  a₀ = 0, a₁ = 0, a₂ = 3.75, a₃ = -1.25
                </p>
                <p className="font-mono text-xs mt-2">
                  <strong>q(t) = 3.75t² - 1.25t³</strong>
                </p>
              </div>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            5차 다항식 (Quintic Polynomial)
          </h3>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg border border-purple-300 dark:border-purple-700">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              3차보다 부드러운 움직임을 위해 <strong>가속도 경계 조건</strong>까지 지정:
            </p>
            <div className="bg-white dark:bg-gray-900 p-4 rounded font-mono text-sm mb-4">
              <p>q(t) = a₀ + a₁·t + a₂·t² + a₃·t³ + a₄·t⁴ + a₅·t⁵</p>
            </div>
            <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-300">경계 조건 (6개):</h4>
            <div className="grid grid-cols-2 gap-2 text-sm font-mono">
              <div className="bg-white dark:bg-gray-900 p-3 rounded">q(0) = q₀</div>
              <div className="bg-white dark:bg-gray-900 p-3 rounded">q(T) = q₁</div>
              <div className="bg-white dark:bg-gray-900 p-3 rounded">q̇(0) = v₀</div>
              <div className="bg-white dark:bg-gray-900 p-3 rounded">q̇(T) = v₁</div>
              <div className="bg-white dark:bg-gray-900 p-3 rounded">q̈(0) = a₀</div>
              <div className="bg-white dark:bg-gray-900 p-3 rounded">q̈(T) = a₁</div>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
              💡 장점: 가속도가 연속적이어서 저크(Jerk) 최소화 → 진동 감소
            </p>
          </div>
        </section>

        {/* 3. 스플라인 궤적 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3. 스플라인 궤적 (Spline Trajectory)
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            여러 웨이포인트를 통과해야 할 때, 각 구간을 다항식으로 연결하면서
            연결점에서 연속성을 보장하는 방법입니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            Cubic Spline (3차 스플라인)
          </h3>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-300 dark:border-blue-700 mb-6">
            <h4 className="font-semibold mb-3 text-blue-900 dark:text-blue-300">
              n개의 웨이포인트 [q₀, q₁, ..., qₙ₋₁]
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              각 구간 [qᵢ, qᵢ₊₁]을 3차 다항식으로 연결하되, 연결점에서:
            </p>
            <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>C⁰ 연속성</strong>: 위치가 연속 (qᵢ(T) = qᵢ₊₁(0))</li>
              <li><strong>C¹ 연속성</strong>: 속도가 연속 (q̇ᵢ(T) = q̇ᵢ₊₁(0))</li>
              <li><strong>C² 연속성</strong>: 가속도가 연속 (q̈ᵢ(T) = q̈ᵢ₊₁(0))</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            스플라인 종류별 특징
          </h3>
          <div className="space-y-4">
            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold text-lg text-green-900 dark:text-green-300 mb-2">
                Natural Cubic Spline
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                경계 조건: 양 끝점에서 2차 미분(가속도)이 0
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                ✅ 자연스러운 움직임 | ⚠️ 끝점에서 가속도 제어 불가
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border border-purple-300 dark:border-purple-700">
              <h4 className="font-bold text-lg text-purple-900 dark:text-purple-300 mb-2">
                Clamped Cubic Spline
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                경계 조건: 양 끝점에서 1차 미분(속도) 지정
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                ✅ 속도 제어 가능 | ✅ 실용적
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border border-orange-300 dark:border-orange-700">
              <h4 className="font-bold text-lg text-orange-900 dark:text-orange-300 mb-2">
                B-Spline
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                제어점 변경 시 지역적 영향만 (전체 재계산 불필요)
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                ✅ 실시간 수정 용이 | ✅ CAD/그래픽스에서 표준 | ⚠️ 제어점 통과 안 함
              </p>
            </div>
          </div>
        </section>

        {/* 4. S-Curve 프로파일 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            4. S-Curve 프로파일 (Trapezoidal with S-Curve)
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            산업용 로봇에서 가장 널리 사용되는 속도 프로파일로, 저크(Jerk)를 제한하여
            기계적 충격과 진동을 최소화합니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            Trapezoidal 속도 프로파일 (기본형)
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <h4 className="font-semibold mb-3 text-gray-900 dark:text-white">3단계 구조:</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm mb-4">
              <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded border border-blue-400">
                <p className="font-semibold text-blue-900 dark:text-blue-300">1. 가속 (Acceleration)</p>
                <p className="text-xs text-gray-700 dark:text-gray-300 mt-1">0 → v_max</p>
              </div>
              <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded border border-green-400">
                <p className="font-semibold text-green-900 dark:text-green-300">2. 등속 (Constant)</p>
                <p className="text-xs text-gray-700 dark:text-gray-300 mt-1">v_max 유지</p>
              </div>
              <div className="bg-red-100 dark:bg-red-900/30 p-3 rounded border border-red-400">
                <p className="font-semibold text-red-900 dark:text-red-300">3. 감속 (Deceleration)</p>
                <p className="text-xs text-gray-700 dark:text-gray-300 mt-1">v_max → 0</p>
              </div>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              ⚠️ 문제점: 가속도가 불연속 → 무한대 저크 → 기계적 충격
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            S-Curve 프로파일 (개선형)
          </h3>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-300 dark:border-blue-700 mb-6">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              저크를 제한하여 가속도가 부드럽게 변하도록 개선:
            </p>
            <h4 className="font-semibold mb-3 text-blue-900 dark:text-blue-300">7단계 구조:</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-3">
                <span className="font-mono w-8 text-center bg-blue-200 dark:bg-blue-800 rounded px-2 py-1">1</span>
                <span>Jerk-up (저크 양수) → 가속도 증가</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono w-8 text-center bg-blue-200 dark:bg-blue-800 rounded px-2 py-1">2</span>
                <span>일정 가속도 (a_max 유지)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono w-8 text-center bg-blue-200 dark:bg-blue-800 rounded px-2 py-1">3</span>
                <span>Jerk-down (저크 음수) → 가속도 감소 → 0</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono w-8 text-center bg-green-200 dark:bg-green-800 rounded px-2 py-1">4</span>
                <span>등속 구간 (v_max 유지, a = 0)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono w-8 text-center bg-red-200 dark:bg-red-800 rounded px-2 py-1">5</span>
                <span>Jerk-down (감속 시작, a 음수로)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono w-8 text-center bg-red-200 dark:bg-red-800 rounded px-2 py-1">6</span>
                <span>일정 감속도 (-a_max 유지)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono w-8 text-center bg-red-200 dark:bg-red-800 rounded px-2 py-1">7</span>
                <span>Jerk-up (감속도 완화 → 0)</span>
              </div>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            S-Curve 수식
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              저크 제한 j_max 하에서 각 단계의 지속 시간:
            </p>
            <div className="font-mono text-xs bg-white dark:bg-gray-800 p-4 rounded space-y-2">
              <p>T_jerk = a_max / j_max (저크 구간 시간)</p>
              <p>T_acc = v_max / a_max (가속 구간 총 시간)</p>
              <p>T_const = (Distance - v_max·T_acc) / v_max (등속 구간)</p>
              <p className="text-green-600 dark:text-green-400 mt-3">
                <strong>Total Time = 2·T_acc + T_const</strong>
              </p>
            </div>
          </div>
        </section>

        {/* 5. 시간 최적 궤적 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            5. 시간 최적 궤적 (Time-Optimal Trajectory)
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            주어진 제약 조건(속도/가속도 한계) 내에서 가장 빠르게 이동하는 궤적을 찾는 문제입니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            Bang-Bang Control
          </h3>
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg border border-green-300 dark:border-green-700 mb-6">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              최단 시간 달성을 위해 가속도를 항상 <strong>최대 또는 최소</strong>로 유지:
            </p>
            <div className="bg-white dark:bg-gray-900 p-4 rounded font-mono text-sm mb-3">
              <p>a(t) = +a_max (가속 시)</p>
              <p>a(t) = -a_max (감속 시)</p>
              <p>a(t) = 0 (등속 시, 선택적)</p>
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              <p>✅ <strong>장점</strong>: 이론적 최단 시간</p>
              <p>⚠️ <strong>단점</strong>: 불연속 가속도 → 높은 저크 → 기계적 부담</p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            최적 제어 접근법
          </h3>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg border border-purple-300 dark:border-purple-700">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              비용 함수를 최소화하는 최적화 문제로 변환:
            </p>
            <div className="bg-white dark:bg-gray-900 p-4 rounded mb-4">
              <p className="font-mono text-sm mb-3">
                minimize: J = ∫₀ᵀ 1 dt = T (이동 시간)
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">제약 조건:</p>
              <div className="font-mono text-xs space-y-1 ml-4">
                <p>|q̇(t)| ≤ v_max</p>
                <p>|q̈(t)| ≤ a_max</p>
                <p>q(0) = q₀, q(T) = q₁</p>
              </div>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              💡 해법: Pontryagin's Maximum Principle, Dynamic Programming
            </p>
          </div>
        </section>

        {/* 6. 실전 구현 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            6. Python 구현 예제
          </h2>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            3차 다항식 궤적 생성
          </h3>
          <div className="bg-gray-900 text-gray-100 p-6 rounded-lg overflow-x-auto mb-6">
            <pre className="text-sm font-mono">
{`import numpy as np
import matplotlib.pyplot as plt

def cubic_trajectory(q0, q1, v0, v1, T, dt=0.01):
    """
    3차 다항식 궤적 생성

    Parameters:
    -----------
    q0, q1 : float
        시작/도착 위치
    v0, v1 : float
        시작/도착 속도
    T : float
        이동 시간
    dt : float
        샘플링 간격

    Returns:
    --------
    t, q, v, a : arrays
        시간, 위치, 속도, 가속도 배열
    """
    # 계수 행렬
    A = np.array([
        [1, 0, 0, 0],
        [1, T, T**2, T**3],
        [0, 1, 0, 0],
        [0, 1, 2*T, 3*T**2]
    ])

    # 경계 조건
    b = np.array([q0, q1, v0, v1])

    # 계수 계산
    coeffs = np.linalg.solve(A, b)
    a0, a1, a2, a3 = coeffs

    # 시간 배열
    t = np.arange(0, T + dt, dt)

    # 위치
    q = a0 + a1*t + a2*t**2 + a3*t**3

    # 속도
    v = a1 + 2*a2*t + 3*a3*t**2

    # 가속도
    a = 2*a2 + 6*a3*t

    return t, q, v, a

# 사용 예제
t, q, v, a = cubic_trajectory(
    q0=0, q1=10,
    v0=0, v1=0,
    T=2.0
)

# 시각화
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(t, q, 'b-', linewidth=2)
axes[0].set_ylabel('Position (m)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, v, 'g-', linewidth=2)
axes[1].set_ylabel('Velocity (m/s)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, a, 'r-', linewidth=2)
axes[2].set_ylabel('Acceleration (m/s²)')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()`}
            </pre>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            S-Curve 프로파일 구현
          </h3>
          <div className="bg-gray-900 text-gray-100 p-6 rounded-lg overflow-x-auto">
            <pre className="text-sm font-mono">
{`def s_curve_profile(distance, v_max, a_max, j_max, dt=0.01):
    """
    S-Curve 속도 프로파일 생성

    Parameters:
    -----------
    distance : float
        이동 거리
    v_max : float
        최대 속도
    a_max : float
        최대 가속도
    j_max : float
        최대 저크
    """
    # 시간 계산
    T_jerk = a_max / j_max
    T_acc = v_max / a_max
    T_const = distance / v_max - T_acc

    # 총 시간
    T_total = 2 * T_acc + T_const

    t = np.arange(0, T_total + dt, dt)
    v = np.zeros_like(t)
    a = np.zeros_like(t)
    j = np.zeros_like(t)

    for i, time in enumerate(t):
        if time < T_jerk:
            # Phase 1: Jerk-up
            j[i] = j_max
            a[i] = j_max * time
            v[i] = 0.5 * j_max * time**2

        elif time < T_acc - T_jerk:
            # Phase 2: Constant acceleration
            j[i] = 0
            a[i] = a_max
            v[i] = a_max * (time - 0.5*T_jerk)

        elif time < T_acc:
            # Phase 3: Jerk-down
            dt_phase = time - (T_acc - T_jerk)
            j[i] = -j_max
            a[i] = a_max - j_max * dt_phase
            v[i] = v_max - 0.5*j_max*dt_phase**2

        elif time < T_acc + T_const:
            # Phase 4: Constant velocity
            j[i] = 0
            a[i] = 0
            v[i] = v_max

        # Phases 5-7: 감속 (대칭)
        # ... (가속의 역순)

    return t, v, a, j

# 사용 예제
t, v, a, j = s_curve_profile(
    distance=100,
    v_max=10,
    a_max=5,
    j_max=50
)`}
            </pre>
          </div>
        </section>

        {/* 7. 요약 */}
        <section className="mb-12 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-8 border border-orange-300 dark:border-orange-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            📌 핵심 요약
          </h2>

          <div className="overflow-x-auto mb-6">
            <table className="min-w-full border border-gray-300 dark:border-gray-600">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-3 border-b text-left">방법</th>
                  <th className="px-4 py-3 border-b text-left">특징</th>
                  <th className="px-4 py-3 border-b text-left">적합한 상황</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">3차 다항식</td>
                  <td className="px-4 py-3">간단, 위치+속도 경계</td>
                  <td className="px-4 py-3">Point-to-Point 이동</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">5차 다항식</td>
                  <td className="px-4 py-3">부드러움, 가속도 제어</td>
                  <td className="px-4 py-3">정밀 작업, 저진동 요구</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">Cubic Spline</td>
                  <td className="px-4 py-3">다중 웨이포인트, C² 연속</td>
                  <td className="px-4 py-3">복잡한 경로, CAD 궤적</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">S-Curve</td>
                  <td className="px-4 py-3">저크 제한, 산업 표준</td>
                  <td className="px-4 py-3">산업용 로봇, CNC</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 font-semibold">Time-Optimal</td>
                  <td className="px-4 py-3">최단 시간, 복잡한 계산</td>
                  <td className="px-4 py-3">고속 픽앤플레이스</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                🔑 실무 선택 가이드
              </h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li><strong>협동 로봇</strong>: 5차 다항식 (부드러움)</li>
                <li><strong>CNC/3D 프린터</strong>: S-Curve (저크 제한)</li>
                <li><strong>Pick-and-Place</strong>: Time-Optimal</li>
                <li><strong>용접 로봇</strong>: Cubic Spline (복잡 경로)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                🛠️ 구현 라이브러리
              </h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li><strong>scipy.interpolate</strong>: Spline 함수</li>
                <li><strong>MoveIt</strong>: Time Parameterization</li>
                <li><strong>Reflexxes</strong>: 실시간 궤적 생성</li>
                <li><strong>TOPP-RA</strong>: Time-Optimal 알고리즘</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 bg-orange-100 dark:bg-orange-900/30 p-5 rounded-lg border border-orange-400 dark:border-orange-600">
            <p className="text-sm text-gray-800 dark:text-gray-200">
              <strong>다음 장 미리보기:</strong> Chapter 6에서는 그리핑과 조작(Grasping & Manipulation)을 학습합니다.
              로봇 그리퍼의 종류, Force Closure 이론, Visual Servoing 등을 다룹니다.
            </p>
          </div>
        </section>

        {/* Chapter Navigation */}
        <ChapterNavigation
          currentChapter={5}
          totalChapters={8}
          moduleSlug="robotics-manipulation"
        />
      </div>
    </div>
  )
}
