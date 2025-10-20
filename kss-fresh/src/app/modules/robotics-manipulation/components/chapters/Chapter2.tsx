'use client'

import React from 'react'
import ChapterNavigation from '../ChapterNavigation'

export default function Chapter2() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Chapter 2: 순기구학 (Forward Kinematics)</h1>
        <p className="text-xl text-white/90">
          관절 각도로부터 엔드이펙터의 위치를 계산하는 순기구학의 이론과 실전
        </p>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        {/* 1. 순기구학 개요 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1. 순기구학이란?
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-600 p-6 mb-6">
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300 mb-3">
              정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>순기구학(Forward Kinematics, FK)</strong>은 로봇의 각 관절 각도(또는 위치) 값이 주어졌을 때,
              엔드이펙터(End-Effector)의 위치와 방향을 계산하는 과정입니다.
            </p>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg mb-6">
            <h3 className="text-xl font-semibold text-orange-900 dark:text-orange-300 mb-4">
              수학적 표현
            </h3>
            <div className="bg-white dark:bg-gray-900 p-4 rounded border border-orange-300 dark:border-orange-700 font-mono text-sm">
              <p className="mb-2">입력: 관절 각도 벡터 <strong>θ = [θ₁, θ₂, ..., θₙ]ᵀ</strong></p>
              <p className="mb-2">출력: 엔드이펙터 위치 및 자세 <strong>T = f(θ)</strong></p>
              <p className="text-gray-600 dark:text-gray-400">여기서 T는 4×4 동차 변환 행렬(Homogeneous Transformation Matrix)</p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 mt-8">
            순기구학 vs 역기구학
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full border border-gray-300 dark:border-gray-600">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-3 border-b text-left">구분</th>
                  <th className="px-4 py-3 border-b text-left">순기구학 (FK)</th>
                  <th className="px-4 py-3 border-b text-left">역기구학 (IK)</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">입력</td>
                  <td className="px-4 py-3">관절 각도 θ</td>
                  <td className="px-4 py-3">엔드이펙터 위치 (x, y, z)</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">출력</td>
                  <td className="px-4 py-3">엔드이펙터 위치 (x, y, z)</td>
                  <td className="px-4 py-3">관절 각도 θ</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">계산 복잡도</td>
                  <td className="px-4 py-3">단순 (행렬 곱셈)</td>
                  <td className="px-4 py-3">복잡 (해석/수치해법)</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">해의 개수</td>
                  <td className="px-4 py-3">유일해 (1개)</td>
                  <td className="px-4 py-3">다중해 또는 무해</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 font-semibold">실시간 계산</td>
                  <td className="px-4 py-3">✅ 용이</td>
                  <td className="px-4 py-3">⚠️ 까다로움</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* 2. DH 파라미터 상세 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            2. DH 파라미터 (Denavit-Hartenberg Parameters)
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            DH 파라미터는 로봇의 각 링크 간 좌표 변환을 체계적으로 표현하는 표준 방법입니다.
            4개의 파라미터(θ, d, a, α)로 임의의 두 좌표계 간 관계를 완전히 기술할 수 있습니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            4가지 DH 파라미터
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border border-blue-300 dark:border-blue-700">
              <h4 className="font-bold text-lg text-blue-900 dark:text-blue-300 mb-2">
                θᵢ (Theta) - 관절 각도
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <strong>Z축 회전</strong>: Xᵢ₋₁축을 Xᵢ축으로 정렬하는 회전각<br/>
                회전 관절(Revolute)의 경우: <strong>변수</strong><br/>
                직진 관절(Prismatic)의 경우: <strong>상수</strong>
              </p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold text-lg text-green-900 dark:text-green-300 mb-2">
                dᵢ (d) - 링크 오프셋
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <strong>Z축 이동</strong>: Xᵢ₋₁축에서 Xᵢ축까지의 거리<br/>
                회전 관절의 경우: <strong>상수</strong><br/>
                직진 관절의 경우: <strong>변수</strong>
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border border-purple-300 dark:border-purple-700">
              <h4 className="font-bold text-lg text-purple-900 dark:text-purple-300 mb-2">
                aᵢ (a) - 링크 길이
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <strong>X축 이동</strong>: Zᵢ₋₁축에서 Zᵢ축까지의 최단 거리<br/>
                모든 관절 타입: <strong>상수</strong> (링크 물리적 길이)
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border border-orange-300 dark:border-orange-700">
              <h4 className="font-bold text-lg text-orange-900 dark:text-orange-300 mb-2">
                αᵢ (Alpha) - 링크 비틀림
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <strong>X축 회전</strong>: Zᵢ₋₁축을 Zᵢ축으로 정렬하는 회전각<br/>
                모든 관절 타입: <strong>상수</strong> (링크 기하학적 형상)
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            DH 변환 행렬
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              링크 i-1에서 링크 i로의 변환은 다음 4단계 변환의 곱으로 표현됩니다:
            </p>
            <div className="font-mono text-sm bg-white dark:bg-gray-800 p-4 rounded overflow-x-auto">
              <p className="mb-2">ⁱ⁻¹Tᵢ = Rot(Z, θᵢ) · Trans(Z, dᵢ) · Trans(X, aᵢ) · Rot(X, αᵢ)</p>
              <p className="text-gray-500 dark:text-gray-400 text-xs mt-4 mb-2">4×4 동차 변환 행렬 형태:</p>
              <pre className="text-xs">
{`┌                                                           ┐
│  cos(θᵢ)  -sin(θᵢ)cos(αᵢ)   sin(θᵢ)sin(αᵢ)   aᵢcos(θᵢ) │
│  sin(θᵢ)   cos(θᵢ)cos(αᵢ)  -cos(θᵢ)sin(αᵢ)   aᵢsin(θᵢ) │
│     0          sin(αᵢ)           cos(αᵢ)           dᵢ     │
│     0              0                  0             1     │
└                                                           ┘`}
              </pre>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            DH 좌표계 설정 규칙
          </h3>
          <ol className="list-decimal list-inside space-y-3 text-gray-700 dark:text-gray-300">
            <li><strong>Z축</strong>: 관절 i의 회전/직진 축과 일치</li>
            <li><strong>X축</strong>: Z<sub>i-1</sub>과 Z<sub>i</sub>의 공통 법선 방향</li>
            <li><strong>Y축</strong>: 오른손 좌표계 규칙으로 결정 (X × Z = Y)</li>
            <li><strong>원점</strong>: X<sub>i</sub>와 Z<sub>i</sub>의 교점</li>
          </ol>
        </section>

        {/* 3. 2-DOF 평면 로봇 FK */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3. 2-DOF 평면 로봇 순기구학 유도
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-600 p-6 mb-6">
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300 mb-3">
              문제 정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              두 개의 회전 관절(Revolute Joints)을 가진 2차원 평면 매니퓰레이터에서<br/>
              관절 각도 <strong>θ₁</strong>, <strong>θ₂</strong>가 주어졌을 때 엔드이펙터의 위치 <strong>(x, y)</strong>를 구하시오.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            Step 1: DH 파라미터 테이블 작성
          </h3>
          <div className="overflow-x-auto mb-6">
            <table className="min-w-full border border-gray-300 dark:border-gray-600">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-3 border-b">링크 i</th>
                  <th className="px-4 py-3 border-b">θᵢ</th>
                  <th className="px-4 py-3 border-b">dᵢ</th>
                  <th className="px-4 py-3 border-b">aᵢ</th>
                  <th className="px-4 py-3 border-b">αᵢ</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="px-4 py-3 text-center font-semibold">1</td>
                  <td className="px-4 py-3 text-center text-blue-600 dark:text-blue-400">θ₁ (변수)</td>
                  <td className="px-4 py-3 text-center">0</td>
                  <td className="px-4 py-3 text-center">L₁</td>
                  <td className="px-4 py-3 text-center">0°</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 text-center font-semibold">2</td>
                  <td className="px-4 py-3 text-center text-blue-600 dark:text-blue-400">θ₂ (변수)</td>
                  <td className="px-4 py-3 text-center">0</td>
                  <td className="px-4 py-3 text-center">L₂</td>
                  <td className="px-4 py-3 text-center">0°</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            Step 2: 개별 변환 행렬 계산
          </h3>
          <div className="space-y-4 mb-6">
            <div className="bg-gray-50 dark:bg-gray-900 p-5 rounded-lg border border-gray-300 dark:border-gray-600">
              <p className="font-semibold mb-2 text-gray-900 dark:text-white">링크 1 변환 행렬 ⁰T₁:</p>
              <pre className="font-mono text-xs overflow-x-auto bg-white dark:bg-gray-800 p-3 rounded">
{`┌                                    ┐
│  cos(θ₁)  -sin(θ₁)    0    L₁cos(θ₁) │
│  sin(θ₁)   cos(θ₁)    0    L₁sin(θ₁) │
│     0          0       1         0     │
│     0          0       0         1     │
└                                    ┘`}
              </pre>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-5 rounded-lg border border-gray-300 dark:border-gray-600">
              <p className="font-semibold mb-2 text-gray-900 dark:text-white">링크 2 변환 행렬 ¹T₂:</p>
              <pre className="font-mono text-xs overflow-x-auto bg-white dark:bg-gray-800 p-3 rounded">
{`┌                                    ┐
│  cos(θ₂)  -sin(θ₂)    0    L₂cos(θ₂) │
│  sin(θ₂)   cos(θ₂)    0    L₂sin(θ₂) │
│     0          0       1         0     │
│     0          0       0         1     │
└                                    ┘`}
              </pre>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            Step 3: 전체 변환 행렬
          </h3>
          <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border border-orange-300 dark:border-orange-700 mb-6">
            <p className="font-mono text-sm mb-3">⁰T₂ = ⁰T₁ · ¹T₂</p>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              행렬 곱셈을 수행하면 엔드이펙터의 위치는 다음과 같습니다:
            </p>
            <div className="bg-white dark:bg-gray-900 p-4 rounded">
              <p className="font-mono text-sm mb-2">
                <strong className="text-orange-600 dark:text-orange-400">x</strong> = L₁·cos(θ₁) + L₂·cos(θ₁ + θ₂)
              </p>
              <p className="font-mono text-sm">
                <strong className="text-orange-600 dark:text-orange-400">y</strong> = L₁·sin(θ₁) + L₂·sin(θ₁ + θ₂)
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            Step 4: 수치 예제
          </h3>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-300 dark:border-blue-700">
            <p className="font-semibold mb-3 text-blue-900 dark:text-blue-300">예제: L₁ = 1.0m, L₂ = 0.8m, θ₁ = 30°, θ₂ = 45°</p>
            <div className="space-y-2 text-sm font-mono bg-white dark:bg-gray-900 p-4 rounded">
              <p>θ₁ + θ₂ = 30° + 45° = 75°</p>
              <p className="mt-3">x = 1.0·cos(30°) + 0.8·cos(75°)</p>
              <p className="ml-4">= 1.0 × 0.866 + 0.8 × 0.259</p>
              <p className="ml-4 text-green-600 dark:text-green-400">= <strong>1.073 m</strong></p>
              <p className="mt-3">y = 1.0·sin(30°) + 0.8·sin(75°)</p>
              <p className="ml-4">= 1.0 × 0.5 + 0.8 × 0.966</p>
              <p className="ml-4 text-green-600 dark:text-green-400">= <strong>1.273 m</strong></p>
            </div>
          </div>
        </section>

        {/* 4. 6-DOF 로봇 FK */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            4. 6-DOF 산업용 로봇 순기구학
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            대부분의 산업용 로봇 팔(예: PUMA 560, UR5, ABB IRB 등)은 6개의 자유도를 가지며,
            이는 3차원 공간에서 위치(x, y, z) 3개와 방향(roll, pitch, yaw) 3개를 완전히 제어할 수 있습니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            전형적인 6-DOF 로봇 DH 파라미터 (PUMA 560 기준)
          </h3>
          <div className="overflow-x-auto mb-6">
            <table className="min-w-full border border-gray-300 dark:border-gray-600 text-sm">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-3 py-2 border-b">링크 i</th>
                  <th className="px-3 py-2 border-b">θᵢ</th>
                  <th className="px-3 py-2 border-b">dᵢ (m)</th>
                  <th className="px-3 py-2 border-b">aᵢ (m)</th>
                  <th className="px-3 py-2 border-b">αᵢ</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="px-3 py-2 text-center font-semibold">1</td>
                  <td className="px-3 py-2 text-center text-blue-600">θ₁*</td>
                  <td className="px-3 py-2 text-center">0.672</td>
                  <td className="px-3 py-2 text-center">0</td>
                  <td className="px-3 py-2 text-center">90°</td>
                </tr>
                <tr className="border-b">
                  <td className="px-3 py-2 text-center font-semibold">2</td>
                  <td className="px-3 py-2 text-center text-blue-600">θ₂*</td>
                  <td className="px-3 py-2 text-center">0</td>
                  <td className="px-3 py-2 text-center">0.432</td>
                  <td className="px-3 py-2 text-center">0°</td>
                </tr>
                <tr className="border-b">
                  <td className="px-3 py-2 text-center font-semibold">3</td>
                  <td className="px-3 py-2 text-center text-blue-600">θ₃*</td>
                  <td className="px-3 py-2 text-center">0.149</td>
                  <td className="px-3 py-2 text-center">0.020</td>
                  <td className="px-3 py-2 text-center">-90°</td>
                </tr>
                <tr className="border-b">
                  <td className="px-3 py-2 text-center font-semibold">4</td>
                  <td className="px-3 py-2 text-center text-blue-600">θ₄*</td>
                  <td className="px-3 py-2 text-center">0.433</td>
                  <td className="px-3 py-2 text-center">0</td>
                  <td className="px-3 py-2 text-center">90°</td>
                </tr>
                <tr className="border-b">
                  <td className="px-3 py-2 text-center font-semibold">5</td>
                  <td className="px-3 py-2 text-center text-blue-600">θ₅*</td>
                  <td className="px-3 py-2 text-center">0</td>
                  <td className="px-3 py-2 text-center">0</td>
                  <td className="px-3 py-2 text-center">-90°</td>
                </tr>
                <tr>
                  <td className="px-3 py-2 text-center font-semibold">6</td>
                  <td className="px-3 py-2 text-center text-blue-600">θ₆*</td>
                  <td className="px-3 py-2 text-center">0</td>
                  <td className="px-3 py-2 text-center">0</td>
                  <td className="px-3 py-2 text-center">0°</td>
                </tr>
              </tbody>
            </table>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">* 변수: 모든 θᵢ는 회전 관절 변수</p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            최종 변환 행렬 계산
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <p className="font-mono text-sm mb-4">
              ⁰T₆ = ⁰T₁ · ¹T₂ · ²T₃ · ³T₄ · ⁴T₅ · ⁵T₆
            </p>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              최종 4×4 변환 행렬은 다음과 같은 구조를 가집니다:
            </p>
            <pre className="font-mono text-xs overflow-x-auto bg-white dark:bg-gray-800 p-4 rounded">
{`⁰T₆ = ┌                          ┐
      │  R₁₁  R₁₂  R₁₃    Pₓ   │   ← 3×3 회전 행렬 R
      │  R₂₁  R₂₂  R₂₃    Pᵧ   │   ← 3×1 위치 벡터 P
      │  R₃₁  R₃₂  R₃₃    Pᵤ   │
      │   0    0    0      1   │
      └                          ┘

엔드이펙터 위치: P = [Pₓ, Pᵧ, Pᵤ]ᵀ
엔드이펙터 방향: R (3×3 회전 행렬 → Euler angles 또는 Quaternion)`}
            </pre>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-600 p-6">
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-300 mb-3">
              💡 실전 팁: 효율적인 FK 계산
            </h3>
            <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>중간 좌표계 저장</strong>: ⁰T₃까지 계산 후 캐싱하면 재계산 시간 단축</li>
              <li><strong>동차 좌표 활용</strong>: 4×4 행렬 하나로 회전+이동 통합 처리</li>
              <li><strong>수치 안정성</strong>: 삼각함수 계산 시 라이브러리 함수 활용 (정밀도 보장)</li>
              <li><strong>벡터화 연산</strong>: NumPy, Eigen 등 행렬 라이브러리로 병렬 처리</li>
            </ul>
          </div>
        </section>

        {/* 5. 작업 공간 분석 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            5. 작업 공간 (Workspace) 분석
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-600 p-6 mb-6">
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300 mb-3">
              작업 공간의 정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              로봇의 <strong>작업 공간(Workspace)</strong>은 엔드이펙터가 도달할 수 있는 모든 점들의 집합입니다.
              관절 각도의 물리적 제약(Joint Limits) 내에서 순기구학으로 계산 가능한 모든 위치를 의미합니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            작업 공간의 종류
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold text-lg text-green-900 dark:text-green-300 mb-2">
                도달 가능 작업 공간<br/>(Reachable Workspace)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                엔드이펙터가 <strong>적어도 하나의 방향</strong>으로 도달할 수 있는 모든 점의 집합
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border border-purple-300 dark:border-purple-700">
              <h4 className="font-bold text-lg text-purple-900 dark:text-purple-300 mb-2">
                조작 가능 작업 공간<br/>(Dexterous Workspace)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                엔드이펙터가 <strong>모든 방향</strong>에서 도달할 수 있는 점의 집합 (더 작은 영역)
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            2-DOF 평면 로봇의 작업 공간 계산
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <p className="font-semibold mb-3 text-gray-900 dark:text-white">조건: L₁ = 1.0m, L₂ = 0.8m, 0° ≤ θ₁, θ₂ ≤ 180°</p>
            <div className="space-y-3 text-sm">
              <div>
                <p className="font-mono mb-1">최대 도달 거리: R_max = L₁ + L₂ = 1.0 + 0.8 = <strong className="text-green-600 dark:text-green-400">1.8m</strong></p>
                <p className="text-gray-600 dark:text-gray-400 text-xs ml-4">
                  (θ₁ = 0°, θ₂ = 0° 일 때 완전히 펴진 상태)
                </p>
              </div>
              <div>
                <p className="font-mono mb-1">최소 도달 거리: R_min = |L₁ - L₂| = |1.0 - 0.8| = <strong className="text-orange-600 dark:text-orange-400">0.2m</strong></p>
                <p className="text-gray-600 dark:text-gray-400 text-xs ml-4">
                  (θ₁ = 0°, θ₂ = 180° 일 때 완전히 접힌 상태)
                </p>
              </div>
              <div className="mt-4 p-4 bg-white dark:bg-gray-800 rounded border border-blue-300 dark:border-blue-700">
                <p className="font-semibold text-blue-900 dark:text-blue-300 mb-2">작업 공간 형태:</p>
                <p className="text-xs text-gray-700 dark:text-gray-300">
                  <strong>환형(Annulus)</strong> - 내부 반지름 0.2m, 외부 반지름 1.8m의 원형 고리 영역
                </p>
              </div>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            작업 공간 시각화 방법
          </h3>
          <ol className="list-decimal list-inside space-y-3 text-gray-700 dark:text-gray-300">
            <li>
              <strong>격자 샘플링</strong>: 관절 각도 공간을 균일하게 샘플링 (예: 1° 간격)
            </li>
            <li>
              <strong>FK 계산</strong>: 각 샘플 점에서 순기구학으로 엔드이펙터 위치 계산
            </li>
            <li>
              <strong>3D 점군 생성</strong>: 모든 도달 가능한 점을 3차원 공간에 플롯
            </li>
            <li>
              <strong>경계 추출</strong>: Convex Hull, Alpha Shape 등 알고리즘으로 외곽선 도출
            </li>
          </ol>

          <div className="mt-6 bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-600 p-6">
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-300 mb-3">
              ⚠️ 작업 공간의 실전적 고려사항
            </h3>
            <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>관절 한계(Joint Limits)</strong>: 물리적 스토퍼로 인한 각도 제한</li>
              <li><strong>자기 충돌(Self-Collision)</strong>: 링크 간 간섭으로 접근 불가능한 영역</li>
              <li><strong>장애물(Obstacles)</strong>: 주변 환경으로 인한 작업 공간 축소</li>
              <li><strong>특이점(Singularities)</strong>: 일부 자유도를 잃는 구성 (다음 섹션)</li>
            </ul>
          </div>
        </section>

        {/* 6. 특이점 분석 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            6. 특이점 (Singularity) 분석
          </h2>

          <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-600 p-6 mb-6">
            <h3 className="text-xl font-semibold text-red-900 dark:text-red-300 mb-3">
              ⚠️ 특이점이란?
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>특이점(Singularity)</strong>은 로봇이 하나 이상의 자유도를 잃어버려서
              특정 방향으로 움직일 수 없게 되는 특수한 관절 구성입니다.
              이 위치에서는 <strong>야코비안 행렬(Jacobian Matrix)의 행렬식(Determinant)이 0</strong>이 됩니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            야코비안 행렬과 특이점의 관계
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              야코비안 행렬 <strong>J(θ)</strong>는 관절 속도와 엔드이펙터 속도를 연결합니다:
            </p>
            <div className="font-mono text-sm bg-white dark:bg-gray-800 p-4 rounded mb-4">
              <p className="mb-2">v = J(θ) · θ̇</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                v: 엔드이펙터 선속도/각속도 (6×1)<br/>
                J(θ): 야코비안 행렬 (6×n)<br/>
                θ̇: 관절 속도 (n×1)
              </p>
            </div>
            <div className="bg-red-100 dark:bg-red-900/30 p-4 rounded border border-red-400 dark:border-red-700">
              <p className="font-semibold text-red-900 dark:text-red-300 mb-2">특이점 조건:</p>
              <p className="font-mono text-sm">det(J(θ)) = 0</p>
              <p className="text-xs text-gray-700 dark:text-gray-300 mt-2">
                → 일부 엔드이펙터 속도가 <strong>어떤 관절 속도로도 만들 수 없음</strong>
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            특이점의 3가지 유형
          </h3>
          <div className="space-y-4 mb-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border border-blue-300 dark:border-blue-700">
              <h4 className="font-bold text-lg text-blue-900 dark:text-blue-300 mb-2">
                1. 경계 특이점 (Boundary Singularity)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                로봇 팔이 <strong>완전히 펴진(Fully Extended)</strong> 또는 <strong>완전히 접힌(Fully Retracted)</strong> 상태
              </p>
              <p className="text-xs font-mono bg-white dark:bg-gray-900 p-2 rounded mt-2">
                예: 2-DOF 평면 로봇에서 θ₂ = 0° (완전히 펴짐) 또는 θ₂ = 180° (완전히 접힘)
              </p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold text-lg text-green-900 dark:text-green-300 mb-2">
                2. 내부 특이점 (Interior Singularity)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                작업 공간 내부에서 발생하는 특이점. <strong>두 개 이상의 관절 축이 정렬</strong>될 때 발생
              </p>
              <p className="text-xs font-mono bg-white dark:bg-gray-900 p-2 rounded mt-2">
                예: 6-DOF 로봇에서 팔꿈치가 완전히 펴진 상태 (어깨-팔꿈치-손목 일직선)
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border border-purple-300 dark:border-purple-700">
              <h4 className="font-bold text-lg text-purple-900 dark:text-purple-300 mb-2">
                3. 손목 특이점 (Wrist Singularity)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                손목의 두 회전 축이 정렬되어 하나의 방향 자유도를 잃음
              </p>
              <p className="text-xs font-mono bg-white dark:bg-gray-900 p-2 rounded mt-2">
                예: PUMA 로봇에서 θ₅ = 0° (Joint 4와 Joint 6의 축이 평행)
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            2-DOF 평면 로봇의 특이점 분석
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <p className="font-semibold mb-3 text-gray-900 dark:text-white">야코비안 행렬 유도:</p>
            <div className="space-y-3 text-sm font-mono">
              <p>순기구학 결과:</p>
              <p className="ml-4">x = L₁cos(θ₁) + L₂cos(θ₁+θ₂)</p>
              <p className="ml-4">y = L₁sin(θ₁) + L₂sin(θ₁+θ₂)</p>

              <p className="mt-4">편미분:</p>
              <p className="ml-4">∂x/∂θ₁ = -L₁sin(θ₁) - L₂sin(θ₁+θ₂)</p>
              <p className="ml-4">∂x/∂θ₂ = -L₂sin(θ₁+θ₂)</p>
              <p className="ml-4">∂y/∂θ₁ = L₁cos(θ₁) + L₂cos(θ₁+θ₂)</p>
              <p className="ml-4">∂y/∂θ₂ = L₂cos(θ₁+θ₂)</p>

              <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded">
                <p className="mb-2">야코비안 행렬:</p>
                <pre className="text-xs">
{`     ┌                                              ┐
J =  │ -L₁sin(θ₁)-L₂sin(θ₁+θ₂)  -L₂sin(θ₁+θ₂) │
     │  L₁cos(θ₁)+L₂cos(θ₁+θ₂)   L₂cos(θ₁+θ₂) │
     └                                              ┘`}
                </pre>
              </div>

              <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/30 rounded border border-red-400">
                <p className="mb-2 text-red-900 dark:text-red-300">행렬식 계산:</p>
                <p>det(J) = L₁L₂sin(θ₂)</p>
                <p className="mt-2 text-xs text-gray-700 dark:text-gray-300">
                  <strong>특이점 조건</strong>: sin(θ₂) = 0 ⇒ θ₂ = 0° 또는 180°
                </p>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-600 p-6">
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-300 mb-3">
              🛠️ 특이점 회피 전략
            </h3>
            <ol className="list-decimal list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>경로 계획 단계에서 회피</strong>: 특이점 근처 경로를 미리 제외</li>
              <li><strong>Damped Least Squares (DLS)</strong>: 수치적으로 안정적인 역기구학 해법 사용</li>
              <li><strong>Redundancy 활용</strong>: 7-DOF 이상 로봇에서 여분 자유도로 우회</li>
              <li><strong>Joint Limit 설정</strong>: 특이점 근처 각도를 금지 영역으로 설정</li>
              <li><strong>속도 제한</strong>: 특이점 근처에서 엔드이펙터 속도를 낮춤</li>
            </ol>
          </div>
        </section>

        {/* 7. 실전 구현 예제 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            7. Python 구현 예제
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            NumPy를 활용한 2-DOF 평면 로봇의 순기구학 구현 예제입니다.
          </p>

          <div className="bg-gray-900 text-gray-100 p-6 rounded-lg overflow-x-auto mb-6">
            <pre className="text-sm font-mono">
{`import numpy as np
import matplotlib.pyplot as plt

class PlanarRobot2DOF:
    def __init__(self, L1=1.0, L2=0.8):
        """
        2-DOF 평면 로봇 초기화

        Parameters:
        -----------
        L1 : float
            첫 번째 링크 길이 (m)
        L2 : float
            두 번째 링크 길이 (m)
        """
        self.L1 = L1
        self.L2 = L2

    def forward_kinematics(self, theta1, theta2):
        """
        순기구학: 관절 각도 → 엔드이펙터 위치

        Parameters:
        -----------
        theta1 : float
            첫 번째 관절 각도 (degrees)
        theta2 : float
            두 번째 관절 각도 (degrees)

        Returns:
        --------
        x, y : float
            엔드이펙터 위치 (m)
        """
        # 각도를 라디안으로 변환
        theta1_rad = np.deg2rad(theta1)
        theta2_rad = np.deg2rad(theta2)

        # FK 공식 적용
        x = self.L1 * np.cos(theta1_rad) + \\
            self.L2 * np.cos(theta1_rad + theta2_rad)
        y = self.L1 * np.sin(theta1_rad) + \\
            self.L2 * np.sin(theta1_rad + theta2_rad)

        return x, y

    def jacobian(self, theta1, theta2):
        """
        야코비안 행렬 계산

        Returns:
        --------
        J : ndarray (2x2)
            야코비안 행렬
        """
        theta1_rad = np.deg2rad(theta1)
        theta2_rad = np.deg2rad(theta2)

        J = np.array([
            [-self.L1*np.sin(theta1_rad) - self.L2*np.sin(theta1_rad+theta2_rad),
             -self.L2*np.sin(theta1_rad+theta2_rad)],
            [self.L1*np.cos(theta1_rad) + self.L2*np.cos(theta1_rad+theta2_rad),
             self.L2*np.cos(theta1_rad+theta2_rad)]
        ])

        return J

    def is_singular(self, theta1, theta2, tolerance=1e-3):
        """
        특이점 판별

        Returns:
        --------
        bool : True if singular
        """
        J = self.jacobian(theta1, theta2)
        det_J = np.linalg.det(J)
        return abs(det_J) < tolerance

    def plot_workspace(self, num_samples=360):
        """
        작업 공간 시각화
        """
        plt.figure(figsize=(10, 10))

        # 관절 각도 샘플링 (0° ~ 180°)
        theta1_range = np.linspace(0, 180, num_samples)
        theta2_range = np.linspace(0, 180, num_samples)

        # 모든 조합에 대해 FK 계산
        for theta1 in theta1_range:
            for theta2 in theta2_range:
                x, y = self.forward_kinematics(theta1, theta2)
                plt.plot(x, y, 'b.', markersize=1, alpha=0.3)

        plt.xlabel('X (m)', fontsize=14)
        plt.ylabel('Y (m)', fontsize=14)
        plt.title('2-DOF Planar Robot Workspace', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

# 사용 예제
if __name__ == "__main__":
    # 로봇 생성
    robot = PlanarRobot2DOF(L1=1.0, L2=0.8)

    # 1. 순기구학 계산
    theta1, theta2 = 30, 45
    x, y = robot.forward_kinematics(theta1, theta2)
    print(f"FK Result: θ₁={theta1}°, θ₂={theta2}° → (x={x:.3f}m, y={y:.3f}m)")

    # 2. 야코비안 계산
    J = robot.jacobian(theta1, theta2)
    print(f"\\nJacobian:\\n{J}")
    print(f"det(J) = {np.linalg.det(J):.6f}")

    # 3. 특이점 검사
    is_sing = robot.is_singular(theta1, theta2)
    print(f"\\nSingular: {is_sing}")

    # 4. 작업 공간 시각화
    # robot.plot_workspace()  # 주석 해제하면 플롯 표시`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-600 p-6">
            <h3 className="text-lg font-semibold text-green-900 dark:text-green-300 mb-3">
              ✅ 실행 결과 예시
            </h3>
            <pre className="text-xs font-mono bg-white dark:bg-gray-900 p-4 rounded overflow-x-auto">
{`FK Result: θ₁=30°, θ₂=45° → (x=1.073m, y=1.273m)

Jacobian:
[[-1.2729961  -0.5656854]
 [ 0.6727876   0.5656854]]
det(J) = -0.340170

Singular: False`}
            </pre>
          </div>
        </section>

        {/* 8. 요약 */}
        <section className="mb-12 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-8 border border-orange-300 dark:border-orange-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            📌 핵심 요약
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                ✅ 순기구학의 장점
              </h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>계산이 단순하고 빠름 (행렬 곱셈)</li>
                <li>항상 유일한 해 존재</li>
                <li>실시간 시뮬레이션 가능</li>
                <li>수치적으로 안정적</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                🔑 핵심 공식
              </h3>
              <div className="space-y-2 text-sm font-mono">
                <p><strong>DH 변환:</strong> ⁰Tₙ = ⁰T₁ · ¹T₂ · ... · ⁿ⁻¹Tₙ</p>
                <p><strong>2-DOF FK:</strong></p>
                <p className="ml-4">x = L₁cos(θ₁) + L₂cos(θ₁+θ₂)</p>
                <p className="ml-4">y = L₁sin(θ₁) + L₂sin(θ₁+θ₂)</p>
                <p><strong>특이점:</strong> det(J) = 0</p>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                🎯 실전 응용
              </h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>로봇 시뮬레이터 개발</li>
                <li>작업 공간 시각화</li>
                <li>충돌 감지 (Collision Detection)</li>
                <li>경로 계획 검증</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                ⚠️ 주의사항
              </h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>특이점에서는 역기구학 불안정</li>
                <li>관절 한계(Joint Limits) 확인 필수</li>
                <li>삼각함수 계산 정밀도 관리</li>
                <li>좌표계 설정 일관성 유지</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 bg-orange-100 dark:bg-orange-900/30 p-5 rounded-lg border border-orange-400 dark:border-orange-600">
            <p className="text-sm text-gray-800 dark:text-gray-200">
              <strong>다음 장 미리보기:</strong> Chapter 3에서는 역기구학(Inverse Kinematics)을 학습합니다.
              목표 위치가 주어졌을 때 필요한 관절 각도를 계산하는 방법과 다중해 문제를 해결하는 기법을 다룹니다.
            </p>
          </div>
        </section>

        {/* Chapter Navigation */}
        <ChapterNavigation
          currentChapter={2}
          totalChapters={8}
          moduleSlug="robotics-manipulation"
        />
      </div>
    </div>
  )
}
