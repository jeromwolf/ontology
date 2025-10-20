'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, CheckCircle, AlertCircle } from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <div className="border-b border-purple-500/20 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <Link
            href="/modules/physics-fundamentals"
            className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors mb-3"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Physics Fundamentals 모듈로 돌아가기</span>
          </Link>
          <h1 className="text-3xl font-bold mb-2">Chapter 2: 운동학</h1>
          <p className="text-slate-300">직선 운동, 포물선 운동, 원운동</p>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
        {/* Section 1: 등가속도 직선 운동 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">1. 등가속도 직선 운동</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">운동학 기본 방정식</h3>
            <p className="text-slate-300 mb-4">
              가속도가 일정한 직선 운동에서 사용하는 세 가지 핵심 방정식
            </p>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm mb-1">1. v = v₀ + at</p>
                <p className="text-xs text-slate-400">최종 속도 = 초기 속도 + 가속도 × 시간</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm mb-1">2. x = x₀ + v₀t + ½at²</p>
                <p className="text-xs text-slate-400">위치 방정식 (시간의 함수)</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm mb-1">3. v² = v₀² + 2a(x - x₀)</p>
                <p className="text-xs text-slate-400">시간이 없는 방정식</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 자동차 가속</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 정지 상태에서 출발하여 10초 후 20 m/s가 된 자동차. 이동 거리는?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>주어진 값: v₀ = 0, v = 20 m/s, t = 10 s</p>
                <p>1. 가속도: a = (v - v₀)/t = 20/10 = 2 m/s²</p>
                <p>2. 거리: x = v₀t + ½at² = 0 + ½(2)(10²)</p>
                <p>3. x = 0 + ½(2)(100) = 100 m</p>
                <p className="text-purple-400 font-bold">답: 100 m</p>
              </div>
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-blue-400" />
              자유 낙하 운동
            </h3>
            <p className="text-slate-300 mb-3">
              공기 저항을 무시하면 모든 물체는 같은 가속도 g = 9.8 m/s²로 떨어진다
            </p>
            <ul className="space-y-2 text-slate-300">
              <li>• 위로 던진 물체: 최고점에서 v = 0</li>
              <li>• 대칭성: 올라가는 시간 = 내려오는 시간</li>
              <li>• 지면 도달 속력 = 던진 속력 (같은 높이)</li>
            </ul>
          </div>
        </section>

        {/* Section 2: 포물선 운동 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">2. 포물선 운동 (Projectile Motion)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">2차원 운동 분해</h3>
            <p className="text-slate-300 mb-4">
              수평 방향과 수직 방향을 독립적으로 분석
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h4 className="font-semibold mb-2">수평 방향 (x)</h4>
                <div className="font-mono text-sm space-y-1">
                  <p>vₓ = v₀ cos(θ) (일정)</p>
                  <p>x = v₀ cos(θ) · t</p>
                  <p>aₓ = 0</p>
                </div>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h4 className="font-semibold mb-2">수직 방향 (y)</h4>
                <div className="font-mono text-sm space-y-1">
                  <p>v_y = v₀ sin(θ) - gt</p>
                  <p>y = v₀ sin(θ) · t - ½gt²</p>
                  <p>a_y = -g</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">핵심 공식</h3>
            <div className="space-y-3">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">최고 높이 (H)</h4>
                <p className="font-mono text-sm">H = (v₀² sin²θ) / (2g)</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">비행 시간 (T)</h4>
                <p className="font-mono text-sm">T = (2v₀ sinθ) / g</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">수평 도달 거리 (R)</h4>
                <p className="font-mono text-sm">R = (v₀² sin(2θ)) / g</p>
                <p className="text-xs text-slate-400 mt-1">최대 거리: θ = 45°</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 대포알 발사</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 100 m/s로 30° 각도로 발사된 대포알의 최고 높이와 수평 거리는?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>주어진 값: v₀ = 100 m/s, θ = 30°, g = 9.8 m/s²</p>
                <p>1. 최고 높이: H = (100² × sin²30°) / (2 × 9.8)</p>
                <p>   H = (10000 × 0.25) / 19.6 = 127.6 m</p>
                <p>2. 수평 거리: R = (100² × sin(60°)) / 9.8</p>
                <p>   R = (10000 × 0.866) / 9.8 = 883.7 m</p>
                <p className="text-purple-400 font-bold">답: H = 127.6 m, R = 883.7 m</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: 원운동 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">3. 원운동 (Circular Motion)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">등속 원운동</h3>
            <p className="text-slate-300 mb-4">
              속력은 일정하지만 방향이 계속 변하므로 가속도가 존재 (구심 가속도)
            </p>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h4 className="font-semibold mb-2">구심 가속도</h4>
                <p className="font-mono text-sm">a_c = v² / r = ω²r</p>
                <p className="text-xs text-slate-400 mt-1">항상 원의 중심을 향함</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h4 className="font-semibold mb-2">구심력</h4>
                <p className="font-mono text-sm">F_c = ma_c = mv² / r</p>
                <p className="text-xs text-slate-400 mt-1">원운동을 유지하는 데 필요한 힘</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">각속도와 주기</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">각속도 (ω)</h4>
                <p className="font-mono text-sm">ω = v / r = 2π / T</p>
                <p className="text-xs text-slate-400 mt-1">단위: rad/s</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">주기 (T)</h4>
                <p className="font-mono text-sm">T = 2πr / v = 2π / ω</p>
                <p className="text-xs text-slate-400 mt-1">한 바퀴 도는 시간</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">진동수 (f)</h4>
                <p className="font-mono text-sm">f = 1 / T = ω / 2π</p>
                <p className="text-xs text-slate-400 mt-1">단위: Hz (헤르츠)</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">각도 (θ)</h4>
                <p className="font-mono text-sm">θ = ωt = 2πft</p>
                <p className="text-xs text-slate-400 mt-1">단위: rad (라디안)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-yellow-400" />
              실생활 예시
            </h3>
            <div className="space-y-4">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">1. 인공위성</h4>
                <p className="text-slate-300 text-sm">
                  중력이 구심력 역할: mg = mv²/r → v = √(gr)
                </p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">2. 자동차 커브</h4>
                <p className="text-slate-300 text-sm">
                  마찰력이 구심력: f = mv²/r (미끄러지지 않으려면 f ≤ μN)
                </p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">3. 놀이공원 회전 기구</h4>
                <p className="text-slate-300 text-sm">
                  줄의 장력이 구심력: T = mv²/r + mg cos(θ)
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: 상대 운동 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">4. 상대 운동</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">상대 속도</h3>
            <p className="text-slate-300 mb-4">
              관찰자에 따라 속도가 다르게 측정된다
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm mb-2">v<sub>AB</sub> = v<sub>A</sub> - v<sub>B</sub></p>
              <p className="text-xs text-slate-400">
                B에 대한 A의 상대 속도 = A의 속도 - B의 속도
              </p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 강을 건너는 배</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 배의 속력 5 m/s, 물의 흐름 속력 3 m/s (수직). 배의 실제 속도는?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>1. 벡터 합: v_total = √(v_배² + v_물²)</p>
                <p>2. v_total = √(5² + 3²) = √(25 + 9) = √34</p>
                <p>3. v_total ≈ 5.83 m/s</p>
                <p>4. 방향: tan(θ) = 3/5, θ = arctan(0.6) ≈ 31°</p>
                <p className="text-purple-400 font-bold">답: 5.83 m/s, 31° (물의 흐름 방향)</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: 연습 문제 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">5. 연습 문제</h2>
          </div>

          <div className="space-y-4">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">문제 1: 자유 낙하</h3>
              <p className="text-slate-300 mb-4">
                건물 꼭대기에서 공을 떨어뜨렸더니 3초 후 지면에 도달했다. 건물의 높이는?
              </p>
              <details className="bg-slate-900/50 p-4 rounded-lg">
                <summary className="cursor-pointer font-semibold text-purple-300">풀이 보기</summary>
                <div className="mt-3 space-y-2 text-sm text-slate-300 font-mono">
                  <p>주어진 값: v₀ = 0, t = 3 s, g = 9.8 m/s²</p>
                  <p>h = ½gt² = ½ × 9.8 × 3²</p>
                  <p>h = 4.9 × 9 = 44.1 m</p>
                  <p className="text-purple-400 font-bold">답: 44.1 m</p>
                </div>
              </details>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">문제 2: 원운동</h3>
              <p className="text-slate-300 mb-4">
                반지름 10 m인 원을 20 m/s로 도는 물체의 구심 가속도는?
              </p>
              <details className="bg-slate-900/50 p-4 rounded-lg">
                <summary className="cursor-pointer font-semibold text-purple-300">풀이 보기</summary>
                <div className="mt-3 space-y-2 text-sm text-slate-300 font-mono">
                  <p>주어진 값: r = 10 m, v = 20 m/s</p>
                  <p>a_c = v² / r = 20² / 10</p>
                  <p>a_c = 400 / 10 = 40 m/s²</p>
                  <p className="text-purple-400 font-bold">답: 40 m/s²</p>
                </div>
              </details>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">문제 3: 포물선 운동</h3>
              <p className="text-slate-300 mb-4">
                45° 각도로 50 m/s로 던진 공의 최대 수평 거리는?
              </p>
              <details className="bg-slate-900/50 p-4 rounded-lg">
                <summary className="cursor-pointer font-semibold text-purple-300">풀이 보기</summary>
                <div className="mt-3 space-y-2 text-sm text-slate-300 font-mono">
                  <p>주어진 값: v₀ = 50 m/s, θ = 45°, g = 9.8 m/s²</p>
                  <p>R = v₀² sin(2θ) / g</p>
                  <p>R = 50² × sin(90°) / 9.8</p>
                  <p>R = 2500 × 1 / 9.8 ≈ 255.1 m</p>
                  <p className="text-purple-400 font-bold">답: 255.1 m</p>
                </div>
              </details>
            </div>
          </div>
        </section>

        {/* Summary */}
        <section className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <CheckCircle className="w-8 h-8 text-purple-400" />
            핵심 요약
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">직선 운동</h3>
              <p className="text-sm text-slate-300">
                v = v₀ + at<br/>
                x = v₀t + ½at²
              </p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">포물선 운동</h3>
              <p className="text-sm text-slate-300">
                수평/수직 분리 분석<br/>
                R = v₀²sin(2θ)/g
              </p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">원운동</h3>
              <p className="text-sm text-slate-300">
                a_c = v²/r<br/>
                F_c = mv²/r
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-8 border-t border-purple-500/20">
          <Link
            href="/modules/physics-fundamentals?chapter=mechanics-basics"
            className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>이전: Chapter 1</span>
          </Link>
          <Link
            href="/modules/physics-fundamentals?chapter=energy-work"
            className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors"
          >
            <span>다음: Chapter 3 - 일과 에너지</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
