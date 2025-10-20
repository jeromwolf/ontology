'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, CheckCircle, AlertCircle } from 'lucide-react'

export default function Chapter3() {
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
          <h1 className="text-3xl font-bold mb-2">Chapter 3: 일과 에너지</h1>
          <p className="text-slate-300">일-에너지 정리, 위치에너지, 운동에너지</p>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
        {/* Section 1: 일 (Work) */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">1. 일 (Work)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">일의 정의</h3>
            <p className="text-slate-300 mb-4">
              힘이 물체를 이동시킬 때 한 일의 양
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-lg font-bold">W = F · d · cos(θ)</p>
              <div className="text-sm text-slate-300 space-y-1">
                <p>• W: 일 (Work, Joule = N·m)</p>
                <p>• F: 힘 (Force, N)</p>
                <p>• d: 변위 (Displacement, m)</p>
                <p>• θ: 힘과 변위 사이의 각도</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">일의 부호</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h4 className="font-semibold text-green-400 mb-2">양의 일 (W &gt; 0)</h4>
                <p className="text-sm text-slate-300">
                  힘과 변위가 같은 방향<br/>
                  θ = 0°, cos(0°) = 1<br/>
                  에너지 증가
                </p>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                <h4 className="font-semibold text-red-400 mb-2">음의 일 (W &lt; 0)</h4>
                <p className="text-sm text-slate-300">
                  힘과 변위가 반대 방향<br/>
                  θ = 180°, cos(180°) = -1<br/>
                  에너지 감소
                </p>
              </div>
              <div className="bg-slate-500/10 border border-slate-500/30 rounded-lg p-4">
                <h4 className="font-semibold text-slate-400 mb-2">일이 0 (W = 0)</h4>
                <p className="text-sm text-slate-300">
                  힘과 변위가 수직<br/>
                  θ = 90°, cos(90°) = 0<br/>
                  에너지 변화 없음
                </p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 수평면에서 상자 밀기</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 20 N의 힘으로 상자를 5 m 밀었다. 한 일은?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>주어진 값: F = 20 N, d = 5 m, θ = 0° (같은 방향)</p>
                <p>W = F · d · cos(θ) = 20 × 5 × cos(0°)</p>
                <p>W = 20 × 5 × 1 = 100 J</p>
                <p className="text-purple-400 font-bold">답: 100 J</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: 운동 에너지 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">2. 운동 에너지 (Kinetic Energy)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">운동 에너지</h3>
            <p className="text-slate-300 mb-4">
              운동하는 물체가 가진 에너지
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-lg font-bold">KE = ½mv²</p>
              <div className="text-sm text-slate-300 space-y-1">
                <p>• KE: 운동 에너지 (Joule)</p>
                <p>• m: 질량 (kg)</p>
                <p>• v: 속력 (m/s)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">일-에너지 정리</h3>
            <p className="text-slate-300 mb-4">
              알짜힘이 한 일은 운동 에너지의 변화량과 같다
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-lg font-bold mb-2">W_net = ΔKE = KE_f - KE_i</p>
              <p className="font-mono text-sm">W_net = ½mv_f² - ½mv_i²</p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 자동차 가속</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 1000 kg 자동차가 정지 상태에서 20 m/s가 되었다. 한 일은?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>주어진 값: m = 1000 kg, v_i = 0, v_f = 20 m/s</p>
                <p>W = ΔKE = ½m(v_f² - v_i²)</p>
                <p>W = ½ × 1000 × (20² - 0²)</p>
                <p>W = 500 × 400 = 200,000 J = 200 kJ</p>
                <p className="text-purple-400 font-bold">답: 200 kJ</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: 위치 에너지 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">3. 위치 에너지 (Potential Energy)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">중력 위치 에너지</h3>
            <p className="text-slate-300 mb-4">
              높이에 따라 저장된 에너지
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-lg font-bold">PE = mgh</p>
              <div className="text-sm text-slate-300 space-y-1">
                <p>• PE: 위치 에너지 (Joule)</p>
                <p>• m: 질량 (kg)</p>
                <p>• g: 중력 가속도 (9.8 m/s²)</p>
                <p>• h: 기준점으로부터 높이 (m)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">탄성 위치 에너지</h3>
            <p className="text-slate-300 mb-4">
              용수철이나 탄성체에 저장된 에너지
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-lg font-bold">PE_spring = ½kx²</p>
              <div className="text-sm text-slate-300 space-y-1">
                <p>• k: 용수철 상수 (N/m)</p>
                <p>• x: 평형 위치로부터의 변위 (m)</p>
              </div>
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-blue-400" />
              기준점의 선택
            </h3>
            <p className="text-slate-300 mb-3">
              위치 에너지는 기준점에 따라 달라진다 (상대적)
            </p>
            <ul className="space-y-2 text-slate-300">
              <li>• 보통 지면을 h = 0으로 설정</li>
              <li>• 기준점이 달라져도 에너지 차이는 같다</li>
              <li>• 물리적으로 의미 있는 것은 에너지 변화량</li>
            </ul>
          </div>
        </section>

        {/* Section 4: 역학적 에너지 보존 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">4. 역학적 에너지 보존</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">에너지 보존 법칙</h3>
            <p className="text-slate-300 mb-4">
              마찰이 없고 보존력만 작용하면 역학적 에너지는 일정하다
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-lg font-bold">E = KE + PE = constant</p>
              <p className="font-mono text-sm">½mv₁² + mgh₁ = ½mv₂² + mgh₂</p>
              <p className="text-xs text-slate-400 mt-2">
                (운동에너지 + 위치에너지 = 일정)
              </p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-yellow-400" />
              실생활 예시
            </h3>
            <div className="space-y-4">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">1. 진자 운동</h4>
                <p className="text-slate-300 text-sm">
                  최고점: PE 최대, KE = 0<br/>
                  최저점: KE 최대, PE 최소<br/>
                  항상 E_total = 일정
                </p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">2. 롤러코스터</h4>
                <p className="text-slate-300 text-sm">
                  높은 곳: 느리고 PE 높음<br/>
                  낮은 곳: 빠르고 KE 높음<br/>
                  총 에너지는 보존
                </p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">3. 자유 낙하</h4>
                <p className="text-slate-300 text-sm">
                  낙하하면서 PE → KE 전환<br/>
                  지면 도달 시 모든 PE가 KE로
                </p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 높이에서 떨어뜨린 공</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 10 m 높이에서 2 kg 공을 떨어뜨렸다. 지면 도달 속력은?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>초기(높이 10m): KE₁ = 0, PE₁ = mgh = 2×9.8×10 = 196 J</p>
                <p>최종(지면): KE₂ = ½mv², PE₂ = 0</p>
                <p>에너지 보존: PE₁ = KE₂</p>
                <p>196 = ½ × 2 × v²</p>
                <p>v² = 196, v = 14 m/s</p>
                <p className="text-purple-400 font-bold">답: 14 m/s</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: 비보존력과 에너지 손실 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">5. 비보존력과 에너지 손실</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">마찰이 있을 때</h3>
            <p className="text-slate-300 mb-4">
              마찰력 같은 비보존력이 있으면 역학적 에너지가 감소한다
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm mb-2">E_initial - E_final = W_friction</p>
              <p className="font-mono text-sm">(KE₁ + PE₁) - (KE₂ + PE₂) = f·d</p>
              <p className="text-xs text-slate-400 mt-2">
                (에너지 손실 = 마찰력이 한 일)
              </p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 경사면 미끄러짐 (마찰 있음)</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 10 m 높이 경사면에서 5 kg 물체가 미끄러진다. 마찰계수 0.2, 경사면 길이 15 m. 바닥 속력은?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>초기: PE = mgh = 5×9.8×10 = 490 J</p>
                <p>마찰력: f = μmg cos(θ) ≈ 0.2×5×9.8×0.8 = 7.84 N</p>
                <p>마찰 일: W_f = 7.84 × 15 = 117.6 J</p>
                <p>최종 KE: 490 - 117.6 = 372.4 J</p>
                <p>½mv² = 372.4, v = √(372.4/2.5) ≈ 12.2 m/s</p>
                <p className="text-purple-400 font-bold">답: 약 12.2 m/s</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 6: 일률 (Power) */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">6. 일률 (Power)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">일률의 정의</h3>
            <p className="text-slate-300 mb-4">
              단위 시간당 한 일의 양 (에너지 변환 속도)
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-lg font-bold">P = W / t = F · v</p>
              <div className="text-sm text-slate-300 space-y-1">
                <p>• P: 일률 (Power, Watt = J/s)</p>
                <p>• W: 일 (Joule)</p>
                <p>• t: 시간 (초)</p>
                <p>• v: 속도 (m/s)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">단위 환산</h3>
            <div className="space-y-2 text-slate-300">
              <p>• 1 W (Watt) = 1 J/s</p>
              <p>• 1 kW (Kilowatt) = 1000 W</p>
              <p>• 1 HP (마력) ≈ 746 W</p>
              <p>• 1 kWh (킬로와트시) = 3.6 × 10⁶ J</p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 계단 오르기</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 70 kg 사람이 10초 동안 5 m 높이 계단을 올랐다. 평균 일률은?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>일: W = mgh = 70 × 9.8 × 5 = 3430 J</p>
                <p>일률: P = W / t = 3430 / 10 = 343 W</p>
                <p className="text-purple-400 font-bold">답: 343 W (약 0.46 HP)</p>
              </div>
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
              <h3 className="font-semibold text-purple-300 mb-2">일과 에너지</h3>
              <p className="text-sm text-slate-300">
                W = F·d·cos(θ)<br/>
                W = ΔKE
              </p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">에너지 보존</h3>
              <p className="text-sm text-slate-300">
                KE + PE = 일정<br/>
                (마찰 없을 때)
              </p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">일률</h3>
              <p className="text-sm text-slate-300">
                P = W/t = F·v<br/>
                단위: Watt
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-8 border-t border-purple-500/20">
          <Link
            href="/modules/physics-fundamentals?chapter=kinematics"
            className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>이전: Chapter 2</span>
          </Link>
          <Link
            href="/modules/physics-fundamentals?chapter=momentum"
            className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors"
          >
            <span>다음: Chapter 4 - 운동량과 충돌</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
