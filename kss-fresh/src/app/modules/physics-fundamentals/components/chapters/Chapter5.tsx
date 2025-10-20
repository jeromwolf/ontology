'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="border-b border-purple-500/20 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <Link href="/modules/physics-fundamentals" className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors mb-3">
            <ArrowLeft className="w-4 h-4" />
            <span>Physics Fundamentals 모듈로 돌아가기</span>
          </Link>
          <h1 className="text-3xl font-bold mb-2">Chapter 5: 회전 운동</h1>
          <p className="text-slate-300">각속도, 관성모멘트, 토크</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">1. 회전 운동학</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">각도, 각속도, 각가속도</h3>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">θ: 각도 (rad)</p>
                <p className="font-mono text-sm">ω = dθ/dt: 각속도 (rad/s)</p>
                <p className="font-mono text-sm">α = dω/dt: 각가속도 (rad/s²)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">선운동과 회전운동 대응</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">선운동</h4>
                <p className="font-mono text-sm">x, v, a</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">회전운동</h4>
                <p className="font-mono text-sm">θ, ω, α</p>
              </div>
            </div>
            <div className="mt-4 bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">v = rω, a_t = rα, a_c = rω²</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">2. 관성모멘트 (Moment of Inertia)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">정의</h3>
            <p className="text-slate-300 mb-4">회전 관성의 척도 (질량의 회전 버전)</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">I = Σmᵢrᵢ² = ∫r²dm</p>
              <p className="text-xs text-slate-400 mt-2">단위: kg·m²</p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">주요 물체의 관성모멘트</h3>
            <div className="space-y-2 text-sm">
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="font-semibold text-purple-300">얇은 막대 (중심축): I = (1/12)ML²</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="font-semibold text-purple-300">원판 (중심축): I = (1/2)MR²</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="font-semibold text-purple-300">구 (중심축): I = (2/5)MR²</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="font-semibold text-purple-300">고리 (중심축): I = MR²</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">3. 토크 (Torque)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">토크의 정의</h3>
            <p className="text-slate-300 mb-4">회전력, 물체를 회전시키는 능력</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-2">
              <p className="font-mono text-sm">τ = r × F = rF sin(θ)</p>
              <p className="text-xs text-slate-400">r: 회전축으로부터 거리, F: 힘, θ: 각도</p>
              <p className="text-xs text-slate-400 mt-2">단위: N·m</p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">회전 운동 방정식</h3>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">Στ = Iα</p>
              <p className="text-xs text-slate-400 mt-2">(F = ma의 회전 버전)</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">4. 각운동량 (Angular Momentum)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">각운동량과 보존</h3>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">L = Iω = r × p</p>
                <p className="text-xs text-slate-400 mt-2">단위: kg·m²/s</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">외부 토크가 0이면: L = constant</p>
                <p className="text-xs text-slate-400 mt-2">각운동량 보존 법칙</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">실생활 예시</h3>
            <div className="space-y-3">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">피겨 스케이팅</h4>
                <p className="text-slate-300 text-sm">팔을 오므리면 I 감소 → ω 증가 (L 보존)</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">태양계</h4>
                <p className="text-slate-300 text-sm">행성의 공전: 타원 궤도에서 각운동량 보존</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">5. 회전 운동 에너지</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">회전 에너지</h3>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-sm">KE_rot = (1/2)Iω²</p>
              <p className="font-mono text-sm">총 에너지 = (1/2)mv² + (1/2)Iω²</p>
              <p className="text-xs text-slate-400 mt-2">(병진 + 회전)</p>
            </div>
          </div>
        </section>

        <section className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <CheckCircle className="w-8 h-8 text-purple-400" />
            핵심 요약
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">관성모멘트</h3>
              <p className="text-sm text-slate-300">I = Σmr²<br/>회전 관성</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">토크</h3>
              <p className="text-sm text-slate-300">τ = rF sin(θ)<br/>Στ = Iα</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">각운동량</h3>
              <p className="text-sm text-slate-300">L = Iω<br/>보존 법칙</p>
            </div>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-purple-500/20">
          <Link href="/modules/physics-fundamentals?chapter=momentum" className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: Chapter 4</span>
          </Link>
          <Link href="/modules/physics-fundamentals?chapter=oscillations" className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors">
            <span>다음: Chapter 6 - 진동과 파동</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
