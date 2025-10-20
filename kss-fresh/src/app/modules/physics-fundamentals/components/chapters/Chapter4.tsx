'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="border-b border-purple-500/20 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <Link href="/modules/physics-fundamentals" className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors mb-3">
            <ArrowLeft className="w-4 h-4" />
            <span>Physics Fundamentals 모듈로 돌아가기</span>
          </Link>
          <h1 className="text-3xl font-bold mb-2">Chapter 4: 운동량과 충돌</h1>
          <p className="text-slate-300">운동량 보존, 탄성/비탄성 충돌</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">1. 운동량 (Momentum)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">운동량의 정의</h3>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-lg font-bold">p = mv</p>
              <div className="text-sm text-slate-300 space-y-1">
                <p>• p: 운동량 (Momentum, kg·m/s)</p>
                <p>• m: 질량 (kg)</p>
                <p>• v: 속도 (m/s, 벡터)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">충격량 (Impulse)</h3>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm mb-2">J = FΔt = Δp = m(v_f - v_i)</p>
              <p className="text-xs text-slate-400">힘 × 시간 = 운동량 변화</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">2. 운동량 보존 법칙</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">보존 법칙</h3>
            <p className="text-slate-300 mb-4">외력이 없으면 계의 총 운동량은 보존된다</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">p_before = p_after</p>
              <p className="font-mono text-sm">m₁v₁ᵢ + m₂v₂ᵢ = m₁v₁f + m₂v₂f</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">3. 탄성 충돌 (Elastic Collision)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">탄성 충돌의 특징</h3>
            <p className="text-slate-300 mb-4">운동량과 운동에너지가 모두 보존</p>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">운동량 보존: m₁v₁ᵢ + m₂v₂ᵢ = m₁v₁f + m₂v₂f</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">에너지 보존: ½m₁v₁ᵢ² + ½m₂v₂ᵢ² = ½m₁v₁f² + ½m₂v₂f²</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">1차원 탄성 충돌 공식</h3>
            <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
              <p>v₁f = ((m₁-m₂)/(m₁+m₂))v₁ᵢ + (2m₂/(m₁+m₂))v₂ᵢ</p>
              <p>v₂f = (2m₁/(m₁+m₂))v₁ᵢ + ((m₂-m₁)/(m₁+m₂))v₂ᵢ</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">4. 비탄성 충돌 (Inelastic Collision)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">완전 비탄성 충돌</h3>
            <p className="text-slate-300 mb-4">충돌 후 한 덩어리로 붙어서 함께 움직임</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">m₁v₁ᵢ + m₂v₂ᵢ = (m₁+m₂)v_f</p>
              <p className="text-xs text-slate-400 mt-2">운동량만 보존, 에너지는 손실</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">5. 2차원 충돌</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">성분별 보존</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h4 className="font-semibold mb-2">x 방향</h4>
                <p className="font-mono text-sm">Σp_x,i = Σp_x,f</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h4 className="font-semibold mb-2">y 방향</h4>
                <p className="font-mono text-sm">Σp_y,i = Σp_y,f</p>
              </div>
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
              <h3 className="font-semibold text-purple-300 mb-2">운동량</h3>
              <p className="text-sm text-slate-300">p = mv<br/>J = FΔt = Δp</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">탄성 충돌</h3>
              <p className="text-sm text-slate-300">운동량 + 에너지<br/>모두 보존</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">비탄성 충돌</h3>
              <p className="text-sm text-slate-300">운동량만 보존<br/>에너지 손실</p>
            </div>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-purple-500/20">
          <Link href="/modules/physics-fundamentals?chapter=energy-work" className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: Chapter 3</span>
          </Link>
          <Link href="/modules/physics-fundamentals?chapter=rotation" className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors">
            <span>다음: Chapter 5 - 회전 운동</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
