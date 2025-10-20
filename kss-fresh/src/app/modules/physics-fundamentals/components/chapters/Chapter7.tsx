'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter7() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="border-b border-purple-500/20 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <Link href="/modules/physics-fundamentals" className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors mb-3">
            <ArrowLeft className="w-4 h-4" />
            <span>Physics Fundamentals 모듈로 돌아가기</span>
          </Link>
          <h1 className="text-3xl font-bold mb-2">Chapter 7: 전자기학 입문</h1>
          <p className="text-slate-300">전기장, 자기장, 맥스웰 방정식</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">1. 전하와 쿨롱의 법칙</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">전하의 성질</h3>
            <ul className="space-y-2 text-slate-300">
              <li>• 양전하(+)와 음전하(-) 두 종류</li>
              <li>• 같은 전하는 반발, 다른 전하는 끌어당김</li>
              <li>• 전하량은 양자화: q = ne (e = 1.6 × 10⁻¹⁹ C)</li>
              <li>• 전하 보존: 고립계에서 총 전하량 일정</li>
            </ul>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">쿨롱의 법칙 (Coulomb's Law)</h3>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-sm">F = k(q₁q₂)/r²</p>
              <div className="text-xs text-slate-400 space-y-1">
                <p>• k = 8.99 × 10⁹ N·m²/C² (쿨롱 상수)</p>
                <p>• r: 전하 사이 거리</p>
                <p>• F: 전기력 (벡터)</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">2. 전기장 (Electric Field)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">전기장의 정의</h3>
            <p className="text-slate-300 mb-4">단위 양전하가 받는 힘</p>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">E = F/q = kQ/r²</p>
                <p className="text-xs text-slate-400 mt-2">단위: N/C or V/m</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">F = qE (전하가 받는 힘)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">전기장 선 (Field Lines)</h3>
            <ul className="space-y-2 text-slate-300">
              <li>• 양전하에서 나와 음전하로 들어감</li>
              <li>• 선이 조밀할수록 전기장이 강함</li>
              <li>• 접선 방향이 전기장 방향</li>
              <li>• 서로 교차하지 않음</li>
            </ul>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">3. 전위와 전위 에너지</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">전위 (Electric Potential)</h3>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">V = kQ/r</p>
                <p className="text-xs text-slate-400 mt-2">단위: Volt (V) = J/C</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">ΔV = -∫E·dl (전위차)</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">PE = qV (전위 에너지)</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">4. 자기장 (Magnetic Field)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">자기력</h3>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">F = qvB sin(θ) (전하가 받는 힘)</p>
                <p className="text-xs text-slate-400 mt-2">방향: 오른손 법칙</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">F = ILB sin(θ) (전류가 받는 힘)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">비오-사바르 법칙</h3>
            <p className="text-slate-300 mb-3">전류에 의한 자기장</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">B = (μ₀I)/(2πr) (무한히 긴 직선 전류)</p>
              <p className="text-xs text-slate-400 mt-2">μ₀ = 4π × 10⁻⁷ T·m/A</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">5. 전자기 유도 (Faraday's Law)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">패러데이 법칙</h3>
            <p className="text-slate-300 mb-4">자기 선속의 변화가 유도 기전력을 만든다</p>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">ε = -dΦ_B/dt = -N(dΦ_B/dt)</p>
                <p className="text-xs text-slate-400 mt-2">Φ_B = BA cos(θ): 자기 선속</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">렌츠 법칙: 유도 전류는 원인을 방해하는 방향</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">실생활 응용</h3>
            <div className="space-y-2">
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="text-sm text-slate-300">• 발전기: 역학 에너지 → 전기 에너지</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="text-sm text-slate-300">• 변압기: 전압 변환</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="text-sm text-slate-300">• 유도 가열: 와전류 이용</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">6. 맥스웰 방정식</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">전자기학의 4대 법칙</h3>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-semibold mb-1">1. 가우스 법칙 (전기)</p>
                <p className="font-mono text-sm">∇·E = ρ/ε₀</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-semibold mb-1">2. 가우스 법칙 (자기)</p>
                <p className="font-mono text-sm">∇·B = 0</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-semibold mb-1">3. 패러데이 법칙</p>
                <p className="font-mono text-sm">∇×E = -∂B/∂t</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-semibold mb-1">4. 앙페르-맥스웰 법칙</p>
                <p className="font-mono text-sm">∇×B = μ₀J + μ₀ε₀∂E/∂t</p>
              </div>
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">전자기파</h3>
            <p className="text-slate-300 mb-3">맥스웰 방정식으로부터 빛의 속도 유도:</p>
            <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm">
              <p>c = 1/√(μ₀ε₀) ≈ 3 × 10⁸ m/s</p>
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
              <h3 className="font-semibold text-purple-300 mb-2">전기장</h3>
              <p className="text-sm text-slate-300">E = kQ/r²<br/>V = kQ/r</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">자기장</h3>
              <p className="text-sm text-slate-300">F = qvB<br/>F = ILB</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">전자기 유도</h3>
              <p className="text-sm text-slate-300">ε = -dΦ/dt<br/>맥스웰 방정식</p>
            </div>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-purple-500/20">
          <Link href="/modules/physics-fundamentals?chapter=oscillations" className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: Chapter 6</span>
          </Link>
          <Link href="/modules/physics-fundamentals?chapter=thermodynamics" className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors">
            <span>다음: Chapter 8 - 열역학</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
