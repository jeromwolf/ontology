'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter6() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="border-b border-purple-500/20 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <Link href="/modules/physics-fundamentals" className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors mb-3">
            <ArrowLeft className="w-4 h-4" />
            <span>Physics Fundamentals 모듈로 돌아가기</span>
          </Link>
          <h1 className="text-3xl font-bold mb-2">Chapter 6: 진동과 파동</h1>
          <p className="text-slate-300">단순 조화 운동, 파동 방정식</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">1. 단순 조화 운동 (SHM)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">SHM의 정의</h3>
            <p className="text-slate-300 mb-4">복원력이 변위에 비례하는 진동 운동</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-sm">F = -kx (Hooke's Law)</p>
              <p className="font-mono text-sm">x(t) = A cos(ωt + φ)</p>
              <div className="text-xs text-slate-400 mt-2 space-y-1">
                <p>• A: 진폭 (Amplitude)</p>
                <p>• ω: 각진동수 (Angular Frequency, rad/s)</p>
                <p>• φ: 위상 (Phase)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">주요 공식</h3>
            <div className="space-y-2 font-mono text-sm">
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p>주기: T = 2π/ω = 2π√(m/k)</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p>진동수: f = 1/T = ω/(2π)</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p>각진동수: ω = √(k/m) = 2πf</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p>속도: v(t) = -Aω sin(ωt + φ)</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p>가속도: a(t) = -Aω² cos(ωt + φ) = -ω²x</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">2. 단진자 (Simple Pendulum)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">단진자의 주기</h3>
            <p className="text-slate-300 mb-4">작은 각도 진동 (θ &lt; 15°)</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">T = 2π√(L/g)</p>
              <p className="text-xs text-slate-400 mt-2">L: 줄 길이, g: 중력가속도</p>
              <p className="text-xs text-slate-400">주기는 질량과 무관!</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">3. SHM의 에너지</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">에너지 보존</h3>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">운동에너지: KE = (1/2)mv² = (1/2)mω²A²sin²(ωt+φ)</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">위치에너지: PE = (1/2)kx² = (1/2)kA²cos²(ωt+φ)</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">총 에너지: E = KE + PE = (1/2)kA² = constant</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">4. 파동 (Waves)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">파동의 종류</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">횡파 (Transverse)</h4>
                <p className="text-slate-300 text-sm">진행 방향에 수직으로 진동<br/>예: 빛, 전자기파</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">종파 (Longitudinal)</h4>
                <p className="text-slate-300 text-sm">진행 방향과 평행하게 진동<br/>예: 소리, 압력파</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">파동 방정식</h3>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-sm">y(x,t) = A sin(kx - ωt + φ)</p>
              <div className="text-xs text-slate-400 space-y-1">
                <p>• k: 파수 (Wave number, k = 2π/λ)</p>
                <p>• λ: 파장 (Wavelength)</p>
                <p>• v: 파동 속력 (v = λf = ω/k)</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">5. 파동의 간섭과 공명</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">간섭 (Interference)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h4 className="font-semibold text-green-400 mb-2">보강 간섭</h4>
                <p className="text-sm text-slate-300">위상차 = 2nπ<br/>진폭이 증가</p>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                <h4 className="font-semibold text-red-400 mb-2">상쇄 간섭</h4>
                <p className="text-sm text-slate-300">위상차 = (2n+1)π<br/>진폭이 감소</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">공명 (Resonance)</h3>
            <p className="text-slate-300 mb-4">외부 진동수가 고유 진동수와 일치할 때 진폭이 크게 증가</p>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <p className="text-sm text-slate-300">예: 악기, 다리, 건물, MRI</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">6. 도플러 효과 (Doppler Effect)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">진동수 변화</h3>
            <p className="text-slate-300 mb-4">음원 또는 관찰자가 움직일 때 진동수가 변화</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">f' = f(v ± v_o)/(v ∓ v_s)</p>
              <p className="text-xs text-slate-400 mt-2">v: 파동 속력, v_o: 관찰자 속력, v_s: 음원 속력</p>
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
              <h3 className="font-semibold text-purple-300 mb-2">SHM</h3>
              <p className="text-sm text-slate-300">x = A cos(ωt)<br/>T = 2π√(m/k)</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">파동</h3>
              <p className="text-sm text-slate-300">v = λf<br/>간섭과 공명</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">도플러</h3>
              <p className="text-sm text-slate-300">진동수 변화<br/>f' ≠ f</p>
            </div>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-purple-500/20">
          <Link href="/modules/physics-fundamentals?chapter=rotation" className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: Chapter 5</span>
          </Link>
          <Link href="/modules/physics-fundamentals?chapter=electromagnetism" className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors">
            <span>다음: Chapter 7 - 전자기학 입문</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
