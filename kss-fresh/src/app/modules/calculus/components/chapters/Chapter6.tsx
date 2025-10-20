'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter6() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <Link href="/modules/calculus" className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700 mb-8">
          <ArrowLeft className="w-4 h-4" />
          <span className="text-sm">모듈로 돌아가기</span>
        </Link>

        <div className="flex items-center gap-3 mb-8">
          <BookOpen className="w-8 h-8 text-green-400" />
          <div>
            <h1 className="text-4xl font-bold">Chapter 6: 급수와 수열</h1>
            <p className="text-slate-400 mt-2">Sequences and Series</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">수열 (Sequence)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">순서가 있는 수의 나열: a₁, a₂, a₃, ...</p>
            <div className="space-y-3">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">수렴</h3>
                <p className="font-mono text-sm">lim<sub>n→∞</sub> aₙ = L (극한 L이 존재)</p>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-red-400 mb-2">발산</h3>
                <p className="font-mono text-sm">극한이 존재하지 않거나 무한대</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">무한급수 (Infinite Series)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center mb-4">
              Σ<sub>n=1</sub><sup>∞</sup> aₙ = a₁ + a₂ + a₃ + ...
            </div>
            <p className="text-slate-300 text-sm">부분합 Sₙ = Σ<sub>k=1</sub><sup>n</sup> aₖ가 수렴하면 급수가 수렴</p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">수렴 판정법</h2>
          <div className="space-y-4">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">기하급수 (Geometric Series)</h3>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm mb-2">
                Σ ar<sup>n</sup> = a/(1-r), |r| {"<"} 1일 때 수렴
              </div>
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-green-400 mb-3">비율 판정법 (Ratio Test)</h3>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                L = lim<sub>n→∞</sub> |aₙ₊₁/aₙ|<br />
                L {"<"} 1: 수렴, L {">"} 1: 발산
              </div>
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-purple-400 mb-3">적분 판정법</h3>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                ∫<sub>1</sub><sup>∞</sup> f(x) dx가 수렴 ⟺ Σ f(n)이 수렴
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">테일러 급수 (Taylor Series)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">함수 f(x)의 x = a에서의 테일러 전개:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center mb-4">
              f(x) = Σ<sub>n=0</sub><sup>∞</sup> [f<sup>(n)</sup>(a)/n!]·(x-a)<sup>n</sup>
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <p className="font-semibold text-blue-400 mb-2">매클로린 급수 (a = 0):</p>
              <div className="font-mono text-sm space-y-1">
                <div>e<sup>x</sup> = 1 + x + x²/2! + x³/3! + ...</div>
                <div>sin x = x - x³/3! + x⁵/5! - ...</div>
                <div>cos x = 1 - x²/2! + x⁴/4! - ...</div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <CheckCircle className="w-6 h-6 text-green-400" />
            핵심 요약
          </h2>
          <div className="bg-gradient-to-br from-green-500/10 to-blue-500/10 border border-green-500/30 rounded-xl p-6">
            <ul className="space-y-2 text-slate-300 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>수열: 순서 있는 수의 나열</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>급수: 무한 개 항의 합</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>테일러 급수: 함수의 무한 전개</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/calculus/applications-integration" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 적분의 응용</span>
          </Link>
          <Link href="/modules/calculus/multivariable" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <span>다음: 다변수 미적분</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
