'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter4() {
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
            <h1 className="text-4xl font-bold">Chapter 4: 적분법</h1>
            <p className="text-slate-400 mt-2">Integration</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">부정적분 (Indefinite Integral)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">F'(x) = f(x)일 때, F(x)를 f(x)의 부정적분이라 합니다:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center">
              ∫ f(x) dx = F(x) + C
            </div>
            <p className="text-slate-400 text-sm mt-4">C는 적분상수</p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">기본 적분 공식</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-3 font-mono text-sm">
              <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                ∫ x<sup>n</sup> dx = x<sup>n+1</sup>/(n+1) + C (n ≠ -1)
              </div>
              <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                ∫ 1/x dx = ln|x| + C
              </div>
              <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                ∫ e<sup>x</sup> dx = e<sup>x</sup> + C
              </div>
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                ∫ sin x dx = -cos x + C
              </div>
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                ∫ cos x dx = sin x + C
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">정적분 (Definite Integral)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center mb-4">
              ∫<sub>a</sub><sup>b</sup> f(x) dx = F(b) - F(a)
            </div>
            <p className="text-slate-300 text-sm">기하학적 의미: 곡선 아래의 넓이</p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">미적분학의 기본 정리</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-6">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">제1 기본정리</h3>
                <p className="font-mono text-sm">d/dx [∫<sub>a</sub><sup>x</sup> f(t) dt] = f(x)</p>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">제2 기본정리</h3>
                <p className="font-mono text-sm">∫<sub>a</sub><sup>b</sup> f(x) dx = F(b) - F(a)</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">치환적분 (Substitution)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">u = g(x)로 치환:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center mb-4">
              ∫ f(g(x))g'(x) dx = ∫ f(u) du
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <p className="font-semibold text-blue-400 mb-2">예시:</p>
              <div className="font-mono text-sm space-y-2">
                <div>∫ 2x·e<sup>x²</sup> dx</div>
                <div>u = x², du = 2x dx</div>
                <div>= ∫ e<sup>u</sup> du = e<sup>u</sup> + C</div>
                <div className="text-green-400">= e<sup>x²</sup> + C</div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">부분적분 (Integration by Parts)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center mb-4">
              ∫ u dv = uv - ∫ v du
            </div>
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
              <p className="font-semibold text-green-400 mb-2">예시:</p>
              <div className="font-mono text-sm space-y-2">
                <div>∫ x·e<sup>x</sup> dx</div>
                <div>u = x, dv = e<sup>x</sup> dx</div>
                <div>du = dx, v = e<sup>x</sup></div>
                <div>= x·e<sup>x</sup> - ∫ e<sup>x</sup> dx</div>
                <div className="text-green-400">= x·e<sup>x</sup> - e<sup>x</sup> + C</div>
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
                <span>부정적분: ∫ f(x) dx = F(x) + C</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>정적분: ∫<sub>a</sub><sup>b</sup> f(x) dx = F(b) - F(a)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>치환적분: u = g(x) 치환</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>부분적분: ∫ u dv = uv - ∫ v du</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/calculus/applications-derivatives" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 미분의 응용</span>
          </Link>
          <Link href="/modules/calculus/applications-integration" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <span>다음: 적분의 응용</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
