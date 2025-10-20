'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter7() {
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
            <h1 className="text-4xl font-bold">Chapter 7: 다변수 미적분</h1>
            <p className="text-slate-400 mt-2">Multivariable Calculus</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">편미분 (Partial Derivative)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">다변수 함수 f(x, y)의 편미분:</p>
            <div className="space-y-3">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">x에 대한 편미분</h3>
                <p className="font-mono text-sm">∂f/∂x = lim<sub>h→0</sub> [f(x+h,y) - f(x,y)]/h</p>
                <p className="text-slate-400 text-sm mt-2">y를 상수로 취급</p>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">y에 대한 편미분</h3>
                <p className="font-mono text-sm">∂f/∂y = lim<sub>h→0</sub> [f(x,y+h) - f(x,y)]/h</p>
                <p className="text-slate-400 text-sm mt-2">x를 상수로 취급</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">그래디언트 (Gradient)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">함수의 최대 증가 방향을 나타내는 벡터:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center mb-4">
              ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)
            </div>
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <p className="text-slate-300 text-sm">
                <span className="font-semibold text-yellow-400">성질:</span><br />
                • ∇f는 f가 가장 빠르게 증가하는 방향<br />
                • |∇f|는 그 방향으로의 변화율<br />
                • ∇f ⊥ 등고선
              </p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">방향도함수 (Directional Derivative)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">단위벡터 u 방향으로의 변화율:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center">
              D<sub>u</sub>f = ∇f · u = |∇f| cos θ
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">이중적분 (Double Integral)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">영역 R 위에서의 적분:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center mb-4">
              ∬<sub>R</sub> f(x,y) dA
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <h3 className="font-semibold text-blue-400 mb-2">푸비니 정리:</h3>
              <div className="font-mono text-sm">
                ∬<sub>R</sub> f(x,y) dA = ∫<sub>a</sub><sup>b</sup> ∫<sub>c</sub><sup>d</sup> f(x,y) dy dx
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">응용: 질량과 질량중심</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                <h3 className="font-semibold text-green-400 mb-2">질량 (Mass)</h3>
                <p className="font-mono text-sm">m = ∬<sub>R</sub> ρ(x,y) dA</p>
              </div>
              <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <h3 className="font-semibold text-purple-400 mb-2">질량중심 (Center of Mass)</h3>
                <p className="font-mono text-sm">
                  x̄ = (1/m)∬ xρ(x,y) dA<br />
                  ȳ = (1/m)∬ yρ(x,y) dA
                </p>
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
                <span>편미분: 다른 변수를 상수로 취급</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>그래디언트: 최대 증가 방향 벡터</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>이중적분: 영역 위의 적분</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/calculus/sequences-series" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 급수와 수열</span>
          </Link>
          <Link href="/modules/calculus/vector-calculus" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <span>다음: 벡터 미적분</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
