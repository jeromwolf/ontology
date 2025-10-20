'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter5() {
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
            <h1 className="text-4xl font-bold">Chapter 5: 적분의 응용</h1>
            <p className="text-slate-400 mt-2">Applications of Integration</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">넓이 (Area)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">곡선과 x축 사이</h3>
                <p className="font-mono text-sm">A = ∫<sub>a</sub><sup>b</sup> f(x) dx</p>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">두 곡선 사이</h3>
                <p className="font-mono text-sm">A = ∫<sub>a</sub><sup>b</sup> [f(x) - g(x)] dx</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">회전체의 부피 (Volume)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">원판법 (Disk Method)</h3>
                <p className="text-slate-300 text-sm mb-2">x축 중심 회전:</p>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                  V = π ∫<sub>a</sub><sup>b</sup> [f(x)]<sup>2</sup> dx
                </div>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-green-400 mb-3">껍질법 (Shell Method)</h3>
                <p className="text-slate-300 text-sm mb-2">y축 중심 회전:</p>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                  V = 2π ∫<sub>a</sub><sup>b</sup> x·f(x) dx
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">곡선의 길이 (Arc Length)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center mb-4">
              L = ∫<sub>a</sub><sup>b</sup> √[1 + (f'(x))<sup>2</sup>] dx
            </div>
            <p className="text-slate-300 text-sm">y = f(x) 곡선의 x = a부터 x = b까지의 길이</p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">평균값 (Average Value)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">[a, b]에서 f(x)의 평균값:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center">
              f<sub>avg</sub> = 1/(b-a) · ∫<sub>a</sub><sup>b</sup> f(x) dx
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
                <span>넓이: ∫<sub>a</sub><sup>b</sup> f(x) dx</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>부피: π ∫[f(x)]<sup>2</sup> dx (원판법)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>곡선 길이: ∫√[1 + (f')<sup>2</sup>] dx</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/calculus/integration" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 적분법</span>
          </Link>
          <Link href="/modules/calculus/sequences-series" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <span>다음: 급수와 수열</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
