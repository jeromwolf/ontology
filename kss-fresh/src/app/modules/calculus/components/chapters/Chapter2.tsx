'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter2() {
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
            <h1 className="text-4xl font-bold">Chapter 2: 미분법</h1>
            <p className="text-slate-400 mt-2">Derivatives</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">도함수의 정의</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">함수 f(x)의 도함수 f'(x)는 다음과 같이 정의됩니다:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center">
              f'(x) = lim<sub>h→0</sub> [f(x + h) - f(x)] / h
            </div>
            <div className="mt-4 text-slate-300 text-sm">
              <p>기하학적 의미: x에서의 접선의 기울기</p>
              <p>물리학적 의미: 순간 변화율</p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">기본 미분 공식</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-3 font-mono text-sm">
              <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <p>d/dx (c) = 0 (상수의 미분)</p>
              </div>
              <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                <p>d/dx (x<sup>n</sup>) = nx<sup>n-1</sup> (거듭제곱 법칙)</p>
              </div>
              <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <p>d/dx (sin x) = cos x</p>
              </div>
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <p>d/dx (cos x) = -sin x</p>
              </div>
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p>d/dx (e<sup>x</sup>) = e<sup>x</sup></p>
              </div>
              <div className="p-3 bg-pink-500/10 border border-pink-500/30 rounded-lg">
                <p>d/dx (ln x) = 1/x</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">미분 법칙</h2>
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">합과 차의 미분</h3>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                [f(x) ± g(x)]' = f'(x) ± g'(x)
              </div>
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-green-400 mb-3">곱의 미분 (Product Rule)</h3>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                [f(x)g(x)]' = f'(x)g(x) + f(x)g'(x)
              </div>
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-purple-400 mb-3">몫의 미분 (Quotient Rule)</h3>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                [f(x)/g(x)]' = [f'(x)g(x) - f(x)g'(x)] / [g(x)]<sup>2</sup>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">연쇄법칙 (Chain Rule)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">합성함수의 미분:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center mb-4">
              [f(g(x))]' = f'(g(x)) · g'(x)
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <p className="font-semibold text-blue-400 mb-2">예시:</p>
              <div className="font-mono text-sm space-y-2">
                <div>y = (x<sup>2</sup> + 1)<sup>3</sup></div>
                <div>외부 함수: f(u) = u<sup>3</sup>, f'(u) = 3u<sup>2</sup></div>
                <div>내부 함수: g(x) = x<sup>2</sup> + 1, g'(x) = 2x</div>
                <div className="text-green-400">dy/dx = 3(x<sup>2</sup> + 1)<sup>2</sup> · 2x = 6x(x<sup>2</sup> + 1)<sup>2</sup></div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">음함수 미분 (Implicit Differentiation)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">y가 x에 대해 명시적으로 표현되지 않은 경우:</p>
            <div className="bg-slate-900/50 rounded-lg p-6">
              <div className="text-green-400 text-sm mb-3">예시: x<sup>2</sup> + y<sup>2</sup> = 25</div>
              <div className="font-mono text-sm space-y-2">
                <div>양변을 x에 대해 미분:</div>
                <div>2x + 2y(dy/dx) = 0</div>
                <div>2y(dy/dx) = -2x</div>
                <div className="text-green-400">dy/dx = -x/y</div>
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
                <span>도함수: f'(x) = lim<sub>h→0</sub> [f(x+h) - f(x)]/h</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>거듭제곱: (x<sup>n</sup>)' = nx<sup>n-1</sup></span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>곱의 미분: (fg)' = f'g + fg'</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>연쇄법칙: [f(g(x))]' = f'(g(x))·g'(x)</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/calculus/limits" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 극한과 연속</span>
          </Link>
          <Link href="/modules/calculus/applications-derivatives" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <span>다음: 미분의 응용</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
