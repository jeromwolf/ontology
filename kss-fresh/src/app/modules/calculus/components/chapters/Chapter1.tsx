'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, CheckCircle, AlertCircle } from 'lucide-react'

export default function Chapter1() {
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
            <h1 className="text-4xl font-bold">Chapter 1: 극한과 연속</h1>
            <p className="text-slate-400 mt-2">Limits and Continuity</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Lightbulb className="w-6 h-6 text-yellow-400" />
            극한(Limit)이란?
          </h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">함수 f(x)에서 x가 a에 가까워질 때 f(x)가 L에 가까워지면, L을 극한값이라 합니다.</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center">
              lim<sub>x→a</sub> f(x) = L
            </div>
            <div className="mt-4 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <p className="text-slate-300 text-sm">
                "x가 a에 한없이 가까워질 때, f(x)는 L에 한없이 가까워진다"
              </p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">극한의 기본 성질</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">1. 극한의 합</h3>
                <p className="font-mono text-sm">lim<sub>x→a</sub> [f(x) + g(x)] = lim<sub>x→a</sub> f(x) + lim<sub>x→a</sub> g(x)</p>
              </div>
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">2. 극한의 곱</h3>
                <p className="font-mono text-sm">lim<sub>x→a</sub> [f(x) · g(x)] = lim<sub>x→a</sub> f(x) · lim<sub>x→a</sub> g(x)</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-purple-400 mb-2">3. 극한의 나눗셈</h3>
                <p className="font-mono text-sm">
                  lim<sub>x→a</sub> [f(x) / g(x)] = lim<sub>x→a</sub> f(x) / lim<sub>x→a</sub> g(x)
                  <span className="text-slate-400"> (단, lim<sub>x→a</sub> g(x) ≠ 0)</span>
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">좌극한과 우극한</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">좌극한 (Left-hand Limit)</h3>
                <p className="text-slate-300 mb-3">x가 a보다 작은 값에서 a로 접근</p>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                  lim<sub>x→a<sup>-</sup></sub> f(x) = L
                </div>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-green-400 mb-3">우극한 (Right-hand Limit)</h3>
                <p className="text-slate-300 mb-3">x가 a보다 큰 값에서 a로 접근</p>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                  lim<sub>x→a<sup>+</sup></sub> f(x) = L
                </div>
              </div>
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                <p className="text-slate-300 text-sm">
                  <span className="font-semibold text-yellow-400">중요!</span> 극한이 존재하려면 좌극한과 우극한이 같아야 합니다:<br />
                  lim<sub>x→a</sub> f(x) = L ⟺ lim<sub>x→a<sup>-</sup></sub> f(x) = lim<sub>x→a<sup>+</sup></sub> f(x) = L
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">무한대의 극한</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div className="bg-slate-900/50 rounded-lg p-6">
                <div className="text-green-400 text-sm mb-3">예시 1: x → ∞</div>
                <div className="font-mono text-sm space-y-2">
                  <div>lim<sub>x→∞</sub> (1/x) = 0</div>
                  <div>lim<sub>x→∞</sub> (1/x<sup>2</sup>) = 0</div>
                  <div>lim<sub>x→∞</sub> (3x<sup>2</sup> + 2x + 1)/(x<sup>2</sup> + 5) = 3</div>
                </div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-6">
                <div className="text-blue-400 text-sm mb-3">예시 2: 함수 → ∞</div>
                <div className="font-mono text-sm space-y-2">
                  <div>lim<sub>x→0</sub> (1/x<sup>2</sup>) = ∞</div>
                  <div>lim<sub>x→0<sup>+</sup></sub> (1/x) = ∞</div>
                  <div>lim<sub>x→0<sup>-</sup></sub> (1/x) = -∞</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">연속 함수 (Continuous Function)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">함수 f가 x = a에서 연속이려면 다음 3가지 조건을 만족해야 합니다:</p>
            <div className="space-y-3">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">1. f(a)가 정의됨</h3>
                <p className="text-slate-300 text-sm">함수값 f(a)가 존재해야 함</p>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">2. 극한이 존재</h3>
                <p className="text-slate-300 text-sm">lim<sub>x→a</sub> f(x)가 존재해야 함</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-purple-400 mb-2">3. 극한값 = 함수값</h3>
                <p className="text-slate-300 text-sm">lim<sub>x→a</sub> f(x) = f(a)</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">중간값 정리 (Intermediate Value Theorem)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">함수 f가 [a, b]에서 연속이고 f(a) ≠ f(b)이면,</p>
            <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 border border-green-500/30 rounded-lg p-6">
              <p className="text-slate-300 text-center">
                f(a)와 f(b) 사이의 임의의 값 k에 대해,<br />
                <span className="font-semibold text-green-400">f(c) = k를 만족하는 c ∈ (a, b)가 존재합니다</span>
              </p>
            </div>
            <div className="mt-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <p className="text-slate-300 text-sm">
                <span className="font-semibold text-yellow-400">응용:</span> 방정식의 근 존재성 증명에 사용됩니다.
              </p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">연습 문제</h2>
          <div className="space-y-4">
            <details className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 cursor-pointer">
              <summary className="font-semibold text-lg mb-4 cursor-pointer">문제 1: 극한 계산</summary>
              <div className="space-y-4 mt-4">
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                  <p className="font-mono">lim<sub>x→2</sub> (x<sup>2</sup> - 4)/(x - 2)를 구하시오.</p>
                </div>
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                  <p className="font-semibold text-green-400 mb-2">풀이:</p>
                  <div className="font-mono text-sm space-y-2 text-slate-300">
                    <div>분자를 인수분해: (x<sup>2</sup> - 4) = (x - 2)(x + 2)</div>
                    <div>= lim<sub>x→2</sub> (x - 2)(x + 2)/(x - 2)</div>
                    <div>= lim<sub>x→2</sub> (x + 2)</div>
                    <div className="text-green-400">= 2 + 2 = 4</div>
                  </div>
                </div>
              </div>
            </details>

            <details className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 cursor-pointer">
              <summary className="font-semibold text-lg mb-4 cursor-pointer">문제 2: 연속성 판별</summary>
              <div className="space-y-4 mt-4">
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                  <p className="font-mono">
                    f(x) = (x<sup>2</sup> - 1)/(x - 1), x ≠ 1<br />
                    f(x) = 2, x = 1<br />
                    이 함수가 x = 1에서 연속인지 판별하시오.
                  </p>
                </div>
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                  <p className="font-semibold text-green-400 mb-2">풀이:</p>
                  <div className="font-mono text-sm space-y-2 text-slate-300">
                    <div>1. f(1) = 2 (정의됨 ✓)</div>
                    <div>2. lim<sub>x→1</sub> f(x) = lim<sub>x→1</sub> (x - 1)(x + 1)/(x - 1) = lim<sub>x→1</sub> (x + 1) = 2 (존재 ✓)</div>
                    <div>3. lim<sub>x→1</sub> f(x) = f(1) = 2 (같음 ✓)</div>
                    <div className="text-green-400">따라서 x = 1에서 연속입니다.</div>
                  </div>
                </div>
              </div>
            </details>

            <details className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 cursor-pointer">
              <summary className="font-semibold text-lg mb-4 cursor-pointer">문제 3: 무한대 극한</summary>
              <div className="space-y-4 mt-4">
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                  <p className="font-mono">lim<sub>x→∞</sub> (2x<sup>3</sup> + 3x<sup>2</sup> - 1)/(x<sup>3</sup> - 5x + 2)를 구하시오.</p>
                </div>
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                  <p className="font-semibold text-green-400 mb-2">풀이:</p>
                  <div className="font-mono text-sm space-y-2 text-slate-300">
                    <div>분자와 분모를 최고차항 x<sup>3</sup>로 나눔:</div>
                    <div>= lim<sub>x→∞</sub> (2 + 3/x - 1/x<sup>3</sup>)/(1 - 5/x<sup>2</sup> + 2/x<sup>3</sup>)</div>
                    <div>x → ∞일 때, 1/x → 0, 1/x<sup>2</sup> → 0, 1/x<sup>3</sup> → 0</div>
                    <div className="text-green-400">= (2 + 0 - 0)/(1 - 0 + 0) = 2</div>
                  </div>
                </div>
              </div>
            </details>
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
                <span>극한: x → a일 때 f(x) → L</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>극한의 연산: 합, 곱, 나눗셈 가능</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>연속: lim<sub>x→a</sub> f(x) = f(a)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>중간값 정리: 연속함수는 사잇값을 모두 가짐</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/calculus" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>모듈로 돌아가기</span>
          </Link>
          <Link href="/modules/calculus/derivatives" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <span>다음: 미분법</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
