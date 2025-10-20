'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, AlertCircle, CheckCircle, Code } from 'lucide-react'

export default function Chapter1() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <Link
          href="/modules/linear-algebra"
          className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700 mb-8"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="text-sm">모듈로 돌아가기</span>
        </Link>

        <div className="flex items-center gap-3 mb-8">
          <BookOpen className="w-8 h-8 text-blue-400" />
          <div>
            <h1 className="text-4xl font-bold">Chapter 1: 벡터의 기초</h1>
            <p className="text-slate-400 mt-2">Vector Basics and Operations</p>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Lightbulb className="w-6 h-6 text-yellow-400" />
            벡터란 무엇인가?
          </h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 leading-relaxed mb-4">
              벡터(Vector)는 크기와 방향을 가진 물리량입니다. 스칼라(Scalar)가 단순히 크기만 가지는 것과 달리,
              벡터는 방향 정보도 함께 포함합니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4 mt-4">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">스칼라 예시</h3>
                <ul className="text-slate-300 text-sm space-y-1">
                  <li>• 온도: 25°C</li>
                  <li>• 질량: 5kg</li>
                  <li>• 시간: 10초</li>
                </ul>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">벡터 예시</h3>
                <ul className="text-slate-300 text-sm space-y-1">
                  <li>• 속도: 30m/s 북쪽으로</li>
                  <li>• 힘: 10N 오른쪽으로</li>
                  <li>• 변위: 5m 위쪽으로</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Vector Notation */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">벡터 표기법</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-2">1. 화살표 표기</h3>
                <p className="text-slate-300 mb-2">벡터 <span className="font-mono bg-slate-700/50 px-2 py-1 rounded">v⃗</span> 또는 굵은 글씨 <span className="font-mono bg-slate-700/50 px-2 py-1 rounded font-bold">v</span></p>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-2">2. 성분 표기 (2차원)</h3>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                  v = (v₁, v₂) = v₁i + v₂j
                </div>
                <p className="text-slate-400 text-sm mt-2">
                  여기서 i, j는 각각 x축, y축 방향의 단위 벡터입니다.
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-2">3. 성분 표기 (3차원)</h3>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                  v = (v₁, v₂, v₃) = v₁i + v₂j + v₃k
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-2">4. 열벡터 표기</h3>
                <div className="bg-slate-900/50 rounded-lg p-4">
                  <div className="font-mono text-sm">
                    v = [v₁]<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;[v₂]<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;[v₃]
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Vector Magnitude */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">벡터의 크기 (Magnitude)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">
              벡터의 크기는 ||v|| 또는 |v|로 표기하며, 피타고라스 정리를 사용하여 계산합니다.
            </p>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-green-400 mb-3">2차원 벡터</h3>
                <div className="bg-slate-900/50 rounded-lg p-4 mb-3">
                  <div className="font-mono text-sm mb-2">v = (v₁, v₂)</div>
                  <div className="font-mono text-lg text-green-400">
                    ||v|| = √(v₁² + v₂²)
                  </div>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                  <div className="text-sm font-semibold text-blue-400 mb-2">예시:</div>
                  <div className="font-mono text-sm text-slate-300">
                    v = (3, 4)<br/>
                    ||v|| = √(3² + 4²) = √25 = 5
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-green-400 mb-3">3차원 벡터</h3>
                <div className="bg-slate-900/50 rounded-lg p-4 mb-3">
                  <div className="font-mono text-sm mb-2">v = (v₁, v₂, v₃)</div>
                  <div className="font-mono text-lg text-green-400">
                    ||v|| = √(v₁² + v₂² + v₃²)
                  </div>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                  <div className="text-sm font-semibold text-blue-400 mb-2">예시:</div>
                  <div className="font-mono text-sm text-slate-300">
                    v = (2, 3, 6)<br/>
                    ||v|| = √(2² + 3² + 6²)<br/>
                    = √(4 + 9 + 36) = √49 = 7
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Vector Operations - Addition */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">벡터 연산</h2>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 mb-6">
            <h3 className="text-xl font-semibold text-blue-400 mb-4">1. 벡터 덧셈 (Vector Addition)</h3>
            <p className="text-slate-300 mb-4">
              두 벡터의 덧셈은 각 성분을 더하여 계산합니다.
            </p>

            <div className="bg-slate-900/50 rounded-lg p-4 mb-4">
              <div className="font-mono text-sm">
                u = (u₁, u₂, u₃)<br/>
                v = (v₁, v₂, v₃)<br/>
                <span className="text-green-400 text-base mt-2 block">
                  u + v = (u₁ + v₁, u₂ + v₂, u₃ + v₃)
                </span>
              </div>
            </div>

            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <div className="text-sm font-semibold text-blue-400 mb-2">예시:</div>
              <div className="font-mono text-sm text-slate-300">
                u = (1, 2, 3)<br/>
                v = (4, 5, 6)<br/>
                u + v = (1+4, 2+5, 3+6) = (5, 7, 9)
              </div>
            </div>
          </div>

          {/* Scalar Multiplication */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 mb-6">
            <h3 className="text-xl font-semibold text-blue-400 mb-4">2. 스칼라 곱셈 (Scalar Multiplication)</h3>
            <p className="text-slate-300 mb-4">
              벡터에 스칼라(실수)를 곱하면 각 성분에 그 스칼라를 곱합니다.
            </p>

            <div className="bg-slate-900/50 rounded-lg p-4 mb-4">
              <div className="font-mono text-sm">
                v = (v₁, v₂, v₃)<br/>
                k = 스칼라<br/>
                <span className="text-green-400 text-base mt-2 block">
                  k·v = (k·v₁, k·v₂, k·v₃)
                </span>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <div className="text-sm font-semibold text-blue-400 mb-2">예시 1: k = 2</div>
                <div className="font-mono text-sm text-slate-300">
                  v = (3, 4, 5)<br/>
                  2·v = (6, 8, 10)
                </div>
              </div>
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <div className="text-sm font-semibold text-blue-400 mb-2">예시 2: k = -1</div>
                <div className="font-mono text-sm text-slate-300">
                  v = (3, 4, 5)<br/>
                  -1·v = (-3, -4, -5)
                </div>
              </div>
            </div>
          </div>

          {/* Dot Product */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-xl font-semibold text-blue-400 mb-4">3. 내적 (Dot Product)</h3>
            <p className="text-slate-300 mb-4">
              두 벡터의 내적은 스칼라 값을 반환하며, 벡터 간의 관계를 나타냅니다.
            </p>

            <div className="bg-slate-900/50 rounded-lg p-4 mb-4">
              <div className="font-mono text-sm">
                u = (u₁, u₂, u₃)<br/>
                v = (v₁, v₂, v₃)<br/>
                <span className="text-green-400 text-base mt-2 block">
                  u · v = u₁v₁ + u₂v₂ + u₃v₃
                </span>
              </div>
            </div>

            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-4">
              <div className="text-sm font-semibold text-blue-400 mb-2">예시:</div>
              <div className="font-mono text-sm text-slate-300">
                u = (1, 2, 3)<br/>
                v = (4, 5, 6)<br/>
                u · v = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
              </div>
            </div>

            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold text-yellow-400 mb-1">중요한 성질</div>
                  <ul className="text-slate-300 text-sm space-y-1">
                    <li>• u · v = 0 ⟺ u와 v가 직교 (perpendicular)</li>
                    <li>• u · v &gt; 0 ⟺ 예각</li>
                    <li>• u · v &lt; 0 ⟺ 둔각</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <CheckCircle className="w-6 h-6 text-green-400" />
            핵심 요약
          </h2>
          <div className="bg-gradient-to-br from-green-500/10 to-blue-500/10 border border-green-500/30 rounded-xl p-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-green-400 mb-3">벡터 기본 개념</h3>
                <ul className="space-y-2 text-slate-300 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>벡터는 크기와 방향을 가진 물리량</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>크기는 ||v|| = √(v₁² + v₂² + v₃²)</span>
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-blue-400 mb-3">벡터 연산</h3>
                <ul className="space-y-2 text-slate-300 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                    <span>덧셈: 각 성분을 더함</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                    <span>내적: 스칼라 결과, 수직 판별</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Practice Problems */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Code className="w-6 h-6 text-purple-400" />
            연습 문제
          </h2>
          <div className="space-y-4">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <div className="font-semibold text-purple-400 mb-2">문제 1: 벡터 크기 계산</div>
              <p className="text-slate-300 text-sm mb-3">
                벡터 v = (6, 8, 0)의 크기를 구하시오.
              </p>
              <details className="cursor-pointer">
                <summary className="text-blue-400 hover:text-blue-300 text-sm">풀이 보기</summary>
                <div className="mt-3 bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-slate-300">
                  ||v|| = √(6² + 8² + 0²) = √100 = 10
                </div>
              </details>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <div className="font-semibold text-purple-400 mb-2">문제 2: 벡터 내적</div>
              <p className="text-slate-300 text-sm mb-3">
                u = (2, 3, 1), v = (1, 0, 4)일 때, u · v를 구하시오.
              </p>
              <details className="cursor-pointer">
                <summary className="text-blue-400 hover:text-blue-300 text-sm">풀이 보기</summary>
                <div className="mt-3 bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-slate-300">
                  u · v = 2×1 + 3×0 + 1×4 = 6
                </div>
              </details>
            </div>
          </div>
        </section>

        {/* Next Chapter */}
        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link
            href="/modules/linear-algebra"
            className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>모듈로 돌아가기</span>
          </Link>
          <Link
            href="/modules/linear-algebra/matrices"
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <span>다음: 행렬과 행렬 연산</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
