'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, CheckCircle, Code, AlertCircle } from 'lucide-react'

export default function Chapter3() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="max-w-4xl mx-auto px-6 py-12">
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
            <h1 className="text-4xl font-bold">Chapter 3: 선형 시스템과 가우스 소거법</h1>
            <p className="text-slate-400 mt-2">Linear Systems and Gaussian Elimination</p>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Lightbulb className="w-6 h-6 text-yellow-400" />
            선형 연립방정식이란?
          </h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 leading-relaxed mb-4">
              여러 개의 선형 방정식을 동시에 만족하는 해를 찾는 문제입니다.
            </p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
              2x + 3y = 8<br/>
              4x - y = 2
            </div>
          </div>
        </section>

        {/* Matrix Form */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">행렬 형태로 표현</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">선형 시스템을 Ax = b 형태로 표현할 수 있습니다.</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm mb-4">
              [2  3] [x]   [8]<br/>
              [4 -1] [y] = [2]
            </div>
            <ul className="text-slate-300 text-sm space-y-2">
              <li>• A: 계수 행렬 (coefficient matrix)</li>
              <li>• x: 미지수 벡터 (variable vector)</li>
              <li>• b: 상수 벡터 (constant vector)</li>
            </ul>
          </div>
        </section>

        {/* Augmented Matrix */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">첨가 행렬 (Augmented Matrix)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">계수 행렬과 상수 벡터를 결합한 행렬 [A|b]</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
              [2  3 | 8]<br/>
              [4 -1 | 2]
            </div>
          </div>
        </section>

        {/* Row Operations */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">기본 행 연산 (Elementary Row Operations)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">1. 행 교환 (Row Swap)</h3>
                <p className="text-slate-300 text-sm">R<sub>i</sub> ↔ R<sub>j</sub>: i번째 행과 j번째 행을 바꿈</p>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">2. 행 스케일링 (Row Scaling)</h3>
                <p className="text-slate-300 text-sm">kR<sub>i</sub>: i번째 행에 0이 아닌 상수 k를 곱함</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-purple-400 mb-2">3. 행 덧셈 (Row Addition)</h3>
                <p className="text-slate-300 text-sm">R<sub>i</sub> + kR<sub>j</sub> → R<sub>i</sub>: j번째 행의 k배를 i번째 행에 더함</p>
              </div>
            </div>
          </div>
        </section>

        {/* Gaussian Elimination */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">가우스 소거법 (Gaussian Elimination)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">기본 행 연산을 이용하여 행 사다리꼴 형태(Row Echelon Form)로 변환</p>

            <div className="bg-slate-900/50 rounded-lg p-6 mb-4">
              <div className="text-green-400 text-sm mb-3">예시:</div>
              <div className="font-mono text-xs space-y-3">
                <div>
                  <div className="text-slate-400 mb-1">원래 첨가 행렬:</div>
                  [2  3 | 8]<br/>
                  [4 -1 | 2]
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Step 1: R₂ - 2R₁ → R₂</div>
                  [2  3 |  8]<br/>
                  [0 -7 | -14]
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Step 2: R₂/(-7) → R₂</div>
                  [2  3 | 8]<br/>
                  [0  1 | 2]
                </div>
                <div className="text-green-400">
                  해: y = 2, x = 1
                </div>
              </div>
            </div>

            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold text-yellow-400 mb-1">목표</div>
                  <p className="text-slate-300 text-sm">
                    대각선 아래의 모든 원소를 0으로 만들기 (상삼각 행렬)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Row Echelon Form */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">행 사다리꼴과 기약 행 사다리꼴</h2>
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">행 사다리꼴 (Row Echelon Form - REF)</h3>
              <ul className="text-slate-300 text-sm space-y-2 mb-4">
                <li>• 영행이 아닌 각 행의 첫 번째 0이 아닌 원소는 1 (선도 원소, leading entry)</li>
                <li>• 아래 행의 선도 원소는 위 행의 선도 원소보다 오른쪽에 위치</li>
                <li>• 모든 영행은 맨 아래에 위치</li>
              </ul>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                [1  2  3]<br/>
                [0  1  4]<br/>
                [0  0  1]
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-green-400 mb-3">기약 행 사다리꼴 (Reduced Row Echelon Form - RREF)</h3>
              <ul className="text-slate-300 text-sm space-y-2 mb-4">
                <li>• 행 사다리꼴의 조건 만족</li>
                <li>• 선도 원소를 포함하는 열의 다른 모든 원소는 0</li>
              </ul>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                [1  0  0]<br/>
                [0  1  0]<br/>
                [0  0  1]
              </div>
            </div>
          </div>
        </section>

        {/* Solution Types */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">해의 유형</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">1. 유일해 (Unique Solution)</h3>
                <p className="text-slate-300 text-sm">선도 원소가 모든 변수에 대응하는 경우</p>
              </div>
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">2. 무한히 많은 해 (Infinite Solutions)</h3>
                <p className="text-slate-300 text-sm">자유 변수(free variable)가 존재하는 경우</p>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-red-400 mb-2">3. 해 없음 (No Solution)</h3>
                <p className="text-slate-300 text-sm">모순된 방정식(0 = k, k≠0)이 나타나는 경우</p>
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
            <ul className="space-y-2 text-slate-300 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>선형 시스템은 Ax = b로 표현</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>가우스 소거법: 기본 행 연산으로 해를 구함</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>행 사다리꼴: 상삼각 형태</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>해의 유형: 유일해, 무한해, 해 없음</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Practice */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Code className="w-6 h-6 text-purple-400" />
            연습 문제
          </h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="font-semibold text-purple-400 mb-2">다음 연립방정식을 가우스 소거법으로 풀어보세요</div>
            <div className="font-mono text-sm text-slate-300 mb-3">
              x + 2y = 5<br/>
              3x - y = 4
            </div>
            <details className="cursor-pointer">
              <summary className="text-blue-400 hover:text-blue-300 text-sm">풀이 보기</summary>
              <div className="mt-3 bg-slate-900/50 rounded-lg p-4 font-mono text-xs text-slate-300">
                [1  2 | 5]<br/>
                [3 -1 | 4]<br/><br/>
                R₂ - 3R₁ → R₂<br/>
                [1  2 |  5]<br/>
                [0 -7 | -11]<br/><br/>
                <span className="text-green-400">해: x = 1, y = 2</span>
              </div>
            </details>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link
            href="/modules/linear-algebra/matrices"
            className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 행렬과 행렬 연산</span>
          </Link>
          <Link
            href="/modules/linear-algebra/vector-spaces"
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <span>다음: 벡터 공간</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
