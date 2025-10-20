'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, AlertCircle, CheckCircle, Code } from 'lucide-react'

export default function Chapter2() {
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
            <h1 className="text-4xl font-bold">Chapter 2: 행렬과 행렬 연산</h1>
            <p className="text-slate-400 mt-2">Matrices and Matrix Operations</p>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Lightbulb className="w-6 h-6 text-yellow-400" />
            행렬이란?
          </h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 leading-relaxed mb-4">
              행렬(Matrix)은 숫자들을 직사각형 배열로 나타낸 것입니다. m개의 행(row)과 n개의 열(column)을 가진 행렬을
              m×n 행렬이라고 합니다.
            </p>

            <div className="bg-slate-900/50 rounded-lg p-6 mb-4">
              <div className="text-center font-mono text-sm mb-2">3×3 행렬 예시:</div>
              <div className="flex justify-center">
                <div className="font-mono text-lg bg-slate-800/50 p-4 rounded-lg">
                  <div>A = [&nbsp;1&nbsp;&nbsp;2&nbsp;&nbsp;3&nbsp;]</div>
                  <div>&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;4&nbsp;&nbsp;5&nbsp;&nbsp;6&nbsp;]</div>
                  <div>&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;7&nbsp;&nbsp;8&nbsp;&nbsp;9&nbsp;]</div>
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">행렬의 원소 표기</h3>
                <p className="text-slate-300 text-sm">
                  A의 i번째 행, j번째 열의 원소를 a<sub>ij</sub>로 표기합니다.
                </p>
                <div className="font-mono text-sm text-slate-400 mt-2">
                  예: a<sub>23</sub> = 6 (2행 3열)
                </div>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">행렬의 크기</h3>
                <p className="text-slate-300 text-sm">
                  m×n 행렬: m개의 행, n개의 열
                </p>
                <div className="font-mono text-sm text-slate-400 mt-2">
                  위 예시: 3×3 행렬 (정방행렬)
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Special Matrices */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">특수한 행렬</h2>

          <div className="space-y-6">
            {/* Square Matrix */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">1. 정방행렬 (Square Matrix)</h3>
              <p className="text-slate-300 mb-3">행의 개수와 열의 개수가 같은 행렬 (n×n)</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                [&nbsp;1&nbsp;&nbsp;2&nbsp;]<br/>
                [&nbsp;3&nbsp;&nbsp;4&nbsp;]&nbsp;&nbsp;&nbsp;← 2×2 정방행렬
              </div>
            </div>

            {/* Identity Matrix */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">2. 단위행렬 (Identity Matrix) - I</h3>
              <p className="text-slate-300 mb-3">대각선 원소가 모두 1이고 나머지는 0인 정방행렬</p>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-slate-900/50 rounded-lg p-4">
                  <div className="text-green-400 text-sm mb-2">2×2 단위행렬:</div>
                  <div className="font-mono text-sm">
                    I₂ = [&nbsp;1&nbsp;&nbsp;0&nbsp;]<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;0&nbsp;&nbsp;1&nbsp;]
                  </div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4">
                  <div className="text-green-400 text-sm mb-2">3×3 단위행렬:</div>
                  <div className="font-mono text-sm">
                    I₃ = [&nbsp;1&nbsp;&nbsp;0&nbsp;&nbsp;0&nbsp;]<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;0&nbsp;&nbsp;1&nbsp;&nbsp;0&nbsp;]<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;0&nbsp;&nbsp;0&nbsp;&nbsp;1&nbsp;]
                  </div>
                </div>
              </div>
              <div className="mt-3 text-slate-400 text-sm">
                성질: AI = IA = A (단위행렬과의 곱은 자기 자신)
              </div>
            </div>

            {/* Zero Matrix */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">3. 영행렬 (Zero Matrix) - O</h3>
              <p className="text-slate-300 mb-3">모든 원소가 0인 행렬</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                O = [&nbsp;0&nbsp;&nbsp;0&nbsp;&nbsp;0&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;0&nbsp;&nbsp;0&nbsp;&nbsp;0&nbsp;]
              </div>
            </div>

            {/* Diagonal Matrix */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">4. 대각행렬 (Diagonal Matrix)</h3>
              <p className="text-slate-300 mb-3">대각선 이외의 원소가 모두 0인 정방행렬</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                D = [&nbsp;2&nbsp;&nbsp;0&nbsp;&nbsp;0&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;0&nbsp;&nbsp;5&nbsp;&nbsp;0&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;0&nbsp;&nbsp;0&nbsp;&nbsp;3&nbsp;]
              </div>
            </div>

            {/* Transpose */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">5. 전치행렬 (Transpose Matrix) - A<sup>T</sup></h3>
              <p className="text-slate-300 mb-3">행과 열을 바꾼 행렬 (i행 j열 → j행 i열)</p>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-slate-900/50 rounded-lg p-4">
                  <div className="text-green-400 text-sm mb-2">원래 행렬 A:</div>
                  <div className="font-mono text-sm">
                    [&nbsp;1&nbsp;&nbsp;2&nbsp;&nbsp;3&nbsp;]<br/>
                    [&nbsp;4&nbsp;&nbsp;5&nbsp;&nbsp;6&nbsp;]
                  </div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4">
                  <div className="text-green-400 text-sm mb-2">전치행렬 A<sup>T</sup>:</div>
                  <div className="font-mono text-sm">
                    [&nbsp;1&nbsp;&nbsp;4&nbsp;]<br/>
                    [&nbsp;2&nbsp;&nbsp;5&nbsp;]<br/>
                    [&nbsp;3&nbsp;&nbsp;6&nbsp;]
                  </div>
                </div>
              </div>
            </div>

            {/* Symmetric Matrix */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">6. 대칭행렬 (Symmetric Matrix)</h3>
              <p className="text-slate-300 mb-3">A = A<sup>T</sup>인 정방행렬 (대각선을 기준으로 대칭)</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                S = [&nbsp;1&nbsp;&nbsp;2&nbsp;&nbsp;3&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;2&nbsp;&nbsp;4&nbsp;&nbsp;5&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;3&nbsp;&nbsp;5&nbsp;&nbsp;6&nbsp;]
              </div>
            </div>
          </div>
        </section>

        {/* Matrix Addition */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">행렬의 덧셈과 뺄셈</h2>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="mb-4">
              <div className="flex items-start gap-2 mb-3">
                <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <p className="text-slate-300">
                  <span className="text-yellow-400 font-semibold">조건:</span> 두 행렬의 크기가 같아야 합니다.
                </p>
              </div>
              <p className="text-slate-300">같은 위치의 원소끼리 더하거나 뺍니다.</p>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4 mb-4">
              <div className="font-mono text-sm mb-3">
                A = [&nbsp;1&nbsp;&nbsp;2&nbsp;]&nbsp;&nbsp;&nbsp;&nbsp;B = [&nbsp;5&nbsp;&nbsp;6&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;3&nbsp;&nbsp;4&nbsp;]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;7&nbsp;&nbsp;8&nbsp;]
              </div>
              <div className="font-mono text-sm text-green-400">
                A + B = [&nbsp;1+5&nbsp;&nbsp;2+6&nbsp;] = [&nbsp;6&nbsp;&nbsp;&nbsp;8&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;3+7&nbsp;&nbsp;4+8&nbsp;]&nbsp;&nbsp;&nbsp;[&nbsp;10&nbsp;&nbsp;12&nbsp;]
              </div>
            </div>

            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <div className="font-semibold text-blue-400 mb-2">행렬 덧셈의 성질</div>
              <ul className="text-slate-300 text-sm space-y-1">
                <li>• 교환법칙: A + B = B + A</li>
                <li>• 결합법칙: (A + B) + C = A + (B + C)</li>
                <li>• 영행렬: A + O = A</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Scalar Multiplication */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">스칼라 곱셈</h2>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">
              행렬의 각 원소에 스칼라(실수)를 곱합니다.
            </p>

            <div className="bg-slate-900/50 rounded-lg p-4 mb-4">
              <div className="font-mono text-sm mb-3">
                A = [&nbsp;1&nbsp;&nbsp;2&nbsp;]&nbsp;&nbsp;&nbsp;&nbsp;k = 3<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;3&nbsp;&nbsp;4&nbsp;]
              </div>
              <div className="font-mono text-sm text-green-400">
                3A = [&nbsp;3×1&nbsp;&nbsp;3×2&nbsp;] = [&nbsp;3&nbsp;&nbsp;&nbsp;6&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;3×3&nbsp;&nbsp;3×4&nbsp;]&nbsp;&nbsp;&nbsp;[&nbsp;9&nbsp;&nbsp;12&nbsp;]
              </div>
            </div>

            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <div className="font-semibold text-blue-400 mb-2">스칼라 곱셈의 성질</div>
              <ul className="text-slate-300 text-sm space-y-1">
                <li>• k(A + B) = kA + kB</li>
                <li>• (k + m)A = kA + mA</li>
                <li>• k(mA) = (km)A</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Matrix Multiplication */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">행렬의 곱셈</h2>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="mb-4">
              <div className="flex items-start gap-2 mb-3">
                <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <p className="text-slate-300">
                  <span className="text-yellow-400 font-semibold">조건:</span> A가 m×n 행렬일 때, B는 n×p 행렬이어야 합니다.
                  <br/>(A의 열 개수 = B의 행 개수)
                </p>
              </div>
              <p className="text-slate-300">결과는 m×p 행렬이 됩니다.</p>
            </div>

            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 mb-4">
              <div className="font-semibold text-purple-400 mb-2">곱셈 규칙</div>
              <p className="text-slate-300 text-sm mb-2">
                AB의 (i, j) 원소 = A의 i번째 행과 B의 j번째 열의 내적
              </p>
              <div className="font-mono text-xs text-slate-400">
                (AB)<sub>ij</sub> = a<sub>i1</sub>b<sub>1j</sub> + a<sub>i2</sub>b<sub>2j</sub> + ... + a<sub>in</sub>b<sub>nj</sub>
              </div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-6 mb-4">
              <div className="text-green-400 text-sm mb-3">예시: 2×3 행렬 × 3×2 행렬 = 2×2 행렬</div>
              <div className="font-mono text-sm mb-3">
                A = [&nbsp;1&nbsp;&nbsp;2&nbsp;&nbsp;3&nbsp;]&nbsp;&nbsp;&nbsp;&nbsp;B = [&nbsp;7&nbsp;&nbsp;&nbsp;8&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;4&nbsp;&nbsp;5&nbsp;&nbsp;6&nbsp;]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;9&nbsp;&nbsp;10&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[11&nbsp;&nbsp;12&nbsp;]
              </div>
              <div className="font-mono text-xs text-slate-300 mb-2">
                (AB)<sub>11</sub> = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58<br/>
                (AB)<sub>12</sub> = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64<br/>
                (AB)<sub>21</sub> = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139<br/>
                (AB)<sub>22</sub> = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154
              </div>
              <div className="font-mono text-sm text-green-400">
                AB = [&nbsp;58&nbsp;&nbsp;&nbsp;64&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[139&nbsp;&nbsp;154&nbsp;]
              </div>
            </div>

            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold text-yellow-400 mb-2">중요한 성질</div>
                  <ul className="text-slate-300 text-sm space-y-1">
                    <li>• 결합법칙: (AB)C = A(BC)</li>
                    <li>• 분배법칙: A(B + C) = AB + AC</li>
                    <li>• <span className="text-yellow-400">교환법칙 성립 안 함:</span> AB ≠ BA (일반적으로)</li>
                    <li>• 단위행렬: AI = IA = A</li>
                    <li>• 전치: (AB)<sup>T</sup> = B<sup>T</sup>A<sup>T</sup></li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Matrix Inverse */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">역행렬 (Inverse Matrix)</h2>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">
              정방행렬 A에 대해 AB = BA = I를 만족하는 행렬 B를 A의 역행렬이라 하고, A<sup>-1</sup>로 표기합니다.
            </p>

            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 mb-4">
              <div className="font-semibold text-purple-400 mb-2">역행렬의 조건</div>
              <p className="text-slate-300 text-sm">
                역행렬이 존재하려면 행렬의 행렬식(determinant)이 0이 아니어야 합니다.
              </p>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-6 mb-4">
              <div className="text-green-400 text-sm mb-3">2×2 행렬의 역행렬 공식:</div>
              <div className="font-mono text-sm mb-3">
                A = [&nbsp;a&nbsp;&nbsp;b&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;c&nbsp;&nbsp;d&nbsp;]
              </div>
              <div className="font-mono text-sm text-green-400 mb-2">
                A<sup>-1</sup> = 1/(ad-bc) × [&nbsp;d&nbsp;&nbsp;-b&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-c&nbsp;&nbsp;&nbsp;a&nbsp;]
              </div>
              <p className="text-slate-400 text-xs">단, ad - bc ≠ 0</p>
            </div>

            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-4">
              <div className="font-semibold text-blue-400 mb-2">예시:</div>
              <div className="font-mono text-sm text-slate-300">
                A = [&nbsp;2&nbsp;&nbsp;1&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;1&nbsp;&nbsp;1&nbsp;]<br/><br/>
                det(A) = 2×1 - 1×1 = 1 ≠ 0 ✓<br/><br/>
                A<sup>-1</sup> = 1/1 × [&nbsp;1&nbsp;&nbsp;-1&nbsp;] = [&nbsp;1&nbsp;&nbsp;-1&nbsp;]<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-1&nbsp;&nbsp;&nbsp;2&nbsp;]&nbsp;&nbsp;&nbsp;[-1&nbsp;&nbsp;&nbsp;2&nbsp;]
              </div>
            </div>

            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <div className="font-semibold text-yellow-400 mb-2">역행렬의 성질</div>
              <ul className="text-slate-300 text-sm space-y-1">
                <li>• (A<sup>-1</sup>)<sup>-1</sup> = A</li>
                <li>• (AB)<sup>-1</sup> = B<sup>-1</sup>A<sup>-1</sup></li>
                <li>• (A<sup>T</sup>)<sup>-1</sup> = (A<sup>-1</sup>)<sup>T</sup></li>
                <li>• (kA)<sup>-1</sup> = (1/k)A<sup>-1</sup>, k ≠ 0</li>
              </ul>
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
                <h3 className="font-semibold text-green-400 mb-3">기본 개념</h3>
                <ul className="space-y-2 text-slate-300 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>행렬: 숫자를 직사각형 배열로 표현</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>단위행렬: 대각선이 1, 나머지 0</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>전치행렬: 행과 열을 바꿈</span>
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-blue-400 mb-3">연산</h3>
                <ul className="space-y-2 text-slate-300 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                    <span>덧셈: 같은 크기 행렬만 가능</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                    <span>곱셈: A의 열 = B의 행 개수</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                    <span>역행렬: AA<sup>-1</sup> = I</span>
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
              <div className="font-semibold text-purple-400 mb-2">문제 1: 행렬 곱셈</div>
              <p className="text-slate-300 text-sm mb-3">
                다음 두 행렬의 곱 AB를 구하시오.
              </p>
              <div className="font-mono text-sm text-slate-300 mb-3">
                A = [1  2]    B = [5  6]<br/>
                    [3  4]        [7  8]
              </div>
              <details className="cursor-pointer">
                <summary className="text-blue-400 hover:text-blue-300 text-sm">풀이 보기</summary>
                <div className="mt-3 bg-slate-900/50 rounded-lg p-4 font-mono text-xs text-slate-300">
                  AB<sub>11</sub> = 1×5 + 2×7 = 5 + 14 = 19<br/>
                  AB<sub>12</sub> = 1×6 + 2×8 = 6 + 16 = 22<br/>
                  AB<sub>21</sub> = 3×5 + 4×7 = 15 + 28 = 43<br/>
                  AB<sub>22</sub> = 3×6 + 4×8 = 18 + 32 = 50<br/><br/>
                  <span className="text-green-400">
                  AB = [19  22]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[43  50]
                  </span>
                </div>
              </details>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <div className="font-semibold text-purple-400 mb-2">문제 2: 역행렬</div>
              <p className="text-slate-300 text-sm mb-3">
                다음 행렬의 역행렬을 구하시오.
              </p>
              <div className="font-mono text-sm text-slate-300 mb-3">
                A = [3  1]<br/>
                    [2  1]
              </div>
              <details className="cursor-pointer">
                <summary className="text-blue-400 hover:text-blue-300 text-sm">풀이 보기</summary>
                <div className="mt-3 bg-slate-900/50 rounded-lg p-4 font-mono text-xs text-slate-300">
                  det(A) = 3×1 - 1×2 = 1<br/><br/>
                  <span className="text-green-400">
                  A<sup>-1</sup> = 1/1 × [1  -1] = [&nbsp;1  -1]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-2  3]&nbsp;&nbsp;&nbsp;[-2&nbsp;&nbsp;3]
                  </span>
                </div>
              </details>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link
            href="/modules/linear-algebra/vectors-basics"
            className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 벡터의 기초</span>
          </Link>
          <Link
            href="/modules/linear-algebra/linear-systems"
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <span>다음: 선형 시스템</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
