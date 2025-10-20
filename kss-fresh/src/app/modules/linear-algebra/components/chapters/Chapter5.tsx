'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, CheckCircle } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <Link href="/modules/linear-algebra" className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700 mb-8">
          <ArrowLeft className="w-4 h-4" />
          <span className="text-sm">모듈로 돌아가기</span>
        </Link>

        <div className="flex items-center gap-3 mb-8">
          <BookOpen className="w-8 h-8 text-blue-400" />
          <div>
            <h1 className="text-4xl font-bold">Chapter 5: 고유값과 고유벡터</h1>
            <p className="text-slate-400 mt-2">Eigenvalues and Eigenvectors</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Lightbulb className="w-6 h-6 text-yellow-400" />
            고유값과 고유벡터란?
          </h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">정방행렬 A에 대해 Av = λv를 만족하는 0이 아닌 벡터 v를 고유벡터, λ를 고유값이라 합니다.</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
              Av = λv<br/>
              (A - λI)v = 0
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">특성 방정식 (Characteristic Equation)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">고유값을 구하기 위한 방정식: det(A - λI) = 0</p>
            <div className="bg-slate-900/50 rounded-lg p-6">
              <div className="text-green-400 text-sm mb-3">예시: 2×2 행렬</div>
              <div className="font-mono text-sm space-y-2">
                <div>A = [3  1]</div>
                <div>&nbsp;&nbsp;&nbsp;&nbsp;[1  3]</div>
                <div className="mt-3">det(A - λI) = (3-λ)² - 1 = 0</div>
                <div>λ² - 6λ + 8 = 0</div>
                <div className="text-green-400">(λ-4)(λ-2) = 0</div>
                <div className="text-green-400">고유값: λ₁ = 4, λ₂ = 2</div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">대각화 (Diagonalization)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">A = PDP⁻¹ 형태로 표현 (P: 고유벡터 행렬, D: 대각 행렬)</p>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <h3 className="font-semibold text-blue-400 mb-2">대각화 가능 조건</h3>
              <ul className="text-slate-300 text-sm space-y-1">
                <li>• n×n 행렬이 n개의 선형 독립인 고유벡터를 가짐</li>
                <li>• 모든 고유값이 서로 다르면 대각화 가능</li>
              </ul>
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
                <span>Av = λv: 고유벡터 v, 고유값 λ</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>특성 방정식: det(A - λI) = 0</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>대각화: A = PDP⁻¹</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/linear-algebra/vector-spaces" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 벡터 공간</span>
          </Link>
          <Link href="/modules/linear-algebra/orthogonality" className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
            <span>다음: 직교성</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
