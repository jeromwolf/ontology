'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, CheckCircle, Code } from 'lucide-react'

export default function Chapter4() {
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
            <h1 className="text-4xl font-bold">Chapter 4: 벡터 공간과 부분 공간</h1>
            <p className="text-slate-400 mt-2">Vector Spaces and Subspaces</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Lightbulb className="w-6 h-6 text-yellow-400" />
            벡터 공간이란?
          </h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">벡터 공간(Vector Space)은 벡터의 덧셈과 스칼라 곱셈이 정의된 집합입니다.</p>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <h3 className="font-semibold text-blue-400 mb-2">8가지 공리 (Axioms)</h3>
              <ul className="text-slate-300 text-sm space-y-1">
                <li>• 덧셈 교환법칙: u + v = v + u</li>
                <li>• 덧셈 결합법칙: (u + v) + w = u + (v + w)</li>
                <li>• 영벡터 존재: v + 0 = v</li>
                <li>• 역벡터 존재: v + (-v) = 0</li>
                <li>• 스칼라 곱 결합: k(cv) = (kc)v</li>
                <li>• 단위원소: 1v = v</li>
                <li>• 분배법칙 1: k(u + v) = ku + kv</li>
                <li>• 분배법칙 2: (k + c)v = kv + cv</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">부분 공간 (Subspace)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">벡터 공간의 부분집합이 다음 3가지 조건을 만족하면 부분 공간입니다.</p>
            <div className="space-y-3">
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">1. 영벡터 포함</h3>
                <p className="text-slate-300 text-sm">0 ∈ H</p>
              </div>
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">2. 덧셈에 닫혀있음</h3>
                <p className="text-slate-300 text-sm">u, v ∈ H ⟹ u + v ∈ H</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-purple-400 mb-2">3. 스칼라 곱에 닫혀있음</h3>
                <p className="text-slate-300 text-sm">v ∈ H, k ∈ ℝ ⟹ kv ∈ H</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">선형 독립과 선형 종속</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="mb-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">선형 독립 (Linearly Independent)</h3>
              <p className="text-slate-300 mb-3">c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 이 c₁ = c₂ = ... = cₙ = 0 일 때만 성립</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                예: v₁ = (1, 0), v₂ = (0, 1)는 선형 독립
              </div>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-red-400 mb-3">선형 종속 (Linearly Dependent)</h3>
              <p className="text-slate-300 mb-3">0이 아닌 계수로 영벡터를 만들 수 있음</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                예: v₁ = (1, 2), v₂ = (2, 4)는 선형 종속 (v₂ = 2v₁)
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">기저와 차원</h2>
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-green-400 mb-3">기저 (Basis)</h3>
              <p className="text-slate-300 mb-3">벡터 공간을 생성하는 선형 독립인 벡터들의 집합</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                ℝ³의 표준 기저: e₁ = (1,0,0), e₂ = (0,1,0), e₃ = (0,0,1)
              </div>
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-purple-400 mb-3">차원 (Dimension)</h3>
              <p className="text-slate-300 mb-3">기저를 이루는 벡터의 개수</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                dim(ℝ²) = 2, dim(ℝ³) = 3
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
                <span>벡터 공간: 8가지 공리를 만족하는 집합</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>부분 공간: 영벡터 포함 + 덧셈/스칼라 곱에 닫혀있음</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>선형 독립: 자명한 해만 존재</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>기저: 공간을 생성하는 선형 독립 벡터들</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/linear-algebra/linear-systems" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 선형 시스템</span>
          </Link>
          <Link href="/modules/linear-algebra/eigenvalues" className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
            <span>다음: 고유값과 고유벡터</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
