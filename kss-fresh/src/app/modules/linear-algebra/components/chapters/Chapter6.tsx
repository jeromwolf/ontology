'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter6() {
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
            <h1 className="text-4xl font-bold">Chapter 6: 직교성과 정규직교기저</h1>
            <p className="text-slate-400 mt-2">Orthogonality and Orthonormal Basis</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">직교 벡터 (Orthogonal Vectors)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">두 벡터 u, v가 직교하면 u·v = 0</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
              u = (1, 2), v = (2, -1)<br/>
              u·v = 1×2 + 2×(-1) = 0  ✓ 직교
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">정규직교기저 (Orthonormal Basis)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">서로 직교하고 크기가 1인 벡터들로 이루어진 기저</p>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <h3 className="font-semibold text-blue-400 mb-2">조건</h3>
              <ul className="text-slate-300 text-sm space-y-1">
                <li>• u<sub>i</sub>·u<sub>j</sub> = 0 (i ≠ j): 서로 직교</li>
                <li>• ||u<sub>i</sub>|| = 1: 크기가 1 (정규화)</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">그람-슈미트 정규직교화</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">임의의 기저를 정규직교기저로 변환하는 알고리즘</p>
            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="font-mono text-sm space-y-2">
                <div>1. v₁ 선택</div>
                <div>2. v₂에서 v₁ 성분 제거: v₂′ = v₂ - (v₂·v₁)v₁</div>
                <div>3. 정규화: u = v/||v||</div>
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
                <span>직교: u·v = 0</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>정규직교기저: 직교 + 크기 1</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>그람-슈미트: 기저를 정규직교기저로 변환</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/linear-algebra/eigenvalues" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 고유값</span>
          </Link>
          <Link href="/modules/linear-algebra/linear-transformations" className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
            <span>다음: 선형 변환</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
