'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter8() {
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
            <h1 className="text-4xl font-bold">Chapter 8: 벡터 미적분</h1>
            <p className="text-slate-400 mt-2">Vector Calculus</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">벡터장 (Vector Field)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">공간의 각 점에 벡터를 대응시키는 함수:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center">
              F(x, y) = P(x,y)i + Q(x,y)j
            </div>
            <div className="mt-4 text-slate-400 text-sm">
              예: 중력장, 전기장, 유체의 속도장
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">선적분 (Line Integral)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">스칼라장의 선적분</h3>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                  ∫<sub>C</sub> f(x,y) ds
                </div>
                <p className="text-slate-400 text-sm mt-2">곡선을 따라 스칼라 함수를 적분</p>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-green-400 mb-3">벡터장의 선적분</h3>
                <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                  ∫<sub>C</sub> F · dr = ∫<sub>C</sub> (P dx + Q dy)
                </div>
                <p className="text-slate-400 text-sm mt-2">일 (Work) 계산에 사용</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">발산과 회전</h2>
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">발산 (Divergence)</h3>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm mb-3">
                div F = ∇ · F = ∂P/∂x + ∂Q/∂y + ∂R/∂z
              </div>
              <p className="text-slate-300 text-sm">벡터장의 "확산" 정도</p>
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-green-400 mb-3">회전 (Curl)</h3>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm mb-3">
                curl F = ∇ × F = (∂R/∂y - ∂Q/∂z)i + ...
              </div>
              <p className="text-slate-300 text-sm">벡터장의 "회전" 정도</p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">그린의 정리 (Green's Theorem)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">평면 영역 D의 경계 C에서:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center mb-4">
              ∮<sub>C</sub> (P dx + Q dy) = ∬<sub>D</sub> (∂Q/∂x - ∂P/∂y) dA
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <p className="text-slate-300 text-sm">선적분과 이중적분을 연결하는 정리</p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">스토크스 정리 (Stokes' Theorem)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">곡면 S의 경계 C에서:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center mb-4">
              ∮<sub>C</sub> F · dr = ∬<sub>S</sub> (curl F) · dS
            </div>
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
              <p className="text-slate-300 text-sm">그린 정리의 3차원 확장</p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">발산 정리 (Divergence Theorem)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">입체 E의 경계면 S에서:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm text-center mb-4">
              ∬<sub>S</sub> F · dS = ∭<sub>E</sub> div F dV
            </div>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="text-slate-300 text-sm">면적분과 삼중적분을 연결 (가우스 정리)</p>
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
                <span>벡터장: 공간의 각 점에 벡터 대응</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>선적분: 곡선을 따라 적분</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>그린: 선적분 ↔ 이중적분</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>스토크스: 선적분 ↔ 면적분</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>발산: 면적분 ↔ 삼중적분</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/calculus/multivariable" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 다변수 미적분</span>
          </Link>
          <Link href="/modules/calculus" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <CheckCircle className="w-4 h-4" />
            <span>모듈 완료</span>
          </Link>
        </div>
      </div>
    </div>
  )
}
