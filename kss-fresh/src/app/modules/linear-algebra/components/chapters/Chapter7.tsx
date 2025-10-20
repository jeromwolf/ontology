'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter7() {
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
            <h1 className="text-4xl font-bold">Chapter 7: 선형 변환</h1>
            <p className="text-slate-400 mt-2">Linear Transformations</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">선형 변환이란?</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">T: V → W가 다음 두 조건을 만족하면 선형 변환입니다.</p>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <ul className="text-slate-300 text-sm space-y-2">
                <li>• T(u + v) = T(u) + T(v) (가법성)</li>
                <li>• T(cu) = cT(u) (동차성)</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">행렬 변환</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">모든 선형 변환은 행렬로 표현할 수 있습니다: T(x) = Ax</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
              A = [2  0]<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;[0  3]<br/><br/>
              T([1]) = [2]<br/>
              &nbsp;&nbsp;([0])&nbsp;&nbsp;&nbsp;([0])<br/><br/>
              → x축 2배, y축 3배로 스케일링
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">핵과 상 (Kernel and Image)</h2>
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">핵 (Kernel / Null Space)</h3>
              <p className="text-slate-300 mb-3">T(v) = 0을 만족하는 모든 벡터 v의 집합</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                Ker(T) = {`{v ∈ V | T(v) = 0}`}
              </div>
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-green-400 mb-3">상 (Image / Range)</h3>
              <p className="text-slate-300 mb-3">T로 변환할 수 있는 모든 벡터의 집합</p>
              <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
                Im(T) = {`{w ∈ W | w = T(v) for some v ∈ V}`}
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">계수-무효성 정리</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">dim(V) = dim(Ker(T)) + dim(Im(T))</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="text-slate-300 text-sm">정의역의 차원 = 핵의 차원 + 상의 차원</p>
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
                <span>선형 변환: T(u+v) = T(u)+T(v), T(cu) = cT(u)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>행렬 표현: T(x) = Ax</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>계수-무효성: dim(V) = dim(Ker) + dim(Im)</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/linear-algebra/orthogonality" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 직교성</span>
          </Link>
          <Link href="/modules/linear-algebra/svd" className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
            <span>다음: SVD</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
