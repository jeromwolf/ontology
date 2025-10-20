'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'

export default function Chapter3() {
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
            <h1 className="text-4xl font-bold">Chapter 3: 미분의 응용</h1>
            <p className="text-slate-400 mt-2">Applications of Derivatives</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">극값 (Extrema)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">극댓값 (Local Maximum)</h3>
                <p className="text-slate-300 text-sm">f'(c) = 0이고 f'가 c에서 양에서 음으로 변함</p>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">극솟값 (Local Minimum)</h3>
                <p className="text-slate-300 text-sm">f'(c) = 0이고 f'가 c에서 음에서 양으로 변함</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">1차 도함수 판정법</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <ul className="space-y-3 text-slate-300">
              <li>• f'(x) {">"} 0 ⟹ f는 증가</li>
              <li>• f'(x) {"<"} 0 ⟹ f는 감소</li>
              <li>• f'(c) = 0 ⟹ c는 임계점 (critical point)</li>
            </ul>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">2차 도함수 판정법</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">f'(c) = 0일 때:</p>
            <div className="space-y-3">
              <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <p>f''(c) {">"} 0 ⟹ c에서 극솟값 (볼록)</p>
              </div>
              <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                <p>f''(c) {"<"} 0 ⟹ c에서 극댓값 (오목)</p>
              </div>
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <p>f''(c) = 0 ⟹ 판정 불가 (변곡점 가능)</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">최적화 문제</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">최댓값 또는 최솟값을 구하는 단계:</p>
            <div className="space-y-3">
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                  <span className="text-blue-400 font-bold">1</span>
                </div>
                <div>
                  <p className="font-semibold">함수 설정</p>
                  <p className="text-sm text-slate-400">최적화할 함수 f(x) 정의</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                  <span className="text-green-400 font-bold">2</span>
                </div>
                <div>
                  <p className="font-semibold">도함수 계산</p>
                  <p className="text-sm text-slate-400">f'(x)를 구하고 0이 되는 점 찾기</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                  <span className="text-purple-400 font-bold">3</span>
                </div>
                <div>
                  <p className="font-semibold">극값 판정</p>
                  <p className="text-sm text-slate-400">2차 도함수 또는 1차 도함수 판정법 사용</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">로피탈의 정리 (L'Hospital's Rule)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">0/0 또는 ∞/∞ 꼴의 극한:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center mb-4">
              lim<sub>x→a</sub> f(x)/g(x) = lim<sub>x→a</sub> f'(x)/g'(x)
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <p className="font-semibold text-blue-400 mb-2">예시:</p>
              <div className="font-mono text-sm space-y-2">
                <div>lim<sub>x→0</sub> sin(x)/x</div>
                <div>= lim<sub>x→0</sub> cos(x)/1</div>
                <div className="text-green-400">= 1</div>
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
                <span>극값: f'(c) = 0인 점에서 발생 가능</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>증가/감소: f'(x)의 부호로 판정</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>볼록/오목: f''(x)의 부호로 판정</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>로피탈: 0/0, ∞/∞ 꼴에 적용</span>
              </li>
            </ul>
          </div>
        </section>

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/calculus/derivatives" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 미분법</span>
          </Link>
          <Link href="/modules/calculus/integration" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <span>다음: 적분법</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
