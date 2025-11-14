'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, CheckCircle } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter8() {
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
            <h1 className="text-4xl font-bold">Chapter 8: 특이값 분해 (SVD)</h1>
            <p className="text-slate-400 mt-2">Singular Value Decomposition</p>
          </div>
        </div>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Lightbulb className="w-6 h-6 text-yellow-400" />
            SVD란?
          </h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">모든 m×n 행렬 A는 다음과 같이 분해할 수 있습니다:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-lg text-center">
              A = UΣV<sup>T</sup>
            </div>
            <div className="mt-4 space-y-2 text-slate-300 text-sm">
              <div>• U: m×m 직교 행렬 (left singular vectors)</div>
              <div>• Σ: m×n 대각 행렬 (singular values)</div>
              <div>• V: n×n 직교 행렬 (right singular vectors)</div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">특이값 (Singular Values)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">Σ의 대각 원소들을 특이값이라 하며, 내림차순으로 정렬됩니다.</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
              Σ = [σ₁  0   0 ]<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;[0   σ₂  0 ]<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;[0   0   σ₃]<br/><br/>
              σ₁ ≥ σ₂ ≥ σ₃ ≥ 0
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">SVD 계산 방법</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-4">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-blue-400 mb-2">Step 1: A<sup>T</sup>A의 고유값과 고유벡터 구하기</h3>
                <p className="text-slate-300 text-sm">A<sup>T</sup>A의 고유값의 제곱근이 특이값, 고유벡터가 V의 열</p>
              </div>
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-green-400 mb-2">Step 2: AA<sup>T</sup>의 고유벡터 구하기</h3>
                <p className="text-slate-300 text-sm">AA<sup>T</sup>의 고유벡터가 U의 열</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <h3 className="font-semibold text-purple-400 mb-2">Step 3: Σ 구성하기</h3>
                <p className="text-slate-300 text-sm">특이값을 대각선에 배치</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">SVD 응용</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-blue-400 font-semibold">1</span>
                </div>
                <div>
                  <h3 className="font-semibold text-blue-400 mb-1">데이터 압축</h3>
                  <p className="text-slate-300 text-sm">작은 특이값 제거로 근사 행렬 생성</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-green-400 font-semibold">2</span>
                </div>
                <div>
                  <h3 className="font-semibold text-green-400 mb-1">추천 시스템</h3>
                  <p className="text-slate-300 text-sm">협업 필터링에서 잠재 요인 추출</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-purple-400 font-semibold">3</span>
                </div>
                <div>
                  <h3 className="font-semibold text-purple-400 mb-1">주성분 분석 (PCA)</h3>
                  <p className="text-slate-300 text-sm">차원 축소 및 데이터 시각화</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-yellow-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-yellow-400 font-semibold">4</span>
                </div>
                <div>
                  <h3 className="font-semibold text-yellow-400 mb-1">이미지 압축</h3>
                  <p className="text-slate-300 text-sm">이미지 행렬을 저계수 근사로 압축</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4">저계수 근사 (Low-Rank Approximation)</h2>
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <p className="text-slate-300 mb-4">상위 k개 특이값만 사용하여 근사 행렬 생성:</p>
            <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-sm">
              A ≈ A<sub>k</sub> = σ₁u₁v₁<sup>T</sup> + σ₂u₂v₂<sup>T</sup> + ... + σₖuₖvₖ<sup>T</sup>
            </div>
            <div className="mt-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <p className="text-slate-300 text-sm">
                이는 Frobenius norm 의미에서 최적의 계수-k 근사입니다.
              </p>
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
                <span>SVD: A = UΣV<sup>T</sup> (모든 행렬에 적용 가능)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>특이값: Σ의 대각 원소 (내림차순 정렬)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>응용: 데이터 압축, 추천 시스템, PCA, 이미지 처리</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                <span>저계수 근사: 상위 k개 특이값으로 행렬 근사</span>
              </li>
            </ul>
          </div>
        </section>

        <References
          sections={[
            {
              title: '📚 온라인 강의 & 교재',
              icon: 'web' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'MIT OCW 18.06 - Linear Algebra (Gilbert Strang)',
                  url: 'https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/',
                  description: '전설적인 MIT 선형대수학 강의 - SVD 완벽 설명 포함 (2024)',
                  year: 2024
                },
                {
                  title: '3Blue1Brown - Essence of Linear Algebra',
                  url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab',
                  description: '시각적 선형대수학 강의 - 직관적 이해를 위한 최고의 애니메이션 시리즈 (2024)',
                  year: 2024
                },
                {
                  title: 'Khan Academy Linear Algebra',
                  url: 'https://www.khanacademy.org/math/linear-algebra',
                  description: '무료 선형대수학 강의 - 벡터, 행렬, 고유값부터 SVD까지 완전 커버 (2024)',
                  year: 2024
                },
                {
                  title: 'Linear Algebra (Jim Hefferon) - Free Textbook',
                  url: 'http://joshua.smcvt.edu/linearalgebra/',
                  description: '무료 공개 교재 - 체계적인 이론 설명과 풍부한 연습문제 (2024)',
                  year: 2024
                }
              ]
            },
            {
              title: '📖 핵심 교재',
              icon: 'research' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'Linear Algebra Done Right (Sheldon Axler)',
                  url: 'https://linear.axler.net/',
                  description: '직관적 접근의 명저 - 행렬식 없이 선형대수 완벽 이해 (4판, 2024)',
                  year: 2024
                },
                {
                  title: 'Introduction to Linear Algebra (Gilbert Strang)',
                  url: 'https://math.mit.edu/~gs/linearalgebra/',
                  description: 'MIT 표준 교재 - 응용 중심 명쾌한 설명 (6판, 2023)',
                  year: 2023
                },
                {
                  title: 'Linear Algebra (Hoffman & Kunze)',
                  url: 'https://www.pearson.com/en-us/subject-catalog/p/linear-algebra/P200000006210',
                  description: '엄밀한 이론 중심 고급 교재 - 수학 전공자 필독서 (2판)',
                  year: 1971
                },
                {
                  title: 'Matrix Analysis (Horn & Johnson)',
                  url: 'https://www.cambridge.org/core/books/matrix-analysis/C7BE870A0C3FF8E0D0F8A8C8C8E8E8E8',
                  description: '행렬 이론 전문서 - SVD 및 고급 응용 완벽 다룸 (2판, 2012)',
                  year: 2012
                }
              ]
            },
            {
              title: '🛠️ 실전 도구',
              icon: 'tools' as const,
              color: 'border-green-500',
              items: [
                {
                  title: 'NumPy Linear Algebra',
                  url: 'https://numpy.org/doc/stable/reference/routines.linalg.html',
                  description: 'Python 선형대수 라이브러리 - SVD, 고유값, 행렬 연산 (2024)',
                  year: 2024
                },
                {
                  title: 'SciPy linalg',
                  url: 'https://docs.scipy.org/doc/scipy/reference/linalg.html',
                  description: '과학 계산용 선형대수 - 고급 행렬 분해 및 최적화 (2024)',
                  year: 2024
                },
                {
                  title: 'MATLAB',
                  url: 'https://www.mathworks.com/help/matlab/linear-algebra.html',
                  description: '산업 표준 수치 계산 소프트웨어 - 선형대수 전용 함수 풍부 (2024)',
                  year: 2024
                },
                {
                  title: 'Julia LinearAlgebra',
                  url: 'https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/',
                  description: '고성능 과학 계산 언어 - 빠른 행렬 연산 및 SVD (2024)',
                  year: 2024
                },
                {
                  title: 'SymPy',
                  url: 'https://docs.sympy.org/latest/modules/matrices/matrices.html',
                  description: 'Python 기호 수학 라이브러리 - 정확한 행렬 계산 (2024)',
                  year: 2024
                }
              ]
            }
          ]}
        />

        <div className="flex justify-between items-center pt-8 border-t border-slate-700">
          <Link href="/modules/linear-algebra/linear-transformations" className="flex items-center gap-2 px-6 py-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: 선형 변환</span>
          </Link>
          <Link href="/modules/linear-algebra" className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
            <CheckCircle className="w-4 h-4" />
            <span>모듈 완료</span>
          </Link>
        </div>
      </div>
    </div>
  )
}
