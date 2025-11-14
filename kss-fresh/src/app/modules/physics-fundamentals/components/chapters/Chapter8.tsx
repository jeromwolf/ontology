'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, CheckCircle } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter8() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="border-b border-purple-500/20 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <Link href="/modules/physics-fundamentals" className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors mb-3">
            <ArrowLeft className="w-4 h-4" />
            <span>Physics Fundamentals 모듈로 돌아가기</span>
          </Link>
          <h1 className="text-3xl font-bold mb-2">Chapter 8: 열역학</h1>
          <p className="text-slate-300">열역학 법칙, 엔트로피, 열기관</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">1. 온도와 열</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">온도의 정의</h3>
            <p className="text-slate-300 mb-4">물질의 평균 운동 에너지를 나타내는 척도</p>
            <div className="space-y-2 text-sm">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                <p className="font-mono">K = °C + 273.15 (켈빈 온도)</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                <p className="font-mono">°F = (9/5)°C + 32 (화씨 온도)</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                <p className="font-mono">T = (2/3)(KE_avg/k_B)</p>
                <p className="text-xs text-slate-400">k_B = 1.38 × 10⁻²³ J/K (볼츠만 상수)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">열의 전달 방식</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">전도 (Conduction)</h4>
                <p className="text-sm text-slate-300">직접 접촉으로 열 전달<br/>Q = kAΔT/L</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">대류 (Convection)</h4>
                <p className="text-sm text-slate-300">유체의 이동으로 열 전달<br/>공기, 물</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">복사 (Radiation)</h4>
                <p className="text-sm text-slate-300">전자기파로 열 전달<br/>P = σAT⁴</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">2. 열역학 제0법칙</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">열평형</h3>
            <p className="text-slate-300 mb-4">A와 B가 평형이고, B와 C가 평형이면, A와 C도 평형이다</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="text-sm text-slate-300">온도의 개념과 온도계의 이론적 근거</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">3. 열역학 제1법칙 (에너지 보존)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">제1법칙</h3>
            <p className="text-slate-300 mb-4">에너지는 생성되거나 소멸되지 않고, 형태만 변환된다</p>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">ΔU = Q - W</p>
                <div className="text-xs text-slate-400 mt-2 space-y-1">
                  <p>• ΔU: 내부 에너지 변화</p>
                  <p>• Q: 흡수한 열 (양수: 흡수, 음수: 방출)</p>
                  <p>• W: 한 일 (양수: 계가 외부에, 음수: 외부가 계에)</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">특수한 과정</h3>
            <div className="space-y-2 text-sm">
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="font-semibold text-purple-300">등적 과정 (Isochoric): ΔV = 0, W = 0, ΔU = Q</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="font-semibold text-purple-300">등압 과정 (Isobaric): ΔP = 0, W = PΔV</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="font-semibold text-purple-300">등온 과정 (Isothermal): ΔT = 0, ΔU = 0, Q = W</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg">
                <p className="font-semibold text-purple-300">단열 과정 (Adiabatic): Q = 0, ΔU = -W</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">4. 열역학 제2법칙 (엔트로피)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">제2법칙의 표현</h3>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-semibold mb-2">클라우지우스 표현:</p>
                <p className="text-sm text-slate-300">열은 자발적으로 저온에서 고온으로 이동하지 않는다</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-semibold mb-2">켈빈-플랑크 표현:</p>
                <p className="text-sm text-slate-300">열을 100% 일로 변환하는 열기관은 불가능하다</p>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-semibold mb-2">엔트로피 표현:</p>
                <p className="text-sm text-slate-300">고립계의 엔트로피는 항상 증가한다 (ΔS ≥ 0)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">엔트로피 (Entropy)</h3>
            <p className="text-slate-300 mb-4">무질서도, 불확실성의 척도</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">ΔS = Q/T (가역 과정)</p>
              <p className="font-mono text-sm">S = k_B ln(Ω) (미시적 정의)</p>
              <p className="text-xs text-slate-400 mt-2">Ω: 미시 상태의 수</p>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">5. 열기관 (Heat Engines)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">열기관의 효율</h3>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">η = W/Q_H = (Q_H - Q_C)/Q_H = 1 - Q_C/Q_H</p>
                <div className="text-xs text-slate-400 mt-2 space-y-1">
                  <p>• Q_H: 고온부에서 흡수한 열</p>
                  <p>• Q_C: 저온부로 방출한 열</p>
                  <p>• W: 한 일</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">카르노 사이클 (Carnot Cycle)</h3>
            <p className="text-slate-300 mb-3">이론적으로 가능한 최대 효율의 열기관</p>
            <div className="space-y-3">
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                <p className="font-mono text-sm">η_Carnot = 1 - T_C/T_H</p>
                <p className="text-xs text-slate-400 mt-2">T는 절대온도 (K)</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <p className="text-sm text-slate-300">4단계: 등온 팽창 → 단열 팽창 → 등온 압축 → 단열 압축</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">실제 열기관</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">오토 사이클</h4>
                <p className="text-sm text-slate-300">가솔린 엔진<br/>η ≈ 20-30%</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">디젤 사이클</h4>
                <p className="text-sm text-slate-300">디젤 엔진<br/>η ≈ 30-40%</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">랭킨 사이클</h4>
                <p className="text-sm text-slate-300">증기 터빈<br/>η ≈ 30-35%</p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">브레이튼 사이클</h4>
                <p className="text-sm text-slate-300">가스 터빈<br/>η ≈ 35-45%</p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">6. 열역학 제3법칙</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">제3법칙</h3>
            <p className="text-slate-300 mb-4">절대영도에서 완전한 결정의 엔트로피는 0이다</p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">lim(T→0) S = 0</p>
              <p className="text-xs text-slate-400 mt-2">절대영도(0 K)는 도달 불가능</p>
            </div>
          </div>
        </section>

        <section className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <CheckCircle className="w-8 h-8 text-purple-400" />
            핵심 요약
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">제0법칙</h3>
              <p className="text-sm text-slate-300">열평형<br/>온도 정의</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">제1법칙</h3>
              <p className="text-sm text-slate-300">ΔU = Q - W<br/>에너지 보존</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">제2법칙</h3>
              <p className="text-sm text-slate-300">ΔS ≥ 0<br/>무질서 증가</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">제3법칙</h3>
              <p className="text-sm text-slate-300">T→0, S→0<br/>절대영도</p>
            </div>
          </div>
        </section>

        <References
          sections={[
            {
              title: '📚 온라인 강의 & 리소스',
              icon: 'web' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'MIT OCW 8.01 - Classical Mechanics',
                  url: 'https://ocw.mit.edu/courses/8-01-classical-mechanics-fall-2016/',
                  description: 'MIT 물리학 입문 강의 - Walter Lewin 교수의 전설적인 강의 (2024)',
                  year: 2024
                },
                {
                  title: 'Khan Academy Physics',
                  url: 'https://www.khanacademy.org/science/physics',
                  description: '무료 물리학 강의 - 역학, 전자기, 열역학 완전 커버 (2024)',
                  year: 2024
                },
                {
                  title: 'The Feynman Lectures on Physics',
                  url: 'https://www.feynmanlectures.caltech.edu/',
                  description: '파인만 물리학 강의 - 노벨상 수상자의 명강의 온라인 공개 (2024)',
                  year: 2024
                },
                {
                  title: 'Walter Lewin Lectures - MIT',
                  url: 'https://www.youtube.com/playlist?list=PLyQSN7X0ro203puVhQsmCj9qhlFQ-As8e',
                  description: '물리학 실험 중심 강의 - 재미있고 직관적인 설명 (YouTube, 2024)',
                  year: 2024
                }
              ]
            },
            {
              title: '📖 핵심 교재',
              icon: 'research' as const,
              color: 'border-pink-500',
              items: [
                {
                  title: 'University Physics (Young & Freedman)',
                  url: 'https://www.pearson.com/en-us/subject-catalog/p/university-physics-with-modern-physics/P200000006228',
                  description: '전 세계 대학 표준 물리학 교재 - 명확한 설명과 풍부한 예제 (15판, 2024)',
                  year: 2024
                },
                {
                  title: 'Physics for Scientists and Engineers (Serway & Jewett)',
                  url: 'https://www.cengage.com/c/physics-for-scientists-and-engineers-10e-serway',
                  description: '공학도를 위한 물리학 - 응용 중심 설명 (10판, 2024)',
                  year: 2024
                },
                {
                  title: 'Classical Mechanics (Goldstein)',
                  url: 'https://www.pearson.com/en-us/subject-catalog/p/classical-mechanics/P200000006154',
                  description: '고전역학 고급 교재 - 라그랑주, 해밀턴 역학 완벽 다룸 (3판)',
                  year: 2002
                },
                {
                  title: 'Fundamentals of Physics (Halliday, Resnick, Walker)',
                  url: 'https://www.wiley.com/en-us/Fundamentals+of+Physics%2C+11th+Edition-p-9781119306856',
                  description: '물리학 기초 명저 - 개념 중심 명쾌한 설명 (11판, 2018)',
                  year: 2018
                }
              ]
            },
            {
              title: '🛠️ 실전 도구',
              icon: 'tools' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'PhET Interactive Simulations',
                  url: 'https://phet.colorado.edu/',
                  description: 'University of Colorado 물리학 시뮬레이션 - 80+ 인터랙티브 시뮬레이터 (2024)',
                  year: 2024
                },
                {
                  title: 'Tracker Video Analysis',
                  url: 'https://physlets.org/tracker/',
                  description: '비디오 분석 도구 - 실험 영상에서 운동 데이터 추출 (2024)',
                  year: 2024
                },
                {
                  title: 'Algodoo',
                  url: 'http://www.algodoo.com/',
                  description: '2D 물리 시뮬레이션 - 드래그 앤 드롭으로 물리 실험 (2024)',
                  year: 2024
                },
                {
                  title: 'VPython',
                  url: 'https://www.glowscript.org/docs/VPythonDocs/index.html',
                  description: 'Python 3D 물리 시뮬레이션 - 코드로 물리 현상 구현 (2024)',
                  year: 2024
                }
              ]
            }
          ]}
        />

        <div className="flex justify-between items-center pt-8 border-t border-purple-500/20">
          <Link href="/modules/physics-fundamentals?chapter=electromagnetism" className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span>이전: Chapter 7</span>
          </Link>
          <Link href="/modules/physics-fundamentals" className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors">
            <span>모듈 완료</span>
            <CheckCircle className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </div>
  )
}
