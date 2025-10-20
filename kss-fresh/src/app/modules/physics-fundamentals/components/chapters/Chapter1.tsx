'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Lightbulb, CheckCircle, AlertCircle } from 'lucide-react'

export default function Chapter1() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <div className="border-b border-purple-500/20 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <Link
            href="/modules/physics-fundamentals"
            className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors mb-3"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Physics Fundamentals 모듈로 돌아가기</span>
          </Link>
          <h1 className="text-3xl font-bold mb-2">Chapter 1: 역학의 기초</h1>
          <p className="text-slate-300">뉴턴의 운동 법칙, 힘과 운동</p>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
        {/* Section 1: 뉴턴의 제1법칙 - 관성의 법칙 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">1. 뉴턴의 제1법칙 - 관성의 법칙</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">관성의 법칙</h3>
            <p className="text-slate-300 mb-4">
              외부에서 힘이 작용하지 않으면, 정지한 물체는 계속 정지해 있고, 운동하는 물체는 등속 직선 운동을 계속한다.
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">
                ΣF = 0 ⟹ v = constant (등속도)
              </p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-yellow-400" />
              실생활 예시
            </h3>
            <ul className="space-y-2 text-slate-300">
              <li className="flex gap-2">
                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                <span><strong>버스 급정거:</strong> 버스가 갑자기 멈추면 승객은 관성으로 앞으로 쏠린다</span>
              </li>
              <li className="flex gap-2">
                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                <span><strong>우주선:</strong> 우주 공간에서는 마찰력이 없어 한번 추진하면 계속 등속 운동</span>
              </li>
              <li className="flex gap-2">
                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                <span><strong>탁자보 실험:</strong> 빠르게 잡아당기면 컵은 관성으로 제자리에 남는다</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Section 2: 뉴턴의 제2법칙 - 가속도의 법칙 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">2. 뉴턴의 제2법칙 - 가속도의 법칙</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">운동 방정식</h3>
            <p className="text-slate-300 mb-4">
              물체에 작용하는 알짜힘은 질량과 가속도의 곱과 같다.
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-3">
              <p className="font-mono text-lg font-bold">
                F = ma
              </p>
              <div className="text-sm text-slate-300 space-y-1">
                <p>• F: 알짜힘 (Net Force, N = kg·m/s²)</p>
                <p>• m: 질량 (Mass, kg)</p>
                <p>• a: 가속도 (Acceleration, m/s²)</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 자동차 가속</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 질량 1000 kg의 자동차가 2 m/s²로 가속한다. 필요한 힘은?</p>
              <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm space-y-2">
                <p>주어진 값: m = 1000 kg, a = 2 m/s²</p>
                <p>F = ma = 1000 kg × 2 m/s²</p>
                <p className="text-purple-400 font-bold">F = 2000 N</p>
              </div>
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-blue-400" />
              중요 개념
            </h3>
            <ul className="space-y-2 text-slate-300">
              <li>• 힘은 벡터량: 크기와 방향을 모두 가진다</li>
              <li>• 여러 힘이 작용할 때는 벡터 합을 구한다</li>
              <li>• 가속도 방향은 항상 알짜힘 방향과 같다</li>
              <li>• 질량이 클수록 같은 힘으로 가속하기 어렵다</li>
            </ul>
          </div>
        </section>

        {/* Section 3: 뉴턴의 제3법칙 - 작용 반작용 법칙 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">3. 뉴턴의 제3법칙 - 작용 반작용 법칙</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">작용과 반작용</h3>
            <p className="text-slate-300 mb-4">
              모든 작용에는 크기가 같고 방향이 반대인 반작용이 존재한다.
            </p>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
              <p className="font-mono text-sm">
                F<sub>AB</sub> = -F<sub>BA</sub>
              </p>
              <p className="text-sm text-slate-400 mt-2">
                (A가 B에 가하는 힘 = - B가 A에 가하는 힘)
              </p>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-yellow-400" />
              실생활 예시
            </h3>
            <div className="space-y-4">
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">1. 로켓 추진</h4>
                <p className="text-slate-300 text-sm">
                  로켓이 연료를 뒤로 분출하면(작용), 로켓은 앞으로 나아간다(반작용)
                </p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">2. 걷기</h4>
                <p className="text-slate-300 text-sm">
                  발로 땅을 뒤로 밀면(작용), 땅이 사람을 앞으로 민다(반작용)
                </p>
              </div>
              <div className="bg-slate-900/50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-300 mb-2">3. 총 발사</h4>
                <p className="text-slate-300 text-sm">
                  총알이 앞으로 나가면(작용), 총이 뒤로 반동한다(반작용)
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: 자유 물체 다이어그램 (Free Body Diagram) */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">4. 자유 물체 다이어그램 (FBD)</h2>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4 text-purple-300">FBD 그리기</h3>
            <p className="text-slate-300 mb-4">
              물체에 작용하는 모든 힘을 화살표로 표시한 다이어그램
            </p>
            <div className="space-y-3">
              <h4 className="font-semibold text-purple-300">단계:</h4>
              <ol className="space-y-2 text-slate-300 list-decimal list-inside">
                <li>물체를 점이나 상자로 표시</li>
                <li>중력(mg)을 아래 방향으로 그린다</li>
                <li>수직항력(N)을 수직 위 방향으로 그린다</li>
                <li>마찰력(f)을 운동 반대 방향으로 그린다</li>
                <li>장력(T), 외력(F) 등 추가 힘을 그린다</li>
              </ol>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-3">예제: 경사면 위의 물체</h3>
            <div className="space-y-3 text-slate-300">
              <p><strong>문제:</strong> 30° 경사면에 질량 5 kg 물체가 있다. 힘을 분석하라.</p>
              <div className="bg-slate-900/50 p-4 rounded-lg space-y-3">
                <p className="font-semibold">작용하는 힘:</p>
                <ul className="space-y-2 list-disc list-inside">
                  <li>중력: mg = 5 × 9.8 = 49 N (수직 아래)</li>
                  <li>수직항력: N (경사면에 수직)</li>
                  <li>마찰력: f (경사면 위쪽 방향)</li>
                </ul>
                <p className="font-semibold mt-3">성분 분해:</p>
                <ul className="space-y-2 list-disc list-inside font-mono text-sm">
                  <li>경사면 평행: mg sin(30°) = 49 × 0.5 = 24.5 N</li>
                  <li>경사면 수직: mg cos(30°) = 49 × 0.866 = 42.4 N</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: 일반적인 힘의 종류 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">5. 일반적인 힘의 종류</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">1. 중력 (Gravity)</h3>
              <p className="text-slate-300 text-sm mb-2">F<sub>g</sub> = mg</p>
              <p className="text-slate-400 text-sm">
                지구가 물체를 끌어당기는 힘<br/>
                g = 9.8 m/s² (지구 표면)
              </p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">2. 수직항력 (Normal Force)</h3>
              <p className="text-slate-300 text-sm mb-2">N ⊥ surface</p>
              <p className="text-slate-400 text-sm">
                표면이 물체를 미는 힘<br/>
                표면에 수직 방향
              </p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">3. 마찰력 (Friction)</h3>
              <p className="text-slate-300 text-sm mb-2">f = μN</p>
              <p className="text-slate-400 text-sm">
                운동을 방해하는 힘<br/>
                μ: 마찰계수 (0 ~ 1)
              </p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">4. 장력 (Tension)</h3>
              <p className="text-slate-300 text-sm mb-2">T (줄이나 로프)</p>
              <p className="text-slate-400 text-sm">
                줄이 당기는 힘<br/>
                줄 방향으로 작용
              </p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">5. 공기저항 (Air Resistance)</h3>
              <p className="text-slate-300 text-sm mb-2">F<sub>d</sub> ∝ v²</p>
              <p className="text-slate-400 text-sm">
                공기와의 마찰력<br/>
                속도의 제곱에 비례
              </p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">6. 탄성력 (Spring Force)</h3>
              <p className="text-slate-300 text-sm mb-2">F = -kx (Hooke's Law)</p>
              <p className="text-slate-400 text-sm">
                용수철이나 탄성체<br/>
                변위에 비례
              </p>
            </div>
          </div>
        </section>

        {/* Section 6: 연습 문제 */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-400" />
            <h2 className="text-2xl font-bold">6. 연습 문제</h2>
          </div>

          <div className="space-y-4">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">문제 1: 수평면 위의 물체</h3>
              <p className="text-slate-300 mb-4">
                질량 10 kg의 상자를 수평 방향으로 50 N의 힘으로 민다. 마찰계수 μ = 0.3일 때, 가속도는?
              </p>
              <details className="bg-slate-900/50 p-4 rounded-lg">
                <summary className="cursor-pointer font-semibold text-purple-300">풀이 보기</summary>
                <div className="mt-3 space-y-2 text-sm text-slate-300 font-mono">
                  <p>1. 수직항력: N = mg = 10 × 9.8 = 98 N</p>
                  <p>2. 마찰력: f = μN = 0.3 × 98 = 29.4 N</p>
                  <p>3. 알짜힘: F_net = 50 - 29.4 = 20.6 N</p>
                  <p>4. 가속도: a = F_net / m = 20.6 / 10 = 2.06 m/s²</p>
                  <p className="text-purple-400 font-bold">답: 2.06 m/s²</p>
                </div>
              </details>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">문제 2: 도르래 시스템</h3>
              <p className="text-slate-300 mb-4">
                5 kg과 3 kg의 두 물체가 도르래로 연결되어 있다. 5 kg 물체의 가속도는?
              </p>
              <details className="bg-slate-900/50 p-4 rounded-lg">
                <summary className="cursor-pointer font-semibold text-purple-300">풀이 보기</summary>
                <div className="mt-3 space-y-2 text-sm text-slate-300 font-mono">
                  <p>1. 5 kg 물체: m₁g - T = m₁a</p>
                  <p>2. 3 kg 물체: T - m₂g = m₂a</p>
                  <p>3. 두 식을 더하면: (m₁ - m₂)g = (m₁ + m₂)a</p>
                  <p>4. a = (5 - 3) × 9.8 / (5 + 3)</p>
                  <p>5. a = 2 × 9.8 / 8 = 2.45 m/s²</p>
                  <p className="text-purple-400 font-bold">답: 2.45 m/s²</p>
                </div>
              </details>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3 text-purple-300">문제 3: 경사면 운동</h3>
              <p className="text-slate-300 mb-4">
                45° 경사면에서 질량 2 kg 물체가 마찰 없이 미끄러진다. 가속도는?
              </p>
              <details className="bg-slate-900/50 p-4 rounded-lg">
                <summary className="cursor-pointer font-semibold text-purple-300">풀이 보기</summary>
                <div className="mt-3 space-y-2 text-sm text-slate-300 font-mono">
                  <p>1. 경사면 평행 성분: F = mg sin(45°)</p>
                  <p>2. a = g sin(45°) = 9.8 × 0.707</p>
                  <p>3. a = 6.93 m/s²</p>
                  <p className="text-purple-400 font-bold">답: 6.93 m/s²</p>
                </div>
              </details>
            </div>
          </div>
        </section>

        {/* Summary */}
        <section className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <CheckCircle className="w-8 h-8 text-purple-400" />
            핵심 요약
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">제1법칙: 관성</h3>
              <p className="text-sm text-slate-300">
                힘이 없으면 등속도 운동
              </p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">제2법칙: F = ma</h3>
              <p className="text-sm text-slate-300">
                힘은 가속도를 만든다
              </p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-300 mb-2">제3법칙: 작용-반작용</h3>
              <p className="text-sm text-slate-300">
                모든 힘은 쌍으로 존재
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-8 border-t border-purple-500/20">
          <Link
            href="/modules/physics-fundamentals"
            className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>모듈 홈</span>
          </Link>
          <Link
            href="/modules/physics-fundamentals?chapter=kinematics"
            className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-lg transition-colors"
          >
            <span>다음: Chapter 2 - 운동학</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Link>
        </div>
      </div>
    </div>
  )
}
