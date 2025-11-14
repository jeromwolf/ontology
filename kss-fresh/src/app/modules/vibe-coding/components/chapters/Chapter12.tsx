'use client'

import React from 'react'
import References from '@/components/common/References'

export default function Chapter12() {
  const chapterInfo = {
    title: '실전 프로젝트 - AI로 앱 처음부터 끝까지',
    subtitle: '48시간 프로젝트: 기획부터 배포까지',
    icon: '🚀'
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-12">
      <div className="text-center space-y-4 py-8">
        <div className="text-6xl mb-4">{chapterInfo.icon}</div>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
          {chapterInfo.title}
        </h1>
        <p className="text-xl text-gray-400">{chapterInfo.subtitle}</p>
      </div>

      <section className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 rounded-xl p-8 border border-purple-500/20">
        <h2 className="text-2xl font-bold text-white mb-4">챕터 개요</h2>
        <p className="text-gray-300 leading-relaxed">
          이 챕터에서는 <span className="text-purple-400 font-semibold">{chapterInfo.title.toLowerCase()}</span>에
          대해 심층적으로 학습합니다. 실전 예제와 베스트 프랙티스를 통해 AI 코딩의 고급 기법을 마스터합니다.
        </p>
      </section>

      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">주요 학습 내용</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((num) => (
            <div key={num} className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
              <h3 className="text-xl font-semibold text-purple-400 mb-3">주제 {num}</h3>
              <p className="text-gray-300">
                {chapterInfo.title}의 핵심 개념과 실전 적용 방법을 배웁니다.
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">실전 예제</h2>
        <div className="bg-black/30 rounded-lg p-6 border border-purple-500/20">
          <h4 className="text-purple-400 font-semibold mb-4">예제 프로젝트</h4>
          <div className="bg-gray-900 rounded p-4 font-mono text-sm text-gray-400">
            <div className="text-gray-500">// AI를 활용한 실전 예제 코드</div>
            <div className="text-purple-400 mt-2">
              {`// ${chapterInfo.title} 구현 예제\n// 상세한 실습 내용이 여기 포함됩니다`}
            </div>
          </div>
        </div>
      </section>

      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">요약</h2>
        <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 rounded-xl p-8 border border-purple-500/30">
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">•</span>
              <span>{chapterInfo.title}의 핵심 개념 이해</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">•</span>
              <span>실전 프로젝트 경험을 통한 실무 능력 향상</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">•</span>
              <span>AI 도구를 활용한 생산성 극대화</span>
            </li>
          </ul>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 창의적 코딩 플랫폼',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Processing',
                url: 'https://processing.org/',
                description: '비주얼 아트와 코딩 교육을 위한 프로그래밍 언어 (Java 기반)'
              },
              {
                title: 'p5.js',
                url: 'https://p5js.org/',
                description: 'Processing의 JavaScript 버전 (웹 브라우저에서 실행)'
              },
              {
                title: 'openFrameworks',
                url: 'https://openframeworks.cc/',
                description: 'C++ 창의적 코딩 프레임워크 (고성능 인터랙티브 아트)'
              },
              {
                title: 'Creative Coding - YouTube Channels',
                url: 'https://www.youtube.com/c/TheCodingTrain',
                description: 'The Coding Train (Daniel Shiffman) - 창의적 코딩 튜토리얼'
              },
              {
                title: 'OpenProcessing',
                url: 'https://openprocessing.org/',
                description: '창의적 코딩 커뮤니티 및 작품 공유 플랫폼'
              }
            ]
          },
          {
            title: '📖 핵심 리소스',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'The Nature of Code (Daniel Shiffman)',
                url: 'https://natureofcode.com/',
                description: '생성 예술 및 시뮬레이션 바이블 (무료 온라인, 2024년 v2)'
              },
              {
                title: 'Generative Design',
                url: 'http://www.generative-gestaltung.de/',
                description: '생성 디자인 교과서 (Processing 기반, 인터랙티브 예제)'
              },
              {
                title: 'Creative Coding Book',
                url: 'https://timrodenbroeker.de/courses/creative-coding/',
                description: 'Tim Rodenbröker 창의적 코딩 강좌'
              },
              {
                title: 'Casey Reas - Artist',
                url: 'https://reas.com/',
                description: 'Processing 공동 창시자, 생성 예술가'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구',
            icon: 'tools' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Three.js',
                url: 'https://threejs.org/',
                description: 'JavaScript 3D 라이브러리 (WebGL 기반, 2024 r169)'
              },
              {
                title: 'WebGL Fundamentals',
                url: 'https://webglfundamentals.org/',
                description: 'WebGL 튜토리얼 (3D 그래픽 기초)'
              },
              {
                title: 'Shader (GLSL)',
                url: 'https://www.shadertoy.com/',
                description: 'GLSL 셰이더 플레이그라운드 (실시간 렌더링)'
              },
              {
                title: 'TouchDesigner',
                url: 'https://derivative.ca/',
                description: '비주얼 프로그래밍 플랫폼 (실시간 인터랙티브 미디어)'
              },
              {
                title: 'Max/MSP',
                url: 'https://cycling74.com/products/max',
                description: '오디오-비주얼 프로그래밍 환경 (음악, VJ)'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
