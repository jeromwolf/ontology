'use client'

import React from 'react'

export default function Chapter9() {
  const chapterInfo = {
    title: 'AI 자동 문서화 시스템',
    subtitle: 'API 문서, README, 주석을 AI로 자동 생성',
    icon: '📚'
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
    </div>
  )
}
