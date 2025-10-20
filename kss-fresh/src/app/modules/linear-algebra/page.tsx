'use client'

import React from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Code, Clock, Users, Star, Target } from 'lucide-react'
import { metadata } from './metadata'

export default function LinearAlgebraModule() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="max-w-7xl mx-auto px-8 py-12">
        {/* Header */}
        <div className="mb-12">
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm">홈으로 돌아가기</span>
          </Link>

          <div className="flex items-center gap-4 mb-4">
            <span className="text-6xl">{metadata.icon}</span>
            <div>
              <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent mb-2">
                {metadata.title}
              </h1>
              <p className="text-xl text-slate-300">{metadata.description}</p>
            </div>
          </div>

          {/* Stats */}
          <div className="flex flex-wrap gap-6 mt-6">
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-blue-400" />
              <span className="text-slate-300">{metadata.duration}</span>
            </div>
            <div className="flex items-center gap-2">
              <Users className="w-5 h-5 text-blue-400" />
              <span className="text-slate-300">{metadata.students.toLocaleString()}명 수강</span>
            </div>
            <div className="flex items-center gap-2">
              <Star className="w-5 h-5 text-yellow-400 fill-yellow-400" />
              <span className="text-slate-300">{metadata.rating} / 5.0</span>
            </div>
            <div className="inline-flex px-3 py-1 bg-blue-500/20 border border-blue-500/50 rounded-full">
              <span className="text-blue-300 text-sm font-medium capitalize">{metadata.difficulty}</span>
            </div>
          </div>
        </div>

        {/* Chapters */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
            <BookOpen className="w-8 h-8 text-blue-400" />
            강의 목차
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {metadata.chapters.map((chapter) => (
              <Link
                key={chapter.id}
                href={`/modules/linear-algebra/${chapter.id}`}
                className="group bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 hover:border-blue-500/50 hover:bg-slate-800/80 transition-all"
              >
                <div className="flex items-start justify-between mb-3">
                  <h3 className="text-lg font-semibold text-white group-hover:text-blue-400 transition-colors">
                    {chapter.title}
                  </h3>
                  <span className="text-sm text-slate-400">{chapter.duration}</span>
                </div>
                <p className="text-slate-300 text-sm mb-4">{chapter.description}</p>
                <div className="space-y-1">
                  {chapter.objectives.slice(0, 2).map((obj, idx) => (
                    <div key={idx} className="flex items-start gap-2 text-xs text-slate-400">
                      <Target className="w-3 h-3 mt-0.5 text-blue-400 flex-shrink-0" />
                      <span>{obj}</span>
                    </div>
                  ))}
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Simulators */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
            <Code className="w-8 h-8 text-indigo-400" />
            인터랙티브 시뮬레이터
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {metadata.simulators.map((sim) => (
              <Link
                key={sim.id}
                href={`/modules/linear-algebra/simulators/${sim.id}`}
                className="group bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 hover:border-indigo-500/50 hover:bg-slate-800/80 transition-all"
              >
                <h3 className="text-lg font-semibold text-white group-hover:text-indigo-400 transition-colors mb-2">
                  {sim.title}
                </h3>
                <p className="text-slate-300 text-sm mb-3">{sim.description}</p>
                <div className="inline-flex px-2 py-1 bg-indigo-500/20 border border-indigo-500/50 rounded text-xs text-indigo-300">
                  {sim.difficulty}
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Resources */}
        <section>
          <h2 className="text-3xl font-bold mb-6">추천 학습 자료</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {metadata.resources.map((resource, idx) => (
              <a
                key={idx}
                href={resource.url}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 hover:border-blue-500/50 hover:bg-slate-800/80 transition-all"
              >
                <h3 className="text-white font-semibold mb-2">{resource.title}</h3>
                <p className="text-slate-400 text-sm">외부 링크 →</p>
              </a>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
