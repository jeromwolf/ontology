'use client'

import { useState } from 'react'
import Link from 'next/link'
import { englishConversationMetadata } from './metadata'
import { 
  MessageCircle, Volume2, Users, Headphones, 
  ChevronRight, PlayCircle, BookOpen, Target,
  Globe, Award, Clock, TrendingUp
} from 'lucide-react'

const iconMap: { [key: string]: React.ElementType } = {
  'conversation-basics': MessageCircle,
  'daily-situations': Users,
  'business-english': Target,
  'travel-english': Globe,
  'pronunciation-intonation': Volume2,
  'listening-comprehension': Headphones,
  'cultural-context': BookOpen,
  'advanced-conversation': Award
}

export default function EnglishConversationPage() {
  const [hoveredChapter, setHoveredChapter] = useState<string | null>(null)
  const [activeSimulator, setActiveSimulator] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50 to-pink-50 dark:from-gray-900 dark:to-rose-950/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl">
        <div className="space-y-8">
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-rose-500 to-pink-600 rounded-3xl p-8 text-white shadow-2xl">
        <div className="flex items-center gap-4 mb-6">
          <div className="p-3 bg-white/20 rounded-xl">
            <MessageCircle className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-3xl font-bold">AI와 함께하는 실전 영어회화</h2>
            <p className="text-rose-100 mt-1">자연스러운 대화로 마스터하는 영어 말하기</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-white/10 rounded-xl p-4">
            <div className="text-2xl font-bold">15+</div>
            <div className="text-rose-100">학습 시간</div>
          </div>
          <div className="bg-white/10 rounded-xl p-4">
            <div className="text-2xl font-bold">8</div>
            <div className="text-rose-100">챕터</div>
          </div>
          <div className="bg-white/10 rounded-xl p-4">
            <div className="text-2xl font-bold">4</div>
            <div className="text-rose-100">AI 시뮬레이터</div>
          </div>
          <div className="bg-white/10 rounded-xl p-4">
            <div className="text-2xl font-bold">1247</div>
            <div className="text-rose-100">수강생</div>
          </div>
        </div>
      </section>

      {/* Learning Features */}
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
        <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          🎯 학습 특징
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center gap-3 p-4 bg-rose-50 dark:bg-rose-950/20 rounded-xl">
            <MessageCircle className="w-6 h-6 text-rose-600 dark:text-rose-400" />
            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200">실시간 AI 대화</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">자연스러운 대화로 실력 향상</p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-4 bg-pink-50 dark:bg-pink-950/20 rounded-xl">
            <Volume2 className="w-6 h-6 text-pink-600 dark:text-pink-400" />
            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200">AI 발음 교정</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">정확한 발음과 억양 연습</p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-4 bg-purple-50 dark:bg-purple-950/20 rounded-xl">
            <Users className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200">상황별 연습</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">실생활 시나리오 기반 훈련</p>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Path */}
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
        <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          🗣️ 학습 로드맵
        </h3>
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          <div className="flex-shrink-0 px-4 py-2 bg-red-100 dark:bg-red-900/50 rounded-lg text-sm">
            기초 회화
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-orange-100 dark:bg-orange-900/50 rounded-lg text-sm">
            일상 대화
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-yellow-100 dark:bg-yellow-900/50 rounded-lg text-sm">
            비즈니스
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-green-100 dark:bg-green-900/50 rounded-lg text-sm">
            여행 영어
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg text-sm">
            발음 교정
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-indigo-100 dark:bg-indigo-900/50 rounded-lg text-sm">
            고급 회화
          </div>
        </div>
      </section>

      {/* Chapters Grid */}
      <section>
        <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-6">
          📚 학습 챕터
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {englishConversationMetadata.chapters.map((chapter) => {
            const Icon = iconMap[chapter.id] || BookOpen
            return (
              <Link
                key={chapter.id}
                href={`/modules/english-conversation/${chapter.id}`}
                className="group"
                onMouseEnter={() => setHoveredChapter(chapter.id)}
                onMouseLeave={() => setHoveredChapter(null)}
              >
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border-2 border-transparent hover:border-rose-500 transition-all duration-300 hover:shadow-xl">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-gradient-to-br from-rose-100 to-pink-100 dark:from-rose-900/50 dark:to-pink-900/50 rounded-xl group-hover:scale-110 transition-transform">
                      <Icon className="w-6 h-6 text-rose-600 dark:text-rose-400" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-2">
                        {chapter.title}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                        {chapter.description}
                      </p>
                      {hoveredChapter === chapter.id && (
                        <div className="space-y-1 animate-in slide-in-from-top-2">
                          {chapter.objectives.map((obj, idx) => (
                            <div key={idx} className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                              <div className="w-1 h-1 bg-rose-500 rounded-full" />
                              <span>{obj}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="flex items-center justify-between mt-4">
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {chapter.duration}
                        </span>
                        <ChevronRight className="w-4 h-4 text-rose-500 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      </section>

      {/* AI Simulators */}
      <section className="bg-gradient-to-br from-white to-rose-50 dark:from-gray-800 dark:to-rose-950/20 rounded-2xl p-6 shadow-lg">
        <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-6">
          🤖 AI 학습 시뮬레이터
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {englishConversationMetadata.simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={`/modules/english-conversation/simulators/${simulator.id}`}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-md hover:shadow-xl transition-all cursor-pointer block"
              onClick={() => {
                console.log('Card clicked, navigating to:', `/modules/english-conversation/simulators/${simulator.id}`)
                setActiveSimulator(simulator.id)
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-bold text-gray-800 dark:text-gray-200">
                  {simulator.name}
                </h4>
                <div className="p-2 bg-rose-100 dark:bg-rose-900/50 rounded-lg">
                  <PlayCircle className="w-5 h-5 text-rose-600 dark:text-rose-400" />
                </div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {simulator.description}
              </p>
              {activeSimulator === simulator.id && (
                <div className="mt-4 p-3 bg-rose-50 dark:bg-rose-900/20 rounded-lg animate-in slide-in-from-top-2">
                  <p className="text-xs text-rose-700 dark:text-rose-300">
                    클릭하여 시뮬레이터 시작 →
                  </p>
                </div>
              )}
            </Link>
          ))}
        </div>
      </section>

      {/* Learning Stats */}
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
        <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          📊 학습 성과
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4">
            <div className="w-12 h-12 bg-rose-100 dark:bg-rose-900/50 rounded-full flex items-center justify-center mx-auto mb-2">
              <TrendingUp className="w-6 h-6 text-rose-600 dark:text-rose-400" />
            </div>
            <div className="text-2xl font-bold text-gray-800 dark:text-gray-200">89%</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">평균 향상률</div>
          </div>
          <div className="text-center p-4">
            <div className="w-12 h-12 bg-pink-100 dark:bg-pink-900/50 rounded-full flex items-center justify-center mx-auto mb-2">
              <Clock className="w-6 h-6 text-pink-600 dark:text-pink-400" />
            </div>
            <div className="text-2xl font-bold text-gray-800 dark:text-gray-200">12분</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">일평균 학습시간</div>
          </div>
          <div className="text-center p-4">
            <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/50 rounded-full flex items-center justify-center mx-auto mb-2">
              <Award className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div className="text-2xl font-bold text-gray-800 dark:text-gray-200">4.9</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">만족도 평점</div>
          </div>
        </div>
      </section>

      {/* Prerequisites */}
      <section className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-2xl p-6 border border-amber-200 dark:border-amber-800">
        <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200 mb-3">
          📋 선수 지식
        </h3>
        <ul className="space-y-2">
          {englishConversationMetadata.prerequisites.map((prereq, idx) => (
            <li key={idx} className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
              <div className="w-2 h-2 bg-amber-500 rounded-full" />
              <span>{prereq}</span>
            </li>
          ))}
        </ul>
      </section>
        </div>
      </div>
    </div>
  )
}