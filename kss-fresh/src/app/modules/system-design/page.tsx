'use client'

import Link from 'next/link'
import { useState } from 'react'
import { 
  ArrowLeft, Clock, Target, Users, BookOpen, 
  Play, ChevronRight, Server, Database, Cloud,
  Network, Shield, Activity, Layers, Cpu,
  HardDrive, Zap, GitBranch, Box
} from 'lucide-react'
import { metadata } from './metadata'

export default function SystemDesignModule() {
  const [hoveredChapter, setHoveredChapter] = useState<number | null>(null)
  const [activeSimulator, setActiveSimulator] = useState<string | null>(null)

  const getChapterIcon = (chapterId: string) => {
    const icons: { [key: string]: any } = {
      'fundamentals': Server,
      'scaling': Layers,
      'caching': HardDrive,
      'database': Database,
      'messaging': GitBranch,
      'microservices': Box,
      'monitoring': Activity,
      'case-studies': Cpu
    }
    return icons[chapterId] || BookOpen
  }

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/"
          className="inline-flex items-center text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          홈으로 돌아가기
        </Link>
        
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
                {metadata.name}
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300 mb-6">
                {metadata.description}
              </p>
              
              <div className="flex flex-wrap gap-4 mb-6">
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <Clock className="w-5 h-5" />
                  <span>{metadata.duration}</span>
                </div>
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <Target className="w-5 h-5" />
                  <span className="capitalize">{metadata.level}</span>
                </div>
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <BookOpen className="w-5 h-5" />
                  <span>{metadata.chapters.length} 챕터</span>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                {metadata.prerequisites.map((prereq, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-sm"
                  >
                    {prereq}
                  </span>
                ))}
              </div>
            </div>
            
            <div className="hidden lg:block">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-indigo-400 rounded-full blur-2xl opacity-20"></div>
                <Server className="w-32 h-32 text-purple-600 dark:text-purple-400 relative" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Learning Path */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          학습 경로
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {metadata.chapters.map((chapter) => {
            const Icon = getChapterIcon(chapter.id)
            return (
              <Link
                key={chapter.id}
                href={`/modules/system-design/${chapter.id}`}
                className="group relative"
                onMouseEnter={() => setHoveredChapter(chapter.number)}
                onMouseLeave={() => setHoveredChapter(null)}
              >
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-2 border-transparent hover:border-purple-500 dark:hover:border-purple-400">
                  <div className="flex items-start justify-between mb-4">
                    <div className="p-3 bg-gradient-to-br from-purple-500 to-indigo-500 rounded-lg text-white">
                      <Icon className="w-6 h-6" />
                    </div>
                    <span className="text-sm font-semibold text-purple-600 dark:text-purple-400">
                      Chapter {chapter.number}
                    </span>
                  </div>
                  
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    {chapter.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {chapter.description}
                  </p>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500 dark:text-gray-500">
                      {chapter.duration}
                    </span>
                    <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors" />
                  </div>

                  {hoveredChapter === chapter.number && (
                    <div className="absolute top-full left-0 right-0 mt-2 p-4 bg-white dark:bg-gray-800 rounded-lg shadow-xl z-10 border border-purple-200 dark:border-purple-700">
                      <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
                        학습 목표:
                      </p>
                      <ul className="space-y-1">
                        {chapter.objectives.map((objective, idx) => (
                          <li key={idx} className="text-xs text-gray-600 dark:text-gray-400 flex items-start gap-2">
                            <div className="w-1 h-1 bg-purple-500 rounded-full mt-1.5 flex-shrink-0" />
                            {objective}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </Link>
            )
          })}
        </div>
      </div>

      {/* Interactive Simulators */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          인터랙티브 시뮬레이터
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {metadata.simulators.map((simulator) => (
            <div
              key={simulator.id}
              className="bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl p-1"
            >
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 h-full">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                  {simulator.title}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  {simulator.description}
                </p>
                <Link
                  href={`/modules/system-design/simulators/${simulator.id}`}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
                >
                  <Play className="w-4 h-4" />
                  시작하기
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Design Principles */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          핵심 설계 원칙
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-100 dark:bg-purple-900/30 rounded-full mb-4">
              <Zap className="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">확장성</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              트래픽 증가에 대응하는 수평/수직 확장 전략
            </p>
          </div>
          
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-indigo-100 dark:bg-indigo-900/30 rounded-full mb-4">
              <Shield className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">신뢰성</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              장애 복구와 데이터 일관성 보장
            </p>
          </div>
          
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full mb-4">
              <Cloud className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">가용성</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              99.9% 이상의 서비스 가동 시간 달성
            </p>
          </div>
          
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full mb-4">
              <Network className="w-8 h-8 text-green-600 dark:text-green-400" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">성능</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              낮은 지연시간과 높은 처리량 최적화
            </p>
          </div>
        </div>
      </div>

      {/* Real-world Applications */}
      <div className="mt-12 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-2xl shadow-xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">
          실무 적용 사례
        </h2>
        <p className="mb-6 text-purple-100">
          Netflix, Uber, Twitter 등 글로벌 서비스의 실제 아키텍처를 분석하고 학습합니다
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="font-semibold mb-1">Netflix</p>
            <p className="text-sm text-purple-200">마이크로서비스</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="font-semibold mb-1">Uber</p>
            <p className="text-sm text-purple-200">실시간 매칭</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="font-semibold mb-1">Twitter</p>
            <p className="text-sm text-purple-200">타임라인 설계</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="font-semibold mb-1">YouTube</p>
            <p className="text-sm text-purple-200">동영상 스트리밍</p>
          </div>
        </div>
      </div>
    </div>
  )
}