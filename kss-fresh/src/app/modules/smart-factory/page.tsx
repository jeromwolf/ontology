'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  Factory, Settings, Cpu, Activity, 
  ChevronRight, Clock, BookOpen, Play, Star,
  Cog, Eye, Bot, Shield, Zap, Gauge
} from 'lucide-react'
import { smartFactoryModule } from './metadata'

export default function SmartFactoryPage() {
  const [progress, setProgress] = useState<Record<string, boolean>>({})
  const [completedChapters, setCompletedChapters] = useState(0)

  useEffect(() => {
    const saved = localStorage.getItem('smart-factory-progress')
    if (saved) {
      const parsed = JSON.parse(saved)
      setProgress(parsed)
      setCompletedChapters(Object.values(parsed).filter(Boolean).length)
    }
  }, [])

  return (
    <>
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-to-r from-slate-700 to-slate-800 p-6 mb-8 border border-slate-600">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-slate-600/30 backdrop-blur-sm rounded-lg flex items-center justify-center border border-slate-500/30">
            <Factory className="w-6 h-6 text-slate-200" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-100">{smartFactoryModule.nameKo}</h1>
            <p className="text-sm text-slate-300">{smartFactoryModule.description}</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          <div className="bg-slate-600/20 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-500/20">
            <div className="flex items-center gap-1 text-slate-200">
              <Clock className="w-4 h-4" />
              <span className="text-xs font-medium">학습시간</span>
            </div>
            <div className="text-lg font-bold text-slate-100">
              {smartFactoryModule.estimatedHours > 0 
                ? `${smartFactoryModule.estimatedHours}시간` 
                : '자유 진도'}
            </div>
          </div>
          
          <div className="bg-slate-600/20 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-500/20">
            <div className="flex items-center gap-1 text-slate-200">
              <BookOpen className="w-4 h-4" />
              <span className="text-xs font-medium">진도</span>
            </div>
            <div className="text-lg font-bold text-slate-100">
              {completedChapters} / {smartFactoryModule.chapters.length}
            </div>
          </div>
          
          <div className="bg-slate-600/20 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-500/20">
            <div className="flex items-center gap-1 text-slate-200">
              <Gauge className="w-4 h-4" />
              <span className="text-xs font-medium">난이도</span>
            </div>
            <div className="text-lg font-bold text-slate-100">중급</div>
          </div>
          
          <div className="bg-slate-600/20 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-500/20">
            <div className="flex items-center gap-1 text-slate-200">
              <Zap className="w-4 h-4" />
              <span className="text-xs font-medium">시뮬레이터</span>
            </div>
            <div className="text-lg font-bold text-slate-100">{smartFactoryModule.simulators.length}개</div>
          </div>
        </div>
      </div>

      {/* Technology Showcase */}
      <div className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">🏭 핵심 기술 스택</h2>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-3 text-center border border-slate-200 dark:border-slate-700">
            <Settings className="w-6 h-6 text-slate-600 dark:text-slate-400 mx-auto mb-2" />
            <h3 className="font-semibold text-sm text-slate-700 dark:text-slate-300 mb-1">IoT</h3>
            <div className="text-xs text-slate-500 dark:text-slate-500 bg-slate-200 dark:bg-slate-700 rounded px-2 py-1">MQTT</div>
          </div>
          
          <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-3 text-center border border-slate-200 dark:border-slate-700">
            <Cpu className="w-6 h-6 text-slate-600 dark:text-slate-400 mx-auto mb-2" />
            <h3 className="font-semibold text-sm text-slate-700 dark:text-slate-300 mb-1">AI 예측</h3>
            <div className="text-xs text-slate-500 dark:text-slate-500 bg-slate-200 dark:bg-slate-700 rounded px-2 py-1">LSTM</div>
          </div>
          
          <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-3 text-center border border-slate-200 dark:border-slate-700">
            <Eye className="w-6 h-6 text-slate-600 dark:text-slate-400 mx-auto mb-2" />
            <h3 className="font-semibold text-sm text-slate-700 dark:text-slate-300 mb-1">머신 비전</h3>
            <div className="text-xs text-slate-500 dark:text-slate-500 bg-slate-200 dark:bg-slate-700 rounded px-2 py-1">YOLO</div>
          </div>
          
          <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-3 text-center border border-slate-200 dark:border-slate-700">
            <Cog className="w-6 h-6 text-slate-600 dark:text-slate-400 mx-auto mb-2" />
            <h3 className="font-semibold text-sm text-slate-700 dark:text-slate-300 mb-1">디지털 트윈</h3>
            <div className="text-xs text-slate-500 dark:text-slate-500 bg-slate-200 dark:bg-slate-700 rounded px-2 py-1">Unity</div>
          </div>
          
          <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-3 text-center border border-slate-200 dark:border-slate-700">
            <Bot className="w-6 h-6 text-slate-600 dark:text-slate-400 mx-auto mb-2" />
            <h3 className="font-semibold text-sm text-slate-700 dark:text-slate-300 mb-1">로봇 자동화</h3>
            <div className="text-xs text-slate-500 dark:text-slate-500 bg-slate-200 dark:bg-slate-700 rounded px-2 py-1">ROS</div>
          </div>
          
          <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-3 text-center border border-slate-200 dark:border-slate-700">
            <Shield className="w-6 h-6 text-slate-600 dark:text-slate-400 mx-auto mb-2" />
            <h3 className="font-semibold text-sm text-slate-700 dark:text-slate-300 mb-1">OT 보안</h3>
            <div className="text-xs text-slate-500 dark:text-slate-500 bg-slate-200 dark:bg-slate-700 rounded px-2 py-1">IEC62443</div>
          </div>
        </div>
      </div>

      {/* Chapters */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">📚 학습 커리큘럼</h2>
        <div className="space-y-4">
          {smartFactoryModule.chapters.map((chapter) => (
            <Link
              key={chapter.id}
              href={`/modules/smart-factory/${chapter.id}`}
              className="block group"
            >
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 hover:border-slate-500 dark:hover:border-slate-400 transition-all hover:shadow-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center border border-slate-200 dark:border-slate-600">
                        <span className="text-slate-600 dark:text-slate-400 font-bold">
                          {smartFactoryModule.chapters.indexOf(chapter) + 1}
                        </span>
                      </div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white group-hover:text-slate-600 dark:group-hover:text-slate-400 transition-colors">
                        {chapter.title}
                      </h3>
                      {progress[chapter.id] && (
                        <span className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs rounded-full border border-slate-200 dark:border-slate-600">
                          완료
                        </span>
                      )}
                    </div>
                    
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      {chapter.description}
                    </p>
                    
                    <div className="space-y-2">
                      {chapter.learningObjectives.map((objective, idx) => (
                        <div key={idx} className="flex items-start gap-2">
                          <Star className="w-4 h-4 text-slate-500 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-gray-700 dark:text-gray-300">{objective}</span>
                        </div>
                      ))}
                    </div>
                    
                    <div className="flex items-center gap-4 mt-4">
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        <Clock className="w-4 h-4 inline mr-1" />
                        {chapter.estimatedMinutes}분
                      </span>
                    </div>
                  </div>
                  
                  <ChevronRight className="w-6 h-6 text-gray-400 group-hover:text-slate-500 dark:group-hover:text-slate-400 transition-colors ml-4" />
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Simulators */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">🎮 인터랙티브 시뮬레이터</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {smartFactoryModule.simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={`/modules/smart-factory/simulators/${simulator.id}`}
              className="group"
            >
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 hover:border-slate-500 dark:hover:border-slate-400 transition-all hover:shadow-lg h-full">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center border border-slate-200 dark:border-slate-600">
                    <Factory className="w-6 h-6 text-slate-600 dark:text-slate-400" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-slate-600 dark:group-hover:text-slate-400 transition-colors mb-2">
                      {simulator.name}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {simulator.description}
                    </p>
                    <div className="mt-4 flex items-center gap-2 text-slate-600 dark:text-slate-400">
                      <Play className="w-4 h-4" />
                      <span className="text-sm font-semibold">시뮬레이터 실행</span>
                    </div>
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Key Features */}
      <div className="bg-gradient-to-br from-slate-50 to-gray-50 dark:from-slate-800 dark:to-gray-800 rounded-2xl p-8 border border-slate-200 dark:border-slate-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">✨ 이 모듈의 특별함</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-slate-700 rounded-xl p-6 border border-slate-200 dark:border-slate-600">
            <Factory className="w-10 h-10 text-slate-600 dark:text-slate-400 mb-4" />
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">실제 스마트 팩토리</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              삼성, LG, 현대차가 운영하는 실제 스마트 팩토리 사례와 기술 학습
            </p>
          </div>
          
          <div className="bg-white dark:bg-slate-700 rounded-xl p-6 border border-slate-200 dark:border-slate-600">
            <Activity className="w-10 h-10 text-slate-600 dark:text-slate-400 mb-4" />
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">예측 유지보수</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              AI 기반 장비 고장 예측으로 다운타임 최소화와 비용 절감 실현
            </p>
          </div>
          
          <div className="bg-white dark:bg-slate-700 rounded-xl p-6 border border-slate-200 dark:border-slate-600">
            <Cog className="w-10 h-10 text-slate-600 dark:text-slate-400 mb-4" />
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">디지털 트윈</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              3D 가상 공장으로 실제 생산 라인 최적화와 시뮬레이션 체험
            </p>
          </div>
        </div>
      </div>
    </>
  )
}