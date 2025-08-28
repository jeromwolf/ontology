'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { intermediateCurriculum } from '@/data/rag/intermediateCurriculum'
import { 
  Calendar, 
  Clock, 
  CheckCircle2, 
  Circle,
  ChevronRight,
  Target,
  BookOpen,
  Code,
  Users,
  Trophy,
  Lightbulb,
  Database,
  Search,
  Layers,
  BarChart3,
  Video,
  Shield,
  Rocket,
  AlertCircle,
  ArrowLeft
} from 'lucide-react'

export default function IntermediatePage() {
  const [completedWeeks, setCompletedWeeks] = useState<number[]>([])
  const [completedTopics, setCompletedTopics] = useState<string[]>([])
  const [prerequisitesChecked, setPrerequisitesChecked] = useState(false)

  useEffect(() => {
    try {
      // 로컬 스토리지에서 진행 상황 불러오기
      const saved = localStorage.getItem('rag-intermediate-progress')
      if (saved) {
        const progress = JSON.parse(saved)
        setCompletedWeeks(progress.weeks || [])
        setCompletedTopics(progress.topics || [])
      }

      // 초급 과정 완료 여부 확인
      const beginnerProgress = localStorage.getItem('rag-beginner-progress')
      if (beginnerProgress) {
        const progress = JSON.parse(beginnerProgress)
        const allWeeksCompleted = progress.weeks?.length >= 4
        setPrerequisitesChecked(allWeeksCompleted)
      }
    } catch (error) {
      console.error('Error loading progress:', error)
    }
  }, [])

  const toggleWeek = (week: number) => {
    try {
      const newCompleted = completedWeeks.includes(week)
        ? completedWeeks.filter(w => w !== week)
        : [...completedWeeks, week]
      
      setCompletedWeeks(newCompleted)
      localStorage.setItem('rag-intermediate-progress', JSON.stringify({
        weeks: newCompleted,
        topics: completedTopics
      }))
    } catch (error) {
      console.error('Error saving week progress:', error)
    }
  }

  const toggleTopic = (topic: string) => {
    try {
      const newCompleted = completedTopics.includes(topic)
        ? completedTopics.filter(t => t !== topic)
        : [...completedTopics, topic]
      
      setCompletedTopics(newCompleted)
      localStorage.setItem('rag-intermediate-progress', JSON.stringify({
        weeks: completedWeeks,
        topics: newCompleted
      }))
    } catch (error) {
      console.error('Error saving topic progress:', error)
    }
  }

  const weekIcons = [Database, Search, Code, Database, Video, Shield]
  const modules = intermediateCurriculum.modules || []
  const progressPercentage = modules.length > 0 ? (completedWeeks.length / modules.length) * 100 : 0
  

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 to-orange-600 rounded-2xl p-8 text-white">
        <Link
          href="/modules/rag"
          className="inline-flex items-center gap-2 text-amber-100 hover:text-white mb-6 transition-colors"
        >
          <ArrowLeft size={20} />
          RAG 모듈로 돌아가기
        </Link>
        
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
            <Target size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Step 2: 중급 과정</h1>
            <p className="text-amber-100 text-lg">{intermediateCurriculum.description}</p>
          </div>
        </div>
        
        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-amber-100">전체 진행률</span>
              <span className="font-bold">{Math.round(progressPercentage)}%</span>
            </div>
            <div className="w-full bg-white/20 rounded-full h-3">
              <div 
                className="bg-white h-3 rounded-full transition-all duration-500"
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
          </div>
          
          <div className="bg-white/10 rounded-xl p-4">
            <h3 className="font-bold mb-2">과정 정보</h3>
            <div className="space-y-1 text-sm text-amber-100">
              <p>• 기간: {intermediateCurriculum.duration}</p>
              <p>• 레벨: {intermediateCurriculum.level}</p>
              <p>• 모듈 수: {modules.length}개</p>
            </div>
          </div>
        </div>
      </div>

      {/* Prerequisites Check */}
      {!prerequisitesChecked && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <AlertCircle className="text-amber-600" size={24} />
            <h3 className="text-xl font-bold text-amber-800 dark:text-amber-200">전제조건 확인</h3>
          </div>
          <p className="text-amber-700 dark:text-amber-300 mb-4">
            중급 과정을 시작하기 전에 다음 전제조건을 확인해주세요:
          </p>
          <ul className="space-y-2 mb-6">
            {intermediateCurriculum.prerequisites.map((req, idx) => (
              <li key={idx} className="flex items-center gap-2 text-amber-700 dark:text-amber-300">
                <Circle size={12} className="text-amber-500" />
                <span>{req}</span>
              </li>
            ))}
          </ul>
          <Link
            href="/modules/rag/beginner"
            className="inline-flex items-center gap-2 bg-amber-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-amber-700 transition-colors"
          >
            초급 과정부터 시작하기
            <ChevronRight size={16} />
          </Link>
        </div>
      )}

      {/* Course Modules */}
      <div className="space-y-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">주차별 커리큘럼</h2>
        
        {modules.map((module, idx) => {
          const IconComponent = weekIcons[idx] || Database
          const isWeekCompleted = completedWeeks.includes(module.week)
          
          return (
            <div 
              key={module.week}
              className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700"
            >
              {/* Module Header */}
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                    isWeekCompleted 
                      ? 'bg-amber-500 text-white' 
                      : 'bg-amber-100 dark:bg-amber-900 text-amber-600 dark:text-amber-400'
                  }`}>
                    {isWeekCompleted ? <CheckCircle2 size={24} /> : <IconComponent size={24} />}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                      Week {module.week}: {module.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">{module.description}</p>
                  </div>
                </div>
                
                <button
                  onClick={() => toggleWeek(module.week)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    isWeekCompleted
                      ? 'bg-amber-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {isWeekCompleted ? '완료됨' : '완료 표시'}
                </button>
              </div>

              {/* 학습 주제 */}
              <div className="mb-6">
                <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                  학습 주제
                </h4>
                <div className="grid md:grid-cols-2 gap-2">
                  {module.topics.map((topic, tidx) => {
                    const topicId = `${module.week}-${tidx}`
                    const isTopicCompleted = completedTopics.includes(topicId)
                    
                    return (
                      <label 
                        key={tidx}
                        className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                      >
                        <input
                          type="checkbox"
                          checked={isTopicCompleted}
                          onChange={() => toggleTopic(topicId)}
                          className="w-5 h-5 rounded border-gray-300 text-amber-600 focus:ring-amber-500"
                        />
                        <span className={`text-gray-700 dark:text-gray-300 ${
                          isTopicCompleted ? 'line-through opacity-60' : ''
                        }`}>
                          {topic}
                        </span>
                      </label>
                    )
                  })}
                </div>
              </div>

              {/* 관련 리소스 */}
              <div className="flex flex-wrap gap-3">
                {module.chapters && module.chapters.length > 0 && (
                  <div className="flex items-center gap-2">
                    <BookOpen size={16} className="text-amber-600 dark:text-amber-400" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      챕터: {module.chapters.map((ch, chIdx) => {
                        if (!ch || !ch.url) return null
                        return (
                          <Link 
                            key={chIdx}
                            href={ch.url}
                            className="text-amber-600 hover:text-amber-700 dark:text-amber-400 dark:hover:text-amber-300 font-medium ml-1 hover:underline"
                          >
                            {ch.title}
                          </Link>
                        )
                      })}
                    </span>
                  </div>
                )}
                
                {module.simulators && module.simulators.length > 0 && (
                  <div className="flex items-center gap-2">
                    <Code size={16} className="text-blue-600 dark:text-blue-400" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      시뮬레이터: {module.simulators.join(', ')}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Learning Outcomes */}
      <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-emerald-800 dark:text-emerald-200 mb-6 flex items-center gap-2">
          <Trophy size={24} />
          학습 성과
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          {intermediateCurriculum.learningOutcomes.map((outcome, idx) => (
            <div key={idx} className="flex items-start gap-3">
              <CheckCircle2 size={20} className="text-emerald-600 mt-0.5" />
              <span className="text-emerald-700 dark:text-emerald-300">{outcome}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Project Ideas */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-purple-800 dark:text-purple-200 mb-6 flex items-center gap-2">
          <Lightbulb size={24} />
          프로젝트 아이디어
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          {intermediateCurriculum.projectIdeas.map((project, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-2">{project.title}</h3>
              <p className="text-purple-700 dark:text-purple-300 text-sm mb-4">{project.description}</p>
              <div className="flex justify-between text-xs">
                <span className={`px-2 py-1 rounded ${
                  project.difficulty === '중급' ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                }`}>
                  {project.difficulty}
                </span>
                <span className="text-purple-600">{project.estimatedTime}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/beginner"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            초급 과정
          </Link>
          
          <Link
            href="/modules/rag/advanced"
            className="inline-flex items-center gap-2 bg-amber-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-amber-600 transition-colors"
          >
            고급 과정으로
            <ChevronRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}