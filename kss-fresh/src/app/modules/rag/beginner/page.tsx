'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  CheckCircle2, GraduationCap, BookOpen, Zap, ExternalLink, 
  Lightbulb, Clock, ArrowLeft, Trophy, ChevronRight
} from 'lucide-react'
import { beginnerCurriculum, beginnerChecklist } from '@/data/rag/beginnerCurriculum'

export default function BeginnerCurriculumPage() {
  const [completedCurriculumItems, setCompletedCurriculumItems] = useState<string[]>([])
  const [completedChecklistItems, setCompletedChecklistItems] = useState<string[]>([])
  
  // Calculate curriculum progress
  const getCurriculumProgress = () => {
    const completed = beginnerCurriculum.filter(item => 
      completedCurriculumItems.includes(item.id)
    ).length
    return (completed / beginnerCurriculum.length) * 100
  }

  // Calculate checklist progress
  const getChecklistProgress = () => {
    return (completedChecklistItems.length / beginnerChecklist.length) * 100
  }

  const toggleCurriculumItem = (itemId: string) => {
    setCompletedCurriculumItems(prev => 
      prev.includes(itemId) 
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    )
  }

  const toggleChecklistItem = (item: string) => {
    setCompletedChecklistItems(prev => 
      prev.includes(item) 
        ? prev.filter(i => i !== item)
        : [...prev, item]
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl p-8 text-white">
        <Link
          href="/modules/rag"
          className="inline-flex items-center gap-2 text-emerald-100 hover:text-white mb-6 transition-colors"
        >
          <ArrowLeft size={20} />
          RAG 모듈로 돌아가기
        </Link>
        
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
            <GraduationCap size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Step 1: 초급 과정</h1>
            <p className="text-emerald-100 text-lg">RAG 기본 개념 이해</p>
          </div>
        </div>
        
        <p className="text-emerald-100 mb-6">
          RAG의 필요성과 기본 작동 원리를 학습합니다. 대화형 AI의 한계를 이해하고, 
          외부 지식을 활용한 답변 생성의 기본 원리를 체험해보세요.
        </p>

        {/* Progress Overview */}
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-emerald-100">커리큘럼 진행률</span>
              <span className="font-bold">{Math.round(getCurriculumProgress())}%</span>
            </div>
            <div className="w-full bg-white/20 rounded-full h-3">
              <div 
                className="bg-white h-3 rounded-full transition-all duration-500"
                style={{ width: `${getCurriculumProgress()}%` }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-emerald-100">체크리스트 진행률</span>
              <span className="font-bold">{Math.round(getChecklistProgress())}%</span>
            </div>
            <div className="w-full bg-white/20 rounded-full h-3">
              <div 
                className="bg-white h-3 rounded-full transition-all duration-500"
                style={{ width: `${getChecklistProgress()}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Course Overview */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">과정 개요</h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-green-100 dark:bg-green-900/20 rounded-xl flex items-center justify-center mb-4">
              <Clock className="text-green-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">학습 시간</h3>
            <p className="text-gray-600 dark:text-gray-400">약 10시간</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-blue-100 dark:bg-blue-900/20 rounded-xl flex items-center justify-center mb-4">
              <BookOpen className="text-blue-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">학습 방식</h3>
            <p className="text-gray-600 dark:text-gray-400">이론 + 실습</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-purple-100 dark:bg-purple-900/20 rounded-xl flex items-center justify-center mb-4">
              <Trophy className="text-purple-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">완료 후</h3>
            <p className="text-gray-600 dark:text-gray-400">중급 과정 진행</p>
          </div>
        </div>
      </div>

      {/* Detailed Curriculum */}
      <div className="space-y-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">상세 커리큘럼</h2>
        
        {beginnerCurriculum.map((item, index) => {
          const isCompleted = completedCurriculumItems.includes(item.id)
          
          return (
            <div key={item.id} className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                    isCompleted 
                      ? 'bg-green-500 text-white' 
                      : 'bg-emerald-100 dark:bg-emerald-900 text-emerald-600 dark:text-emerald-400'
                  }`}>
                    {isCompleted ? <CheckCircle2 size={24} /> : <span className="font-bold">{index + 1}</span>}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
                      {item.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">{item.description}</p>
                  </div>
                </div>
                
                <button
                  onClick={() => toggleCurriculumItem(item.id)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    isCompleted
                      ? 'bg-green-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {isCompleted ? '완료됨' : '완료 표시'}
                </button>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">학습 내용</h4>
                  <ul className="space-y-2">
                    {item.topics.map((topic, i) => (
                      <li key={i} className="text-sm text-gray-600 dark:text-gray-400 flex items-start gap-2">
                        <CheckCircle2 size={14} className="text-green-500 mt-0.5" />
                        <span>{topic}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">학습 자료</h4>
                  <div className="space-y-2">
                    {item.resources.map((resource, i) => (
                      <Link
                        key={i}
                        href={resource.url || '#'}
                        className={`block p-3 rounded-lg border transition-all duration-200 hover:shadow-sm ${
                          resource.type === 'chapter' 
                            ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700 hover:bg-emerald-100 dark:hover:bg-emerald-900/30'
                            : resource.type === 'simulator'
                            ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-900/30'
                            : 'bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            {resource.type === 'chapter' && <BookOpen size={16} className="text-emerald-600" />}
                            {resource.type === 'simulator' && <Zap size={16} className="text-blue-600" />}
                            {resource.type === 'external' && <ExternalLink size={16} className="text-gray-600" />}
                            <span className="text-sm font-medium text-gray-900 dark:text-white">
                              {resource.title}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            {resource.duration && (
                              <span className="text-xs text-gray-500 dark:text-gray-400">
                                {resource.duration}
                              </span>
                            )}
                            <ChevronRight size={16} className="text-gray-400" />
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
              
              {item.quiz && item.quiz.length > 0 && (
                <div className="mt-6 bg-amber-50 dark:bg-amber-900/20 p-4 rounded-lg">
                  <h4 className="font-semibold text-amber-800 dark:text-amber-200 mb-3 flex items-center gap-2">
                    <Lightbulb size={16} />
                    퀴즈 ({item.quiz.length}문)
                  </h4>
                  {item.quiz.map((q, i) => (
                    <div key={i} className="mb-4 last:mb-0">
                      <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Q{i + 1}. {q.question}
                      </p>
                      <div className="space-y-1 ml-4">
                        {q.options.map((opt, j) => (
                          <label key={j} className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer hover:text-gray-800 dark:hover:text-gray-200">
                            <input type="radio" name={`quiz-${item.id}-${i}`} className="text-emerald-500" />
                            {opt}
                          </label>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Checklist */}
      <div className="bg-green-50 dark:bg-green-900/20 rounded-2xl p-8">
        <h3 className="text-xl font-bold text-green-800 dark:text-green-200 mb-6 flex items-center gap-2">
          <CheckCircle2 size={24} />
          초급 과정 체크리스트
        </h3>
        
        <p className="text-green-700 dark:text-green-300 mb-6">
          아래 항목들을 체크하여 학습 진도를 관리하세요. 모든 항목을 완료하면 중급 과정으로 진행할 수 있습니다.
        </p>
        
        <div className="grid md:grid-cols-2 gap-4">
          {beginnerChecklist.map((item, i) => {
            const isChecked = completedChecklistItems.includes(item)
            
            return (
              <label key={i} className="flex items-start gap-3 cursor-pointer p-3 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors">
                <input 
                  type="checkbox" 
                  checked={isChecked}
                  onChange={() => toggleChecklistItem(item)}
                  className="mt-0.5 text-green-500 rounded"
                />
                <span className={`text-sm ${isChecked ? 'text-green-800 dark:text-green-200 line-through' : 'text-gray-700 dark:text-gray-300'}`}>
                  {item}
                </span>
              </label>
            )
          })}
        </div>
        
        {getChecklistProgress() === 100 && (
          <div className="mt-6 p-4 bg-green-100 dark:bg-green-900/40 rounded-lg border border-green-200 dark:border-green-700">
            <p className="text-green-800 dark:text-green-200 font-medium mb-2">🎉 초급 과정을 완료했습니다!</p>
            <Link
              href="/modules/rag/intermediate"
              className="inline-flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-700 transition-colors"
            >
              중급 과정으로 이동
              <ChevronRight size={16} />
            </Link>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            RAG 메인으로
          </Link>
          
          <Link
            href="/modules/rag/intermediate"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            중급 과정 확인하기
            <ChevronRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}