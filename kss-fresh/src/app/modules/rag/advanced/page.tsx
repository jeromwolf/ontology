'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  CheckCircle2, Trophy, BookOpen, ExternalLink, 
  Clock, ArrowLeft, ChevronRight, Zap, Target, Award
} from 'lucide-react'
import { advancedCurriculum, advancedChecklist } from '@/data/rag/advancedCurriculum'

export default function AdvancedCurriculumPage() {
  const [completedCurriculumItems, setCompletedCurriculumItems] = useState<string[]>([])
  const [completedChecklistItems, setCompletedChecklistItems] = useState<string[]>([])
  
  // Calculate curriculum progress
  const getCurriculumProgress = () => {
    const completed = advancedCurriculum.filter(item => 
      completedCurriculumItems.includes(item.id)
    ).length
    return (completed / advancedCurriculum.length) * 100
  }

  // Calculate checklist progress
  const getChecklistProgress = () => {
    return (completedChecklistItems.length / advancedChecklist.length) * 100
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
      <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
        <Link
          href="/modules/rag"
          className="inline-flex items-center gap-2 text-purple-100 hover:text-white mb-6 transition-colors"
        >
          <ArrowLeft size={20} />
          RAG 모듈로 돌아가기
        </Link>
        
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
            <Trophy size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Step 3: 고급 과정</h1>
            <p className="text-purple-100 text-lg">프로덕션 레벨 구현</p>
          </div>
        </div>
        
        <p className="text-purple-100 mb-6">
          실제 서비스에 적용 가능한 고급 기법을 마스터합니다. 
          GraphRAG, 하이브리드 검색, 프롬프트 엔지니어링 등 최신 기술을 활용한 고성능 RAG 시스템을 구축해보세요.
        </p>

        {/* Progress Overview */}
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-purple-100">커리큘럼 진행률</span>
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
              <span className="text-purple-100">체크리스트 진행률</span>
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

      {/* Prerequisites Check */}
      <div className="bg-amber-50 dark:bg-amber-900/20 rounded-2xl p-6 border border-amber-200 dark:border-amber-700">
        <h3 className="font-bold text-amber-800 dark:text-amber-200 mb-3">⚠️ 선행 학습 확인</h3>
        <p className="text-amber-700 dark:text-amber-300 mb-4">
          고급 과정을 시작하기 전에 다음 과정들을 완료하셨나요?
        </p>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/modules/rag/beginner"
            className="inline-flex items-center gap-2 px-3 py-2 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-lg text-sm hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
          >
            Step 1: 초급 과정
            <ChevronRight size={14} />
          </Link>
          <Link
            href="/modules/rag/intermediate"
            className="inline-flex items-center gap-2 px-3 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-lg text-sm hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors"
          >
            Step 2: 중급 과정
            <ChevronRight size={14} />
          </Link>
        </div>
      </div>

      {/* Course Overview */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">과정 개요</h2>
        
        <div className="grid md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-purple-100 dark:bg-purple-900/20 rounded-xl flex items-center justify-center mb-4">
              <Clock className="text-purple-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">학습 시간</h3>
            <p className="text-gray-600 dark:text-gray-400">약 20시간</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-pink-100 dark:bg-pink-900/20 rounded-xl flex items-center justify-center mb-4">
              <Target className="text-pink-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">학습 방식</h3>
            <p className="text-gray-600 dark:text-gray-400">프로젝트 기반</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-indigo-100 dark:bg-indigo-900/20 rounded-xl flex items-center justify-center mb-4">
              <BookOpen className="text-indigo-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">난이도</h3>
            <p className="text-gray-600 dark:text-gray-400">고급/전문가</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-emerald-100 dark:bg-emerald-900/20 rounded-xl flex items-center justify-center mb-4">
              <Award className="text-emerald-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">완료 후</h3>
            <p className="text-gray-600 dark:text-gray-400">RAG 전문가</p>
          </div>
        </div>
      </div>

      {/* Special Features Alert */}
      <div className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl p-6 border border-purple-200 dark:border-purple-700">
        <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-3 flex items-center gap-2">
          <Trophy size={20} />
          고급 과정 특별 기능
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">GraphRAG 실습</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">지식 그래프를 활용한 차세대 RAG 시스템 구현</p>
          </div>
          <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">프로덕션 최적화</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">대규모 시스템에서의 성능 최적화 기법</p>
          </div>
        </div>
      </div>

      {/* Detailed Curriculum */}
      <div className="space-y-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">상세 커리큘럼</h2>
        
        {advancedCurriculum.map((item, index) => {
          const isCompleted = completedCurriculumItems.includes(item.id)
          
          return (
            <div key={item.id} className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                    isCompleted 
                      ? 'bg-purple-500 text-white' 
                      : 'bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400'
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
                      ? 'bg-purple-500 text-white'
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
                        <CheckCircle2 size={14} className="text-purple-500 mt-0.5" />
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
                            ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-700 hover:bg-purple-100 dark:hover:bg-purple-900/30'
                            : resource.type === 'simulator'
                            ? 'bg-pink-50 dark:bg-pink-900/20 border-pink-200 dark:border-pink-700 hover:bg-pink-100 dark:hover:bg-pink-900/30'
                            : 'bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            {resource.type === 'chapter' && <BookOpen size={16} className="text-purple-600" />}
                            {resource.type === 'simulator' && <Zap size={16} className="text-pink-600" />}
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
            </div>
          )
        })}
      </div>

      {/* Checklist */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-2xl p-8">
        <h3 className="text-xl font-bold text-purple-800 dark:text-purple-200 mb-6 flex items-center gap-2">
          <CheckCircle2 size={24} />
          고급 과정 체크리스트
        </h3>
        
        <p className="text-purple-700 dark:text-purple-300 mb-6">
          아래 항목들을 체크하여 학습 진도를 관리하세요. 모든 항목을 완료하면 RAG 전문가가 됩니다!
        </p>
        
        <div className="grid md:grid-cols-2 gap-4">
          {advancedChecklist.map((item, i) => {
            const isChecked = completedChecklistItems.includes(item)
            
            return (
              <label key={i} className="flex items-start gap-3 cursor-pointer p-3 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors">
                <input 
                  type="checkbox" 
                  checked={isChecked}
                  onChange={() => toggleChecklistItem(item)}
                  className="mt-0.5 text-purple-500 rounded"
                />
                <span className={`text-sm ${isChecked ? 'text-purple-800 dark:text-purple-200 line-through' : 'text-gray-700 dark:text-gray-300'}`}>
                  {item}
                </span>
              </label>
            )
          })}
        </div>
        
        {getChecklistProgress() === 100 && (
          <div className="mt-6 p-6 bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/40 dark:to-pink-900/40 rounded-lg border border-purple-200 dark:border-purple-700">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mb-4">
                <Trophy className="text-white" size={32} />
              </div>
              <h4 className="text-xl font-bold text-purple-800 dark:text-purple-200 mb-2">
                🏆 축하합니다! RAG 전문가가 되셨습니다!
              </h4>
              <p className="text-purple-700 dark:text-purple-300 mb-4">
                이제 실무에서 RAG 시스템을 설계하고 구현할 수 있는 전문가 수준에 도달했습니다.
              </p>
              <div className="flex justify-center gap-4">
                <Link
                  href="/modules/rag/simulators/graphrag-explorer"
                  className="inline-flex items-center gap-2 bg-purple-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-purple-700 transition-colors"
                >
                  GraphRAG 체험하기
                  <ChevronRight size={16} />
                </Link>
                <Link
                  href="/modules/rag/simulators/rag-playground"
                  className="inline-flex items-center gap-2 bg-pink-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-pink-700 transition-colors"
                >
                  RAG 플레이그라운드
                  <ChevronRight size={16} />
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/intermediate"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            중급 과정으로
          </Link>
          
          <Link
            href="/modules/rag"
            className="inline-flex items-center gap-2 bg-purple-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-purple-600 transition-colors"
          >
            RAG 모듈 메인으로
            <ChevronRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}