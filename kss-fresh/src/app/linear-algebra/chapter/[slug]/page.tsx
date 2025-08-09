'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { ArrowLeft, ArrowRight, CheckCircle, Clock, BookOpen } from 'lucide-react'
import ChapterContent from '../../components/ChapterContent'

const chapters = [
  { id: 1, title: '벡터와 벡터공간', slug: 'vectors' },
  { id: 2, title: '행렬과 행렬연산', slug: 'matrices' },
  { id: 3, title: '선형변환', slug: 'linear-transformations' },
  { id: 4, title: '고유값과 고유벡터', slug: 'eigenvalues' },
  { id: 5, title: '직교성과 정규화', slug: 'orthogonality' },
  { id: 6, title: 'SVD와 차원축소', slug: 'svd' },
  { id: 7, title: '선형시스템', slug: 'linear-systems' },
  { id: 8, title: 'AI/ML 응용', slug: 'ml-applications' }
]

export default function ChapterPage() {
  const params = useParams()
  const slug = params.slug as string
  const [completed, setCompleted] = useState(false)
  const [readingTime, setReadingTime] = useState(0)
  
  const currentChapter = chapters.find(ch => ch.slug === slug)
  const currentIndex = chapters.findIndex(ch => ch.slug === slug)
  const prevChapter = currentIndex > 0 ? chapters[currentIndex - 1] : null
  const nextChapter = currentIndex < chapters.length - 1 ? chapters[currentIndex + 1] : null
  
  useEffect(() => {
    // 읽기 시간 추적
    const startTime = Date.now()
    
    return () => {
      const endTime = Date.now()
      const timeSpent = Math.floor((endTime - startTime) / 1000)
      setReadingTime(prev => prev + timeSpent)
    }
  }, [slug])
  
  useEffect(() => {
    // 진행 상태 로드
    const savedProgress = JSON.parse(
      localStorage.getItem('linear-algebra-progress') || '[]'
    )
    if (currentChapter && savedProgress.includes(currentChapter.id)) {
      setCompleted(true)
    }
  }, [currentChapter])
  
  const markAsCompleted = () => {
    if (!currentChapter) return
    
    const savedProgress = JSON.parse(
      localStorage.getItem('linear-algebra-progress') || '[]'
    )
    
    if (!savedProgress.includes(currentChapter.id)) {
      savedProgress.push(currentChapter.id)
      localStorage.setItem('linear-algebra-progress', JSON.stringify(savedProgress))
      setCompleted(true)
    }
  }
  
  if (!currentChapter) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-8 text-center">
            <h2 className="text-2xl font-bold text-red-700 dark:text-red-300 mb-4">
              챕터를 찾을 수 없습니다
            </h2>
            <Link
              href="/linear-algebra"
              className="inline-flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              목록으로 돌아가기
            </Link>
          </div>
        </div>
      </div>
    )
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/linear-algebra"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>목록으로</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Chapter {currentChapter.id}: {currentChapter.title}
              </h1>
            </div>
            <div className="flex items-center gap-3">
              {completed ? (
                <span className="flex items-center gap-1 px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-full text-sm font-medium">
                  <CheckCircle className="w-4 h-4" />
                  완료됨
                </span>
              ) : (
                <button
                  onClick={markAsCompleted}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium"
                >
                  완료로 표시
                </button>
              )}
            </div>
          </div>
        </div>
      </header>
      
      {/* Progress Bar */}
      <div className="bg-gray-100 dark:bg-gray-800">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between py-2 text-sm">
            <span className="text-gray-600 dark:text-gray-400">
              진행률: {currentChapter.id} / {chapters.length}
            </span>
            <div className="flex items-center gap-4 text-gray-600 dark:text-gray-400">
              <span className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                예상 시간: 25분
              </span>
              <span className="flex items-center gap-1">
                <BookOpen className="w-4 h-4" />
                난이도: 중급
              </span>
            </div>
          </div>
          <div className="pb-2">
            <div className="w-full h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 transition-all duration-500"
                style={{ width: `${(currentChapter.id / chapters.length) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        <ChapterContent chapterId={slug} />
        
        {/* Navigation */}
        <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
          {prevChapter ? (
            <Link
              href={`/linear-algebra/chapter/${prevChapter.slug}`}
              className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <div className="text-left">
                <div className="text-xs text-gray-500 dark:text-gray-400">이전</div>
                <div className="font-medium">{prevChapter.title}</div>
              </div>
            </Link>
          ) : (
            <div></div>
          )}
          
          {nextChapter ? (
            <Link
              href={`/linear-algebra/chapter/${nextChapter.slug}`}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all"
            >
              <div className="text-right">
                <div className="text-xs opacity-90">다음</div>
                <div className="font-medium">{nextChapter.title}</div>
              </div>
              <ArrowRight className="w-5 h-5" />
            </Link>
          ) : (
            <Link
              href="/linear-algebra"
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:shadow-lg transition-all"
            >
              <CheckCircle className="w-5 h-5" />
              과정 완료
            </Link>
          )}
        </div>
      </main>
    </div>
  )
}