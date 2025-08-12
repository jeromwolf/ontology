'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  BookOpen, Clock, Users, Star, Calendar, Target, BarChart3,
  ChevronRight, Play, FileText, ArrowLeft, Menu,
  CheckCircle, Circle, Lock, Brain, LineChart, Database,
  Zap, FlaskConical, Network, TrendingUp, MessageSquare,
  TestTube, Settings, Lightbulb, Bot, Wine, Gavel, GitBranch
} from 'lucide-react'
import { moduleMetadata } from './metadata'
import dynamic from 'next/dynamic'

const ChapterContent = dynamic(
  () => import('./components/ChapterContent'),
  { 
    ssr: false,
    loading: () => <div className="flex items-center justify-center h-96">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
    </div>
  }
)

interface ProgressData {
  [chapterId: string]: {
    completed: boolean
    progress: number
    lastAccessed: string
  }
}

type ViewMode = 'overview' | 'chapters' | 'simulators'

export default function DataSciencePage() {
  const [selectedChapter, setSelectedChapter] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('overview')
  const [progressData, setProgressData] = useState<ProgressData>({})
  const [selectedDifficulty, setSelectedDifficulty] = useState<'beginner' | 'intermediate' | 'advanced' | null>(null)

  // 진행 상황 로드
  useEffect(() => {
    const loadProgress = () => {
      const saved = localStorage.getItem(`module-progress-${moduleMetadata.id}`)
      if (saved) {
        setProgressData(JSON.parse(saved))
      }
    }
    loadProgress()
  }, [])

  // 진행 상황 저장
  const updateProgress = (chapterId: string, completed: boolean, progress: number = 100) => {
    const updated = {
      ...progressData,
      [chapterId]: {
        completed,
        progress,
        lastAccessed: new Date().toISOString()
      }
    }
    setProgressData(updated)
    localStorage.setItem(`module-progress-${moduleMetadata.id}`, JSON.stringify(updated))
  }

  // 완료된 챕터 수 계산
  const completedCount = Object.values(progressData).filter(p => p.completed).length
  const totalProgress = Math.round((completedCount / moduleMetadata.chapters.length) * 100)

  // 시뮬레이터 아이콘 매핑
  const simulatorIcons: { [key: string]: React.ReactNode } = {
    'ml-playground': <Brain className="w-6 h-6" />,
    'ml-playground-pycaret': <Zap className="w-6 h-6" />,
    'statistical-lab': <BarChart3 className="w-6 h-6" />,
    'neural-network-builder': <Network className="w-6 h-6" />,
    'clustering-visualizer': <Zap className="w-6 h-6" />,
    'time-series-forecaster': <TrendingUp className="w-6 h-6" />,
    'nlp-analyzer': <MessageSquare className="w-6 h-6" />,
    'ab-test-simulator': <TestTube className="w-6 h-6" />,
    'feature-engineering-lab': <Settings className="w-6 h-6" />,
    'model-explainer': <Lightbulb className="w-6 h-6" />,
    'recommendation-engine': <Bot className="w-6 h-6" />,
    'wine-price-predictor': <Wine className="w-6 h-6" />,
    'bidding-price-predictor': <Gavel className="w-6 h-6" />,
    'classification-model-comparator': <GitBranch className="w-6 h-6" />
  }

  // 난이도별 챕터 필터링
  const getChaptersByDifficulty = (difficulty: 'beginner' | 'intermediate' | 'advanced') => {
    const ranges = {
      beginner: [0, 4],
      intermediate: [4, 8],
      advanced: [8, 12]
    }
    const [start, end] = ranges[difficulty]
    return moduleMetadata.chapters.slice(start, end)
  }

  // 챕터 뷰 렌더링
  if (selectedChapter && viewMode === 'chapters') {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700 sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setSelectedChapter(null)}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  <ArrowLeft className="w-5 h-5" />
                </button>
                <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {moduleMetadata.chapters.find(ch => ch.id === selectedChapter)?.title}
                </h1>
              </div>
            </div>
          </div>
        </header>
        <div className="max-w-4xl mx-auto p-8">
          <ChapterContent 
            chapterId={selectedChapter}
            onComplete={() => updateProgress(selectedChapter, true)}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* 헤더 */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link href="/modules" className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 bg-gradient-to-r ${moduleMetadata.gradient} rounded-xl flex items-center justify-center`}>
                  <span className="text-white text-xl">{moduleMetadata.icon}</span>
                </div>
                <div>
                  <h1 className="text-lg font-semibold text-gray-900 dark:text-white">{moduleMetadata.title}</h1>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{moduleMetadata.description}</p>
                </div>
              </div>
            </div>
            
            {/* 네비게이션 탭 */}
            <div className="hidden md:flex items-center gap-2">
              <button
                onClick={() => setViewMode('overview')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  viewMode === 'overview'
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                개요
              </button>
              <button
                onClick={() => setViewMode('chapters')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  viewMode === 'chapters'
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                챕터 학습
              </button>
              <button
                onClick={() => setViewMode('simulators')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  viewMode === 'simulators'
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                시뮬레이터
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 개요 뷰 */}
        {viewMode === 'overview' && (
          <div className="space-y-8">
            {/* 모듈 소개 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
              <div className="text-center max-w-3xl mx-auto">
                <div className={`w-24 h-24 bg-gradient-to-r ${moduleMetadata.gradient} rounded-2xl flex items-center justify-center mx-auto mb-6`}>
                  <span className="text-5xl">{moduleMetadata.icon}</span>
                </div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                  {moduleMetadata.title}
                </h1>
                <p className="text-lg text-gray-600 dark:text-gray-400 mb-8">
                  {moduleMetadata.longDescription || moduleMetadata.description}
                </p>
                
                {/* 모듈 정보 */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <Clock className="w-8 h-8 text-blue-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 dark:text-gray-400">예상 시간</p>
                    <p className="font-semibold">{moduleMetadata.estimatedHours}시간</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <Target className="w-8 h-8 text-green-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 dark:text-gray-400">난이도</p>
                    <p className="font-semibold">{moduleMetadata.difficulty}</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <Users className="w-8 h-8 text-purple-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 dark:text-gray-400">수강생</p>
                    <p className="font-semibold">{moduleMetadata.students}명</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <Star className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 dark:text-gray-400">평점</p>
                    <p className="font-semibold">{moduleMetadata.rating}</p>
                  </div>
                </div>

                {/* 학습 목표 */}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-8">
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                    이 모듈에서 배우게 될 내용
                  </h3>
                  <div className="grid md:grid-cols-2 gap-4 text-left">
                    {moduleMetadata.skills.slice(0, 8).map((skill, index) => (
                      <div key={index} className="flex items-center gap-3">
                        <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                        <span className="text-gray-700 dark:text-gray-300">{skill}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 시작 버튼 */}
                <div className="flex gap-4 justify-center">
                  <button
                    onClick={() => setViewMode('chapters')}
                    className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all flex items-center gap-2"
                  >
                    <BookOpen className="w-5 h-5" />
                    챕터 학습 시작
                  </button>
                  <button
                    onClick={() => setViewMode('simulators')}
                    className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all flex items-center gap-2"
                  >
                    <FlaskConical className="w-5 h-5" />
                    시뮬레이터 체험
                  </button>
                </div>
              </div>
            </div>

            {/* 학습 경로 추천 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 text-center">
                나에게 맞는 학습 경로 선택
              </h2>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 border-2 border-transparent hover:border-green-400 transition-all cursor-pointer"
                     onClick={() => {
                       setSelectedDifficulty('beginner')
                       setViewMode('chapters')
                     }}>
                  <h3 className="text-xl font-semibold text-green-700 dark:text-green-400 mb-3">초급 과정</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    데이터 사이언스의 기초 개념과 통계 기본기를 탄탄하게 다집니다.
                  </p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>기초 통계와 확률</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>데이터 전처리 기법</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>기본 머신러닝 알고리즘</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border-2 border-transparent hover:border-blue-400 transition-all cursor-pointer"
                     onClick={() => {
                       setSelectedDifficulty('intermediate')
                       setViewMode('chapters')
                     }}>
                  <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-400 mb-3">중급 과정</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    실무에 바로 적용 가능한 머신러닝과 딥러닝 기법을 학습합니다.
                  </p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-blue-500" />
                      <span>고급 머신러닝 기법</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-blue-500" />
                      <span>딥러닝 기초</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-blue-500" />
                      <span>모델 평가와 최적화</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border-2 border-transparent hover:border-purple-400 transition-all cursor-pointer"
                     onClick={() => {
                       setSelectedDifficulty('advanced')
                       setViewMode('chapters')
                     }}>
                  <h3 className="text-xl font-semibold text-purple-700 dark:text-purple-400 mb-3">고급 과정</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    최신 연구 동향과 대규모 시스템 구축 경험을 쌓을 수 있습니다.
                  </p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-purple-500" />
                      <span>대규모 ML 시스템</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-purple-500" />
                      <span>MLOps와 배포</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-purple-500" />
                      <span>최신 AI 연구 동향</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 챕터 뷰 */}
        {viewMode === 'chapters' && (
          <div className="grid lg:grid-cols-3 gap-8">
            {/* 챕터 목록 */}
            <div className="lg:col-span-2 space-y-4">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
                챕터 목록
              </h2>
              
              {selectedDifficulty && (
                <div className="mb-4 flex items-center justify-between">
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedDifficulty === 'beginner' && '초급 과정 챕터'}
                    {selectedDifficulty === 'intermediate' && '중급 과정 챕터'}
                    {selectedDifficulty === 'advanced' && '고급 과정 챕터'}
                  </p>
                  <button
                    onClick={() => setSelectedDifficulty(null)}
                    className="text-sm text-blue-600 hover:text-blue-700"
                  >
                    전체 보기
                  </button>
                </div>
              )}

              {(selectedDifficulty ? getChaptersByDifficulty(selectedDifficulty) : moduleMetadata.chapters).map((chapter, index) => {
                const isCompleted = progressData[chapter.id]?.completed
                const isLocked = index > 0 && !progressData[moduleMetadata.chapters[index - 1].id]?.completed && !selectedDifficulty
                
                return (
                  <div
                    key={chapter.id}
                    className={`bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border-2 ${
                      isLocked ? 'border-gray-200 dark:border-gray-700 opacity-60' : 'border-transparent hover:border-blue-400'
                    } transition-all`}
                  >
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0">
                        {isLocked ? (
                          <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center">
                            <Lock className="w-6 h-6 text-gray-400" />
                          </div>
                        ) : isCompleted ? (
                          <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center">
                            <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
                          </div>
                        ) : (
                          <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                            <span className="text-blue-600 dark:text-blue-400 font-bold">{index + 1}</span>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                          {chapter.title}
                        </h3>
                        <p className="text-gray-600 dark:text-gray-400 mb-3">
                          {chapter.description}
                        </p>
                        <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                          <span className="flex items-center gap-1">
                            <Clock className="w-4 h-4" />
                            {chapter.estimatedMinutes}분
                          </span>
                          {isCompleted && (
                            <span className="text-green-600 dark:text-green-400">
                              ✓ 완료됨
                            </span>
                          )}
                        </div>
                      </div>
                      
                      <button
                        onClick={() => !isLocked && setSelectedChapter(chapter.id)}
                        disabled={isLocked}
                        className={`px-4 py-2 rounded-lg font-medium transition-all ${
                          isLocked
                            ? 'bg-gray-100 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                      >
                        {isCompleted ? '다시 학습' : '학습 시작'}
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* 진행 상황 */}
            <div className="space-y-6">
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  학습 진행률
                </h3>
                <div className="relative w-32 h-32 mx-auto mb-4">
                  <svg className="transform -rotate-90 w-32 h-32">
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="none"
                      className="text-gray-200 dark:text-gray-700"
                    />
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="url(#progress-gradient)"
                      strokeWidth="12"
                      fill="none"
                      strokeDasharray={`${2 * Math.PI * 56}`}
                      strokeDashoffset={`${2 * Math.PI * 56 * (1 - totalProgress / 100)}`}
                      className="transition-all duration-500"
                    />
                    <defs>
                      <linearGradient id="progress-gradient">
                        <stop offset="0%" stopColor="#3b82f6" />
                        <stop offset="100%" stopColor="#8b5cf6" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl font-bold text-gray-900 dark:text-white">{totalProgress}%</span>
                  </div>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">완료한 챕터</span>
                    <span className="font-medium">{completedCount} / {moduleMetadata.chapters.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">남은 시간</span>
                    <span className="font-medium">
                      {Math.ceil((moduleMetadata.chapters.reduce((sum, ch) => sum + ch.estimatedMinutes, 0) - 
                        moduleMetadata.chapters.filter(ch => progressData[ch.id]?.completed).reduce((sum, ch) => sum + ch.estimatedMinutes, 0)) / 60)}시간
                    </span>
                  </div>
                </div>
              </div>

              {/* 추천 시뮬레이터 */}
              <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  추천 시뮬레이터
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  학습한 내용을 바로 실습해보세요!
                </p>
                <button
                  onClick={() => setViewMode('simulators')}
                  className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition-colors"
                >
                  시뮬레이터 보기
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 시뮬레이터 뷰 */}
        {viewMode === 'simulators' && (
          <div className="space-y-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                인터랙티브 시뮬레이터
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400">
                실시간으로 데이터 사이언스 개념을 체험하고 실습해보세요
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {moduleMetadata.simulators.map((simulator) => (
                <Link
                  key={simulator.id}
                  href={`/modules/data-science/simulators/${simulator.id}`}
                  className="group"
                >
                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-all transform group-hover:-translate-y-1">
                    <div className={`h-32 bg-gradient-to-r ${simulator.gradient || moduleMetadata.gradient} flex items-center justify-center`}>
                      <div className="text-white">
                        {simulatorIcons[simulator.id] || <FlaskConical className="w-12 h-12" />}
                      </div>
                    </div>
                    <div className="p-6">
                      <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                        {simulator.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        {simulator.description}
                      </p>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-500 dark:text-gray-400">
                          난이도: {simulator.difficulty || '중급'}
                        </span>
                        <span className="text-blue-600 dark:text-blue-400 group-hover:translate-x-1 transition-transform inline-flex items-center gap-1">
                          시작하기 <ChevronRight className="w-4 h-4" />
                        </span>
                      </div>
                    </div>
                  </div>
                </Link>
              ))}
            </div>

            {/* 시뮬레이터 카테고리 */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8">
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
                시뮬레이터 카테고리
              </h3>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
                  <Brain className="w-10 h-10 text-purple-500 mb-3" />
                  <h4 className="font-semibold mb-2">머신러닝</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    분류, 회귀, 클러스터링 등 ML 알고리즘 실습
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
                  <Network className="w-10 h-10 text-blue-500 mb-3" />
                  <h4 className="font-semibold mb-2">딥러닝</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    신경망 구조 설계와 학습 과정 시각화
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
                  <BarChart3 className="w-10 h-10 text-green-500 mb-3" />
                  <h4 className="font-semibold mb-2">통계 분석</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    가설 검정, 회귀 분석 등 통계 기법 실습
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}