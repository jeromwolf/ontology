'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  BarChart3, Play, Clock, Target, ChevronRight, 
  BookOpen, Sparkles, TrendingUp, Brain,
  Dice1, BarChart2, FlaskConical, Calculator
} from 'lucide-react'
import { probabilityStatisticsModule } from './metadata'
import Navigation from '@/components/Navigation'
import ModuleProgress from '@/components/ModuleProgress'

export default function ProbabilityStatisticsPage() {
  const [progress, setProgress] = useState(0)
  const [startedChapters, setStartedChapters] = useState<string[]>([])

  useEffect(() => {
    const savedProgress = localStorage.getItem(`module-progress-${probabilityStatisticsModule.id}`)
    if (savedProgress) {
      const { chapters } = JSON.parse(savedProgress)
      setStartedChapters(chapters)
      setProgress((chapters.length / probabilityStatisticsModule.chapters.length) * 100)
    }
  }, [])

  const getChapterIcon = (chapterId: string) => {
    const icons: { [key: string]: JSX.Element } = {
      'probability-basics': <Dice1 className="w-5 h-5" />,
      'distributions': <BarChart2 className="w-5 h-5" />,
      'descriptive-statistics': <BarChart3 className="w-5 h-5" />,
      'inferential-statistics': <FlaskConical className="w-5 h-5" />,
      'bayesian-statistics': <Brain className="w-5 h-5" />,
      'regression-analysis': <TrendingUp className="w-5 h-5" />,
      'time-series': <Sparkles className="w-5 h-5" />,
      'ml-statistics': <Calculator className="w-5 h-5" />
    }
    return icons[chapterId] || <BookOpen className="w-5 h-5" />
  }

  const getSimulatorIcon = (simulatorId: string) => {
    const icons: { [key: string]: JSX.Element } = {
      'probability-playground': <Dice1 className="w-5 h-5" />,
      'distribution-visualizer': <BarChart2 className="w-5 h-5" />,
      'hypothesis-tester': <FlaskConical className="w-5 h-5" />,
      'regression-lab': <TrendingUp className="w-5 h-5" />,
      'monte-carlo': <Brain className="w-5 h-5" />
    }
    return icons[simulatorId] || <Play className="w-5 h-5" />
  }

  return (
    <div className="min-h-screen">
      <Navigation />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-4 mb-6">
            <div className="p-3 rounded-xl bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800 border">
              <BarChart3 className="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
                {probabilityStatisticsModule.name}
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-400 mt-2">
                {probabilityStatisticsModule.description}
              </p>
            </div>
          </div>

          <ModuleProgress 
            completedChapters={startedChapters.length}
            totalChapters={probabilityStatisticsModule.chapters.length}
            completedSimulators={0}
            totalSimulators={probabilityStatisticsModule.simulators.length}
          />

          <div className="flex flex-wrap gap-4 mt-6">
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
              <Clock className="w-5 h-5" />
              <span>{probabilityStatisticsModule.estimatedHours}시간</span>
            </div>
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
              <Target className="w-5 h-5" />
              <span>난이도: 중급</span>
            </div>
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
              <BookOpen className="w-5 h-5" />
              <span>{probabilityStatisticsModule.chapters.length}개 챕터</span>
            </div>
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
              <Sparkles className="w-5 h-5" />
              <span>{probabilityStatisticsModule.simulators.length}개 시뮬레이터</span>
            </div>
          </div>
        </div>

        {/* Learning Objectives */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            학습 목표
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
{[
              '확률과 통계의 기본 개념을 이해하고 실제 문제에 적용',
              '다양한 확률분포의 특성을 파악하고 상황에 맞게 선택',
              '통계적 추론을 통한 가설 검정과 신뢰구간 계산',
              '베이지안 통계의 원리와 MCMC 샘플링 방법 이해',
              '회귀분석과 시계열 분석을 통한 예측 모델 구축',
              '머신러닝을 위한 통계적 개념과 검증 방법론 습득'
            ].map((objective, index) => (
              <div 
                key={index}
                className="flex items-start gap-3 p-4 rounded-xl bg-white dark:bg-gray-800 border border-purple-200 dark:border-purple-700"
              >
                <div className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-xs font-semibold text-purple-600 dark:text-purple-400">
                    {index + 1}
                  </span>
                </div>
                <p className="text-gray-700 dark:text-gray-300">{objective}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Chapters */}
          <div className="lg:col-span-2">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              챕터 목록
            </h2>
            <div className="space-y-4">
              {probabilityStatisticsModule.chapters.map((chapter, index) => {
                const isStarted = startedChapters.includes(chapter.id)
                return (
                  <Link
                    key={chapter.id}
                    href={`/modules/probability-statistics/${chapter.id}`}
                    className={`
                      block p-6 rounded-xl border transition-all duration-200
                      ${isStarted 
                        ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-300 dark:border-purple-700' 
                        : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'
                      }
                      hover:shadow-lg hover:scale-[1.02] hover:border-purple-400 dark:hover:border-purple-600
                    `}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4">
                        <div className={`
                          p-2 rounded-lg
                          ${isStarted 
                            ? 'bg-purple-200 dark:bg-purple-800' 
                            : 'bg-purple-100 dark:bg-purple-900'
                          }
                        `}>
                          {getChapterIcon(chapter.id)}
                        </div>
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                            Chapter {index + 1}: {chapter.title}
                          </h3>
                          <p className="text-gray-600 dark:text-gray-400">
                            {chapter.description}
                          </p>
                          {isStarted && (
                            <div className="mt-2 inline-flex items-center gap-1 text-sm text-purple-600 dark:text-purple-400">
                              <Play className="w-4 h-4" />
                              학습 중
                            </div>
                          )}
                        </div>
                      </div>
                      <ChevronRight className="w-5 h-5 text-gray-400 dark:text-gray-600 flex-shrink-0" />
                    </div>
                  </Link>
                )
              })}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-8">
            {/* Simulators */}
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                시뮬레이터
              </h2>
              <div className="space-y-3">
                {probabilityStatisticsModule.simulators.map((simulator) => (
                  <Link
                    key={simulator.id}
                    href={`/modules/probability-statistics/simulators/${simulator.id}`}
                    className="block p-4 rounded-xl bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30 border border-purple-200 dark:border-purple-700 hover:shadow-lg transition-all duration-200 hover:scale-[1.02]"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-white dark:bg-gray-800">
                        {getSimulatorIcon(simulator.id)}
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900 dark:text-white">
                          {simulator.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {simulator.description}
                        </p>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>

            {/* Career Paths */}
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                진로 방향
              </h2>
              <div className="space-y-3">
{[
                  { title: '데이터 사이언티스트', description: '빅데이터 분석을 통한 인사이트 도출 및 예측 모델 구축' },
                  { title: '머신러닝 엔지니어', description: 'AI 모델 개발과 배포, 통계적 기법을 활용한 모델 최적화' },
                  { title: '퀀트 애널리스트', description: '금융 상품 개발, 위험 관리, 알고리즘 트레이딩 전략 수립' },
                  { title: '바이오통계학자', description: '임상시험 설계, 의학 연구 데이터 분석 및 해석' },
                  { title: '품질관리 전문가', description: '통계적 품질관리, 식스시그마, 공정 최적화' }
                ].map((career, index) => (
                  <div 
                    key={index}
                    className="p-4 rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700"
                  >
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                      {career.title}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {career.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Tools */}
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                추천 도구
              </h2>
              <div className="space-y-3">
                {probabilityStatisticsModule.tools.map((tool, index) => (
                  <a
                    key={index}
                    href={tool.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block p-4 rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-purple-400 dark:hover:border-purple-600 transition-colors"
                  >
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                      {tool.name}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {tool.description}
                    </p>
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}