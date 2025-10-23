'use client'

import { useState } from 'react'
import Link from 'next/link'
import {
  Play, Clock, BookOpen, Link as LinkIcon, Zap, Users,
  ChevronRight, Award, ArrowRight, GraduationCap, Trophy,
  Sparkles, Code, Settings, TrendingUp
} from 'lucide-react'
import { langchainModule } from './metadata'

interface LearningPath {
  id: string
  level: string
  title: string
  description: string
  icon: React.ReactNode
  color: string
  duration: string
  topics: string[]
  chapters: string[]
  prerequisites?: string[]
}

export default function LangChainMainPage() {
  const [completedChapters, setCompletedChapters] = useState<string[]>([])

  const progress = (completedChapters.length / langchainModule.chapters.length) * 100

  const learningPaths: LearningPath[] = [
    {
      id: 'beginner',
      level: 'Step 1: 초급',
      title: 'LangChain 기초',
      description: 'LangChain의 기본 개념과 핵심 컴포넌트를 학습합니다',
      icon: <GraduationCap size={24} />,
      color: 'from-amber-500 to-orange-600',
      duration: '8시간',
      topics: [
        'LangChain 아키텍처',
        'Chains와 Prompts',
        'Memory 시스템',
        '기본 예제 실습'
      ],
      chapters: ['01-langchain-basics', '02-chains-prompts', '03-memory-context']
    },
    {
      id: 'intermediate',
      level: 'Step 2: 중급',
      title: 'Agents와 Tools',
      description: 'AI 에이전트를 구축하고 외부 도구를 통합합니다',
      icon: <Zap size={24} />,
      color: 'from-blue-500 to-indigo-600',
      duration: '10시간',
      topics: [
        'Agent 작동 원리',
        'Tool Integration',
        'Custom Tool 개발',
        'Agent 최적화'
      ],
      chapters: ['04-agents-tools'],
      prerequisites: ['Step 1 완료']
    },
    {
      id: 'advanced',
      level: 'Step 3: 고급',
      title: 'LangGraph 마스터',
      description: 'State Graph로 복잡한 워크플로우를 설계합니다',
      icon: <Trophy size={24} />,
      color: 'from-purple-500 to-pink-600',
      duration: '12시간',
      topics: [
        'LangGraph 설계',
        '복잡한 워크플로우',
        '멀티 에이전트',
        'Human-in-the-loop'
      ],
      chapters: ['05-langgraph-intro', '06-complex-workflows'],
      prerequisites: ['Step 2 완료']
    },
    {
      id: 'production',
      level: 'Step 4: 프로덕션',
      title: '실전 배포',
      description: '실제 서비스에 적용 가능한 애플리케이션을 구축합니다',
      icon: <Award size={24} />,
      color: 'from-green-500 to-emerald-600',
      duration: '14시간',
      topics: [
        'LangSmith 모니터링',
        'LangServe 배포',
        '성능 최적화',
        '실전 프로젝트'
      ],
      chapters: ['07-production-deployment', '08-real-world-projects'],
      prerequisites: ['Step 3 완료']
    }
  ]

  const simulators = [
    {
      id: 'chain-builder',
      name: 'Chain Builder',
      description: '드래그앤드롭으로 체인 구성 시각화',
      path: '/modules/langchain/simulators/chain-builder',
      icon: <LinkIcon className="text-amber-500" />
    },
    {
      id: 'prompt-optimizer',
      name: 'Prompt Optimizer',
      description: '프롬프트 엔지니어링 플레이그라운드',
      path: '/modules/langchain/simulators/prompt-optimizer',
      icon: <Code className="text-blue-500" />
    },
    {
      id: 'memory-manager',
      name: 'Memory Manager',
      description: '메모리 시스템 비교 및 시각화',
      path: '/modules/langchain/simulators/memory-manager',
      icon: <Settings className="text-purple-500" />
    },
    {
      id: 'agent-debugger',
      name: 'Agent Debugger',
      description: 'Agent 실행 과정 단계별 추적',
      path: '/modules/langchain/simulators/agent-debugger',
      icon: <Zap className="text-orange-500" />
    },
    {
      id: 'langgraph-designer',
      name: 'LangGraph Designer',
      description: '그래프 워크플로우 시각적 설계',
      path: '/modules/langchain/simulators/langgraph-designer',
      icon: <Sparkles className="text-pink-500" />
    },
    {
      id: 'tool-integrator',
      name: 'Tool Integrator',
      description: 'Custom Tool 빌더와 테스트',
      path: '/modules/langchain/simulators/tool-integrator',
      icon: <Award className="text-indigo-500" />
    },
    {
      id: 'multi-agent-coordinator',
      name: 'Multi-Agent Coordinator',
      description: '여러 AI 에이전트의 협업 시뮬레이션',
      path: '/modules/langchain/simulators/multi-agent-coordinator',
      icon: <Users className="text-green-500" />
    }
  ]

  const quickStats = [
    { label: '학습 시간', value: `${langchainModule.estimatedHours}시간`, icon: <Clock size={20} /> },
    { label: '챕터 수', value: `${langchainModule.chapters.length}개`, icon: <BookOpen size={20} /> },
    { label: '시뮬레이터', value: `${langchainModule.simulators.length}개`, icon: <Play size={20} /> },
    { label: '난이도', value: '중급', icon: <TrendingUp size={20} /> }
  ]

  return (
    <div className="space-y-12">
      {/* Quick Navigation */}
      <nav className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-4 mb-8">
        <div className="flex flex-wrap justify-center gap-4">
          <a href="#learning-paths" className="text-sm font-medium text-amber-600 dark:text-amber-400 hover:underline">
            학습 경로
          </a>
          <span className="text-gray-400">•</span>
          <a href="#simulators" className="text-sm font-medium text-amber-600 dark:text-amber-400 hover:underline">
            시뮬레이터
          </a>
          <span className="text-gray-400">•</span>
          <a href="#resources" className="text-sm font-medium text-amber-600 dark:text-amber-400 hover:underline">
            학습 자료
          </a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="text-center py-16 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-amber-100/50 to-orange-100/50 dark:from-amber-900/20 dark:to-orange-900/20 -z-10"></div>

        <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-white text-4xl mb-6 shadow-lg">
          {langchainModule.icon}
        </div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          {langchainModule.nameKo}
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
          {langchainModule.description}
        </p>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto mb-8">
          {quickStats.map((stat, index) => (
            <div key={index} className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
              <div className="flex items-center justify-center gap-2 text-amber-600 dark:text-amber-400 mb-1">
                {stat.icon}
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {stat.value}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {stat.label}
              </div>
            </div>
          ))}
        </div>

        {/* Progress */}
        <div className="max-w-md mx-auto mb-8">
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>전체 학습 진도</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-amber-500 to-orange-600 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>
      </section>

      {/* Learning Path Selection */}
      <section id="learning-paths" className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <GraduationCap className="text-amber-500" size={24} />
          학습 경로 선택
        </h2>

        <p className="text-gray-600 dark:text-gray-400 text-center mb-8">
          LangChain과 LangGraph를 체계적으로 학습할 수 있는 4단계 과정을 제공합니다.
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {learningPaths.map((path) => (
            <div
              key={path.id}
              className="group relative p-6 rounded-xl border-2 transition-all duration-200 text-left border-gray-200 dark:border-gray-700 hover:border-amber-300 dark:hover:border-amber-600 hover:shadow-lg cursor-pointer"
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${path.color} flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform duration-200`}>
                {path.icon}
              </div>

              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-1">
                {path.level}
              </h3>
              <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                {path.title}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                {path.description}
              </p>

              <div className="space-y-2 mb-4">
                {path.topics.slice(0, 3).map((topic, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                    <div className="w-1 h-1 rounded-full bg-amber-500"></div>
                    <span>{topic}</span>
                  </div>
                ))}
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                  <Clock size={14} />
                  <span>{path.duration}</span>
                </div>
                <ArrowRight size={16} className="text-amber-500 group-hover:translate-x-1 transition-transform duration-200" />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Chapter List */}
      <section className="bg-gray-50 dark:bg-gray-800/50 rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="text-amber-500" size={24} />
          전체 챕터
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {langchainModule.chapters.map((chapter, index) => (
            <Link
              key={chapter.id}
              href={`/modules/langchain/${chapter.id}`}
              className="group bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-200 border border-gray-200 dark:border-gray-700 hover:border-amber-300 dark:hover:border-amber-600"
            >
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-white font-bold flex-shrink-0">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                    {chapter.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    {chapter.description}
                  </p>
                  <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock size={12} />
                      <span>{chapter.estimatedMinutes}분</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span>{chapter.keywords.length}개 키워드</span>
                    </div>
                  </div>
                </div>
                <ChevronRight size={20} className="text-gray-400 group-hover:text-amber-500 transition-colors duration-200 flex-shrink-0" />
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Simulators Grid */}
      <section id="simulators" className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Sparkles className="text-amber-500" size={24} />
          인터랙티브 시뮬레이터
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={simulator.path}
              className="group bg-gray-50 dark:bg-gray-900 rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-200 border border-gray-200 dark:border-gray-700 hover:border-amber-300 dark:hover:border-amber-600"
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-white dark:bg-gray-800 flex items-center justify-center group-hover:scale-110 transition-transform duration-200 shadow-sm">
                  {simulator.icon}
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                    {simulator.name}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {simulator.description}
                  </p>
                </div>
                <ChevronRight size={20} className="text-gray-400 group-hover:text-amber-500 transition-colors duration-200" />
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Resources & Community */}
      <section id="resources" className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Users className="text-amber-500" size={24} />
          학습 자료 & 커뮤니티
        </h2>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-amber-800 dark:text-amber-200 mb-3">공식 문서</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• LangChain Documentation</li>
              <li>• LangGraph Guide</li>
              <li>• LangSmith Platform</li>
              <li>• API Reference</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">실습 자료</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 예제 코드 저장소</li>
              <li>• 프로젝트 템플릿</li>
              <li>• 비디오 튜토리얼</li>
              <li>• 커뮤니티 프로젝트</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-3">지원</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 전문가 멘토링</li>
              <li>• 실시간 Q&A</li>
              <li>• 코드 리뷰</li>
              <li>• 취업 연계</li>
            </ul>
          </div>
        </div>

        <div className="mt-8 text-center">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            LangChain 전문가가 되는 여정, 지금 시작하세요!
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="px-8 py-3 bg-gradient-to-r from-amber-500 to-orange-600 text-white rounded-lg font-medium hover:shadow-lg transition-all duration-200">
              학습 시작하기
            </button>
            <button className="px-8 py-3 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:shadow-lg transition-all duration-200 border border-gray-200 dark:border-gray-700">
              데모 보기
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}
