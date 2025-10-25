'use client'

import { useState } from 'react'
import Link from 'next/link'
import {
  Play, Clock, BookOpen, Code, Zap, Users,
  ChevronRight, Award, ArrowRight, GraduationCap, Trophy,
  Sparkles, Settings, TrendingUp, Wand2, Shield, Rocket
} from 'lucide-react'
import { vibeCodingMetadata } from './metadata'

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

export default function VibeCodingMainPage() {
  const [completedChapters, setCompletedChapters] = useState<string[]>([])

  const progress = (completedChapters.length / vibeCodingMetadata.chapters.length) * 100

  const learningPaths: LearningPath[] = [
    {
      id: 'beginner',
      level: 'Step 1: 초급',
      title: 'AI 코딩 기초',
      description: 'Cursor, Copilot, Claude Code 등 AI 코딩 도구의 핵심 기능을 학습합니다',
      icon: <GraduationCap size={24} />,
      color: 'from-purple-500 to-pink-600',
      duration: '11시간',
      topics: [
        'AI 코딩 혁명의 이해',
        'Cursor IDE 마스터',
        'GitHub Copilot 활용',
        'Claude Code 엔지니어링'
      ],
      chapters: ['ai-coding-revolution', 'cursor-mastery', 'github-copilot']
    },
    {
      id: 'intermediate',
      level: 'Step 2: 중급',
      title: '고급 AI 코딩 기법',
      description: '프롬프트 엔지니어링, 테스트 자동화, AI 코드 리뷰 시스템을 구축합니다',
      icon: <Zap size={24} />,
      color: 'from-blue-500 to-indigo-600',
      duration: '17시간',
      topics: [
        'AI 프롬프트 엔지니어링',
        'AI 테스트 자동 생성',
        'AI 코드 리뷰 시스템',
        'AI 리팩토링 전략'
      ],
      chapters: ['claude-code-engineering', 'prompt-engineering', 'ai-test-generation', 'ai-code-review'],
      prerequisites: ['Step 1 완료']
    },
    {
      id: 'advanced',
      level: 'Step 3: 고급',
      title: '자동화 & 보안',
      description: 'AI 기반 문서화, 워크플로우 자동화, 보안 모범 사례를 마스터합니다',
      icon: <Trophy size={24} />,
      color: 'from-emerald-500 to-teal-600',
      duration: '16시간',
      topics: [
        'AI 대규모 리팩토링',
        'AI 자동 문서화',
        'CI/CD 워크플로우 자동화',
        'AI 코딩 보안'
      ],
      chapters: ['ai-refactoring', 'ai-documentation', 'ai-workflow-automation', 'ai-security-practices'],
      prerequisites: ['Step 2 완료']
    },
    {
      id: 'production',
      level: 'Step 4: 실전',
      title: '48시간 풀스택 앱',
      description: 'AI 도구만으로 기획부터 배포까지 완전한 풀스택 애플리케이션을 구축합니다',
      icon: <Award size={24} />,
      color: 'from-orange-500 to-red-600',
      duration: '8시간',
      topics: [
        'AI 프로젝트 기획',
        'Next.js + TypeScript 앱',
        'Database & API 자동 설계',
        'UI/UX to Code 변환',
        '자동 테스트 & 배포'
      ],
      chapters: ['real-world-projects'],
      prerequisites: ['Step 3 완료']
    }
  ]

  const simulators = [
    {
      id: 'ai-code-assistant',
      name: 'AI Code Assistant Playground',
      description: 'Cursor, Copilot, Claude 스타일 AI 코딩 체험',
      path: '/modules/vibe-coding/simulators/ai-code-assistant',
      icon: <Wand2 className="text-purple-500" />
    },
    {
      id: 'prompt-optimizer',
      name: 'Prompt Optimizer',
      description: 'AI 코딩 프롬프트 분석 및 최적화',
      path: '/modules/vibe-coding/simulators/prompt-optimizer',
      icon: <Code className="text-blue-500" />
    },
    {
      id: 'code-review-ai',
      name: 'AI Code Reviewer',
      description: '보안, 성능, 스타일 자동 체크',
      path: '/modules/vibe-coding/simulators/code-review-ai',
      icon: <Shield className="text-pink-500" />
    },
    {
      id: 'refactoring-engine',
      name: 'AI Refactoring Engine',
      description: '레거시 코드 자동 현대화',
      path: '/modules/vibe-coding/simulators/refactoring-engine',
      icon: <Zap className="text-orange-500" />
    },
    {
      id: 'test-generator',
      name: 'AI Test Generator',
      description: '단위/통합 테스트 자동 생성',
      path: '/modules/vibe-coding/simulators/test-generator',
      icon: <Settings className="text-indigo-500" />
    },
    {
      id: 'doc-generator',
      name: 'AI Documentation Generator',
      description: 'README, API 문서 자동 생성',
      path: '/modules/vibe-coding/simulators/doc-generator',
      icon: <BookOpen className="text-green-500" />
    }
  ]

  const quickStats = [
    { label: '학습 시간', value: '48시간', icon: <Clock size={20} /> },
    { label: '챕터 수', value: `${vibeCodingMetadata.chapters.length}개`, icon: <BookOpen size={20} /> },
    { label: '시뮬레이터', value: `${vibeCodingMetadata.simulators.length}개`, icon: <Play size={20} /> },
    { label: '난이도', value: '전체 레벨', icon: <TrendingUp size={20} /> }
  ]

  return (
    <div className="space-y-12">
      {/* Quick Navigation */}
      <nav className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-4 mb-8">
        <div className="flex flex-wrap justify-center gap-4">
          <a href="#learning-paths" className="text-sm font-medium text-purple-600 dark:text-purple-400 hover:underline">
            학습 경로
          </a>
          <span className="text-gray-400">•</span>
          <a href="#simulators" className="text-sm font-medium text-purple-600 dark:text-purple-400 hover:underline">
            시뮬레이터
          </a>
          <span className="text-gray-400">•</span>
          <a href="#resources" className="text-sm font-medium text-purple-600 dark:text-purple-400 hover:underline">
            학습 자료
          </a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="text-center py-16 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-100/50 to-pink-100/50 dark:from-purple-900/20 dark:to-pink-900/20 -z-10"></div>

        <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center text-white text-4xl mb-6 shadow-lg">
          <Wand2 size={40} />
        </div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Vibe Coding with AI
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
          {vibeCodingMetadata.description}
        </p>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto mb-8">
          {quickStats.map((stat, index) => (
            <div key={index} className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
              <div className="flex items-center justify-center gap-2 text-purple-600 dark:text-purple-400 mb-1">
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
              className="bg-gradient-to-r from-purple-500 to-pink-600 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>
      </section>

      {/* Learning Path Selection */}
      <section id="learning-paths" className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <GraduationCap className="text-purple-500" size={24} />
          학습 경로 선택
        </h2>

        <p className="text-gray-600 dark:text-gray-400 text-center mb-8">
          AI 코딩 도구를 체계적으로 마스터할 수 있는 4단계 과정을 제공합니다. Cursor부터 Claude Code까지 개발 생산성을 10배 향상시키세요.
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {learningPaths.map((path) => (
            <div
              key={path.id}
              className="group relative p-6 rounded-xl border-2 transition-all duration-200 text-left border-gray-200 dark:border-gray-700 hover:border-purple-300 dark:hover:border-purple-600 hover:shadow-lg cursor-pointer"
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
                    <div className="w-1 h-1 rounded-full bg-purple-500"></div>
                    <span>{topic}</span>
                  </div>
                ))}
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                  <Clock size={14} />
                  <span>{path.duration}</span>
                </div>
                <ArrowRight size={16} className="text-purple-500 group-hover:translate-x-1 transition-transform duration-200" />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Chapter List */}
      <section className="bg-gray-50 dark:bg-gray-800/50 rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="text-purple-500" size={24} />
          전체 챕터
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {vibeCodingMetadata.chapters.map((chapter, index) => (
            <Link
              key={chapter.id}
              href={`/modules/vibe-coding/${chapter.id}`}
              className="group bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-200 border border-gray-200 dark:border-gray-700 hover:border-purple-300 dark:hover:border-purple-600"
            >
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center text-white font-bold flex-shrink-0">
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
                      <span>{chapter.duration}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span>{chapter.topics.length}개 주제</span>
                    </div>
                  </div>
                </div>
                <ChevronRight size={20} className="text-gray-400 group-hover:text-purple-500 transition-colors duration-200 flex-shrink-0" />
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Simulators Grid */}
      <section id="simulators" className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Sparkles className="text-purple-500" size={24} />
          인터랙티브 시뮬레이터
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={simulator.path}
              className="group bg-gray-50 dark:bg-gray-900 rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-200 border border-gray-200 dark:border-gray-700 hover:border-purple-300 dark:hover:border-purple-600"
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
                <ChevronRight size={20} className="text-gray-400 group-hover:text-purple-500 transition-colors duration-200" />
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Resources & Community */}
      <section id="resources" className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Users className="text-purple-500" size={24} />
          학습 자료 & 커뮤니티
        </h2>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-3">AI 코딩 도구</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• Cursor IDE 공식 가이드</li>
              <li>• GitHub Copilot 문서</li>
              <li>• Claude Code API</li>
              <li>• VSCode Extensions</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">실습 자료</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 예제 코드 저장소</li>
              <li>• 프롬프트 라이브러리</li>
              <li>• 실전 프로젝트 템플릿</li>
              <li>• 비디오 튜토리얼</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <h3 className="font-bold text-pink-800 dark:text-pink-200 mb-3">지원</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• AI 코딩 전문가 멘토링</li>
              <li>• 실시간 Q&A</li>
              <li>• 코드 리뷰 세션</li>
              <li>• 커리어 컨설팅</li>
            </ul>
          </div>
        </div>

        <div className="mt-8 text-center">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            AI 코딩으로 개발 생산성을 10배 향상시키는 여정, 지금 시작하세요!
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="px-8 py-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg font-medium hover:shadow-lg transition-all duration-200">
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
