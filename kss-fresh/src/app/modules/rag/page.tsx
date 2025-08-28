'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Play, Clock, BookOpen, FileText, Search, Database, 
  Sparkles, CheckCircle2, GraduationCap, Trophy, Zap, Users,
  ChevronRight, Award, ArrowRight
} from 'lucide-react'
import { ragModule } from './metadata'

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
  simulators: { name: string; description: string }[]
  prerequisites?: string[]
}

export default function RAGMainPage() {
  const [completedChapters, setCompletedChapters] = useState<string[]>([])
  
  const progress = (completedChapters.length / ragModule.chapters.length) * 100

  const learningPaths: LearningPath[] = [
    {
      id: 'beginner',
      level: 'Step 1: 초급',
      title: 'RAG 기본 개념 이해',
      description: 'RAG의 필요성과 기본 작동 원리를 학습합니다',
      icon: <GraduationCap size={24} />,
      color: 'from-green-500 to-emerald-600',
      duration: '10시간',
      topics: [
        'LLM의 한계점',
        'RAG란 무엇인가',
        '기본 파이프라인 이해',
        '간단한 예제 실습'
      ],
      chapters: ['01-what-is-rag', '02-document-processing'],
      simulators: [
        { name: 'Document Uploader', description: '문서 업로드 체험' },
        { name: 'Chunking Demo', description: '청킹 전략 비교' }
      ]
    },
    {
      id: 'intermediate',
      level: 'Step 2: 중급',
      title: '핵심 컴포넌트 마스터',
      description: '임베딩, 벡터 검색 등 핵심 기술을 심화 학습합니다',
      icon: <Zap size={24} />,
      color: 'from-blue-500 to-indigo-600',
      duration: '15시간',
      topics: [
        '임베딩 모델 이해',
        '벡터 데이터베이스',
        '검색 알고리즘',
        '성능 최적화'
      ],
      chapters: ['03-embeddings', '04-vector-search', '05-answer-generation'],
      simulators: [
        { name: 'Embedding Visualizer', description: '임베딩 시각화' },
        { name: 'Vector Search Demo', description: '검색 방식 비교' }
      ],
      prerequisites: ['Step 1 완료']
    },
    {
      id: 'advanced',
      level: 'Step 3: 고급',
      title: '프로덕션 레벨 구현',
      description: '실제 서비스에 적용 가능한 고급 기법을 마스터합니다',
      icon: <Trophy size={24} />,
      color: 'from-purple-500 to-pink-600',
      duration: '20시간',
      topics: [
        'GraphRAG 아키텍처',
        '하이브리드 검색',
        '프롬프트 엔지니어링',
        '대규모 시스템 설계'
      ],
      chapters: ['06-advanced-rag'],
      simulators: [
        { name: 'GraphRAG Explorer', description: '그래프 기반 RAG' },
        { name: 'RAG Playground', description: '전체 파이프라인' }
      ],
      prerequisites: ['Step 2 완료']
    },
    {
      id: 'supplementary',
      level: 'Step 4: 보충',
      title: '실무 필수 요소',
      description: '프로덕션 환경에서 필요한 평가, 보안, 비용 최적화를 학습합니다',
      icon: <Award size={24} />,
      color: 'from-amber-500 to-orange-600',
      duration: '8시간',
      topics: [
        'RAG 평가 및 품질 관리',
        '보안 및 프라이버시',
        '비용 최적화 전략',
        '실패 처리 및 복구'
      ],
      chapters: ['07-evaluation', '08-security', '09-optimization'],
      simulators: [
        { name: 'RAGAS Evaluator', description: 'RAG 성능 평가 도구' },
        { name: 'Cost Monitor', description: '비용 분석 대시보드' }
      ],
      prerequisites: ['Step 3 완료']
    }
  ]


  // 시뮬레이터 정보
  const simulators = [
    {
      id: 'document-uploader',
      name: '문서 업로더',
      description: 'PDF, Word, HTML 문서를 업로드하고 처리 과정 체험',
      path: '/modules/rag/simulators/document-uploader',
      icon: <FileText className="text-emerald-500" />
    },
    {
      id: 'chunking-demo',
      name: '청킹 데모',
      description: '5가지 청킹 전략을 실시간으로 비교',
      path: '/modules/rag/simulators/chunking-demo',
      icon: <Database className="text-blue-500" />
    },
    {
      id: 'embedding-visualizer',
      name: '임베딩 시각화',
      description: '텍스트가 벡터 공간에서 표현되는 방식 시각화',
      path: '/modules/rag/simulators/embedding-visualizer',
      icon: <Search className="text-purple-500" />
    },
    {
      id: 'vector-search',
      name: '벡터 검색',
      description: '벡터, 키워드, 하이브리드 검색 비교',
      path: '/modules/rag/simulators/vector-search',
      icon: <Zap className="text-orange-500" />
    },
    {
      id: 'graphrag-explorer',
      name: 'GraphRAG 탐색기',
      description: '지식 그래프 기반 RAG 시스템',
      path: '/modules/rag/simulators/graphrag-explorer',
      icon: <Sparkles className="text-pink-500" />
    },
    {
      id: 'rag-playground',
      name: 'RAG 플레이그라운드',
      description: '전체 파이프라인 통합 체험',
      path: '/modules/rag/simulators/rag-playground',
      icon: <Award className="text-indigo-500" />
    }
  ]

  return (
    <div className="space-y-12">
      {/* Quick Navigation */}
      <nav className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-4 mb-8">
        <div className="flex flex-wrap justify-center gap-4">
          <a href="#learning-paths" className="text-sm font-medium text-emerald-600 dark:text-emerald-400 hover:underline">
            학습 경로
          </a>
          <span className="text-gray-400">•</span>
          <a href="#simulators" className="text-sm font-medium text-emerald-600 dark:text-emerald-400 hover:underline">
            시뮬레이터
          </a>
          <span className="text-gray-400">•</span>
          <a href="#community" className="text-sm font-medium text-emerald-600 dark:text-emerald-400 hover:underline">
            커뮤니티
          </a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="text-center py-16 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-100/50 to-green-100/50 dark:from-emerald-900/20 dark:to-green-900/20 -z-10"></div>
        
        <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center text-white text-4xl mb-6 shadow-lg">
          {ragModule.icon}
        </div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          {ragModule.nameKo}
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
          {ragModule.description}
        </p>
        
        {/* Progress */}
        <div className="max-w-md mx-auto mb-8">
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>전체 학습 진도</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-emerald-500 to-green-600 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>
      </section>

      {/* Learning Path Selection */}
      <section id="learning-paths" className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <GraduationCap className="text-emerald-500" size={24} />
          학습 경로 선택
        </h2>
        
        <p className="text-gray-600 dark:text-gray-400 text-center mb-8">
          RAG 기술을 체계적으로 학습할 수 있는 4단계 과정을 제공합니다. 각 레벨별로 상세한 커리큘럼을 확인하세요.
        </p>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {learningPaths.map((path) => (
            <Link
              key={path.id}
              href={`/modules/rag/${path.id}`}
              className="group relative p-6 rounded-xl border-2 transition-all duration-200 text-left border-gray-200 dark:border-gray-700 hover:border-emerald-300 dark:hover:border-emerald-600 hover:shadow-lg"
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
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                  <Clock size={14} />
                  <span>{path.duration}</span>
                </div>
                <ArrowRight size={16} className="text-emerald-500 group-hover:translate-x-1 transition-transform duration-200" />
              </div>
            </Link>
          ))}
        </div>
      </section>


      {/* Simulators Grid */}
      <section id="simulators" className="bg-gray-50 dark:bg-gray-800/50 rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Sparkles className="text-emerald-500" size={24} />
          시뮬레이터
        </h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={simulator.path}
              className="group bg-white dark:bg-gray-900 rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-200 border border-gray-200 dark:border-gray-700 hover:border-emerald-300 dark:hover:border-emerald-600"
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-gray-100 dark:bg-gray-800 flex items-center justify-center group-hover:scale-110 transition-transform duration-200">
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
                <ChevronRight size={20} className="text-gray-400 group-hover:text-emerald-500 transition-colors duration-200" />
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Community Section */}
      <section id="community" className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Users className="text-emerald-500" size={24} />
          커뮤니티 & 지원
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6">
            <h3 className="font-bold text-emerald-800 dark:text-emerald-200 mb-3">학습 지원</h3>
            <ul className="space-y-2 text-sm text-emerald-700 dark:text-emerald-300">
              <li>• 전문가 멘토링 제공</li>
              <li>• 실시간 Q&A 지원</li>
              <li>• 프로젝트 피드백</li>
              <li>• 취업 연계 프로그램</li>
            </ul>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
            <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">추가 리소스</h3>
            <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-300">
              <li>• 최신 RAG 논문 리뷰</li>
              <li>• 업계 동향 분석</li>
              <li>• 오픈소스 프로젝트 기여</li>
              <li>• 컨퍼런스 발표 기회</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6 text-center">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            RAG 전문가가 되는 여정, 함께 시작하세요!
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="px-6 py-3 bg-emerald-500 text-white rounded-lg font-medium hover:bg-emerald-600 transition-colors">
              멘토와 상담하기
            </button>
            <button className="px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors">
              커뮤니티 가입
            </button>
          </div>
        </div>
      </section>

    </div>
  )
}