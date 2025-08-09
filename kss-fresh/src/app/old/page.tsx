'use client'

import { useState } from 'react'
import { 
  Brain, TrendingUp, Network, Sparkles, 
  Atom, Cpu, Database, Globe, Car, Factory,
  ChevronRight, Star, Clock, Users, Zap, Server, Settings, Eye
} from 'lucide-react'
import Link from 'next/link'

interface Course {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
  color: string;
  category: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  duration: string;
  students?: number;
  rating?: number;
  status: 'active' | 'coming-soon' | 'planned';
  link?: string;
}

const courses: Course[] = [
  {
    id: 'llm',
    title: 'Large Language Models (LLM)',
    description: 'Transformer, GPT, Claude 등 최신 LLM 기술 완전 정복',
    icon: Cpu,
    color: 'from-indigo-500 to-purple-600',
    category: '인공지능',
    difficulty: 'intermediate',
    duration: '6주',
    students: 856,
    rating: 4.9,
    status: 'active',
    link: '/modules/llm'
  },
  {
    id: 'ontology',
    title: 'Ontology & Semantic Web',
    description: 'RDF, SPARQL, 지식 그래프를 통한 시맨틱 웹 기술 마스터',
    icon: Brain,
    color: 'from-purple-500 to-pink-500',
    category: '지식공학',
    difficulty: 'intermediate',
    duration: '8주',
    students: 1234,
    rating: 4.8,
    status: 'active',
    link: '/modules/ontology'
  },
  {
    id: 'stock-analysis',
    title: '주식투자분석 시뮬레이터',
    description: '실전 투자 전략과 심리까지 포함한 종합 투자 마스터 과정',
    icon: TrendingUp,
    color: 'from-red-500 to-orange-500',
    category: '금융',
    difficulty: 'beginner',
    duration: '16주',
    students: 2341,
    rating: 4.9,
    status: 'active',
    link: '/modules/stock-analysis'
  },
  {
    id: 'rag',
    title: 'RAG Systems',
    description: 'Retrieval-Augmented Generation 시스템 설계와 구현',
    icon: Database,
    color: 'from-emerald-500 to-green-600',
    category: 'AI/ML',
    difficulty: 'intermediate',
    duration: '12시간',
    students: 423,
    rating: 4.9,
    status: 'active',
    link: '/modules/rag'
  },
  {
    id: 'system-design',
    title: 'System Design',
    description: '대규모 분산 시스템 설계의 핵심 원칙과 실전 패턴을 학습합니다',
    icon: Server,
    color: 'from-purple-500 to-indigo-600',
    category: '시스템 설계',
    difficulty: 'advanced',
    duration: '20시간',
    students: 785,
    rating: 4.9,
    status: 'active',
    link: '/modules/system-design'
  },
  {
    id: 'agent-mcp',
    title: 'AI Agent & MCP',
    description: 'AI 에이전트 개발과 Model Context Protocol 마스터하기',
    icon: Zap,
    color: 'from-yellow-500 to-orange-600',
    category: 'AI/ML',
    difficulty: 'advanced',
    duration: '18시간',
    students: 445,
    rating: 4.9,
    status: 'active',
    link: '/modules/agent-mcp'
  },
  {
    id: 'multi-agent',
    title: 'Multi-Agent Systems',
    description: '여러 AI 에이전트가 협력하는 시스템 구축',
    icon: Network,
    color: 'from-orange-500 to-red-600',
    category: 'AI/ML',
    difficulty: 'advanced',
    duration: '22시간',
    students: 178,
    rating: 4.8,
    status: 'active',
    link: '/modules/multi-agent'
  },
  {
    id: 'web3',
    title: 'Web3 & Blockchain',
    description: '블록체인 기술과 Web3 생태계 완전 정복',
    icon: Globe,
    color: 'from-indigo-500 to-cyan-500',
    category: '블록체인',
    difficulty: 'intermediate',
    duration: '12시간',
    students: 567,
    rating: 4.8,
    status: 'active',
    link: '/modules/web3'
  },
  {
    id: 'computer-vision',
    title: 'Computer Vision',
    description: '이미지 처리부터 3D 변환까지',
    icon: Eye,
    color: 'from-teal-500 to-cyan-600',
    category: 'AI/ML',
    difficulty: 'intermediate',
    duration: '20시간',
    students: 320,
    rating: 4.9,
    status: 'active',
    link: '/modules/computer-vision'
  },
  {
    id: 'smart-factory',
    title: 'Smart Factory & Industry 4.0',
    description: '스마트 팩토리 구축과 산업 4.0 디지털 전환',
    icon: Factory,
    color: 'from-gray-500 to-gray-700',
    category: '제조업',
    difficulty: 'intermediate',
    duration: '14시간',
    students: 267,
    rating: 4.7,
    status: 'active',
    link: '/modules/smart-factory'
  },
  {
    id: 'quantum-computing',
    title: 'Quantum Computing',
    description: '양자 컴퓨팅과 알고리즘',
    icon: Atom,
    color: 'from-purple-500 to-violet-600',
    category: '양자',
    difficulty: 'advanced',
    duration: '24시간',
    students: 89,
    rating: 4.8,
    status: 'active',
    link: '/modules/quantum-computing'
  },
  {
    id: 'autonomous-mobility',
    title: '자율주행 & 미래 모빌리티',
    description: 'SAE 자율주행 레벨부터 UAM, 하이퍼루프까지',
    icon: Car,
    color: 'from-cyan-500 to-blue-600',
    category: '자율주행',
    difficulty: 'intermediate',
    duration: '16시간',
    students: 156,
    rating: 4.9,
    status: 'active',
    link: '/modules/autonomous-mobility'
  },
  {
    id: 'english-conversation',
    title: 'AI English Conversation',
    description: 'AI와 함께하는 실전 영어 회화 마스터',
    icon: Globe,
    color: 'from-blue-500 to-purple-600',
    category: '언어학습',
    difficulty: 'beginner',
    duration: '10시간',
    students: 892,
    rating: 4.8,
    status: 'active',
    link: '/modules/english-conversation'
  },
  {
    id: 'neo4j',
    title: 'Neo4j Graph Database',
    description: '그래프 데이터베이스 설계부터 Cypher 쿼리 최적화까지',
    icon: Network,
    color: 'from-green-500 to-emerald-600',
    category: '데이터베이스',
    difficulty: 'intermediate',
    duration: '20시간',
    students: 234,
    rating: 4.8,
    status: 'active',
    link: '/modules/neo4j'
  },
  {
    id: 'ai-security',
    title: 'AI Security & Privacy',
    description: 'AI 시스템 보안과 개인정보 보호 완전 가이드',
    icon: Settings,
    color: 'from-red-500 to-pink-600',
    category: '보안',
    difficulty: 'advanced',
    duration: '16시간',
    students: 234,
    rating: 4.9,
    status: 'active',
    link: '/modules/ai-security'
  }
]

export default function OldHomePage() {
  const [hoveredCourse, setHoveredCourse] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-indigo-900">
      {/* Hero Section with KSS Logo */}
      <div className="relative pt-20 pb-20 px-6">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="absolute top-0 right-0 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-cyan-400/20 rounded-full blur-3xl"></div>
        
        <div className="relative z-10 max-w-6xl mx-auto text-center">
          {/* KSS Logo */}
          <div className="mb-12">
            <div className="inline-flex items-center justify-center w-32 h-32 bg-white/10 backdrop-blur-sm rounded-3xl mb-6">
              <svg viewBox="0 0 100 100" className="w-20 h-20">
                <defs>
                  <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="50%" stopColor="#06B6D4" />
                    <stop offset="100%" stopColor="#10B981" />
                  </linearGradient>
                </defs>
                
                {/* Knowledge Graph Network */}
                <circle cx="20" cy="30" r="8" fill="url(#logoGradient)" opacity="0.8">
                  <animate attributeName="r" values="6;10;6" dur="3s" repeatCount="indefinite" />
                </circle>
                <circle cx="80" cy="30" r="6" fill="url(#logoGradient)" opacity="0.7">
                  <animate attributeName="r" values="4;8;4" dur="2s" repeatCount="indefinite" />
                </circle>
                <circle cx="50" cy="70" r="7" fill="url(#logoGradient)" opacity="0.9">
                  <animate attributeName="r" values="5;9;5" dur="2.5s" repeatCount="indefinite" />
                </circle>
                <circle cx="20" cy="70" r="5" fill="url(#logoGradient)" opacity="0.6">
                  <animate attributeName="r" values="3;7;3" dur="2.2s" repeatCount="indefinite" />
                </circle>
                
                {/* Connections */}
                <line x1="20" y1="30" x2="80" y2="30" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.6">
                  <animate attributeName="opacity" values="0.3;0.9;0.3" dur="2s" repeatCount="indefinite" />
                </line>
                <line x1="20" y1="30" x2="50" y2="70" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.5">
                  <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2.5s" repeatCount="indefinite" />
                </line>
                <line x1="50" y1="70" x2="20" y2="70" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.4">
                  <animate attributeName="opacity" values="0.1;0.7;0.1" dur="3s" repeatCount="indefinite" />
                </line>
                <line x1="80" y1="30" x2="50" y2="70" stroke="url(#logoGradient)" strokeWidth="2" opacity="0.3">
                  <animate attributeName="opacity" values="0.1;0.6;0.1" dur="2.8s" repeatCount="indefinite" />
                </line>
              </svg>
            </div>
            
            <h1 className="text-6xl md:text-8xl font-bold text-white mb-4 tracking-tight">
              <span className="bg-gradient-to-r from-purple-400 via-cyan-400 to-green-400 bg-clip-text text-transparent">
                KSS
              </span>
            </h1>
            <div className="text-2xl md:text-3xl text-white/90 mb-2 font-light">
              Knowledge Space Simulator
            </div>
            <div className="text-lg text-white/70">
              차세대 학습 혁신 플랫폼
            </div>
          </div>
          
          <p className="text-xl md:text-2xl text-white/90 mb-12 max-w-4xl mx-auto font-light leading-relaxed">
            복잡한 기술 개념을 3D 시뮬레이션으로 체험하고, AI와 함께 학습하는 지식 우주에 오신 것을 환영합니다.
          </p>

          <div className="flex flex-col sm:flex-row gap-6 justify-center mb-16">
            <Link href="/modules">
              <button className="px-8 py-4 bg-gradient-to-r from-purple-600 to-cyan-600 text-white rounded-xl text-lg font-semibold hover:from-purple-700 hover:to-cyan-700 transition-all transform hover:scale-105 shadow-lg">
                플랫폼 시작하기
              </button>
            </Link>
            <Link href="/3d-graph">
              <button className="px-8 py-4 bg-white/10 backdrop-blur-sm text-white rounded-xl text-lg font-semibold hover:bg-white/20 transition-all border border-white/20">
                3D 데모 체험
              </button>
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-4xl font-bold text-white mb-2">15+</div>
              <div className="text-white/70">전문 모듈</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-white mb-2">100+</div>
              <div className="text-white/70">학습 챕터</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-white mb-2">50+</div>
              <div className="text-white/70">시뮬레이터</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-white mb-2">8,000+</div>
              <div className="text-white/70">학습자</div>
            </div>
          </div>
        </div>
      </div>

      {/* Featured Modules */}
      <div className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-6">
              🚀 전문 AI 교육 플랫폼
            </h2>
            <p className="text-xl text-white/80 max-w-3xl mx-auto">
              실무에 바로 적용할 수 있는 시뮬레이션 기반 학습 경험을 제공합니다
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {courses.map((course) => {
              const Icon = course.icon
              
              return (
                <Link
                  key={course.id}
                  href={course.link || `/modules/${course.id}`}
                  className="group"
                  onMouseEnter={() => setHoveredCourse(course.id)}
                  onMouseLeave={() => setHoveredCourse(null)}
                >
                  <div className="relative bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:border-white/40 transition-all hover:bg-white/15 hover:transform hover:scale-105 h-full">
                    <div className={`w-14 h-14 rounded-xl bg-gradient-to-r ${course.color} flex items-center justify-center mb-4`}>
                      <Icon className="w-7 h-7 text-white" />
                    </div>
                    
                    <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-cyan-300 transition-colors">
                      {course.title}
                    </h3>
                    
                    <p className="text-sm text-white/70 mb-4 leading-relaxed">
                      {course.description}
                    </p>
                    
                    <div className="flex items-center justify-between text-xs text-white/60 mb-4">
                      <span className="px-2 py-1 bg-white/10 rounded">{course.category}</span>
                      <span>{course.duration}</span>
                    </div>
                    
                    {course.students && (
                      <div className="flex items-center justify-between text-xs text-white/60">
                        <div className="flex items-center">
                          <Users className="w-3 h-3 mr-1" />
                          <span>{course.students.toLocaleString()}명</span>
                        </div>
                        {course.rating && (
                          <div className="flex items-center">
                            <Star className="w-3 h-3 mr-1 fill-current text-yellow-400" />
                            <span>{course.rating}</span>
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                      <ChevronRight className="w-5 h-5 text-cyan-300" />
                    </div>
                  </div>
                </Link>
              )
            })}
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="py-20 px-6 bg-black/20">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mx-auto mb-6">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">
                실시간 시뮬레이션
              </h3>
              <p className="text-white/70 leading-relaxed">
                복잡한 개념을 직접 만지고 실험하며 학습하는 환경 제공
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-xl flex items-center justify-center mx-auto mb-6">
                <Network className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">
                3D 시각화
              </h3>
              <p className="text-white/70 leading-relaxed">
                추상적 개념을 3차원 공간에서 탐색하여 직관적 이해
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mx-auto mb-6">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">
                AI 기반 학습
              </h3>
              <p className="text-white/70 leading-relaxed">
                개인별 학습 속도와 스타일에 최적화된 AI 기반 커리큘럼
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-white mb-6">
            지금 바로 시작해보세요
          </h2>
          <p className="text-xl text-white/80 mb-12">
            미래의 AI 교육을 경험하고 전문가로 성장하세요
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/modules">
              <button className="px-8 py-4 bg-gradient-to-r from-purple-600 to-cyan-600 text-white rounded-xl text-lg font-semibold hover:from-purple-700 hover:to-cyan-700 transition-all transform hover:scale-105 shadow-lg">
                무료로 시작하기
              </button>
            </Link>
            <Link href="/about">
              <button className="px-8 py-4 bg-white/10 backdrop-blur-sm text-white rounded-xl text-lg font-semibold hover:bg-white/20 transition-all border border-white/20">
                더 알아보기
              </button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}