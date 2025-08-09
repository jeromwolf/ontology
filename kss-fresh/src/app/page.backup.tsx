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
    icon: Cpu,
    color: 'from-purple-500 to-purple-600',
    category: 'Agent/AI',
    difficulty: 'intermediate',
    duration: '10시간',
    students: 234,
    rating: 4.9,
    status: 'active',
    link: '/modules/agent-mcp'
  },
  {
    id: 'multi-agent',
    title: 'Multi-Agent Systems',
    description: '멀티 에이전트 시스템과 A2A 통신, 오케스트레이션',
    icon: Network,
    color: 'from-orange-500 to-orange-600',
    category: 'Agent/AI',
    difficulty: 'advanced',
    duration: '8시간',
    students: 89,
    rating: 4.8,
    status: 'active',
    link: '/modules/multi-agent'
  },
  {
    id: 'web3',
    title: 'Web3 & Blockchain',
    description: '블록체인 기술과 Web3 생태계를 체험하는 실전 학습 플랫폼',
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
    id: 'neo4j',
    title: 'Neo4j Knowledge Graph',
    description: '그래프 데이터베이스로 모든 지식을 연결하는 통합 지식 허브',
    icon: Network,
    color: 'from-blue-600 to-indigo-600',
    category: '지식공학',
    difficulty: 'intermediate',
    duration: '6주',
    status: 'active',
    link: '/modules/neo4j'
  },
  {
    id: 'autonomous-mobility',
    title: '자율주행 & 미래 모빌리티',
    description: 'AI 기반 자율주행 기술과 차세대 모빌리티 생태계 완전 정복',
    icon: Car,
    color: 'from-cyan-500 to-blue-600',
    category: '자율주행',
    difficulty: 'advanced',
    duration: '16시간',
    students: 112,
    rating: 4.9,
    status: 'active',
    link: '/modules/autonomous-mobility'
  },
  {
    id: 'medical-ai',
    title: 'Medical AI',
    description: '의료 영상 분석, 진단 보조, 신약 개발 AI 기술',
    icon: Brain,
    color: 'from-red-500 to-pink-500',
    category: '의료/바이오',
    difficulty: 'advanced',
    duration: '15시간',
    students: 234,
    rating: 4.8,
    status: 'active',
    link: '/medical-ai'
  },
  {
    id: 'physical-ai',
    title: 'Physical AI & 실세계 지능',
    description: '현실 세계와 상호작용하는 AI 시스템의 설계와 구현',
    icon: Cpu,
    color: 'from-slate-600 to-gray-700',
    category: '물리AI',
    difficulty: 'advanced',
    duration: '20시간',
    students: 156,
    rating: 4.9,
    status: 'active',
    link: '/modules/physical-ai'
  },
  {
    id: 'bioinformatics',
    title: 'Bio-informatics & Computational Biology',
    description: '유전체학, 단백질체학, 약물 설계를 위한 컴퓨터 생물학의 모든 것',
    icon: Sparkles,
    color: 'from-emerald-500 to-teal-600',
    category: '바이오/의료',
    difficulty: 'advanced',
    duration: '8주',
    students: 342,
    rating: 4.8,
    status: 'active',
    link: '/modules/bioinformatics'
  },
  {
    id: 'english-conversation',
    title: 'English Conversation Master',
    description: 'AI와 함께하는 실전 영어회화, 상황별 대화 연습과 발음 교정',
    icon: Globe,
    color: 'from-rose-500 to-pink-600',
    category: '언어/교육',
    difficulty: 'beginner',
    duration: '12주',
    students: 1247,
    rating: 4.9,
    status: 'active',
    link: '/modules/english-conversation'
  },
  {
    id: 'quantum-computing',
    title: 'Quantum Computing & 양자 알고리즘',
    description: '양자역학 기초부터 양자 머신러닝까지 차세대 컴퓨팅 기술 완전 정복',
    icon: Atom,
    color: 'from-purple-500 to-violet-600',
    category: '양자컴퓨팅',
    difficulty: 'advanced',
    duration: '24시간',
    students: 89,
    rating: 4.8,
    status: 'active',
    link: '/modules/quantum-computing'
  },
  {
    id: 'smart-factory',
    title: 'Smart Factory & Industry 4.0',
    description: '스마트 팩토리 자동화와 예측 유지보수, 디지털 트윈까지 산업 AI 기술 완전 정복',
    icon: Factory,
    color: 'from-amber-500 to-orange-600',
    category: '산업AI',
    difficulty: 'intermediate',
    duration: '20시간',
    status: 'active',
    link: '/modules/smart-factory'
  },
  {
    id: 'linear-algebra',
    title: 'Linear Algebra',
    description: 'AI를 위한 선형대수 - 벡터, 행렬, 고유값, SVD 완전 정복',
    icon: Sparkles,
    color: 'from-indigo-600 to-purple-600',
    category: '수학/이론',
    difficulty: 'intermediate',
    duration: '20시간',
    students: 342,
    rating: 4.9,
    status: 'active',
    link: '/linear-algebra'
  },
  {
    id: 'probability-statistics',
    title: 'Probability & Statistics',
    description: '머신러닝을 위한 확률론과 통계학',
    icon: TrendingUp,
    color: 'from-purple-500 to-pink-600',
    category: '수학/이론',
    difficulty: 'intermediate',
    duration: '8주',
    students: 423,
    rating: 4.9,
    status: 'active',
    link: '/modules/probability-statistics'
  },
  {
    id: 'optimization',
    title: 'Optimization Theory',
    description: '최적화 이론과 경사하강법, 메타휴리스틱',
    icon: Sparkles,
    color: 'from-violet-500 to-purple-600',
    category: '수학/이론',
    difficulty: 'advanced',
    duration: '8주',
    status: 'planned'
  },
  {
    id: 'distributed-computing',
    title: 'Distributed Computing',
    description: '분산 컴퓨팅과 병렬 처리, MapReduce',
    icon: Network,
    color: 'from-orange-500 to-red-600',
    category: '시스템/이론',
    difficulty: 'advanced',
    duration: '10주',
    status: 'planned'
  },
  {
    id: 'computer-vision',
    title: 'Computer Vision',
    description: '이미지 처리부터 2D to 3D 변환, 얼굴 인식까지 컴퓨터 비전의 모든 것',
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
    id: 'ai-security',
    title: 'AI 보안',
    description: 'AI 시스템의 보안 위협과 방어 기법을 학습합니다',
    icon: Zap,
    color: 'from-red-600 to-gray-700',
    category: '국방/보안',
    difficulty: 'advanced',
    duration: '8주',
    students: 156,
    rating: 4.8,
    status: 'active',
    link: '/modules/ai-security'
  },
  {
    id: 'ai-automation',
    title: '바이블코딩',
    description: 'Claude Code, Cursor, Windsurf 등 최신 AI 코딩 도구 마스터',
    icon: Cpu,
    color: 'from-violet-500 to-purple-600',
    category: 'AI/ML',
    difficulty: 'intermediate',
    duration: '8시간',
    status: 'active',
    link: '/modules/ai-automation'
  }
];

const categories = ['전체', '지식공학', 'AI/ML', '금융', 'Agent/AI', '물리컴퓨팅', '블록체인', '의료/바이오', '산업AI', '국방/보안', '수학/이론', '시스템/이론', '자율주행', '언어/교육'];

export default function Home() {
  const [selectedCategory, setSelectedCategory] = useState('전체');
  const [hoveredCourse, setHoveredCourse] = useState<string | null>(null);

  const filteredCourses = selectedCategory === '전체' 
    ? courses 
    : courses.filter(course => course.category === selectedCategory);

  const getDifficultyLabel = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '초급';
      case 'intermediate': return '중급';
      case 'advanced': return '고급';
      default: return '';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30';
      case 'intermediate': return 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/30';
      case 'advanced': return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30';
      default: return '';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <span className="text-xs px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded-full">학습 가능</span>;
      case 'coming-soon':
        return <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 rounded-full">곧 공개</span>;
      case 'planned':
        return <span className="text-xs px-2 py-1 bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-400 rounded-full">개발 예정</span>;
      default:
        return null;
    }
  };

  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-white dark:bg-gray-900">
        <div className="absolute inset-0 bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 opacity-50" />
        <div className="relative max-w-7xl mx-auto px-4 py-12">
          <div className="text-center">
            {/* Sophisticated KSS Logo with Knowledge Space Icon */}
            <div className="flex justify-center mb-8 pt-8">
              <div className="relative group cursor-pointer">
                {/* Multiple layer glow effects */}
                <div className="absolute inset-0 bg-gradient-to-r from-kss-primary/30 to-kss-secondary/30 rounded-3xl blur-3xl opacity-70 group-hover:opacity-100 transition-all duration-1000 animate-pulse"></div>
                <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 via-pink-600/20 to-blue-600/20 rounded-3xl blur-2xl opacity-50 group-hover:opacity-80 transition-all duration-700 rotate-180"></div>
                
                {/* Main logo container - Larger and more prominent */}
                <div className="relative bg-white/90 dark:bg-gray-900/90 backdrop-blur-xl px-20 py-16 rounded-3xl border-2 border-gray-200/30 dark:border-gray-700/30 shadow-2xl group-hover:shadow-[0_20px_60px_-15px_rgba(123,63,242,0.5)] transform group-hover:scale-110 transition-all duration-700">
                  
                  {/* Knowledge Space Visualization - Top - Larger */}
                  <div className="absolute -top-16 left-1/2 transform -translate-x-1/2">
                    <div className="relative w-32 h-32">
                      {/* Central node - Larger with ring */}
                      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                        <div className="w-8 h-8 bg-gradient-to-r from-kss-primary to-kss-secondary rounded-full shadow-2xl animate-pulse"></div>
                        <div className="absolute inset-0 w-8 h-8 bg-gradient-to-r from-kss-primary to-kss-secondary rounded-full animate-ping opacity-30"></div>
                      </div>
                      
                      {/* Orbiting nodes - Multiple rings */}
                      <div className="absolute inset-0 animate-spin" style={{animationDuration: '10s'}}>
                        <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full shadow-lg"></div>
                        <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full shadow-lg"></div>
                        <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-4 h-4 bg-gradient-to-r from-pink-400 to-rose-400 rounded-full shadow-lg"></div>
                        <div className="absolute right-0 top-1/2 transform -translate-y-1/2 w-4 h-4 bg-gradient-to-r from-orange-400 to-amber-400 rounded-full shadow-lg"></div>
                      </div>
                      {/* Second ring */}
                      <div className="absolute inset-4 animate-spin" style={{animationDuration: '15s', animationDirection: 'reverse'}}>
                        <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-gradient-to-r from-indigo-400 to-blue-400 rounded-full shadow-md opacity-70"></div>
                        <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-gradient-to-r from-green-400 to-teal-400 rounded-full shadow-md opacity-70"></div>
                      </div>
                      
                      {/* Connection lines - Dynamic */}
                      <div className="absolute inset-0 opacity-40 group-hover:opacity-80 transition-opacity duration-700">
                        <div className="absolute top-1/2 left-1/2 w-16 h-0.5 bg-gradient-to-r from-transparent via-kss-primary to-transparent transform -translate-y-1/2 -translate-x-1/2 rotate-0 animate-pulse"></div>
                        <div className="absolute top-1/2 left-1/2 w-16 h-0.5 bg-gradient-to-r from-transparent via-kss-secondary to-transparent transform -translate-y-1/2 -translate-x-1/2 rotate-45 animate-pulse delay-100"></div>
                        <div className="absolute top-1/2 left-1/2 w-16 h-0.5 bg-gradient-to-r from-transparent via-purple-500 to-transparent transform -translate-y-1/2 -translate-x-1/2 rotate-90 animate-pulse delay-200"></div>
                        <div className="absolute top-1/2 left-1/2 w-16 h-0.5 bg-gradient-to-r from-transparent via-pink-500 to-transparent transform -translate-y-1/2 -translate-x-1/2 rotate-135 animate-pulse delay-300"></div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Logo text with icon integration */}
                  <div className="relative overflow-hidden flex flex-col items-center gap-3">
                    <div className="flex items-center gap-6">
                      {/* Brain/Network Icon - Much Larger with effects */}
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-kss-primary/40 to-kss-secondary/40 rounded-full blur-2xl animate-pulse"></div>
                        <Brain className="w-28 h-28 text-kss-primary group-hover:text-kss-secondary transition-all duration-700 transform group-hover:rotate-12" />
                        <div className="absolute inset-2 bg-gradient-to-br from-kss-primary/20 to-transparent rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                      </div>
                      
                      <div className="text-center">
                        <h1 className="text-8xl font-black bg-gradient-to-r from-kss-primary via-purple-600 to-kss-secondary bg-clip-text text-transparent tracking-tight mb-3 animate-gradient bg-300% transition-all duration-700 group-hover:tracking-wide">
                          KSS
                        </h1>
                        {/* Full name with effects */}
                        <div className="text-base font-bold text-gray-600 dark:text-gray-300 tracking-[0.3em] uppercase group-hover:tracking-[0.4em] transition-all duration-700">
                          <span className="inline-block transform group-hover:scale-110 transition-transform duration-500">KNOWLEDGE</span>
                          <span className="inline-block mx-2 text-kss-primary">•</span>
                          <span className="inline-block transform group-hover:scale-110 transition-transform duration-500 delay-100">SPACE</span>
                          <span className="inline-block mx-2 text-kss-secondary">•</span>
                          <span className="inline-block transform group-hover:scale-110 transition-transform duration-500 delay-200">SIMULATOR</span>
                        </div>
                      </div>
                      
                      {/* Cube/Space Icon - 3D Enhanced */}
                      <div className="relative perspective-1000">
                        <div className="w-28 h-28 relative transform-style-3d group-hover:animate-spin" style={{animationDuration: '4s'}}>
                          {/* 3D Cube with multiple layers */}
                          <div className="absolute inset-0 border-3 border-kss-secondary/60 rounded-xl transform rotate-12 group-hover:rotate-45 transition-transform duration-1000 shadow-lg"></div>
                          <div className="absolute inset-2 border-3 border-kss-primary/60 rounded-xl transform -rotate-12 group-hover:rotate-0 transition-transform duration-1000 shadow-lg"></div>
                          <div className="absolute inset-4 border-2 border-purple-500/40 rounded-lg transform rotate-6 group-hover:-rotate-12 transition-transform duration-1000"></div>
                          <div className="absolute inset-5 bg-gradient-to-br from-kss-primary/30 via-purple-600/20 to-kss-secondary/30 rounded-lg backdrop-blur-sm"></div>
                          
                          {/* Floating particles with trails */}
                          <div className="absolute top-1/2 left-1/2 w-3 h-3 bg-gradient-to-r from-kss-primary to-purple-500 rounded-full transform -translate-x-1/2 -translate-y-1/2 animate-ping shadow-lg"></div>
                          <div className="absolute top-1/3 left-1/3 w-2 h-2 bg-gradient-to-r from-kss-secondary to-pink-500 rounded-full animate-bounce shadow-md" style={{animationDelay: '0.3s'}}></div>
                          <div className="absolute bottom-1/3 right-1/3 w-2 h-2 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full animate-bounce shadow-md" style={{animationDelay: '0.6s'}}></div>
                          <div className="absolute top-1/4 right-1/4 w-1.5 h-1.5 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full animate-pulse"></div>
                          <div className="absolute bottom-1/4 left-1/4 w-1.5 h-1.5 bg-gradient-to-r from-pink-400 to-rose-400 rounded-full animate-pulse delay-500"></div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Animated underline - Thicker and glowing */}
                    <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 h-2 bg-gradient-to-r from-kss-primary via-purple-500 to-kss-secondary rounded-full w-0 group-hover:w-4/5 transition-all duration-700 ease-out shadow-lg group-hover:shadow-[0_0_20px_rgba(123,63,242,0.6)]"></div>
                    
                    {/* Multiple shine animations */}
                    <div className="absolute inset-0 overflow-hidden rounded-3xl">
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent opacity-0 group-hover:opacity-100 transform -translate-x-full group-hover:translate-x-full transition-all duration-1500 ease-in-out"></div>
                      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transform -translate-y-full group-hover:translate-y-full transition-all duration-2000 ease-in-out delay-300"></div>
                    </div>
                  </div>
                  
                  {/* Enhanced corner accents with animation */}
                  <div className="absolute top-6 left-6 w-6 h-6 border-l-3 border-t-3 border-kss-primary/50 opacity-0 group-hover:opacity-100 transition-all duration-500 group-hover:w-8 group-hover:h-8"></div>
                  <div className="absolute top-6 right-6 w-6 h-6 border-r-3 border-t-3 border-kss-secondary/50 opacity-0 group-hover:opacity-100 transition-all duration-500 delay-100 group-hover:w-8 group-hover:h-8"></div>
                  <div className="absolute bottom-6 left-6 w-6 h-6 border-l-3 border-b-3 border-purple-500/50 opacity-0 group-hover:opacity-100 transition-all duration-500 delay-200 group-hover:w-8 group-hover:h-8"></div>
                  <div className="absolute bottom-6 right-6 w-6 h-6 border-r-3 border-b-3 border-pink-500/50 opacity-0 group-hover:opacity-100 transition-all duration-500 delay-300 group-hover:w-8 group-hover:h-8"></div>
                </div>
                
                {/* Enhanced floating elements with trails */}
                <div className="absolute -top-6 left-20 animate-float" style={{animationDelay: '0s', animationDuration: '3s'}}>
                  <div className="w-4 h-4 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full shadow-lg shadow-blue-400/50"></div>
                  <div className="absolute inset-0 w-4 h-4 bg-blue-400 rounded-full animate-ping opacity-30"></div>
                </div>
                <div className="absolute top-16 -right-6 animate-float" style={{animationDelay: '0.5s', animationDuration: '3.5s'}}>
                  <div className="w-3 h-3 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full shadow-lg shadow-purple-400/50"></div>
                  <div className="absolute inset-0 w-3 h-3 bg-purple-400 rounded-full animate-ping opacity-30"></div>
                </div>
                <div className="absolute -bottom-6 right-20 animate-float" style={{animationDelay: '1s', animationDuration: '4s'}}>
                  <div className="w-5 h-5 bg-gradient-to-r from-pink-400 to-rose-400 rounded-full shadow-lg shadow-pink-400/50"></div>
                  <div className="absolute inset-0 w-5 h-5 bg-pink-400 rounded-full animate-ping opacity-30"></div>
                </div>
                <div className="absolute bottom-16 -left-6 animate-float" style={{animationDelay: '1.5s', animationDuration: '3.2s'}}>
                  <div className="w-3.5 h-3.5 bg-gradient-to-r from-orange-400 to-amber-400 rounded-full shadow-lg shadow-orange-400/50"></div>
                  <div className="absolute inset-0 w-3.5 h-3.5 bg-orange-400 rounded-full animate-ping opacity-30"></div>
                </div>
                
                {/* Data flow lines */}
                <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-30 transition-opacity duration-700">
                  <div className="absolute top-1/4 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-kss-primary/50 to-transparent transform rotate-12"></div>
                  <div className="absolute bottom-1/4 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-kss-secondary/50 to-transparent transform -rotate-12"></div>
                </div>
              </div>
            </div>
            {/* Dynamic tagline with typing effect */}
            <div className="relative max-w-4xl mx-auto mb-12 mt-8">
              <p className="text-3xl font-bold text-gray-800 dark:text-gray-200 mb-2 animate-pulse">
                차세대 학습 혁신
              </p>
              <p className="text-xl font-medium text-gray-600 dark:text-gray-300 leading-relaxed">
                복잡한 기술을 <span className="font-bold bg-gradient-to-r from-kss-primary to-kss-secondary bg-clip-text text-transparent">시뮬레이션</span>으로 체험하고, 
                <span className="font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent"> AI와 함께</span> 학습하는 
                <span className="font-bold bg-gradient-to-r from-blue-500 to-cyan-500 bg-clip-text text-transparent">지식 우주</span>
              </p>
            </div>
            
            
            {/* Stats - Professional Style */}
            <div className="flex justify-center gap-6 mb-8">
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-10 hover:shadow-lg transition-shadow">
                <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  {courses.filter(c => c.status === 'active').length}+
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">활성 코스</div>
              </div>
              
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-10 hover:shadow-lg transition-shadow">
                <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  5,000+
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">학습자</div>
              </div>
              
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-10 hover:shadow-lg transition-shadow">
                <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  4.8★
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">평균 평점</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Enterprise Knowledge Simulators */}
      <section className="py-32 bg-gray-50 dark:bg-gray-900/50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="mb-12">
            <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Enterprise Knowledge Simulators
              </h2>
              <p className="text-xl font-medium text-gray-600 dark:text-gray-400">
                Production-grade simulation environments for advanced technical domains
              </p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500 dark:text-gray-400">Platform Status</div>
              <div className="flex items-center gap-2 mt-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm font-bold text-gray-700 dark:text-gray-300">Operational</span>
              </div>
            </div>
          </div>
          </div>
          
          {/* Domain Filter - Multi-row layout */}
          <div className="mb-8">
            <div className="flex flex-wrap gap-2 justify-center">
              {categories.map(category => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-all transform hover:scale-105 ${
                    selectedCategory === category
                      ? 'bg-kss-primary text-white shadow-lg shadow-kss-primary/30'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-kss-primary/30'
                  }`}
                >
                  {category === '전체' ? 'All Domains' : 
                   category === '지식공학' ? 'Knowledge Engineering' :
                   category === '금융' ? 'Financial Systems' :
                   category === 'Agent/AI' ? 'Agent/AI Systems' :
                   category === '의료/바이오' ? 'Medical AI' :
                   category === '산업AI' ? 'Industrial AI' :
                   category === '국방/보안' ? 'Defense AI' :
                   category === 'AI/ML' ? 'AI/Machine Learning' :
                   category === '물리컴퓨팅' ? 'Physical Computing' :
                   category === '블록체인' ? 'Blockchain' :
                   category === '수학/이론' ? 'Mathematics' :
                   category === '시스템/이론' ? 'System Design' :
                   category === '자율주행' ? 'Autonomous Mobility' :
                   category === '언어/교육' ? 'Language Learning' : category}
                </button>
              ))}
            </div>
          </div>
        </div>
        
        {/* Professional Simulator Grid */}
        <div className="max-w-7xl mx-auto px-4 mt-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-10">
          {filteredCourses.map((course) => {
            const Icon = course.icon;
            
            return (
              <div
                key={course.id}
                className="group"
              >
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 transition-colors">
                  
                  {/* Header Bar */}
                  <div className="flex items-center justify-between p-4 border-b border-gray-100 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded bg-gradient-to-r ${course.color} flex items-center justify-center`}>
                        <Icon className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <div className="font-mono text-xs text-gray-500 dark:text-gray-400">
                          {course.id.toUpperCase().replace('-', '_')}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${
                        course.status === 'active' ? 'bg-green-500' :
                        course.status === 'coming-soon' ? 'bg-yellow-500' : 'bg-gray-400'
                      }`}></div>
                      <span className="text-xs font-bold text-gray-600 dark:text-gray-400">
                        {course.status === 'active' ? 'ACTIVE' :
                         course.status === 'coming-soon' ? 'BETA' : 'DEV'}
                      </span>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="p-8">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      {course.title}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                      {course.description}
                    </p>
                    
                    {/* Technical Specs */}
                    <div className="grid grid-cols-2 gap-4 mb-4 text-xs">
                      <div>
                        <div className="text-gray-500 dark:text-gray-400">Complexity</div>
                        <div className="font-bold text-gray-700 dark:text-gray-300">
                          {getDifficultyLabel(course.difficulty)}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-500 dark:text-gray-400">Duration</div>
                        <div className="font-bold text-gray-700 dark:text-gray-300">
                          {course.duration}
                        </div>
                      </div>
                    </div>
                    
                    {/* Core Modules */}
                    {(course.status === 'active') && (
                      <div className="mb-4">
                        <div className="text-xs font-bold text-gray-700 dark:text-gray-300 mb-2">Core Modules</div>
                        <div className="flex flex-wrap gap-1">
                          {course.id === 'ontology' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded">RDF 에디터</span>
                              <span className="text-xs px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded">SPARQL</span>
                              <span className="text-xs px-2 py-1 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 rounded">3D 그래프</span>
                            </>
                          )}
                          {course.id === 'stock-analysis' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 rounded">차트 분석</span>
                              <span className="text-xs px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded">AI 예측</span>
                            </>
                          )}
                          {course.id === 'llm' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded">Transformer</span>
                              <span className="text-xs px-2 py-1 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded">파인튜닝</span>
                              <span className="text-xs px-2 py-1 bg-violet-50 dark:bg-violet-900/20 text-violet-600 dark:text-violet-400 rounded">추론 엔진</span>
                            </>
                          )}
                          {course.id === 'neo4j' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded">그래프 탐색</span>
                              <span className="text-xs px-2 py-1 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded">Cypher 쿼리</span>
                              <span className="text-xs px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded">지식 통합</span>
                            </>
                          )}
                          {course.id === 'rag' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded">벡터 DB</span>
                              <span className="text-xs px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded">검색 시스템</span>
                              <span className="text-xs px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded">생성 파이프라인</span>
                            </>
                          )}
                          {course.id === 'agent-mcp' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-cyan-50 dark:bg-cyan-900/20 text-cyan-600 dark:text-cyan-400 rounded">MCP 프로토콜</span>
                              <span className="text-xs px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded">A2A 통신</span>
                              <span className="text-xs px-2 py-1 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded">에이전트 오케스트레이션</span>
                            </>
                          )}
                          {course.id === 'medical-ai' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded">영상 분석</span>
                              <span className="text-xs px-2 py-1 bg-pink-50 dark:bg-pink-900/20 text-pink-600 dark:text-pink-400 rounded">진단 AI</span>
                              <span className="text-xs px-2 py-1 bg-rose-50 dark:bg-rose-900/20 text-rose-600 dark:text-rose-400 rounded">치료 최적화</span>
                            </>
                          )}
                          {course.id === 'physical-ai' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400 rounded">로봇 제어</span>
                              <span className="text-xs px-2 py-1 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded">센서 융합</span>
                              <span className="text-xs px-2 py-1 bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400 rounded">동역학 시뮬레이션</span>
                            </>
                          )}
                          {course.id === 'iot-systems' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 rounded">디바이스 관리</span>
                              <span className="text-xs px-2 py-1 bg-teal-50 dark:bg-teal-900/20 text-teal-600 dark:text-teal-400 rounded">엣지 컴퓨팅</span>
                              <span className="text-xs px-2 py-1 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400 rounded">네트워크 시뮬레이션</span>
                            </>
                          )}
                          {course.id === 'defense-ai' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-slate-50 dark:bg-slate-900/20 text-slate-600 dark:text-slate-400 rounded">전술 AI</span>
                              <span className="text-xs px-2 py-1 bg-gray-50 dark:bg-gray-900/20 text-gray-600 dark:text-gray-400 rounded">사이버 보안</span>
                              <span className="text-xs px-2 py-1 bg-zinc-50 dark:bg-zinc-900/20 text-zinc-600 dark:text-zinc-400 rounded">의사결정 지원</span>
                            </>
                          )}
                          {course.id === 'autonomous-mobility' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-cyan-50 dark:bg-cyan-900/20 text-cyan-600 dark:text-cyan-400 rounded">센서 퓨전</span>
                              <span className="text-xs px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded">경로 계획</span>
                              <span className="text-xs px-2 py-1 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded">V2X 통신</span>
                              <span className="text-xs px-2 py-1 bg-teal-50 dark:bg-teal-900/20 text-teal-600 dark:text-teal-400 rounded">CARLA</span>
                            </>
                          )}
                          {course.id === 'quantum-computing' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded">양자 회로</span>
                              <span className="text-xs px-2 py-1 bg-violet-50 dark:bg-violet-900/20 text-violet-600 dark:text-violet-400 rounded">Grover 알고리즘</span>
                              <span className="text-xs px-2 py-1 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded">양자 ML</span>
                              <span className="text-xs px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded">Qiskit</span>
                            </>
                          )}
                          {course.id === 'smart-factory' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400 rounded">예측 유지보수</span>
                              <span className="text-xs px-2 py-1 bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400 rounded">디지털 트윈</span>
                              <span className="text-xs px-2 py-1 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded">AI 품질검사</span>
                              <span className="text-xs px-2 py-1 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-600 dark:text-yellow-400 rounded">로봇 자동화</span>
                            </>
                          )}
                          {course.id === 'linear-algebra' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded">벡터 시각화</span>
                              <span className="text-xs px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded">행렬 계산기</span>
                              <span className="text-xs px-2 py-1 bg-violet-50 dark:bg-violet-900/20 text-violet-600 dark:text-violet-400 rounded">고유값 탐색</span>
                              <span className="text-xs px-2 py-1 bg-pink-50 dark:bg-pink-900/20 text-pink-600 dark:text-pink-400 rounded">SVD 분해</span>
                            </>
                          )}
                          {course.id === 'english-conversation' && (
                            <>
                              <span className="text-xs px-2 py-1 bg-rose-50 dark:bg-rose-900/20 text-rose-600 dark:text-rose-400 rounded">AI 대화 파트너</span>
                              <span className="text-xs px-2 py-1 bg-pink-50 dark:bg-pink-900/20 text-pink-600 dark:text-pink-400 rounded">발음 트레이너</span>
                              <span className="text-xs px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded">상황별 연습</span>
                              <span className="text-xs px-2 py-1 bg-violet-50 dark:bg-violet-900/20 text-violet-600 dark:text-violet-400 rounded">듣기 실험실</span>
                            </>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Metrics */}
                    {course.rating && (
                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <div>
                          <div className="text-gray-500 dark:text-gray-400">Satisfaction</div>
                          <div className="font-bold text-gray-700 dark:text-gray-300">
                            {course.rating}/5.0
                          </div>
                        </div>
                        {course.students && (
                          <div>
                            <div className="text-gray-500 dark:text-gray-400">Enrollments</div>
                            <div className="font-bold text-gray-700 dark:text-gray-300">
                              {course.students.toLocaleString()}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  {/* Action Bar */}
                  <div className="border-t border-gray-100 dark:border-gray-700 p-4">
                    {course.status === 'active' && course.link ? (
                      <Link
                        href={course.link || `/modules/${course.id}`}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-900 dark:bg-white text-white dark:text-gray-900 text-sm font-bold hover:bg-gray-800 dark:hover:bg-gray-100 transition-colors"
                      >
                        Access Environment
                        <ChevronRight className="w-4 h-4" />
                      </Link>
                    ) : (
                      <button
                        disabled
                        className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 text-sm font-bold cursor-not-allowed"
                      >
                        {course.status === 'coming-soon' ? 'Coming Soon' : 'In Development'}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
          </div>
        </div>
      </section>

      {/* Simulator Tools Section */}
      <section className="bg-gray-50 dark:bg-gray-900/50 py-32 border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-16 text-gray-900 dark:text-white">Interactive Simulators</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Multi-Agent Simulators */}
            <Link href="/modules/multi-agent/tools/a2a-orchestrator" className="group">
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-6 hover:shadow-xl transition-all h-full">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-orange-600 rounded flex items-center justify-center">
                    <Users className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400">A2A Orchestrator</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  에이전트 간 통신과 작업 흐름 시뮬레이터
                </p>
                <div className="text-xs text-orange-600 dark:text-orange-400 font-semibold">
                  실습 도구 →
                </div>
              </div>
            </Link>

            <Link href="/modules/multi-agent/tools/crewai-builder" className="group">
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-6 hover:shadow-xl transition-all h-full">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-orange-600 rounded flex items-center justify-center">
                    <Zap className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400">CrewAI Builder</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  역할 기반 에이전트 팀 구성 도구
                </p>
                <div className="text-xs text-orange-600 dark:text-orange-400 font-semibold">
                  실습 도구 →
                </div>
              </div>
            </Link>

            <Link href="/modules/multi-agent/tools/consensus-simulator" className="group">
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-6 hover:shadow-xl transition-all h-full">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-orange-600 rounded flex items-center justify-center">
                    <Cpu className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400">Consensus Simulator</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  분산 합의 알고리즘 시각화 도구
                </p>
                <div className="text-xs text-orange-600 dark:text-orange-400 font-semibold">
                  실습 도구 →
                </div>
              </div>
            </Link>

            {/* RAG Simulators */}
            <Link href="/modules/rag" className="group">
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-6 hover:shadow-xl transition-all h-full">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-green-600 rounded flex items-center justify-center">
                    <Database className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-emerald-600 dark:group-hover:text-emerald-400">RAG Playground</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  전체 RAG 파이프라인을 실시간으로 체험
                </p>
                <div className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold">
                  RAG Module →
                </div>
              </div>
            </Link>

            {/* Ontology Simulators */}
            <Link href="/rdf-editor" className="group">
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-6 hover:shadow-xl transition-all h-full">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-600 rounded flex items-center justify-center">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400">RDF Editor</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  시각적 RDF Triple 편집기와 지식 그래프 구축
                </p>
                <div className="text-xs text-blue-600 dark:text-blue-400 font-semibold">
                  Ontology Module →
                </div>
              </div>
            </Link>

            {/* Stock Analysis Simulators - Redirect to new module */}
            <Link href="/modules/stock-analysis" className="group">
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-6 hover:shadow-xl transition-all h-full">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-orange-500 rounded flex items-center justify-center">
                    <TrendingUp className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-red-600 dark:group-hover:text-red-400">Stock Simulator</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  실전 투자 전략과 기술적 분석 시뮬레이터
                </p>
                <div className="text-xs text-red-600 dark:text-red-400 font-semibold">
                  Stock Analysis Module →
                </div>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* System Tools Section - Admin & Management */}
      <section className="bg-gradient-to-br from-gray-900 to-gray-800 dark:from-black dark:to-gray-900 py-16 border-t border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold text-white">System Management Tools</h2>
            <div className="flex items-center gap-2">
              <Settings className="w-5 h-5 text-gray-400" />
              <span className="text-sm text-gray-400">Admin Access</span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {/* Content Manager */}
            <Link href="/modules/content-manager" className="group">
              <div className="bg-gray-800/50 dark:bg-gray-900/50 border border-gray-700 p-4 hover:bg-gray-700/50 transition-all">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-indigo-500 rounded flex items-center justify-center">
                    <Database className="w-4 h-4 text-white" />
                  </div>
                  <h3 className="text-sm font-bold text-white">Content Manager</h3>
                </div>
                <p className="text-xs text-gray-400 mb-2">
                  Monitor and update all module content
                </p>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-green-400">• Active</span>
                  <span className="text-gray-500">Dashboard →</span>
                </div>
              </div>
            </Link>

            {/* Video Creator */}
            <Link href="/video-creator" className="group">
              <div className="bg-gray-800/50 dark:bg-gray-900/50 border border-gray-700 p-4 hover:bg-gray-700/50 transition-all">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-red-500 to-pink-500 rounded flex items-center justify-center">
                    <Zap className="w-4 h-4 text-white" />
                  </div>
                  <h3 className="text-sm font-bold text-white">Video Creator</h3>
                </div>
                <p className="text-xs text-gray-400 mb-2">
                  Generate educational video content
                </p>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-green-400">• Active</span>
                  <span className="text-gray-500">Studio →</span>
                </div>
              </div>
            </Link>

            {/* 3D Graph Viewer */}
            <Link href="/3d-graph" className="group">
              <div className="bg-gray-800/50 dark:bg-gray-900/50 border border-gray-700 p-4 hover:bg-gray-700/50 transition-all">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-cyan-500 to-blue-500 rounded flex items-center justify-center">
                    <Network className="w-4 h-4 text-white" />
                  </div>
                  <h3 className="text-sm font-bold text-white">3D Graph Viewer</h3>
                </div>
                <p className="text-xs text-gray-400 mb-2">
                  Visualize knowledge graphs in 3D
                </p>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-green-400">• Active</span>
                  <span className="text-gray-500">Viewer →</span>
                </div>
              </div>
            </Link>

            {/* Stock Dictionary */}
            <Link href="/stock-dictionary" className="group">
              <div className="bg-gray-800/50 dark:bg-gray-900/50 border border-gray-700 p-4 hover:bg-gray-700/50 transition-all">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-emerald-500 rounded flex items-center justify-center">
                    <TrendingUp className="w-4 h-4 text-white" />
                  </div>
                  <h3 className="text-sm font-bold text-white">Stock Dictionary</h3>
                </div>
                <p className="text-xs text-gray-400 mb-2">
                  Financial terms and concepts database
                </p>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-green-400">• Active</span>
                  <span className="text-gray-500">Reference →</span>
                </div>
              </div>
            </Link>

            {/* YouTube Summarizer */}
            <Link href="/youtube-summarizer" className="group">
              <div className="bg-gray-800/50 dark:bg-gray-900/50 border border-gray-700 p-4 hover:bg-gray-700/50 transition-all">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-red-600 to-orange-500 rounded flex items-center justify-center">
                    <Eye className="w-4 h-4 text-white" />
                  </div>
                  <h3 className="text-sm font-bold text-white">YouTube Summarizer</h3>
                </div>
                <p className="text-xs text-gray-400 mb-2">
                  AI-powered video content summarization
                </p>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-green-400">• Active</span>
                  <span className="text-gray-500">Analyze →</span>
                </div>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section - Professional Style */}
      <section className="bg-white dark:bg-gray-900 py-32 border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-16 text-gray-900 dark:text-white">KSS만의 특별한 학습 경험</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-10 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white">인터랙티브 시뮬레이션</h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                복잡한 개념을 직접 만지고 실험하며 학습하는 환경 제공
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-10 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-500 rounded flex items-center justify-center">
                  <Network className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white">3D 시각화</h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                추상적 개념을 3차원 공간에서 탐색하여 직관적 이해
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-10 hover:shadow-lg transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-red-500 rounded flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white">AI 맞춤 학습</h3>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                개인별 학습 속도와 스타일에 최적화된 AI 기반 커리큘럼
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}