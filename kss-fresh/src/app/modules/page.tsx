'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Brain, TrendingUp, Network, Sparkles, 
  Atom, Cpu, Database, Globe, Car, Factory,
  ChevronRight, Star, Clock, Users, Zap, Server, Settings, Eye,
  Play, BarChart3, Activity, ArrowRight, Search, Filter, BookOpen
} from 'lucide-react'

interface Module {
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
  chapters?: number;
  simulators?: number;
}

const modules: Module[] = [
  {
    id: 'llm',
    title: 'Large Language Models',
    description: 'Transformer, GPT, Claude 등 최신 LLM 기술 완전 정복',
    icon: Cpu,
    color: 'from-indigo-500 to-purple-600',
    category: 'AI/ML',
    difficulty: 'intermediate',
    duration: '6주',
    students: 856,
    rating: 4.9,
    status: 'active',
    link: '/modules/llm',
    chapters: 8,
    simulators: 6
  },
  {
    id: 'ontology',
    title: 'Knowledge Graphs & Ontology',
    description: 'RDF, SPARQL, 시맨틱 웹을 통한 지식 그래프 마스터',
    icon: Brain,
    color: 'from-purple-500 to-pink-500',
    category: 'Knowledge',
    difficulty: 'intermediate',
    duration: '8주',
    students: 1234,
    rating: 4.8,
    status: 'active',
    link: '/modules/ontology',
    chapters: 16,
    simulators: 4
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
    link: '/modules/rag',
    chapters: 6,
    simulators: 5
  },
  {
    id: 'system-design',
    title: 'System Design',
    description: '대규모 분산 시스템 설계의 핵심 원칙과 실전 패턴',
    icon: Server,
    color: 'from-purple-500 to-indigo-600',
    category: 'Systems',
    difficulty: 'advanced',
    duration: '20시간',
    students: 785,
    rating: 4.9,
    status: 'active',
    link: '/modules/system-design',
    chapters: 8,
    simulators: 6
  },
  {
    id: 'web3',
    title: 'Web3 & Blockchain',
    description: '블록체인 기술과 Web3 생태계 완전 정복',
    icon: Globe,
    color: 'from-indigo-500 to-cyan-500',
    category: 'Blockchain',
    difficulty: 'intermediate',
    duration: '12시간',
    students: 567,
    rating: 4.8,
    status: 'active',
    link: '/modules/web3',
    chapters: 8,
    simulators: 6
  },
  {
    id: 'computer-vision',
    title: 'Computer Vision',
    description: '이미지 처리부터 3D 변환, 객체 인식까지',
    icon: Eye,
    color: 'from-teal-500 to-cyan-600',
    category: 'AI/ML',
    difficulty: 'intermediate',
    duration: '20시간',
    students: 320,
    rating: 4.9,
    status: 'active',
    link: '/modules/computer-vision',
    chapters: 8,
    simulators: 5
  },
  {
    id: 'deep-learning',
    title: 'Deep Learning',
    description: '신경망 기초부터 CNN, Transformer, GAN까지 완전 정복',
    icon: Brain,
    color: 'from-violet-500 to-purple-600',
    category: 'AI/ML',
    difficulty: 'intermediate',
    duration: '25시간',
    students: 0,
    rating: 5.0,
    status: 'active',
    link: '/modules/deep-learning',
    chapters: 8,
    simulators: 6
  },
  {
    id: 'stock-analysis',
    title: 'Financial Analysis',
    description: '실전 투자 전략과 심리까지 포함한 종합 투자 마스터',
    icon: TrendingUp,
    color: 'from-red-500 to-orange-500',
    category: 'Finance',
    difficulty: 'beginner',
    duration: '16주',
    students: 2341,
    rating: 4.9,
    status: 'active',
    link: '/modules/stock-analysis',
    chapters: 10,
    simulators: 8
  },
  {
    id: 'quantum-computing',
    title: 'Quantum Computing',
    description: '양자 컴퓨팅 원리부터 양자 알고리즘까지',
    icon: Atom,
    color: 'from-purple-500 to-violet-600',
    category: 'Quantum',
    difficulty: 'advanced',
    duration: '24시간',
    students: 89,
    rating: 4.8,
    status: 'active',
    link: '/modules/quantum-computing',
    chapters: 8,
    simulators: 4
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
    link: '/modules/autonomous-mobility',
    chapters: 8,
    simulators: 4
  },
  {
    id: 'neo4j',
    title: 'Neo4j Graph Database',
    description: '그래프 데이터베이스 설계부터 Cypher 쿼리 최적화까지',
    icon: Network,
    color: 'from-green-500 to-emerald-600',
    category: 'Database',
    difficulty: 'intermediate',
    duration: '20시간',
    students: 234,
    rating: 4.8,
    status: 'active',
    link: '/modules/neo4j',
    chapters: 8,
    simulators: 5
  },
  {
    id: 'agent-mcp',
    title: 'AI Agent & MCP',
    description: 'AI 에이전트 개발과 Model Context Protocol 마스터',
    icon: Zap,
    color: 'from-yellow-500 to-orange-600',
    category: 'AI/ML',
    difficulty: 'advanced',
    duration: '18시간',
    students: 445,
    rating: 4.9,
    status: 'active',
    link: '/modules/agent-mcp',
    chapters: 8,
    simulators: 4
  },
  {
    id: 'multi-agent',
    title: 'Multi-Agent Systems',
    description: '협력하는 AI 에이전트 시스템 설계와 구현',
    icon: Network,
    color: 'from-orange-500 to-red-600',
    category: 'AI/ML',
    difficulty: 'advanced',
    duration: '22시간',
    students: 178,
    rating: 4.8,
    status: 'active',
    link: '/modules/multi-agent',
    chapters: 8,
    simulators: 3
  },
  {
    id: 'smart-factory',
    title: 'Smart Factory & Industry 4.0',
    description: '스마트 팩토리 구축과 산업 4.0 디지털 전환',
    icon: Factory,
    color: 'from-gray-500 to-gray-700',
    category: 'Industry',
    difficulty: 'intermediate',
    duration: '14시간',
    students: 267,
    rating: 4.7,
    status: 'active',
    link: '/modules/smart-factory',
    chapters: 6,
    simulators: 4
  },
  {
    id: 'english-conversation',
    title: 'AI English Conversation',
    description: 'AI와 함께하는 실전 영어 회화 마스터',
    icon: Globe,
    color: 'from-blue-500 to-purple-600',
    category: 'Language',
    difficulty: 'beginner',
    duration: '10시간',
    students: 892,
    rating: 4.8,
    status: 'active',
    link: '/modules/english-conversation',
    chapters: 6,
    simulators: 5
  },
  {
    id: 'ai-security',
    title: 'AI Security & Privacy',
    description: 'AI 시스템 보안과 개인정보 보호 완전 가이드',
    icon: Settings,
    color: 'from-red-500 to-pink-600',
    category: 'Security',
    difficulty: 'advanced',
    duration: '16시간',
    students: 234,
    rating: 4.9,
    status: 'active',
    link: '/modules/ai-security',
    chapters: 8,
    simulators: 10
  },
  {
    id: 'youtube-summarizer',
    title: 'YouTube Summarizer',
    description: 'AI 기반 YouTube 비디오 요약 및 분석 도구',
    icon: Play,
    color: 'from-red-500 to-red-600',
    category: 'System Tools',
    difficulty: 'beginner',
    duration: '즉시 사용',
    students: 1520,
    rating: 4.8,
    status: 'active',
    link: '/youtube-summarizer',
    chapters: 1,
    simulators: 1
  },
  {
    id: 'video-creator',
    title: 'Video Creator',  
    description: 'Remotion 기반 자동 비디오 생성 스튜디오',
    icon: Activity,
    color: 'from-purple-500 to-indigo-600',
    category: 'System Tools',
    difficulty: 'intermediate',
    duration: '30분',
    students: 892,
    rating: 4.7,
    status: 'active',
    link: '/video-creator',
    chapters: 3,
    simulators: 6
  },
  {
    id: 'ai-automation',
    title: 'AI Automation',
    description: 'AI 기반 워크플로우 자동화와 코드 생성',
    icon: Zap,
    color: 'from-yellow-500 to-orange-500',
    category: 'AI/ML',
    difficulty: 'intermediate',
    duration: '14시간',
    students: 567,
    rating: 4.8,
    status: 'active',
    link: '/modules/ai-automation',
    chapters: 8,
    simulators: 4
  },
  {
    id: 'bioinformatics',
    title: 'Bioinformatics',
    description: '생물정보학과 AI를 활용한 생명과학 연구',
    icon: Activity,
    color: 'from-green-500 to-teal-600',
    category: 'BioTech',
    difficulty: 'advanced',
    duration: '18시간',
    students: 234,
    rating: 4.7,
    status: 'active',
    link: '/modules/bioinformatics',
    chapters: 8,
    simulators: 4
  },
  {
    id: 'physical-ai',
    title: 'Physical AI',
    description: '물리 시뮬레이션과 디지털 트윈, Omniverse',
    icon: Cpu,
    color: 'from-blue-500 to-indigo-600',
    category: 'AI/ML',
    difficulty: 'advanced',
    duration: '20시간',
    students: 145,
    rating: 4.9,
    status: 'active',
    link: '/modules/physical-ai',
    chapters: 8,
    simulators: 7
  },
  {
    id: 'probability-statistics',
    title: 'Probability & Statistics',
    description: '확률통계와 베이지안 추론, 몬테카를로 시뮬레이션',
    icon: BarChart3,
    color: 'from-purple-500 to-blue-600',
    category: 'Math',
    difficulty: 'intermediate',
    duration: '16시간',
    students: 678,
    rating: 4.8,
    status: 'active',
    link: '/modules/probability-statistics',
    chapters: 8,
    simulators: 5
  },
  // 🚀 새로운 모듈들 추가
  {
    id: 'ai-ethics',
    title: 'AI 윤리 & 거버넌스',
    description: 'AI 윤리 원칙과 책임감 있는 AI 개발, 규제 대응',
    icon: Settings,
    color: 'from-rose-500 to-pink-600',
    category: 'Ethics',
    difficulty: 'intermediate',
    duration: '12시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/ai-ethics',
    chapters: 6,
    simulators: 4
  },
  {
    id: 'cyber-security',
    title: 'Cyber Security',
    description: '사이버 보안 실습과 해킹 시뮬레이션, 보안 아키텍처',
    icon: Settings,
    color: 'from-red-500 to-red-700',
    category: 'Security',
    difficulty: 'advanced',
    duration: '20시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/cyber-security',
    chapters: 8,
    simulators: 6
  },
  {
    id: 'cloud-computing',
    title: 'Cloud Computing',
    description: 'AWS, Azure, GCP 실무와 클라우드 아키텍처 설계',
    icon: Server,
    color: 'from-blue-500 to-sky-600',
    category: 'Cloud',
    difficulty: 'intermediate',
    duration: '18시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/cloud-computing',
    chapters: 10,
    simulators: 8
  },
  {
    id: 'data-engineering',
    title: 'Data Engineering',
    description: 'ETL 파이프라인, 실시간 데이터 처리, 데이터 레이크 구축',
    icon: Database,
    color: 'from-emerald-500 to-teal-600',
    category: 'Data',
    difficulty: 'advanced',
    duration: '22시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/data-engineering',
    chapters: 12,
    simulators: 10
  },
  {
    id: 'creative-ai',
    title: 'Creative AI',
    description: 'Midjourney, DALL-E, 생성형 AI로 창작하는 디지털 아트',
    icon: Sparkles,
    color: 'from-pink-500 to-purple-600',
    category: 'Creative',
    difficulty: 'beginner',
    duration: '14시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/creative-ai',
    chapters: 8,
    simulators: 12
  },
  {
    id: 'devops-cicd',
    title: 'DevOps & CI/CD',
    description: 'Docker, Kubernetes, GitOps로 구축하는 현대적 개발 운영',
    icon: Settings,
    color: 'from-gray-500 to-slate-600',
    category: 'DevOps',
    difficulty: 'intermediate',
    duration: '16시간',
    students: 234,
    rating: 4.8,
    status: 'active',
    link: '/modules/devops-cicd',
    chapters: 8,
    simulators: 6
  },
  // 🚀 고급 기술 모듈들 추가
  {
    id: 'hpc-computing',
    title: 'High-Performance Computing',
    description: '분산컴퓨팅, CUDA, 병렬처리로 구축하는 초고성능 시스템',
    icon: Cpu,
    color: 'from-orange-500 to-red-600',
    category: 'HPC',
    difficulty: 'advanced',
    duration: '24시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/hpc-computing',
    chapters: 10,
    simulators: 8
  },
  {
    id: 'multimodal-ai',
    title: 'Multimodal AI Systems',
    description: '텍스트+이미지+음성을 통합하는 차세대 멀티모달 AI 구축',
    icon: Brain,
    color: 'from-violet-500 to-purple-600',
    category: 'AI/ML',
    difficulty: 'advanced',
    duration: '20시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/multimodal-ai',
    chapters: 8,
    simulators: 6
  },
  {
    id: 'optimization-theory',
    title: 'Mathematical Optimization',
    description: '최적화 이론과 실무 알고리즘, AI 최적화 응용',
    icon: BarChart3,
    color: 'from-indigo-500 to-blue-600',
    category: 'Math',
    difficulty: 'advanced',
    duration: '18시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/optimization-theory',
    chapters: 10,
    simulators: 7
  },
  {
    id: 'ai-infrastructure',
    title: 'AI Infrastructure & MLOps',
    description: '대규모 AI 시스템 인프라 설계와 ML 파이프라인 최적화',
    icon: Server,
    color: 'from-emerald-500 to-green-600',
    category: 'MLOps',
    difficulty: 'advanced',
    duration: '22시간',
    students: 0,
    rating: 0,
    status: 'coming-soon',
    link: '/modules/ai-infrastructure',
    chapters: 12,
    simulators: 10
  }
]

const categories = ['All', 'AI/ML', 'Knowledge', 'Systems', 'Blockchain', 'Quantum', '자율주행', 'Database', 'Industry', 'Language', 'Security', 'Finance', 'System Tools', 'BioTech', 'Math', 'Ethics', 'Cloud', 'Data', 'Creative', 'DevOps', 'HPC', 'MLOps']
const difficulties = ['All', 'beginner', 'intermediate', 'advanced']

export default function ModulesPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [selectedDifficulty, setSelectedDifficulty] = useState('All')

  const filteredModules = modules.filter(module => {
    const matchesSearch = module.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         module.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = selectedCategory === 'All' || module.category === selectedCategory
    const matchesDifficulty = selectedDifficulty === 'All' || module.difficulty === selectedDifficulty
    
    return matchesSearch && matchesCategory && matchesDifficulty
  })

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
      case 'intermediate': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'advanced': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
      default: return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getDifficultyText = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '초급'
      case 'intermediate': return '중급'
      case 'advanced': return '고급'
      default: return difficulty
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-indigo-600 via-purple-600 to-cyan-600 text-white">
        <div className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold mb-6">
              Knowledge Space Simulator
            </h1>
            <p className="text-xl text-white/90 max-w-3xl mx-auto">
              차세대 학습 혁신 플랫폼. 복잡한 기술을 시뮬레이션으로 체험하고, AI와 함께 학습하는 지식 우주.
            </p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">{modules.length}+</div>
              <div className="text-white/80">전문 모듈</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">
                {modules.reduce((acc, mod) => acc + (mod.chapters || 0), 0)}+
              </div>
              <div className="text-white/80">학습 챕터</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">
                {modules.reduce((acc, mod) => acc + (mod.simulators || 0), 0)}+
              </div>
              <div className="text-white/80">시뮬레이터</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">
                {modules.reduce((acc, mod) => acc + (mod.students || 0), 0).toLocaleString()}+
              </div>
              <div className="text-white/80">수강생</div>
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filter Section */}
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="모듈 검색..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400 focus:border-transparent"
              />
            </div>

            {/* Filters */}
            <div className="flex gap-4">
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400"
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category === 'All' ? '전체 카테고리' : category}
                  </option>
                ))}
              </select>

              <select
                value={selectedDifficulty}
                onChange={(e) => setSelectedDifficulty(e.target.value)}
                className="px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400"
              >
                {difficulties.map(difficulty => (
                  <option key={difficulty} value={difficulty}>
                    {difficulty === 'All' ? '전체 난이도' : getDifficultyText(difficulty)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-4 text-gray-600 dark:text-gray-400">
            총 {filteredModules.length}개의 모듈을 찾았습니다
          </div>
        </div>

        {/* Modules Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filteredModules.map((module) => {
            const Icon = module.icon
            
            return (
              <Link
                key={module.id}
                href={module.link || `/modules/${module.id}`}
                className="group"
              >
                <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 border border-gray-200 dark:border-gray-700 hover:border-indigo-500 dark:hover:border-indigo-400 transition-all hover:shadow-xl hover:-translate-y-1 h-full">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-6">
                    <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${module.color} flex items-center justify-center`}>
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${getDifficultyColor(module.difficulty)}`}>
                      {getDifficultyText(module.difficulty)}
                    </div>
                  </div>

                  {/* Content */}
                  <div className="mb-6">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                      {module.title}
                    </h3>
                    
                    <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed mb-4">
                      {module.description}
                    </p>

                    <div className="flex items-center gap-4 mb-4 text-sm text-gray-500 dark:text-gray-400">
                      <div className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        <span>{module.duration}</span>
                      </div>
                      {module.chapters && (
                        <div className="flex items-center gap-1">
                          <BookOpen className="w-4 h-4" />
                          <span>{module.chapters}챕터</span>
                        </div>
                      )}
                      {module.simulators && (
                        <div className="flex items-center gap-1">
                          <Play className="w-4 h-4" />
                          <span>{module.simulators}개</span>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg text-sm">
                        {module.category}
                      </span>
                      
                      {module.students && (
                        <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                          <div className="flex items-center gap-1">
                            <Users className="w-4 h-4" />
                            <span>{module.students.toLocaleString()}</span>
                          </div>
                          {module.rating && (
                            <div className="flex items-center gap-1">
                              <Star className="w-4 h-4 fill-current text-yellow-400" />
                              <span>{module.rating}</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Footer */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                      module.status === 'active' 
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                    }`}>
                      {module.status === 'active' ? '이용 가능' : '준비 중'}
                    </div>
                    
                    <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-medium">
                      <span className="text-sm">시작하기</span>
                      <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>

        {/* Empty State */}
        {filteredModules.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-6">
              <Search className="w-12 h-12 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              검색 결과가 없습니다
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              다른 검색어나 필터를 시도해보세요
            </p>
          </div>
        )}
      </div>

      {/* CTA Section */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-gray-800 dark:to-gray-800">
        <div className="container mx-auto px-4 py-16 text-center">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
            지금 바로 시작해보세요
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
            실무에 바로 적용할 수 있는 시뮬레이션 기반 학습 경험을 제공합니다.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/dashboard"
              className="px-8 py-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-medium transition-colors"
            >
              학습 시작하기
            </Link>
            <Link
              href="/about"
              className="px-8 py-4 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:border-gray-400 dark:hover:border-gray-500 rounded-xl font-medium transition-colors"
            >
              더 알아보기
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}