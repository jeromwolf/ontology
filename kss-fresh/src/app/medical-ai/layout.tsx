'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  Brain, 
  Activity, 
  Microscope, 
  Pill, 
  Scan, 
  Heart,
  Database,
  Dna,
  ChevronRight,
  Home,
  BookOpen,
  Beaker,
  Menu,
  X
} from 'lucide-react'

const chapters = [
  { id: 1, title: 'Medical AI 개요', slug: 'introduction', icon: Brain },
  { id: 2, title: '의료 영상 분석', slug: 'medical-imaging', icon: Scan },
  { id: 3, title: '진단 보조 시스템', slug: 'diagnosis-assistant', icon: Activity },
  { id: 4, title: '신약 개발 AI', slug: 'drug-discovery', icon: Pill },
  { id: 5, title: '유전체 분석', slug: 'genomics', icon: Dna },
  { id: 6, title: '환자 모니터링', slug: 'patient-monitoring', icon: Heart },
  { id: 7, title: '의료 데이터 관리', slug: 'medical-data', icon: Database },
  { id: 8, title: '윤리와 규제', slug: 'ethics-regulation', icon: Microscope }
]

const simulators = [
  { title: 'X-Ray Analyzer', slug: 'xray-analyzer', icon: Scan },
  { title: 'Diagnosis AI', slug: 'diagnosis-ai', icon: Brain },
  { title: 'Drug Discovery', slug: 'drug-discovery-sim', icon: Beaker },
  { title: 'Patient Dashboard', slug: 'patient-dashboard', icon: Activity }
]

export default function MedicalAILayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [progress, setProgress] = useState<number[]>([])

  useEffect(() => {
    const savedProgress = JSON.parse(
      localStorage.getItem('medical-ai-progress') || '[]'
    )
    setProgress(savedProgress)
  }, [])

  const isActive = (path: string) => pathname === path

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950">
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg"
      >
        {isSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Sidebar */}
      <aside className={`${
        isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } lg:translate-x-0 fixed lg:sticky top-0 left-0 z-40 w-80 h-screen bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-r border-gray-200 dark:border-gray-800 transition-transform duration-300`}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="p-6 border-b border-gray-200 dark:border-gray-800">
            <Link href="/medical-ai" className="flex items-center gap-3 group">
              <div className="p-2.5 bg-gradient-to-br from-red-500 to-pink-600 rounded-xl shadow-lg shadow-red-500/20 group-hover:shadow-xl group-hover:shadow-red-500/30 transition-all">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold bg-gradient-to-r from-red-600 to-pink-600 bg-clip-text text-transparent">
                  Medical AI
                </h2>
                <p className="text-xs text-gray-500 dark:text-gray-400">의료 인공지능</p>
              </div>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto p-4">
            {/* Home Link */}
            <Link
              href="/"
              className="flex items-center gap-3 px-3 py-2 mb-4 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-all"
            >
              <Home className="w-4 h-4" />
              <span className="text-sm">홈으로</span>
            </Link>

            {/* Overview Link */}
            <Link
              href="/medical-ai"
              className={`flex items-center gap-3 px-3 py-2.5 mb-2 rounded-lg transition-all ${
                pathname === '/medical-ai'
                  ? 'bg-gradient-to-r from-red-500 to-pink-600 text-white shadow-lg shadow-red-500/25'
                  : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              <span className="font-medium">Overview</span>
            </Link>

            {/* Chapters */}
            <div className="mb-6">
              <h3 className="px-3 mb-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">
                학습 콘텐츠
              </h3>
              <div className="space-y-1">
                {chapters.map((chapter) => {
                  const Icon = chapter.icon
                  const isCompleted = progress.includes(chapter.id)
                  const chapterPath = `/medical-ai/chapter/${chapter.slug}`
                  
                  return (
                    <Link
                      key={chapter.id}
                      href={chapterPath}
                      className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                        isActive(chapterPath)
                          ? 'bg-gradient-to-r from-red-500 to-pink-600 text-white shadow-lg shadow-red-500/25'
                          : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
                      }`}
                    >
                      <div className={`flex items-center justify-center w-8 h-8 rounded-lg ${
                        isActive(chapterPath)
                          ? 'bg-white/20'
                          : isCompleted
                          ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                          : 'bg-gray-100 dark:bg-gray-800 text-gray-400'
                      }`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <span className="flex-1 text-sm font-medium">
                        {chapter.title}
                      </span>
                      <ChevronRight className={`w-4 h-4 transition-transform ${
                        isActive(chapterPath) ? 'translate-x-1' : 'group-hover:translate-x-1'
                      }`} />
                    </Link>
                  )
                })}
              </div>
            </div>

            {/* Simulators */}
            <div>
              <h3 className="px-3 mb-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">
                시뮬레이터
              </h3>
              <div className="space-y-1">
                {simulators.map((simulator) => {
                  const Icon = simulator.icon
                  const simPath = `/medical-ai/simulator/${simulator.slug}`
                  
                  return (
                    <Link
                      key={simulator.slug}
                      href={simPath}
                      className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                        isActive(simPath)
                          ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg shadow-purple-500/25'
                          : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
                      }`}
                    >
                      <div className={`flex items-center justify-center w-8 h-8 rounded-lg ${
                        isActive(simPath)
                          ? 'bg-white/20'
                          : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                      }`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <span className="flex-1 text-sm font-medium">
                        {simulator.title}
                      </span>
                      <ChevronRight className={`w-4 h-4 transition-transform ${
                        isActive(simPath) ? 'translate-x-1' : 'group-hover:translate-x-1'
                      }`} />
                    </Link>
                  )
                })}
              </div>
            </div>
          </nav>

          {/* Progress */}
          <div className="p-6 border-t border-gray-200 dark:border-gray-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                학습 진행률
              </span>
              <span className="text-sm font-bold text-red-600 dark:text-red-400">
                {Math.round((progress.length / chapters.length) * 100)}%
              </span>
            </div>
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-red-500 to-pink-600 transition-all duration-500"
                style={{ width: `${(progress.length / chapters.length) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 lg:ml-0">
        {children}
      </main>
    </div>
  )
}