'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  Grid3x3, 
  Move, 
  Maximize2, 
  GitBranch, 
  Layers, 
  TrendingUp,
  BarChart3,
  Activity,
  ChevronRight,
  Home,
  BookOpen,
  Beaker,
  Menu,
  X,
  Sigma,
  Binary,
  Sparkles,
  Axis3d
} from 'lucide-react'

const chapters = [
  { id: 1, title: '벡터와 벡터공간', slug: 'vectors', icon: Move },
  { id: 2, title: '행렬과 행렬연산', slug: 'matrices', icon: Grid3x3 },
  { id: 3, title: '선형변환', slug: 'linear-transformations', icon: Maximize2 },
  { id: 4, title: '고유값과 고유벡터', slug: 'eigenvalues', icon: GitBranch },
  { id: 5, title: '직교성과 정규화', slug: 'orthogonality', icon: Axis3d },
  { id: 6, title: 'SVD와 차원축소', slug: 'svd', icon: Layers },
  { id: 7, title: '선형시스템', slug: 'linear-systems', icon: Binary },
  { id: 8, title: 'AI/ML 응용', slug: 'ml-applications', icon: Sparkles }
]

const simulators = [
  { title: 'Vector Visualizer', slug: 'vector-visualizer', icon: Move },
  { title: 'Matrix Calculator', slug: 'matrix-calculator', icon: Grid3x3 },
  { title: 'Eigenvalue Explorer', slug: 'eigenvalue-explorer', icon: GitBranch },
  { title: 'SVD Decomposer', slug: 'svd-decomposer', icon: Layers }
]

export default function LinearAlgebraLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [progress, setProgress] = useState<number[]>([])

  useEffect(() => {
    const savedProgress = JSON.parse(
      localStorage.getItem('linear-algebra-progress') || '[]'
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
            <Link href="/linear-algebra" className="flex items-center gap-3 group">
              <div className="p-2.5 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl shadow-lg shadow-indigo-500/20 group-hover:shadow-xl group-hover:shadow-indigo-500/30 transition-all">
                <Sigma className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  Linear Algebra
                </h2>
                <p className="text-xs text-gray-500 dark:text-gray-400">선형대수학</p>
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
              href="/linear-algebra"
              className={`flex items-center gap-3 px-3 py-2.5 mb-2 rounded-lg transition-all ${
                pathname === '/linear-algebra'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/25'
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
                  const chapterPath = `/linear-algebra/chapter/${chapter.slug}`
                  
                  return (
                    <Link
                      key={chapter.id}
                      href={chapterPath}
                      className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                        isActive(chapterPath)
                          ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/25'
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
                  const simPath = `/linear-algebra/simulator/${simulator.slug}`
                  
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
              <span className="text-sm font-bold text-indigo-600 dark:text-indigo-400">
                {Math.round((progress.length / chapters.length) * 100)}%
              </span>
            </div>
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 transition-all duration-500"
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