'use client'

import { ReactNode } from 'react'
import Link from 'next/link'
import { ChevronLeft, Clock, BookOpen, Server } from 'lucide-react'
import { devopsMetadata } from './metadata'

interface DevOpsLayoutProps {
  children: ReactNode
}

export default function DevOpsLayout({ children }: DevOpsLayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-slate-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Header */}
      <header className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-md border-b border-gray-200/30 dark:border-gray-700/30 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Link 
                href="/modules"
                className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-gray-600 dark:hover:text-gray-400 transition-colors"
              >
                <ChevronLeft size={20} />
                <span>모듈 목록</span>
              </Link>
              
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-600"></div>
              
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${devopsMetadata.moduleColor} flex items-center justify-center text-white`}>
                  <Server size={20} />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                    {devopsMetadata.title}
                  </h1>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {devopsMetadata.category}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
              <div className="flex items-center gap-1">
                <Clock size={16} />
                <span>{devopsMetadata.duration}</span>
              </div>
              <div className="flex items-center gap-1">
                <BookOpen size={16} />
                <span>{devopsMetadata.chapters.length}개 챕터</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main>
        {children}
      </main>
    </div>
  )
}