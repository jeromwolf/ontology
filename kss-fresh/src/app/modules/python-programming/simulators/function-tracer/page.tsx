'use client'

import Link from 'next/link'
import { ArrowLeft, GitBranch, Code, Play, CheckCircle2 } from 'lucide-react'

export default function FunctionTracerPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900/10 dark:to-gray-900">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <Link
          href="/modules/python-programming"
          className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 mb-8"
        >
          <ArrowLeft className="w-4 h-4" />
          Python Programming 모듈로 돌아가기
        </Link>

        <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 mb-8 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center">
              <GitBranch className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Function Execution Tracer
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Step-by-step function call visualization
              </p>
            </div>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl p-6 mb-6">
            <div className="flex items-start gap-3">
              <Code className="w-6 h-6 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-1" />
              <div>
                <h3 className="font-semibold text-amber-900 dark:text-amber-200 mb-2">
                  Development in Progress
                </h3>
                <p className="text-amber-800 dark:text-amber-300 text-sm">
                  This simulator is currently under active development. Check back soon for the full interactive experience!
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Planned Features
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Call Stack Visualization
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Watch the call stack grow and shrink as functions execute
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Variable Scope Tracking
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    See how variables change in local, enclosing, and global scopes
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Recursion Debugging
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Understand recursive function calls with visual tree diagrams
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Return Value Flow
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Trace how return values propagate through function calls
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Development Status
              </h2>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                  <span className="text-gray-700 dark:text-gray-300">UI Design Complete</span>
                </div>
                <div className="flex items-center gap-3">
                  <Play className="w-5 h-5 text-blue-500 animate-pulse" />
                  <span className="text-gray-700 dark:text-gray-300">Execution Tracer - In Progress</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
                  <span className="text-gray-500 dark:text-gray-500">Advanced Debugging Tools - Planned</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
