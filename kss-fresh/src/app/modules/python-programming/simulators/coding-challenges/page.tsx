'use client'

import Link from 'next/link'
import { ArrowLeft, Trophy, Code, Play, CheckCircle2 } from 'lucide-react'

export default function CodingChallengesPage() {
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
            <div className="w-12 h-12 bg-gradient-to-br from-yellow-500 to-orange-600 rounded-xl flex items-center justify-center">
              <Trophy className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Python Coding Challenges
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                50+ interactive coding problems with instant feedback
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
                    Difficulty Levels
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Beginner, Intermediate, and Advanced challenges
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Topic Categories
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Strings, Arrays, Algorithms, Data Structures, OOP
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Instant Feedback
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Run test cases and get immediate results with hints
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Solutions & Explanations
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Learn from detailed solutions and multiple approaches
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Challenge Categories Preview
              </h2>
              <div className="grid md:grid-cols-3 gap-3">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
                  <div className="font-semibold text-blue-900 dark:text-blue-200 mb-1">Easy (20)</div>
                  <p className="text-xs text-blue-700 dark:text-blue-400">Basic syntax & logic</p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3 border border-orange-200 dark:border-orange-800">
                  <div className="font-semibold text-orange-900 dark:text-orange-200 mb-1">Medium (20)</div>
                  <p className="text-xs text-orange-700 dark:text-orange-400">Algorithms & patterns</p>
                </div>
                <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3 border border-red-200 dark:border-red-800">
                  <div className="font-semibold text-red-900 dark:text-red-200 mb-1">Hard (10)</div>
                  <p className="text-xs text-red-700 dark:text-red-400">Advanced problems</p>
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
                  <span className="text-gray-700 dark:text-gray-300">Challenge Database - In Progress</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
                  <span className="text-gray-500 dark:text-gray-500">Test Runner & Leaderboard - Planned</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
