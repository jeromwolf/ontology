'use client'

import { CheckCircle2, Circle } from 'lucide-react'

interface ModuleProgressProps {
  completedChapters: number
  totalChapters: number
  completedSimulators: number
  totalSimulators: number
}

export default function ModuleProgress({
  completedChapters,
  totalChapters,
  completedSimulators,
  totalSimulators
}: ModuleProgressProps) {
  const chapterProgress = totalChapters > 0 ? (completedChapters / totalChapters) * 100 : 0
  const simulatorProgress = totalSimulators > 0 ? (completedSimulators / totalSimulators) * 100 : 0
  const overallProgress = (chapterProgress + simulatorProgress) / 2

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
      <h3 className="text-lg font-semibold mb-4">학습 진도</h3>
      
      <div className="space-y-4">
        {/* Overall Progress */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-600 dark:text-gray-400">전체 진도</span>
            <span className="font-medium">{Math.round(overallProgress)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div
              className="h-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-600 transition-all duration-500"
              style={{ width: `${overallProgress}%` }}
            />
          </div>
        </div>

        {/* Chapter Progress */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-600 dark:text-gray-400">챕터 학습</span>
            <span className="font-medium">{completedChapters} / {totalChapters}</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="h-2 rounded-full bg-blue-500 transition-all duration-500"
              style={{ width: `${chapterProgress}%` }}
            />
          </div>
        </div>

        {/* Simulator Progress */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-600 dark:text-gray-400">시뮬레이터 실습</span>
            <span className="font-medium">{completedSimulators} / {totalSimulators}</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="h-2 rounded-full bg-green-500 transition-all duration-500"
              style={{ width: `${simulatorProgress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Achievement Badges */}
      {overallProgress === 100 && (
        <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
            <span className="text-sm font-medium text-green-700 dark:text-green-300">
              모듈 학습 완료! 🎉
            </span>
          </div>
        </div>
      )}
    </div>
  )
}