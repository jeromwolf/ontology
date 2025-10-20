'use client'

import Link from 'next/link'
import { ArrowLeft, Scale, AlertTriangle, FileCheck, TrendingUp } from 'lucide-react'

export default function SimulatorNav() {
  const simulators = [
    {
      id: 'bias-detector',
      name: 'Bias Detector',
      icon: AlertTriangle,
      description: '데이터셋 편향 탐지'
    },
    {
      id: 'fairness-analyzer',
      name: 'Fairness Analyzer',
      icon: Scale,
      description: '공정성 메트릭 분석'
    },
    {
      id: 'ethics-framework',
      name: 'Ethics Framework',
      icon: FileCheck,
      description: '윤리 프레임워크 구축'
    },
    {
      id: 'impact-assessment',
      name: 'Impact Assessment',
      icon: TrendingUp,
      description: '사회적 영향 평가'
    }
  ]

  return (
    <nav className="bg-gradient-to-r from-rose-50 to-pink-50 dark:from-gray-900 dark:to-rose-950 border-b border-rose-200 dark:border-rose-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        {/* Back Button */}
        <Link
          href="/modules/ai-ethics"
          className="inline-flex items-center gap-2 text-rose-600 dark:text-rose-400 hover:text-rose-700 dark:hover:text-rose-300 transition-colors mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>AI Ethics 모듈로 돌아가기</span>
        </Link>

        {/* Simulator Links */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {simulators.map((sim) => {
            const Icon = sim.icon
            return (
              <Link
                key={sim.id}
                href={`/modules/ai-ethics/simulators/${sim.id}`}
                className="group p-4 bg-white dark:bg-gray-800 rounded-lg border border-rose-200 dark:border-rose-800 hover:border-rose-400 dark:hover:border-rose-600 hover:shadow-lg transition-all"
              >
                <div className="flex items-start gap-3">
                  <div className="p-2 bg-gradient-to-br from-rose-500 to-pink-600 text-white rounded-lg group-hover:scale-110 transition-transform">
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-gray-900 dark:text-white text-sm group-hover:text-rose-600 dark:group-hover:text-rose-400 transition-colors">
                      {sim.name}
                    </h3>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {sim.description}
                    </p>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      </div>
    </nav>
  )
}
