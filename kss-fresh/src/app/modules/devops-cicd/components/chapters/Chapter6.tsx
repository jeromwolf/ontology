'use client'

import React from 'react'
import { GitBranch } from 'lucide-react'

export default function Chapter6() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-2xl p-8 mb-8 border border-green-200 dark:border-green-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center">
            <GitBranch className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">CI/CD 파이프라인 구축</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          GitHub Actions와 Jenkins를 활용하여 자동화된 빌드, 테스트, 배포 파이프라인을 구축합니다.
        </p>
      </div>

      <p>
        이 챕터에서는 CI/CD의 개념부터 실제 파이프라인 구축까지 실습합니다.
      </p>

      <p className="text-gray-600 dark:text-gray-400">
        상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
      </p>
    </div>
  )
}