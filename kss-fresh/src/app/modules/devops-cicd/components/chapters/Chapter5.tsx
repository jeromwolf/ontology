'use client'

import React from 'react'
import { Server } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-2xl p-8 mb-8 border border-purple-200 dark:border-purple-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center">
            <Server className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Kubernetes 운영</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          Ingress, ConfigMap, Secret, 스케일링 등 Kubernetes 클러스터 운영에 필요한 고급 기능들을 학습합니다.
        </p>
      </div>

      <p>
        이 챕터에서는 프로덕션 환경에서 Kubernetes를 운영하기 위한 고급 기능들을 다룹니다.
      </p>

      <p className="text-gray-600 dark:text-gray-400">
        상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
      </p>
    </div>
  )
}