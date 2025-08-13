'use client'

import React from 'react'
import { Cpu } from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl p-8 mb-8 border border-blue-200 dark:border-blue-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Kubernetes 기초</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          Kubernetes 아키텍처와 핵심 오브젝트들을 이해하고, kubectl로 클러스터를 관리하는 방법을 학습합니다.
        </p>
      </div>

      <p>
        이 챕터에서는 Kubernetes의 기본 개념과 아키텍처, 주요 오브젝트들에 대해 학습합니다.
        실습을 통해 Pod, Service, Deployment를 만들고 관리하는 방법을 익히겠습니다.
      </p>

      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 my-6">
        <h3 className="text-blue-800 dark:text-blue-300 mt-0">주요 학습 내용</h3>
        <ul className="text-blue-700 dark:text-blue-300">
          <li>• Kubernetes 아키텍처 이해</li>
          <li>• Pod, Service, Deployment 개념</li>
          <li>• kubectl 기본 명령어</li>
          <li>• YAML 매니페스트 작성</li>
        </ul>
      </div>

      <p className="text-gray-600 dark:text-gray-400">
        상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
      </p>
    </div>
  )
}