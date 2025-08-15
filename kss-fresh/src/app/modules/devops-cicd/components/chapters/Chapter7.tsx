'use client';

import React from 'react';
import { GitBranch } from 'lucide-react';

export default function Chapter7() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-indigo-50 to-cyan-50 dark:from-indigo-900/20 dark:to-cyan-900/20 rounded-2xl p-8 mb-8 border border-indigo-200 dark:border-indigo-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-indigo-500 rounded-xl flex items-center justify-center">
            <GitBranch className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">GitOps와 배포 전략</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          GitOps 원칙에 따른 선언적 배포와 Blue-Green, Canary, Rolling Update 등 다양한 배포 전략을 학습합니다.
        </p>
      </div>

      <p>
        이 챕터에서는 GitOps 개념과 ArgoCD 사용법, 그리고 다양한 배포 전략들을 실습합니다.
      </p>

      <p className="text-gray-600 dark:text-gray-400">
        상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
      </p>
    </div>
  )
}