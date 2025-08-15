'use client';

import React from 'react';
import { Monitor } from 'lucide-react';

export default function Chapter8() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-2xl p-8 mb-8 border border-red-200 dark:border-red-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-red-500 rounded-xl flex items-center justify-center">
            <Monitor className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">모니터링, 로깅, 보안</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          Prometheus, Grafana, ELK Stack을 활용한 모니터링과 로깅, 그리고 컨테이너 보안 모범 사례를 학습합니다.
        </p>
      </div>

      <p>
        이 챕터에서는 프로덕션 시스템의 모니터링, 로깅, 보안에 대해 학습합니다.
      </p>

      <p className="text-gray-600 dark:text-gray-400">
        상세한 콘텐츠는 개발 중입니다. 곧 업데이트될 예정입니다.
      </p>
    </div>
  )
}