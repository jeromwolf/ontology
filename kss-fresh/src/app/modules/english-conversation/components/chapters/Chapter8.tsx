'use client'

import { useState, useEffect } from 'react'
import { Volume2, Pause, MessageCircle, Users, Globe, Copy, CheckCircle, Play } from 'lucide-react'

export default function Chapter8() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          고급 회화 기법과 설득력 있는 소통
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          복잡한 주제에 대한 토론, 논리적 설득, 감정적 뉘앙스 표현 등 고급 수준의 영어 회화 기법을 마스터합니다.
        </p>
      </div>

      <div className="bg-indigo-50 dark:bg-indigo-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🎯 논리적 설득 구조
        </h3>
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">1. 주장 제시 (Claim)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              "I believe that remote work should be the default option for our company."
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">2. 근거 제시 (Evidence)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              "Studies show that remote workers are 13% more productive, and our team's performance has improved by 25% since going remote."
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">3. 결론 강화 (Warrant)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              "Therefore, implementing a remote-first policy would benefit both the company and employees."
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            💡 고급 표현법
          </h3>
          <div className="space-y-3 text-sm">
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">가정법:</span>
              <p className="text-gray-600 dark:text-gray-400">"If I were in your position..."</p>
            </div>
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">강조법:</span>
              <p className="text-gray-600 dark:text-gray-400">"What really matters is..."</p>
            </div>
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">대조법:</span>
              <p className="text-gray-600 dark:text-gray-400">"On the one hand... On the other hand..."</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            🔥 토론 기법
          </h3>
          <div className="space-y-3 text-sm">
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">의견 제시:</span>
              <p className="text-gray-600 dark:text-gray-400">"From my perspective..."</p>
            </div>
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">반박:</span>
              <p className="text-gray-600 dark:text-gray-400">"I see your point, however..."</p>
            </div>
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">타협:</span>
              <p className="text-gray-600 dark:text-gray-400">"Perhaps we could find a middle ground..."</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
