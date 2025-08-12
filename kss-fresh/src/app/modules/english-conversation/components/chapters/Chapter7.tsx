'use client'

import { useState, useEffect } from 'react'
import { Volume2, Pause, MessageCircle, Users, Globe, Copy, CheckCircle, Play } from 'lucide-react'

export default function Chapter7() {
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
          영어권 문화와 소통 에티켓
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          언어는 문화와 밀접한 관련이 있습니다. 영어권 문화를 이해하고 상황에 맞는 적절한 표현을 사용하는 방법을 배워보겠습니다.
        </p>
      </div>

      <div className="bg-amber-50 dark:bg-amber-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🤝 소통 스타일 차이
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">직접적 표현 (Direct)</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• "I disagree with that"</li>
              <li>• "That won't work"</li>
              <li>• "I need this by Friday"</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">간접적 표현 (Indirect)</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• "I see what you mean, but..."</li>
              <li>• "That might be challenging"</li>
              <li>• "If possible, could you...?"</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

