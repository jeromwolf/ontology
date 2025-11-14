'use client';

import { useState, useEffect } from 'react';
import { Volume2, Pause, MessageCircle, Users, Globe, Copy, CheckCircle, Play } from 'lucide-react';
import References from '@/components/common/References';

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

      <References
        sections={[
          {
            title: '📚 학습 플랫폼 & 리소스',
            icon: 'web' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'BBC Learning English',
                url: 'https://www.bbc.co.uk/learningenglish',
                description: 'BBC 영어 학습 플랫폼 - 뉴스, 비즈니스, 문법 강의 무료 제공 (2024)',
                year: 2024
              },
              {
                title: 'VOA Learning English',
                url: 'https://learningenglish.voanews.com/',
                description: 'Voice of America - 천천히 말하는 뉴스 및 다양한 레벨 콘텐츠 (2024)',
                year: 2024
              },
              {
                title: 'ESL Pod - English as a Second Language',
                url: 'https://www.eslpod.com/',
                description: '일상 대화 팟캐스트 - 실생활 표현과 문화 설명 포함 (2024)',
                year: 2024
              },
              {
                title: "Rachel's English",
                url: 'https://www.youtube.com/c/rachelsenglish',
                description: 'YouTube 발음 강의 - 미국 영어 발음 세밀 교정 (2024)',
                year: 2024
              }
            ]
          },
          {
            title: '📖 핵심 교재',
            icon: 'research' as const,
            color: 'border-rose-500',
            items: [
              {
                title: 'English Grammar in Use (Raymond Murphy)',
                url: 'https://www.cambridge.org/elt/grammarinuse',
                description: '세계적 베스트셀러 문법서 - 자습용 명쾌한 설명 (5판, 2019)',
                year: 2019
              },
              {
                title: 'Practical English Usage (Michael Swan)',
                url: 'https://global.oup.com/academic/product/practical-english-usage-9780194202411',
                description: '영어 사용법 백과사전 - 실무 영어 완벽 정리 (4판, 2016)',
                year: 2016
              },
              {
                title: 'Oxford Collocations Dictionary',
                url: 'https://www.oxfordlearnersdictionaries.com/about/collocations',
                description: '연어 사전 - 자연스러운 영어 표현 조합 완벽 수록 (2판, 2009)',
                year: 2009
              },
              {
                title: 'Cambridge Advanced Learner\'s Dictionary',
                url: 'https://dictionary.cambridge.org/dictionary/english/',
                description: '온라인 영영사전 - 발음, 예문, 문법 정보 완벽 제공 (2024)',
                year: 2024
              }
            ]
          },
          {
            title: '🛠️ 실전 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Grammarly',
                url: 'https://www.grammarly.com/',
                description: 'AI 영문법 교정 도구 - 스펠링, 문법, 스타일 자동 검사 (2024)',
                year: 2024
              },
              {
                title: 'Anki',
                url: 'https://apps.ankiweb.net/',
                description: '플래시카드 암기 앱 - 간격 반복 학습법 기반 무료 도구 (2024)',
                year: 2024
              },
              {
                title: 'Forvo',
                url: 'https://forvo.com/',
                description: '발음 사전 - 네이티브 발음 녹음 450만+ 단어 (2024)',
                year: 2024
              },
              {
                title: 'YouGlish',
                url: 'https://youglish.com/',
                description: 'YouTube 영상 속 실제 발음 검색 - 문맥 속 발음 학습 (2024)',
                year: 2024
              },
              {
                title: 'Reverso Context',
                url: 'https://context.reverso.net/',
                description: '번역 및 예문 검색 - 실제 사용 예문 수백만 개 제공 (2024)',
                year: 2024
              }
            ]
          }
        ]}
      />
    </div>
  )
}
