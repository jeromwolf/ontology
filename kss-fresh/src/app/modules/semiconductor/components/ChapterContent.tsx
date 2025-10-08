'use client'

import dynamic from 'next/dynamic'
import Link from 'next/link'

// 동적 임포트로 각 챕터 컴포넌트 로드
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })
const Chapter9 = dynamic(() => import('./chapters/Chapter9'), { ssr: false })

// 챕터 순서 정의
const chapters = [
  { id: 'basics', title: '반도체 기초' },
  { id: 'design', title: '회로 설계' },
  { id: 'lithography', title: '포토리소그래피' },
  { id: 'fabrication', title: '제조 공정' },
  { id: 'advanced', title: '첨단 기술' },
  { id: 'ai-chips', title: 'AI 반도체' },
  { id: 'memory', title: '차세대 메모리' },
  { id: 'future', title: '미래 기술' },
  { id: 'image-display', title: '이미지센서 & 디스플레이' }
]

export default function ChapterContent({ chapterId }: { chapterId: string }) {
  const currentIndex = chapters.findIndex(ch => ch.id === chapterId)
  const prevChapter = currentIndex > 0 ? chapters[currentIndex - 1] : null
  const nextChapter = currentIndex < chapters.length - 1 ? chapters[currentIndex + 1] : null

  const getChapterComponent = () => {
    switch (chapterId) {
      case 'basics':
        return <Chapter1 />
      case 'design':
        return <Chapter2 />
      case 'lithography':
        return <Chapter3 />
      case 'fabrication':
        return <Chapter4 />
      case 'advanced':
        return <Chapter5 />
      case 'ai-chips':
        return <Chapter6 />
      case 'memory':
        return <Chapter7 />
      case 'future':
        return <Chapter8 />
      case 'image-display':
        return <Chapter9 />
      default:
        return (
          <div className="text-center py-20">
            <h2 className="text-2xl font-bold">챕터를 찾을 수 없습니다</h2>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900">
      {getChapterComponent()}

      {/* 네비게이션 */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="flex justify-between items-center border-t border-gray-200 dark:border-gray-700 pt-6">
          <div className="flex-1">
            {prevChapter && (
              <Link
                href={`/modules/semiconductor/${prevChapter.id}`}
                className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <div className="text-left">
                  <div className="text-xs text-gray-500 dark:text-gray-400">이전</div>
                  <div className="font-medium">{prevChapter.title}</div>
                </div>
              </Link>
            )}
          </div>

          <Link
            href="/modules/semiconductor"
            className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors"
          >
            목록으로
          </Link>

          <div className="flex-1 flex justify-end">
            {nextChapter && (
              <Link
                href={`/modules/semiconductor/${nextChapter.id}`}
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
              >
                <div className="text-right">
                  <div className="text-xs text-blue-100">다음</div>
                  <div className="font-medium">{nextChapter.title}</div>
                </div>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
