'use client'

import dynamic from 'next/dynamic'

const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false })
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false })
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false })
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false })
const Chapter7 = dynamic(() => import('./chapters/Chapter7'), { ssr: false })
const Chapter8 = dynamic(() => import('./chapters/Chapter8'), { ssr: false })

export default function ChapterContent({ chapterId }: { chapterId: string }) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'limits':
        return <Chapter1 />
      case 'derivatives':
        return <Chapter2 />
      case 'applications-derivatives':
        return <Chapter3 />
      case 'integration':
        return <Chapter4 />
      case 'applications-integration':
        return <Chapter5 />
      case 'sequences-series':
        return <Chapter6 />
      case 'multivariable':
        return <Chapter7 />
      case 'vector-calculus':
        return <Chapter8 />
      default:
        return (
          <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900 text-white flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-4xl font-bold mb-4">챕터를 찾을 수 없습니다</h1>
              <p className="text-slate-300">요청하신 챕터가 존재하지 않습니다.</p>
            </div>
          </div>
        )
    }
  }

  return getChapterComponent()
}
