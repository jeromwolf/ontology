'use client'

import dynamic from 'next/dynamic'

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
const Chapter10 = dynamic(() => import('./chapters/Chapter10'), { ssr: false })
const Chapter11 = dynamic(() => import('./chapters/Chapter11'), { ssr: false })
const Chapter12 = dynamic(() => import('./chapters/Chapter12'), { ssr: false })

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'ai-infrastructure-overview':
        return <Chapter1 />
      case 'gpu-parallel-computing':
        return <Chapter2 />
      case 'containers-orchestration':
        return <Chapter3 />
      case 'ml-framework-basics':
        return <Chapter4 />
      case 'distributed-training':
        return <Chapter5 />
      case 'model-serving-optimization':
        return <Chapter6 />
      case 'feature-store-pipelines':
        return <Chapter7 />
      case 'experiment-tracking':
        return <Chapter8 />
      case 'large-scale-training':
        return <Chapter9 />
      case 'monitoring-observability':
        return <Chapter10 />
      case 'ml-cicd':
        return <Chapter11 />
      case 'production-mlops':
        return <Chapter12 />
      default:
        return (
          <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-slate-900 text-white flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-4xl font-bold mb-4">챕터를 찾을 수 없습니다</h1>
              <p className="text-slate-300">요청하신 챕터가 존재하지 않습니다.</p>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {getChapterComponent()}
    </div>
  )
}
