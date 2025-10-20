'use client'

import { useParams } from 'next/navigation'
import Navigation from '@/components/Navigation'
import ChapterContent from '../components/ChapterContent'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function ChapterPage() {
  const params = useParams()
  const chapterId = params?.chapterId as string

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Link
            href="/modules/robotics-manipulation"
            className="inline-flex items-center gap-2 text-orange-600 dark:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Robotics & Manipulation 모듈로 돌아가기
          </Link>
        </div>

        <ChapterContent chapterId={chapterId} />
      </main>
    </div>
  )
}
