'use client'

import { useParams } from 'next/navigation'
import ChapterContent from '../components/ChapterContent'
import Navigation from '@/components/Navigation'

export default function ChapterPage() {
  const params = useParams()
  const chapterId = params?.chapterId as string

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />
      <ChapterContent chapterId={chapterId} />
    </div>
  )
}
