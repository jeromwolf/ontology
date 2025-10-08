'use client'

import ChapterContent from '../components/ChapterContent'

export default function ChapterPage({ params }: { params: { chapterId: string } }) {
  return <ChapterContent chapterId={params.chapterId} />
}
