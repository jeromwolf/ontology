import ChapterContent from '../components/ChapterContent'

export default function OptimizationTheoryChapterPage({ params }: { params: { chapterId: string } }) {
  return <ChapterContent chapterId={params.chapterId} />
}
