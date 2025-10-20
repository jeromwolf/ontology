import ChapterContent from '../components/ChapterContent'

export default function AIInfrastructureChapterPage({ params }: { params: { chapterId: string } }) {
  return <ChapterContent chapterId={params.chapterId} />
}
