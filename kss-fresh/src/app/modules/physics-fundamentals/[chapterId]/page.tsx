import ChapterContent from '../components/ChapterContent'

export default function PhysicsChapterPage({ params }: { params: { chapterId: string } }) {
  return <ChapterContent chapterId={params.chapterId} />
}
