import ChapterContent from '../components/ChapterContent'

export default function CalculusChapterPage({ params }: { params: { chapterId: string } }) {
  return <ChapterContent chapterId={params.chapterId} />
}
