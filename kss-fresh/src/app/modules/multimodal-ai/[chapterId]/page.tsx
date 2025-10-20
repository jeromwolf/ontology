import ChapterContent from '../components/ChapterContent'

export default function MultimodalAIChapterPage({ params }: { params: { chapterId: string } }) {
  return <ChapterContent chapterId={params.chapterId} />
}
