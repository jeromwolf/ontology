import ChapterContent from '../components/ChapterContent'

export default function DataEngineeringChapterPage({ params }: { params: { chapterId: string } }) {
  return <ChapterContent chapterId={params.chapterId} />
}
