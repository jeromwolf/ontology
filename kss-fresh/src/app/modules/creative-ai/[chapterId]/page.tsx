import ChapterContent from '../components/ChapterContent'

interface PageProps {
  params: Promise<{
    chapterId: string
  }>
}

export default async function ChapterPage({ params }: PageProps) {
  const { chapterId } = await params
  return <ChapterContent chapterId={chapterId} />
}
