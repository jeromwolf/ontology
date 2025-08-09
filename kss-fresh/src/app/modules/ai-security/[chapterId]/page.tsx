import { notFound } from 'next/navigation';
import { aiSecurityMetadata } from '../metadata';
import ChapterContent from '../components/ChapterContent';

interface Props {
  params: {
    chapterId: string;
  };
}

export default function ChapterPage({ params }: Props) {
  const chapter = aiSecurityMetadata.chapters.find(ch => ch.id === params.chapterId);
  
  if (!chapter) {
    notFound();
  }

  return <ChapterContent chapterId={params.chapterId} />;
}

export function generateStaticParams() {
  return aiSecurityMetadata.chapters.map(chapter => ({
    chapterId: chapter.id,
  }));
}