import dynamic from 'next/dynamic';

const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false });
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false });
const Chapter3 = dynamic(() => import('./chapters/Chapter3'), { ssr: false });
const Chapter4 = dynamic(() => import('./chapters/Chapter4'), { ssr: false });
const Chapter5 = dynamic(() => import('./chapters/Chapter5'), { ssr: false });
const Chapter6 = dynamic(() => import('./chapters/Chapter6'), { ssr: false });

interface ChapterContentProps {
  chapterId: string;
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'introduction':
        return <Chapter1 />;
      case 'bias-fairness':
        return <Chapter2 />;
      case 'transparency':
        return <Chapter3 />;
      case 'privacy-security':
        return <Chapter4 />;
      case 'regulation':
        return <Chapter5 />;
      case 'case-studies':
        return <Chapter6 />;
      default:
        return (
          <div className="text-center py-12">
            <p className="text-gray-600 dark:text-gray-400">챕터를 찾을 수 없습니다.</p>
          </div>
        );
    }
  };

  return <div className="chapter-content">{getChapterComponent()}</div>;
}
