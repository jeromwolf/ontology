'use client';

import { FileText, BookOpen, Globe } from 'lucide-react';

interface Reference {
  title: string;
  authors?: string;
  year?: string;
  link?: string;
  description?: string;
}

interface ReferenceSection {
  title: string;
  icon?: 'paper' | 'book' | 'web';
  color: string;
  items: Reference[];
}

interface ReferencesProps {
  sections: ReferenceSection[];
}

export default function References({ sections }: ReferencesProps) {
  const getIcon = (iconType?: string) => {
    switch (iconType) {
      case 'paper':
        return <FileText className="w-5 h-5" />;
      case 'book':
        return <BookOpen className="w-5 h-5" />;
      case 'web':
        return <Globe className="w-5 h-5" />;
      default:
        return <FileText className="w-5 h-5" />;
    }
  };

  return (
    <section className="mt-12 bg-gradient-to-r from-gray-50 to-slate-50 dark:from-gray-900/50 dark:to-slate-900/50 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-2 mb-6">
        <BookOpen className="w-6 h-6 text-blue-600 dark:text-blue-400" />
        <h3 className="text-xl font-bold text-gray-900 dark:text-white">
          References & Further Reading
        </h3>
      </div>

      <div className="space-y-4">
        {sections.map((section, idx) => (
          <div key={idx} className={`border-l-4 ${section.color} pl-4 py-2`}>
            <div className="flex items-center gap-2 mb-3">
              {getIcon(section.icon)}
              <h4 className="font-semibold text-gray-900 dark:text-white">
                {section.title}
              </h4>
            </div>

            <ul className="space-y-2">
              {section.items.map((item, itemIdx) => (
                <li key={itemIdx} className="text-sm text-gray-700 dark:text-gray-300">
                  <div className="flex items-start gap-2">
                    <span className="text-gray-400 mt-1">•</span>
                    <div className="flex-1">
                      {item.authors && (
                        <span className="font-medium">{item.authors} </span>
                      )}
                      {item.year && (
                        <span className="text-gray-500">({item.year}) </span>
                      )}
                      <span className="font-semibold">"{item.title}"</span>
                      {item.description && (
                        <span className="text-gray-600 dark:text-gray-400">
                          {' '}
                          - {item.description}
                        </span>
                      )}
                      {item.link && (
                        <a
                          href={item.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-2 text-blue-600 dark:text-blue-400 hover:underline inline-flex items-center gap-1"
                        >
                          <Globe className="w-3 h-3" />
                          <span className="text-xs">Link</span>
                        </a>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400 italic">
          💡 이 자료들은 해당 주제에 대한 심화 학습과 최신 연구 동향을 파악하는 데 도움이 됩니다.
        </p>
      </div>
    </section>
  );
}
