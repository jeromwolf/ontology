import React from 'react';
import { ExternalLink, BookOpen, Wrench, FileText } from 'lucide-react';

interface ReferenceItem {
  title: string;
  url: string;
  description: string;
}

interface ReferenceSection {
  title: string;
  icon: 'web' | 'research' | 'tools' | 'docs';
  color: string;
  items: ReferenceItem[];
}

interface ReferencesProps {
  sections: ReferenceSection[];
}

export default function References({ sections }: ReferencesProps) {
  const getIcon = (iconType: string) => {
    switch (iconType) {
      case 'web':
        return <ExternalLink className="w-5 h-5" />;
      case 'research':
        return <BookOpen className="w-5 h-5" />;
      case 'tools':
        return <Wrench className="w-5 h-5" />;
      case 'docs':
        return <FileText className="w-5 h-5" />;
      default:
        return <ExternalLink className="w-5 h-5" />;
    }
  };

  return (
    <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-6">
      <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">
        📚 References
      </h2>

      <div className="space-y-6">
        {sections.map((section, sectionIdx) => (
          <div key={sectionIdx} className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className={`text-lg font-bold mb-4 flex items-center gap-2 text-gray-900 dark:text-white border-l-4 pl-3 ${section.color}`}>
              {getIcon(section.icon)}
              {section.title}
            </h3>

            <div className="space-y-3">
              {section.items.map((item, itemIdx) => (
                <div
                  key={itemIdx}
                  className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded-lg hover:shadow-md transition-shadow"
                >
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-semibold text-blue-600 dark:text-blue-400 hover:underline flex items-start gap-2"
                  >
                    <ExternalLink className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <span>{item.title}</span>
                  </a>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 ml-6">
                    {item.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
