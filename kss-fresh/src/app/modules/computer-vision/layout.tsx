'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { moduleMetadata } from './metadata';
import { Eye, ArrowLeft, Home } from 'lucide-react';

export default function ComputerVisionLayout({
  children
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const isChapterPage = pathname.includes('/modules/computer-vision/') && 
                       !pathname.endsWith('/computer-vision') &&
                       !pathname.includes('/simulators/');
  const isSimulatorPage = pathname.includes('/simulators/');
  const isMainPage = pathname === '/modules/computer-vision';

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <header className="sticky top-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/"
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
                title="홈으로"
              >
                <Home className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              </Link>
              <Link 
                href="/modules/computer-vision"
                className="flex items-center gap-3 hover:opacity-80 transition-opacity"
              >
                <div className="p-2 bg-teal-100 dark:bg-teal-900/30 rounded-lg">
                  <Eye className="w-6 h-6 text-teal-600 dark:text-teal-400" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                    {moduleMetadata.title}
                  </h1>
                  {!isMainPage && (
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {moduleMetadata.description}
                    </p>
                  )}
                </div>
              </Link>
            </div>
            
            {(isChapterPage || isSimulatorPage) && (
              <Link
                href="/modules/computer-vision"
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                모듈 홈으로
              </Link>
            )}
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {children}
      </main>
    </div>
  );
}