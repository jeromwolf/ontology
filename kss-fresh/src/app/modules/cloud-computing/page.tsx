'use client';

import { useState } from 'react';
import Link from 'next/link';
import { moduleMetadata } from './metadata';
import ChapterContent from './components/ChapterContent';

export default function CloudComputingPage() {
  const [currentChapterId, setCurrentChapterId] = useState<string>('cloud-fundamentals');

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link href="/" className="text-sky-600 hover:text-sky-800 dark:text-sky-400 mb-4 inline-block">
            ← 홈으로 돌아가기
          </Link>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
            <div className="flex items-center gap-4 mb-6">
              <div className={`w-16 h-16 bg-gradient-to-r ${moduleMetadata.gradient} rounded-xl flex items-center justify-center`}>
                <span className="text-3xl">{moduleMetadata.icon}</span>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{moduleMetadata.title}</h1>
                <p className="text-gray-600 dark:text-gray-400 mt-2">{moduleMetadata.description}</p>
              </div>
            </div>
            <div className="flex gap-4 text-sm">
              <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full">{moduleMetadata.category}</span>
              <span className="px-3 py-1 bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-300 rounded-full">{moduleMetadata.difficulty}</span>
              <span className="px-3 py-1 bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300 rounded-full">{moduleMetadata.estimatedHours}시간</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar - Chapter Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 sticky top-4">
              <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">챕터</h2>
              <nav className="space-y-2">
                {moduleMetadata.chapters.map((chapter, index) => (
                  <button
                    key={chapter.id}
                    onClick={() => setCurrentChapterId(chapter.id)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                      currentChapterId === chapter.id
                        ? 'bg-sky-100 dark:bg-sky-900 text-sky-800 dark:text-sky-200 font-semibold'
                        : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    <div className="text-sm font-medium">Chapter {index + 1}</div>
                    <div className="text-xs mt-1">{chapter.title}</div>
                  </button>
                ))}
              </nav>

              {/* Simulators Link */}
              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <h3 className="text-sm font-bold mb-3 text-gray-900 dark:text-white">시뮬레이터</h3>
                <div className="space-y-2">
                  {moduleMetadata.simulators.map((simulator) => (
                    <Link
                      key={simulator.id}
                      href={`/modules/cloud-computing/simulators/${simulator.id}`}
                      className="block px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                    >
                      {simulator.title}
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
              <ChapterContent chapterId={currentChapterId} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
