'use client';

import Link from 'next/link';
import { moduleMetadata } from './metadata';

export default function AIEthicsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link href="/" className="text-blue-600 hover:text-blue-800 mb-4 inline-block">
            ← 홈으로 돌아가기
          </Link>
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="flex items-center gap-4 mb-6">
              <div className={`w-16 h-16 bg-gradient-to-r ${moduleMetadata.gradient} rounded-xl flex items-center justify-center`}>
                <span className="text-3xl">{moduleMetadata.icon}</span>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">{moduleMetadata.title}</h1>
                <p className="text-gray-600 mt-2">{moduleMetadata.description}</p>
              </div>
            </div>
            <div className="flex gap-4 text-sm">
              <span className="px-3 py-1 bg-gray-100 rounded-full">{moduleMetadata.category}</span>
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full">{moduleMetadata.difficulty}</span>
              <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full">{moduleMetadata.estimatedHours}시간</span>
            </div>
          </div>
        </div>

        {/* Under Development Notice */}
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-6 mb-8">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                이 모듈은 현재 개발 중입니다. 곧 완성된 콘텐츠로 만나보실 수 있습니다.
              </p>
            </div>
          </div>
        </div>

        {/* Planned Chapters */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-6">예정된 챕터</h2>
          <div className="grid gap-4">
            {moduleMetadata.chapters.map((chapter) => (
              <div key={chapter.id} className="bg-white rounded-lg shadow p-6 opacity-60">
                <h3 className="text-xl font-semibold mb-2">{chapter.title}</h3>
                <p className="text-gray-600 mb-2">{chapter.description}</p>
                <p className="text-sm text-gray-500">{chapter.estimatedMinutes}분</p>
              </div>
            ))}
          </div>
        </div>

        {/* Planned Simulators */}
        <div>
          <h2 className="text-2xl font-bold mb-6">예정된 시뮬레이터</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {moduleMetadata.simulators.map((simulator) => (
              <div key={simulator.id} className="bg-white rounded-lg shadow p-6 opacity-60">
                <h3 className="text-lg font-semibold mb-2">{simulator.title}</h3>
                <p className="text-gray-600">{simulator.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}