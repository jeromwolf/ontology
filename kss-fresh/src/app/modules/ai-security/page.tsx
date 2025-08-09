'use client';

import Link from 'next/link';
import { Shield, Lock, Eye, Brain, AlertTriangle, Search, Server, FileWarning } from 'lucide-react';
import { aiSecurityMetadata } from './metadata';

const chapterIcons = {
  'fundamentals': Shield,
  'adversarial-attacks': AlertTriangle,
  'model-security': Lock,
  'privacy-preserving': Eye,
  'robustness': Brain,
  'security-testing': Search,
  'deployment-security': Server,
  'case-studies': FileWarning
};

export default function AISecurityPage() {
  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          {aiSecurityMetadata.title}
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          {aiSecurityMetadata.description}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {aiSecurityMetadata.chapters.map((chapter) => {
          const Icon = chapterIcons[chapter.id as keyof typeof chapterIcons];
          return (
            <Link
              key={chapter.id}
              href={`/modules/ai-security/${chapter.id}`}
              className="group relative overflow-hidden rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6 shadow-sm transition-all hover:shadow-lg hover:-translate-y-1"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-red-500 to-gray-600 opacity-0 group-hover:opacity-10 transition-opacity" />
              
              <div className="relative">
                <div className="flex items-center justify-between mb-4">
                  <Icon className="w-8 h-8 text-red-600 dark:text-red-400" />
                  <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Chapter {aiSecurityMetadata.chapters.indexOf(chapter) + 1}
                  </span>
                </div>
                
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  {chapter.title}
                </h3>
                
                <p className="text-gray-600 dark:text-gray-300 text-sm">
                  {chapter.description}
                </p>
              </div>
            </Link>
          );
        })}
      </div>

      {/* 시뮬레이터 섹션 */}
      {aiSecurityMetadata.simulators && aiSecurityMetadata.simulators.length > 0 && (
        <div className="mt-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            인터랙티브 시뮬레이터
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {aiSecurityMetadata.simulators.map((simulator) => (
              <Link
                key={simulator.id}
                href={`/modules/ai-security/simulators/${simulator.id}`}
                className="group relative overflow-hidden rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6 shadow-sm transition-all hover:shadow-lg hover:-translate-y-1"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-red-500 to-gray-600 opacity-0 group-hover:opacity-10 transition-opacity" />
                
                <div className="relative">
                  <div className="flex items-center justify-between mb-4">
                    <Shield className="w-8 h-8 text-red-600 dark:text-red-400" />
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                      simulator.difficulty === 'beginner' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      simulator.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      simulator.difficulty === 'advanced' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
                    }`}>
                      {simulator.difficulty === 'beginner' ? '초급' :
                       simulator.difficulty === 'intermediate' ? '중급' :
                       simulator.difficulty === 'advanced' ? '고급' : '전문가'}
                    </span>
                  </div>
                  
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    {simulator.title}
                  </h3>
                  
                  <p className="text-gray-600 dark:text-gray-300 text-sm">
                    {simulator.description}
                  </p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      <div className="mt-12 p-6 bg-red-50 dark:bg-red-950/30 rounded-xl border border-red-200 dark:border-red-800">
        <h2 className="text-2xl font-bold text-red-900 dark:text-red-100 mb-4">
          학습 목표
        </h2>
        <ul className="space-y-2 text-red-800 dark:text-red-200">
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>AI 시스템의 주요 보안 위협과 취약점 이해</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>적대적 공격과 방어 기법 실습</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>프라이버시 보호 머신러닝 구현</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>AI 시스템 보안 평가 및 감사 방법</span>
          </li>
        </ul>
      </div>
    </div>
  );
}