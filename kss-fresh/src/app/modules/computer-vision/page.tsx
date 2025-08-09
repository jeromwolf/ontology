'use client';

import { useState } from 'react';
import Link from 'next/link';
import { moduleMetadata } from './metadata';
import { 
  BookOpen, 
  PlayCircle, 
  ChevronRight, 
  GraduationCap,
  Sparkles,
  Clock,
  Target,
  Box,
  Scan,
  UserCheck,
  ImagePlus,
  User
} from 'lucide-react';

const iconMap: { [key: string]: any } = {
  Box,
  Scan,
  UserCheck,
  ImagePlus,
  User
};

export default function ComputerVisionPage() {
  const [activeTab, setActiveTab] = useState<'chapters' | 'simulators'>('chapters');

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-r from-teal-500 to-cyan-600 rounded-3xl p-8 md:p-12 text-white overflow-hidden">
        <div className="absolute inset-0 bg-black/10"></div>
        <div className="relative z-10">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            {moduleMetadata.title}
          </h1>
          <p className="text-xl md:text-2xl text-white/90 mb-8 max-w-3xl">
            {moduleMetadata.description}
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
              <GraduationCap className="w-8 h-8 mb-3 text-teal-200" />
              <h3 className="font-semibold text-lg mb-2">체계적 학습</h3>
              <p className="text-white/80">기초부터 최신 딥러닝까지 단계별 학습</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
              <Sparkles className="w-8 h-8 mb-3 text-cyan-200" />
              <h3 className="font-semibold text-lg mb-2">실시간 시뮬레이터</h3>
              <p className="text-white/80">5개의 인터랙티브 비전 처리 도구</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
              <Target className="w-8 h-8 mb-3 text-blue-200" />
              <h3 className="font-semibold text-lg mb-2">실무 응용</h3>
              <p className="text-white/80">AR, 자율주행, 의료 영상 등 실제 활용</p>
            </div>
          </div>
        </div>
      </section>

      {/* Tab Navigation */}
      <div className="flex gap-4 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setActiveTab('chapters')}
          className={`px-6 py-3 font-medium transition-all relative ${
            activeTab === 'chapters'
              ? 'text-teal-600 dark:text-teal-400'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
          }`}
        >
          <span className="flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            챕터 ({moduleMetadata.chapters.length})
          </span>
          {activeTab === 'chapters' && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-teal-600 dark:bg-teal-400"></div>
          )}
        </button>
        <button
          onClick={() => setActiveTab('simulators')}
          className={`px-6 py-3 font-medium transition-all relative ${
            activeTab === 'simulators'
              ? 'text-teal-600 dark:text-teal-400'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
          }`}
        >
          <span className="flex items-center gap-2">
            <PlayCircle className="w-5 h-5" />
            시뮬레이터 ({moduleMetadata.simulators.length})
          </span>
          {activeTab === 'simulators' && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-teal-600 dark:bg-teal-400"></div>
          )}
        </button>
      </div>

      {/* Content */}
      <div className="grid gap-6">
        {activeTab === 'chapters' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {moduleMetadata.chapters.map((chapter, index) => (
              <Link
                key={chapter.id}
                href={`/modules/computer-vision/${chapter.id}`}
                className="group bg-white dark:bg-gray-800 rounded-xl p-6 hover:shadow-lg transition-all hover:-translate-y-1 border border-gray-200 dark:border-gray-700"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center text-teal-600 dark:text-teal-400 font-bold text-lg">
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">
                      {chapter.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      {chapter.description}
                    </p>
                    <div className="space-y-2">
                      {chapter.topics.map((topic, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-500">
                          <div className="w-1.5 h-1.5 bg-teal-400 rounded-full"></div>
                          {topic}
                        </div>
                      ))}
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors mt-1" />
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {moduleMetadata.simulators.map((simulator) => {
              const Icon = iconMap[simulator.icon] || Box;
              return (
                <Link
                  key={simulator.id}
                  href={`/modules/computer-vision/simulators/${simulator.id}`}
                  className="group bg-white dark:bg-gray-800 rounded-xl p-6 hover:shadow-lg transition-all hover:-translate-y-1 border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex flex-col items-center text-center">
                    <div className="w-16 h-16 bg-teal-100 dark:bg-teal-900/30 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                      <Icon className="w-8 h-8 text-teal-600 dark:text-teal-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">
                      {simulator.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">
                      {simulator.description}
                    </p>
                    <div className="mt-4 flex items-center gap-2 text-teal-600 dark:text-teal-400 font-medium">
                      <PlayCircle className="w-4 h-4" />
                      <span className="text-sm">시작하기</span>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>

      {/* Learning Path */}
      <section className="mt-12 bg-gray-50 dark:bg-gray-800/50 rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Clock className="w-6 h-6 text-teal-600 dark:text-teal-400" />
          추천 학습 경로
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-lg mb-3 text-gray-900 dark:text-white">초급자 코스</h3>
            <ol className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <span className="text-teal-600 dark:text-teal-400 font-medium">1.</span>
                컴퓨터 비전 기초 → 이미지 처리 기본
              </li>
              <li className="flex items-start gap-2">
                <span className="text-teal-600 dark:text-teal-400 font-medium">2.</span>
                Image Enhancement Studio에서 실습
              </li>
              <li className="flex items-start gap-2">
                <span className="text-teal-600 dark:text-teal-400 font-medium">3.</span>
                특징점 검출 이론 학습
              </li>
            </ol>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-lg mb-3 text-gray-900 dark:text-white">고급자 코스</h3>
            <ol className="space-y-2 text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <span className="text-teal-600 dark:text-teal-400 font-medium">1.</span>
                딥러닝 비전 → 객체 탐지 알고리즘
              </li>
              <li className="flex items-start gap-2">
                <span className="text-teal-600 dark:text-teal-400 font-medium">2.</span>
                Object Detection Lab 고급 기능 활용
              </li>
              <li className="flex items-start gap-2">
                <span className="text-teal-600 dark:text-teal-400 font-medium">3.</span>
                실시간 응용 프로젝트 구현
              </li>
            </ol>
          </div>
        </div>
      </section>
    </div>
  );
}