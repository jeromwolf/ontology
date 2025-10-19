'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Code2, Zap, BookOpen, Play, CheckCircle2, Clock, Award, TrendingUp } from 'lucide-react';
import { moduleMetadata } from './metadata';

export default function PythonProgrammingPage() {
  const [progress, setProgress] = useState<Record<number, boolean>>({});

  useEffect(() => {
    const saved = localStorage.getItem('python-progress');
    if (saved) {
      setProgress(JSON.parse(saved));
    }
  }, []);

  const completedCount = Object.values(progress).filter(Boolean).length;
  const progressPercent = (completedCount / moduleMetadata.chapters.length) * 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900/10 dark:to-gray-900">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Hero Section */}
        <div className="mb-12">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center">
              <Code2 className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
                {moduleMetadata.title}
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-400 mt-2">
                {moduleMetadata.description}
              </p>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 mb-1">
                <Clock className="w-4 h-4" />
                <span className="text-sm">Duration</span>
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {moduleMetadata.duration}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 mb-1">
                <BookOpen className="w-4 h-4" />
                <span className="text-sm">Chapters</span>
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {moduleMetadata.chapters.length}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 mb-1">
                <Play className="w-4 h-4" />
                <span className="text-sm">Simulators</span>
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {moduleMetadata.simulators.length}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 mb-1">
                <Award className="w-4 h-4" />
                <span className="text-sm">Level</span>
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                {moduleMetadata.level}
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                Your Progress
              </span>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {completedCount} / {moduleMetadata.chapters.length} chapters
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-blue-500 to-indigo-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        </div>

        {/* Learning Path */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            Learning Path
          </h2>

          <div className="grid md:grid-cols-3 gap-6">
            {/* Beginner */}
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 border border-green-200 dark:border-green-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
                Beginner
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Chapters 1-4 • 3.5 hours
              </p>
              <ul className="space-y-2 text-sm">
                {moduleMetadata.chapters.slice(0, 4).map((chapter) => (
                  <li key={chapter.id} className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400 mt-0.5" />
                    <span className="text-gray-700 dark:text-gray-300">{chapter.title}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Intermediate */}
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
                Intermediate
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Chapters 5-7 • 3.5 hours
              </p>
              <ul className="space-y-2 text-sm">
                {moduleMetadata.chapters.slice(4, 7).map((chapter) => (
                  <li key={chapter.id} className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-0.5" />
                    <span className="text-gray-700 dark:text-gray-300">{chapter.title}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Advanced */}
            <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-700">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
                Advanced
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Chapters 8-10 • 3.5 hours
              </p>
              <ul className="space-y-2 text-sm">
                {moduleMetadata.chapters.slice(7, 10).map((chapter) => (
                  <li key={chapter.id} className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-purple-600 dark:text-purple-400 mt-0.5" />
                    <span className="text-gray-700 dark:text-gray-300">{chapter.title}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Chapters */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Course Chapters
          </h2>

          <div className="space-y-4">
            {moduleMetadata.chapters.map((chapter) => (
              <Link
                key={chapter.id}
                href={`/modules/python-programming/${chapter.slug}`}
                className="block"
              >
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 transition-all hover:shadow-lg group">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
                          Chapter {chapter.id}
                        </span>
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          chapter.difficulty === 'beginner'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                            : chapter.difficulty === 'intermediate'
                            ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
                            : 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'
                        }`}>
                          {chapter.difficulty}
                        </span>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          {chapter.duration}
                        </span>
                      </div>
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400">
                        {chapter.title}
                      </h3>
                      <ul className="space-y-1">
                        {chapter.learningObjectives.map((obj, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                            <span className="text-blue-600 dark:text-blue-400 mt-0.5">•</span>
                            <span>{obj}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    {progress[chapter.id] && (
                      <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400" />
                    )}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* Simulators */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
              <Zap className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              Interactive Simulators
            </h2>
            <Link
              href="/modules/python-programming/tools"
              className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium"
            >
              View All →
            </Link>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {moduleMetadata.simulators.map((simulator) => (
              <Link
                key={simulator.id}
                href={`/modules/python-programming/simulators/${simulator.id}`}
                className="group"
              >
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 transition-all hover:shadow-lg">
                  <Play className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-3" />
                  <h3 className="font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400">
                    {simulator.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {simulator.description}
                  </p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
