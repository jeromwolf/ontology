'use client';

import Link from 'next/link';
import { ArrowLeft, Code, RefreshCw, Grid3x3, Activity, Box, AlertCircle, FileCode, Target, Clock, Zap } from 'lucide-react';

const simulators = [
  {
    id: 'python-repl',
    title: 'Python REPL Simulator',
    description: 'Interactive Python interpreter with real-time code execution and output visualization',
    icon: Code,
    difficulty: 'Beginner',
    duration: '15 min',
    features: [
      'Live code execution',
      'Syntax highlighting',
      'Error handling & debugging',
      'Variable inspection'
    ],
    color: 'from-blue-500 to-indigo-600',
    order: 1
  },
  {
    id: 'data-type-converter',
    title: 'Data Type Converter',
    description: 'Explore Python data types with interactive conversion and type checking tools',
    icon: RefreshCw,
    difficulty: 'Beginner',
    duration: '10 min',
    features: [
      'All basic data types',
      'Type conversion rules',
      'Memory representation',
      'Best practices'
    ],
    color: 'from-indigo-500 to-purple-600',
    order: 2
  },
  {
    id: 'collection-visualizer',
    title: 'Collection Visualizer',
    description: 'Visualize lists, tuples, sets, and dictionaries with interactive operations',
    icon: Grid3x3,
    difficulty: 'Intermediate',
    duration: '20 min',
    features: [
      'Visual data structures',
      'Common operations demo',
      'Performance comparison',
      'Memory layout'
    ],
    color: 'from-purple-500 to-pink-600',
    order: 3
  },
  {
    id: 'function-tracer',
    title: 'Function Tracer',
    description: 'Step-by-step function execution tracer with call stack visualization',
    icon: Activity,
    difficulty: 'Intermediate',
    duration: '25 min',
    features: [
      'Call stack tracking',
      'Variable scope visualization',
      'Recursion debugger',
      'Execution timeline'
    ],
    color: 'from-pink-500 to-rose-600',
    order: 4
  },
  {
    id: 'oop-diagram-generator',
    title: 'OOP Diagram Generator',
    description: 'Generate UML class diagrams from Python code with inheritance visualization',
    icon: Box,
    difficulty: 'Advanced',
    duration: '30 min',
    features: [
      'Class diagram generation',
      'Inheritance tree',
      'Method overriding tracker',
      'Design pattern detection'
    ],
    color: 'from-rose-500 to-orange-600',
    order: 5
  },
  {
    id: 'exception-simulator',
    title: 'Exception Simulator',
    description: 'Practice exception handling with interactive try-except scenarios',
    icon: AlertCircle,
    difficulty: 'Intermediate',
    duration: '20 min',
    features: [
      'Common exception types',
      'Exception hierarchy',
      'Custom exceptions',
      'Best practices'
    ],
    color: 'from-orange-500 to-amber-600',
    order: 6
  },
  {
    id: 'file-io-playground',
    title: 'File I/O Playground',
    description: 'Hands-on file operations with virtual file system and path manipulation',
    icon: FileCode,
    difficulty: 'Intermediate',
    duration: '25 min',
    features: [
      'File read/write operations',
      'Path manipulation',
      'CSV & JSON handling',
      'Context managers'
    ],
    color: 'from-amber-500 to-yellow-600',
    order: 7
  },
  {
    id: 'coding-challenges',
    title: 'Coding Challenges',
    description: 'Progressive coding challenges with automated testing and hints',
    icon: Target,
    difficulty: 'All Levels',
    duration: '60 min',
    features: [
      '30+ coding challenges',
      'Automated test cases',
      'Progressive hints',
      'Solution explanations'
    ],
    color: 'from-yellow-500 to-green-600',
    order: 8
  }
];

const getDifficultyColor = (difficulty: string) => {
  switch (difficulty.toLowerCase()) {
    case 'beginner': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
    case 'intermediate': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
    case 'advanced': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
    default: return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
  }
};

export default function PythonToolsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900/20 dark:to-purple-900/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/modules/python-programming"
            className="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Python Programming
          </Link>

          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-4 mb-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl">
                <Zap className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Interactive Python Simulators
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  8 hands-on tools to master Python programming
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-4 mt-6">
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <Target className="w-4 h-4" />
                <span>8 Interactive Tools</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <Clock className="w-4 h-4" />
                <span>3+ Hours Total</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <Code className="w-4 h-4" />
                <span>Real-time Execution</span>
              </div>
            </div>
          </div>
        </div>

        {/* Learning Path Guide */}
        <div className="mb-8 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
            Recommended Learning Path
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="text-sm font-medium text-green-600 dark:text-green-400 mb-2">
                1. Beginner Start
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Python REPL → Data Type Converter
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="text-sm font-medium text-yellow-600 dark:text-yellow-400 mb-2">
                2. Intermediate Practice
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Collection Visualizer → Function Tracer → Exception Simulator
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="text-sm font-medium text-red-600 dark:text-red-400 mb-2">
                3. Advanced Mastery
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                OOP Diagram Generator → File I/O → Coding Challenges
              </p>
            </div>
          </div>
        </div>

        {/* Simulators Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {simulators.sort((a, b) => a.order - b.order).map((simulator) => {
            const Icon = simulator.icon;
            return (
              <Link
                key={simulator.id}
                href={`/modules/python-programming/simulators/${simulator.id}`}
                className="group bg-white dark:bg-gray-800 rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border border-gray-200 dark:border-gray-700 overflow-hidden hover:scale-[1.02]"
              >
                <div className={`h-2 bg-gradient-to-r ${simulator.color}`} />

                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 bg-gradient-to-br ${simulator.color} rounded-lg`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex flex-col gap-2">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getDifficultyColor(simulator.difficulty)}`}>
                        {simulator.difficulty}
                      </span>
                      <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                        <Clock className="w-3 h-3" />
                        {simulator.duration}
                      </div>
                    </div>
                  </div>

                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    {simulator.title}
                  </h3>

                  <p className="text-gray-600 dark:text-gray-400 mb-4 text-sm">
                    {simulator.description}
                  </p>

                  <div className="space-y-2">
                    <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Key Features:
                    </div>
                    <ul className="space-y-1.5">
                      {simulator.features.map((feature, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                          <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 flex-shrink-0" />
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="mt-6 flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      Tool #{simulator.order}
                    </span>
                    <span className="text-blue-600 dark:text-blue-400 group-hover:translate-x-1 transition-transform">
                      Launch →
                    </span>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>

        {/* Footer */}
        <div className="mt-12 text-center">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
              Ready to Start Learning?
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Choose a simulator above or follow the recommended learning path
            </p>
            <Link
              href="/modules/python-programming"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all"
            >
              Back to Course Overview
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
