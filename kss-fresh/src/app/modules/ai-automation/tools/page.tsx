'use client';

import Link from 'next/link';
import { ArrowLeft, Sparkles, FileText, Code2, Users, Clock, TrendingUp, Award } from 'lucide-react';

const simulators = [
  {
    id: 'prompt-optimizer',
    title: 'Prompt Optimizer',
    description: 'AI prompt patterns and optimization techniques',
    icon: Sparkles,
    difficulty: 'Beginner',
    duration: '15 min',
    features: [
      '5 prompt patterns',
      '15 best practice templates',
      'Token & cost calculator',
      'Before/After comparison'
    ],
    color: 'from-violet-500 to-purple-600',
    bgColor: 'bg-violet-50 dark:bg-violet-900/20',
    order: 1
  },
  {
    id: 'context-manager',
    title: 'Context Manager',
    description: 'CLAUDE.md authoring and project context management',
    icon: FileText,
    difficulty: 'Intermediate',
    duration: '20 min',
    features: [
      '6 section templates',
      '200K token budget manager',
      'Optimization recommendations',
      'Real-time section checklist'
    ],
    color: 'from-blue-500 to-cyan-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    order: 2
  },
  {
    id: 'code-generator',
    title: 'Multi-AI Code Generator',
    description: 'Generate and compare code from Claude, GPT-4, and Gemini',
    icon: Code2,
    difficulty: 'Intermediate',
    duration: '25 min',
    features: [
      '3 AI models comparison',
      'Diff comparison system',
      'Quality metrics analysis',
      '10 code templates'
    ],
    color: 'from-green-500 to-emerald-600',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
    order: 3
  },
  {
    id: 'workflow-builder',
    title: 'AI Workflow Builder',
    description: 'Visual AI automation workflow construction',
    icon: Users,
    difficulty: 'Advanced',
    duration: '30 min',
    features: [
      'Canvas-based node editor',
      '8 node types',
      'JSON/YAML code generation',
      '6 workflow templates'
    ],
    color: 'from-orange-500 to-red-600',
    bgColor: 'bg-orange-50 dark:bg-orange-900/20',
    order: 4
  }
];

const difficultyColors = {
  'Beginner': 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
  'Intermediate': 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
  'Advanced': 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'
};

export default function AIAutomationToolsPage() {
  const sortedSimulators = [...simulators].sort((a, b) => a.order - b.order);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Navigation */}
      <div className="mb-8">
        <Link
          href="/modules/ai-automation"
          className="inline-flex items-center text-violet-600 dark:text-violet-400 hover:text-violet-700 dark:hover:text-violet-300 transition-colors"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to AI Automation
        </Link>
      </div>

      {/* Header */}
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          AI Automation Simulators
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 mb-6">
          Hands-on experience with AI automation tools
        </p>

        {/* Learning Path Guide */}
        <div className="bg-gradient-to-r from-violet-100 to-purple-100 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-violet-600 dark:text-violet-400" />
            Recommended Learning Path
          </h2>
          <div className="flex flex-wrap gap-2 text-sm">
            {sortedSimulators.map((sim, index) => (
              <div key={sim.id} className="flex items-center">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300">
                  <span className="font-semibold">{index + 1}.</span>
                  {sim.title}
                </span>
                {index < sortedSimulators.length - 1 && (
                  <span className="mx-2 text-gray-400">→</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Simulator Cards Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {sortedSimulators.map((simulator) => {
          const Icon = simulator.icon;

          return (
            <Link
              key={simulator.id}
              href={`/modules/ai-automation/simulators/${simulator.id}`}
              className="group"
            >
              <div className="h-full bg-white dark:bg-gray-800 rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 overflow-hidden border border-gray-200 dark:border-gray-700 hover:border-violet-500 dark:hover:border-violet-500">
                {/* Card Header with Gradient */}
                <div className={`bg-gradient-to-r ${simulator.color} p-6 text-white`}>
                  <div className="flex items-start justify-between mb-3">
                    <Icon className="w-10 h-10" />
                    <div className="flex gap-2">
                      <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${difficultyColors[simulator.difficulty]}`}>
                        {simulator.difficulty}
                      </span>
                    </div>
                  </div>
                  <h3 className="text-2xl font-bold mb-2 group-hover:underline">
                    {simulator.title}
                  </h3>
                  <p className="text-white/90 text-sm">
                    {simulator.description}
                  </p>
                </div>

                {/* Card Body */}
                <div className="p-6">
                  {/* Duration */}
                  <div className="flex items-center gap-2 mb-4 text-gray-600 dark:text-gray-400">
                    <Clock className="w-4 h-4" />
                    <span className="text-sm">Duration: {simulator.duration}</span>
                  </div>

                  {/* Features */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                      <Award className="w-4 h-4 text-violet-600 dark:text-violet-400" />
                      Key Features
                    </h4>
                    <ul className="space-y-1.5">
                      {simulator.features.map((feature, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                          <span className="text-violet-600 dark:text-violet-400 mt-0.5">✓</span>
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* CTA */}
                  <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <span className="text-violet-600 dark:text-violet-400 font-medium group-hover:underline flex items-center gap-2">
                      Start Simulator
                      <ArrowLeft className="w-4 h-4 rotate-180 group-hover:translate-x-1 transition-transform" />
                    </span>
                  </div>
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {/* Additional Info */}
      <div className="mt-12 bg-gray-50 dark:bg-gray-800/50 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          Learning Tips
        </h2>
        <ul className="space-y-2 text-gray-600 dark:text-gray-400">
          <li className="flex items-start gap-2">
            <span className="text-violet-600 dark:text-violet-400 mt-1">•</span>
            <span>Each simulator can be learned independently, but following the order helps build systematic understanding</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-violet-600 dark:text-violet-400 mt-1">•</span>
            <span>Experiment with different parameters within each simulator</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-violet-600 dark:text-violet-400 mt-1">•</span>
            <span>Combine theory from chapters with hands-on simulator practice</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-violet-600 dark:text-violet-400 mt-1">•</span>
            <span>Apply what you learn to real projects to maximize learning effectiveness</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
