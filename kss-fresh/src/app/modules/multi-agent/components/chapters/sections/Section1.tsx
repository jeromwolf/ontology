'use client';

import React from 'react';
import { Users, Target, Network, Settings } from 'lucide-react';

export default function Section1() {
  return (
    <>
      {/* CrewAI 프레임워크 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          CrewAI: 역할 기반 에이전트 오케스트레이션
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            CrewAI는 <strong>인간 조직의 협업 방식을 모방</strong>하여 AI 에이전트들이
            팀으로 작업할 수 있게 하는 프레임워크입니다. 각 에이전트는 명확한 역할, 목표, 배경을 가지고
            협력하여 복잡한 작업을 수행합니다.
          </p>
        </div>
      </section>

      <section className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          CrewAI 핵심 컴포넌트
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Users className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                <h4 className="font-semibold text-gray-900 dark:text-white">Agent</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                특정 역할과 전문성을 가진 AI 워커
              </p>
              <div className="mt-2 p-2 bg-gray-50 dark:bg-gray-700 rounded text-xs">
                <code>role, goal, backstory, tools, llm</code>
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                <h4 className="font-semibold text-gray-900 dark:text-white">Task</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                에이전트가 수행할 구체적인 작업
              </p>
              <div className="mt-2 p-2 bg-gray-50 dark:bg-gray-700 rounded text-xs">
                <code>description, expected_output, agent</code>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Network className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                <h4 className="font-semibold text-gray-900 dark:text-white">Crew</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                에이전트 팀과 작업 워크플로우
              </p>
              <div className="mt-2 p-2 bg-gray-50 dark:bg-gray-700 rounded text-xs">
                <code>agents, tasks, process, verbose</code>
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Settings className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                <h4 className="font-semibold text-gray-900 dark:text-white">Process</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                작업 실행 방식과 순서
              </p>
              <div className="mt-2 p-2 bg-gray-50 dark:bg-gray-700 rounded text-xs">
                <code>sequential, hierarchical, parallel</code>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
