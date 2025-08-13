'use client';

import React from 'react';
import { Users, Target, Network, Settings } from 'lucide-react';

export default function Chapter3() {
  return (
    <div className="space-y-8">
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

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          CrewAI 실전 코드
        </h3>
        <div className="bg-gray-900 rounded-xl p-6 text-white">
          <pre className="overflow-x-auto">
            <code className="text-sm">{`from crewai import Agent, Task, Crew, Process

# 1. 에이전트 정의
researcher = Agent(
    role='Senior Research Analyst',
    goal='Find accurate information about {topic}',
    backstory='Expert researcher with 10 years experience',
    tools=[search_tool, web_scraper],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Create engaging content based on research',
    backstory='Professional writer specializing in tech',
    tools=[writing_tool],
    verbose=True
)

editor = Agent(
    role='Editor',
    goal='Ensure high quality and accuracy',
    backstory='Meticulous editor with attention to detail',
    verbose=True
)

# 2. 작업 정의
research_task = Task(
    description='Research latest trends in {topic}',
    expected_output='Comprehensive research report',
    agent=researcher
)

writing_task = Task(
    description='Write article based on research',
    expected_output='1000-word article',
    agent=writer
)

editing_task = Task(
    description='Edit and polish the article',
    expected_output='Final polished article',
    agent=editor
)

# 3. Crew 구성 및 실행
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={'topic': 'AI Agents'})`}</code>
          </pre>
        </div>
      </section>

      <section className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          💼 실전 사례: 마케팅 캠페인 Crew
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Market Analyst</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              시장 조사 및 타겟 고객 분석
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Creative Director</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              캠페인 컨셉 및 크리에이티브 개발
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Campaign Manager</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              캠페인 실행 계획 및 일정 관리
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}