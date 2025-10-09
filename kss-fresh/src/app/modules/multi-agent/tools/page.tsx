'use client';

import Link from 'next/link';
import { ArrowLeft, Network, Users, Vote, GitBranch, Zap, Clock, TrendingUp, Award } from 'lucide-react';

const simulators = [
  {
    id: 'a2a-orchestrator',
    title: 'A2A Communication Orchestrator',
    description: 'Agent 간 통신과 협업 워크플로우를 시뮬레이션합니다',
    icon: Network,
    difficulty: '초급',
    duration: '20분',
    features: [
      '17개 에이전트 타입',
      '4개 워크플로우 패턴',
      '실시간 네트워크 시각화',
      '성능 메트릭 대시보드'
    ],
    color: 'from-orange-500 to-red-500',
    bgColor: 'bg-orange-50 dark:bg-orange-900/20',
    order: 1
  },
  {
    id: 'crewai-builder',
    title: 'CrewAI Team Builder',
    description: '역할 기반 AI 에이전트 팀을 구성하고 작업을 할당합니다',
    icon: Users,
    difficulty: '중급',
    duration: '30분',
    features: [
      '6개 전문 팀 템플릿',
      'Canvas 조직도 시각화',
      'Python 코드 생성',
      'Sequential & Parallel 프로세스'
    ],
    color: 'from-blue-500 to-cyan-500',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    order: 2
  },
  {
    id: 'consensus-simulator',
    title: 'Distributed Consensus Simulator',
    description: '분산 합의 알고리즘(Raft, Paxos, PBFT)을 시뮬레이션합니다',
    icon: Vote,
    difficulty: '고급',
    duration: '40분',
    features: [
      'Raft 리더 선출 & 로그 복제',
      'Paxos 2-phase 프로토콜',
      'PBFT 3-phase 합의',
      '5가지 장애 시나리오'
    ],
    color: 'from-green-500 to-emerald-500',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
    order: 4
  },
  {
    id: 'langgraph-workflow',
    title: 'LangGraph Workflow Builder',
    description: 'LangGraph를 활용한 복잡한 에이전트 워크플로우를 구축합니다',
    icon: GitBranch,
    difficulty: '중급',
    duration: '25분',
    features: [
      '그래프 기반 워크플로우',
      '조건부 라우팅',
      '상태 관리',
      '시각적 에디터'
    ],
    color: 'from-purple-500 to-pink-500',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
    order: 3
  },
  {
    id: 'swarm-handoff',
    title: 'Swarm Handoff Visualizer',
    description: 'OpenAI Swarm 프레임워크의 에이전트 핸드오프를 시각화합니다',
    icon: Zap,
    difficulty: '중급',
    duration: '20분',
    features: [
      '동적 에이전트 핸드오프',
      '컨텍스트 전달 시각화',
      '실시간 워크플로우',
      'Swarm 패턴 분석'
    ],
    color: 'from-yellow-500 to-orange-500',
    bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
    order: 5
  }
];

const difficultyColors = {
  '초급': 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
  '중급': 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
  '고급': 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'
};

export default function MultiAgentToolsPage() {
  const sortedSimulators = [...simulators].sort((a, b) => a.order - b.order);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Navigation */}
      <div className="mb-8">
        <Link
          href="/modules/multi-agent"
          className="inline-flex items-center text-orange-600 dark:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 transition-colors"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          멀티 에이전트 시스템으로 돌아가기
        </Link>
      </div>

      {/* Header */}
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Multi-Agent System Simulators
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 mb-6">
          멀티 에이전트 시스템의 핵심 개념을 직접 체험해보세요
        </p>

        {/* Learning Path Guide */}
        <div className="bg-gradient-to-r from-orange-100 to-yellow-100 dark:from-orange-900/20 dark:to-yellow-900/20 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            추천 학습 순서
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
              href={`/modules/multi-agent/tools/${simulator.id}`}
              className="group"
            >
              <div className="h-full bg-white dark:bg-gray-800 rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 overflow-hidden border border-gray-200 dark:border-gray-700 hover:border-orange-500 dark:hover:border-orange-500">
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
                    <span className="text-sm">예상 학습 시간: {simulator.duration}</span>
                  </div>

                  {/* Features */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                      <Award className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                      핵심 기능
                    </h4>
                    <ul className="space-y-1.5">
                      {simulator.features.map((feature, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                          <span className="text-orange-600 dark:text-orange-400 mt-0.5">✓</span>
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* CTA */}
                  <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <span className="text-orange-600 dark:text-orange-400 font-medium group-hover:underline flex items-center gap-2">
                      시뮬레이터 시작하기
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
          💡 학습 팁
        </h2>
        <ul className="space-y-2 text-gray-600 dark:text-gray-400">
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-1">•</span>
            <span>각 시뮬레이터는 독립적으로 학습 가능하지만, 순서대로 진행하면 체계적인 이해가 가능합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-1">•</span>
            <span>시뮬레이터 내 파라미터를 직접 조정하며 실험해보세요</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-1">•</span>
            <span>각 챕터의 이론과 시뮬레이터를 병행 학습하면 효과적입니다</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
