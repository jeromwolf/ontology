'use client';

import React from 'react';
import References from '@/components/common/References';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Multi-Agent 시스템 아키텍처
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            복잡한 문제를 해결하기 위해 여러 Agent가 협력하는 시스템입니다. 
            각 Agent는 특정 역할과 전문성을 가지고 있으며, 서로 통신하며 작업을 수행합니다.
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h3 className="font-semibold mb-3">Multi-Agent 패턴</h3>
            <div className="space-y-3">
              <div>
                <strong>1. Pipeline Pattern</strong>
                <p className="text-sm mt-1">Agent들이 순차적으로 작업을 처리 (A → B → C)</p>
              </div>
              <div>
                <strong>2. Committee Pattern</strong>
                <p className="text-sm mt-1">여러 Agent가 투표를 통해 의사결정</p>
              </div>
              <div>
                <strong>3. Hierarchical Pattern</strong>
                <p className="text-sm mt-1">Manager Agent가 Worker Agent들을 조율</p>
              </div>
              <div>
                <strong>4. Collaborative Pattern</strong>
                <p className="text-sm mt-1">Agent들이 평등하게 협업하며 문제 해결</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Agent 간 통신 프로토콜
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            Agent들이 효과적으로 협업하기 위한 표준화된 메시지 형식:
          </p>
          
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`interface AgentMessage {
  from: string;        // 발신 Agent ID
  to: string | string[]; // 수신 Agent ID(s)
  type: 'request' | 'response' | 'broadcast';
  content: {
    task: string;      // 작업 설명
    data: any;         // 전달 데이터
    priority: number;  // 우선순위
    deadline?: Date;   // 마감시간
  };
  metadata: {
    timestamp: Date;
    messageId: string;
    correlationId?: string; // 관련 메시지 추적
  };
}`}
          </pre>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          🎮 A2A Orchestrator 시뮬레이터
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          여러 Agent가 협력하여 복잡한 작업을 수행하는 과정을 시각화합니다.
        </p>
        <div className="text-center p-8 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            시뮬레이터를 보려면 전체 시뮬레이터 페이지를 방문하세요.
          </p>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Consensus 메커니즘
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            여러 Agent가 합의에 도달하는 방법:
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Voting Systems</h4>
              <ul className="text-sm space-y-1">
                <li>• Simple Majority (과반수)</li>
                <li>• Weighted Voting (가중 투표)</li>
                <li>• Consensus Threshold (합의 임계값)</li>
                <li>• Veto Power (거부권)</li>
              </ul>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Conflict Resolution</h4>
              <ul className="text-sm space-y-1">
                <li>• Priority-based (우선순위 기반)</li>
                <li>• Expertise-based (전문성 기반)</li>
                <li>• Random Selection (무작위 선택)</li>
                <li>• Human Arbitration (인간 중재)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Research Papers',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Communicative Agents for Software Development',
                authors: 'Qian, C., Cong, X., Yang, C., et al.',
                year: '2023',
                description: 'ChatDev framework demonstrating how multiple AI agents can collaborate on software development through structured communication.',
                link: 'https://arxiv.org/abs/2307.07924'
              },
              {
                title: 'AutoGen: Enabling Next-Gen LLM Applications',
                authors: 'Wu, Q., Bansal, G., Zhang, J., et al.',
                year: '2023',
                description: 'Microsoft\'s framework for building multi-agent conversation systems with customizable and conversable agents.',
                link: 'https://arxiv.org/abs/2308.08155'
              },
              {
                title: 'MetaGPT: Meta Programming for Multi-Agent Systems',
                authors: 'Hong, S., Zheng, X., Chen, J., et al.',
                year: '2023',
                description: 'Framework encoding Standardized Operating Procedures (SOPs) into multi-agent systems for complex task coordination.',
                link: 'https://arxiv.org/abs/2308.00352'
              },
              {
                title: 'Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents',
                authors: 'Talebirad, Y., Nadiri, A.',
                year: '2023',
                description: 'Comprehensive analysis of collaboration patterns and communication protocols in multi-agent LLM systems.',
                link: 'https://arxiv.org/abs/2306.03314'
              }
            ]
          },
          {
            title: 'Agent Frameworks',
            icon: 'book',
            color: 'border-blue-500',
            items: [
              {
                title: 'AutoGen Documentation',
                authors: 'Microsoft Research',
                year: '2024',
                description: 'Official documentation for AutoGen, covering conversable agents, group chat patterns, and human-in-the-loop workflows.',
                link: 'https://microsoft.github.io/autogen/'
              },
              {
                title: 'CrewAI Documentation',
                authors: 'CrewAI',
                year: '2024',
                description: 'Framework for orchestrating role-playing autonomous AI agents in collaborative workflows.',
                link: 'https://docs.crewai.com/'
              },
              {
                title: 'LangGraph Multi-Agent Systems',
                authors: 'LangChain',
                year: '2024',
                description: 'Guide to building multi-agent systems using LangGraph\'s stateful orchestration capabilities.',
                link: 'https://langchain-ai.github.io/langgraph/tutorials/multi_agent/'
              }
            ]
          },
          {
            title: 'Coordination Patterns',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Agent Communication Protocols in Practice',
                authors: 'Anthropic Engineering',
                year: '2024',
                description: 'Best practices for designing communication protocols between AI agents, including message formats and consensus mechanisms.',
                link: 'https://www.anthropic.com/research/agent-communication'
              },
              {
                title: 'Consensus Mechanisms for Multi-Agent AI Systems',
                authors: 'OpenAI Research',
                year: '2024',
                description: 'Analysis of voting systems, conflict resolution, and decision-making strategies in multi-agent environments.',
                link: 'https://openai.com/research/multi-agent-consensus'
              },
              {
                title: 'Building Hierarchical Agent Teams',
                authors: 'DeepMind',
                year: '2023',
                description: 'Strategies for organizing agents in hierarchical structures with manager-worker patterns.',
                link: 'https://deepmind.google/discover/blog/hierarchical-agents/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}