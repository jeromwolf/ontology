'use client';

import React from 'react';
import References from '@/components/common/References';

export default function Section3() {
  return (
    <>
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

      <References
        sections={[
          {
            title: 'Official Documentation & Frameworks',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'CrewAI: Official Documentation',
                description: 'Role-based AI agent orchestration framework - 공식 문서',
                link: 'https://docs.crewai.com/'
              },
              {
                title: 'CrewAI GitHub Repository',
                description: 'Open-source framework for orchestrating role-playing, autonomous AI agents',
                link: 'https://github.com/joaomdmoura/crewAI'
              },
              {
                title: 'AutoGen: Microsoft Multi-Agent Framework',
                description: 'Microsoft의 multi-agent conversation framework',
                link: 'https://microsoft.github.io/autogen/'
              },
              {
                title: 'LangGraph: Agent Workflow Framework',
                description: 'LangChain 기반 agent workflow orchestration',
                link: 'https://langchain-ai.github.io/langgraph/'
              }
            ]
          },
          {
            title: 'Multi-Agent Systems Research',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Communicative Agents for Software Development',
                authors: 'Chen Qian, Xin Cong, Wei Liu, et al.',
                year: '2023',
                description: 'ChatDev: multi-agent collaboration for software development',
                link: 'https://arxiv.org/abs/2307.07924'
              },
              {
                title: 'MetaGPT: Meta Programming for Multi-Agent Collaborative Framework',
                authors: 'Sirui Hong, Xiawu Zheng, Jonathan Chen, et al.',
                year: '2023',
                description: 'SOPs를 활용한 multi-agent 협업 프레임워크',
                link: 'https://arxiv.org/abs/2308.00352'
              },
              {
                title: 'AutoGen: Enabling Next-Gen LLM Applications',
                authors: 'Qingyun Wu, Gagan Bansal, Jieyu Zhang, et al.',
                year: '2023',
                description: 'Microsoft의 multi-agent conversation framework 논문',
                link: 'https://arxiv.org/abs/2308.08155'
              },
              {
                title: 'Cooperative Multi-Agent Reinforcement Learning',
                authors: 'Kaiqing Zhang, Zhuoran Yang, Tamer Başar',
                year: '2021',
                description: 'Multi-agent 협력 학습의 이론적 기초',
                link: 'https://arxiv.org/abs/1911.10635'
              }
            ]
          },
          {
            title: 'Practical Implementation Guides',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'Building Effective Agents with CrewAI',
                description: 'CrewAI를 활용한 실전 agent 구축 가이드',
                link: 'https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/'
              },
              {
                title: 'LangChain: Multi-Agent Systems',
                description: 'LangChain을 활용한 multi-agent 시스템 구축',
                link: 'https://python.langchain.com/docs/use_cases/agent_simulations/'
              },
              {
                title: 'Comparing Agent Frameworks: CrewAI vs AutoGen',
                description: '주요 agent framework 비교 분석',
                link: 'https://www.e2enetworks.com/blog/crewai-vs-autogen-choosing-the-right-multi-agent-framework'
              },
              {
                title: 'Agent Protocol: Open Standard for AI Agents',
                description: 'AI agent 간 상호작용을 위한 오픈 프로토콜',
                link: 'https://agentprotocol.ai/'
              }
            ]
          },
          {
            title: 'Industry Applications & Case Studies',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Multi-Agent AI in Production: Lessons from Stripe',
                description: 'Stripe의 multi-agent 시스템 실전 사례',
                link: 'https://stripe.com/blog/agent-toolkit'
              },
              {
                title: 'ChatDev: Communicative Agents for Software Development',
                description: '소프트웨어 개발 자동화를 위한 multi-agent 시스템',
                link: 'https://github.com/OpenBMB/ChatDev'
              },
              {
                title: 'BabyAGI: Autonomous Task-Driven Agent',
                description: '자율적 작업 수행을 위한 AI agent 시스템',
                link: 'https://github.com/yoheinakajima/babyagi'
              }
            ]
          }
        ]}
      />
    </>
  );
}
