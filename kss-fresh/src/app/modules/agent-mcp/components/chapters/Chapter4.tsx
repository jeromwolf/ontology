'use client';

import React from 'react';
import References from '@/components/common/References';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          LangChain Agent 구현
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            LangChain은 가장 인기 있는 Agent 개발 프레임워크입니다. 
            다양한 도구와 LLM을 쉽게 통합할 수 있습니다.
          </p>
          
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 도구 정의
tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="수학 계산을 수행합니다"
    ),
    Tool(
        name="Search",
        func=search_web,
        description="웹에서 정보를 검색합니다"
    )
]

# Agent 생성
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 실행
result = agent_executor.invoke({
    "input": "서울 인구의 제곱근은?"
})`}
          </pre>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          AutoGPT 아키텍처 분석
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            AutoGPT는 완전 자율적인 Agent 시스템의 선구자입니다. 주요 컴포넌트:
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Core Components</h4>
              <ul className="text-sm space-y-1">
                <li>🧠 <strong>Planning</strong>: 작업 계획 수립</li>
                <li>💾 <strong>Memory</strong>: 장/단기 기억 관리</li>
                <li>🔧 <strong>Tools</strong>: 도구 실행 엔진</li>
                <li>🔄 <strong>Reflection</strong>: 자기 평가 및 개선</li>
              </ul>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Execution Flow</h4>
              <ol className="text-sm space-y-1">
                <li>1. Goal Setting (목표 설정)</li>
                <li>2. Task Decomposition (작업 분해)</li>
                <li>3. Action Execution (행동 실행)</li>
                <li>4. Result Evaluation (결과 평가)</li>
                <li>5. Plan Adjustment (계획 조정)</li>
              </ol>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          CrewAI로 팀 에이전트 구성
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            CrewAI는 여러 Agent가 팀으로 협업하는 시스템을 쉽게 구축할 수 있게 해줍니다:
          </p>
          
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`from crewai import Agent, Task, Crew

# Agent 정의
researcher = Agent(
    role='연구원',
    goal='정확한 정보 수집',
    backstory='데이터 분석 전문가',
    tools=[search_tool, scrape_tool]
)

writer = Agent(
    role='작가',
    goal='명확한 콘텐츠 작성',
    backstory='기술 문서 전문가',
    tools=[write_tool]
)

# Task 정의
research_task = Task(
    description='AI Agent에 대해 조사',
    agent=researcher
)

write_task = Task(
    description='조사 내용을 블로그 포스트로 작성',
    agent=writer
)

# Crew 구성
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

# 실행
result = crew.kickoff()`}
          </pre>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Custom Agent Framework 설계
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            특정 요구사항에 맞는 커스텀 Agent 프레임워크 설계 원칙:
          </p>
          
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h4 className="font-semibold mb-2">설계 고려사항</h4>
            <ul className="space-y-2">
              <li>✓ <strong>Modularity</strong>: 컴포넌트 교체 가능성</li>
              <li>✓ <strong>Scalability</strong>: 다중 Agent 지원</li>
              <li>✓ <strong>Observability</strong>: 디버깅과 모니터링</li>
              <li>✓ <strong>Safety</strong>: 안전장치와 제한사항</li>
              <li>✓ <strong>Performance</strong>: 효율적인 리소스 관리</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Official Framework Documentation',
            icon: 'book',
            color: 'border-blue-500',
            items: [
              {
                title: 'LangChain Agents Documentation',
                authors: 'LangChain',
                year: '2024',
                description: 'Comprehensive guide to building agents with LangChain, including ReAct pattern, tool integration, and memory systems.',
                link: 'https://python.langchain.com/docs/modules/agents/'
              },
              {
                title: 'AutoGPT Repository and Documentation',
                authors: 'Significant Gravitas',
                year: '2024',
                description: 'Open-source autonomous AI agent with self-directed task planning and execution capabilities.',
                link: 'https://github.com/Significant-Gravitas/AutoGPT'
              },
              {
                title: 'CrewAI Documentation',
                authors: 'CrewAI',
                year: '2024',
                description: 'Framework for orchestrating role-playing autonomous AI agents working together on complex tasks.',
                link: 'https://docs.crewai.com/'
              },
              {
                title: 'BabyAGI: Task-Driven Autonomous Agent',
                authors: 'Nakajima, Y.',
                year: '2023',
                description: 'Minimalist autonomous agent system demonstrating task creation, prioritization, and execution.',
                link: 'https://github.com/yoheinakajima/babyagi'
              }
            ]
          },
          {
            title: 'Research & Architecture',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'ReAct: Synergizing Reasoning and Acting in Language Models',
                authors: 'Yao, S., Zhao, J., Yu, D., et al.',
                year: '2022',
                description: 'Foundational paper on the ReAct pattern used in LangChain agents, combining reasoning traces with task-specific actions.',
                link: 'https://arxiv.org/abs/2210.03629'
              },
              {
                title: 'Cognitive Architectures for Language Agents',
                authors: 'Sumers, T. R., Yao, S., Narasimhan, K., et al.',
                year: '2023',
                description: 'Analysis of cognitive architectures (planning, memory, tools, reflection) in autonomous language agents.',
                link: 'https://arxiv.org/abs/2309.02427'
              },
              {
                title: 'A Survey on Large Language Model based Autonomous Agents',
                authors: 'Wang, L., Ma, C., Feng, X., et al.',
                year: '2023',
                description: 'Comprehensive survey covering agent construction, application domains, and evaluation methods.',
                link: 'https://arxiv.org/abs/2308.11432'
              }
            ]
          },
          {
            title: 'Implementation Guides',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Building Production-Ready LangChain Agents',
                authors: 'LangChain Blog',
                year: '2024',
                description: 'Best practices for deploying LangChain agents in production, including error handling and monitoring.',
                link: 'https://blog.langchain.dev/production-ready-agents/'
              },
              {
                title: 'CrewAI Tutorial: Building Your First Agent Crew',
                authors: 'CrewAI Community',
                year: '2024',
                description: 'Step-by-step tutorial for creating multi-agent systems with CrewAI, from basic to advanced patterns.',
                link: 'https://docs.crewai.com/getting-started/start-building'
              },
              {
                title: 'Custom Agent Framework Design Patterns',
                authors: 'Anthropic Developer Relations',
                year: '2024',
                description: 'Guide to designing custom agent frameworks, covering modularity, scalability, and observability.',
                link: 'https://www.anthropic.com/research/custom-agent-frameworks'
              }
            ]
          },
          {
            title: 'Comparisons & Benchmarks',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Agent Framework Comparison: LangChain vs AutoGPT vs CrewAI',
                authors: 'Towards Data Science',
                year: '2024',
                description: 'Detailed comparison of popular agent frameworks, evaluating ease of use, flexibility, and performance.',
                link: 'https://towardsdatascience.com/agent-framework-comparison'
              },
              {
                title: 'AgentBench: Evaluating LLMs as Agents',
                authors: 'Liu, X., Yu, H., Zhang, H., et al.',
                year: '2023',
                description: 'Systematic benchmark for evaluating LLMs in agent tasks across diverse environments.',
                link: 'https://arxiv.org/abs/2308.03688'
              }
            ]
          }
        ]}
      />
    </div>
  );
}