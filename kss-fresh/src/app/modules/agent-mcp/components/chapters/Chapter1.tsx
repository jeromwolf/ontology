'use client';

import React from 'react';
import References from '@/components/common/References';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Agent vs LLM: 근본적인 차이
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            단순한 LLM은 질문에 대답하는 수동적인 시스템입니다. 하지만 <strong>Agent</strong>는 
            목표를 달성하기 위해 능동적으로 행동하고, 도구를 사용하며, 자율적으로 의사결정을 내립니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 my-6">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">일반 LLM</h3>
              <ul className="space-y-2 text-sm">
                <li>• 단순 질의응답</li>
                <li>• 정적인 컨텍스트</li>
                <li>• 단일 턴 상호작용</li>
                <li>• 외부 도구 사용 불가</li>
              </ul>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/30 rounded-lg p-4">
              <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">AI Agent</h3>
              <ul className="space-y-2 text-sm">
                <li>• 목표 지향적 행동</li>
                <li>• 동적 컨텍스트 관리</li>
                <li>• 멀티 턴 작업 수행</li>
                <li>• 도구 사용 및 API 호출</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          ReAct 패턴: Reasoning + Acting
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            ReAct는 Agent가 작업을 수행하는 핵심 패턴입니다. 각 단계에서 Agent는:
          </p>
          <ol className="list-decimal list-inside space-y-2 ml-4">
            <li><strong>Thought (생각)</strong>: 현재 상황을 분석하고 다음 행동을 계획</li>
            <li><strong>Action (행동)</strong>: 도구를 사용하거나 작업을 수행</li>
            <li><strong>Observation (관찰)</strong>: 행동의 결과를 확인</li>
            <li><strong>Repeat</strong>: 목표 달성까지 반복</li>
          </ol>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 mt-4">
            <h4 className="font-semibold mb-2">ReAct 예시</h4>
            <pre className="text-sm bg-gray-900 text-gray-100 p-3 rounded overflow-x-auto">
{`Thought: 사용자가 서울의 날씨를 물어봤다. 날씨 API를 호출해야 한다.
Action: weather_api.get("Seoul")
Observation: {"temp": 15, "condition": "맑음", "humidity": 60}
Thought: 날씨 정보를 받았다. 사용자에게 친근하게 전달하자.
Action: respond("서울은 현재 15도로 선선하고 맑은 날씨입니다!")`}
            </pre>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          🎮 Agent Playground 시뮬레이터
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          ReAct 패턴을 직접 체험해보세요. Agent가 어떻게 생각하고 행동하는지 실시간으로 확인할 수 있습니다.
        </p>
        <div className="text-center p-8 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            시뮬레이터를 보려면 전체 시뮬레이터 페이지를 방문하세요.
          </p>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Tool Use와 Function Calling
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            Agent의 핵심 능력은 도구를 사용하는 것입니다. Function Calling을 통해 Agent는:
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>외부 API 호출 (날씨, 검색, 데이터베이스)</li>
            <li>파일 시스템 조작 (읽기, 쓰기, 생성)</li>
            <li>코드 실행 (Python, JavaScript)</li>
            <li>웹 브라우징 및 스크래핑</li>
          </ul>
          
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mt-4">
            <h4 className="font-semibold mb-2">Tool Definition 예시</h4>
            <pre className="text-sm bg-gray-900 text-gray-100 p-3 rounded overflow-x-auto">
{`{
  "name": "search_web",
  "description": "웹에서 정보를 검색합니다",
  "parameters": {
    "query": {
      "type": "string",
      "description": "검색 쿼리"
    },
    "max_results": {
      "type": "integer",
      "default": 5
    }
  }
}`}
            </pre>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Official Documentation',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'Anthropic Model Context Protocol Documentation',
                authors: 'Anthropic',
                year: '2024',
                description: 'Official MCP specification covering agent architectures, tool use patterns, and implementation guidelines for building AI agents with Claude.',
                link: 'https://modelcontextprotocol.io'
              },
              {
                title: 'OpenAI Function Calling Guide',
                authors: 'OpenAI',
                year: '2023',
                description: 'Comprehensive guide on implementing function calling in LLMs, the foundation for agent tool use and action execution.',
                link: 'https://platform.openai.com/docs/guides/function-calling'
              },
              {
                title: 'LangChain Agent Documentation',
                authors: 'LangChain',
                year: '2024',
                description: 'Complete documentation for building agents with LangChain, including ReAct, Plan-and-Execute, and custom agent architectures.',
                link: 'https://python.langchain.com/docs/modules/agents'
              }
            ]
          },
          {
            title: 'Research Papers',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'ReAct: Synergizing Reasoning and Acting in Language Models',
                authors: 'Yao, S., Zhao, J., Yu, D., et al.',
                year: '2023',
                description: 'Seminal paper introducing the ReAct framework that combines reasoning traces and task-specific actions for improved agent performance.',
                link: 'https://arxiv.org/abs/2210.03629'
              },
              {
                title: 'Toolformer: Language Models Can Teach Themselves to Use Tools',
                authors: 'Schick, T., Dwivedi-Yu, J., Dessì, R., et al.',
                year: '2023',
                description: 'Meta research demonstrating how LLMs can learn to use external tools through self-supervised learning.',
                link: 'https://arxiv.org/abs/2302.04761'
              },
              {
                title: 'Language Models as Agent Models',
                authors: 'Sumers, T., Yao, S., Narasimhan, K., Griffiths, T.',
                year: '2023',
                description: 'Theoretical framework analyzing LLMs as cognitive agents, exploring the transition from passive Q&A systems to goal-oriented agents.',
                link: 'https://arxiv.org/abs/2212.01681'
              }
            ]
          },
          {
            title: 'Implementation Guides',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'Building ReAct Agents from Scratch',
                authors: 'Harrison Chase',
                year: '2024',
                description: 'Practical tutorial implementing ReAct pattern with code examples in Python.',
                link: 'https://blog.langchain.dev/react-agents'
              },
              {
                title: 'Anthropic Claude Function Calling Cookbook',
                authors: 'Anthropic',
                year: '2024',
                description: 'Production-ready examples of implementing tool use with Claude.',
                link: 'https://github.com/anthropics/anthropic-cookbook/tree/main/function_calling'
              },
              {
                title: 'LLM Agent Architecture Patterns',
                authors: 'Eugene Yan',
                year: '2024',
                description: 'Comprehensive guide to different agent patterns including ReAct, Plan-and-Execute, and Reflection agents.',
                link: 'https://eugeneyan.com/writing/llm-agents'
              }
            ]
          },
          {
            title: 'Real-World Applications',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'GitHub Copilot Workspace: AI Agent for Software Development',
                authors: 'GitHub',
                year: '2024',
                description: 'Case study of production AI agents using ReAct patterns for code generation, debugging, and project planning tasks.',
                link: 'https://github.blog/2024-04-29-github-copilot-workspace'
              },
              {
                title: 'Notion AI: Building a Multi-Tool Agent Assistant',
                authors: 'Notion Labs',
                year: '2023',
                description: 'Technical deep-dive into Notion AI agent that uses multiple tools to assist users with knowledge work.',
                link: 'https://www.notion.so/blog/introducing-notion-ai'
              },
              {
                title: 'Perplexity AI: Search Agent Architecture',
                authors: 'Perplexity',
                year: '2024',
                description: 'How Perplexity built a production search agent that combines web search, citation, and answer synthesis.',
                link: 'https://www.perplexity.ai/hub/blog/how-perplexity-works'
              }
            ]
          }
        ]}
      />
    </div>
  );
}