'use client';

import React from 'react';
import References from '@/components/common/References';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Sequential vs Parallel 실행
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            Agent 작업을 효율적으로 조율하는 두 가지 주요 패턴:
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Sequential Execution</h4>
              <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded">
{`Agent A → Agent B → Agent C
✅ 간단한 의존성 관리
✅ 예측 가능한 흐름
❌ 느린 전체 실행 시간`}
              </pre>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Parallel Execution</h4>
              <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded">
{`Agent A ┐
Agent B ├→ Merge
Agent C ┘
✅ 빠른 실행
❌ 복잡한 동기화`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Agent Pipeline 설계
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            복잡한 워크플로우를 위한 파이프라인 설계:
          </p>
          
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`class AgentPipeline:
    def __init__(self):
        self.stages = []
        self.context = {}
    
    def add_stage(self, agent, condition=None):
        """파이프라인에 Agent 스테이지 추가"""
        self.stages.append({
            'agent': agent,
            'condition': condition
        })
    
    async def execute(self, input_data):
        """파이프라인 실행"""
        result = input_data
        
        for stage in self.stages:
            # 조건 확인
            if stage['condition'] and not stage['condition'](result):
                continue
            
            # Agent 실행
            try:
                result = await stage['agent'].run(result, self.context)
            except Exception as e:
                result = await self.handle_error(e, stage, result)
        
        return result
    
    async def handle_error(self, error, stage, data):
        """에러 처리 및 복구"""
        # Retry logic
        # Fallback agent
        # Error logging
        pass`}
          </pre>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Error Handling과 Retry 전략
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            Agent 시스템의 안정성을 위한 에러 처리 패턴:
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-semibold mb-2">Retry 전략</h4>
            <ul className="space-y-2 text-sm">
              <li>📈 <strong>Exponential Backoff</strong>: 2^n 초 간격으로 재시도</li>
              <li>🔄 <strong>Circuit Breaker</strong>: 연속 실패 시 차단</li>
              <li>🎯 <strong>Selective Retry</strong>: 특정 에러만 재시도</li>
              <li>🔀 <strong>Fallback Agent</strong>: 대체 Agent로 전환</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Monitoring과 Observability
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            Agent 시스템의 상태를 추적하고 디버깅하기 위한 도구:
          </p>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <h5 className="font-semibold text-sm mb-1">Metrics</h5>
              <ul className="text-xs space-y-1">
                <li>• Response Time</li>
                <li>• Success Rate</li>
                <li>• Token Usage</li>
              </ul>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <h5 className="font-semibold text-sm mb-1">Logging</h5>
              <ul className="text-xs space-y-1">
                <li>• Agent Decisions</li>
                <li>• Tool Calls</li>
                <li>• Error Traces</li>
              </ul>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <h5 className="font-semibold text-sm mb-1">Tracing</h5>
              <ul className="text-xs space-y-1">
                <li>• Request Flow</li>
                <li>• Agent Chain</li>
                <li>• Latency Analysis</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Workflow & Orchestration',
            icon: 'book',
            color: 'border-blue-500',
            items: [
              {
                title: 'LangGraph Documentation',
                authors: 'LangChain',
                year: '2024',
                description: 'Official guide to building stateful, multi-actor applications with LLMs, including cyclic graphs and state management.',
                link: 'https://langchain-ai.github.io/langgraph/'
              },
              {
                title: 'Temporal.io for AI Agent Workflows',
                authors: 'Temporal Technologies',
                year: '2024',
                description: 'Durable execution framework for building reliable, long-running agent workflows with automatic retries.',
                link: 'https://temporal.io/use-cases/ai-agent-orchestration'
              },
              {
                title: 'Apache Airflow for ML Orchestration',
                authors: 'Apache Software Foundation',
                year: '2024',
                description: 'Platform for programmatically authoring, scheduling, and monitoring agent pipelines as DAGs.',
                link: 'https://airflow.apache.org/docs/apache-airflow/stable/ml-orchestration.html'
              }
            ]
          },
          {
            title: 'Error Handling & Reliability',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Building Reliable Agent Systems',
                authors: 'Anthropic Research',
                year: '2024',
                description: 'Best practices for error handling, retry strategies, and fallback mechanisms in production agent systems.',
                link: 'https://www.anthropic.com/research/reliable-agents'
              },
              {
                title: 'Circuit Breaker Pattern for AI Services',
                authors: 'Microsoft Azure',
                year: '2024',
                description: 'Implementation guide for circuit breakers to prevent cascading failures in agent systems.',
                link: 'https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker'
              },
              {
                title: 'Exponential Backoff and Jitter',
                authors: 'AWS Architecture Blog',
                year: '2023',
                description: 'Strategies for implementing robust retry logic with exponential backoff and jitter.',
                link: 'https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/'
              }
            ]
          },
          {
            title: 'Monitoring & Observability',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'LangSmith: LLM Application Observability',
                authors: 'LangChain',
                year: '2024',
                description: 'Platform for debugging, testing, and monitoring LLM applications with detailed tracing.',
                link: 'https://docs.smith.langchain.com/'
              },
              {
                title: 'OpenTelemetry for AI Agents',
                authors: 'CNCF',
                year: '2024',
                description: 'Observability framework for collecting metrics, logs, and traces from agent systems.',
                link: 'https://opentelemetry.io/docs/instrumentation/js/examples/ai-agents/'
              },
              {
                title: 'Weights & Biases LLM Monitoring',
                authors: 'Weights & Biases',
                year: '2024',
                description: 'Tools for tracking agent performance, token usage, and debugging conversation flows.',
                link: 'https://wandb.ai/site/solutions/llm'
              },
              {
                title: 'Arize AI: LLM Observability Platform',
                authors: 'Arize AI',
                year: '2024',
                description: 'End-to-end observability for LLM applications, including prompt engineering and agent workflows.',
                link: 'https://arize.com/llm-observability/'
              }
            ]
          },
          {
            title: 'Production Patterns',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Designing Data-Intensive Applications (Chapter 8)',
                authors: 'Kleppmann, M.',
                year: '2017',
                description: 'Classic patterns for handling faults, including retries, circuit breakers, and compensating transactions.',
                link: 'https://dataintensive.net/'
              },
              {
                title: 'Agent Pipeline Design Patterns',
                authors: 'Google Cloud',
                year: '2024',
                description: 'Reference architectures for sequential and parallel agent execution patterns.',
                link: 'https://cloud.google.com/architecture/ai-ml/agent-pipelines'
              }
            ]
          }
        ]}
      />
    </div>
  );
}