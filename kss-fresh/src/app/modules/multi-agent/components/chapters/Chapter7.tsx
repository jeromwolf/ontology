'use client';

import React from 'react';
import { GitBranch, Workflow, Users, Database } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      {/* LangGraph 소개 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          LangGraph: 상태 기반 멀티에이전트 시스템
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <strong>LangGraph</strong>는 LangChain 팀이 2024년 출시한 차세대 멀티에이전트 프레임워크입니다.
            상태 그래프(State Graph)를 통해 복잡한 에이전트 워크플로우를 직관적으로 설계하고 관리할 수 있습니다.
          </p>
        </div>
      </section>

      <section className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          LangGraph의 핵심 특징
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <GitBranch className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Stateful Graphs</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              에이전트 간 상태를 그래프로 표현하여 복잡한 플로우를 시각화하고 관리
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Workflow className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Conditional Edges</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              조건부 라우팅으로 동적으로 다음 에이전트를 선택
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Users className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Human-in-the-Loop</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              중요한 의사결정 시점에 인간의 개입을 쉽게 통합
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Database className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Persistence</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              상태를 저장하고 복원하여 장기 실행 워크플로우 지원
            </p>
          </div>
        </div>
      </section>

      {/* 상태 그래프 설계 */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          상태 그래프 설계
        </h3>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            LangGraph에서는 노드(Node)와 엣지(Edge)로 에이전트 워크플로우를 정의합니다:
          </p>
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

# 상태 정의
class AgentState(TypedDict):
    messages: list[str]
    next_agent: str
    data: dict

# 그래프 생성
workflow = StateGraph(AgentState)

# 노드 추가 (각 에이전트)
workflow.add_node("researcher", research_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

# 엣지 추가 (워크플로우)
workflow.add_edge("researcher", "writer")
workflow.add_conditional_edges(
    "writer",
    should_continue,  # 조건 함수
    {
        "continue": "reviewer",
        "end": END
    }
)

# 시작점 설정
workflow.set_entry_point("researcher")

# 컴파일
app = workflow.compile()`}
          </pre>
        </div>
      </section>

      {/* Conditional Routing */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          조건부 라우팅
        </h3>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            상태를 기반으로 동적으로 다음 에이전트를 선택할 수 있습니다:
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">조건 함수</h4>
              <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded overflow-x-auto">
{`def route_decision(state):
    if state["confidence"] > 0.8:
        return "approved"
    elif state["needs_review"]:
        return "human_review"
    else:
        return "revise"`}
              </pre>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">엣지 매핑</h4>
              <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded overflow-x-auto">
{`workflow.add_conditional_edges(
    "agent",
    route_decision,
    {
        "approved": END,
        "human_review": "human",
        "revise": "agent"
    }
)`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Human-in-the-Loop */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Human-in-the-Loop 패턴
        </h3>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            중요한 의사결정 시점에 사람의 입력을 받을 수 있습니다:
          </p>
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`from langgraph.checkpoint.memory import MemorySaver

# Checkpointer로 상태 저장
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 실행 중 중단
config = {"configurable": {"thread_id": "1"}}
result = app.invoke(initial_state, config)

# 사람이 검토하고 수정
human_feedback = input("승인하시겠습니까? (y/n): ")
state["human_decision"] = human_feedback

# 이어서 실행
final_result = app.invoke(state, config)`}
          </pre>
        </div>
      </section>

      {/* 실전 예제 */}
      <section className="bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          💼 실전 예제: 콘텐츠 생성 파이프라인
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</span>
              <div>
                <strong>Research Agent:</strong> 주제에 대한 자료 수집 (웹 검색, 문서 분석)
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</span>
              <div>
                <strong>Outline Agent:</strong> 수집한 자료로 아웃라인 작성
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</span>
              <div>
                <strong>Writer Agent:</strong> 아웃라인 기반으로 초안 작성
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center text-sm font-bold">4</span>
              <div>
                <strong>Editor Agent:</strong> 문법, 스타일 검토 및 수정
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center text-sm font-bold">👤</span>
              <div>
                <strong>Human Review:</strong> 최종 승인 또는 수정 요청 (조건부)
              </div>
            </div>
          </div>
          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded text-sm">
            <strong>조건부 라우팅:</strong> Editor의 confidence score가 0.9 이상이면 자동 완료,
            그렇지 않으면 Human Review로 전달
          </div>
        </div>
      </section>

      {/* 성능 최적화 */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          성능 최적화 전략
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">병렬 실행</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              독립적인 노드를 병렬로 실행하여 전체 실행 시간 단축
            </p>
          </div>
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">Checkpointing</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              중간 상태를 저장하여 실패 시 처음부터 다시 실행하지 않음
            </p>
          </div>
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">스트리밍</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              각 노드의 출력을 실시간으로 스트리밍하여 UX 개선
            </p>
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
                title: 'LangGraph Official Documentation',
                authors: 'LangChain',
                year: '2024',
                description: 'Complete guide to building stateful multi-agent systems with LangGraph, including tutorials, API reference, and examples.',
                link: 'https://langchain-ai.github.io/langgraph/'
              },
              {
                title: 'LangGraph Tutorials',
                authors: 'LangChain',
                year: '2024',
                description: 'Step-by-step tutorials for building agent workflows with conditional routing, human-in-the-loop, and persistence.',
                link: 'https://langchain-ai.github.io/langgraph/tutorials/'
              },
              {
                title: 'LangGraph Python API Reference',
                authors: 'LangChain',
                year: '2024',
                description: 'Detailed API documentation for StateGraph, nodes, edges, and checkpointing.',
                link: 'https://langchain-ai.github.io/langgraph/reference/'
              }
            ]
          },
          {
            title: 'Research & Concepts',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'State Machines for Agent Workflows',
                authors: 'Harrison Chase, LangChain Team',
                year: '2024',
                description: 'Blog post introducing the concept of state graphs for modeling complex agent interactions.',
                link: 'https://blog.langchain.dev/langgraph/'
              },
              {
                title: 'Human-in-the-Loop Machine Learning',
                authors: 'Robert Munro',
                year: '2021',
                description: 'Comprehensive guide to integrating human feedback in AI systems, foundational for LangGraph\'s HITL patterns.',
                link: 'https://www.manning.com/books/human-in-the-loop-machine-learning'
              },
              {
                title: 'Workflow Orchestration Patterns',
                authors: 'Gregor Hohpe, Bobby Woolf',
                year: '2003',
                description: 'Classic enterprise integration patterns applicable to agent orchestration.',
                link: 'https://www.enterpriseintegrationpatterns.com/'
              }
            ]
          },
          {
            title: 'Implementation Examples',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'LangGraph Example Gallery',
                authors: 'LangChain Community',
                year: '2024',
                description: 'Collection of production-ready examples including customer support, research assistant, and code generation workflows.',
                link: 'https://github.com/langchain-ai/langgraph/tree/main/examples'
              },
              {
                title: 'Building a Multi-Agent Content Pipeline with LangGraph',
                authors: 'LangChain',
                year: '2024',
                description: 'Tutorial on building a research → outline → write → edit pipeline with conditional routing.',
                link: 'https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/'
              },
              {
                title: 'Persistence and Checkpointing in LangGraph',
                authors: 'LangChain',
                year: '2024',
                description: 'Guide to implementing state persistence for long-running workflows.',
                link: 'https://langchain-ai.github.io/langgraph/how-tos/persistence/'
              }
            ]
          },
          {
            title: 'Production Use Cases',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Customer Support Agent with LangGraph',
                authors: 'LangChain',
                year: '2024',
                description: 'Case study of building a production customer support system with escalation paths and human handoff.',
                link: 'https://blog.langchain.dev/customer-support-bot-langraph/'
              },
              {
                title: 'Code Generation Pipeline Using LangGraph',
                authors: 'LangChain',
                year: '2024',
                description: 'Multi-agent system for planning, coding, testing, and reviewing code with human approval gates.',
                link: 'https://github.com/langchain-ai/langgraph/tree/main/examples/code-assistant'
              },
              {
                title: 'Research Assistant with Conditional Routing',
                authors: 'LangChain Community',
                year: '2024',
                description: 'Intelligent research assistant that dynamically routes to web search, document retrieval, or calculation tools.',
                link: 'https://github.com/langchain-ai/langgraph/tree/main/examples/research-assistant'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
