'use client';

import React from 'react';
import { ArrowRightLeft, Zap, Shield, Code } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      {/* Swarm 소개 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Swarm: 경량 멀티에이전트 프레임워크
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <strong>Swarm</strong>은 OpenAI가 2024년 10월 공개한 실험적 멀티에이전트 프레임워크입니다.
            복잡한 오케스트레이션 없이 <strong>Handoff 패턴</strong>을 통해 간단하게 에이전트 간 작업을 이관할 수 있습니다.
          </p>
        </div>
      </section>

      <section className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Swarm의 핵심 철학
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Zap className="w-6 h-6 text-orange-600 dark:text-orange-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">경량 & 단순</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              복잡한 프레임워크 없이 순수 Python 함수로 에이전트 정의
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <ArrowRightLeft className="w-6 h-6 text-orange-600 dark:text-orange-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Handoff 중심</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              에이전트가 스스로 다른 에이전트에게 작업을 넘기는 패턴
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Shield className="w-6 h-6 text-orange-600 dark:text-orange-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">제어 가능</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              에이전트 행동을 명확히 정의하고 디버깅 용이
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Code className="w-6 h-6 text-orange-600 dark:text-orange-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">교육 목적</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              프로덕션보다는 패턴 학습과 프로토타이핑에 최적화
            </p>
          </div>
        </div>
      </section>

      {/* Handoff 패턴 */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Handoff 패턴
        </h3>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            에이전트가 자신의 작업이 끝나면 다음 에이전트로 컨텍스트를 전달합니다:
          </p>
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`from swarm import Swarm, Agent

client = Swarm()

# Triage Agent (라우팅 담당)
def transfer_to_sales():
    return sales_agent

def transfer_to_support():
    return support_agent

triage_agent = Agent(
    name="Triage Agent",
    instructions="사용자 의도를 파악하고 적절한 팀에 연결",
    functions=[transfer_to_sales, transfer_to_support]
)

# Sales Agent
sales_agent = Agent(
    name="Sales Agent",
    instructions="제품 정보 제공 및 견적 작성"
)

# Support Agent
support_agent = Agent(
    name="Support Agent",
    instructions="기술 지원 및 문제 해결"
)

# 실행
messages = [{"role": "user", "content": "제품 가격이 궁금해요"}]
response = client.run(
    agent=triage_agent,
    messages=messages
)`}
          </pre>
        </div>
      </section>

      {/* Context Variables */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Context Variables
        </h3>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            에이전트 간 공유 정보를 Context Variables로 관리합니다:
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">컨텍스트 정의</h4>
              <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded overflow-x-auto">
{`context_variables = {
    "user_id": "12345",
    "session_id": "abc",
    "cart": [],
    "total_price": 0
}`}
              </pre>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">함수에서 접근</h4>
              <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded overflow-x-auto">
{`def add_to_cart(item, context):
    context["cart"].append(item)
    return f"{item} 추가됨"`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Routines와 Instructions */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Routines와 Instructions
        </h3>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            각 에이전트의 행동을 Instructions로 명확히 정의합니다:
          </p>
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`agent = Agent(
    name="Customer Service Agent",
    instructions="""
    당신은 친절한 고객 서비스 담당자입니다.

    **절차:**
    1. 고객의 문제를 경청하고 공감 표현
    2. 필요한 정보를 정중하게 요청
    3. 문제를 해결하거나 적절한 팀에 전달

    **규칙:**
    - 항상 존댓말 사용
    - 확신이 없으면 전문가에게 전달
    - 고객 정보는 절대 공유하지 않음
    """,
    functions=[transfer_to_tech_support, transfer_to_billing]
)`}
          </pre>
        </div>
      </section>

      {/* 실전 예제 */}
      <section className="bg-gradient-to-r from-orange-100 to-red-100 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          💼 실전 예제: 항공사 고객 서비스
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-10 h-10 bg-orange-500 text-white rounded-full flex items-center justify-center text-sm font-bold">📞</span>
              <div>
                <strong>Triage Agent:</strong> "예약 변경인가요? 취소인가요?"
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  → 의도 파악 후 적절한 에이전트로 handoff
                </div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">✈️</span>
              <div>
                <strong>Flight Agent:</strong> 예약 번호로 항공편 정보 조회
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  → 필요시 Refund Agent로 handoff
                </div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-10 h-10 bg-green-500 text-white rounded-full flex items-center justify-center text-sm font-bold">💳</span>
              <div>
                <strong>Refund Agent:</strong> 환불 정책 확인 및 처리
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  → 완료 후 종료
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded text-sm">
            <strong>Context Variables:</strong> booking_id, customer_tier, refund_amount 등을
            에이전트 간 공유하여 원활한 handoff 구현
          </div>
        </div>
      </section>

      {/* Swarm vs 다른 프레임워크 */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Swarm vs 다른 프레임워크
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-700">
                <th className="px-4 py-2 text-left">특징</th>
                <th className="px-4 py-2 text-left">Swarm</th>
                <th className="px-4 py-2 text-left">LangGraph</th>
                <th className="px-4 py-2 text-left">AutoGen</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-t dark:border-gray-700">
                <td className="px-4 py-2 font-semibold">복잡도</td>
                <td className="px-4 py-2">⭐ 매우 단순</td>
                <td className="px-4 py-2">⭐⭐⭐ 복잡</td>
                <td className="px-4 py-2">⭐⭐ 중간</td>
              </tr>
              <tr className="border-t dark:border-gray-700">
                <td className="px-4 py-2 font-semibold">주요 패턴</td>
                <td className="px-4 py-2">Handoff</td>
                <td className="px-4 py-2">State Graph</td>
                <td className="px-4 py-2">대화형</td>
              </tr>
              <tr className="border-t dark:border-gray-700">
                <td className="px-4 py-2 font-semibold">프로덕션 준비</td>
                <td className="px-4 py-2">❌ 실험적</td>
                <td className="px-4 py-2">✅ 준비됨</td>
                <td className="px-4 py-2">✅ 준비됨</td>
              </tr>
              <tr className="border-t dark:border-gray-700">
                <td className="px-4 py-2 font-semibold">학습 곡선</td>
                <td className="px-4 py-2">낮음</td>
                <td className="px-4 py-2">높음</td>
                <td className="px-4 py-2">중간</td>
              </tr>
              <tr className="border-t dark:border-gray-700">
                <td className="px-4 py-2 font-semibold">최적 용도</td>
                <td className="px-4 py-2">프로토타입, 학습</td>
                <td className="px-4 py-2">복잡한 워크플로우</td>
                <td className="px-4 py-2">코드 생성, 협업</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 한계와 주의사항 */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          ⚠️ 한계와 주의사항
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">실험적 프로젝트</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              OpenAI가 공식적으로 지원하지 않는 실험적 라이브러리. 프로덕션보다는 학습 목적으로 활용
            </p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">기능 제한</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              복잡한 상태 관리, 병렬 실행, 조건부 라우팅 등의 고급 기능은 부족
            </p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">스케일링 어려움</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              대규모 에이전트 시스템에는 LangGraph나 AutoGen이 더 적합
            </p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">모니터링 부족</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              프로덕션 수준의 로깅, 추적, 디버깅 도구가 없음
            </p>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Official Resources',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'Swarm GitHub Repository',
                authors: 'OpenAI',
                year: '2024',
                description: 'Official experimental framework for lightweight multi-agent orchestration using handoff patterns.',
                link: 'https://github.com/openai/swarm'
              },
              {
                title: 'Swarm Documentation',
                authors: 'OpenAI',
                year: '2024',
                description: 'Complete guide to building agent systems with Swarm, including examples and best practices.',
                link: 'https://github.com/openai/swarm/blob/main/README.md'
              },
              {
                title: 'Swarm Examples Gallery',
                authors: 'OpenAI',
                year: '2024',
                description: 'Collection of example implementations including customer service, triage systems, and airline booking.',
                link: 'https://github.com/openai/swarm/tree/main/examples'
              }
            ]
          },
          {
            title: 'Handoff Pattern Research',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Handoff Protocols in Multi-Agent Systems',
                authors: 'Gerkey, B. P., Matarić, M. J.',
                year: '2004',
                description: 'Foundational research on task allocation and handoff mechanisms in multi-robot systems.',
                link: 'https://ieeexplore.ieee.org/document/1389727'
              },
              {
                title: 'Conversational Handoffs in AI Assistants',
                authors: 'Google AI',
                year: '2023',
                description: 'Research on seamless handoffs between specialized AI agents in conversation systems.',
                link: 'https://ai.google/research/pubs/pub52147'
              },
              {
                title: 'Context Preservation in Agent Transitions',
                authors: 'Microsoft Research',
                year: '2023',
                description: 'Study on maintaining context during agent handoffs to preserve conversation quality.',
                link: 'https://www.microsoft.com/en-us/research/publication/context-handoff/'
              }
            ]
          },
          {
            title: 'Implementation Tutorials',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'Building a Customer Service Bot with Swarm',
                authors: 'OpenAI Community',
                year: '2024',
                description: 'Tutorial on implementing triage agent pattern for customer service automation.',
                link: 'https://github.com/openai/swarm/tree/main/examples/customer_service'
              },
              {
                title: 'Airline Booking System Example',
                authors: 'OpenAI',
                year: '2024',
                description: 'Complete implementation of multi-agent flight booking with refund handling.',
                link: 'https://github.com/openai/swarm/tree/main/examples/airline'
              },
              {
                title: 'Swarm vs LangGraph Comparison',
                authors: 'AI Engineering Blog',
                year: '2024',
                description: 'Technical comparison of Swarm and LangGraph for different use cases.',
                link: 'https://www.latent.space/p/swarm'
              }
            ]
          },
          {
            title: 'Alternative Frameworks',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'LangGraph: Production Multi-Agent Framework',
                authors: 'LangChain',
                year: '2024',
                description: 'More robust alternative to Swarm with state management and persistence.',
                link: 'https://langchain-ai.github.io/langgraph/'
              },
              {
                title: 'AutoGen: Conversational Multi-Agent Systems',
                authors: 'Microsoft',
                year: '2023',
                description: 'Framework for building conversational agent systems with code execution.',
                link: 'https://microsoft.github.io/autogen/'
              },
              {
                title: 'CrewAI: Role-Playing Agent Teams',
                authors: 'CrewAI',
                year: '2024',
                description: 'Framework for orchestrating role-playing AI agents in collaborative workflows.',
                link: 'https://docs.crewai.com/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
