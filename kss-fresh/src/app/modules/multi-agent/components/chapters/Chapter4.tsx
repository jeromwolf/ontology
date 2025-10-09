'use client';

import React from 'react';
import { MessageSquare, Brain, Users, Zap } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      {/* AutoGen 멀티 에이전트 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Microsoft AutoGen: 대화형 멀티 에이전트
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            AutoGen은 <strong>대화를 통해 협력하는 AI 에이전트</strong>를 구축하는 Microsoft의 프레임워크입니다. 
            인간과 AI, AI와 AI 간의 자연스러운 대화를 통해 복잡한 작업을 수행합니다.
          </p>
        </div>
      </section>

      <section className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          AutoGen의 핵심 특징
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <MessageSquare className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">대화형 인터페이스</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              자연어 대화를 통한 에이전트 간 협업
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Brain className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">코드 실행 능력</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Python 코드를 직접 작성하고 실행
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Users className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Human-in-the-loop</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              필요시 인간의 개입과 피드백 지원
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Zap className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">유연한 워크플로우</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              동적으로 변경 가능한 대화 흐름
            </p>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          AutoGen Agent 타입
        </h3>
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">AssistantAgent</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              LLM 기반 대화형 에이전트, 코드 작성 및 문제 해결
            </p>
          </div>
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">UserProxyAgent</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              인간 사용자를 대표하거나 코드 실행을 담당
            </p>
          </div>
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">GroupChatManager</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              여러 에이전트의 그룹 대화를 관리하고 조정
            </p>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          AutoGen 코드 예시
        </h3>
        <div className="bg-gray-900 rounded-xl p-6 text-white">
          <pre className="overflow-x-auto">
            <code className="text-sm">{`import autogen

# Configuration
config_list = [{
    "model": "gpt-4",
    "api_key": "your-api-key"
}]

# Create agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)

critic = autogen.AssistantAgent(
    name="critic",
    system_message="You are a code reviewer.",
    llm_config={"config_list": config_list}
)

# Group chat
groupchat = autogen.GroupChat(
    agents=[assistant, user_proxy, critic],
    messages=[],
    max_round=20
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# Start conversation
user_proxy.initiate_chat(
    manager,
    message="Create a Python function to calculate fibonacci"
)`}</code>
          </pre>
        </div>
      </section>

      {/* 실전 사례 추가 */}
      <section className="bg-gradient-to-r from-purple-100 to-blue-100 dark:from-purple-900/20 dark:to-blue-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          💼 실전 사례: 프로덕션 환경 AutoGen 활용
        </h3>

        <div className="space-y-4">
          {/* 사례 1: 코드 리뷰 자동화 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
              🔍 사례 1: 코드 리뷰 자동화 시스템
            </h4>
            <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <p><strong>요구사항:</strong> GitHub PR 생성 시 자동 코드 리뷰 및 개선 제안</p>
              <div className="ml-4 space-y-1">
                <p>• <strong>Reviewer Agent:</strong> 코드 품질, 보안, 성능 이슈 검토</p>
                <p>• <strong>Tester Agent:</strong> 테스트 커버리지 분석 및 테스트 코드 생성</p>
                <p>• <strong>Documentation Agent:</strong> 코드 문서화 및 README 업데이트</p>
              </div>
              <div className="mt-2 p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>효과:</strong> 리뷰 시간 70% 단축, 버그 발견률 35% 향상
              </div>
            </div>
          </div>

          {/* 사례 2: 고객 지원 자동화 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">
              💬 사례 2: 멀티 에이전트 고객 지원
            </h4>
            <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <p><strong>요구사항:</strong> 복잡한 기술 문의에 대한 자동 응답 시스템</p>
              <div className="ml-4 space-y-1">
                <p>• <strong>Triage Agent:</strong> 문의 내용 분석 및 우선순위 설정</p>
                <p>• <strong>Technical Agent:</strong> 기술 문서 검색 및 해결책 제시</p>
                <p>• <strong>Human Agent:</strong> 복잡한 케이스는 실제 상담원에게 전달</p>
              </div>
              <div className="mt-2 p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>효과:</strong> 응답 시간 80% 단축, 고객 만족도 25% 향상
              </div>
            </div>
          </div>

          {/* 사례 3: 데이터 분석 파이프라인 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">
              📊 사례 3: 자동화된 데이터 분석 워크플로우
            </h4>
            <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <p><strong>요구사항:</strong> 대량의 데이터를 분석하고 인사이트 도출</p>
              <div className="ml-4 space-y-1">
                <p>• <strong>Data Collector:</strong> 다양한 소스에서 데이터 수집</p>
                <p>• <strong>Analyst Agent:</strong> 통계 분석 및 패턴 발견</p>
                <p>• <strong>Visualization Agent:</strong> 차트 생성 및 대시보드 구성</p>
                <p>• <strong>Report Agent:</strong> 경영진을 위한 요약 리포트 작성</p>
              </div>
              <div className="mt-2 p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>효과:</strong> 분석 시간 90% 단축, 인사이트 발견 속도 5배 향상
              </div>
            </div>
          </div>
        </div>

        {/* 비용 최적화 섹션 */}
        <div className="mt-6 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
          <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3 flex items-center gap-2">
            💰 비용 최적화 전략
          </h4>
          <div className="grid md:grid-cols-2 gap-3 text-sm">
            <div>
              <strong>모델 선택 전략:</strong>
              <ul className="ml-4 mt-1 space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 간단한 작업: GPT-3.5-turbo ($0.0015/1K tokens)</li>
                <li>• 복잡한 작업: GPT-4o ($0.0075/1K tokens)</li>
                <li>• 대량 처리: Claude Haiku ($0.00025/1K tokens)</li>
              </ul>
            </div>
            <div>
              <strong>캐싱 활용:</strong>
              <ul className="ml-4 mt-1 space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 시스템 프롬프트 캐싱 (90% 비용 절감)</li>
                <li>• 반복적인 쿼리 결과 저장</li>
                <li>• Context 재사용으로 토큰 절약</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'AutoGen Official Resources',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'AutoGen: Official Documentation',
                description: 'Microsoft AutoGen 공식 문서',
                link: 'https://microsoft.github.io/autogen/'
              },
              {
                title: 'AutoGen GitHub Repository',
                description: 'Microsoft AutoGen 오픈소스 프로젝트',
                link: 'https://github.com/microsoft/autogen'
              },
              {
                title: 'AutoGen Studio: Low-Code Interface',
                description: 'No-code/Low-code AutoGen 개발 환경',
                link: 'https://microsoft.github.io/autogen/docs/autogen-studio/getting-started'
              },
              {
                title: 'AutoGen: API Reference',
                description: '전체 API 레퍼런스 문서',
                link: 'https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent'
              }
            ]
          },
          {
            title: 'AutoGen Research & Papers',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'AutoGen: Enabling Next-Gen LLM Applications',
                authors: 'Qingyun Wu, Gagan Bansal, Jieyu Zhang, et al.',
                year: '2023',
                description: 'AutoGen 프레임워크 소개 논문',
                link: 'https://arxiv.org/abs/2308.08155'
              },
              {
                title: 'An Empirical Study on Challenging Math Problem Solving with GPT-4',
                authors: 'Yiran Wu, Feiran Jia, Shaokun Zhang, et al.',
                year: '2023',
                description: 'AutoGen을 활용한 수학 문제 해결 연구',
                link: 'https://arxiv.org/abs/2306.01337'
              },
              {
                title: 'Large Language Model Guided Tree-of-Thought',
                authors: 'Jieyi Long',
                year: '2023',
                description: 'LLM 기반 Tree-of-Thought reasoning',
                link: 'https://arxiv.org/abs/2305.08291'
              }
            ]
          },
          {
            title: 'Tutorials & Implementation Guides',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'Building Agentic RAG with LlamaIndex',
                description: 'AutoGen과 LlamaIndex를 활용한 RAG 구축',
                link: 'https://microsoft.github.io/autogen/blog/2023/11/13/OAI-assistants'
              },
              {
                title: 'AutoGen Tutorial: Multi-Agent Conversation',
                description: '대화형 multi-agent 시스템 구축 가이드',
                link: 'https://microsoft.github.io/autogen/docs/tutorial/introduction'
              },
              {
                title: 'Code Execution in AutoGen',
                description: '안전한 코드 실행 환경 설정',
                link: 'https://microsoft.github.io/autogen/docs/tutorial/code-executors'
              },
              {
                title: 'Human-in-the-Loop with AutoGen',
                description: '인간 개입 워크플로우 구현',
                link: 'https://microsoft.github.io/autogen/docs/tutorial/human-in-the-loop'
              }
            ]
          },
          {
            title: 'Community & Real-World Applications',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'AutoGen Discord Community',
                description: '활발한 개발자 커뮤니티 및 지원',
                link: 'https://discord.gg/pAbnFJrkgZ'
              },
              {
                title: 'Awesome AutoGen: Curated Resources',
                description: 'AutoGen 관련 리소스 모음',
                link: 'https://github.com/thinkloop/awesome-autogen'
              },
              {
                title: 'AutoGen Examples Gallery',
                description: '실전 예제 코드 컬렉션',
                link: 'https://microsoft.github.io/autogen/docs/Examples'
              },
              {
                title: 'Building Production AutoGen Applications',
                description: '프로덕션 환경 배포 가이드',
                link: 'https://microsoft.github.io/autogen/blog/2024/01/25/AutoGenBench'
              }
            ]
          }
        ]}
      />
    </div>
  );
}