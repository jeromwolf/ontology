'use client';

import React from 'react';
import { MessageSquare, Brain, Users, Zap } from 'lucide-react';

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
    </div>
  );
}