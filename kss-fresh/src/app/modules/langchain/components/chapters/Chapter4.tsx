'use client';

import React from 'react';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 4: Agents와 Tools</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          자율적으로 도구를 사용하는 AI 에이전트 구축
        </p>
      </div>

      {/* Agent 개념 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-amber-600 dark:text-amber-400">
          1. Agent란 무엇인가?
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          Agent는 LLM이 도구(Tools)를 사용하여 복잡한 작업을 자율적으로 수행하도록 하는 시스템입니다.
          ReAct(Reasoning + Acting) 패턴을 따릅니다.
        </p>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold text-lg mb-3">🤖 Agent vs Chain</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-2">Chain</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 고정된 순서</li>
                <li>• 결정론적</li>
                <li>• 빠르고 예측 가능</li>
                <li>• 간단한 작업에 적합</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-2">Agent</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 동적 순서</li>
                <li>• 비결정론적</li>
                <li>• 유연하고 강력</li>
                <li>• 복잡한 작업에 적합</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# 1. Tools 정의
from langchain_community.tools import DuckDuckGoSearchRun

tools = [
    DuckDuckGoSearchRun(name="Search"),
]

# 2. LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 3. Prompt 가져오기
prompt = hub.pull("hwchase17/react")

# 4. Agent 생성
agent = create_react_agent(llm, tools, prompt)

# 5. Executor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 6. 실행
result = agent_executor.invoke({
    "input": "2024년 AI 트렌드는?"
})`}
          </pre>
        </div>
      </section>

      {/* ReAct 패턴 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
          2. ReAct 패턴 이해하기
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold text-lg mb-3 text-purple-800 dark:text-purple-200">
            🔄 ReAct 실행 사이클
          </h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold flex-shrink-0">1</div>
              <div>
                <h4 className="font-bold">Thought (생각)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  "날씨 정보가 필요하니 Weather API를 사용해야겠다"
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold flex-shrink-0">2</div>
              <div>
                <h4 className="font-bold">Action (행동)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  Tool: Weather API, Input: "Seoul"
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold flex-shrink-0">3</div>
              <div>
                <h4 className="font-bold">Observation (관찰)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  "Temperature: 15°C, Sunny"
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center font-bold flex-shrink-0">✓</div>
              <div>
                <h4 className="font-bold">Final Answer (최종 답변)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  "서울의 현재 날씨는 맑고 기온은 15도입니다"
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Built-in Tools */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
          3. Built-in Tools 활용
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm">
{`from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    PythonREPLTool,
    ShellTool,
    RequestsGetTool
)
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    SerpAPIWrapper
)

# 1. 웹 검색
search = DuckDuckGoSearchRun()

# 2. Wikipedia
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

# 3. Python 코드 실행
python_repl = PythonREPLTool()

# 4. API 호출
requests_tool = RequestsGetTool()

# Tools 리스트
tools = [search, wikipedia, python_repl, requests_tool]`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
            <h4 className="font-bold text-yellow-800 dark:text-yellow-200 mb-2">
              ⚠️ 보안 주의사항
            </h4>
            <p className="text-sm text-yellow-700 dark:text-yellow-300">
              PythonREPLTool과 ShellTool은 임의 코드를 실행할 수 있어 위험합니다.
              프로덕션에서는 샌드박스 환경에서만 사용하세요!
            </p>
          </div>
        </div>
      </section>

      {/* Custom Tools */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-green-600 dark:text-green-400">
          4. Custom Tool 개발
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          비즈니스 로직을 Tool로 만들어 Agent에 통합할 수 있습니다.
        </p>

        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-bold mb-3">📦 방법 1: @tool 데코레이터</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.tools import tool

@tool
def calculate_product_price(
    base_price: float,
    discount_percent: float
) -> float:
    """제품 가격 계산. 할인율을 적용합니다.

    Args:
        base_price: 원래 가격
        discount_percent: 할인율 (0-100)
    """
    discount = base_price * (discount_percent / 100)
    return base_price - discount

# 사용
tools = [calculate_product_price]`}
              </pre>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-bold mb-3">🏗️ 방법 2: BaseTool 상속</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.tools import BaseTool
from typing import Optional

class CustomerLookupTool(BaseTool):
    name = "customer_lookup"
    description = "고객 ID로 고객 정보를 조회합니다."

    def _run(
        self,
        customer_id: str,
        run_manager: Optional[...] = None
    ) -> str:
        # 실제 DB 쿼리 로직
        customer = db.query_customer(customer_id)
        return f"Name: {customer.name}, Email: {customer.email}"

    async def _arun(self, customer_id: str):
        # 비동기 버전
        raise NotImplementedError("Async not implemented")

tools = [CustomerLookupTool()]`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Agent Types */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-indigo-600 dark:text-indigo-400">
          5. Agent 타입별 특징
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 pl-6">
            <h3 className="text-xl font-bold mb-2">🔵 ReAct Agent</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              가장 범용적. 생각-행동-관찰 사이클 반복.
            </p>
            <div className="bg-gray-100 dark:bg-gray-900 rounded p-3 text-sm">
              <code>create_react_agent(llm, tools, prompt)</code>
            </div>
          </div>

          <div className="border-l-4 border-green-500 pl-6">
            <h3 className="text-xl font-bold mb-2">🟢 OpenAI Functions Agent</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              OpenAI Function Calling 사용. 가장 안정적.
            </p>
            <div className="bg-gray-100 dark:bg-gray-900 rounded p-3 text-sm">
              <code>create_openai_functions_agent(llm, tools, prompt)</code>
            </div>
          </div>

          <div className="border-l-4 border-purple-500 pl-6">
            <h3 className="text-xl font-bold mb-2">🟣 Structured Chat Agent</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              복잡한 입력 처리. 멀티모달 가능.
            </p>
            <div className="bg-gray-100 dark:bg-gray-900 rounded p-3 text-sm">
              <code>create_structured_chat_agent(llm, tools, prompt)</code>
            </div>
          </div>

          <div className="border-l-4 border-orange-500 pl-6">
            <h3 className="text-xl font-bold mb-2">🟠 Plan-and-Execute Agent</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              계획 수립 후 실행. 복잡한 작업에 적합.
            </p>
            <div className="bg-gray-100 dark:bg-gray-900 rounded p-3 text-sm">
              <code>PlanAndExecute(planner, executor, ...)</code>
            </div>
          </div>
        </div>
      </section>

      {/* Error Handling */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-red-600 dark:text-red-400">
          6. Agent 에러 핸들링
        </h2>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,        # 최대 반복 횟수
    max_execution_time=60,    # 최대 실행 시간 (초)
    early_stopping_method="generate",  # 조기 종료
    handle_parsing_errors=True,        # 파싱 에러 처리
    return_intermediate_steps=True     # 중간 단계 반환
)

try:
    result = agent_executor.invoke({"input": "..."})

    # 중간 단계 확인
    for step in result["intermediate_steps"]:
        print(f"Tool: {step[0].tool}")
        print(f"Input: {step[0].tool_input}")
        print(f"Output: {step[1]}")

except Exception as e:
    print(f"Agent failed: {e}")`}
          </pre>
        </div>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-4 text-amber-800 dark:text-amber-200">
          📚 이 챕터에서 배운 것
        </h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Agent의 개념과 ReAct 패턴 이해
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Built-in Tools 활용과 Custom Tool 개발
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              4가지 Agent 타입별 특징과 사용 사례
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Agent 에러 핸들링과 디버깅
            </span>
          </li>
        </ul>
      </section>
    </div>
  );
}
