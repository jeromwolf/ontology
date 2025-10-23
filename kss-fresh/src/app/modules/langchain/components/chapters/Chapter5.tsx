'use client';

import React from 'react';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 5: LangGraph 시작하기</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          State Graph 기반의 차세대 워크플로우 프레임워크
        </p>
      </div>

      {/* LangGraph 소개 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-amber-600 dark:text-amber-400">
          1. LangGraph란?
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          LangGraph는 LangChain팀이 개발한 상태 기반 워크플로우 프레임워크입니다.
          복잡한 멀티 에이전트 시스템과 조건부 분기를 쉽게 구현할 수 있습니다.
        </p>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold text-lg mb-3">🆚 LangChain vs LangGraph</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-2">LangChain</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 선형/순차적 워크플로우</li>
                <li>• 단순한 체인 구성</li>
                <li>• 빠른 프로토타이핑</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-2">LangGraph</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 복잡한 그래프 구조</li>
                <li>• 조건부 분기와 루프</li>
                <li>• 멀티 에이전트 협업</li>
                <li>• 사람 개입(Human-in-the-loop)</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`# LangGraph 설치
pip install langgraph

# 기본 사용
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. State 정의
class State(TypedDict):
    message: str
    count: int

# 2. Node 함수 정의
def node_a(state: State) -> State:
    return {"message": "Node A", "count": state["count"] + 1}

def node_b(state: State) -> State:
    return {"message": "Node B", "count": state["count"] + 1}

# 3. Graph 생성
workflow = StateGraph(State)
workflow.add_node("a", node_a)
workflow.add_node("b", node_b)

# 4. Edge 연결
workflow.set_entry_point("a")
workflow.add_edge("a", "b")
workflow.add_edge("b", END)

# 5. 컴파일 및 실행
app = workflow.compile()
result = app.invoke({"message": "start", "count": 0})`}
          </pre>
        </div>
      </section>

      {/* Core Concepts */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
          2. 핵심 개념
        </h2>

        <div className="space-y-6">
          <div className="border-l-4 border-purple-500 pl-6">
            <h3 className="text-xl font-bold mb-3">📊 State (상태)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              그래프 전체에서 공유되는 데이터 구조. TypedDict로 정의.
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]
    current_agent: str
    iteration: int
    final_answer: str`}
              </pre>
            </div>
          </div>

          <div className="border-l-4 border-green-500 pl-6">
            <h3 className="text-xl font-bold mb-3">🔵 Nodes (노드)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              State를 받아 처리하고 업데이트된 State를 반환하는 함수.
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`def research_node(state: AgentState) -> AgentState:
    # 웹 검색 수행
    results = search_web(state["messages"][-1])

    return {
        "messages": state["messages"] + [results],
        "current_agent": "researcher"
    }`}
              </pre>
            </div>
          </div>

          <div className="border-l-4 border-blue-500 pl-6">
            <h3 className="text-xl font-bold mb-3">➡️ Edges (엣지)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              노드 간 연결. 일반 엣지와 조건부 엣지 2가지.
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`# 일반 엣지
workflow.add_edge("node_a", "node_b")

# 조건부 엣지
def should_continue(state: State) -> str:
    if state["count"] > 5:
        return "end"
    return "continue"

workflow.add_conditional_edges(
    "node_a",
    should_continue,
    {
        "continue": "node_b",
        "end": END
    }
)`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 예제 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
          3. 실전: 리서치 에이전트
        </h2>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# State 정의
class ResearchState(TypedDict):
    query: str
    search_results: List[str]
    summary: str
    iteration: int

# Node 함수들
def search_node(state: ResearchState):
    results = web_search(state["query"])
    return {
        "search_results": results,
        "iteration": state["iteration"] + 1
    }

def summarize_node(state: ResearchState):
    summary = llm.invoke(
        f"다음 검색 결과를 요약: {state['search_results']}"
    )
    return {"summary": summary}

def should_research_more(state: ResearchState) -> str:
    if state["iteration"] >= 3:
        return "summarize"
    if len(state["search_results"]) < 5:
        return "search"
    return "summarize"

# Graph 구성
workflow = StateGraph(ResearchState)
workflow.add_node("search", search_node)
workflow.add_node("summarize", summarize_node)

workflow.set_entry_point("search")
workflow.add_conditional_edges(
    "search",
    should_research_more,
    {
        "search": "search",      # 더 검색
        "summarize": "summarize"  # 요약으로
    }
)
workflow.add_edge("summarize", END)

app = workflow.compile()

# 실행
result = app.invoke({
    "query": "LangGraph의 장점은?",
    "search_results": [],
    "summary": "",
    "iteration": 0
})`}
          </pre>
        </div>
      </section>

      {/* Advanced Features */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-green-600 dark:text-green-400">
          4. 고급 기능
        </h2>

        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-bold mb-3">🔄 Checkpointing (상태 저장)</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langgraph.checkpoint.sqlite import SqliteSaver

# Checkpoint 저장소 설정
memory = SqliteSaver.from_conn_string("checkpoints.db")

# Graph 컴파일 시 checkpointer 지정
app = workflow.compile(checkpointer=memory)

# Thread ID로 대화 관리
config = {"configurable": {"thread_id": "user_123"}}

# 실행 (자동으로 checkpoint 저장)
result = app.invoke(input_data, config)

# 이전 상태에서 재개 가능!`}
              </pre>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-3">👤 Human-in-the-Loop</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langgraph.checkpoint import interrupt

def approval_node(state):
    # 사용자 승인 대기
    if not state.get("approved"):
        interrupt("Waiting for approval")

    return {"approved": True}

# 실행
result = app.invoke({"task": "..."})

# 중단된 상태 확인
if result.get("interrupted"):
    # 사용자 입력 받기
    user_approval = input("승인하시겠습니까? (y/n)")

    # 재개
    result = app.invoke(
        {"approved": user_approval == "y"},
        config={"thread_id": "same_thread"}
    )`}
              </pre>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-bold mb-3">📊 Graph 시각화</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`# Mermaid 다이어그램 생성
from langgraph.graph import draw_mermaid

mermaid_code = draw_mermaid(app)
print(mermaid_code)

# 또는 이미지로 저장
from IPython.display import Image

Image(app.get_graph().draw_mermaid_png())`}
              </pre>
            </div>
          </div>
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
              LangGraph의 필요성과 LangChain과의 차이
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              State, Nodes, Edges의 핵심 개념
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              조건부 분기를 활용한 리서치 에이전트 구축
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Checkpointing과 Human-in-the-Loop 구현
            </span>
          </li>
        </ul>
      </section>
    </div>
  );
}
