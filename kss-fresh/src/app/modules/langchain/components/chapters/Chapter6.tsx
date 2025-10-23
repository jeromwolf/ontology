'use client';

import React from 'react';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 6: 복잡한 워크플로우와 분기</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          멀티 에이전트 협업과 고급 패턴
        </p>
      </div>

      {/* 조건부 라우팅 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-amber-600 dark:text-amber-400">
          1. 조건부 라우팅
        </h2>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`from langgraph.graph import StateGraph, END

def router(state):
    task_type = state["task_type"]

    if task_type == "coding":
        return "code_agent"
    elif task_type == "research":
        return "research_agent"
    elif task_type == "writing":
        return "writer_agent"
    else:
        return END

workflow.add_conditional_edges(
    "classifier",
    router,
    {
        "code_agent": "code_agent",
        "research_agent": "research_agent",
        "writer_agent": "writer_agent",
        END: END
    }
)`}
          </pre>
        </div>
      </section>

      {/* 멀티 에이전트 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
          2. 멀티 에이전트 협업
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          여러 전문 에이전트가 협력하여 복잡한 작업 수행
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`# 에이전트 정의
researcher = create_agent("researcher", research_tools)
coder = create_agent("coder", code_tools)
writer = create_agent("writer", writing_tools)

# Supervisor 패턴
def supervisor_node(state):
    messages = state["messages"]
    last_message = messages[-1]

    # LLM이 다음 에이전트 선택
    response = llm.invoke([
        SystemMessage(
            "다음 중 어느 에이전트가 작업해야 하는지 선택: "
            "researcher, coder, writer, FINISH"
        ),
        *messages
    ])

    if "FINISH" in response.content:
        return {"next": END}

    return {"next": response.content}

# Graph 구성
workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher)
workflow.add_node("coder", coder)
workflow.add_node("writer", writer)

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "researcher": "researcher",
        "coder": "coder",
        "writer": "writer",
        END: END
    }
)

# 모든 에이전트 -> supervisor로 복귀
for agent in ["researcher", "coder", "writer"]:
    workflow.add_edge(agent, "supervisor")`}
          </pre>
        </div>
      </section>

      {/* Human-in-the-Loop */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
          3. Human-in-the-Loop 구현
        </h2>

        <div className="space-y-4">
          <p className="text-gray-700 dark:text-gray-300">
            중요한 결정에 사람의 승인을 받는 시스템
          </p>

          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm">
{`from langgraph.checkpoint.memory import MemorySaver

def approval_required_node(state):
    # 중요한 작업 전 승인 대기
    return {
        "messages": state["messages"] + [
            "다음 작업을 수행하시겠습니까?"
        ],
        "awaiting_approval": True
    }

# Checkpointer로 상태 저장
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 1단계: 초기 실행
config = {"configurable": {"thread_id": "session_1"}}
result = app.invoke({"task": "..."}, config)

# 2단계: 중단 지점에서 사용자 입력
if result.get("awaiting_approval"):
    user_input = input("승인하시겠습니까? (y/n): ")

    # 3단계: 사용자 응답으로 재개
    final_result = app.invoke(
        {"approval": user_input == "y"},
        config  # 같은 thread_id 사용
    )`}
            </pre>
          </div>
        </div>
      </section>

      {/* Error Handling */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-red-600 dark:text-red-400">
          4. 에러 핸들링과 Retry
        </h2>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`def resilient_node(state):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            result = risky_operation(state)
            return {"result": result, "error": None}

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "result": None,
                    "error": str(e),
                    "retry_count": attempt + 1
                }

            # 지수 백오프
            time.sleep(2 ** attempt)

def error_handler(state):
    if state.get("error"):
        return "fallback"
    return "success"

workflow.add_conditional_edges(
    "risky_node",
    error_handler,
    {
        "success": "next_node",
        "fallback": "fallback_node"
    }
)`}
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
            <span className="text-gray-700 dark:text-gray-300">조건부 라우팅으로 동적 워크플로우 구성</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">Supervisor 패턴 기반 멀티 에이전트 시스템</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">Human-in-the-Loop 실전 구현</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">에러 핸들링과 Retry 로직</span>
          </li>
        </ul>
      </section>
    </div>
  );
}
