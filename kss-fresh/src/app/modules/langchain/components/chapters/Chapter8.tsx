'use client';

import React from 'react';
import References from '@/components/common/References';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 8: 실전 프로젝트와 사례 연구</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          완전한 애플리케이션 구축 실습
        </p>
      </div>

      {/* 프로젝트 1: 문서 QA */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-amber-600 dark:text-amber-400">
          프로젝트 1: 문서 기반 QA 시스템
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          RAG 패턴을 활용한 회사 문서 질의응답 시스템
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-6">
          <pre className="text-sm">
{`from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 문서 로드
loader = DirectoryLoader('./company_docs', glob="**/*.pdf")
documents = loader.load()

# 2. 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 3. 벡터 스토어 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    return_source_documents=True
)

# 5. 실행
result = qa_chain.invoke({
    "query": "회사의 휴가 정책은?"
})

print(result["result"])
for doc in result["source_documents"]:
    print(f"Source: {doc.metadata['source']}")`}
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold text-lg mb-3">🎯 구현 포인트</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
            <li>• PDF, Word, Markdown 등 다양한 형식 지원</li>
            <li>• 메타데이터 보존 (파일명, 페이지 번호)</li>
            <li>• 하이브리드 검색 (키워드 + 벡터)</li>
            <li>• 소스 인용 기능</li>
          </ul>
        </div>
      </section>

      {/* 프로젝트 2: Code Assistant */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
          프로젝트 2: AI Code Assistant
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          코드 생성, 리뷰, 리팩토링을 수행하는 AI 어시스턴트
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool
from langchain import hub

@tool
def execute_python(code: str) -> str:
    """Python 코드를 실행하고 결과를 반환합니다."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals.get('result', 'Success'))
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def analyze_complexity(code: str) -> str:
    """코드의 시간 복잡도를 분석합니다."""
    # 간단한 정적 분석
    loops = code.count("for") + code.count("while")
    if loops == 0:
        return "O(1) - 상수 시간"
    elif loops == 1:
        return "O(n) - 선형 시간"
    else:
        return f"O(n^{loops}) - 다항 시간"

@tool
def suggest_refactoring(code: str) -> str:
    """코드 리팩토링 제안을 제공합니다."""
    suggestions = []

    if len(code.split('\\n')) > 50:
        suggestions.append("함수를 더 작은 단위로 분리하세요")

    if "TODO" in code or "FIXME" in code:
        suggestions.append("TODO/FIXME 주석을 해결하세요")

    return "\\n".join(suggestions) if suggestions else "코드가 깔끔합니다!"

# Agent 생성
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [execute_python, analyze_complexity, suggest_refactoring]

prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# 사용 예시
result = agent_executor.invoke({
    "input": """
    다음 Python 함수를 분석하고 최적화 제안해줘:

    def find_duplicates(arr):
        result = []
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if arr[i] == arr[j] and arr[i] not in result:
                    result.append(arr[i])
        return result
    """
})`}
          </pre>
        </div>
      </section>

      {/* 프로젝트 3: Research Agent */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
          프로젝트 3: 자동화 리서치 에이전트
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          LangGraph를 활용한 자율적 리서치 시스템
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class ResearchState(TypedDict):
    topic: str
    search_queries: List[str]
    search_results: List[str]
    outline: str
    content: str
    iteration: int

# Node 함수들
def generate_queries(state):
    queries = llm.invoke(
        f"{state['topic']}에 대해 검색할 3개 쿼리 생성"
    ).content.split("\\n")

    return {"search_queries": queries}

def search_web(state):
    results = []
    for query in state["search_queries"]:
        # 실제로는 DuckDuckGo, SerpAPI 등 사용
        result = web_search_tool.invoke(query)
        results.append(result)

    return {"search_results": results}

def create_outline(state):
    outline = llm.invoke(f"""
    다음 검색 결과를 바탕으로 '{state['topic']}'에 대한 개요 작성:
    {state['search_results']}
    """).content

    return {"outline": outline}

def write_content(state):
    content = llm.invoke(f"""
    다음 개요를 바탕으로 완전한 글 작성:
    {state['outline']}

    검색 결과 참고:
    {state['search_results']}
    """).content

    return {"content": content}

def should_continue(state):
    if state["iteration"] >= 2:
        return "write"
    if len(state["search_results"]) < 5:
        return "search"
    return "outline"

# Graph 구성
workflow = StateGraph(ResearchState)

workflow.add_node("generate_queries", generate_queries)
workflow.add_node("search", search_web)
workflow.add_node("outline", create_outline)
workflow.add_node("write", write_content)

workflow.set_entry_point("generate_queries")
workflow.add_edge("generate_queries", "search")

workflow.add_conditional_edges(
    "search",
    should_continue,
    {
        "search": "search",
        "outline": "outline",
        "write": "write"
    }
)

workflow.add_edge("outline", "write")
workflow.add_edge("write", END)

app = workflow.compile()

# 실행
result = app.invoke({
    "topic": "LangGraph의 실전 활용 사례",
    "search_queries": [],
    "search_results": [],
    "outline": "",
    "content": "",
    "iteration": 0
})

print(result["content"])`}
          </pre>
        </div>
      </section>

      {/* 프로젝트 4: Multi-Modal */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-green-600 dark:text-green-400">
          프로젝트 4: Multi-Modal 애플리케이션
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          이미지와 텍스트를 함께 처리하는 시스템
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# GPT-4 Vision 사용
llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1000)

# 이미지 분석
image_data = encode_image("product_screenshot.png")

message = HumanMessage(
    content=[
        {"type": "text", "text": "이 제품 페이지의 UI/UX를 분석하고 개선점을 제안해줘"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"}
        }
    ]
)

response = llm.invoke([message])
print(response.content)`}
          </pre>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 mt-6">
          <h3 className="font-bold text-lg mb-3">🎨 활용 사례</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
            <li>• UI 디자인 자동 분석 및 개선 제안</li>
            <li>• 문서 OCR + 내용 요약</li>
            <li>• 제품 이미지 → 설명 자동 생성</li>
            <li>• 차트/그래프 해석</li>
          </ul>
        </div>
      </section>

      {/* Best Practices 요약 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-indigo-600 dark:text-indigo-400">
          💡 프로덕션 체크리스트
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-bold mb-3 text-lg">✅ 개발 단계</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>☐ 명확한 사용 사례 정의</li>
              <li>☐ 적절한 모델 선택</li>
              <li>☐ 프롬프트 엔지니어링</li>
              <li>☐ 에러 핸들링 구현</li>
              <li>☐ 로깅 및 모니터링</li>
            </ul>
          </div>

          <div>
            <h3 className="font-bold mb-3 text-lg">✅ 배포 단계</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>☐ LangSmith 통합</li>
              <li>☐ 캐싱 전략 수립</li>
              <li>☐ Rate limiting 설정</li>
              <li>☐ 보안 검증</li>
              <li>☐ 성능 테스트</li>
              <li>☐ 비용 모니터링</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 다음 단계 */}
      <section className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-4 text-purple-800 dark:text-purple-200">
          🚀 다음 단계
        </h2>

        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <div>
            <h3 className="font-bold mb-2">1. 실전 프로젝트 구축</h3>
            <p className="text-sm">
              배운 내용을 활용하여 실제 문제를 해결하는 프로젝트를 만들어보세요.
            </p>
          </div>

          <div>
            <h3 className="font-bold mb-2">2. 커뮤니티 참여</h3>
            <p className="text-sm">
              LangChain Discord, GitHub Discussions에 참여하여 최신 트렌드를 따라가세요.
            </p>
          </div>

          <div>
            <h3 className="font-bold mb-2">3. 오픈소스 기여</h3>
            <p className="text-sm">
              LangChain, LangGraph 프로젝트에 기여하며 더 깊이 학습하세요.
            </p>
          </div>

          <div>
            <h3 className="font-bold mb-2">4. 고급 주제 탐구</h3>
            <ul className="text-sm space-y-1 mt-2">
              <li>• LangGraph Cloud 배포</li>
              <li>• Custom LLM 통합</li>
              <li>• Agent Memory 최적화</li>
              <li>• Multi-Agent 협업 패턴</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 최종 요약 */}
      <section className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-4 text-amber-800 dark:text-amber-200">
          🎓 전체 과정 완료!
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          축하합니다! LangChain과 LangGraph의 기초부터 프로덕션 배포까지 모든 과정을 완료하셨습니다.
          이제 여러분은 실전에서 LLM 애플리케이션을 구축할 준비가 되었습니다!
        </p>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-bold mb-3">📚 학습한 내용 총정리</h3>
          <ul className="grid md:grid-cols-2 gap-2 text-sm text-gray-700 dark:text-gray-300">
            <li>✓ LangChain 기초와 LCEL</li>
            <li>✓ Chains와 Prompt Templates</li>
            <li>✓ Memory 시스템</li>
            <li>✓ Agents와 Tools</li>
            <li>✓ LangGraph 설계</li>
            <li>✓ 복잡한 워크플로우</li>
            <li>✓ 프로덕션 배포</li>
            <li>✓ 실전 프로젝트 4개</li>
          </ul>
        </div>
      </section>

      {/* References Section */}
      <References
        sections={[
          {
            title: '📚 공식 문서 & 튜토리얼',
            icon: 'web' as const,
            color: 'border-amber-500',
            items: [
              {
                title: 'LangChain Documentation',
                authors: 'LangChain',
                year: '2025',
                description: 'LangChain 공식 문서 - 전체 API 레퍼런스와 사용 가이드',
                link: 'https://python.langchain.com/docs/'
              },
              {
                title: 'LangGraph Documentation',
                authors: 'LangChain',
                year: '2025',
                description: 'LangGraph 공식 문서 - 복잡한 워크플로우 구축 가이드',
                link: 'https://langchain-ai.github.io/langgraph/'
              },
              {
                title: 'LangSmith Platform',
                authors: 'LangChain',
                year: '2025',
                description: 'LLM 애플리케이션 디버깅, 테스팅, 모니터링 플랫폼',
                link: 'https://smith.langchain.com/'
              },
              {
                title: 'LangServe Deployment',
                authors: 'LangChain',
                year: '2025',
                description: 'LangChain 애플리케이션을 REST API로 배포하는 프레임워크',
                link: 'https://python.langchain.com/docs/langserve'
              },
              {
                title: 'LangChain Cookbook',
                authors: 'LangChain Community',
                year: '2025',
                description: '실전 예제와 베스트 프랙티스 모음',
                link: 'https://github.com/langchain-ai/langchain-cookbook'
              }
            ]
          },
          {
            title: '📖 핵심 논문 & 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'ReAct: Synergizing Reasoning and Acting in Language Models',
                authors: 'Yao et al.',
                year: '2023',
                description: 'Reasoning + Acting을 결합한 Agent 아키텍처 - LangChain Agent의 기반',
                link: 'https://arxiv.org/abs/2210.03629'
              },
              {
                title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP',
                authors: 'Lewis et al., Meta AI',
                year: '2020',
                description: 'RAG 패턴의 원조 논문 - 외부 지식 검색 + 생성 결합',
                link: 'https://arxiv.org/abs/2005.11401'
              },
              {
                title: 'Chain-of-Thought Prompting Elicits Reasoning',
                authors: 'Wei et al., Google',
                year: '2022',
                description: 'CoT 프롬프팅 기법 - 단계별 추론으로 성능 향상',
                link: 'https://arxiv.org/abs/2201.11903'
              },
              {
                title: 'Self-Consistency Improves Chain of Thought Reasoning',
                authors: 'Wang et al., Google',
                year: '2023',
                description: '다중 추론 경로 + 다수결 투표로 정확도 향상',
                link: 'https://arxiv.org/abs/2203.11171'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 템플릿',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'LangChain Templates',
                authors: 'LangChain',
                year: '2025',
                description: '다양한 사용 사례별 즉시 사용 가능한 템플릿 모음',
                link: 'https://github.com/langchain-ai/langchain/tree/master/templates'
              },
              {
                title: 'LangChain Hub',
                authors: 'LangChain',
                year: '2025',
                description: '검증된 Prompt 템플릿과 Chain 구성 공유 플랫폼',
                link: 'https://smith.langchain.com/hub'
              },
              {
                title: 'LangChain Benchmarks',
                authors: 'LangChain',
                year: '2025',
                description: 'RAG, Agent 성능 벤치마크 도구 및 데이터셋',
                link: 'https://github.com/langchain-ai/langchain-benchmarks'
              },
              {
                title: 'Awesome LangChain',
                authors: 'Community',
                year: '2025',
                description: 'LangChain 관련 도구, 튜토리얼, 프로젝트 큐레이션',
                link: 'https://github.com/kyrolabs/awesome-langchain'
              },
              {
                title: 'LangChain GitHub Examples',
                authors: 'LangChain',
                year: '2025',
                description: '공식 예제 코드 리포지토리 - RAG, Agent, Multi-modal 등',
                link: 'https://github.com/langchain-ai/langchain/tree/master/cookbook'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
