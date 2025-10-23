'use client';

import React from 'react';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 3: Memory와 Context 관리</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          대화의 맥락을 유지하는 핵심 메커니즘
        </p>
      </div>

      {/* Memory 개념 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-amber-600 dark:text-amber-400">
          1. Memory의 필요성
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          LLM은 상태를 유지하지 않습니다(stateless). 이전 대화를 기억하려면 Memory 컴포넌트가 필요합니다.
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-6">
          <pre className="text-sm">
{`from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# LLM 초기화
llm = ChatOpenAI(temperature=0.7)

# Memory 설정
memory = ConversationBufferMemory()

# 대화 체인 생성
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 첫 번째 대화
conversation.predict(input="내 이름은 민수야")
# "안녕하세요 민수님!"

# 두 번째 대화 - 이름을 기억함
conversation.predict(input="내 이름이 뭐였지?")
# "민수님이셨죠!"`}
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold text-lg mb-3">🧠 Memory가 해결하는 문제</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 이전 대화 맥락 유지</li>
            <li>• 사용자 선호도 기억</li>
            <li>• 복잡한 멀티턴 대화 처리</li>
            <li>• 장기 컨텍스트 관리</li>
          </ul>
        </div>
      </section>

      {/* Memory 타입 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
          2. Memory 타입별 특징
        </h2>

        <div className="space-y-6">
          <div className="border-l-4 border-green-500 pl-6">
            <h3 className="text-xl font-bold mb-3">💬 ConversationBufferMemory</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              모든 대화를 그대로 저장. 가장 단순하지만 토큰 사용량이 많음.
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "안녕"},
    {"output": "안녕하세요!"}
)

print(memory.load_memory_variables({}))
# {'history': 'Human: 안녕\\nAI: 안녕하세요!'}`}
              </pre>
            </div>
          </div>

          <div className="border-l-4 border-blue-500 pl-6">
            <h3 className="text-xl font-bold mb-3">📝 ConversationBufferWindowMemory</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              최근 K개의 대화만 저장. 토큰 효율적.
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.memory import ConversationBufferWindowMemory

# 최근 5개 대화만 저장
memory = ConversationBufferWindowMemory(k=5)

for i in range(10):
    memory.save_context(
        {"input": f"질문 {i}"},
        {"output": f"답변 {i}"}
    )

# 최근 5개만 반환
print(memory.load_memory_variables({}))`}
              </pre>
            </div>
          </div>

          <div className="border-l-4 border-purple-500 pl-6">
            <h3 className="text-xl font-bold mb-3">📊 ConversationSummaryMemory</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              대화를 요약하여 저장. 장기 대화에 적합.
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)

# 대화가 쌓일수록 자동으로 요약
memory.save_context(
    {"input": "LangChain에 대해 알려줘"},
    {"output": "LangChain은 LLM 앱 프레임워크입니다..."}
)

# 요약된 내용 반환
summary = memory.load_memory_variables({})
print(summary["history"])`}
              </pre>
            </div>
          </div>

          <div className="border-l-4 border-orange-500 pl-6">
            <h3 className="text-xl font-bold mb-3">🔍 VectorStoreRetrieverMemory</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              벡터 검색으로 관련 대화만 가져옴. 매우 효율적.
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 벡터 스토어 초기화
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["초기 텍스트"], embeddings
)

# Memory 설정
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}  # 관련 3개만
    )
)

memory.save_context(
    {"input": "내 좋아하는 색은 파란색이야"},
    {"output": "파란색을 좋아하시는군요!"}
)

# 관련 대화 검색
relevant = memory.load_memory_variables(
    {"prompt": "내가 좋아하는 색은?"}
)
print(relevant)`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 컨텍스트 윈도우 관리 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
          3. 컨텍스트 윈도우 최적화
        </h2>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 mb-6">
          <h3 className="font-bold text-yellow-800 dark:text-yellow-200 mb-3">
            ⚠️ 토큰 제한 문제
          </h3>
          <div className="text-gray-700 dark:text-gray-300 space-y-2 text-sm">
            <p>• GPT-4: 8K-128K tokens</p>
            <p>• Claude 3: 200K tokens</p>
            <p>• Gemini 1.5 Pro: 2M tokens</p>
            <p className="mt-3 font-semibold">
              컨텍스트가 너무 길면 비용 증가 + 성능 저하!
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <h3 className="text-xl font-bold mb-3">📏 Token Counting</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"input": "..."})

    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost: $" + "{cb.total_cost}")`}
              </pre>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-3">✂️ 컨텍스트 압축 전략</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>Window Memory</strong>: 최근 N개만 유지</li>
              <li>• <strong>Summary Memory</strong>: 오래된 대화는 요약</li>
              <li>• <strong>Vector Memory</strong>: 관련성 높은 것만 검색</li>
              <li>• <strong>Hybrid</strong>: 최근 대화 + 관련 과거 대화</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 실전 예제 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-green-600 dark:text-green-400">
          4. 실전: 고객 지원 챗봇
        </h2>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

# Hybrid Memory: Summary + Recent Buffer
memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(),
    max_token_limit=500,  # 500토큰 이상이면 요약
    return_messages=True
)

# 고객 지원 체인
conversation = ConversationChain(
    llm=ChatOpenAI(temperature=0),
    memory=memory,
    verbose=True
)

# 시뮬레이션
conversation.predict(
    input="제품 반품하고 싶어요"
)
conversation.predict(
    input="주문번호는 ABC123이에요"
)
conversation.predict(
    input="언제 환불되나요?"
)
# 이전 맥락(주문번호)을 기억하여 답변`}
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mt-6">
          <h3 className="font-bold text-lg mb-3">💡 Best Practices</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
            <li>✓ 짧은 대화: BufferMemory</li>
            <li>✓ 중간 대화: BufferWindowMemory (k=10)</li>
            <li>✓ 긴 대화: SummaryBufferMemory</li>
            <li>✓ 매우 긴 대화: VectorStoreMemory</li>
            <li>✓ 프로덕션: 데이터베이스에 영구 저장</li>
          </ul>
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
              Memory의 필요성과 4가지 주요 타입
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              컨텍스트 윈도우 관리와 토큰 최적화
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              상황별 적절한 Memory 타입 선택
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              고객 지원 챗봇 실전 예제
            </span>
          </li>
        </ul>
      </section>
    </div>
  );
}
