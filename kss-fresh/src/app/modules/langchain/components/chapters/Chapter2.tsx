'use client';

import React from 'react';

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 2: Chains와 Prompt Templates</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          체인 구성의 예술과 효과적인 프롬프트 설계
        </p>
      </div>

      {/* Prompt Templates */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-amber-600 dark:text-amber-400">
          1. Prompt Templates 마스터하기
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          Prompt Template은 재사용 가능한 프롬프트 구조를 정의합니다.
          변수를 포함하여 동적으로 프롬프트를 생성할 수 있습니다.
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-6">
          <pre className="text-sm">
{`from langchain_core.prompts import PromptTemplate

# 기본 템플릿
template = """질문: {question}

위 질문에 대해 {style} 스타일로 답변해주세요."""

prompt = PromptTemplate(
    template=template,
    input_variables=["question", "style"]
)

# 사용
formatted = prompt.format(
    question="LangChain이 뭔가요?",
    style="친근한"
)
print(formatted)`}
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold text-lg mb-3">💬 ChatPromptTemplate</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-3">
            대화형 모델을 위한 멀티 메시지 템플릿
          </p>
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm">
{`from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {expertise} 전문가입니다."),
    ("human", "{question}"),
    ("ai", "제가 도와드리겠습니다!"),
    ("human", "{followup}")
])

messages = prompt.format_messages(
    expertise="AI",
    question="LangChain 설명해줘",
    followup="더 자세히 알려줘"
)`}
            </pre>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="font-bold text-lg mb-3">🎯 Few-Shot Prompting</h3>
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm">
{`from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "😊"},
    {"input": "sad", "output": "😢"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="감정을 이모지로 변환:",
    suffix="Input: {input}\\nOutput:",
    input_variables=["input"]
)`}
            </pre>
          </div>
        </div>
      </section>

      {/* Chains */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
          2. Chains: 컴포넌트 연결하기
        </h2>

        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-bold mb-3">⛓️ Sequential Chain</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              여러 체인을 순차적으로 실행하여 복잡한 워크플로우 구성
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Chain 1: 주제 생성
topic_chain = (
    ChatPromptTemplate.from_template("블로그 주제 3개 추천: {interest}")
    | ChatOpenAI()
    | StrOutputParser()
)

# Chain 2: 내용 작성
content_chain = (
    ChatPromptTemplate.from_template("다음 주제로 글 작성: {topic}")
    | ChatOpenAI()
    | StrOutputParser()
)

# 순차 실행
from langchain_core.runnables import RunnablePassthrough

full_chain = (
    {"topic": topic_chain}
    | RunnablePassthrough()
    | content_chain
)

result = full_chain.invoke({"interest": "AI"})`}
              </pre>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-3">🔀 Router Chain</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              입력에 따라 다른 체인으로 라우팅
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain_core.runnables import RunnableBranch

# 조건부 라우팅
branch = RunnableBranch(
    (lambda x: "python" in x["query"].lower(), python_chain),
    (lambda x: "javascript" in x["query"].lower(), js_chain),
    default_chain  # 기본값
)

chain = {"query": RunnablePassthrough()} | branch
result = chain.invoke("How to use Python?")`}
              </pre>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-3">🔄 Transform Chain</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              데이터 변환 로직을 체인에 포함
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain_core.runnables import RunnableLambda

def clean_text(text):
    return text.strip().lower()

def count_words(text):
    return {"text": text, "word_count": len(text.split())}

chain = (
    RunnableLambda(clean_text)
    | RunnableLambda(count_words)
    | ChatPromptTemplate.from_template(
        "텍스트: {text}\\n단어 수: {word_count}"
    )
    | llm
)`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Advanced Patterns */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
          3. 고급 Chain 패턴
        </h2>

        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-bold mb-3">⚡ Parallel Execution</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain_core.runnables import RunnableParallel

# 여러 체인을 병렬로 실행
parallel_chain = RunnableParallel(
    summary=summary_chain,
    translation=translation_chain,
    sentiment=sentiment_chain
)

results = parallel_chain.invoke({"text": "..."})
# {
#   "summary": "...",
#   "translation": "...",
#   "sentiment": "positive"
# }`}
              </pre>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-3">🔁 Retry Logic</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain_core.runnables import RunnableRetry

# 실패 시 재시도
chain_with_retry = RunnableRetry(
    bound=chain,
    max_attempt_number=3,
    wait_exponential_jitter=True
)

# 타임아웃 설정
from langchain_core.runnables import RunnableTimeout

chain_with_timeout = RunnableTimeout(
    bound=chain,
    timeout=10.0  # 10초
)`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-green-600 dark:text-green-400">
          4. Prompt Engineering Best Practices
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-amber-500 pl-6">
            <h3 className="font-bold mb-2">✓ 명확하고 구체적으로</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-red-600 dark:text-red-400 font-bold mb-2">❌ 나쁜 예</p>
                <div className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm">
                  "이것에 대해 말해줘"
                </div>
              </div>
              <div>
                <p className="text-sm text-green-600 dark:text-green-400 font-bold mb-2">✅ 좋은 예</p>
                <div className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-sm">
                  "LangChain의 Memory 컴포넌트에 대해 사용 예제와 함께 200자로 설명해줘"
                </div>
              </div>
            </div>
          </div>

          <div className="border-l-4 border-blue-500 pl-6">
            <h3 className="font-bold mb-2">✓ 역할과 페르소나 부여</h3>
            <div className="bg-gray-100 dark:bg-gray-900 p-3 rounded">
              <code className="text-sm">
                "당신은 10년 경력의 Python 개발자이자 LangChain 전문가입니다.
                초보자도 이해할 수 있도록 친절하게 설명해주세요."
              </code>
            </div>
          </div>

          <div className="border-l-4 border-purple-500 pl-6">
            <h3 className="font-bold mb-2">✓ 출력 형식 지정</h3>
            <div className="bg-gray-100 dark:bg-gray-900 p-3 rounded">
              <code className="text-sm">
                {`다음 형식으로 답변하세요:
1. 개념 정의
2. 사용 예제
3. 주의사항
4. 추가 리소스`}
              </code>
            </div>
          </div>

          <div className="border-l-4 border-green-500 pl-6">
            <h3 className="font-bold mb-2">✓ Few-Shot Examples 활용</h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm">
              원하는 출력 형식의 예시를 2-3개 제공하면 일관된 결과를 얻을 수 있습니다.
            </p>
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
              Prompt Template의 종류: PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Sequential Chain으로 복잡한 워크플로우 구성
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Router Chain과 Transform Chain의 활용
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              병렬 실행과 에러 핸들링 (Retry, Timeout)
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              효과적인 Prompt Engineering Best Practices
            </span>
          </li>
        </ul>
      </section>
    </div>
  );
}
