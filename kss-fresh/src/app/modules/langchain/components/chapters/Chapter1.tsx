'use client';

import React from 'react';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* 페이지 헤더 */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 1: LangChain 시작하기</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          LLM 애플리케이션 개발의 새로운 패러다임
        </p>
      </div>

      {/* Section 1: LangChain이란? */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-amber-600 dark:text-amber-400">
          1. LangChain이란 무엇인가?
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300 leading-relaxed">
          LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션을 구축하기 위한 오픈소스 프레임워크입니다.
          2022년 10월 Harrison Chase가 시작한 이 프로젝트는 현재 가장 인기 있는 LLM 개발 도구로 자리잡았습니다.
        </p>

        <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold text-lg mb-3 text-amber-800 dark:text-amber-200">
            🎯 LangChain의 핵심 목표
          </h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-amber-600 dark:text-amber-400 mt-1">•</span>
              <span className="text-gray-700 dark:text-gray-300">
                <strong>컴포저빌리티(Composability)</strong>: 다양한 LLM 컴포넌트를 조합하여 복잡한 애플리케이션 구축
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600 dark:text-amber-400 mt-1">•</span>
              <span className="text-gray-700 dark:text-gray-300">
                <strong>재사용성(Reusability)</strong>: 표준화된 인터페이스로 컴포넌트 재활용
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600 dark:text-amber-400 mt-1">•</span>
              <span className="text-gray-700 dark:text-gray-300">
                <strong>확장성(Extensibility)</strong>: 커스텀 컴포넌트 추가 가능
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600 dark:text-amber-400 mt-1">•</span>
              <span className="text-gray-700 dark:text-gray-300">
                <strong>프로덕션 지향</strong>: 실제 서비스 배포를 위한 도구 제공
              </span>
            </li>
          </ul>
        </div>

        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-6">
          <h3 className="font-bold text-lg mb-3">💡 왜 LangChain이 필요한가?</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            순수한 LLM API 호출만으로는 복잡한 애플리케이션을 구축하기 어렵습니다.
            LangChain은 다음과 같은 문제를 해결합니다:
          </p>
          <ul className="space-y-2">
            <li className="text-gray-700 dark:text-gray-300">
              ✓ 여러 LLM 호출을 연결하는 복잡한 로직
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              ✓ 대화 히스토리 관리와 컨텍스트 유지
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              ✓ 외부 도구 및 API 통합
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              ✓ 데이터 검색 및 RAG 구현
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              ✓ 프롬프트 템플릿 관리
            </li>
          </ul>
        </div>
      </section>

      {/* Section 2: 핵심 컴포넌트 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
          2. LangChain 핵심 컴포넌트
        </h2>

        <div className="space-y-6">
          <div className="border-l-4 border-amber-500 pl-6">
            <h3 className="text-xl font-bold mb-2">📦 Models</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              다양한 LLM과 임베딩 모델을 통합하는 인터페이스
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
              <code className="text-sm">
                OpenAI, Anthropic, Cohere, HuggingFace, Local Models 등 30+ 모델 지원
              </code>
            </div>
          </div>

          <div className="border-l-4 border-blue-500 pl-6">
            <h3 className="text-xl font-bold mb-2">⛓️ Chains</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              여러 컴포넌트를 연결하여 복잡한 워크플로우 구성
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
              <code className="text-sm">
                LLMChain, SequentialChain, RouterChain, TransformChain 등
              </code>
            </div>
          </div>

          <div className="border-l-4 border-purple-500 pl-6">
            <h3 className="text-xl font-bold mb-2">💬 Prompts</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              재사용 가능한 프롬프트 템플릿과 관리 도구
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
              <code className="text-sm">
                PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
              </code>
            </div>
          </div>

          <div className="border-l-4 border-green-500 pl-6">
            <h3 className="text-xl font-bold mb-2">🧠 Memory</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              대화 히스토리와 컨텍스트를 저장하고 관리
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
              <code className="text-sm">
                ConversationBufferMemory, ConversationSummaryMemory, VectorStoreMemory
              </code>
            </div>
          </div>

          <div className="border-l-4 border-orange-500 pl-6">
            <h3 className="text-xl font-bold mb-2">🤖 Agents</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              LLM이 도구를 사용하여 자율적으로 문제 해결
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
              <code className="text-sm">
                ReAct Agent, Plan-and-Execute Agent, OpenAI Functions Agent
              </code>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: 설치 및 환경 설정 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
          3. 설치 및 환경 설정
        </h2>

        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-bold mb-3">📦 패키지 설치</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`# Python 3.8 이상 필요
pip install langchain

# 특정 통합 패키지 설치
pip install langchain-openai      # OpenAI 모델
pip install langchain-anthropic   # Claude 모델
pip install langchain-community   # 커뮤니티 통합`}
              </pre>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-bold mb-3">🔑 API 키 설정</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`# .env 파일 생성
OPENAI_API_KEY=sk-your-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Python 코드에서 로드
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")`}
              </pre>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
            <h4 className="font-bold text-yellow-800 dark:text-yellow-200 mb-2">
              ⚠️ 보안 주의사항
            </h4>
            <ul className="text-sm text-yellow-700 dark:text-yellow-300 space-y-1">
              <li>• API 키를 절대 코드에 하드코딩하지 마세요</li>
              <li>• .env 파일을 .gitignore에 추가하세요</li>
              <li>• 프로덕션에서는 환경 변수나 시크릿 관리 서비스 사용</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Section 4: 첫 LangChain 애플리케이션 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-green-600 dark:text-green-400">
          4. 첫 LangChain 애플리케이션
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          간단한 질문-응답 애플리케이션을 만들어봅시다:
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-6">
          <pre className="text-sm">
{`from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. LLM 모델 초기화
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7
)

# 2. 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    ("user", "{question}")
])

# 3. 출력 파서 생성
output_parser = StrOutputParser()

# 4. Chain 구성 (LCEL 문법)
chain = prompt | llm | output_parser

# 5. 실행
response = chain.invoke({"question": "LangChain이 뭔가요?"})
print(response)`}
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold text-lg mb-3 text-blue-800 dark:text-blue-200">
            📖 코드 설명
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li><strong>Line 1-3:</strong> 필요한 컴포넌트 import</li>
            <li><strong>Line 6-9:</strong> OpenAI GPT-4 모델 초기화 (temperature는 창의성 조절)</li>
            <li><strong>Line 12-15:</strong> 시스템 메시지와 사용자 질문 템플릿 정의</li>
            <li><strong>Line 18:</strong> LLM 응답을 문자열로 파싱</li>
            <li><strong>Line 21:</strong> LCEL(LangChain Expression Language)로 컴포넌트 연결</li>
            <li><strong>Line 24:</strong> 질문을 전달하여 응답 생성</li>
          </ul>
        </div>
      </section>

      {/* Section 5: LCEL 소개 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-indigo-600 dark:text-indigo-400">
          5. LCEL (LangChain Expression Language)
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          LCEL은 LangChain 0.1.0에서 도입된 새로운 체인 구성 방식입니다.
          파이프(|) 연산자를 사용하여 컴포넌트를 직관적으로 연결할 수 있습니다.
        </p>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div>
            <h3 className="font-bold mb-2 text-red-600 dark:text-red-400">❌ 기존 방식 (Legacy)</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.chains import LLMChain

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_parser=output_parser
)
chain.run(question="...")`}
              </pre>
            </div>
          </div>

          <div>
            <h3 className="font-bold mb-2 text-green-600 dark:text-green-400">✅ LCEL 방식 (Modern)</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`chain = (
    prompt
    | llm
    | output_parser
)
chain.invoke({"question": "..."})`}
              </pre>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
          <h3 className="font-bold text-lg mb-3 text-green-800 dark:text-green-200">
            ✨ LCEL의 장점
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• <strong>간결함</strong>: 더 짧고 읽기 쉬운 코드</li>
            <li>• <strong>스트리밍</strong>: 자동으로 스트리밍 지원 (.stream() 메서드)</li>
            <li>• <strong>비동기</strong>: 비동기 실행 기본 지원 (.ainvoke())</li>
            <li>• <strong>병렬 실행</strong>: 여러 체인을 동시에 실행 (RunnableParallel)</li>
            <li>• <strong>타입 안전성</strong>: 더 나은 타입 추론</li>
          </ul>
        </div>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-amber-800 dark:text-amber-200">
          📚 이 챕터에서 배운 것
        </h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              LangChain의 필요성과 핵심 목표 (컴포저빌리티, 재사용성, 확장성)
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              5가지 핵심 컴포넌트: Models, Chains, Prompts, Memory, Agents
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              개발 환경 설정 및 API 키 관리 Best Practices
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              첫 LangChain 애플리케이션 구축 (Q&A 시스템)
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              LCEL(LangChain Expression Language)의 개념과 장점
            </span>
          </li>
        </ul>
      </section>

      {/* 다음 챕터 안내 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 text-center">
        <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
          🚀 다음 챕터
        </h3>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          Chapter 2에서는 Chains와 Prompt Templates을 깊이 있게 학습하고,
          다양한 체인 패턴을 실습해보겠습니다.
        </p>
      </div>
    </div>
  );
}
