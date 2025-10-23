'use client';

import React from 'react';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 7: 프로덕션 배포와 Best Practices</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          실제 서비스 운영을 위한 필수 지식
        </p>
      </div>

      {/* LangSmith */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-amber-600 dark:text-amber-400">
          1. LangSmith로 모니터링
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          LangSmith는 LangChain 애플리케이션의 디버깅, 테스팅, 모니터링을 위한 플랫폼입니다.
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-6">
          <pre className="text-sm">
{`# 환경 변수 설정
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# 자동으로 모든 실행 추적
chain.invoke({"input": "..."})

# LangSmith 대시보드에서:
# - 전체 실행 trace
# - 각 단계별 latency
# - Token 사용량
# - 에러 로그`}
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold text-lg mb-3">📊 LangSmith 핵심 기능</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• <strong>Tracing</strong>: 모든 실행 단계 시각화</li>
            <li>• <strong>Dataset</strong>: 테스트 케이스 관리</li>
            <li>• <strong>Evaluation</strong>: 자동화된 평가</li>
            <li>• <strong>Monitoring</strong>: 실시간 성능 추적</li>
          </ul>
        </div>
      </section>

      {/* Caching */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
          2. 캐싱으로 성능 최적화
        </h2>

        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-bold mb-3">💾 In-Memory Cache</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# 메모리 캐시 활성화
set_llm_cache(InMemoryCache())

# 동일한 입력은 캐시에서 반환 (LLM 호출 생략)
llm.invoke("What is LangChain?")  # API 호출
llm.invoke("What is LangChain?")  # 캐시 반환 (즉시!)`}
              </pre>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-bold mb-3">🗄️ Redis Cache</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.cache import RedisCache
from redis import Redis

# Redis 캐시 (영구 저장)
set_llm_cache(RedisCache(
    redis_=Redis(host="localhost", port=6379)
))

# 서버 재시작 후에도 캐시 유지`}
              </pre>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-bold mb-3">🔍 Semantic Cache</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`from langchain.cache import SemanticCache
from langchain_openai import OpenAIEmbeddings

# 의미적으로 유사한 질문도 캐시
set_llm_cache(SemanticCache(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.9
))

llm.invoke("What is LangChain?")
llm.invoke("Tell me about LangChain")  # 유사 → 캐시 hit!`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Security */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-red-600 dark:text-red-400">
          3. 보안 Best Practices
        </h2>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 mb-6">
          <h3 className="font-bold text-yellow-800 dark:text-yellow-200 mb-3">
            ⚠️ 보안 체크리스트
          </h3>
          <ul className="space-y-2 text-yellow-700 dark:text-yellow-300 text-sm">
            <li>✓ API 키를 환경 변수로 관리</li>
            <li>✓ 사용자 입력 검증 및 sanitization</li>
            <li>✓ Rate limiting 구현</li>
            <li>✓ Prompt injection 방어</li>
            <li>✓ 민감 정보 필터링</li>
            <li>✓ 접근 제어 및 인증</li>
          </ul>
        </div>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm">
{`# 1. Input Validation
def validate_input(user_input: str) -> str:
    # 길이 제한
    if len(user_input) > 1000:
        raise ValueError("Input too long")

    # 금지된 패턴 체크
    forbidden = ["<script>", "DROP TABLE"]
    for pattern in forbidden:
        if pattern in user_input:
            raise ValueError("Invalid input")

    return user_input

# 2. Output Filtering
def filter_output(output: str) -> str:
    # PII 제거
    import re

    # 이메일 마스킹
    output = re.sub(
        r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
        '[EMAIL]',
        output
    )

    # 전화번호 마스킹
    output = re.sub(r'\\d{3}-\\d{4}-\\d{4}', '[PHONE]', output)

    return output

# 3. Rate Limiting
from slowapi import Limiter

limiter = Limiter(key_func=lambda: request.client.host)

@limiter.limit("10/minute")
async def invoke_chain(input: str):
    return chain.invoke(input)`}
          </pre>
        </div>
      </section>

      {/* LangServe */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-green-600 dark:text-green-400">
          4. LangServe로 API 배포
        </h2>

        <p className="mb-4 text-gray-700 dark:text-gray-300">
          LangServe를 사용하면 LangChain 체인을 FastAPI 엔드포인트로 쉽게 배포할 수 있습니다.
        </p>

        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-6">
          <pre className="text-sm">
{`# 설치
pip install langserve[all]

# server.py
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(
    title="LangChain Server",
    version="1.0"
)

# Chain 정의
prompt = ChatPromptTemplate.from_template("{topic}에 대해 설명해줘")
model = ChatOpenAI()
chain = prompt | model

# API 엔드포인트 자동 생성
add_routes(
    app,
    chain,
    path="/chat"
)

# 실행: uvicorn server:app --reload

# 자동으로 생성되는 엔드포인트:
# POST /chat/invoke          - 단일 실행
# POST /chat/batch           - 배치 실행
# POST /chat/stream          - 스트리밍
# GET  /chat/playground      - 테스트 UI`}
          </pre>
        </div>

        <div>
          <h3 className="text-xl font-bold mb-3">📱 Client 사용</h3>
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm">
{`from langserve import RemoteRunnable

# 원격 체인 연결
chain = RemoteRunnable("http://localhost:8000/chat")

# 로컬 체인처럼 사용!
result = chain.invoke({"topic": "LangChain"})

# 스트리밍
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)`}
            </pre>
          </div>
        </div>
      </section>

      {/* Performance */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm">
        <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
          5. 성능 최적화 팁
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-bold mb-2">⚡ 속도 향상</h4>
            <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
              <li>• 캐싱 활용</li>
              <li>• 스트리밍 응답</li>
              <li>• 배치 처리</li>
              <li>• 비동기 실행</li>
            </ul>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-bold mb-2">💰 비용 절감</h4>
            <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
              <li>• 작은 모델 사용</li>
              <li>• 프롬프트 압축</li>
              <li>• 토큰 제한</li>
              <li>• 캐시로 중복 방지</li>
            </ul>
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
            <span className="text-gray-700 dark:text-gray-300">LangSmith를 활용한 모니터링과 디버깅</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">다양한 캐싱 전략으로 성능 최적화</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">보안 Best Practices와 Input/Output 필터링</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-600 dark:text-amber-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">LangServe로 프로덕션 API 배포</span>
          </li>
        </ul>
      </section>
    </div>
  );
}
