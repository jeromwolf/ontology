'use client'

import React from 'react'
import { Bot, FileCode, Sparkles, Database, Layers, Workflow } from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-8">
        <div className="inline-block p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl mb-4">
          <Bot className="w-12 h-12 text-purple-500" />
        </div>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
          Claude Code 엔지니어링
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          200K 토큰 컨텍스트로 대규모 프로젝트 완전 정복
        </p>
      </div>

      {/* Introduction */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">Claude Code란?</h2>

        <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 rounded-xl p-8 border border-purple-500/20">
          <p className="text-lg text-gray-300 leading-relaxed mb-6">
            <span className="text-purple-400 font-semibold">Claude Code</span>는 Anthropic의 Claude 3.5 Sonnet을
            기반으로 한 AI 코딩 도구로, <span className="text-pink-400 font-semibold">200,000 토큰</span>의
            초대형 컨텍스트 윈도우를 자랑합니다. 이는 약 <span className="text-purple-400">150,000 단어</span> 또는
            <span className="text-purple-400"> 500페이지 분량의 책</span>에 해당하며, 대규모 코드베이스 전체를 한 번에 이해할 수 있습니다.
          </p>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
              <div className="text-3xl font-bold text-purple-400 mb-1">200K</div>
              <div className="text-sm text-gray-400">토큰 컨텍스트 (GPT-4의 16배)</div>
            </div>
            <div className="bg-pink-500/10 rounded-lg p-4 border border-pink-500/20">
              <div className="text-3xl font-bold text-pink-400 mb-1">94.4%</div>
              <div className="text-sm text-gray-400">HumanEval 코딩 벤치마크</div>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
              <div className="text-3xl font-bold text-purple-400 mb-1">92.0%</div>
              <div className="text-sm text-gray-400">GSM8K 수학 문제 해결</div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Features */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Sparkles className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">Claude Code의 핵심 강점</h2>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">1. 초대형 컨텍스트 윈도우</h3>
            <p className="text-gray-300 mb-4">
              200K 토큰은 실제로 무엇을 의미할까요?
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-purple-900/20 rounded-lg p-4">
                <div className="text-purple-400 font-semibold mb-2">📂 코드베이스</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• 중형 프로젝트 전체 (50-100개 파일)</li>
                  <li>• 대형 프로젝트 주요 모듈</li>
                  <li>• API 문서 + 코드 동시 참조</li>
                </ul>
              </div>
              <div className="bg-pink-900/20 rounded-lg p-4">
                <div className="text-pink-400 font-semibold mb-2">📚 텍스트</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• 소설 1권 분량 (~500 페이지)</li>
                  <li>• 기술 문서 수십 개</li>
                  <li>• 전체 대화 히스토리 유지</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-xl font-semibold text-pink-400 mb-3">2. Artifacts (실시간 미리보기)</h3>
            <p className="text-gray-300 mb-4">
              Claude가 생성한 코드를 <span className="text-pink-400">즉시 실행하고 미리보기</span>할 수 있는
              독특한 기능입니다. HTML, React, SVG, Mermaid 다이어그램 등을 실시간으로 렌더링합니다.
            </p>
            <div className="bg-black/30 rounded-lg p-4">
              <div className="text-gray-500 text-sm mb-2">// 요청 예시</div>
              <div className="text-purple-400 mb-3">
                "사용자 대시보드를 React로 만들어줘. 차트, 테이블, 통계 카드 포함"
              </div>
              <div className="text-gray-400 text-sm">
                → Claude가 React 컴포넌트 생성<br/>
                → 오른쪽 패널에서 실시간 미리보기<br/>
                → 수정 요청 시 즉시 업데이트
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">3. 뛰어난 추론 능력</h3>
            <p className="text-gray-300 mb-4">
              Claude는 단순한 코드 생성을 넘어 <span className="text-purple-400">아키텍처 설계, 버그 분석, 최적화 전략</span>까지
              제안할 수 있습니다.
            </p>
            <div className="space-y-3 text-gray-300 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-purple-400">▸</span>
                <span><strong>아키텍처 리뷰:</strong> "이 마이크로서비스 구조의 문제점과 개선 방안은?"</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-purple-400">▸</span>
                <span><strong>성능 분석:</strong> "왜 이 쿼리가 느린지 분석하고 인덱스 전략 제안해줘"</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-purple-400">▸</span>
                <span><strong>보안 감사:</strong> "이 인증 로직에 보안 취약점이 있는지 확인해줘"</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-xl font-semibold text-pink-400 mb-3">4. 프로젝트 생성 (Projects)</h3>
            <p className="text-gray-300 mb-4">
              Claude Projects는 <span className="text-pink-400">프로젝트별로 독립적인 대화 공간</span>을 제공합니다.
              커스텀 지시사항, 문서, 코드 스니펫을 저장하여 재사용할 수 있습니다.
            </p>
            <div className="bg-black/30 rounded-lg p-4 space-y-2 text-sm">
              <div className="text-purple-400">• 프로젝트당 최대 10MB 문서 업로드</div>
              <div className="text-pink-400">• 커스텀 지시사항으로 AI 동작 조정</div>
              <div className="text-purple-400">• 팀원과 프로젝트 공유 가능</div>
            </div>
          </div>
        </div>
      </section>

      {/* Usage Examples */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <FileCode className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">실전 활용 사례</h2>
        </div>

        <div className="space-y-6">
          <div className="bg-black/30 rounded-lg p-6 border border-purple-500/20">
            <h4 className="text-purple-400 font-semibold mb-4">사례 1: 대규모 리팩토링</h4>
            <div className="space-y-4">
              <div className="bg-gray-900 rounded p-4">
                <div className="text-pink-400 font-semibold mb-2">요청:</div>
                <div className="text-gray-300 text-sm mb-3">
                  "이 Express 앱을 NestJS로 마이그레이션하고 싶어. 현재 구조를 분석하고 단계별 계획을 세워줘."
                </div>
                <div className="text-purple-400 font-semibold mb-2">업로드:</div>
                <div className="text-gray-400 text-sm">
                  • src/ 디렉토리 전체 (50개 파일, ~15,000줄)
                </div>
              </div>
              <div className="bg-gray-900 rounded p-4">
                <div className="text-green-400 font-semibold mb-2">Claude의 응답:</div>
                <div className="text-gray-300 text-sm space-y-2">
                  <div>1. 현재 구조 분석 (라우터, 미들웨어, DB 모델)</div>
                  <div>2. NestJS 모듈 구조 설계</div>
                  <div>3. 우선순위별 마이그레이션 단계 (12단계)</div>
                  <div>4. 각 단계별 상세 코드 예제</div>
                  <div>5. 테스트 전략 및 롤백 계획</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-black/30 rounded-lg p-6 border border-pink-500/20">
            <h4 className="text-pink-400 font-semibold mb-4">사례 2: API 문서로 클라이언트 SDK 생성</h4>
            <div className="space-y-4">
              <div className="bg-gray-900 rounded p-4">
                <div className="text-purple-400 font-semibold mb-2">요청:</div>
                <div className="text-gray-300 text-sm">
                  "Stripe API 문서를 보고 TypeScript SDK를 만들어줘. Payment Intents, Customers, Subscriptions API 포함."
                </div>
              </div>
              <div className="bg-gray-900 rounded p-4">
                <div className="text-gray-500 mb-2 text-sm">// Claude 생성 코드</div>
                <pre className="text-purple-400 text-xs overflow-x-auto">
{`class StripeClient {
  constructor(private apiKey: string) {}

  async createPaymentIntent(params: PaymentIntentParams): Promise<PaymentIntent> {
    const response = await fetch('https://api.stripe.com/v1/payment_intents', {
      method: 'POST',
      headers: {
        'Authorization': \`Bearer \${this.apiKey}\`,
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: new URLSearchParams(params as any)
    });
    return response.json();
  }

  // ... 30+ 메서드 자동 생성
}`}
                </pre>
              </div>
            </div>
          </div>

          <div className="bg-black/30 rounded-lg p-6 border border-purple-500/20">
            <h4 className="text-purple-400 font-semibold mb-4">사례 3: 레거시 코드 이해 및 문서화</h4>
            <div className="space-y-4">
              <div className="bg-gray-900 rounded p-4">
                <div className="text-pink-400 font-semibold mb-2">상황:</div>
                <div className="text-gray-300 text-sm">
                  5년 전 작성된 주석 없는 PHP 코드 10,000줄
                </div>
              </div>
              <div className="bg-gray-900 rounded p-4">
                <div className="text-green-400 font-semibold mb-2">Claude 활용:</div>
                <div className="text-gray-300 text-sm space-y-2">
                  <div>1. 전체 코드를 컨텍스트로 업로드</div>
                  <div>2. "이 시스템의 전체 아키텍처를 설명해줘"</div>
                  <div>3. 클래스 다이어그램, 시퀀스 다이어그램 생성</div>
                  <div>4. 각 모듈별 상세 문서 자동 생성</div>
                  <div>5. 개선 필요 부분 식별 및 제안</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Advanced Techniques */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Layers className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">고급 기법</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-purple-900/20 to-transparent rounded-xl p-6 border border-purple-500/20">
            <h3 className="text-xl font-semibold text-purple-400 mb-4">1. 체인 프롬프팅</h3>
            <p className="text-gray-300 mb-4 text-sm">
              복잡한 작업을 단계별로 나누어 요청하면 더 정확한 결과를 얻습니다.
            </p>
            <div className="space-y-2 text-gray-400 text-xs">
              <div className="border-l-4 border-purple-500 pl-3">
                <strong>Step 1:</strong> "이 프로젝트의 구조를 분석해줘"
              </div>
              <div className="border-l-4 border-pink-500 pl-3">
                <strong>Step 2:</strong> "인증 모듈을 어떻게 개선할 수 있을까?"
              </div>
              <div className="border-l-4 border-purple-500 pl-3">
                <strong>Step 3:</strong> "제안한 방법을 코드로 구현해줘"
              </div>
              <div className="border-l-4 border-pink-500 pl-3">
                <strong>Step 4:</strong> "테스트 코드도 작성해줘"
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-pink-900/20 to-transparent rounded-xl p-6 border border-pink-500/20">
            <h3 className="text-xl font-semibold text-pink-400 mb-4">2. 컨텍스트 최적화</h3>
            <p className="text-gray-300 mb-4 text-sm">
              200K 토큰도 무한하지 않습니다. 효율적으로 사용하는 방법:
            </p>
            <ul className="space-y-2 text-gray-300 text-xs">
              <li className="flex items-start gap-2">
                <span className="text-pink-400">•</span>
                <span>필요한 파일만 선택적으로 업로드</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-pink-400">•</span>
                <span>주요 인터페이스와 타입 정의 우선 제공</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-pink-400">•</span>
                <span>긴 로그 파일 대신 에러 메시지만 추출</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-pink-400">•</span>
                <span>중복 코드는 대표 예시 하나만</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-purple-900/20 to-transparent rounded-xl p-6 border border-purple-500/20">
            <h3 className="text-xl font-semibold text-purple-400 mb-4">3. 프로젝트 템플릿 활용</h3>
            <p className="text-gray-300 mb-4 text-sm">
              자주 사용하는 프로젝트 설정을 Claude Projects에 저장:
            </p>
            <div className="bg-black/30 rounded p-3 text-xs text-gray-400">
              <div className="mb-2 text-purple-400">프로젝트 지시사항 예시:</div>
              <div className="space-y-1">
                <div>• TypeScript strict 모드 사용</div>
                <div>• ESLint + Prettier 적용</div>
                <div>• Jest로 테스트 작성</div>
                <div>• Tailwind CSS 스타일링</div>
                <div>• tRPC API 통신</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-pink-900/20 to-transparent rounded-xl p-6 border border-pink-500/20">
            <h3 className="text-xl font-semibold text-pink-400 mb-4">4. Artifacts 활용</h3>
            <p className="text-gray-300 mb-4 text-sm">
              실시간 미리보기로 UI 개발 속도 극대화:
            </p>
            <div className="space-y-2 text-gray-300 text-xs">
              <div className="flex items-start gap-2">
                <span className="text-pink-400">▸</span>
                <span>React 컴포넌트 즉시 렌더링</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-pink-400">▸</span>
                <span>HTML/CSS 실시간 수정</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-pink-400">▸</span>
                <span>SVG 아이콘/로고 생성</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-pink-400">▸</span>
                <span>Mermaid 다이어그램 시각화</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Workflow Integration */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <Workflow className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">워크플로우 통합</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-pink-500/20">
          <h3 className="text-xl font-semibold text-pink-400 mb-4">Claude Code 중심 개발 프로세스</h3>

          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="bg-purple-500/20 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                <span className="text-purple-400 font-bold">1</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">요구사항 분석</h4>
                <p className="text-gray-400 text-sm">
                  Claude에게 프로젝트 개요를 설명하고 아키텍처 설계 요청
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="bg-pink-500/20 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                <span className="text-pink-400 font-bold">2</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">프로젝트 초기화</h4>
                <p className="text-gray-400 text-sm">
                  "Next.js 14 + TypeScript + Prisma + tRPC 프로젝트 생성해줘" → 전체 구조 생성
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="bg-purple-500/20 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                <span className="text-purple-400 font-bold">3</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">기능 개발</h4>
                <p className="text-gray-400 text-sm">
                  각 기능을 Claude와 대화하며 구현 (인증, API, UI 등)
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="bg-pink-500/20 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                <span className="text-pink-400 font-bold">4</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">코드 리뷰</h4>
                <p className="text-gray-400 text-sm">
                  Claude에게 코드 리뷰 요청: "이 인증 로직에 보안 문제가 있나요?"
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="bg-purple-500/20 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                <span className="text-purple-400 font-bold">5</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">테스트 작성</h4>
                <p className="text-gray-400 text-sm">
                  모든 함수와 컴포넌트에 대한 테스트 코드 자동 생성
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="bg-pink-500/20 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                <span className="text-pink-400 font-bold">6</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">문서화</h4>
                <p className="text-gray-400 text-sm">
                  README, API 문서, 아키텍처 다이어그램 자동 생성
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">Claude Code 마스터 팁</h2>

        <div className="space-y-4">
          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">1. 긴 컨텍스트 활용</h3>
            <p className="text-gray-300">
              프로젝트 전체를 한 번에 업로드하여 <span className="text-purple-400">전역적인 리팩토링</span>이나
              <span className="text-purple-400"> 일관성 검사</span>를 수행하세요.
            </p>
          </div>

          <div className="bg-gradient-to-r from-pink-900/20 to-transparent rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-lg font-semibold text-pink-400 mb-2">2. Artifacts로 빠른 프로토타이핑</h3>
            <p className="text-gray-300">
              UI 아이디어를 즉시 시각화하고 고객에게 <span className="text-pink-400">실시간으로 데모</span>할 수 있습니다.
            </p>
          </div>

          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">3. 프로젝트별 분리</h3>
            <p className="text-gray-300">
              각 프로젝트마다 <span className="text-purple-400">독립적인 Claude Projects</span>를 생성하여
              컨텍스트가 섞이지 않도록 관리하세요.
            </p>
          </div>

          <div className="bg-gradient-to-r from-pink-900/20 to-transparent rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-lg font-semibold text-pink-400 mb-2">4. API 활용</h3>
            <p className="text-gray-300">
              Claude API를 직접 사용하면 <span className="text-pink-400">자동화 스크립트</span>나
              <span className="text-pink-400"> CI/CD 파이프라인</span>에 통합할 수 있습니다.
            </p>
          </div>
        </div>
      </section>

      {/* Summary */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">요약</h2>
        <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 rounded-xl p-8 border border-purple-500/30">
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">1.</span>
              <span>
                Claude Code는 <strong>200K 토큰 컨텍스트</strong>로 대규모 프로젝트 전체를 이해
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">2.</span>
              <span>
                <strong>Artifacts</strong>로 코드를 즉시 실행하고 미리보기 가능
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">3.</span>
              <span>
                <strong>뛰어난 추론 능력</strong>으로 아키텍처 설계부터 보안 감사까지 지원
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">4.</span>
              <span>
                <strong>Claude Projects</strong>로 프로젝트별 독립적인 컨텍스트 관리
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">5.</span>
              <span>
                레거시 코드 이해, 대규모 리팩토링, API SDK 생성에 특히 강력
              </span>
            </li>
          </ul>

          <div className="mt-8 p-6 bg-black/30 rounded-lg border border-purple-500/20">
            <p className="text-lg text-purple-400 font-semibold mb-2">다음 챕터 미리보기</p>
            <p className="text-gray-300">
              Chapter 5에서는 <strong>AI 코딩을 위한 프롬프트 엔지니어링</strong>을 배웁니다.
              명확한 요청부터 고급 패턴까지, AI에게서 최고의 결과를 얻는 기술을 학습합니다.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
