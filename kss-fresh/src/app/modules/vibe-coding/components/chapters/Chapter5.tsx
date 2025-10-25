'use client'

import React from 'react'
import { MessageSquare, Target, Zap, CheckCircle2, AlertCircle, Lightbulb } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-8">
        <div className="inline-block p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl mb-4">
          <MessageSquare className="w-12 h-12 text-purple-500" />
        </div>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
          AI 코딩을 위한 프롬프트 엔지니어링
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          AI에게서 최고의 코드를 얻는 의사소통 기술
        </p>
      </div>

      {/* Introduction */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">프롬프트 엔지니어링이란?</h2>

        <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 rounded-xl p-8 border border-purple-500/20">
          <p className="text-lg text-gray-300 leading-relaxed mb-6">
            <span className="text-purple-400 font-semibold">프롬프트 엔지니어링</span>은 AI 모델로부터
            원하는 결과를 얻기 위해 <span className="text-pink-400 font-semibold">입력(프롬프트)을 최적화</span>하는 기술입니다.
            같은 요청이라도 <span className="text-purple-400">표현 방식에 따라 결과가 크게 달라집니다</span>.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-red-900/20 rounded-lg p-6 border border-red-500/20">
              <div className="text-red-400 font-semibold mb-3 flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                나쁜 프롬프트
              </div>
              <div className="bg-black/30 rounded p-3 text-gray-400 text-sm font-mono">
                "로그인 기능 만들어줘"
              </div>
              <div className="mt-3 text-gray-400 text-sm">
                → 불명확, 구체성 부족, 요구사항 누락
              </div>
            </div>

            <div className="bg-green-900/20 rounded-lg p-6 border border-green-500/20">
              <div className="text-green-400 font-semibold mb-3 flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5" />
                좋은 프롬프트
              </div>
              <div className="bg-black/30 rounded p-3 text-gray-400 text-sm font-mono">
                "JWT 기반 인증 시스템을 TypeScript로 구현. 로그인/회원가입 API, 토큰 갱신, 에러 처리 포함. Express 미들웨어 형태로 작성"
              </div>
              <div className="mt-3 text-gray-400 text-sm">
                → 명확, 구체적, 기술 스택 명시, 요구사항 상세
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Core Principles */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Target className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">프롬프트 작성 5대 원칙</h2>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">1. 명확성 (Clarity)</h3>
            <p className="text-gray-300 mb-4">
              모호한 표현을 피하고 <span className="text-purple-400">정확한 용어</span>를 사용하세요.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="text-red-400 text-sm mb-2">❌ 모호함</div>
                <div className="bg-black/30 rounded p-3 text-sm text-gray-400">
                  "데이터를 저장해줘"
                </div>
              </div>
              <div>
                <div className="text-green-400 text-sm mb-2">✅ 명확함</div>
                <div className="bg-black/30 rounded p-3 text-sm text-purple-400">
                  "사용자 객체를 PostgreSQL 데이터베이스의 users 테이블에 INSERT 쿼리로 저장"
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-xl font-semibold text-pink-400 mb-3">2. 구체성 (Specificity)</h3>
            <p className="text-gray-300 mb-4">
              기술 스택, 라이브러리 버전, 패턴 등을 <span className="text-pink-400">명시</span>하세요.
            </p>
            <div className="bg-black/30 rounded p-4 space-y-2 text-sm">
              <div className="text-gray-500">// 나쁜 예</div>
              <div className="text-gray-400">"React로 컴포넌트 만들어줘"</div>
              <div className="text-gray-500 mt-3">// 좋은 예</div>
              <div className="text-pink-400">
                "React 18 + TypeScript로 함수형 컴포넌트 작성. useState, useEffect 훅 사용. Tailwind CSS로 스타일링. Props는 interface로 타입 정의"
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">3. 컨텍스트 (Context)</h3>
            <p className="text-gray-300 mb-4">
              프로젝트 배경, 기존 코드, 제약사항 등 <span className="text-purple-400">관련 정보</span>를 제공하세요.
            </p>
            <div className="bg-black/30 rounded p-4 text-sm text-gray-300">
              <div className="text-purple-400 mb-2">좋은 컨텍스트 예시:</div>
              <div className="space-y-1">
                <div>• "이 프로젝트는 Next.js 14 App Router를 사용합니다"</div>
                <div>• "기존 API는 tRPC로 작성되어 있으니 같은 패턴 따라주세요"</div>
                <div>• "이 함수는 초당 1000번 호출되므로 성능이 중요합니다"</div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-xl font-semibold text-pink-400 mb-3">4. 예제 (Examples)</h3>
            <p className="text-gray-300 mb-4">
              원하는 <span className="text-pink-400">결과의 예시</span>를 제공하면 AI가 패턴을 이해합니다.
            </p>
            <div className="bg-black/30 rounded p-4 font-mono text-xs text-gray-400">
{`// 예제 제공 프롬프트
"다음과 같은 형식으로 유효성 검사 함수를 만들어줘:

function validateEmail(email: string): { valid: boolean; error?: string } {
  // implementation
}

이 패턴을 따라서 validatePassword, validateUsername도 작성해줘"`}
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">5. 제약사항 (Constraints)</h3>
            <p className="text-gray-300 mb-4">
              <span className="text-purple-400">하지 말아야 할 것</span>을 명확히 알려주세요.
            </p>
            <div className="bg-black/30 rounded p-4 text-sm text-gray-300">
              <div>• "절대 any 타입 사용하지 말 것"</div>
              <div>• "외부 라이브러리 추가 금지, 순수 JavaScript만 사용"</div>
              <div>• "async/await 대신 Promise 체이닝 사용"</div>
              <div>• "코드 길이 50줄 이내로 제한"</div>
            </div>
          </div>
        </div>
      </section>

      {/* Advanced Patterns */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <Zap className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">고급 프롬프트 패턴</h2>
        </div>

        <div className="space-y-6">
          <div className="bg-black/30 rounded-lg p-6 border border-purple-500/20">
            <h4 className="text-purple-400 font-semibold mb-4">패턴 1: 체인 오브 씽킹 (Chain-of-Thought)</h4>
            <p className="text-gray-300 mb-4 text-sm">
              AI에게 단계별로 생각하도록 유도하면 더 정확한 결과를 얻습니다.
            </p>
            <div className="bg-gray-900 rounded p-4 text-sm">
              <div className="text-pink-400 mb-2">프롬프트:</div>
              <div className="text-gray-400">
                "사용자 인증 시스템을 설계해줘. 먼저 다음 단계를 따라주세요:<br/><br/>
                1단계: 필요한 데이터베이스 테이블 설계<br/>
                2단계: API 엔드포인트 목록 작성<br/>
                3단계: 각 엔드포인트의 로직 구현<br/>
                4단계: 에러 처리 및 보안 강화<br/>
                5단계: 테스트 케이스 작성"
              </div>
            </div>
          </div>

          <div className="bg-black/30 rounded-lg p-6 border border-pink-500/20">
            <h4 className="text-pink-400 font-semibold mb-4">패턴 2: 롤 플레잉 (Role-Playing)</h4>
            <p className="text-gray-300 mb-4 text-sm">
              AI에게 특정 역할을 부여하면 해당 관점에서 답변합니다.
            </p>
            <div className="bg-gray-900 rounded p-4 text-sm space-y-3">
              <div>
                <div className="text-purple-400">시니어 개발자 역할:</div>
                <div className="text-gray-400">
                  "당신은 10년 경력의 시니어 백엔드 개발자입니다. 이 API 설계를 리뷰하고 보안, 성능, 확장성 관점에서 개선점을 제안해주세요."
                </div>
              </div>
              <div>
                <div className="text-pink-400">보안 전문가 역할:</div>
                <div className="text-gray-400">
                  "당신은 사이버 보안 전문가입니다. 이 로그인 로직에서 보안 취약점을 찾아내고 OWASP Top 10 기준으로 분석해주세요."
                </div>
              </div>
            </div>
          </div>

          <div className="bg-black/30 rounded-lg p-6 border border-purple-500/20">
            <h4 className="text-purple-400 font-semibold mb-4">패턴 3: 퓨샷 러닝 (Few-Shot Learning)</h4>
            <p className="text-gray-300 mb-4 text-sm">
              2-3개의 예제를 제공하면 AI가 패턴을 학습합니다.
            </p>
            <div className="bg-gray-900 rounded p-4 font-mono text-xs text-gray-400">
{`// 예제 1
function getUserById(id: string): Promise<User> { /* ... */ }

// 예제 2
function createUser(data: CreateUserInput): Promise<User> { /* ... */ }

// 이제 이 패턴을 따라서 updateUser, deleteUser 함수도 작성해줘`}
            </div>
          </div>

          <div className="bg-black/30 rounded-lg p-6 border border-pink-500/20">
            <h4 className="text-pink-400 font-semibold mb-4">패턴 4: 반복 개선 (Iterative Refinement)</h4>
            <p className="text-gray-300 mb-4 text-sm">
              첫 결과를 받은 후 점진적으로 개선 요청을 합니다.
            </p>
            <div className="space-y-3 text-sm">
              <div className="border-l-4 border-purple-500 pl-3">
                <strong className="text-purple-400">1차 요청:</strong> "간단한 TODO 앱 만들어줘"
              </div>
              <div className="border-l-4 border-pink-500 pl-3">
                <strong className="text-pink-400">2차 요청:</strong> "여기에 카테고리 필터 추가해줘"
              </div>
              <div className="border-l-4 border-purple-500 pl-3">
                <strong className="text-purple-400">3차 요청:</strong> "localStorage에 데이터 저장하도록 개선"
              </div>
              <div className="border-l-4 border-pink-500 pl-3">
                <strong className="text-pink-400">4차 요청:</strong> "드래그 앤 드롭으로 순서 변경 기능 추가"
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Real Examples */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Lightbulb className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">실전 프롬프트 예제</h2>
        </div>

        <div className="space-y-6">
          <div className="bg-gray-800/50 rounded-lg p-6">
            <h4 className="text-purple-400 font-semibold mb-3">시나리오 1: React 컴포넌트 생성</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-red-900/10 rounded-lg p-4 border border-red-500/20">
                <div className="text-red-400 text-sm mb-2">❌ 나쁜 프롬프트</div>
                <div className="bg-black/30 rounded p-3 text-xs text-gray-400">
                  "데이터 테이블 컴포넌트 만들어줘"
                </div>
              </div>
              <div className="bg-green-900/10 rounded-lg p-4 border border-green-500/20">
                <div className="text-green-400 text-sm mb-2">✅ 좋은 프롬프트</div>
                <div className="bg-black/30 rounded p-3 text-xs text-purple-400">
{`"React 18 + TypeScript로 DataTable 컴포넌트 작성:
- Props: data (any[]), columns (Column[]), onRowClick
- 기능: 정렬, 페이지네이션 (10개/페이지), 검색
- 스타일: Tailwind CSS, dark mode 지원
- 반응형: 모바일에서는 카드 레이아웃
- 타입: 모든 Props와 State를 interface로 정의"`}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6">
            <h4 className="text-pink-400 font-semibold mb-3">시나리오 2: API 엔드포인트 작성</h4>
            <div className="bg-black/30 rounded p-4 font-mono text-xs text-purple-400">
{`"Node.js + Express로 RESTful API 작성:

엔드포인트: POST /api/orders
기능: 주문 생성 및 결제 처리
요청 바디: { productId, quantity, paymentMethod }
비즈니스 로직:
  1. 재고 확인 (재고 부족 시 400 에러)
  2. 가격 계산 (할인 쿠폰 적용)
  3. 결제 처리 (Stripe API 호출)
  4. 주문 생성 (Prisma ORM 사용)
  5. 재고 차감 (트랜잭션 처리)
  6. 확인 이메일 발송 (비동기, 실패해도 주문은 완료)
에러 처리: 각 단계별 try-catch, 상세한 에러 메시지
응답: { orderId, status, estimatedDelivery }`}
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6">
            <h4 className="text-purple-400 font-semibold mb-3">시나리오 3: 알고리즘 구현</h4>
            <div className="bg-black/30 rounded p-4 font-mono text-xs text-pink-400">
{`"이진 탐색 트리(BST)를 TypeScript로 구현:

클래스: BinarySearchTree<T>
메서드:
  - insert(value: T): void (중복 허용 안 함)
  - search(value: T): boolean
  - delete(value: T): boolean
  - inOrder(): T[] (정렬된 배열 반환)
  - getMin(): T | null
  - getMax(): T | null
제네릭 타입 T는 비교 가능해야 함 (Comparable 인터페이스 정의)
시간 복잡도: 각 메서드 주석으로 명시
테스트: Jest로 단위 테스트 포함 (엣지 케이스 포함)"`}
            </div>
          </div>
        </div>
      </section>

      {/* Common Mistakes */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">흔한 실수와 해결책</h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-red-900/10 rounded-lg p-6 border border-red-500/20">
            <h4 className="text-red-400 font-semibold mb-3">❌ 실수 1: 너무 모호함</h4>
            <div className="text-gray-400 text-sm mb-2">"코드 최적화해줘"</div>
            <div className="text-gray-500 text-xs">
              → 무엇을 최적화? 속도? 메모리? 가독성?
            </div>
          </div>

          <div className="bg-green-900/10 rounded-lg p-6 border border-green-500/20">
            <h4 className="text-green-400 font-semibold mb-3">✅ 해결책</h4>
            <div className="text-gray-300 text-sm mb-2">"이 함수의 시간 복잡도를 O(n²)에서 O(n log n)으로 개선하고 메모리 사용량도 줄여줘"</div>
          </div>

          <div className="bg-red-900/10 rounded-lg p-6 border border-red-500/20">
            <h4 className="text-red-400 font-semibold mb-3">❌ 실수 2: 컨텍스트 부족</h4>
            <div className="text-gray-400 text-sm">"버그 수정해줘"</div>
            <div className="text-gray-500 text-xs">
              → 어떤 버그? 에러 메시지는? 재현 방법은?
            </div>
          </div>

          <div className="bg-green-900/10 rounded-lg p-6 border border-green-500/20">
            <h4 className="text-green-400 font-semibold mb-3">✅ 해결책</h4>
            <div className="text-gray-300 text-sm">
              "useEffect에서 'Cannot read property of undefined' 에러 발생. userId가 null일 때도 API 호출하는 문제. userId가 유효할 때만 실행되도록 수정"
            </div>
          </div>

          <div className="bg-red-900/10 rounded-lg p-6 border border-red-500/20">
            <h4 className="text-red-400 font-semibold mb-3">❌ 실수 3: 한 번에 너무 많이</h4>
            <div className="text-gray-400 text-sm">
              "전체 앱 만들어줘: 로그인, CRUD, 결제, 관리자 대시보드, 모바일 앱"
            </div>
          </div>

          <div className="bg-green-900/10 rounded-lg p-6 border border-green-500/20">
            <h4 className="text-green-400 font-semibold mb-3">✅ 해결책</h4>
            <div className="text-gray-300 text-sm">
              "먼저 사용자 인증 시스템만 구현하자. 완성되면 다음 기능으로 넘어갈게"
            </div>
          </div>

          <div className="bg-red-900/10 rounded-lg p-6 border border-red-500/20">
            <h4 className="text-red-400 font-semibold mb-3">❌ 실수 4: 제약사항 미명시</h4>
            <div className="text-gray-400 text-sm">
              "인증 시스템 만들어줘"
            </div>
            <div className="text-gray-500 text-xs">
              → AI가 Session 기반을 만들었는데 JWT가 필요했다면?
            </div>
          </div>

          <div className="bg-green-900/10 rounded-lg p-6 border border-green-500/20">
            <h4 className="text-green-400 font-semibold mb-3">✅ 해결책</h4>
            <div className="text-gray-300 text-sm">
              "JWT 기반 인증 (Session 사용 금지), httpOnly 쿠키 저장, Refresh Token 포함"
            </div>
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
                프롬프트는 <strong>명확하고, 구체적이고, 컨텍스트가 풍부</strong>해야 함
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">2.</span>
              <span>
                <strong>예제 제공, 제약사항 명시</strong>로 정확도 향상
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">3.</span>
              <span>
                <strong>체인 오브 씽킹, 롤 플레잉</strong> 등 고급 패턴 활용
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">4.</span>
              <span>
                한 번에 완벽한 결과를 기대하지 말고 <strong>반복 개선</strong> 접근
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">5.</span>
              <span>
                프롬프트 작성도 <strong>기술</strong>이며 연습으로 향상됨
              </span>
            </li>
          </ul>

          <div className="mt-8 p-6 bg-black/30 rounded-lg border border-purple-500/20">
            <p className="text-lg text-purple-400 font-semibold mb-2">다음 챕터 미리보기</p>
            <p className="text-gray-300">
              Chapter 6에서는 <strong>AI 기반 테스트 자동 생성</strong>을 배웁니다.
              단위 테스트부터 E2E 테스트까지, AI로 테스트 코드를 빠르게 작성하는 방법을 학습합니다.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
