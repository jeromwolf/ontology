'use client'

import React from 'react'
import { Lightbulb, Zap, Code2, Sparkles, TrendingUp, Users } from 'lucide-react'

export default function Chapter1() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-8">
        <div className="inline-block p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl mb-4">
          <Sparkles className="w-12 h-12 text-purple-500" />
        </div>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
          AI 코딩 혁명의 시작
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          AI가 바꾸는 소프트웨어 개발의 새로운 패러다임과 미래
        </p>
      </div>

      {/* Introduction */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Lightbulb className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">서론: 코딩의 새로운 시대</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-purple-500/20">
          <p className="text-lg text-gray-300 leading-relaxed mb-6">
            2023년 이후, 소프트웨어 개발은 근본적인 변화를 겪고 있습니다. <span className="text-purple-400 font-semibold">AI 코딩 도구</span>의
            등장으로 개발자의 생산성은 <span className="text-pink-400 font-semibold">3-5배</span> 향상되었고,
            코드 작성 방식 자체가 완전히 달라졌습니다.
          </p>

          <div className="grid md:grid-cols-3 gap-4 mt-8">
            <div className="bg-purple-500/10 rounded-lg p-6 border border-purple-500/20">
              <div className="text-4xl font-bold text-purple-400 mb-2">55%</div>
              <div className="text-sm text-gray-400">GitHub Copilot 사용자의 코드 작성 속도 향상</div>
            </div>
            <div className="bg-pink-500/10 rounded-lg p-6 border border-pink-500/20">
              <div className="text-4xl font-bold text-pink-400 mb-2">46%</div>
              <div className="text-sm text-gray-400">버그 발생률 감소 (AI 코드 리뷰 활용 시)</div>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-6 border border-purple-500/20">
              <div className="text-4xl font-bold text-purple-400 mb-2">70%</div>
              <div className="text-sm text-gray-400">개발자가 AI 도구를 일상적으로 사용 (2024)</div>
            </div>
          </div>
        </div>
      </section>

      {/* The AI Coding Revolution */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <Zap className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">AI 코딩 혁명이란?</h2>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">1. 자동 완성의 진화</h3>
            <p className="text-gray-300 mb-4">
              기존 IDE의 단순 자동완성은 변수명이나 메서드 제안에 그쳤습니다.
              AI 코딩 도구는 <span className="text-purple-400">컨텍스트 전체를 이해</span>하고
              다음 코드 블록, 함수, 심지어 전체 파일까지 생성합니다.
            </p>
            <div className="bg-black/30 rounded-lg p-4 font-mono text-sm">
              <div className="text-gray-500">// 기존: 변수명 자동완성</div>
              <div className="text-gray-400">const user<span className="text-green-400">Name</span> = ...</div>
              <div className="text-gray-500 mt-4">// AI: 전체 함수 생성</div>
              <div className="text-gray-400">// "사용자 인증 함수 작성" (주석 입력)</div>
              <div className="text-purple-400 mt-2">
{`async function authenticateUser(email, password) {
  const user = await db.users.findOne({ email });
  if (!user) throw new Error('User not found');
  const isValid = await bcrypt.compare(password, user.passwordHash);
  if (!isValid) throw new Error('Invalid credentials');
  return generateJWT(user);
}`}
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-xl font-semibold text-pink-400 mb-3">2. 자연어 → 코드 변환</h3>
            <p className="text-gray-300 mb-4">
              개발자가 <span className="text-pink-400">자연어로 의도를 설명</span>하면
              AI가 즉시 실행 가능한 코드로 변환합니다. 이는 특히 복잡한 알고리즘이나
              생소한 라이브러리 사용 시 획기적입니다.
            </p>
            <div className="bg-black/30 rounded-lg p-4">
              <div className="mb-2 text-gray-400">
                <span className="text-green-400">입력:</span> "CSV 파일을 읽어서 날짜별로 그룹화하고 합계를 계산해줘"
              </div>
              <div className="text-purple-400 font-mono text-sm">
{`import pandas as pd
from datetime import datetime

def process_csv(file_path):
    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # 날짜 컬럼을 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'])

    # 날짜별로 그룹화하고 합계 계산
    result = df.groupby(df['date'].dt.date).agg({
        'amount': 'sum',
        'quantity': 'sum'
    }).reset_index()

    return result`}
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">3. 코드 설명 및 리팩토링</h3>
            <p className="text-gray-300 mb-4">
              레거시 코드를 이해하거나, 복잡한 로직을 개선할 때 AI가 <span className="text-purple-400">즉각적인 분석과 제안</span>을
              제공합니다. 수천 줄의 코드도 몇 초 만에 분석 가능합니다.
            </p>
          </div>
        </div>
      </section>

      {/* Major AI Coding Tools */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Code2 className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">주요 AI 코딩 도구</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/10 rounded-xl p-6 border border-blue-500/20">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🔵</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-blue-400">GitHub Copilot</h3>
                <p className="text-sm text-gray-400">OpenAI Codex 기반</p>
              </div>
            </div>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-1">•</span>
                <span><strong>출시:</strong> 2021년 6월 (Preview), 2022년 6월 (GA)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-1">•</span>
                <span><strong>강점:</strong> VS Code 완벽 통합, 광범위한 언어 지원</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-1">•</span>
                <span><strong>가격:</strong> $10/월 (개인), $19/월 (비즈니스)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-1">•</span>
                <span><strong>사용자:</strong> 150만+ 유료 구독자 (2024)</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/10 rounded-xl p-6 border border-purple-500/20">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <span className="text-2xl">⚡</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-purple-400">Cursor</h3>
                <p className="text-sm text-gray-400">VS Code 포크 + GPT-4</p>
              </div>
            </div>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                <span><strong>출시:</strong> 2023년 3월</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                <span><strong>강점:</strong> 채팅 인터페이스, 코드베이스 전체 이해</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                <span><strong>가격:</strong> 무료 (제한), $20/월 (Pro)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                <span><strong>혁신:</strong> Cmd+K로 인라인 코드 편집, 멀티파일 수정</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-orange-900/20 to-orange-800/10 rounded-xl p-6 border border-orange-500/20">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-orange-500/20 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🤖</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-orange-400">Claude Code</h3>
                <p className="text-sm text-gray-400">Anthropic Claude 3.5</p>
              </div>
            </div>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-orange-400 mt-1">•</span>
                <span><strong>출시:</strong> 2024년 6월</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-400 mt-1">•</span>
                <span><strong>강점:</strong> 긴 컨텍스트 (200K 토큰), 추론 능력</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-400 mt-1">•</span>
                <span><strong>가격:</strong> Claude Pro $20/월 포함</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-400 mt-1">•</span>
                <span><strong>혁신:</strong> 대규모 리팩토링, 아키텍처 설계 지원</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-green-900/20 to-green-800/10 rounded-xl p-6 border border-green-500/20">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🚀</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-green-400">기타 주요 도구</h3>
                <p className="text-sm text-gray-400">Tabnine, Cody, Replit AI</p>
              </div>
            </div>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">•</span>
                <span><strong>Tabnine:</strong> 온프레미스 배포, 기업용 보안</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">•</span>
                <span><strong>Cody (Sourcegraph):</strong> 코드 검색 + AI</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">•</span>
                <span><strong>Replit AI:</strong> 클라우드 IDE 통합</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* How AI Coding Works */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <TrendingUp className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">AI 코딩은 어떻게 작동하는가?</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-pink-500/20">
          <h3 className="text-xl font-semibold text-pink-400 mb-4">1. 대규모 언어 모델 (LLM) 기반</h3>
          <p className="text-gray-300 mb-6">
            AI 코딩 도구는 수십억 개의 공개 코드 저장소로 학습된 <span className="text-pink-400">거대 언어 모델</span>을
            사용합니다. GPT-4, Claude, Codex 등이 대표적입니다.
          </p>

          <div className="bg-black/30 rounded-lg p-6 space-y-4">
            <div className="flex items-start gap-3">
              <div className="bg-pink-500/20 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">
                <span className="text-pink-400 font-bold">1</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">학습 데이터</h4>
                <p className="text-gray-400 text-sm">
                  GitHub, Stack Overflow, 문서, 오픈소스 프로젝트 (수천억 줄의 코드)
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="bg-pink-500/20 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">
                <span className="text-pink-400 font-bold">2</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">패턴 학습</h4>
                <p className="text-gray-400 text-sm">
                  함수 구조, 명명 규칙, 디자인 패턴, 에러 처리 방식 등을 학습
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="bg-pink-500/20 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">
                <span className="text-pink-400 font-bold">3</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">컨텍스트 이해</h4>
                <p className="text-gray-400 text-sm">
                  현재 파일, 프로젝트 구조, 이전 코드를 분석하여 최적의 제안
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="bg-pink-500/20 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">
                <span className="text-pink-400 font-bold">4</span>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-1">확률 기반 생성</h4>
                <p className="text-gray-400 text-sm">
                  가장 적합한 다음 토큰(코드 조각)을 확률적으로 예측하여 생성
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-purple-500/20">
          <h3 className="text-xl font-semibold text-purple-400 mb-4">2. 실시간 컨텍스트 분석</h3>
          <p className="text-gray-300 mb-4">
            AI는 단순히 코드만 보지 않습니다. 프로젝트의 전체 맥락을 이해합니다:
          </p>
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-purple-400">▸</span>
              <span><strong>파일 구조:</strong> package.json, requirements.txt 등으로 사용 중인 프레임워크 파악</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400">▸</span>
              <span><strong>주변 코드:</strong> 같은 파일 내의 다른 함수, 클래스 참고</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400">▸</span>
              <span><strong>Import 문:</strong> 어떤 라이브러리를 사용하는지 분석</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400">▸</span>
              <span><strong>주석과 문서:</strong> 개발자의 의도를 이해하기 위한 힌트</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400">▸</span>
              <span><strong>코딩 스타일:</strong> 기존 프로젝트의 네이밍, 포맷팅 규칙 학습</span>
            </li>
          </ul>
        </div>
      </section>

      {/* The Impact on Developers */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Users className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">개발자에게 미치는 영향</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-green-900/20 rounded-xl p-6 border border-green-500/20">
            <h3 className="text-xl font-semibold text-green-400 mb-4 flex items-center gap-2">
              <span>✅</span> 긍정적 영향
            </h3>
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">•</span>
                <div>
                  <strong>생산성 급증:</strong> 반복 코드, 보일러플레이트 자동 생성
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">•</span>
                <div>
                  <strong>학습 가속화:</strong> 새로운 언어/프레임워크 빠른 습득
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">•</span>
                <div>
                  <strong>품질 향상:</strong> 베스트 프랙티스 자동 적용, 버그 조기 발견
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-1">•</span>
                <div>
                  <strong>창의성 집중:</strong> 단순 작업 자동화로 아키텍처 설계에 집중
                </div>
              </li>
            </ul>
          </div>

          <div className="bg-red-900/20 rounded-xl p-6 border border-red-500/20">
            <h3 className="text-xl font-semibold text-red-400 mb-4 flex items-center gap-2">
              <span>⚠️</span> 도전 과제
            </h3>
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-red-400 mt-1">•</span>
                <div>
                  <strong>과도한 의존:</strong> 기본 원리 이해 없이 AI에만 의존하는 위험
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-400 mt-1">•</span>
                <div>
                  <strong>보안 우려:</strong> 학습 데이터에 포함된 취약한 코드 패턴 재현 가능
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-400 mt-1">•</span>
                <div>
                  <strong>저작권 문제:</strong> 오픈소스 라이선스 위반 가능성
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-400 mt-1">•</span>
                <div>
                  <strong>환각(Hallucination):</strong> 존재하지 않는 API나 잘못된 로직 생성
                </div>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">AI 코딩 도구 사용 모범 사례</h2>

        <div className="space-y-4">
          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">1. AI는 도구, 당신이 주인</h3>
            <p className="text-gray-300">
              AI가 생성한 코드를 <span className="text-purple-400">맹목적으로 수용하지 마세요</span>.
              항상 검토하고, 이해하고, 필요하면 수정하세요. AI는 제안을 하는 조수일 뿐입니다.
            </p>
          </div>

          <div className="bg-gradient-to-r from-pink-900/20 to-transparent rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-lg font-semibold text-pink-400 mb-2">2. 명확한 프롬프트 작성</h3>
            <p className="text-gray-300">
              좋은 결과는 <span className="text-pink-400">명확한 요청</span>에서 나옵니다.
              "함수 만들어줘" 대신 "이메일 유효성을 정규식으로 검증하고 에러 메시지를 반환하는 TypeScript 함수 작성"처럼 구체적으로 요청하세요.
            </p>
          </div>

          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">3. 테스트 필수</h3>
            <p className="text-gray-300">
              AI가 생성한 코드도 <span className="text-purple-400">반드시 테스트</span>하세요.
              단위 테스트를 작성하거나, AI에게 테스트 코드까지 함께 생성하도록 요청할 수 있습니다.
            </p>
          </div>

          <div className="bg-gradient-to-r from-pink-900/20 to-transparent rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-lg font-semibold text-pink-400 mb-2">4. 보안 검토</h3>
            <p className="text-gray-300">
              특히 인증, 암호화, 데이터베이스 쿼리 등 <span className="text-pink-400">보안이 중요한 코드</span>는
              AI 생성 후 반드시 보안 전문가의 리뷰를 거치세요.
            </p>
          </div>

          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">5. 점진적 도입</h3>
            <p className="text-gray-300">
              한 번에 모든 워크플로우를 AI로 전환하지 마세요.
              <span className="text-purple-400">작은 작업부터 시작</span>하여 신뢰를 쌓고 점차 확대하세요.
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
                AI 코딩 도구는 2023년 이후 소프트웨어 개발의 <strong>표준 도구</strong>가 되었습니다.
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">2.</span>
              <span>
                <strong>GitHub Copilot, Cursor, Claude Code</strong>가 주요 3대 도구입니다.
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">3.</span>
              <span>
                AI는 <strong>대규모 언어 모델</strong>과 <strong>컨텍스트 분석</strong>을 통해 코드를 생성합니다.
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">4.</span>
              <span>
                생산성은 향상되지만, <strong>과도한 의존, 보안, 저작권</strong> 문제를 주의해야 합니다.
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">5.</span>
              <span>
                AI 도구는 <strong>조수</strong>일 뿐, 개발자의 역할은 더욱 중요해졌습니다.
              </span>
            </li>
          </ul>

          <div className="mt-8 p-6 bg-black/30 rounded-lg border border-purple-500/20">
            <p className="text-lg text-purple-400 font-semibold mb-2">다음 챕터 미리보기</p>
            <p className="text-gray-300">
              Chapter 2에서는 <strong>Cursor 에디터</strong>를 완벽하게 마스터합니다.
              설치부터 고급 기능(Cmd+K, Composer, Chat)까지 실전 예제와 함께 배웁니다.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
