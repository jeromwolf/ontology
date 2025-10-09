'use client';

import React from 'react';
import { Briefcase, TrendingUp, DollarSign, Clock } from 'lucide-react';
import References from '@/components/common/References';

export default function Section4() {
  return (
    <>
      <section className="bg-gradient-to-r from-purple-100 to-blue-100 dark:from-purple-900/20 dark:to-blue-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Briefcase className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          💼 실전 프로덕션 사례: CrewAI 활용
        </h3>

        {/* 사례 1: 콘텐츠 제작 자동화 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-5 mb-4">
          <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3 text-lg">
            🎬 사례 1: 멀티미디어 콘텐츠 제작 자동화
          </h4>
          <div className="space-y-3 text-sm">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3">
              <strong className="text-purple-800 dark:text-purple-300">요구사항:</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                기업 블로그, 소셜 미디어, 비디오 스크립트를 일관된 브랜드 보이스로 자동 생성
              </p>
            </div>

            <div className="ml-4 space-y-2">
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-blue-700 dark:text-blue-300">Research Agent (Researcher)</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 트렌드 분석 및 키워드 리서치</li>
                  <li>• 경쟁사 콘텐츠 분석</li>
                  <li>• SEO 최적화 제안</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-green-700 dark:text-green-300">Writer Agent (Content Creator)</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 블로그 포스트 작성 (1,500+ 단어)</li>
                  <li>• 소셜 미디어 카피 생성 (Twitter, LinkedIn)</li>
                  <li>• 브랜드 톤앤매너 일관성 유지</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-orange-700 dark:text-orange-300">Editor Agent (Quality Controller)</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 문법 및 스타일 검토</li>
                  <li>• 사실관계 확인 (Fact-checking)</li>
                  <li>• 브랜드 가이드라인 준수 검증</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-purple-700 dark:text-purple-300">Publisher Agent (Distribution Manager)</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 최적 게시 시간 선택</li>
                  <li>• 멀티채널 동시 배포</li>
                  <li>• A/B 테스트 설정</li>
                </ul>
              </div>
            </div>

            <div className="mt-3 grid md:grid-cols-3 gap-3">
              <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="w-4 h-4 text-green-600 dark:text-green-400" />
                  <strong className="text-green-800 dark:text-green-300">제작 시간</strong>
                </div>
                <p className="text-gray-600 dark:text-gray-400">8시간 → 45분 <span className="text-green-600 font-semibold">(90% 감소)</span></p>
              </div>

              <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <strong className="text-blue-800 dark:text-blue-300">품질 점수</strong>
                </div>
                <p className="text-gray-600 dark:text-gray-400">평균 7.2 → 8.8 <span className="text-blue-600 font-semibold">(+22%)</span></p>
              </div>

              <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                <div className="flex items-center gap-2 mb-1">
                  <DollarSign className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  <strong className="text-purple-800 dark:text-purple-300">비용 절감</strong>
                </div>
                <p className="text-gray-600 dark:text-gray-400">월 $12,000 → $2,400 <span className="text-purple-600 font-semibold">(80%)</span></p>
              </div>
            </div>
          </div>
        </div>

        {/* 사례 2: 고객 지원 자동화 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-5 mb-4">
          <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-3 text-lg">
            🎧 사례 2: 다층 고객 지원 시스템
          </h4>
          <div className="space-y-3 text-sm">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
              <strong className="text-blue-800 dark:text-blue-300">요구사항:</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                24/7 멀티채널 고객 지원 (이메일, 채팅, 전화) + 자동 에스컬레이션
              </p>
            </div>

            <div className="ml-4 space-y-2">
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-blue-700 dark:text-blue-300">Triage Agent (First Contact)</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 문의 유형 자동 분류 (9개 카테고리)</li>
                  <li>• 긴급도 평가 (P0~P3)</li>
                  <li>• 고객 감정 분석 (sentiment analysis)</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-green-700 dark:text-green-300">Technical Support Agent</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• Knowledge Base 검색 및 솔루션 제공</li>
                  <li>• 단계별 트러블슈팅 가이드</li>
                  <li>• 시스템 로그 분석 (자동 진단)</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-orange-700 dark:text-orange-300">Account Manager Agent</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 결제/환불 처리</li>
                  <li>• 계정 보안 이슈 해결</li>
                  <li>• 구독 변경 및 업그레이드 제안</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-red-700 dark:text-red-300">Escalation Manager Agent</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 복잡한 케이스 인간 상담원에게 전달</li>
                  <li>• 전체 대화 컨텍스트 요약</li>
                  <li>• 추천 해결 방안 제시</li>
                </ul>
              </div>
            </div>

            <div className="mt-3 grid md:grid-cols-4 gap-2">
              <div className="p-2 bg-green-50 dark:bg-green-900/20 rounded text-center">
                <strong className="text-green-800 dark:text-green-300 text-xs">해결률</strong>
                <p className="text-lg font-bold text-green-600 dark:text-green-400">89%</p>
                <p className="text-xs text-gray-500">자동 해결</p>
              </div>

              <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-center">
                <strong className="text-blue-800 dark:text-blue-300 text-xs">응답 시간</strong>
                <p className="text-lg font-bold text-blue-600 dark:text-blue-400">&lt;30초</p>
                <p className="text-xs text-gray-500">평균</p>
              </div>

              <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded text-center">
                <strong className="text-purple-800 dark:text-purple-300 text-xs">만족도</strong>
                <p className="text-lg font-bold text-purple-600 dark:text-purple-400">4.7/5</p>
                <p className="text-xs text-gray-500">CSAT 점수</p>
              </div>

              <div className="p-2 bg-orange-50 dark:bg-orange-900/20 rounded text-center">
                <strong className="text-orange-800 dark:text-orange-300 text-xs">비용 절감</strong>
                <p className="text-lg font-bold text-orange-600 dark:text-orange-400">75%</p>
                <p className="text-xs text-gray-500">인건비</p>
              </div>
            </div>
          </div>
        </div>

        {/* 사례 3: 데이터 분석 파이프라인 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-5">
          <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3 text-lg">
            📊 사례 3: 실시간 비즈니스 인텔리전스 Crew
          </h4>
          <div className="space-y-3 text-sm">
            <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
              <strong className="text-green-800 dark:text-green-300">요구사항:</strong>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                대규모 데이터셋 분석 → 인사이트 추출 → 경영진 리포트 자동 생성
              </p>
            </div>

            <div className="ml-4 space-y-2">
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-blue-700 dark:text-blue-300">Data Engineer Agent</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 다중 소스 데이터 수집 (API, DB, CSV)</li>
                  <li>• ETL 파이프라인 자동 구축</li>
                  <li>• 데이터 품질 검증 및 클렌징</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-purple-700 dark:text-purple-300">Analyst Agent</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 통계 분석 (회귀, 상관관계, 시계열)</li>
                  <li>• 이상치 탐지 및 패턴 인식</li>
                  <li>• 예측 모델링 (Prophet, ARIMA)</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-green-700 dark:text-green-300">Visualization Agent</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 인터랙티브 대시보드 생성 (Plotly, D3.js)</li>
                  <li>• 자동 차트 타입 선택</li>
                  <li>• 모바일 최적화 리포트</li>
                </ul>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-orange-700 dark:text-orange-300">Insight Agent</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• 비즈니스 인사이트 추출</li>
                  <li>• 액션 아이템 우선순위 제안</li>
                  <li>• 경영진 요약 보고서 작성 (Executive Summary)</li>
                </ul>
              </div>
            </div>

            <div className="mt-3 p-3 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded">
              <strong className="text-gray-800 dark:text-gray-200">효과:</strong>
              <div className="grid md:grid-cols-2 gap-2 mt-2 text-gray-600 dark:text-gray-400">
                <div>• 분석 시간: <strong>5일 → 2시간</strong> (98% 단축)</div>
                <div>• 리포트 생성: <strong>수작업 → 자동</strong></div>
                <div>• 인사이트 정확도: <strong>87% → 94%</strong></div>
                <div>• ROI: <strong>월 $45K 비용 절감</strong></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 프로덕션 배포 전략 */}
      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6 mt-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <DollarSign className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
          💰 프로덕션 배포 및 비용 최적화 전략
        </h3>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">LLM 모델 선택 전략</h4>
            <div className="space-y-2 text-sm">
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-purple-700 dark:text-purple-300">Tier 1: Simple Tasks</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  • GPT-3.5-turbo ($0.0015/1K tokens)<br/>
                  • Claude Haiku ($0.00025/1K tokens)<br/>
                  • 용도: 분류, 요약, 간단한 변환
                </p>
              </div>

              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-blue-700 dark:text-blue-300">Tier 2: Complex Reasoning</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  • GPT-4o ($0.0075/1K tokens)<br/>
                  • Claude Sonnet ($0.003/1K tokens)<br/>
                  • 용도: 분석, 코드 생성, 전문 작업
                </p>
              </div>

              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong className="text-green-700 dark:text-green-300">Tier 3: Expert Tasks</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  • GPT-4 Turbo ($0.03/1K tokens)<br/>
                  • Claude Opus ($0.015/1K tokens)<br/>
                  • 용도: 최고 품질 요구 작업
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">스케일링 및 안정성</h4>
            <div className="space-y-3 text-sm">
              <div>
                <strong className="text-blue-700 dark:text-blue-300">Rate Limiting & Retry Logic</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• Exponential backoff 구현</li>
                  <li>• Token bucket algorithm</li>
                  <li>• Circuit breaker pattern</li>
                </ul>
              </div>

              <div>
                <strong className="text-green-700 dark:text-green-300">Caching Strategy</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• Semantic caching (벡터 유사도)</li>
                  <li>• Redis 활용 응답 캐싱</li>
                  <li>• 평균 30-50% 비용 절감</li>
                </ul>
              </div>

              <div>
                <strong className="text-orange-700 dark:text-orange-300">Monitoring & Logging</strong>
                <ul className="ml-4 mt-1 text-gray-600 dark:text-gray-400">
                  <li>• LangSmith/LangFuse 활용</li>
                  <li>• 토큰 사용량 실시간 추적</li>
                  <li>• Agent 성능 메트릭 수집</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">💡 비용 최적화 Best Practices</h4>
          <div className="grid md:grid-cols-3 gap-3 text-sm text-gray-600 dark:text-gray-400">
            <div>
              <strong className="text-blue-700 dark:text-blue-300">Prompt Engineering</strong>
              <p className="mt-1">• 토큰 효율적인 프롬프트<br/>• Few-shot 예제 최소화<br/>• 컨텍스트 윈도우 관리</p>
            </div>
            <div>
              <strong className="text-green-700 dark:text-green-300">Task Delegation</strong>
              <p className="mt-1">• 복잡도 기반 모델 선택<br/>• Agent별 역할 명확화<br/>• 불필요한 중간 단계 제거</p>
            </div>
            <div>
              <strong className="text-purple-700 dark:text-purple-300">Batch Processing</strong>
              <p className="mt-1">• 비동기 작업 배치 처리<br/>• OpenAI Batch API 활용<br/>• 50% 할인 가능</p>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 코드 예제 */}
      <section className="mt-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          💻 프로덕션 수준 CrewAI 구현
        </h3>
        <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
import redis

# 캐싱 설정
cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 티어별 LLM 설정
llm_tier1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
llm_tier2 = ChatOpenAI(model="gpt-4o", temperature=0.5)

# 콘텐츠 제작 Crew
researcher = Agent(
    role='Senior Market Researcher',
    goal='Find trending topics and keywords in the tech industry',
    backstory='Expert in market research with 10+ years of experience',
    tools=[DuckDuckGoSearchRun()],
    llm=llm_tier1,  # 간단한 검색 → Tier 1
    verbose=True
)

writer = Agent(
    role='Senior Content Writer',
    goal='Create engaging, SEO-optimized blog posts',
    backstory='Award-winning tech blogger with 1M+ monthly readers',
    llm=llm_tier2,  # 복잡한 글쓰기 → Tier 2
    verbose=True
)

editor = Agent(
    role='Chief Editor',
    goal='Ensure content quality and brand consistency',
    backstory='Former editor-in-chief at TechCrunch',
    llm=llm_tier2,
    verbose=True
)

# Task 정의
research_task = Task(
    description='Research trending AI topics for this week',
    expected_output='List of 5 trending topics with keywords',
    agent=researcher
)

write_task = Task(
    description='Write a 1500-word blog post on the top trending topic',
    expected_output='Complete blog post with SEO optimization',
    agent=writer
)

edit_task = Task(
    description='Review and refine the blog post',
    expected_output='Publication-ready blog post',
    agent=editor
)

# Crew 실행 (Sequential Process)
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
    verbose=2
)

# 캐싱을 활용한 실행
cache_key = f"crew_output_{research_task.description}"
cached_result = cache.get(cache_key)

if cached_result:
    result = cached_result
else:
    result = crew.kickoff()
    cache.set(cache_key, result, ex=3600)  # 1시간 캐싱

print(result)`}
        </pre>
      </section>

      <References
        sections={[
          {
            title: 'CrewAI Official & Community',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'CrewAI: Official Documentation',
                description: 'CrewAI 공식 문서 및 API 레퍼런스',
                link: 'https://docs.crewai.com/'
              },
              {
                title: 'CrewAI GitHub Repository',
                description: 'CrewAI 오픈소스 프로젝트 (15K+ stars)',
                link: 'https://github.com/joaomdmoura/crewAI'
              },
              {
                title: 'CrewAI Examples Repository',
                description: '실전 예제 코드 모음',
                link: 'https://github.com/crewAIInc/crewAI-examples'
              },
              {
                title: 'CrewAI Discord Community',
                description: '활발한 개발자 커뮤니티 및 지원',
                link: 'https://discord.com/invite/X4JWnZnxPb'
              }
            ]
          },
          {
            title: 'Production Deployment Guides',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Building Production-Ready AI Agents',
                authors: 'João Moura (CrewAI Creator)',
                year: '2024',
                description: '프로덕션 환경을 위한 AI 에이전트 구축 가이드',
                link: 'https://www.crewai.com/blog/production-ready-ai-agents'
              },
              {
                title: 'LangChain in Production: Best Practices',
                authors: 'LangChain Team',
                year: '2024',
                description: 'LLM 애플리케이션 프로덕션 배포 전략',
                link: 'https://python.langchain.com/docs/guides/productionization/'
              },
              {
                title: 'Multi-Agent Systems for Enterprise',
                authors: 'Andrew Ng, DeepLearning.AI',
                year: '2024',
                description: '엔터프라이즈 multi-agent 시스템 설계',
                link: 'https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/'
              }
            ]
          },
          {
            title: 'Cost Optimization & Monitoring',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'LangSmith: LLM Monitoring Platform',
                description: 'LLM 애플리케이션 모니터링 및 디버깅',
                link: 'https://smith.langchain.com/'
              },
              {
                title: 'LangFuse: Open-Source LLM Observability',
                description: '오픈소스 LLM 관측성 플랫폼',
                link: 'https://langfuse.com/'
              },
              {
                title: 'Helicone: LLM Observability & Cost Management',
                description: 'LLM API 사용량 추적 및 비용 최적화',
                link: 'https://www.helicone.ai/'
              },
              {
                title: 'OpenAI Batch API: 50% Cost Reduction',
                description: 'Batch 처리를 통한 비용 절감',
                link: 'https://platform.openai.com/docs/guides/batch'
              }
            ]
          },
          {
            title: 'Enterprise Case Studies',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'How Klarna Uses AI Agents',
                description: 'Klarna의 AI 에이전트 활용 사례 (90% 고객 문의 자동화)',
                link: 'https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats/'
              },
              {
                title: 'AI Agents in Financial Services',
                description: '금융 서비스에서의 multi-agent 시스템 활용',
                link: 'https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai'
              },
              {
                title: 'Content Automation at Scale: HubSpot',
                description: 'HubSpot의 AI 기반 콘텐츠 자동화',
                link: 'https://www.hubspot.com/artificial-intelligence'
              },
              {
                title: 'Multi-Agent Customer Support Systems',
                description: '대규모 고객 지원을 위한 agent 시스템',
                link: 'https://www.intercom.com/blog/ai-agents/'
              }
            ]
          }
        ]}
      />
    </>
  );
}
