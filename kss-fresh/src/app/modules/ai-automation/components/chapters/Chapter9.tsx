'use client';

import References from '@/components/common/References';

export default function Chapter9() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          미래를 위한 준비
        </h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            AI 도구는 빠르게 진화하고 있습니다. 새로운 도구를 평가하고, 
            지속적으로 학습하며, AI 시대에 필요한 핵심 역량을 개발하는 전략을 수립합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔮 AI 도구의 진화 방향
        </h3>
        
        <div className="space-y-4 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-l-4 border-indigo-500">
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">2024-2025 트렌드</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>🎯 더 큰 컨텍스트 윈도우 (100만+ 토큰)</li>
                <li>🎯 멀티모달 AI (코드 + 이미지 + 음성)</li>
                <li>🎯 실시간 협업 AI</li>
                <li>🎯 자율 에이전트 시스템</li>
              </ul>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>🎯 IDE 완전 통합</li>
                <li>🎯 프로젝트 수준 이해</li>
                <li>🎯 자동 최적화 및 리팩토링</li>
                <li>🎯 AI 간 협업 프로토콜</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-l-4 border-purple-500">
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">새로운 AI 도구 평가 기준</h4>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
                  <span className="text-purple-600 dark:text-purple-400 font-bold">1</span>
                </div>
                <h5 className="font-semibold text-gray-900 dark:text-white text-sm mb-1">성능</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">속도, 정확도, 컨텍스트 이해</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
                  <span className="text-purple-600 dark:text-purple-400 font-bold">2</span>
                </div>
                <h5 className="font-semibold text-gray-900 dark:text-white text-sm mb-1">통합성</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">기존 워크플로우 호환</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
                  <span className="text-purple-600 dark:text-purple-400 font-bold">3</span>
                </div>
                <h5 className="font-semibold text-gray-900 dark:text-white text-sm mb-1">ROI</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">비용 대비 생산성 향상</p>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💪 AI 시대의 핵심 역량
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">기술적 역량</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
                <span className="text-gray-700 dark:text-gray-300">프롬프트 엔지니어링</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
                <span className="text-gray-700 dark:text-gray-300">AI 도구 선택 및 조합</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
                <span className="text-gray-700 dark:text-gray-300">컨텍스트 관리</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 dark:text-blue-400 mt-0.5">✓</span>
                <span className="text-gray-700 dark:text-gray-300">AI 출력 검증 및 개선</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">소프트 스킬</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-purple-600 dark:text-purple-400 mt-0.5">✓</span>
                <span className="text-gray-700 dark:text-gray-300">문제 정의 능력</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-600 dark:text-purple-400 mt-0.5">✓</span>
                <span className="text-gray-700 dark:text-gray-300">비판적 사고</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-600 dark:text-purple-400 mt-0.5">✓</span>
                <span className="text-gray-700 dark:text-gray-300">창의적 문제 해결</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-600 dark:text-purple-400 mt-0.5">✓</span>
                <span className="text-gray-700 dark:text-gray-300">지속적 학습 마인드셋</span>
              </li>
            </ul>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          📚 지속적 학습 전략
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-green-600 dark:text-green-400 font-bold text-sm">1</span>
              </div>
              <div>
                <h5 className="font-semibold text-gray-900 dark:text-white mb-1">주간 AI 도구 탐색</h5>
                <p className="text-sm text-gray-600 dark:text-gray-400">매주 새로운 도구 1개씩 테스트</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-green-600 dark:text-green-400 font-bold text-sm">2</span>
              </div>
              <div>
                <h5 className="font-semibold text-gray-900 dark:text-white mb-1">커뮤니티 참여</h5>
                <p className="text-sm text-gray-600 dark:text-gray-400">Discord, Reddit, Twitter에서 최신 정보 수집</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-green-600 dark:text-green-400 font-bold text-sm">3</span>
              </div>
              <div>
                <h5 className="font-semibold text-gray-900 dark:text-white mb-1">실전 프로젝트</h5>
                <p className="text-sm text-gray-600 dark:text-gray-400">AI 도구로 실제 프로젝트 완성</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-green-600 dark:text-green-400 font-bold text-sm">4</span>
              </div>
              <div>
                <h5 className="font-semibold text-gray-900 dark:text-white mb-1">지식 공유</h5>
                <p className="text-sm text-gray-600 dark:text-gray-400">블로그, 영상으로 경험 공유</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🎯 핵심 메시지
        </h3>

        <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 text-center text-lg font-medium">
            "AI는 개발자를 대체하지 않습니다.<br/>
            하지만 AI를 활용하는 개발자가<br/>
            그렇지 않은 개발자를 대체할 것입니다."
          </p>
        </div>
      </section>

      {/* References */}
      <section className="mt-12">
        <References
          sections={[
            {
              title: '📚 자동화 플랫폼',
              icon: 'web' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'Make.com (Integromat)',
                  authors: 'Make',
                  year: '2024',
                  description: '노코드 자동화 플랫폼 - 1,500+ 앱 통합, 시각적 워크플로우 빌더',
                  link: 'https://www.make.com/'
                },
                {
                  title: 'Zapier',
                  authors: 'Zapier Inc.',
                  year: '2024',
                  description: '업무 자동화 플랫폼 - 7,000+ 앱 연동, AI 기반 워크플로우',
                  link: 'https://zapier.com/'
                },
                {
                  title: 'n8n',
                  authors: 'n8n.io',
                  year: '2024',
                  description: '오픈소스 워크플로우 자동화 - 400+ 노드, 커스터마이징 가능',
                  link: 'https://n8n.io/'
                },
                {
                  title: 'Microsoft Power Automate',
                  authors: 'Microsoft',
                  year: '2024',
                  description: 'RPA + 클라우드 자동화 - Office 365 통합, AI Builder',
                  link: 'https://powerautomate.microsoft.com/'
                },
                {
                  title: 'UiPath',
                  authors: 'UiPath',
                  year: '2024',
                  description: '엔터프라이즈 RPA 플랫폼 - 문서 자동화, AI Fabric',
                  link: 'https://www.uipath.com/'
                }
              ]
            },
            {
              title: '📖 핵심 리소스',
              icon: 'research' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'The State of RPA 2024',
                  authors: 'Forrester Research',
                  year: '2024',
                  description: 'RPA 시장 동향 보고서 - AI 통합 및 Intelligent Automation',
                  link: 'https://www.forrester.com/'
                },
                {
                  title: 'AI Automation Trends Report',
                  authors: 'Gartner',
                  year: '2024',
                  description: '2024 AI 자동화 트렌드 - Agentic AI, Multi-agent Systems',
                  link: 'https://www.gartner.com/en/insights/automation'
                },
                {
                  title: 'Hyperautomation Guide',
                  authors: 'Gartner',
                  year: '2024',
                  description: '초자동화 전략 가이드 - RPA + AI + Low-code 통합',
                  link: 'https://www.gartner.com/en/information-technology/insights/hyperautomation'
                },
                {
                  title: 'The Future of Work Automation',
                  authors: 'McKinsey & Company',
                  year: '2024',
                  description: '업무 자동화 미래 전망 - 생산성 향상 및 일자리 변화',
                  link: 'https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-future-of-work-after-covid-19'
                }
              ]
            },
            {
              title: '🛠️ 실전 도구',
              icon: 'tools' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'LangChain Agents',
                  authors: 'LangChain',
                  year: '2024',
                  description: 'AI 에이전트 프레임워크 - 자율적 작업 실행, Tool 통합',
                  link: 'https://python.langchain.com/docs/modules/agents/'
                },
                {
                  title: 'AutoGPT',
                  authors: 'Significant Gravitas',
                  year: '2024',
                  description: '자율 AI 에이전트 - 목표 달성을 위한 자동 작업 분해',
                  link: 'https://github.com/Significant-Gravitas/AutoGPT'
                },
                {
                  title: 'BabyAGI',
                  authors: 'Yohei Nakajima',
                  year: '2023',
                  description: 'Task-driven AI 에이전트 - 작업 생성, 우선순위, 실행',
                  link: 'https://github.com/yoheinakajima/babyagi'
                },
                {
                  title: 'Microsoft Semantic Kernel',
                  authors: 'Microsoft',
                  year: '2024',
                  description: 'LLM 오케스트레이션 SDK - Planner, Plugins, Memory',
                  link: 'https://github.com/microsoft/semantic-kernel'
                },
                {
                  title: 'CrewAI',
                  authors: 'CrewAI',
                  year: '2024',
                  description: 'Multi-agent 협업 프레임워크 - 역할 기반 작업 분담',
                  link: 'https://github.com/joaomdmoura/crewAI'
                }
              ]
            }
          ]}
        />
      </section>
    </div>
  )
}