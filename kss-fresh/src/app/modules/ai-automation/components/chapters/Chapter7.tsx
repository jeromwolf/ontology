'use client'

import { Workflow, Zap, GitBranch } from 'lucide-react'

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          AI 워크플로우 자동화
        </h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            No-code/Low-code 플랫폼을 활용하여 복잡한 AI 워크플로우를 시각적으로 
            설계하고 자동화합니다. Make, Zapier, n8n 등을 통해 다양한 AI 서비스를 
            연결하고 오케스트레이션합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔄 주요 자동화 플랫폼
        </h3>
        
        <div className="grid md:grid-cols-3 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center mb-3">
              <Workflow className="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Make (Integromat)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              비주얼 워크플로우 빌더, 1000+ 앱 연동
            </p>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">복잡한 분기 처리</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">데이터 변환 도구</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">에러 핸들링</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center mb-3">
              <Zap className="w-8 h-8 text-orange-600 dark:text-orange-400" />
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Zapier</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              가장 많은 앱 지원, 간단한 자동화
            </p>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">5000+ 앱 연동</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">간단한 설정</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">즉시 실행</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="w-12 h-12 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center mb-3">
              <GitBranch className="w-8 h-8 text-red-600 dark:text-red-400" />
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">n8n</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              오픈소스, 셀프호스팅 가능
            </p>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">코드 노드 지원</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">무제한 실행</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span className="text-gray-700 dark:text-gray-300">커스텀 노드</span>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎯 실전 AI 워크플로우 예시
        </h3>
        
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              📝 콘텐츠 생성 파이프라인
            </h4>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <ol className="space-y-3 text-sm">
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-violet-500 text-white rounded-full flex items-center justify-center text-xs">1</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">RSS/웹 스크래핑</span>
                    <p className="text-xs text-gray-600 dark:text-gray-400">최신 뉴스/트렌드 수집</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-violet-500 text-white rounded-full flex items-center justify-center text-xs">2</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">GPT-4 요약</span>
                    <p className="text-xs text-gray-600 dark:text-gray-400">핵심 내용 추출 및 요약</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-violet-500 text-white rounded-full flex items-center justify-center text-xs">3</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">Claude 리라이팅</span>
                    <p className="text-xs text-gray-600 dark:text-gray-400">톤앤매너 조정, SEO 최적화</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-violet-500 text-white rounded-full flex items-center justify-center text-xs">4</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">DALL-E 3 이미지</span>
                    <p className="text-xs text-gray-600 dark:text-gray-400">썸네일 자동 생성</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-violet-500 text-white rounded-full flex items-center justify-center text-xs">5</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">WordPress 게시</span>
                    <p className="text-xs text-gray-600 dark:text-gray-400">자동 포스팅 및 스케줄링</p>
                  </div>
                </li>
              </ol>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              🤖 고객 지원 자동화
            </h4>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="space-y-3 text-sm">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">이메일/Slack 메시지 수신</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">감정 분석 (Sentiment Analysis)</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">카테고리 자동 분류</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">AI 답변 생성 또는 담당자 할당</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="text-gray-700 dark:text-gray-300">CRM 업데이트 및 팔로우업</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}