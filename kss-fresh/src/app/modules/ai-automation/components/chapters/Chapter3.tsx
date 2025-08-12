'use client'

import { Code2, Brain } from 'lucide-react'

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Gemini CLI & AI Studio
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Google의 Gemini는 최신 멀티모달 AI 모델로, CLI 도구와 AI Studio를 통해
            강력한 개발 경험을 제공합니다. 이미지, 비디오, 오디오를 포함한 다양한 형태의
            입력을 처리할 수 있으며, Function Calling과 Grounding으로 실제 애플리케이션에 통합할 수 있습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚀 Gemini CLI 설치 및 설정
        </h3>
        
        <div className="bg-gray-900 rounded-lg p-6 mb-6">
          <pre className="text-green-400 font-mono text-sm overflow-x-auto">
{`# npm을 통한 설치
npm install -g @google/generative-ai-cli

# 또는 Python pip
pip install google-generativeai-cli

# API 키 설정
export GOOGLE_API_KEY="your-api-key"

# 또는 gcloud를 통한 인증
gcloud auth application-default login

# Gemini CLI 초기화
gemini init

# 프로젝트 설정
gemini config set project-id YOUR_PROJECT_ID`}</pre>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎯 주요 CLI 명령어
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">기본 명령어</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">텍스트 생성</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">gemini generate</kbd>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">이미지 분석</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">gemini vision</kbd>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">코드 생성</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">gemini code</kbd>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-700 dark:text-gray-300">대화형 세션</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">gemini chat</kbd>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">고급 기능</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">Function Calling</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">gemini function</kbd>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">파일 업로드</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">gemini upload</kbd>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">임베딩 생성</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">gemini embed</kbd>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-700 dark:text-gray-300">모델 튜닝</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">gemini tune</kbd>
                </div>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🌟 멀티모달 처리 능력
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              <Code2 className="inline w-5 h-5 mr-2" />
              이미지 & 비디오 분석
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 이미지에서 텍스트 추출 (OCR)</li>
              <li>• 비디오 내용 요약 및 분석</li>
              <li>• 다이어그램과 차트 해석</li>
              <li>• 스크린샷 기반 코드 생성</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              <Brain className="inline w-5 h-5 mr-2" />
              오디오 & 문서 처리
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 음성 파일 텍스트 변환</li>
              <li>• PDF 문서 전체 분석</li>
              <li>• 대용량 파일 처리 (최대 2GB)</li>
              <li>• 다국어 실시간 번역</li>
            </ul>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎨 AI Studio 활용
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">프롬프트 테스트 및 최적화</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-3">
                AI Studio에서 다양한 프롬프트를 테스트하고 최적의 결과를 찾아냅니다
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 예시: 코드 리뷰 프롬프트
gemini generate \\
  --prompt "Review this code for security vulnerabilities" \\
  --file ./src/api/auth.js \\
  --model gemini-2.0-flash \\
  --temperature 0.2`}</pre>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">Function Calling 구현</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-3">
                외부 API와 연동하여 실시간 데이터 처리
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# Function 정의 및 실행
gemini function create \\
  --name "get_weather" \\
  --description "Get current weather for a location" \\
  --parameters '{"location": "string", "unit": "celsius|fahrenheit"}'

# Function과 함께 프롬프트 실행
gemini chat --functions weather_api.json`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          ⚙️ Grounding & 실시간 검색
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Gemini의 Grounding 기능으로 실시간 웹 정보와 Google 검색 결과를 활용할 수 있습니다.
          </p>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs whitespace-nowrap">
{`# Grounding 활성화 예시
gemini generate \\
  --prompt "최신 React 19 기능을 활용한 컴포넌트 작성" \\
  --grounding-source "google-search" \\
  --grounding-threshold 0.7

# 특정 웹사이트 참조  
gemini generate \\
  --prompt "이 라이브러리의 최신 버전 문법으로 코드 작성" \\
  --grounding-urls "https://docs.library.com" \\
  --model gemini-2.0-pro

# 프로젝트 컨텍스트 파일 설정 (.gemini-context.yaml)
context:
  project_type: "Next.js 14 App"
  language: "TypeScript"
  styling: "Tailwind CSS"
  database: "PostgreSQL with Prisma"
  
rules:
  - "Always use App Router patterns"
  - "Implement proper error boundaries"
  - "Use server components by default"
  
grounding:
  enabled: true
  sources:
    - "google-search"
    - "github"
    - "stackoverflow"`}</pre>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🎯 Gemini 활용 실전 팁
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">
              멀티모달 최적화
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              이미지와 코드를 함께 입력하여 UI 구현
            </p>
          </div>
          
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-4">
            <h4 className="font-bold text-indigo-700 dark:text-indigo-400 mb-2">
              컨텍스트 윈도우 활용
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              2M 토큰까지 한 번에 처리 가능
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-bold text-purple-700 dark:text-purple-400 mb-2">
              모델 선택 가이드
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Flash: 빠른 응답, Pro: 복잡한 추론
            </p>
          </div>
          
          <div className="bg-pink-50 dark:bg-pink-900/20 rounded-lg p-4">
            <h4 className="font-bold text-pink-700 dark:text-pink-400 mb-2">
              API 비용 최적화
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              캐싱과 배치 처리로 비용 절감
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}