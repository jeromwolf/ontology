'use client';

import { Code2, Brain } from 'lucide-react';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Cursor IDE 마스터하기
        </h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Cursor는 AI-First 철학으로 만들어진 IDE로, VS Code를 기반으로 하면서도
            AI 기능을 핵심에 둔 혁신적인 개발 환경입니다. Copilot++와 Chat 기능으로
            코딩 속도를 극적으로 향상시킬 수 있습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          ⚡ 핵심 기능과 단축키
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">필수 단축키</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">AI Chat 열기</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">Cmd+K</kbd>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">Composer 모드</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">Cmd+I</kbd>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">코드 생성</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">Cmd+Shift+K</kbd>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-700 dark:text-gray-300">AI 수정 제안</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">Tab</kbd>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">고급 기능</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">Codebase 검색</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">@codebase</kbd>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">웹 검색</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">@web</kbd>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                  <span className="text-gray-700 dark:text-gray-300">문서 참조</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">@docs</kbd>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-700 dark:text-gray-300">Git 정보</span>
                  <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">@git</kbd>
                </div>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚀 Copilot++ 활용법
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              <Code2 className="inline w-5 h-5 mr-2" />
              자동 완성 최적화
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 함수 시그니처만 작성하면 전체 구현 제안</li>
              <li>• 주석으로 의도 설명 → 코드 자동 생성</li>
              <li>• 테스트 케이스 자동 생성</li>
              <li>• 에러 메시지 기반 자동 수정</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              <Brain className="inline w-5 h-5 mr-2" />
              컨텍스트 활용
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 열린 파일들을 자동으로 컨텍스트로 사용</li>
              <li>• 최근 수정 내역 참조</li>
              <li>• 프로젝트 구조 이해</li>
              <li>• 의존성 자동 import</li>
            </ul>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💬 Chat & Composer 모드
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">Chat 모드 (Cmd+K)</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-3">
                코드에 대한 질문, 설명 요청, 버그 수정 제안 등 대화형 인터랙션
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm">
{`// 예시 프롬프트
"이 함수의 시간 복잡도를 O(n)으로 최적화해줘"
"이 컴포넌트를 TypeScript로 변환해줘"
"이 코드에 메모리 누수가 있는지 확인해줘"`}</pre>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">Composer 모드 (Cmd+I)</h4>
              <p className="text-gray-600 dark:text-gray-400 mb-3">
                여러 파일을 동시에 수정하는 대규모 변경 작업
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm">
{`// 예시 프롬프트
"모든 API 엔드포인트에 rate limiting 추가"
"전체 프로젝트를 Tailwind CSS로 마이그레이션"
"모든 클래스 컴포넌트를 함수형으로 변환"`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          ⚙️ 커스텀 Rules 설정
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            .cursorrules 파일로 프로젝트별 AI 동작을 커스터마이징할 수 있습니다.
          </p>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# .cursorrules 예시

You are an expert in React, Next.js, and TypeScript.

## Code Style
- Use functional components with hooks
- Prefer const over let
- Use optional chaining and nullish coalescing
- Always use TypeScript strict mode

## Naming Conventions
- Components: PascalCase
- Functions: camelCase
- Constants: UPPER_SNAKE_CASE
- Files: kebab-case

## Best Practices
- Implement error boundaries for all pages
- Use React.memo for expensive components
- Prefer composition over inheritance
- Always handle loading and error states

## Forbidden
- Never use var
- Avoid any type unless absolutely necessary
- Don't use inline styles`}</pre>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🎯 실전 팁
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-bold text-purple-700 dark:text-purple-400 mb-2">
              대용량 파일 처리
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              10,000줄 이상 파일은 부분 선택 후 처리
            </p>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">
              멀티 커서 활용
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Cmd+D로 동일 단어 선택 후 AI 수정
            </p>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-2">
              컨텍스트 최적화
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              관련 파일만 열어두고 작업
            </p>
          </div>
          
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
            <h4 className="font-bold text-orange-700 dark:text-orange-400 mb-2">
              히스토리 활용
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              이전 대화 참조로 일관성 유지
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}