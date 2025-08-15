'use client';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          GitHub Copilot 고급 활용
        </h2>
        
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            GitHub Copilot은 가장 널리 사용되는 AI 코딩 도구로, 최근 Copilot X와 
            Workspace 기능이 추가되며 단순 자동완성을 넘어 전체 개발 워크플로우를 
            지원하는 플랫폼으로 진화했습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚀 Copilot Workspace
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            이슈에서 PR까지 전체 개발 프로세스를 AI가 관리합니다.
          </p>
          
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-4">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">워크플로우</h4>
              <ol className="space-y-3">
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs">1</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">이슈 분석</span>
                    <p className="text-sm text-gray-600 dark:text-gray-400">GitHub 이슈를 읽고 요구사항 파악</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs">2</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">계획 수립</span>
                    <p className="text-sm text-gray-600 dark:text-gray-400">구현 계획과 파일 목록 생성</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs">3</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">코드 생성</span>
                    <p className="text-sm text-gray-600 dark:text-gray-400">계획에 따라 코드 작성</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs">4</span>
                  <div>
                    <span className="font-semibold text-gray-900 dark:text-white">PR 생성</span>
                    <p className="text-sm text-gray-600 dark:text-gray-400">커밋 메시지와 PR 설명 자동 작성</p>
                  </div>
                </li>
              </ol>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💬 Copilot Chat 고급 기능
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">슬래시 명령어</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between py-1 border-b border-gray-200 dark:border-gray-700">
                <code className="text-orange-600 dark:text-orange-400">/explain</code>
                <span className="text-gray-600 dark:text-gray-400">코드 설명</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-200 dark:border-gray-700">
                <code className="text-orange-600 dark:text-orange-400">/fix</code>
                <span className="text-gray-600 dark:text-gray-400">버그 수정</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-200 dark:border-gray-700">
                <code className="text-orange-600 dark:text-orange-400">/tests</code>
                <span className="text-gray-600 dark:text-gray-400">테스트 생성</span>
              </div>
              <div className="flex justify-between py-1">
                <code className="text-orange-600 dark:text-orange-400">/docs</code>
                <span className="text-gray-600 dark:text-gray-400">문서 생성</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">컨텍스트 변수</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between py-1 border-b border-gray-200 dark:border-gray-700">
                <code className="text-blue-600 dark:text-blue-400">#file</code>
                <span className="text-gray-600 dark:text-gray-400">특정 파일 참조</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-200 dark:border-gray-700">
                <code className="text-blue-600 dark:text-blue-400">#selection</code>
                <span className="text-gray-600 dark:text-gray-400">선택 영역</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-200 dark:border-gray-700">
                <code className="text-blue-600 dark:text-blue-400">#editor</code>
                <span className="text-gray-600 dark:text-gray-400">현재 에디터</span>
              </div>
              <div className="flex justify-between py-1">
                <code className="text-blue-600 dark:text-blue-400">#terminal</code>
                <span className="text-gray-600 dark:text-gray-400">터미널 출력</span>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🤖 Custom Instructions 설정
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`// .github/copilot-instructions.md

## Project Context
- Next.js 14 App Router 사용
- TypeScript strict mode
- Tailwind CSS for styling
- PostgreSQL with Prisma ORM

## Code Style
- 함수형 컴포넌트 사용
- Custom hooks for business logic
- Error boundaries on all pages
- Comprehensive error handling

## Testing
- Jest + React Testing Library
- Minimum 80% coverage
- E2E tests with Playwright

## Documentation
- JSDoc for all public APIs
- README for each module
- Inline comments for complex logic`}</pre>
          </div>
        </div>
      </section>
    </div>
  )
}