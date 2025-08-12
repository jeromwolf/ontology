'use client'

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          AI 자동화 시대의 도래
        </h2>
        
        <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            2024년, 우리는 AI가 단순한 도구를 넘어 개발자의 진정한 파트너가 되는 시대를 맞이했습니다.
            Claude Code, Cursor, Windsurf 같은 혁신적인 도구들이 등장하며, 
            개발 생산성은 문자 그대로 10배 이상 향상되고 있습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚀 패러다임의 변화
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">Before AI (2020)</h4>
            <ul className="space-y-2 text-gray-600 dark:text-gray-400">
              <li>• 수동 코드 작성: 100%</li>
              <li>• 디버깅 시간: 전체의 40%</li>
              <li>• 보일러플레이트: 반복 작성</li>
              <li>• 리팩토링: 수일 소요</li>
              <li>• 문서화: 종종 누락</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-lg p-6 border border-violet-200 dark:border-violet-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">With AI (2024)</h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>✨ AI 제안 코드: 70-80%</li>
              <li>✨ 즉각적 버그 감지</li>
              <li>✨ 자동 생성 템플릿</li>
              <li>✨ 실시간 리팩토링</li>
              <li>✨ 자동 문서화</li>
            </ul>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          📊 실제 생산성 향상 사례
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-700 dark:text-gray-300">CRUD API 개발</span>
              <div className="flex items-center gap-4">
                <span className="text-gray-500">2시간 → </span>
                <span className="font-bold text-violet-600 dark:text-violet-400">10분</span>
                <span className="text-green-600 dark:text-green-400 text-sm">12x 향상</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-700 dark:text-gray-300">React 컴포넌트 생성</span>
              <div className="flex items-center gap-4">
                <span className="text-gray-500">30분 → </span>
                <span className="font-bold text-violet-600 dark:text-violet-400">2분</span>
                <span className="text-green-600 dark:text-green-400 text-sm">15x 향상</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-700 dark:text-gray-300">테스트 코드 작성</span>
              <div className="flex items-center gap-4">
                <span className="text-gray-500">1시간 → </span>
                <span className="font-bold text-violet-600 dark:text-violet-400">5분</span>
                <span className="text-green-600 dark:text-green-400 text-sm">12x 향상</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-700 dark:text-gray-300">버그 수정</span>
              <div className="flex items-center gap-4">
                <span className="text-gray-500">45분 → </span>
                <span className="font-bold text-violet-600 dark:text-violet-400">5분</span>
                <span className="text-green-600 dark:text-green-400 text-sm">9x 향상</span>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎯 AI 도구 선택 가이드
        </h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-l-4 border-blue-500">
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">
              Claude Code - 대규모 프로젝트 자동화
            </h4>
            <p className="text-gray-600 dark:text-gray-400 mb-2">
              전체 프로젝트 컨텍스트를 이해하고 복잡한 리팩토링이나 기능 구현이 필요할 때
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-sm">
                MCP 지원
              </span>
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-sm">
                CLAUDE.md
              </span>
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-sm">
                CLI 기반
              </span>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-l-4 border-purple-500">
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">
              Cursor - 실시간 코딩 파트너
            </h4>
            <p className="text-gray-600 dark:text-gray-400 mb-2">
              IDE 내에서 즉각적인 코드 제안과 수정이 필요한 일상적인 개발 작업
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded-full text-sm">
                Copilot++
              </span>
              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded-full text-sm">
                Chat 모드
              </span>
              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded-full text-sm">
                Composer
              </span>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-l-4 border-green-500">
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">
              Windsurf - 플로우 기반 개발
            </h4>
            <p className="text-gray-600 dark:text-gray-400 mb-2">
              여러 파일을 동시에 수정하며 전체적인 코드 플로우를 관리해야 할 때
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-sm">
                Cascade
              </span>
              <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-sm">
                Multi-file
              </span>
              <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-sm">
                Supercomplete
              </span>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          💡 핵심 인사이트
        </h3>
        
        <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <ul className="space-y-3">
            <li className="flex items-start gap-2">
              <span className="text-violet-600 dark:text-violet-400 mt-1">1.</span>
              <span className="text-gray-700 dark:text-gray-300">
                AI 도구는 개발자를 대체하는 것이 아니라 증강시킵니다
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-violet-600 dark:text-violet-400 mt-1">2.</span>
              <span className="text-gray-700 dark:text-gray-300">
                각 도구의 강점을 이해하고 상황에 맞게 선택하는 것이 중요합니다
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-violet-600 dark:text-violet-400 mt-1">3.</span>
              <span className="text-gray-700 dark:text-gray-300">
                프롬프트 엔지니어링 능력이 새로운 핵심 역량이 되었습니다
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-violet-600 dark:text-violet-400 mt-1">4.</span>
              <span className="text-gray-700 dark:text-gray-300">
                AI와의 협업 워크플로우를 구축하는 것이 경쟁력의 핵심입니다
              </span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}