'use client'

import { Workflow, Zap, Brain, Settings } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Windsurf와 Cascade
        </h2>
        
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Windsurf는 Codeium이 개발한 차세대 AI IDE로, Cascade 플로우 모드를 통해
            복잡한 멀티파일 편집을 자연스럽게 처리합니다. AI가 코드의 흐름을 이해하고
            전체 프로젝트 차원에서 일관된 변경을 수행합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🌊 Cascade 플로우 모드
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Cascade는 단순한 코드 생성을 넘어, 전체 작업 흐름을 이해하고 실행하는 AI 모드입니다.
          </p>
          
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-lg p-4">
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">작동 원리</h4>
              <ol className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>1️⃣ 작업 의도 파악: 자연어로 설명한 목표 이해</li>
                <li>2️⃣ 영향 범위 분석: 수정이 필요한 모든 파일 식별</li>
                <li>3️⃣ 순차적 실행: 의존성을 고려한 단계별 수정</li>
                <li>4️⃣ 일관성 검증: 변경사항의 전체적 일관성 확인</li>
                <li>5️⃣ 자동 테스트: 변경 후 테스트 실행 및 수정</li>
              </ol>
            </div>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <Workflow className="w-8 h-8 text-green-600 dark:text-green-400 mb-2" />
                <h5 className="font-bold text-gray-900 dark:text-white mb-1">멀티파일 리팩토링</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  연관된 모든 파일을 동시에 수정
                </p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <Zap className="w-8 h-8 text-yellow-600 dark:text-yellow-400 mb-2" />
                <h5 className="font-bold text-gray-900 dark:text-white mb-1">스마트 임포트</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  필요한 의존성 자동 추가 및 정리
                </p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <Brain className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-2" />
                <h5 className="font-bold text-gray-900 dark:text-white mb-1">컨텍스트 유지</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  프로젝트 전체 맥락 이해
                </p>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <Settings className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2" />
                <h5 className="font-bold text-gray-900 dark:text-white mb-1">자동 최적화</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  코드 품질 자동 개선
                </p>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          ⚡ Supercomplete 기능
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Supercomplete는 단순 자동완성을 넘어 전체 로직을 이해하고 구현합니다.
          </p>
          
          <div className="space-y-4">
            <div className="bg-gray-900 rounded-lg p-4">
              <pre className="text-green-400 font-mono text-sm overflow-x-auto">
{`// 입력: 함수 시그니처만 작성
async function fetchUserWithPosts(userId: string) {

// Supercomplete가 자동 생성:
async function fetchUserWithPosts(userId: string) {
  try {
    // 사용자 정보 가져오기
    const userResponse = await fetch(\`/api/users/\${userId}\`);
    if (!userResponse.ok) {
      throw new Error(\`Failed to fetch user: \${userResponse.status}\`);
    }
    const user = await userResponse.json();
    
    // 사용자의 포스트 가져오기
    const postsResponse = await fetch(\`/api/users/\${userId}/posts\`);
    if (!postsResponse.ok) {
      throw new Error(\`Failed to fetch posts: \${postsResponse.status}\`);
    }
    const posts = await postsResponse.json();
    
    // 각 포스트의 댓글 수 가져오기
    const postsWithComments = await Promise.all(
      posts.map(async (post) => {
        const commentsResponse = await fetch(\`/api/posts/\${post.id}/comments/count\`);
        const { count } = await commentsResponse.json();
        return { ...post, commentCount: count };
      })
    );
    
    return {
      ...user,
      posts: postsWithComments,
      postCount: postsWithComments.length
    };
  } catch (error) {
    console.error('Error fetching user with posts:', error);
    throw error;
  }
}`}</pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎯 Command 모드 활용
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-4">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">주요 명령어</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                  <code className="text-sm text-gray-700 dark:text-gray-300">@generate</code>
                  <span className="text-xs text-gray-500">전체 기능 생성</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                  <code className="text-sm text-gray-700 dark:text-gray-300">@refactor</code>
                  <span className="text-xs text-gray-500">코드 리팩토링</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                  <code className="text-sm text-gray-700 dark:text-gray-300">@test</code>
                  <span className="text-xs text-gray-500">테스트 코드 생성</span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                  <code className="text-sm text-gray-700 dark:text-gray-300">@optimize</code>
                  <span className="text-xs text-gray-500">성능 최적화</span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <code className="text-sm text-gray-700 dark:text-gray-300">@explain</code>
                  <span className="text-xs text-gray-500">코드 설명</span>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">실전 예시</h4>
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`// 전체 CRUD API 생성
@generate "User CRUD API with validation and error handling"

// 성능 최적화
@optimize "Reduce re-renders in ProductList component"

// 테스트 생성
@test "Add integration tests for checkout flow"`}</pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔥 Windsurf vs 다른 도구
        </h3>
        
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-900">
                <th className="border border-gray-200 dark:border-gray-700 p-3 text-left">기능</th>
                <th className="border border-gray-200 dark:border-gray-700 p-3 text-center">Windsurf</th>
                <th className="border border-gray-200 dark:border-gray-700 p-3 text-center">Cursor</th>
                <th className="border border-gray-200 dark:border-gray-700 p-3 text-center">Copilot</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-200 dark:border-gray-700 p-3">멀티파일 편집</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-green-600">✅ Cascade</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-yellow-600">⚡ Composer</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-red-600">❌</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-900/50">
                <td className="border border-gray-200 dark:border-gray-700 p-3">플로우 이해</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-green-600">✅</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-yellow-600">부분적</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-red-600">❌</td>
              </tr>
              <tr>
                <td className="border border-gray-200 dark:border-gray-700 p-3">속도</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-green-600">매우 빠름</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-yellow-600">빠름</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center text-yellow-600">빠름</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-900/50">
                <td className="border border-gray-200 dark:border-gray-700 p-3">가격</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center">무료/$20</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center">$20</td>
                <td className="border border-gray-200 dark:border-gray-700 p-3 text-center">$10</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          💡 실전 팁
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-2">Cascade 최적화</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              작업을 명확한 단계로 나누어 설명하면 더 정확한 결과
            </p>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">컨텍스트 관리</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              .windsurfignore로 불필요한 파일 제외
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}