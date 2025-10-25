'use client'

import React from 'react'
import { Zap, Keyboard, MessageSquare, Sparkles, FileCode, BookOpen } from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-8">
        <div className="inline-block p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl mb-4">
          <Zap className="w-12 h-12 text-purple-500" />
        </div>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
          Cursor 완벽 마스터
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          VS Code를 넘어선 차세대 AI 에디터의 모든 것
        </p>
      </div>

      {/* Introduction */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">Cursor란 무엇인가?</h2>

        <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 rounded-xl p-8 border border-purple-500/20">
          <p className="text-lg text-gray-300 leading-relaxed mb-6">
            <span className="text-purple-400 font-semibold">Cursor</span>는 VS Code를 포크하여
            <span className="text-pink-400 font-semibold"> GPT-4를 네이티브로 통합</span>한 AI 우선 코드 에디터입니다.
            2023년 출시 이후 빠르게 성장하여 2024년 현재 50만+ 개발자가 사용 중입니다.
          </p>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
              <div className="text-3xl font-bold text-purple-400 mb-1">10x</div>
              <div className="text-sm text-gray-400">코딩 속도 향상 (사용자 리포트)</div>
            </div>
            <div className="bg-pink-500/10 rounded-lg p-4 border border-pink-500/20">
              <div className="text-3xl font-bold text-pink-400 mb-1">200K</div>
              <div className="text-sm text-gray-400">토큰 컨텍스트 윈도우 (GPT-4)</div>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
              <div className="text-3xl font-bold text-purple-400 mb-1">100%</div>
              <div className="text-sm text-gray-400">VS Code 확장 프로그램 호환</div>
            </div>
          </div>
        </div>
      </section>

      {/* Installation */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <FileCode className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">설치 및 초기 설정</h2>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">1. 다운로드 및 설치</h3>
            <div className="bg-black/30 rounded-lg p-4 space-y-3">
              <div className="text-gray-300">
                <strong className="text-purple-400">공식 사이트:</strong> https://cursor.sh
              </div>
              <div className="text-gray-400 text-sm">
                • macOS: .dmg 파일 다운로드 → Applications 폴더로 드래그<br/>
                • Windows: .exe 인스톨러 실행<br/>
                • Linux: .AppImage 또는 .deb 패키지
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-xl font-semibold text-pink-400 mb-3">2. API 키 설정</h3>
            <p className="text-gray-300 mb-4">
              Cursor는 자체 API 크레딧을 제공하지만, 무제한 사용을 위해 OpenAI API 키를 설정할 수 있습니다.
            </p>
            <div className="bg-black/30 rounded-lg p-4 font-mono text-sm space-y-2">
              <div className="text-gray-500">// Settings (Cmd/Ctrl + ,) → Cursor Settings</div>
              <div className="text-purple-400">API Keys → OpenAI API Key</div>
              <div className="text-gray-400">sk-proj-xxxxxxxxxxxxxxxxxxxxx</div>
              <div className="text-gray-500 mt-3">// 또는 환경 변수로 설정</div>
              <div className="text-pink-400">export OPENAI_API_KEY="sk-proj-xxx"</div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">3. VS Code 설정 마이그레이션</h3>
            <p className="text-gray-300 mb-4">
              기존 VS Code 사용자라면 모든 설정, 테마, 확장 프로그램을 자동으로 가져올 수 있습니다.
            </p>
            <div className="bg-black/30 rounded-lg p-4 space-y-2 text-gray-300">
              <div><strong className="text-purple-400">Step 1:</strong> Cursor 첫 실행 시 "Import VS Code Settings" 선택</div>
              <div><strong className="text-purple-400">Step 2:</strong> 자동으로 ~/.vscode 디렉토리에서 설정 복사</div>
              <div><strong className="text-purple-400">Step 3:</strong> 확장 프로그램 재설치 여부 선택</div>
            </div>
          </div>
        </div>
      </section>

      {/* Core Features */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <Keyboard className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">핵심 기능: Cmd+K (인라인 편집)</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-pink-500/20">
          <p className="text-lg text-gray-300 mb-6">
            Cursor의 가장 강력한 기능은 <span className="text-pink-400 font-semibold">Cmd+K</span> (Windows: Ctrl+K)입니다.
            코드를 선택하고 자연어로 수정 요청하면 <span className="text-pink-400">즉시 인라인으로 변경</span>됩니다.
          </p>

          <div className="space-y-6">
            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-purple-400 font-semibold mb-4">예제 1: 함수 리팩토링</h4>
              <div className="space-y-4">
                <div>
                  <div className="text-gray-500 text-sm mb-2">// Before (선택 후 Cmd+K)</div>
                  <div className="bg-gray-900 rounded p-4 font-mono text-sm text-gray-400">
{`function calc(a, b, op) {
  if (op === '+') return a + b;
  if (op === '-') return a - b;
  if (op === '*') return a * b;
  if (op === '/') return a / b;
}`}
                  </div>
                </div>
                <div className="text-center">
                  <div className="inline-block bg-pink-500/20 rounded-lg px-4 py-2 text-pink-400 font-mono text-sm">
                    💬 "switch 문으로 변경하고 TypeScript로 타입 추가"
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 text-sm mb-2">// After (AI 생성)</div>
                  <div className="bg-gray-900 rounded p-4 font-mono text-sm text-purple-400">
{`function calc(a: number, b: number, op: '+' | '-' | '*' | '/'): number {
  switch (op) {
    case '+':
      return a + b;
    case '-':
      return a - b;
    case '*':
      return a * b;
    case '/':
      if (b === 0) throw new Error('Division by zero');
      return a / b;
    default:
      throw new Error(\`Unknown operator: \${op}\`);
  }
}`}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-purple-400 font-semibold mb-4">예제 2: 비동기 변환</h4>
              <div className="space-y-4">
                <div>
                  <div className="text-gray-500 text-sm mb-2">// Synchronous 코드</div>
                  <div className="bg-gray-900 rounded p-4 font-mono text-sm text-gray-400">
{`function fetchUserData(userId) {
  const response = fetch(\`/api/users/\${userId}\`);
  const data = response.json();
  return data;
}`}
                  </div>
                </div>
                <div className="text-center">
                  <div className="inline-block bg-pink-500/20 rounded-lg px-4 py-2 text-pink-400 font-mono text-sm">
                    💬 "async/await로 변경하고 에러 처리 추가"
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 text-sm mb-2">// Asynchronous 코드</div>
                  <div className="bg-gray-900 rounded p-4 font-mono text-sm text-purple-400">
{`async function fetchUserData(userId: string): Promise<User> {
  try {
    const response = await fetch(\`/api/users/\${userId}\`);
    if (!response.ok) {
      throw new Error(\`HTTP error! status: \${response.status}\`);
    }
    const data: User = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to fetch user data:', error);
    throw error;
  }
}`}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
            <h4 className="text-purple-400 font-semibold mb-2">💡 Cmd+K 활용 팁</h4>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong>선택 범위:</strong> 함수 전체, 클래스, 또는 여러 줄 선택 가능</li>
              <li>• <strong>구체적 요청:</strong> "성능 최적화", "가독성 개선", "에러 처리 추가" 등</li>
              <li>• <strong>다중 수정:</strong> 여러 곳을 선택하고 한 번에 같은 변경 적용</li>
              <li>• <strong>Undo:</strong> Cmd+Z로 AI 변경사항 즉시 되돌리기</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Chat Feature */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <MessageSquare className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">Chat 기능 (Cmd+L)</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-purple-500/20">
          <p className="text-lg text-gray-300 mb-6">
            <span className="text-purple-400 font-semibold">Cmd+L</span>로 열리는 Chat 패널은
            ChatGPT처럼 대화형 인터페이스를 제공하지만, <span className="text-purple-400">코드베이스 전체를 이해</span>합니다.
          </p>

          <div className="space-y-6">
            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-pink-400 font-semibold mb-4">사용 시나리오</h4>
              <div className="space-y-4">
                <div className="border-l-4 border-purple-500 pl-4">
                  <div className="text-purple-400 font-semibold mb-2">1. 코드 설명 요청</div>
                  <div className="bg-gray-900 rounded p-3 font-mono text-sm text-gray-400 mb-2">
                    "이 React 컴포넌트가 어떻게 동작하는지 설명해줘"
                  </div>
                  <div className="text-gray-300 text-sm">
                    → AI가 useState, useEffect, props 등을 분석하여 상세 설명
                  </div>
                </div>

                <div className="border-l-4 border-pink-500 pl-4">
                  <div className="text-pink-400 font-semibold mb-2">2. 디버깅 도움</div>
                  <div className="bg-gray-900 rounded p-3 font-mono text-sm text-gray-400 mb-2">
                    "왜 useEffect가 무한 루프에 빠지는지 찾아줘"
                  </div>
                  <div className="text-gray-300 text-sm">
                    → 의존성 배열 문제를 찾아내고 해결책 제시
                  </div>
                </div>

                <div className="border-l-4 border-purple-500 pl-4">
                  <div className="text-purple-400 font-semibold mb-2">3. 아키텍처 질문</div>
                  <div className="bg-gray-900 rounded p-3 font-mono text-sm text-gray-400 mb-2">
                    "이 프로젝트에서 인증을 어떻게 구현했어?"
                  </div>
                  <div className="text-gray-300 text-sm">
                    → auth 관련 파일들을 분석하여 JWT, 세션, OAuth 등 패턴 설명
                  </div>
                </div>

                <div className="border-l-4 border-pink-500 pl-4">
                  <div className="text-pink-400 font-semibold mb-2">4. 새 기능 구현</div>
                  <div className="bg-gray-900 rounded p-3 font-mono text-sm text-gray-400 mb-2">
                    "결제 기능을 Stripe로 추가하고 싶어. 어디서 시작해야 해?"
                  </div>
                  <div className="text-gray-300 text-sm">
                    → 기존 프로젝트 구조를 고려한 단계별 가이드 제공
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-pink-400 font-semibold mb-4">Chat의 강력한 기능</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
                  <div className="font-semibold text-purple-400 mb-2">📂 @-mentions</div>
                  <div className="text-gray-300 text-sm">
                    @파일명 으로 특정 파일 컨텍스트 추가<br/>
                    예: "@auth.ts의 로직을 설명해줘"
                  </div>
                </div>
                <div className="bg-pink-900/20 rounded-lg p-4 border border-pink-500/20">
                  <div className="font-semibold text-pink-400 mb-2">🔍 Codebase Search</div>
                  <div className="text-gray-300 text-sm">
                    전체 프로젝트에서 관련 코드 자동 검색<br/>
                    "결제 처리는 어디서 해?" → 자동으로 관련 파일 찾기
                  </div>
                </div>
                <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
                  <div className="font-semibold text-purple-400 mb-2">✏️ Apply 버튼</div>
                  <div className="text-gray-300 text-sm">
                    AI가 제안한 코드를 클릭 한 번으로 파일에 적용<br/>
                    diff 뷰로 변경사항 미리보기
                  </div>
                </div>
                <div className="bg-pink-900/20 rounded-lg p-4 border border-pink-500/20">
                  <div className="font-semibold text-pink-400 mb-2">📜 대화 히스토리</div>
                  <div className="text-gray-300 text-sm">
                    이전 대화 내용 참조<br/>
                    컨텍스트 유지하며 연속적인 개발 가능
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Composer Mode */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <Sparkles className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">Composer 모드 (멀티파일 편집)</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-pink-500/20">
          <p className="text-lg text-gray-300 mb-6">
            <span className="text-pink-400 font-semibold">Composer</span>는 Cursor의 최신 기능으로,
            <span className="text-pink-400"> 여러 파일을 동시에 수정</span>할 수 있는 강력한 도구입니다.
            Cmd+Shift+I 또는 Chat 패널에서 "Composer Mode" 활성화.
          </p>

          <div className="space-y-6">
            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-purple-400 font-semibold mb-4">실전 예제: API 엔드포인트 추가</h4>
              <div className="space-y-4">
                <div className="bg-gray-900 rounded-lg p-4">
                  <div className="text-pink-400 font-mono text-sm mb-3">
                    💬 "사용자 프로필 수정 API를 추가해줘. 라우트, 컨트롤러, 서비스 레이어 모두 구현"
                  </div>
                  <div className="space-y-3 text-gray-300 text-sm">
                    <div className="border-l-4 border-purple-500 pl-3">
                      <strong>Step 1:</strong> routes/users.ts에 PUT /users/:id 엔드포인트 추가
                    </div>
                    <div className="border-l-4 border-pink-500 pl-3">
                      <strong>Step 2:</strong> controllers/userController.ts에 updateProfile 메서드 생성
                    </div>
                    <div className="border-l-4 border-purple-500 pl-3">
                      <strong>Step 3:</strong> services/userService.ts에 비즈니스 로직 구현
                    </div>
                    <div className="border-l-4 border-pink-500 pl-3">
                      <strong>Step 4:</strong> validators/userValidator.ts에 유효성 검사 추가
                    </div>
                    <div className="border-l-4 border-purple-500 pl-3">
                      <strong>Step 5:</strong> types/user.ts에 TypeScript 타입 업데이트
                    </div>
                  </div>
                </div>

                <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
                  <div className="font-semibold text-purple-400 mb-2">✨ Composer의 마법</div>
                  <ul className="space-y-2 text-gray-300 text-sm">
                    <li>• <strong>자동 파일 생성:</strong> 없는 파일도 자동으로 생성</li>
                    <li>• <strong>일관성 유지:</strong> 프로젝트 패턴에 맞춰 코드 작성</li>
                    <li>• <strong>Import 자동 추가:</strong> 필요한 모든 import 문 자동 생성</li>
                    <li>• <strong>Diff 뷰:</strong> 각 파일의 변경사항을 개별 확인 및 승인</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-purple-400 font-semibold mb-4">Composer vs Cmd+K vs Chat</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-purple-400">기능</th>
                      <th className="text-left py-3 px-4 text-pink-400">Cmd+K</th>
                      <th className="text-left py-3 px-4 text-purple-400">Chat</th>
                      <th className="text-left py-3 px-4 text-pink-400">Composer</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-300">
                    <tr className="border-b border-gray-800">
                      <td className="py-3 px-4">파일 수</td>
                      <td className="py-3 px-4">1개 (현재 파일)</td>
                      <td className="py-3 px-4">1-2개</td>
                      <td className="py-3 px-4 text-pink-400 font-semibold">무제한</td>
                    </tr>
                    <tr className="border-b border-gray-800">
                      <td className="py-3 px-4">사용 사례</td>
                      <td className="py-3 px-4">빠른 수정</td>
                      <td className="py-3 px-4">질문, 탐색</td>
                      <td className="py-3 px-4 text-pink-400 font-semibold">대규모 리팩토링</td>
                    </tr>
                    <tr className="border-b border-gray-800">
                      <td className="py-3 px-4">속도</td>
                      <td className="py-3 px-4 text-green-400">매우 빠름</td>
                      <td className="py-3 px-4">빠름</td>
                      <td className="py-3 px-4">느림 (정확도 우선)</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4">컨텍스트</td>
                      <td className="py-3 px-4">선택 영역</td>
                      <td className="py-3 px-4">코드베이스</td>
                      <td className="py-3 px-4 text-pink-400 font-semibold">전체 프로젝트</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Advanced Features */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <BookOpen className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">고급 기능</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-purple-900/20 to-transparent rounded-xl p-6 border border-purple-500/20">
            <h3 className="text-xl font-semibold text-purple-400 mb-4">1. Codebase Indexing</h3>
            <p className="text-gray-300 mb-4">
              Cursor는 프로젝트를 자동으로 인덱싱하여 모든 함수, 클래스, 변수를 이해합니다.
            </p>
            <div className="bg-black/30 rounded-lg p-4 text-sm text-gray-400">
              <div className="mb-2">Settings → Cursor Settings → Features</div>
              <div className="text-purple-400">✅ Index Codebase (Enable)</div>
              <div className="mt-3 text-gray-500">
                → AI가 "이 프로젝트에서 인증은 어디서 처리?" 같은 질문에 정확히 답변
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-pink-900/20 to-transparent rounded-xl p-6 border border-pink-500/20">
            <h3 className="text-xl font-semibold text-pink-400 mb-4">2. Terminal Integration</h3>
            <p className="text-gray-300 mb-4">
              터미널 명령도 AI가 제안합니다. Cmd+K를 터미널에서 사용 가능.
            </p>
            <div className="bg-black/30 rounded-lg p-4 text-sm">
              <div className="text-pink-400 mb-2">💬 "Docker 컨테이너 실행 명령 알려줘"</div>
              <div className="font-mono text-gray-400">
{`$ docker run -d \\
  --name my-app \\
  -p 3000:3000 \\
  -e NODE_ENV=production \\
  my-app:latest`}
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-900/20 to-transparent rounded-xl p-6 border border-purple-500/20">
            <h3 className="text-xl font-semibold text-purple-400 mb-4">3. Rules for AI</h3>
            <p className="text-gray-300 mb-4">
              .cursorrules 파일로 프로젝트별 AI 동작을 커스터마이징.
            </p>
            <div className="bg-black/30 rounded-lg p-4 text-sm font-mono text-gray-400">
{`# .cursorrules
- Always use TypeScript strict mode
- Prefer functional components over class components
- Use Tailwind CSS for styling
- Add JSDoc comments for public APIs
- Never use 'any' type`}
            </div>
          </div>

          <div className="bg-gradient-to-br from-pink-900/20 to-transparent rounded-xl p-6 border border-pink-500/20">
            <h3 className="text-xl font-semibold text-pink-400 mb-4">4. Privacy Mode</h3>
            <p className="text-gray-300 mb-4">
              민감한 코드는 OpenAI로 전송하지 않도록 설정 가능.
            </p>
            <div className="bg-black/30 rounded-lg p-4 text-sm text-gray-300">
              <div className="mb-2">Settings → Privacy</div>
              <div className="text-pink-400">✅ Privacy Mode (Enable)</div>
              <div className="mt-2 text-gray-500">
                → .gitignore 파일에 명시된 파일은 AI 컨텍스트에서 제외
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">Cursor 마스터 팁</h2>

        <div className="space-y-4">
          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">1. 키보드 단축키 암기</h3>
            <div className="grid md:grid-cols-2 gap-3 text-gray-300 text-sm">
              <div><strong className="text-purple-400">Cmd+K:</strong> 인라인 편집</div>
              <div><strong className="text-purple-400">Cmd+L:</strong> Chat 열기</div>
              <div><strong className="text-purple-400">Cmd+Shift+I:</strong> Composer 모드</div>
              <div><strong className="text-purple-400">Cmd+I:</strong> 새 파일 생성 (AI로)</div>
              <div><strong className="text-purple-400">Tab:</strong> AI 제안 수락</div>
              <div><strong className="text-purple-400">Esc:</strong> AI 제안 거부</div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-pink-900/20 to-transparent rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-lg font-semibold text-pink-400 mb-2">2. 프롬프트 구체화</h3>
            <div className="space-y-2 text-gray-300">
              <div className="flex gap-2">
                <span className="text-red-400">❌</span>
                <span>"코드 고쳐줘"</span>
              </div>
              <div className="flex gap-2">
                <span className="text-green-400">✅</span>
                <span>"이 함수를 async/await로 변경하고 try-catch로 에러 처리 추가. 타입은 Promise&lt;User[]&gt;로 설정"</span>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">3. 반복 작업 자동화</h3>
            <p className="text-gray-300">
              같은 패턴의 코드를 여러 번 작성해야 한다면, 첫 번째를 AI로 생성하고
              <span className="text-purple-400"> @filename</span>으로 참조하여 나머지도 같은 스타일로 생성.
            </p>
          </div>

          <div className="bg-gradient-to-r from-pink-900/20 to-transparent rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-lg font-semibold text-pink-400 mb-2">4. 변경사항 항상 검토</h3>
            <p className="text-gray-300">
              Cursor의 Diff 뷰를 활용하여 AI가 변경한 모든 코드를 <span className="text-pink-400">줄 단위로 검토</span>하세요.
              Accept All 버튼은 신중하게 사용.
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
                Cursor는 VS Code 포크에 GPT-4를 네이티브 통합한 <strong>AI 우선 에디터</strong>
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">2.</span>
              <span>
                <strong>Cmd+K</strong> (인라인 편집), <strong>Cmd+L</strong> (Chat), <strong>Composer</strong> (멀티파일)가 핵심 기능
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">3.</span>
              <span>
                코드베이스 전체를 이해하고 컨텍스트 기반으로 <strong>정확한 제안</strong> 제공
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">4.</span>
              <span>
                .cursorrules 파일로 <strong>프로젝트별 AI 동작 커스터마이징</strong> 가능
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">5.</span>
              <span>
                Privacy Mode로 <strong>민감한 코드 보호</strong> 가능
              </span>
            </li>
          </ul>

          <div className="mt-8 p-6 bg-black/30 rounded-lg border border-purple-500/20">
            <p className="text-lg text-purple-400 font-semibold mb-2">다음 챕터 미리보기</p>
            <p className="text-gray-300">
              Chapter 3에서는 <strong>GitHub Copilot</strong>을 전문가 수준으로 활용합니다.
              Copilot Chat, Labs, CLI까지 모든 기능을 실전 예제와 함께 배웁니다.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
