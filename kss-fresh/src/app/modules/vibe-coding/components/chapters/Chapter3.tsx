'use client'

import React from 'react'
import { Github, Zap, Terminal, Flask, MessageSquare, Code } from 'lucide-react'

export default function Chapter3() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-8">
        <div className="inline-block p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-2xl mb-4">
          <Github className="w-12 h-12 text-purple-500" />
        </div>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
          GitHub Copilot 전문가 되기
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          OpenAI Codex 기반 세계 최초 AI 코딩 어시스턴트 완전 정복
        </p>
      </div>

      {/* Introduction */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">GitHub Copilot이란?</h2>

        <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 rounded-xl p-8 border border-purple-500/20">
          <p className="text-lg text-gray-300 leading-relaxed mb-6">
            <span className="text-purple-400 font-semibold">GitHub Copilot</span>은 2021년 6월 출시된
            세계 최초의 AI 코딩 어시스턴트로, <span className="text-pink-400 font-semibold">OpenAI Codex</span> 모델을
            기반으로 수십억 줄의 공개 코드로 학습되었습니다. 2024년 현재 <span className="text-purple-400">150만+ 유료 구독자</span>를 보유한
            가장 인기 있는 AI 코딩 도구입니다.
          </p>

          <div className="grid md:grid-cols-4 gap-4">
            <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
              <div className="text-3xl font-bold text-purple-400 mb-1">150만+</div>
              <div className="text-sm text-gray-400">유료 구독자</div>
            </div>
            <div className="bg-pink-500/10 rounded-lg p-4 border border-pink-500/20">
              <div className="text-3xl font-bold text-pink-400 mb-1">46%</div>
              <div className="text-sm text-gray-400">코드 자동완성 수락률</div>
            </div>
            <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
              <div className="text-3xl font-bold text-purple-400 mb-1">55%</div>
              <div className="text-sm text-gray-400">코딩 속도 향상</div>
            </div>
            <div className="bg-pink-500/10 rounded-lg p-4 border border-pink-500/20">
              <div className="text-3xl font-bold text-pink-400 mb-1">88%</div>
              <div className="text-sm text-gray-400">생산성 향상 체감</div>
            </div>
          </div>
        </div>
      </section>

      {/* Setup */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Code className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">설치 및 설정</h2>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">1. 구독 플랜 선택</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
                <div className="text-purple-400 font-semibold mb-2">개인 플랜</div>
                <div className="text-2xl font-bold text-white mb-2">$10/월</div>
                <ul className="space-y-1 text-sm text-gray-300">
                  <li>• 무제한 코드 제안</li>
                  <li>• Copilot Chat 포함</li>
                  <li>• VS Code, JetBrains 지원</li>
                  <li>• GPT-4 기반 제안</li>
                </ul>
              </div>
              <div className="bg-pink-900/20 rounded-lg p-4 border border-pink-500/20">
                <div className="text-pink-400 font-semibold mb-2">비즈니스 플랜</div>
                <div className="text-2xl font-bold text-white mb-2">$19/월</div>
                <ul className="space-y-1 text-sm text-gray-300">
                  <li>• 개인 플랜 모든 기능</li>
                  <li>• 라이선스 관리</li>
                  <li>• 정책 제어 (IP 필터링)</li>
                  <li>• 조직 전용 모델 옵션</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-xl font-semibold text-pink-400 mb-3">2. VS Code 확장 설치</h3>
            <div className="space-y-3">
              <div className="bg-black/30 rounded-lg p-4 font-mono text-sm">
                <div className="text-gray-500">// 방법 1: VS Code Marketplace</div>
                <div className="text-purple-400 mt-2">Extensions → "GitHub Copilot" 검색 → Install</div>
                <div className="text-gray-500 mt-4">// 방법 2: CLI</div>
                <div className="text-pink-400 mt-2">code --install-extension GitHub.copilot</div>
              </div>
              <div className="text-gray-300 text-sm">
                설치 후 GitHub 계정으로 인증하면 즉시 사용 가능합니다.
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">3. 초기 설정 최적화</h3>
            <div className="bg-black/30 rounded-lg p-4 space-y-3 text-sm">
              <div>
                <div className="text-purple-400 font-semibold mb-1">자동 제안 활성화</div>
                <div className="text-gray-400">Settings → Copilot → Enable Auto Completions</div>
              </div>
              <div>
                <div className="text-pink-400 font-semibold mb-1">제안 표시 딜레이 조절</div>
                <div className="text-gray-400">Editor: Suggest → Delay: 100ms (기본 500ms보다 빠르게)</div>
              </div>
              <div>
                <div className="text-purple-400 font-semibold mb-1">언어별 활성화</div>
                <div className="text-gray-400">Copilot → Languages → Python, JavaScript, TypeScript 등 선택</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Core Features */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <Zap className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">핵심 기능: 인라인 제안</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-pink-500/20">
          <p className="text-lg text-gray-300 mb-6">
            Copilot의 기본 기능은 <span className="text-pink-400 font-semibold">인라인 코드 제안</span>입니다.
            타이핑하는 동안 자동으로 다음 코드를 예측하여 <span className="text-pink-400">회색 텍스트로 표시</span>합니다.
          </p>

          <div className="space-y-6">
            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-purple-400 font-semibold mb-4">예제 1: 함수 자동완성</h4>
              <div className="space-y-4">
                <div>
                  <div className="text-gray-500 text-sm mb-2">// 주석만 작성하면...</div>
                  <div className="bg-gray-900 rounded p-4 font-mono text-sm">
                    <div className="text-gray-400">// 이메일 유효성 검증 함수</div>
                    <div className="text-gray-500 italic">
                      {`function validateEmail(email) {
  const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
  return regex.test(email);
}`}
                    </div>
                  </div>
                </div>
                <div className="text-center">
                  <div className="inline-block bg-pink-500/20 rounded-lg px-4 py-2 text-pink-400 font-mono text-sm">
                    ⌨️ Tab 키를 누르면 코드 삽입
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-purple-400 font-semibold mb-4">예제 2: 테스트 코드 생성</h4>
              <div className="space-y-4">
                <div className="bg-gray-900 rounded p-4 font-mono text-sm">
                  <div className="text-gray-400">{`// 기존 함수
function add(a, b) {
  return a + b;
}

// 테스트 코드 작성 시작...
describe('add function', () => {`}</div>
                  <div className="text-gray-500 italic ml-4">
{`  it('should add two positive numbers', () => {
    expect(add(2, 3)).toBe(5);
  });

  it('should add negative numbers', () => {
    expect(add(-2, -3)).toBe(-5);
  });

  it('should handle zero', () => {
    expect(add(0, 5)).toBe(5);
  });`}
                  </div>
                  <div className="text-gray-400">{`});`}</div>
                </div>
              </div>
            </div>

            <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
              <h4 className="text-purple-400 font-semibold mb-3">💡 인라인 제안 활용 팁</h4>
              <ul className="space-y-2 text-gray-300 text-sm">
                <li><strong className="text-purple-400">Tab:</strong> 전체 제안 수락</li>
                <li><strong className="text-purple-400">Ctrl+→:</strong> 단어 단위로 수락</li>
                <li><strong className="text-purple-400">Alt+]:</strong> 다음 제안 보기</li>
                <li><strong className="text-purple-400">Alt+[:</strong> 이전 제안 보기</li>
                <li><strong className="text-purple-400">Esc:</strong> 제안 무시</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Copilot Chat */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <MessageSquare className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">Copilot Chat (GPT-4 기반)</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-purple-500/20">
          <p className="text-lg text-gray-300 mb-6">
            <span className="text-purple-400 font-semibold">Copilot Chat</span>은 2023년 추가된 기능으로,
            <span className="text-purple-400"> GPT-4를 활용</span>하여 대화형으로 코드를 작성하고 수정할 수 있습니다.
            Ctrl+I (Windows) 또는 Cmd+I (Mac)로 활성화.
          </p>

          <div className="space-y-6">
            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-pink-400 font-semibold mb-4">주요 기능</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
                  <div className="font-semibold text-purple-400 mb-2">1. 인라인 Chat</div>
                  <div className="text-gray-300 text-sm">
                    코드 내에서 직접 질문하고 수정<br/>
                    선택 영역에 대한 즉각 피드백
                  </div>
                </div>
                <div className="bg-pink-900/20 rounded-lg p-4 border border-pink-500/20">
                  <div className="font-semibold text-pink-400 mb-2">2. 사이드바 Chat</div>
                  <div className="text-gray-300 text-sm">
                    ChatGPT처럼 대화형 인터페이스<br/>
                    파일 전체를 컨텍스트로 전달 가능
                  </div>
                </div>
                <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
                  <div className="font-semibold text-purple-400 mb-2">3. Slash Commands</div>
                  <div className="text-gray-300 text-sm">
                    /explain, /fix, /tests 등<br/>
                    특정 작업에 최적화된 명령
                  </div>
                </div>
                <div className="bg-pink-900/20 rounded-lg p-4 border border-pink-500/20">
                  <div className="font-semibold text-pink-400 mb-2">4. @-mentions</div>
                  <div className="text-gray-300 text-sm">
                    @workspace, @terminal 등<br/>
                    컨텍스트 범위 확장
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-purple-400 font-semibold mb-4">실전 예제</h4>
              <div className="space-y-4">
                <div className="border-l-4 border-purple-500 pl-4">
                  <div className="text-purple-400 font-mono text-sm mb-2">/explain</div>
                  <div className="text-gray-300 text-sm mb-2">선택한 코드의 동작 설명</div>
                  <div className="bg-gray-900 rounded p-3 text-sm text-gray-400">
                    "이 useEffect 훅은 컴포넌트가 마운트될 때 API에서 사용자 데이터를 가져옵니다.
                    userId가 변경될 때마다 재실행되며, cleanup 함수로 메모리 누수를 방지합니다."
                  </div>
                </div>

                <div className="border-l-4 border-pink-500 pl-4">
                  <div className="text-pink-400 font-mono text-sm mb-2">/fix</div>
                  <div className="text-gray-300 text-sm mb-2">버그 자동 수정</div>
                  <div className="bg-gray-900 rounded p-3 text-sm">
                    <div className="text-red-400 mb-2">// Before: Infinite loop</div>
                    <div className="text-gray-400">{`useEffect(() => {
  setCount(count + 1);
});`}</div>
                    <div className="text-green-400 mt-3 mb-2">// After: Fixed dependency</div>
                    <div className="text-purple-400">{`useEffect(() => {
  setCount(prev => prev + 1);
}, []);`}</div>
                  </div>
                </div>

                <div className="border-l-4 border-purple-500 pl-4">
                  <div className="text-purple-400 font-mono text-sm mb-2">/tests</div>
                  <div className="text-gray-300 text-sm mb-2">자동 테스트 코드 생성</div>
                  <div className="bg-gray-900 rounded p-3 text-sm text-purple-400">
{`describe('UserService', () => {
  it('should fetch user by ID', async () => {
    const user = await UserService.getById('123');
    expect(user).toHaveProperty('id', '123');
    expect(user).toHaveProperty('email');
  });

  it('should throw error for invalid ID', async () => {
    await expect(UserService.getById('')).rejects.toThrow();
  });
});`}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Copilot Labs */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/10 rounded-lg">
            <Flask className="w-6 h-6 text-pink-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">Copilot Labs (실험 기능)</h2>
        </div>

        <div className="bg-gradient-to-br from-pink-900/20 to-purple-900/20 rounded-xl p-8 border border-pink-500/20">
          <p className="text-lg text-gray-300 mb-6">
            <span className="text-pink-400 font-semibold">Copilot Labs</span>는 GitHub이 테스트 중인
            최신 AI 기능을 먼저 체험할 수 있는 실험 공간입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-black/30 rounded-lg p-6 border border-purple-500/20">
              <h4 className="text-purple-400 font-semibold mb-3">Explain</h4>
              <p className="text-gray-300 text-sm mb-4">
                복잡한 코드를 단계별로 설명. 특히 레거시 코드나 타인의 코드를 이해할 때 유용.
              </p>
              <div className="bg-gray-900 rounded p-3 text-xs text-gray-400">
                선택 → Copilot Labs → Explain 클릭 →<br/>
                알고리즘, 시간 복잡도, 개선점까지 상세 분석
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6 border border-pink-500/20">
              <h4 className="text-pink-400 font-semibold mb-3">Language Translation</h4>
              <p className="text-gray-300 text-sm mb-4">
                코드를 다른 프로그래밍 언어로 자동 변환. Python → JavaScript, Java → TypeScript 등.
              </p>
              <div className="bg-gray-900 rounded p-3 text-xs text-gray-400">
                Python 코드 선택 → Translate to JavaScript →<br/>
                문법, 라이브러리까지 자동 변환
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6 border border-purple-500/20">
              <h4 className="text-purple-400 font-semibold mb-3">Brushes (코드 스타일 변환)</h4>
              <p className="text-gray-300 text-sm mb-4">
                코드를 특정 스타일로 변환: 가독성 개선, 간결화, 타입 추가, 문서화 등.
              </p>
              <div className="bg-gray-900 rounded p-3 text-xs text-gray-400">
                코드 선택 → Brush: Readable →<br/>
                변수명 개선, 주석 추가, 함수 분리
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6 border border-pink-500/20">
              <h4 className="text-pink-400 font-semibold mb-3">Test Generation</h4>
              <p className="text-gray-300 text-sm mb-4">
                함수나 클래스에 대한 단위 테스트를 자동 생성. Jest, Pytest, JUnit 등 지원.
              </p>
              <div className="bg-gray-900 rounded p-3 text-xs text-gray-400">
                함수 선택 → Generate Tests →<br/>
                엣지 케이스, 에러 케이스까지 포함
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Copilot CLI */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Terminal className="w-6 h-6 text-purple-400" />
          </div>
          <h2 className="text-3xl font-bold text-white">Copilot CLI (터미널에서 AI 활용)</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-8 border border-purple-500/20">
          <p className="text-lg text-gray-300 mb-6">
            <span className="text-purple-400 font-semibold">Copilot CLI</span>는 터미널에서 자연어로
            명령어를 생성하는 도구입니다. 복잡한 CLI 명령을 기억할 필요 없이 AI에게 요청하면 됩니다.
          </p>

          <div className="space-y-6">
            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-purple-400 font-semibold mb-4">설치</h4>
              <div className="bg-gray-900 rounded p-4 font-mono text-sm space-y-2">
                <div className="text-gray-500"># npm으로 설치</div>
                <div className="text-purple-400">npm install -g @githubnext/github-copilot-cli</div>
                <div className="text-gray-500 mt-3"># GitHub 계정 인증</div>
                <div className="text-pink-400">github-copilot-cli auth</div>
              </div>
            </div>

            <div className="bg-black/30 rounded-lg p-6">
              <h4 className="text-pink-400 font-semibold mb-4">사용 예제</h4>
              <div className="space-y-4">
                <div className="bg-gray-900 rounded p-4 font-mono text-sm">
                  <div className="text-gray-500"># Docker 컨테이너 관리</div>
                  <div className="text-purple-400">$ ?? remove all stopped containers</div>
                  <div className="text-gray-400 mt-2">→ docker container prune -f</div>
                </div>

                <div className="bg-gray-900 rounded p-4 font-mono text-sm">
                  <div className="text-gray-500"># Git 명령</div>
                  <div className="text-purple-400">$ ?? undo last commit but keep changes</div>
                  <div className="text-gray-400 mt-2">→ git reset --soft HEAD~1</div>
                </div>

                <div className="bg-gray-900 rounded p-4 font-mono text-sm">
                  <div className="text-gray-500"># 파일 검색</div>
                  <div className="text-purple-400">$ ?? find all .log files older than 7 days</div>
                  <div className="text-gray-400 mt-2">→ find . -name "*.log" -mtime +7</div>
                </div>

                <div className="bg-gray-900 rounded p-4 font-mono text-sm">
                  <div className="text-gray-500"># 프로세스 관리</div>
                  <div className="text-purple-400">$ ?? kill process using port 3000</div>
                  <div className="text-gray-400 mt-2">→ lsof -ti:3000 | xargs kill -9</div>
                </div>
              </div>
            </div>

            <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
              <h4 className="text-purple-400 font-semibold mb-3">💡 CLI 명령어</h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-300">
                <div><strong className="text-purple-400">??</strong> - 쉘 명령 생성</div>
                <div><strong className="text-purple-400">git?</strong> - Git 명령 생성</div>
                <div><strong className="text-purple-400">gh?</strong> - GitHub CLI 명령 생성</div>
                <div><strong className="text-purple-400">--explain</strong> - 명령 설명</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="space-y-6">
        <h2 className="text-3xl font-bold text-white">Copilot 고수 되기</h2>

        <div className="space-y-4">
          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">1. 주석으로 의도 명확히 표현</h3>
            <p className="text-gray-300 mb-3">
              Copilot은 주석을 읽고 다음 코드를 예측합니다. <span className="text-purple-400">구체적이고 명확한 주석</span>을
              작성할수록 정확한 제안을 받습니다.
            </p>
            <div className="bg-black/30 rounded p-3 font-mono text-sm">
              <div className="text-red-400">// Bad: 데이터 가져오기</div>
              <div className="text-green-400">// Good: Fetch user profile from API and cache it in localStorage for 1 hour</div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-pink-900/20 to-transparent rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-lg font-semibold text-pink-400 mb-2">2. 함수명과 변수명으로 컨텍스트 제공</h3>
            <p className="text-gray-300 mb-3">
              함수명이 <span className="text-pink-400">validateEmailAndSendConfirmation</span>이라면
              Copilot은 자동으로 이메일 유효성 검사 + 확인 메일 발송 로직을 제안합니다.
            </p>
          </div>

          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">3. 예제 코드 제공</h3>
            <p className="text-gray-300 mb-3">
              파일 상단에 한두 개의 <span className="text-purple-400">예제 함수</span>를 작성하면
              Copilot이 같은 패턴으로 나머지 함수를 생성합니다.
            </p>
          </div>

          <div className="bg-gradient-to-r from-pink-900/20 to-transparent rounded-lg p-6 border-l-4 border-pink-500">
            <h3 className="text-lg font-semibold text-pink-400 mb-2">4. 여러 제안 비교</h3>
            <p className="text-gray-300">
              <span className="text-pink-400">Alt+]</span>로 다음 제안을 보며 가장 적합한 것을 선택하세요.
              첫 번째 제안이 항상 최선은 아닙니다.
            </p>
          </div>

          <div className="bg-gradient-to-r from-purple-900/20 to-transparent rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-purple-400 mb-2">5. 보안 및 라이선스 검토</h3>
            <p className="text-gray-300">
              Copilot이 생성한 코드에 <span className="text-purple-400">보안 취약점</span>이나
              <span className="text-purple-400"> 라이선스 문제</span>가 있을 수 있으니 항상 검토하세요.
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
                GitHub Copilot은 <strong>150만+ 사용자</strong>가 선택한 세계 1위 AI 코딩 도구
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">2.</span>
              <span>
                <strong>인라인 제안</strong>, <strong>Copilot Chat</strong>, <strong>Labs</strong>, <strong>CLI</strong>의 4대 기능
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">3.</span>
              <span>
                주석과 함수명으로 <strong>명확한 컨텍스트</strong>를 제공할수록 정확한 제안
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-pink-400 text-xl">4.</span>
              <span>
                /explain, /fix, /tests 등 <strong>Slash Commands</strong>로 빠른 작업 수행
              </span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 text-xl">5.</span>
              <span>
                Copilot CLI로 <strong>터미널 명령도 자연어로 생성</strong> 가능
              </span>
            </li>
          </ul>

          <div className="mt-8 p-6 bg-black/30 rounded-lg border border-purple-500/20">
            <p className="text-lg text-purple-400 font-semibold mb-2">다음 챕터 미리보기</p>
            <p className="text-gray-300">
              Chapter 4에서는 <strong>Claude Code</strong>를 활용한 고급 코딩 기법을 배웁니다.
              200K 토큰 컨텍스트와 Artifacts 기능으로 대규모 프로젝트를 효율적으로 관리하는 방법을 학습합니다.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
