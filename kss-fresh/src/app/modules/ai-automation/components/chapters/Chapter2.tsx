'use client'

import { Terminal, FileCode, GitBranch, Settings } from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Claude Code 완벽 가이드
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Claude Code는 Anthropic이 공식 제공하는 CLI 도구로, 
            터미널에서 직접 Claude와 대화하며 코드를 생성하고 수정할 수 있습니다.
            MCP(Model Context Protocol)를 지원하여 프로젝트 전체를 이해하고 작업합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚀 설치 및 초기 설정
        </h3>
        
        <div className="bg-gray-900 rounded-lg p-6 mb-6">
          <pre className="text-green-400 font-mono text-sm overflow-x-auto">
{`# npm을 통한 설치
npm install -g @anthropic/claude-code

# 또는 Homebrew (macOS)
brew install claude-code

# API 키 설정
export ANTHROPIC_API_KEY="your-api-key"

# 프로젝트 초기화
claude-code init

# MCP 설정 파일 생성
claude-code mcp init`}</pre>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          📁 CLAUDE.md - 프로젝트 컨텍스트 관리
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            CLAUDE.md는 프로젝트의 컨텍스트를 Claude에게 전달하는 핵심 파일입니다.
            이 파일을 통해 프로젝트의 구조, 규칙, 목표를 명확히 전달할 수 있습니다.
          </p>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# CLAUDE.md 예시

## Project Overview
Next.js 14 기반의 e-commerce 플랫폼

## Technical Stack
- Framework: Next.js 14 (App Router)
- Language: TypeScript
- Styling: Tailwind CSS
- Database: PostgreSQL + Prisma

## Coding Conventions
- 함수명: camelCase
- 컴포넌트: PascalCase
- 상수: UPPER_SNAKE_CASE

## Current Focus
장바구니 기능 구현 중
- [ ] 상품 추가/삭제
- [ ] 수량 변경
- [ ] 가격 계산

## Important Notes
- 모든 API는 /api/v1 prefix 사용
- 에러 처리는 custom ErrorBoundary 사용
- 테스트는 Jest + RTL 사용`}</pre>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💬 효과적인 프롬프트 작성법
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h4 className="font-bold text-red-700 dark:text-red-400 mb-3">❌ 나쁜 예시</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <code className="text-sm text-gray-700 dark:text-gray-300">
                  "로그인 기능 만들어줘"
                </code>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <code className="text-sm text-gray-700 dark:text-gray-300">
                  "버그 고쳐줘"
                </code>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <code className="text-sm text-gray-700 dark:text-gray-300">
                  "최적화 해줘"
                </code>
              </div>
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-3">✅ 좋은 예시</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <code className="text-sm text-gray-700 dark:text-gray-300">
                  "NextAuth.js를 사용해서 Google OAuth 로그인 구현. 
                  사용자 정보는 Prisma로 PostgreSQL에 저장"
                </code>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <code className="text-sm text-gray-700 dark:text-gray-300">
                  "CartContext의 updateQuantity 함수에서 음수 값이 
                  들어올 때 발생하는 TypeError 수정"
                </code>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <code className="text-sm text-gray-700 dark:text-gray-300">
                  "ProductList 컴포넌트의 리렌더링 최적화. 
                  React.memo와 useMemo 활용"
                </code>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🛠️ MCP (Model Context Protocol) 활용
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            MCP는 Claude가 프로젝트의 도구와 리소스에 접근할 수 있게 하는 프로토콜입니다.
          </p>
          
          <div className="space-y-4">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h5 className="font-bold text-gray-900 dark:text-white mb-2">mcp.json 설정</h5>
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`{
  "servers": {
    "filesystem": {
      "command": "mcp-server-filesystem",
      "args": ["--root", "./src"]
    },
    "git": {
      "command": "mcp-server-git",
      "args": ["--repo", "."]
    },
    "database": {
      "command": "mcp-server-postgres",
      "env": {
        "DATABASE_URL": "$DATABASE_URL"
      }
    }
  }
}`}</pre>
            </div>
            
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-sm">
                파일시스템 접근
              </span>
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-sm">
                Git 히스토리
              </span>
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-sm">
                데이터베이스 스키마
              </span>
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-sm">
                API 문서
              </span>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3 mt-8">
          🎯 실전 활용 예시
        </h3>
        
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              전체 기능 구현
            </h4>
            <div className="bg-gray-900 rounded-lg p-4">
              <pre className="text-green-400 font-mono text-sm overflow-x-auto">
{`claude-code "다음 요구사항으로 상품 리뷰 시스템 구현:
1. Review 모델 생성 (rating, comment, userId, productId)
2. CRUD API 엔드포인트 (/api/v1/reviews)
3. ReviewList, ReviewForm 컴포넌트
4. 평점 평균 계산 및 표시
5. 이미지 업로드 지원 (최대 3장)
6. Jest 테스트 코드 포함"`}</pre>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              리팩토링
            </h4>
            <div className="bg-gray-900 rounded-lg p-4">
              <pre className="text-green-400 font-mono text-sm overflow-x-auto">
{`claude-code refactor "src/components/ProductCard.tsx를 
다음 기준으로 리팩토링:
- 비즈니스 로직을 커스텀 훅으로 분리
- 스타일 컴포넌트 분리
- 메모이제이션 적용
- Storybook 스토리 생성"`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          💡 Pro Tips
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <Terminal className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">세션 관리</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              `--resume` 플래그로 이전 대화 이어가기
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <FileCode className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-2" />
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">컨텍스트 제한</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              `.claudeignore` 파일로 불필요한 파일 제외
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <GitBranch className="w-8 h-8 text-green-600 dark:text-green-400 mb-2" />
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">버전 관리</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              작업 전 브랜치 생성, 완료 후 PR 자동 생성
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <Settings className="w-8 h-8 text-orange-600 dark:text-orange-400 mb-2" />
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">프로필 설정</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              프로젝트별 다른 설정 프로필 사용
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}