'use client';

import { useState } from 'react';
import { FileText, Hash, DollarSign, CheckCircle2, AlertTriangle, Copy, Download, RotateCcw } from 'lucide-react';

interface Section {
  id: string;
  name: string;
  present: boolean;
  priority: 'high' | 'medium' | 'low';
}

interface Template {
  id: string;
  name: string;
  content: string;
}

const TEMPLATES: Template[] = [
  {
    id: 'basic',
    name: '기본 구조',
    content: `# CLAUDE.md

## Project Overview
[프로젝트에 대한 간단한 설명]

## Technical Stack
[사용 기술 스택]

## Development Commands
[개발 명령어들]

## Important Notes
[중요한 참고사항]
`
  },
  {
    id: 'fullstack',
    name: 'Full Stack 프로젝트',
    content: `# CLAUDE.md

## Project Overview
Next.js 14 기반의 풀스택 웹 애플리케이션

## Technical Stack
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Database**: PostgreSQL + Prisma
- **Auth**: NextAuth.js
- **API**: REST + tRPC

## Development Commands
\`\`\`bash
npm install
npm run dev    # 개발 서버 (port 3000)
npm run build  # 프로덕션 빌드
npm run test   # 테스트 실행
\`\`\`

## Coding Conventions
- 파일명: camelCase for utilities, PascalCase for components
- 컴포넌트: 함수형 컴포넌트 + hooks
- 스타일: Tailwind utility classes
- 타입: 명시적 타입 선언

## Current Focus
- [ ] 사용자 인증 시스템 구현
- [ ] 대시보드 UI 개발
- [ ] API 엔드포인트 최적화

## Important Notes
- 모든 API 경로는 /api/v1 prefix 사용
- 컴포넌트는 src/components에 위치
- 공통 타입은 types/index.ts에 정의
`
  },
  {
    id: 'ai-agent',
    name: 'AI Agent 프로젝트',
    content: `# CLAUDE.md

## Project Overview
Claude Code를 활용한 AI 자동화 에이전트 시스템

## Technical Stack
- **CLI Tool**: @anthropic/claude-code
- **Language**: Python 3.11+ / TypeScript
- **MCP Servers**: filesystem, git, database
- **AI Models**: Claude Sonnet 3.5, GPT-4

## MCP Configuration
\`\`\`json
{
  "servers": {
    "filesystem": {
      "command": "mcp-server-filesystem",
      "args": ["--root", "./src"]
    },
    "git": {
      "command": "mcp-server-git"
    }
  }
}
\`\`\`

## Coding Conventions
- 프롬프트는 prompts/ 디렉토리에 저장
- 워크플로우는 YAML로 정의
- 모든 자동화 스크립트는 scripts/ 폴더

## Current Focus
AI 에이전트 워크플로우 구축:
1. 코드 리뷰 자동화
2. 테스트 생성 자동화
3. 문서화 자동화

## Important Notes
- CLAUDE.md는 항상 최신 상태 유지
- .claudeignore로 불필요한 파일 제외
- 세션 히스토리는 .claude/sessions에 저장
`
  },
  {
    id: 'refactoring',
    name: '리팩토링 프로젝트',
    content: `# CLAUDE.md

## Project Context
레거시 코드베이스 현대화 프로젝트

## Refactoring Goals
1. TypeScript 마이그레이션 (JavaScript → TypeScript)
2. 모놀리식 → 마이크로서비스 아키텍처
3. 테스트 커버리지 80% 이상 달성

## Technical Constraints
- 기존 API 호환성 유지
- 단계적 마이그레이션 (빅뱅 금지)
- 다운타임 최소화

## Current Phase
**Phase 1: TypeScript 마이그레이션 (진행중)**
- [x] 타입 정의 파일 생성
- [ ] 유틸리티 함수 마이그레이션
- [ ] React 컴포넌트 마이그레이션
- [ ] API 라우트 마이그레이션

## Code Patterns to Follow
\`\`\`typescript
// ✅ Good: 명시적 타입 정의
interface User {
  id: string;
  name: string;
  email: string;
}

// ❌ Bad: any 타입 사용
function processData(data: any) { }
\`\`\`

## Testing Strategy
- 기존 기능 유지를 위한 회귀 테스트 필수
- E2E 테스트로 전체 플로우 검증
- 단위 테스트는 순수 함수부터 우선 작성
`
  },
  {
    id: 'monorepo',
    name: 'Monorepo 프로젝트',
    content: `# CLAUDE.md

## Project Overview
Turborepo 기반 모노레포 구조

## Repository Structure
\`\`\`
monorepo/
├── apps/
│   ├── web/          # Next.js 웹 앱
│   ├── mobile/       # React Native 앱
│   └── admin/        # 관리자 대시보드
├── packages/
│   ├── ui/           # 공통 UI 컴포넌트
│   ├── config/       # 공통 설정
│   └── utils/        # 유틸리티 함수
└── turbo.json
\`\`\`

## Development Commands
\`\`\`bash
# 전체 빌드
turbo build

# 특정 앱 개발
turbo dev --filter=web

# 패키지 추가
pnpm add <package> --filter=<workspace>
\`\`\`

## Coding Conventions
- **Packages**: packages/는 독립적으로 배포 가능해야 함
- **Apps**: apps/는 packages/를 소비만 함
- **Imports**: 내부 패키지는 @repo/* alias 사용

## Important Notes
- 순환 참조 절대 금지
- 각 패키지는 독립적인 package.json 유지
- 공통 의존성은 루트 package.json에
- 타입스크립트 설정은 base tsconfig 상속
`
  },
  {
    id: 'migration',
    name: '마이그레이션 프로젝트',
    content: `# CLAUDE.md

## Migration Context
Vue 2 → React 18 + Next.js 14 마이그레이션

## Migration Strategy
**단계적 접근 (Strangler Fig Pattern)**
1. 새로운 페이지는 Next.js로 작성
2. 기존 Vue 페이지는 iframe으로 통합
3. 점진적으로 Vue 페이지를 React로 재작성
4. API 레이어는 중립적으로 유지

## Technical Mapping
| Vue 2 | React 18 + Next.js |
|-------|-------------------|
| Vuex  | Zustand / Context API |
| Vue Router | Next.js App Router |
| Composition API | React Hooks |
| <style scoped> | CSS Modules / Tailwind |

## Current Progress
- [x] 프로젝트 설정 완료
- [x] 홈페이지 마이그레이션
- [ ] 대시보드 마이그레이션 (진행중)
- [ ] 설정 페이지 마이그레이션

## Code Conversion Examples
\`\`\`vue
<!-- Vue 2 -->
<template>
  <div>{{ message }}</div>
</template>

<script>
export default {
  data() {
    return { message: 'Hello' }
  }
}
</script>
\`\`\`

\`\`\`tsx
// React 18
export default function Component() {
  const [message, setMessage] = useState('Hello');
  return <div>{message}</div>;
}
\`\`\`

## Important Notes
- 모든 마이그레이션은 기능 테스트 후 진행
- 기존 Vue 앱은 /legacy 경로에 유지
- API는 절대 동시 변경 금지
`
  }
];

const DEFAULT_SECTIONS: Section[] = [
  { id: 'overview', name: 'Project Overview', present: false, priority: 'high' },
  { id: 'stack', name: 'Technical Stack', present: false, priority: 'high' },
  { id: 'commands', name: 'Development Commands', present: false, priority: 'medium' },
  { id: 'conventions', name: 'Coding Conventions', present: false, priority: 'medium' },
  { id: 'focus', name: 'Current Focus', present: false, priority: 'high' },
  { id: 'notes', name: 'Important Notes', present: false, priority: 'low' }
];

export default function ContextManager() {
  const [content, setContent] = useState('');
  const [sections, setSections] = useState<Section[]>(DEFAULT_SECTIONS);

  const estimateTokens = (text: string): number => {
    const words = text.split(/\s+/).length;
    return Math.ceil(words * 1.3);
  };

  const estimateCost = (tokens: number): number => {
    return (tokens / 1000000) * 3; // Claude Sonnet 3.5: $3/1M tokens
  };

  const analyzeContent = (text: string) => {
    const newSections = sections.map(section => ({
      ...section,
      present: text.toLowerCase().includes(section.name.toLowerCase().replace(' ', ''))
    }));
    setSections(newSections);
  };

  const handleContentChange = (newContent: string) => {
    setContent(newContent);
    analyzeContent(newContent);
  };

  const handleTemplateSelect = (template: Template) => {
    handleContentChange(template.content);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
  };

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'CLAUDE.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleReset = () => {
    setContent('');
    setSections(DEFAULT_SECTIONS);
  };

  const tokens = estimateTokens(content);
  const cost = estimateCost(tokens);
  const tokenBudget = 200000; // Claude Code 컨텍스트 윈도우
  const tokenUsagePercent = Math.min(100, (tokens / tokenBudget) * 100);

  const completedSections = sections.filter(s => s.present).length;
  const highPrioritySections = sections.filter(s => s.priority === 'high' && s.present).length;
  const totalHighPriority = sections.filter(s => s.priority === 'high').length;

  const getRecommendations = (): string[] => {
    const recs: string[] = [];

    if (content.length < 100) {
      recs.push('CLAUDE.md가 너무 짧습니다. 프로젝트 정보를 더 추가하세요.');
    }

    if (!sections.find(s => s.id === 'overview')?.present) {
      recs.push('프로젝트 개요 섹션이 없습니다. Overview를 추가하세요.');
    }

    if (!sections.find(s => s.id === 'stack')?.present) {
      recs.push('기술 스택 정보가 없습니다. 사용 중인 기술을 명시하세요.');
    }

    if (!sections.find(s => s.id === 'focus')?.present) {
      recs.push('현재 작업 내용을 추가하면 더 정확한 AI 지원을 받을 수 있습니다.');
    }

    if (tokens > tokenBudget * 0.8) {
      recs.push('⚠️ 토큰 사용량이 80%를 초과했습니다. 불필요한 내용을 제거하세요.');
    }

    if (!content.includes('```')) {
      recs.push('코드 예시를 추가하면 AI가 코딩 스타일을 더 잘 이해합니다.');
    }

    if (recs.length === 0) {
      recs.push('✨ 완벽합니다! CLAUDE.md가 잘 작성되었습니다.');
    }

    return recs;
  };

  const recommendations = getRecommendations();

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl flex items-center justify-center">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              컨텍스트 관리 시뮬레이터
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              효과적인 CLAUDE.md 작성 및 관리
            </p>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6 mb-6">
        {/* Editor */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-gray-900 dark:text-white">
              CLAUDE.md 에디터
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={handleReset}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                title="초기화"
              >
                <RotateCcw className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              </button>
              <button
                onClick={handleCopy}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                title="복사"
                disabled={!content}
              >
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              </button>
              <button
                onClick={handleDownload}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                title="다운로드"
                disabled={!content}
              >
                <Download className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              </button>
            </div>
          </div>

          <textarea
            value={content}
            onChange={(e) => handleContentChange(e.target.value)}
            placeholder="CLAUDE.md 내용을 입력하거나 아래 템플릿을 선택하세요..."
            className="w-full h-96 p-4 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 dark:text-white font-mono text-sm"
          />
        </div>

        {/* Analysis Panel */}
        <div className="space-y-4">
          {/* Token Usage */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="font-bold text-gray-900 dark:text-white mb-4">
              토큰 사용량
            </h3>

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    사용 / 예산
                  </span>
                  <span className="text-sm font-bold text-gray-900 dark:text-white">
                    {tokens.toLocaleString()} / {tokenBudget.toLocaleString()}
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all ${
                      tokenUsagePercent > 80
                        ? 'bg-red-500'
                        : tokenUsagePercent > 50
                        ? 'bg-yellow-500'
                        : 'bg-green-500'
                    }`}
                    style={{ width: `${tokenUsagePercent}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {tokenUsagePercent.toFixed(1)}% 사용 중
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-xs mb-1">
                    <Hash className="w-3 h-3" />
                    토큰
                  </div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">
                    {tokens.toLocaleString()}
                  </div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-xs mb-1">
                    <DollarSign className="w-3 h-3" />
                    비용
                  </div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">
                    ${cost.toFixed(4)}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Section Checklist */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="font-bold text-gray-900 dark:text-white mb-4">
              섹션 체크리스트
            </h3>

            <div className="mb-4">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                완료: {completedSections}/{sections.length}
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${(completedSections / sections.length) * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              {sections.map((section) => (
                <div
                  key={section.id}
                  className="flex items-center justify-between p-2 rounded-lg bg-gray-50 dark:bg-gray-900"
                >
                  <div className="flex items-center gap-2">
                    {section.present ? (
                      <CheckCircle2 className="w-4 h-4 text-green-600" />
                    ) : (
                      <div className="w-4 h-4 border-2 border-gray-300 dark:border-gray-600 rounded-full" />
                    )}
                    <span className={`text-sm ${
                      section.present
                        ? 'text-gray-900 dark:text-white font-medium'
                        : 'text-gray-500 dark:text-gray-400'
                    }`}>
                      {section.name}
                    </span>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    section.priority === 'high'
                      ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                      : section.priority === 'medium'
                      ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                      : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-400'
                  }`}>
                    {section.priority}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl border border-blue-200 dark:border-blue-700 p-6">
            <h3 className="font-bold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              최적화 제안
            </h3>
            <ul className="space-y-2">
              {recommendations.map((rec, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <span className="text-blue-600 dark:text-blue-400 mt-0.5">•</span>
                  <span className="text-gray-700 dark:text-gray-300">{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Templates */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="font-bold text-gray-900 dark:text-white mb-4">
          템플릿 라이브러리
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {TEMPLATES.map((template) => (
            <button
              key={template.id}
              onClick={() => handleTemplateSelect(template)}
              className="p-4 bg-gray-50 dark:bg-gray-900 hover:bg-blue-50 dark:hover:bg-blue-900/20 border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 rounded-lg transition-all text-left"
            >
              <div className="text-sm font-bold text-gray-900 dark:text-white mb-1">
                {template.name}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {estimateTokens(template.content).toLocaleString()} tokens
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
