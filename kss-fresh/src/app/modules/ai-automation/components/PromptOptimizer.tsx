'use client';

import { useState } from 'react';
import { Sparkles, Copy, RotateCcw, DollarSign, Hash, Lightbulb, CheckCircle2 } from 'lucide-react';

type PromptPattern = 'zero-shot' | 'few-shot' | 'chain-of-thought' | 'react' | 'system-user';

interface Template {
  id: string;
  title: string;
  category: string;
  userPrompt: string;
  pattern: PromptPattern;
}

const TEMPLATES: Template[] = [
  {
    id: 'code-review',
    title: '코드 리뷰',
    category: '개발',
    userPrompt: '이 코드를 리뷰해줘',
    pattern: 'system-user'
  },
  {
    id: 'bug-fix',
    title: '버그 수정',
    category: '개발',
    userPrompt: '이 버그 고쳐줘',
    pattern: 'react'
  },
  {
    id: 'refactoring',
    title: '리팩토링',
    category: '개발',
    userPrompt: '이 코드 리팩토링 해줘',
    pattern: 'chain-of-thought'
  },
  {
    id: 'api-design',
    title: 'API 설계',
    category: '개발',
    userPrompt: 'REST API 만들어줘',
    pattern: 'few-shot'
  },
  {
    id: 'test-code',
    title: '테스트 작성',
    category: '개발',
    userPrompt: '테스트 코드 작성해줘',
    pattern: 'few-shot'
  },
  {
    id: 'documentation',
    title: '문서 생성',
    category: '문서',
    userPrompt: 'README 작성해줘',
    pattern: 'system-user'
  },
  {
    id: 'explain-code',
    title: '코드 설명',
    category: '학습',
    userPrompt: '이 코드 설명해줘',
    pattern: 'chain-of-thought'
  },
  {
    id: 'optimization',
    title: '성능 최적화',
    category: '개발',
    userPrompt: '성능 최적화 해줘',
    pattern: 'react'
  },
  {
    id: 'database-query',
    title: 'SQL 쿼리',
    category: '데이터',
    userPrompt: 'SQL 쿼리 작성해줘',
    pattern: 'few-shot'
  },
  {
    id: 'error-handling',
    title: '에러 처리',
    category: '개발',
    userPrompt: '에러 처리 추가해줘',
    pattern: 'system-user'
  },
  {
    id: 'ui-component',
    title: 'UI 컴포넌트',
    category: '개발',
    userPrompt: 'React 컴포넌트 만들어줘',
    pattern: 'few-shot'
  },
  {
    id: 'algorithm',
    title: '알고리즘',
    category: '알고리즘',
    userPrompt: '이 알고리즘 구현해줘',
    pattern: 'chain-of-thought'
  },
  {
    id: 'security',
    title: '보안 검사',
    category: '보안',
    userPrompt: '보안 취약점 찾아줘',
    pattern: 'react'
  },
  {
    id: 'architecture',
    title: '아키텍처 설계',
    category: '설계',
    userPrompt: '시스템 아키텍처 설계해줘',
    pattern: 'chain-of-thought'
  },
  {
    id: 'migration',
    title: '마이그레이션',
    category: '개발',
    userPrompt: '코드 마이그레이션 해줘',
    pattern: 'react'
  }
];

const PATTERN_INFO = {
  'zero-shot': {
    name: 'Zero-Shot',
    description: '직접적인 질문/요청',
    color: 'bg-blue-500',
    example: '단순하고 직접적인 요청'
  },
  'few-shot': {
    name: 'Few-Shot',
    description: '예시를 통한 학습',
    color: 'bg-green-500',
    example: '예시 코드와 함께 요청'
  },
  'chain-of-thought': {
    name: 'Chain-of-Thought',
    description: '단계별 사고 과정',
    color: 'bg-purple-500',
    example: '단계별로 생각하며 해결'
  },
  'react': {
    name: 'ReAct',
    description: '추론 + 행동 순환',
    color: 'bg-orange-500',
    example: '문제 분석 후 해결 방안 실행'
  },
  'system-user': {
    name: 'System-User',
    description: '시스템 프롬프트 활용',
    color: 'bg-pink-500',
    example: '역할과 규칙 정의 후 요청'
  }
};

export default function PromptOptimizer() {
  const [userPrompt, setUserPrompt] = useState('');
  const [selectedPattern, setSelectedPattern] = useState<PromptPattern>('zero-shot');
  const [copiedType, setCopiedType] = useState<'before' | 'after' | null>(null);

  const transformPrompt = (prompt: string, pattern: PromptPattern): string => {
    if (!prompt.trim()) return '';

    switch (pattern) {
      case 'zero-shot':
        return prompt;

      case 'few-shot':
        return `다음 예시를 참고하여 작업해주세요:

예시 1:
입력: "사용자 인증 API 구현"
출력: NextAuth.js를 활용한 Google OAuth 인증 시스템

예시 2:
입력: "데이터 검증 추가"
출력: Zod 스키마를 활용한 타입 안전 검증

이제 다음 요청을 처리해주세요:
${prompt}`;

      case 'chain-of-thought':
        return `다음 작업을 단계별로 수행해주세요:

작업: ${prompt}

단계별 접근:
1. 문제 분석: 현재 상태와 요구사항 파악
2. 해결 방안: 가능한 접근 방법들 검토
3. 최적 선택: 가장 적합한 방법 선택 및 이유
4. 구현: 선택한 방법으로 코드 작성
5. 검증: 예상되는 테스트 케이스와 엣지 케이스

각 단계를 명확히 설명하며 진행해주세요.`;

      case 'react':
        return `다음 작업을 ReAct 패턴으로 수행해주세요:

작업: ${prompt}

프로세스:
[Thought] 현재 상황 분석
[Action] 필요한 정보 수집 또는 조사
[Observation] 수집한 정보 검토
[Thought] 해결 방안 도출
[Action] 코드 구현
[Observation] 결과 검증
[Thought] 최종 결론 및 개선사항

각 단계를 명확히 표시하며 진행해주세요.`;

      case 'system-user':
        return `System: 당신은 10년 경력의 시니어 개발자입니다.
코드 품질, 성능, 보안을 최우선으로 고려하며 작업합니다.
모든 코드는 TypeScript를 사용하고, 테스트 가능하며, 문서화되어야 합니다.

작업 규칙:
- 타입 안전성 보장
- 에러 처리 필수
- 성능 최적화 고려
- 주석과 함께 설명
- 테스트 케이스 제안

User: ${prompt}

위 규칙을 따라 전문가답게 처리해주세요.`;

      default:
        return prompt;
    }
  };

  const estimateTokens = (text: string): number => {
    // 간단한 토큰 추정: 단어 수 * 1.3
    const words = text.split(/\s+/).length;
    return Math.ceil(words * 1.3);
  };

  const estimateCost = (tokens: number): number => {
    // Claude Sonnet 3.5 기준: $3 per 1M input tokens
    return (tokens / 1000000) * 3;
  };

  const handleCopy = (text: string, type: 'before' | 'after') => {
    navigator.clipboard.writeText(text);
    setCopiedType(type);
    setTimeout(() => setCopiedType(null), 2000);
  };

  const handleTemplateSelect = (template: Template) => {
    setUserPrompt(template.userPrompt);
    setSelectedPattern(template.pattern);
  };

  const transformedPrompt = transformPrompt(userPrompt, selectedPattern);
  const beforeTokens = estimateTokens(userPrompt);
  const afterTokens = estimateTokens(transformedPrompt);
  const beforeCost = estimateCost(beforeTokens);
  const afterCost = estimateCost(afterTokens);

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              프롬프트 최적화 실험실
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              다양한 프롬프트 패턴으로 최적의 결과를 얻어보세요
            </p>
          </div>
        </div>
      </div>

      {/* Pattern Selection */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          프롬프트 패턴 선택
        </label>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {(Object.keys(PATTERN_INFO) as PromptPattern[]).map((pattern) => {
            const info = PATTERN_INFO[pattern];
            const isSelected = selectedPattern === pattern;

            return (
              <button
                key={pattern}
                onClick={() => setSelectedPattern(pattern)}
                className={`p-4 rounded-xl border-2 transition-all ${
                  isSelected
                    ? 'border-violet-500 bg-violet-50 dark:bg-violet-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-violet-300'
                }`}
              >
                <div className={`w-3 h-3 rounded-full ${info.color} mb-2`} />
                <div className="font-bold text-sm text-gray-900 dark:text-white mb-1">
                  {info.name}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {info.description}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Workspace */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Before */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-gray-900 dark:text-white">
              원본 프롬프트
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setUserPrompt('')}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                title="초기화"
              >
                <RotateCcw className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              </button>
              <button
                onClick={() => handleCopy(userPrompt, 'before')}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                title="복사"
                disabled={!userPrompt}
              >
                {copiedType === 'before' ? (
                  <CheckCircle2 className="w-4 h-4 text-green-600" />
                ) : (
                  <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                )}
              </button>
            </div>
          </div>

          <textarea
            value={userPrompt}
            onChange={(e) => setUserPrompt(e.target.value)}
            placeholder="프롬프트를 입력하세요..."
            className="w-full h-64 p-4 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-violet-500 text-gray-900 dark:text-white"
          />

          {/* Stats */}
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-xs mb-1">
                <Hash className="w-3 h-3" />
                토큰 수
              </div>
              <div className="text-lg font-bold text-gray-900 dark:text-white">
                {beforeTokens.toLocaleString()}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-xs mb-1">
                <DollarSign className="w-3 h-3" />
                예상 비용
              </div>
              <div className="text-lg font-bold text-gray-900 dark:text-white">
                ${beforeCost.toFixed(6)}
              </div>
            </div>
          </div>
        </div>

        {/* After */}
        <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl border border-violet-200 dark:border-violet-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-gray-900 dark:text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-violet-600" />
              최적화된 프롬프트
            </h3>
            <button
              onClick={() => handleCopy(transformedPrompt, 'after')}
              className="p-2 hover:bg-white/50 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="복사"
              disabled={!transformedPrompt}
            >
              {copiedType === 'after' ? (
                <CheckCircle2 className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-violet-600 dark:text-violet-400" />
              )}
            </button>
          </div>

          <div className="w-full h-64 p-4 bg-white dark:bg-gray-800 border border-violet-200 dark:border-violet-700 rounded-lg overflow-y-auto">
            {transformedPrompt ? (
              <pre className="text-sm text-gray-900 dark:text-white whitespace-pre-wrap font-mono">
                {transformedPrompt}
              </pre>
            ) : (
              <div className="text-gray-400 dark:text-gray-600">
                프롬프트를 입력하면 최적화된 버전이 표시됩니다
              </div>
            )}
          </div>

          {/* Stats */}
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-xs mb-1">
                <Hash className="w-3 h-3" />
                토큰 수
              </div>
              <div className="flex items-center justify-between">
                <div className="text-lg font-bold text-gray-900 dark:text-white">
                  {afterTokens.toLocaleString()}
                </div>
                {afterTokens > beforeTokens && (
                  <span className="text-xs text-orange-600 dark:text-orange-400">
                    +{((afterTokens / beforeTokens - 1) * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-xs mb-1">
                <DollarSign className="w-3 h-3" />
                예상 비용
              </div>
              <div className="flex items-center justify-between">
                <div className="text-lg font-bold text-gray-900 dark:text-white">
                  ${afterCost.toFixed(6)}
                </div>
                {afterCost > beforeCost && (
                  <span className="text-xs text-orange-600 dark:text-orange-400">
                    +{((afterCost / beforeCost - 1) * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pattern Info */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 mb-6">
        <div className="flex items-start gap-3">
          <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
          <div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-1">
              {PATTERN_INFO[selectedPattern].name} 패턴
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              {PATTERN_INFO[selectedPattern].description}
            </p>
            <div className="text-xs text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-800 rounded-lg p-2 inline-block">
              예시: {PATTERN_INFO[selectedPattern].example}
            </div>
          </div>
        </div>
      </div>

      {/* Templates */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-violet-600" />
          베스트 프랙티스 템플릿
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {TEMPLATES.map((template) => (
            <button
              key={template.id}
              onClick={() => handleTemplateSelect(template)}
              className="p-3 bg-gray-50 dark:bg-gray-900 hover:bg-violet-50 dark:hover:bg-violet-900/20 border border-gray-200 dark:border-gray-700 hover:border-violet-300 rounded-lg transition-all text-left"
            >
              <div className="text-xs text-violet-600 dark:text-violet-400 font-medium mb-1">
                {template.category}
              </div>
              <div className="text-sm font-bold text-gray-900 dark:text-white mb-1">
                {template.title}
              </div>
              <div className={`inline-block px-2 py-0.5 rounded text-xs text-white ${PATTERN_INFO[template.pattern].color}`}>
                {PATTERN_INFO[template.pattern].name}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
