'use client'

import React, { useState, useEffect } from 'react'
import { Sparkles, TrendingUp, AlertCircle, CheckCircle, Copy, RotateCcw } from 'lucide-react'

interface Metric {
  name: string
  score: number
  description: string
  suggestions: string[]
}

interface Analysis {
  metrics: Metric[]
  overallScore: number
  optimizedPrompt: string
}

export default function PromptOptimizer() {
  const [userPrompt, setUserPrompt] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [copied, setCopied] = useState(false)

  const examplePrompts = [
    {
      title: '기본 예제',
      prompt: 'Python 코드를 작성해주세요.'
    },
    {
      title: '개선된 예제',
      prompt: 'Python으로 CSV 파일을 읽어서 pandas DataFrame으로 변환하고, 결측치를 제거한 후 평균을 계산하는 함수를 작성해주세요. 타입 힌트와 docstring을 포함해주세요.'
    },
    {
      title: '웹 스크래핑',
      prompt: '웹사이트에서 데이터를 가져오는 코드를 만들어주세요.'
    }
  ]

  const analyzePrompt = (prompt: string): Analysis => {
    const metrics: Metric[] = [
      {
        name: '명확성 (Clarity)',
        score: calculateClarityScore(prompt),
        description: '요청이 얼마나 명확하고 이해하기 쉬운가',
        suggestions: getClaritySuggestions(prompt)
      },
      {
        name: '구체성 (Specificity)',
        score: calculateSpecificityScore(prompt),
        description: '세부 요구사항이 얼마나 구체적인가',
        suggestions: getSpecificitySuggestions(prompt)
      },
      {
        name: '맥락 (Context)',
        score: calculateContextScore(prompt),
        description: '배경 정보와 사용 목적이 포함되어 있는가',
        suggestions: getContextSuggestions(prompt)
      },
      {
        name: '예시 (Examples)',
        score: calculateExamplesScore(prompt),
        description: '입출력 예시가 제공되어 있는가',
        suggestions: getExamplesSuggestions(prompt)
      },
      {
        name: '제약조건 (Constraints)',
        score: calculateConstraintsScore(prompt),
        description: '기술적 제약이나 요구사항이 명시되어 있는가',
        suggestions: getConstraintsSuggestions(prompt)
      }
    ]

    const overallScore = Math.round(
      metrics.reduce((sum, m) => sum + m.score, 0) / metrics.length
    )

    const optimizedPrompt = generateOptimizedPrompt(prompt, metrics)

    return { metrics, overallScore, optimizedPrompt }
  }

  const calculateClarityScore = (prompt: string): number => {
    let score = 50
    if (prompt.length > 20) score += 10
    if (prompt.includes('?')) score += 10
    if (/[가-힣a-zA-Z]{3,}/.test(prompt)) score += 10
    if (prompt.split(' ').length > 5) score += 10
    if (!prompt.includes('...') && !prompt.includes('등등')) score += 10
    return Math.min(100, score)
  }

  const calculateSpecificityScore = (prompt: string): number => {
    let score = 40
    const specificKeywords = ['함수', 'class', '메서드', 'API', '파일', '데이터', '형식', 'JSON', 'CSV']
    specificKeywords.forEach(keyword => {
      if (prompt.includes(keyword)) score += 8
    })
    if (prompt.length > 50) score += 10
    if (/\d+/.test(prompt)) score += 10 // 숫자 포함
    return Math.min(100, score)
  }

  const calculateContextScore = (prompt: string): number => {
    let score = 30
    const contextKeywords = ['위해', '목적', '사용', '필요', '프로젝트', '시스템']
    contextKeywords.forEach(keyword => {
      if (prompt.includes(keyword)) score += 12
    })
    if (prompt.length > 100) score += 20
    return Math.min(100, score)
  }

  const calculateExamplesScore = (prompt: string): number => {
    let score = 20
    if (prompt.includes('예시') || prompt.includes('예제')) score += 30
    if (prompt.includes('입력') || prompt.includes('출력')) score += 25
    if (prompt.includes('->') || prompt.includes('=>')) score += 25
    return Math.min(100, score)
  }

  const calculateConstraintsScore = (prompt: string): number => {
    let score = 30
    const constraintKeywords = ['타입', 'type', '에러', 'error', '최적화', '성능', '제약', '조건']
    constraintKeywords.forEach(keyword => {
      if (prompt.toLowerCase().includes(keyword.toLowerCase())) score += 12
    })
    if (prompt.includes('docstring') || prompt.includes('주석')) score += 15
    return Math.min(100, score)
  }

  const getClaritySuggestions = (prompt: string): string[] => {
    const suggestions = []
    if (prompt.length < 20) suggestions.push('프롬프트를 더 자세히 작성하세요')
    if (!prompt.includes('?') && !prompt.includes('.')) suggestions.push('명확한 질문 형태로 작성하세요')
    if (prompt.includes('...')) suggestions.push('모호한 표현 대신 구체적으로 작성하세요')
    return suggestions
  }

  const getSpecificitySuggestions = (prompt: string): string[] => {
    const suggestions = []
    if (prompt.length < 50) suggestions.push('구체적인 요구사항을 추가하세요')
    if (!prompt.match(/Python|JavaScript|TypeScript|Java/i)) suggestions.push('프로그래밍 언어를 명시하세요')
    if (!prompt.match(/함수|class|메서드/)) suggestions.push('원하는 코드의 형태를 지정하세요')
    return suggestions
  }

  const getContextSuggestions = (prompt: string): string[] => {
    const suggestions = []
    if (prompt.length < 100) suggestions.push('사용 목적이나 배경을 추가하세요')
    if (!prompt.includes('위해') && !prompt.includes('목적')) suggestions.push('왜 이 코드가 필요한지 설명하세요')
    return suggestions
  }

  const getExamplesSuggestions = (prompt: string): string[] => {
    const suggestions = []
    if (!prompt.includes('예시') && !prompt.includes('예제')) {
      suggestions.push('입출력 예시를 포함하세요')
    }
    if (!prompt.includes('입력') && !prompt.includes('출력')) {
      suggestions.push('기대하는 결과를 명시하세요')
    }
    return suggestions
  }

  const getConstraintsSuggestions = (prompt: string): string[] => {
    const suggestions = []
    if (!prompt.toLowerCase().includes('type') && !prompt.includes('타입')) {
      suggestions.push('타입 힌트 필요 여부를 명시하세요')
    }
    if (!prompt.includes('에러') && !prompt.includes('error')) {
      suggestions.push('에러 처리 방식을 지정하세요')
    }
    return suggestions
  }

  const generateOptimizedPrompt = (originalPrompt: string, metrics: Metric[]): string => {
    let optimized = originalPrompt

    // 언어가 명시되지 않았으면 추가
    if (!optimized.match(/Python|JavaScript|TypeScript|Java/i)) {
      optimized = `Python으로 ${optimized}`
    }

    // 구체성 향상
    if (metrics.find(m => m.name.includes('구체성'))!.score < 60) {
      if (!optimized.includes('함수')) {
        optimized += ' 재사용 가능한 함수 형태로 작성해주세요.'
      }
    }

    // 제약조건 추가
    if (metrics.find(m => m.name.includes('제약조건'))!.score < 60) {
      optimized += ' 타입 힌트와 docstring을 포함하고, 에러 처리를 추가해주세요.'
    }

    // 예시 추가
    if (metrics.find(m => m.name.includes('예시'))!.score < 60) {
      optimized += ' 사용 예시도 함께 제공해주세요.'
    }

    return optimized
  }

  const handleAnalyze = () => {
    if (userPrompt.trim().length === 0) return

    setIsAnalyzing(true)
    setTimeout(() => {
      const result = analyzePrompt(userPrompt)
      setAnalysis(result)
      setIsAnalyzing(false)
    }, 1000)
  }

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleReset = () => {
    setUserPrompt('')
    setAnalysis(null)
  }

  const loadExample = (prompt: string) => {
    setUserPrompt(prompt)
    setAnalysis(null)
  }

  const getScoreColor = (score: number): string => {
    if (score >= 80) return 'text-green-600 dark:text-green-400'
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getScoreBgColor = (score: number): string => {
    if (score >= 80) return 'bg-green-500'
    if (score >= 60) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-purple-100 dark:from-gray-900 dark:via-purple-900 dark:to-gray-900 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl">
              <TrendingUp className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-800 dark:text-white">
              프롬프트 최적화 도구
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            AI에게 더 나은 결과를 얻기 위한 프롬프트 작성법
          </p>
        </div>

        {/* Example Prompts */}
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">예시 프롬프트:</h3>
          <div className="flex flex-wrap gap-3">
            {examplePrompts.map((example, index) => (
              <button
                key={index}
                onClick={() => loadExample(example.prompt)}
                className="px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg border border-purple-200 dark:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-900 transition-colors text-sm"
              >
                {example.title}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div>
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 mb-6">
              <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-600" />
                원본 프롬프트
              </h2>
              <textarea
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                placeholder="AI에게 요청할 프롬프트를 입력하세요...&#10;&#10;예: Python으로 웹 스크래핑 코드를 작성해주세요."
                className="w-full h-48 p-4 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 rounded-lg border border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
              />
              <div className="flex items-center justify-between mt-4">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {userPrompt.length} 문자
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={handleReset}
                    className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    초기화
                  </button>
                  <button
                    onClick={handleAnalyze}
                    disabled={userPrompt.trim().length === 0 || isAnalyzing}
                    className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg hover:from-purple-600 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    {isAnalyzing ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        분석 중...
                      </>
                    ) : (
                      <>
                        <TrendingUp className="w-4 h-4" />
                        분석하기
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Metrics */}
            {analysis && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
                <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4">분석 결과</h2>

                {/* Overall Score */}
                <div className="mb-6 p-4 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl text-white">
                  <div className="text-sm mb-2">종합 점수</div>
                  <div className="flex items-end gap-2">
                    <div className="text-5xl font-bold">{analysis.overallScore}</div>
                    <div className="text-xl mb-2">/100</div>
                  </div>
                </div>

                {/* Individual Metrics */}
                <div className="space-y-4">
                  {analysis.metrics.map((metric, index) => (
                    <div key={index} className="border-b border-gray-200 dark:border-gray-700 pb-4 last:border-0">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-gray-800 dark:text-white">{metric.name}</span>
                        <span className={`text-xl font-bold ${getScoreColor(metric.score)}`}>
                          {metric.score}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-500 ${getScoreBgColor(metric.score)}`}
                          style={{ width: `${metric.score}%` }}
                        />
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{metric.description}</p>
                      {metric.suggestions.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {metric.suggestions.map((suggestion, idx) => (
                            <div key={idx} className="flex items-start gap-2 text-sm text-yellow-700 dark:text-yellow-400">
                              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                              <span>{suggestion}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Output Section */}
          <div>
            {analysis && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    최적화된 프롬프트
                  </h2>
                  <button
                    onClick={() => handleCopy(analysis.optimizedPrompt)}
                    className="p-2 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-200 dark:hover:bg-purple-800 transition-colors"
                    title="복사"
                  >
                    {copied ? <CheckCircle className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                  </button>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 border-2 border-green-200 dark:border-green-800 rounded-lg">
                  <p className="text-gray-800 dark:text-gray-200 whitespace-pre-wrap">{analysis.optimizedPrompt}</p>
                </div>

                {/* Comparison */}
                <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <h3 className="font-semibold text-gray-800 dark:text-white mb-2">개선 사항</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-start gap-2 text-green-700 dark:text-green-400">
                      <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>프로그래밍 언어가 명확히 지정됨</span>
                    </div>
                    <div className="flex items-start gap-2 text-green-700 dark:text-green-400">
                      <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>구체적인 요구사항이 추가됨</span>
                    </div>
                    <div className="flex items-start gap-2 text-green-700 dark:text-green-400">
                      <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>코드 품질 기준이 명시됨</span>
                    </div>
                  </div>
                </div>

                {/* Tips */}
                <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <h3 className="font-semibold text-gray-800 dark:text-white mb-2">💡 프롬프트 작성 팁</h3>
                  <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                    <li>• 프로그래밍 언어를 명시하세요</li>
                    <li>• 입출력 형식을 구체적으로 설명하세요</li>
                    <li>• 코드 스타일 요구사항을 추가하세요</li>
                    <li>• 에러 처리 방식을 지정하세요</li>
                    <li>• 사용 예시를 포함하세요</li>
                  </ul>
                </div>
              </div>
            )}

            {!analysis && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 h-full flex items-center justify-center">
                <div className="text-center text-gray-400 dark:text-gray-600">
                  <Sparkles className="w-16 h-16 mx-auto mb-4" />
                  <p className="text-lg">프롬프트를 입력하고<br />분석하기 버튼을 클릭하세요</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
