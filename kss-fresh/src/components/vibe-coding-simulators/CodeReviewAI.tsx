'use client'

import React, { useState } from 'react'
import { Shield, AlertTriangle, Info, CheckCircle, Search, Code, TrendingDown } from 'lucide-react'

type Severity = 'critical' | 'warning' | 'info'
type Category = 'security' | 'performance' | 'style' | 'best-practices'

interface Issue {
  line: number
  severity: Severity
  category: Category
  message: string
  suggestion: string
  code?: string
}

interface ReviewResult {
  issues: Issue[]
  score: number
  summary: {
    critical: number
    warning: number
    info: number
  }
}

export default function CodeReviewAI() {
  const [code, setCode] = useState('')
  const [reviewResult, setReviewResult] = useState<ReviewResult | null>(null)
  const [isReviewing, setIsReviewing] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState<Category | 'all'>('all')

  const exampleCodes = [
    {
      title: '보안 취약점',
      code: `def login(username, password):
    query = "SELECT * FROM users WHERE username='" + username + "' AND password='" + password + "'"
    result = db.execute(query)
    return result

def save_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)`
    },
    {
      title: '성능 문제',
      code: `def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j and arr[i] == arr[j]:
                if arr[i] not in duplicates:
                    duplicates.append(arr[i])
    return duplicates`
    },
    {
      title: '스타일 문제',
      code: `def Calculate_Sum(x,y):
    result=x+y
    return result

def PROCESS_DATA(data):
    for item in data:
        print(item)`
    }
  ]

  const analyzeCode = (sourceCode: string): ReviewResult => {
    const issues: Issue[] = []
    const lines = sourceCode.split('\n')

    lines.forEach((line, index) => {
      const lineNum = index + 1

      // Security checks
      if (line.includes('SELECT') && (line.includes('+') || line.includes('f"'))) {
        issues.push({
          line: lineNum,
          severity: 'critical',
          category: 'security',
          message: 'SQL Injection 취약점',
          suggestion: 'Parameterized queries를 사용하세요',
          code: 'cursor.execute("SELECT * FROM users WHERE username=?", (username,))'
        })
      }

      if (line.includes('open(') && !line.includes('with')) {
        issues.push({
          line: lineNum,
          severity: 'warning',
          category: 'best-practices',
          message: '파일 핸들러가 명시적으로 닫히지 않음',
          suggestion: 'with 문을 사용하여 자동으로 파일을 닫으세요',
          code: 'with open(filename, "r") as f:\n    content = f.read()'
        })
      }

      if (line.includes('eval(') || line.includes('exec(')) {
        issues.push({
          line: lineNum,
          severity: 'critical',
          category: 'security',
          message: 'Code Injection 위험',
          suggestion: 'eval()과 exec() 사용을 피하세요',
          code: '// 대신 ast.literal_eval() 또는 json.loads() 사용'
        })
      }

      // Performance checks
      if (line.includes('for') && sourceCode.includes('for') &&
          sourceCode.split('for').length - 1 >= 2) {
        const nestedForCount = sourceCode.match(/for/g)?.length || 0
        if (nestedForCount >= 2 && line.includes('for')) {
          issues.push({
            line: lineNum,
            severity: 'warning',
            category: 'performance',
            message: '중첩 루프로 인한 O(n²) 시간 복잡도',
            suggestion: 'Set 또는 Dictionary를 활용하여 O(n)으로 개선하세요',
            code: 'seen = set()\nfor item in arr:\n    if item in seen:\n        duplicates.add(item)\n    seen.add(item)'
          })
        }
      }

      if (line.includes('.append(') && lines.some(l => l.includes('for'))) {
        issues.push({
          line: lineNum,
          severity: 'info',
          category: 'performance',
          message: '루프 내 append() 사용',
          suggestion: 'List comprehension을 사용하면 더 빠릅니다',
          code: 'result = [item for item in data if condition]'
        })
      }

      // Style checks
      if (/def [A-Z]/.test(line) || /def .*[A-Z]{2,}/.test(line)) {
        issues.push({
          line: lineNum,
          severity: 'info',
          category: 'style',
          message: '함수명이 PEP 8 규칙을 따르지 않음',
          suggestion: 'snake_case를 사용하세요 (예: calculate_sum)',
          code: 'def calculate_sum(x, y):'
        })
      }

      if (line.includes('=') && !line.includes('==') && !line.includes('!=')) {
        const hasSpaces = line.match(/\s*=\s*/)
        if (!hasSpaces) {
          issues.push({
            line: lineNum,
            severity: 'info',
            category: 'style',
            message: '연산자 주변 공백 누락',
            suggestion: 'PEP 8: 연산자 앞뒤로 공백을 추가하세요',
            code: 'result = x + y'
          })
        }
      }

      // Best practices
      if (line.includes('range(len(')) {
        issues.push({
          line: lineNum,
          severity: 'info',
          category: 'best-practices',
          message: 'range(len()) 패턴 사용',
          suggestion: 'enumerate()를 사용하는 것이 더 pythonic합니다',
          code: 'for index, item in enumerate(arr):'
        })
      }

      if (line.includes('print(') && !line.includes('#')) {
        issues.push({
          line: lineNum,
          severity: 'info',
          category: 'best-practices',
          message: 'Debug용 print 문',
          suggestion: 'logging 모듈을 사용하거나 제거하세요',
          code: 'import logging\nlogging.info(f"Processing {item}")'
        })
      }
    })

    const summary = {
      critical: issues.filter(i => i.severity === 'critical').length,
      warning: issues.filter(i => i.severity === 'warning').length,
      info: issues.filter(i => i.severity === 'info').length
    }

    // Score calculation (100 - penalties)
    let score = 100
    score -= summary.critical * 20
    score -= summary.warning * 10
    score -= summary.info * 3
    score = Math.max(0, score)

    return { issues, score, summary }
  }

  const handleReview = () => {
    if (code.trim().length === 0) return

    setIsReviewing(true)
    setTimeout(() => {
      const result = analyzeCode(code)
      setReviewResult(result)
      setIsReviewing(false)
    }, 1500)
  }

  const loadExample = (exampleCode: string) => {
    setCode(exampleCode)
    setReviewResult(null)
  }

  const getSeverityIcon = (severity: Severity) => {
    switch (severity) {
      case 'critical':
        return <Shield className="w-5 h-5 text-red-600" />
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />
      case 'info':
        return <Info className="w-5 h-5 text-blue-600" />
    }
  }

  const getSeverityColor = (severity: Severity) => {
    switch (severity) {
      case 'critical':
        return 'border-red-500 bg-red-50 dark:bg-red-900/20'
      case 'warning':
        return 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
      case 'info':
        return 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
    }
  }

  const getCategoryIcon = (category: Category) => {
    switch (category) {
      case 'security':
        return '🔒'
      case 'performance':
        return '⚡'
      case 'style':
        return '✨'
      case 'best-practices':
        return '📚'
    }
  }

  const getCategoryName = (category: Category) => {
    switch (category) {
      case 'security':
        return '보안'
      case 'performance':
        return '성능'
      case 'style':
        return '스타일'
      case 'best-practices':
        return '모범 사례'
    }
  }

  const filteredIssues = reviewResult?.issues.filter(issue =>
    selectedCategory === 'all' || issue.category === selectedCategory
  ) || []

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400'
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getScoreGrade = (score: number) => {
    if (score >= 90) return 'A'
    if (score >= 80) return 'B'
    if (score >= 70) return 'C'
    if (score >= 60) return 'D'
    return 'F'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-purple-100 dark:from-gray-900 dark:via-purple-900 dark:to-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl">
              <Search className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-800 dark:text-white">
              AI 코드 리뷰어
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            자동화된 코드 품질 분석 및 개선 제안
          </p>
        </div>

        {/* Example Buttons */}
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">예제 코드:</h3>
          <div className="flex flex-wrap gap-3">
            {exampleCodes.map((example, index) => (
              <button
                key={index}
                onClick={() => loadExample(example.code)}
                className="px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg border border-purple-200 dark:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-900 transition-colors text-sm"
              >
                {example.title}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Code Input */}
          <div>
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
                  <Code className="w-5 h-5 text-purple-600" />
                  코드 입력
                </h2>
                <button
                  onClick={handleReview}
                  disabled={code.trim().length === 0 || isReviewing}
                  className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg hover:from-purple-600 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {isReviewing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      분석 중...
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4" />
                      리뷰 시작
                    </>
                  )}
                </button>
              </div>

              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="코드를 입력하세요...&#10;&#10;예: Python, JavaScript, TypeScript 등"
                className="w-full h-96 p-4 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 font-mono text-sm rounded-lg border border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                spellCheck={false}
              />

              <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                {code.split('\n').length} 줄, {code.length} 문자
              </div>
            </div>
          </div>

          {/* Review Results */}
          <div>
            {reviewResult ? (
              <div className="space-y-6">
                {/* Score Card */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
                  <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4">코드 품질 점수</h2>
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <div className={`text-6xl font-bold ${getScoreColor(reviewResult.score)}`}>
                        {reviewResult.score}
                      </div>
                      <div className="text-gray-600 dark:text-gray-400">/ 100</div>
                    </div>
                    <div className={`text-5xl font-bold ${getScoreColor(reviewResult.score)}`}>
                      {getScoreGrade(reviewResult.score)}
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                        {reviewResult.summary.critical}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">Critical</div>
                    </div>
                    <div className="text-center p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                        {reviewResult.summary.warning}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">Warning</div>
                    </div>
                    <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                        {reviewResult.summary.info}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">Info</div>
                    </div>
                  </div>
                </div>

                {/* Category Filter */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
                  <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-4">카테고리 필터</h3>
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={() => setSelectedCategory('all')}
                      className={`px-4 py-2 rounded-lg transition-colors ${
                        selectedCategory === 'all'
                          ? 'bg-purple-600 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      전체 ({reviewResult.issues.length})
                    </button>
                    {(['security', 'performance', 'style', 'best-practices'] as Category[]).map((cat) => (
                      <button
                        key={cat}
                        onClick={() => setSelectedCategory(cat)}
                        className={`px-4 py-2 rounded-lg transition-colors ${
                          selectedCategory === cat
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                        }`}
                      >
                        {getCategoryIcon(cat)} {getCategoryName(cat)} (
                        {reviewResult.issues.filter((i) => i.category === cat).length})
                      </button>
                    ))}
                  </div>
                </div>

                {/* Issues List */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 max-h-[600px] overflow-y-auto">
                  <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-4">
                    발견된 문제 ({filteredIssues.length})
                  </h3>

                  {filteredIssues.length === 0 ? (
                    <div className="text-center py-8 text-gray-400 dark:text-gray-600">
                      <CheckCircle className="w-12 h-12 mx-auto mb-2" />
                      <p>이 카테고리에서 문제가 발견되지 않았습니다!</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {filteredIssues.map((issue, index) => (
                        <div
                          key={index}
                          className={`p-4 rounded-lg border-l-4 ${getSeverityColor(issue.severity)}`}
                        >
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center gap-2">
                              {getSeverityIcon(issue.severity)}
                              <span className="font-semibold text-gray-800 dark:text-white">
                                Line {issue.line}
                              </span>
                              <span className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded">
                                {getCategoryIcon(issue.category)} {getCategoryName(issue.category)}
                              </span>
                            </div>
                          </div>
                          <p className="text-gray-800 dark:text-gray-200 mb-2">{issue.message}</p>
                          <div className="bg-white dark:bg-gray-900 p-3 rounded-lg mb-2">
                            <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">💡 제안:</div>
                            <p className="text-sm text-gray-700 dark:text-gray-300">{issue.suggestion}</p>
                          </div>
                          {issue.code && (
                            <div className="bg-gray-900 p-3 rounded-lg">
                              <div className="text-xs text-gray-400 mb-1">✅ 권장 코드:</div>
                              <code className="text-sm text-green-400 font-mono">{issue.code}</code>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 h-full flex items-center justify-center">
                <div className="text-center text-gray-400 dark:text-gray-600">
                  <Search className="w-16 h-16 mx-auto mb-4" />
                  <p className="text-lg">코드를 입력하고<br />리뷰 시작 버튼을 클릭하세요</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
