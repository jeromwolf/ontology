'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Sparkles, Code, Zap, Settings, Play, Copy, Check } from 'lucide-react'

type Language = 'python' | 'typescript' | 'javascript'

interface Suggestion {
  text: string
  description: string
  confidence: number
}

export default function AICodeAssistant() {
  const [language, setLanguage] = useState<Language>('python')
  const [code, setCode] = useState('')
  const [cursorPosition, setCursorPosition] = useState(0)
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [selectedSuggestion, setSelectedSuggestion] = useState(0)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [copied, setCopied] = useState(false)
  const [aiEnabled, setAiEnabled] = useState(true)
  const [autoComplete, setAutoComplete] = useState(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const codeTemplates: Record<Language, string> = {
    python: `# AI 코드 어시스턴트 데모
def calculate_fibonacci(n):
    """피보나치 수열을 계산합니다."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# 함수 호출
result = calculate_fibonacci(10)
print(f"결과: {result}")`,
    typescript: `// AI 코드 어시스턴트 데모
interface User {
  id: number;
  name: string;
  email: string;
}

function fetchUsers(): Promise<User[]> {
  return fetch('/api/users')
    .then(res => res.json())
    .catch(err => console.error(err));
}

// 사용 예제
fetchUsers().then(users => {
  console.log('사용자:', users);
});`,
    javascript: `// AI 코드 어시스턴트 데모
const users = [
  { id: 1, name: 'Alice', age: 25 },
  { id: 2, name: 'Bob', age: 30 },
  { id: 3, name: 'Charlie', age: 35 }
];

// 배열 필터링
const adults = users.filter(user => user.age >= 18);
console.log('성인 사용자:', adults);

// 맵 변환
const names = users.map(user => user.name);
console.log('이름 목록:', names);`
  }

  const suggestionDatabase: Record<Language, Suggestion[]> = {
    python: [
      {
        text: 'def fetch_data(url: str) -> dict:',
        description: 'HTTP 요청으로 데이터 가져오기',
        confidence: 95
      },
      {
        text: 'import pandas as pd',
        description: '데이터 분석용 Pandas 라이브러리',
        confidence: 92
      },
      {
        text: 'async def process_data(data):',
        description: '비동기 데이터 처리 함수',
        confidence: 88
      },
      {
        text: 'class DataProcessor:',
        description: '데이터 처리 클래스 정의',
        confidence: 85
      },
      {
        text: 'with open(filename, "r") as f:',
        description: '파일 안전하게 읽기',
        confidence: 90
      }
    ],
    typescript: [
      {
        text: 'interface ApiResponse<T> {',
        description: '제네릭 API 응답 타입',
        confidence: 94
      },
      {
        text: 'const fetchData = async (): Promise<void> => {',
        description: '비동기 데이터 가져오기',
        confidence: 91
      },
      {
        text: 'type User = {',
        description: '사용자 타입 정의',
        confidence: 89
      },
      {
        text: 'export default function Component() {',
        description: 'React 컴포넌트 생성',
        confidence: 93
      },
      {
        text: 'const handleClick = (e: React.MouseEvent) => {',
        description: '클릭 이벤트 핸들러',
        confidence: 87
      }
    ],
    javascript: [
      {
        text: 'const fetchData = async () => {',
        description: '비동기 함수 정의',
        confidence: 93
      },
      {
        text: 'const users = data.map(item => ({',
        description: '배열 변환 및 객체 생성',
        confidence: 90
      },
      {
        text: 'try { ... } catch (error) {',
        description: '에러 처리 블록',
        confidence: 88
      },
      {
        text: 'export default function App() {',
        description: 'React 컴포넌트 내보내기',
        confidence: 92
      },
      {
        text: 'const [state, setState] = useState(',
        description: 'React 상태 관리',
        confidence: 89
      }
    ]
  }

  useEffect(() => {
    setCode(codeTemplates[language])
  }, [language])

  useEffect(() => {
    if (aiEnabled && autoComplete && code.length > 0) {
      const lastLine = code.split('\n').pop() || ''
      if (lastLine.trim().length > 2) {
        const filtered = suggestionDatabase[language].filter(s =>
          s.text.toLowerCase().includes(lastLine.trim().toLowerCase().slice(0, 3))
        )
        setSuggestions(filtered.length > 0 ? filtered : suggestionDatabase[language].slice(0, 3))
        setShowSuggestions(true)
      } else {
        setShowSuggestions(false)
      }
    }
  }, [code, language, aiEnabled, autoComplete])

  const handleCodeChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCode(e.target.value)
    setCursorPosition(e.target.selectionStart)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab' && showSuggestions && suggestions.length > 0) {
      e.preventDefault()
      insertSuggestion(suggestions[selectedSuggestion].text)
    } else if (e.key === 'ArrowDown' && showSuggestions) {
      e.preventDefault()
      setSelectedSuggestion((prev) => (prev + 1) % suggestions.length)
    } else if (e.key === 'ArrowUp' && showSuggestions) {
      e.preventDefault()
      setSelectedSuggestion((prev) => (prev - 1 + suggestions.length) % suggestions.length)
    } else if (e.key === 'Escape') {
      setShowSuggestions(false)
    }
  }

  const insertSuggestion = (suggestion: string) => {
    const lines = code.split('\n')
    lines[lines.length - 1] = suggestion
    const newCode = lines.join('\n')
    setCode(newCode)
    setShowSuggestions(false)
    setSelectedSuggestion(0)
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleRun = () => {
    alert('코드 실행 시뮬레이션\n\n실제 프로덕션에서는 샌드박스 환경에서 코드가 실행됩니다.')
  }

  const loadTemplate = () => {
    setCode(codeTemplates[language])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-purple-100 dark:from-gray-900 dark:via-purple-900 dark:to-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-800 dark:text-white">
              AI 코드 어시스턴트
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            AI 기반 코드 자동완성 및 제안 시스템
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Code Editor */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              {/* Toolbar */}
              <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-4">
                  <Code className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value as Language)}
                    className="px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300"
                  >
                    <option value="python">Python</option>
                    <option value="typescript">TypeScript</option>
                    <option value="javascript">JavaScript</option>
                  </select>
                  <button
                    onClick={loadTemplate}
                    className="px-4 py-2 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-200 dark:hover:bg-purple-800 transition-colors"
                  >
                    템플릿 불러오기
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCopy}
                    className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                    title="코드 복사"
                  >
                    {copied ? <Check className="w-5 h-5 text-green-600" /> : <Copy className="w-5 h-5 text-gray-600 dark:text-gray-400" />}
                  </button>
                  <button
                    onClick={handleRun}
                    className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg hover:from-purple-600 hover:to-pink-700 transition-all flex items-center gap-2"
                  >
                    <Play className="w-4 h-4" />
                    실행
                  </button>
                </div>
              </div>

              {/* Code Editor */}
              <div className="relative">
                <textarea
                  ref={textareaRef}
                  value={code}
                  onChange={handleCodeChange}
                  onKeyDown={handleKeyDown}
                  className="w-full h-96 p-4 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 font-mono text-sm rounded-lg border border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                  spellCheck={false}
                  placeholder="코드를 입력하세요... Tab 키로 AI 제안을 적용할 수 있습니다."
                />

                {/* Suggestions Overlay */}
                {showSuggestions && aiEnabled && (
                  <div className="absolute bottom-4 left-4 right-4 bg-white dark:bg-gray-800 rounded-lg shadow-2xl border border-purple-200 dark:border-purple-700 max-h-48 overflow-y-auto">
                    <div className="p-2">
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-2 px-2 flex items-center gap-2">
                        <Zap className="w-3 h-3 text-purple-500" />
                        AI 제안 (Tab으로 적용, ↑↓로 선택)
                      </div>
                      {suggestions.map((suggestion, index) => (
                        <div
                          key={index}
                          onClick={() => insertSuggestion(suggestion.text)}
                          className={`p-3 rounded-lg cursor-pointer transition-colors ${
                            index === selectedSuggestion
                              ? 'bg-purple-100 dark:bg-purple-900'
                              : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                          }`}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1">
                              <code className="text-sm font-mono text-purple-600 dark:text-purple-400">
                                {suggestion.text}
                              </code>
                              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                                {suggestion.description}
                              </p>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="text-xs font-semibold text-purple-600 dark:text-purple-400">
                                {suggestion.confidence}%
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Status Bar */}
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
                <div className="flex items-center gap-4">
                  <span>줄: {code.split('\n').length}</span>
                  <span>문자: {code.length}</span>
                  <span>언어: {language.toUpperCase()}</span>
                </div>
                <div className="flex items-center gap-2">
                  {aiEnabled && (
                    <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                      <Sparkles className="w-4 h-4" />
                      AI 활성
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Settings & Info Panel */}
          <div className="space-y-6">
            {/* Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <Settings className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                <h3 className="text-lg font-bold text-gray-800 dark:text-white">설정</h3>
              </div>

              <div className="space-y-4">
                <label className="flex items-center justify-between">
                  <span className="text-sm text-gray-700 dark:text-gray-300">AI 어시스턴트</span>
                  <input
                    type="checkbox"
                    checked={aiEnabled}
                    onChange={(e) => setAiEnabled(e.target.checked)}
                    className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                  />
                </label>

                <label className="flex items-center justify-between">
                  <span className="text-sm text-gray-700 dark:text-gray-300">자동 완성</span>
                  <input
                    type="checkbox"
                    checked={autoComplete}
                    onChange={(e) => setAutoComplete(e.target.checked)}
                    disabled={!aiEnabled}
                    className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500 disabled:opacity-50"
                  />
                </label>
              </div>
            </div>

            {/* AI Suggestions Stats */}
            <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl shadow-xl p-6 text-white">
              <h3 className="text-lg font-bold mb-4">AI 통계</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm">제안 횟수</span>
                  <span className="text-xl font-bold">{suggestions.length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">평균 신뢰도</span>
                  <span className="text-xl font-bold">
                    {suggestions.length > 0
                      ? Math.round(suggestions.reduce((acc, s) => acc + s.confidence, 0) / suggestions.length)
                      : 0}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">코드 라인</span>
                  <span className="text-xl font-bold">{code.split('\n').length}</span>
                </div>
              </div>
            </div>

            {/* Quick Tips */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-4">💡 사용 팁</h3>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li>• <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">Tab</kbd>으로 제안 적용</li>
                <li>• <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">↑</kbd><kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">↓</kbd>로 제안 선택</li>
                <li>• <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">Esc</kbd>로 제안 닫기</li>
                <li>• AI는 컨텍스트 기반 제안 제공</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
