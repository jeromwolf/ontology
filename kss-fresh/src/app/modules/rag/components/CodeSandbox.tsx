'use client'

import { useState } from 'react'
import { Copy, Check, Play, Terminal } from 'lucide-react'

interface CodeSandboxProps {
  title: string
  description?: string
  code: string
  language?: 'python' | 'javascript' | 'typescript'
  output?: string
  highlightLines?: number[]
}

export default function CodeSandbox({
  title,
  description,
  code,
  language = 'python',
  output,
  highlightLines = []
}: CodeSandboxProps) {
  const [copied, setCopied] = useState(false)
  const [showOutput, setShowOutput] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleRun = () => {
    setShowOutput(!showOutput)
  }

  const getLanguageIcon = () => {
    switch (language) {
      case 'python':
        return '🐍'
      case 'javascript':
        return '📜'
      case 'typescript':
        return '🔷'
      default:
        return '💻'
    }
  }

  return (
    <div className="bg-gray-900 rounded-xl overflow-hidden shadow-lg border border-gray-700">
      {/* Header */}
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between border-b border-gray-700">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{getLanguageIcon()}</span>
          <div>
            <h3 className="font-semibold text-white">{title}</h3>
            {description && (
              <p className="text-sm text-gray-400">{description}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {output && (
            <button
              onClick={handleRun}
              className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <Play size={14} />
              {showOutput ? '출력 숨기기' : '실행'}
            </button>
          )}
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-medium transition-colors"
          >
            {copied ? (
              <>
                <Check size={14} className="text-green-400" />
                복사됨!
              </>
            ) : (
              <>
                <Copy size={14} />
                복사
              </>
            )}
          </button>
        </div>
      </div>

      {/* Code Block */}
      <div className="relative">
        <pre className="p-6 overflow-x-auto text-sm leading-relaxed">
          <code className="text-gray-100 font-mono">
            {code.split('\n').map((line, index) => (
              <div
                key={index}
                className={`${
                  highlightLines.includes(index + 1)
                    ? 'bg-yellow-500/20 border-l-2 border-yellow-500 pl-2 -ml-2'
                    : ''
                }`}
              >
                <span className="text-gray-500 select-none mr-4">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <span>{line || ' '}</span>
              </div>
            ))}
          </code>
        </pre>
      </div>

      {/* Output */}
      {output && showOutput && (
        <div className="border-t border-gray-700">
          <div className="bg-gray-850 px-4 py-2 flex items-center gap-2 text-gray-400 text-sm">
            <Terminal size={16} />
            <span>출력 결과</span>
          </div>
          <pre className="p-6 bg-black/30 text-green-400 font-mono text-sm overflow-x-auto">
            {output}
          </pre>
        </div>
      )}

      {/* Language Badge */}
      <div className="px-4 py-2 bg-gray-800/50 border-t border-gray-700">
        <span className="text-xs text-gray-400">
          언어: <span className="text-gray-300 font-medium">{language}</span>
        </span>
      </div>
    </div>
  )
}
