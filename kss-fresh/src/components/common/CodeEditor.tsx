'use client'

import { useState, useEffect } from 'react'
import { Copy, Download, Play, Code, Maximize2, Minimize2 } from 'lucide-react'

interface CodeEditorProps {
  code: string
  language: string
  title?: string
  filename?: string
  showLineNumbers?: boolean
  editable?: boolean
  maxHeight?: string
  theme?: 'dark' | 'light'
  showHeader?: boolean
  allowFullscreen?: boolean
  className?: string
}

export default function CodeEditor({ 
  code, 
  language, 
  title, 
  filename, 
  showLineNumbers = true, 
  editable = false,
  maxHeight = "400px",
  theme = 'dark',
  showHeader = true,
  allowFullscreen = true,
  className = ""
}: CodeEditorProps) {
  const [copied, setCopied] = useState(false)
  const [editableCode, setEditableCode] = useState(code)
  const [isFullscreen, setIsFullscreen] = useState(false)

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(editableCode)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const downloadCode = () => {
    const element = document.createElement('a')
    const file = new Blob([editableCode], { type: 'text/plain' })
    element.href = URL.createObjectURL(file)
    element.download = filename || `code.${language}`
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }

  // ESC 키로 전체화면 종료
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false)
      }
    }

    if (isFullscreen) {
      document.addEventListener('keydown', handleKeyDown)
      return () => {
        document.removeEventListener('keydown', handleKeyDown)
      }
    }
  }, [isFullscreen])

  const lines = editableCode.split('\n')

  const themeClasses = {
    dark: {
      container: 'bg-gray-900 border-gray-700',
      header: 'bg-gray-800 border-gray-700',
      text: 'text-gray-100',
      textSecondary: 'text-gray-300',
      textMuted: 'text-gray-400',
      lineNumbers: 'bg-gray-800 text-gray-500 border-gray-700',
      button: 'text-gray-400 hover:text-white hover:bg-gray-700',
      tag: 'bg-gray-700 text-gray-400'
    },
    light: {
      container: 'bg-white border-gray-300',
      header: 'bg-gray-100 border-gray-300',
      text: 'text-gray-900',
      textSecondary: 'text-gray-700',
      textMuted: 'text-gray-600',
      lineNumbers: 'bg-gray-50 text-gray-500 border-gray-300',
      button: 'text-gray-600 hover:text-gray-900 hover:bg-gray-200',
      tag: 'bg-gray-200 text-gray-600'
    }
  }

  const currentTheme = themeClasses[theme]

  return (
    <>
      {/* Fullscreen overlay - 코드 에디터 뒤에 배치 */}
      {isFullscreen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40" 
          onClick={toggleFullscreen}
        />
      )}
      
      <div className={`
        ${currentTheme.container} rounded-lg border shadow-lg
        ${isFullscreen ? 'fixed inset-4 z-50 flex flex-col' : ''}
        ${className}
      `}>
      {/* Header */}
      {showHeader && (
        <div className={`${currentTheme.header} px-4 py-2 flex items-center justify-between border-b flex-shrink-0`}>
          <div className="flex items-center gap-3">
            <div className="flex gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            </div>
            <div className={`flex items-center gap-2 ${currentTheme.textSecondary}`}>
              <Code className="w-4 h-4" />
              <span className="text-sm font-mono">
                {title || filename || `code.${language}`}
              </span>
            </div>
            <div className={`px-2 py-1 ${currentTheme.tag} rounded text-xs font-mono`}>
              {language}
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={copyToClipboard}
              className={`p-2 ${currentTheme.button} rounded transition-colors`}
              title="Copy code"
            >
              <Copy className="w-4 h-4" />
            </button>
            <button
              onClick={downloadCode}
              className={`p-2 ${currentTheme.button} rounded transition-colors`}
              title="Download code"
            >
              <Download className="w-4 h-4" />
            </button>
            {language === 'python' && (
              <button
                className={`p-2 ${currentTheme.button} rounded transition-colors`}
                title="Run code (demo)"
              >
                <Play className="w-4 h-4" />
              </button>
            )}
            {allowFullscreen && (
              <button
                onClick={toggleFullscreen}
                className={`p-2 ${currentTheme.button} rounded transition-colors`}
                title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
              >
                {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Code content */}
      <div 
        className={`relative overflow-auto ${isFullscreen ? 'flex-1' : ''}`}
        style={{ 
          maxHeight: isFullscreen ? undefined : maxHeight 
        }}
      >
        {editable ? (
          <textarea
            value={editableCode}
            onChange={(e) => setEditableCode(e.target.value)}
            className={`w-full h-full p-4 bg-transparent ${currentTheme.text} font-mono text-sm leading-6 resize-none focus:outline-none`}
            style={{ minHeight: '200px' }}
          />
        ) : (
          <div className="flex">
            {showLineNumbers && (
              <div className={`select-none ${currentTheme.lineNumbers} px-4 py-4 font-mono text-sm leading-6 border-r`}>
                {lines.map((_, index) => (
                  <div key={index + 1} className="text-right">
                    {index + 1}
                  </div>
                ))}
              </div>
            )}
            <pre className={`flex-1 p-4 ${currentTheme.text} font-mono text-sm leading-6 overflow-x-auto`}>
              <code className={`language-${language}`}>
                {editableCode}
              </code>
            </pre>
          </div>
        )}
      </div>

      {/* Footer with copy notification */}
      {copied && (
        <div className="bg-green-800 px-4 py-2 text-green-100 text-sm">
          ✓ Code copied to clipboard!
        </div>
      )}
    </div>
    </>
  )
}