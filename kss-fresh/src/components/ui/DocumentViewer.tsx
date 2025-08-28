'use client'

import { useState, useCallback } from 'react'
import { FileText, Download, Copy, Eye } from 'lucide-react'
import PDFViewer, { extractTextFromPDF } from './PDFViewer'

interface DocumentViewerProps {
  file: File
  className?: string
  showTextExtraction?: boolean
  onTextExtracted?: (text: string) => void
}

export default function DocumentViewer({ 
  file, 
  className = '',
  showTextExtraction = true,
  onTextExtracted 
}: DocumentViewerProps) {
  const [extractedText, setExtractedText] = useState<string>('')
  const [isExtracting, setIsExtracting] = useState(false)
  const [showTextPreview, setShowTextPreview] = useState(false)

  const getFileType = () => {
    const fileName = file.name.toLowerCase()
    const fileType = file.type.toLowerCase()
    
    if (fileName.endsWith('.pdf') || fileType.includes('pdf')) return 'pdf'
    if (fileName.endsWith('.txt') || fileType.includes('text/plain')) return 'txt'
    if (fileName.endsWith('.json') || fileType.includes('application/json')) return 'json'
    if (fileName.endsWith('.md') || fileName.endsWith('.markdown')) return 'md'
    if (fileName.endsWith('.csv')) return 'csv'
    if (fileName.endsWith('.xml')) return 'xml'
    return 'unknown'
  }

  const extractText = useCallback(async () => {
    setIsExtracting(true)
    try {
      const fileType = getFileType()
      let text = ''

      if (fileType === 'pdf') {
        text = await extractTextFromPDF(file)
      } else if (fileType === 'json') {
        const jsonText = await file.text()
        try {
          const parsed = JSON.parse(jsonText)
          text = `JSON 파일 "${file.name}" 내용:\n\n${JSON.stringify(parsed, null, 2)}`
        } catch {
          text = `JSON 파일 "${file.name}" 내용:\n\n${jsonText}`
        }
      } else if (['txt', 'md', 'csv', 'xml'].includes(fileType)) {
        text = await file.text()
      } else {
        // 기타 파일은 텍스트로 읽기 시도
        try {
          text = await file.text()
        } catch (error) {
          text = `파일 "${file.name}"을 텍스트로 읽을 수 없습니다.\n\n파일 정보:\n- 크기: ${(file.size / 1024).toFixed(1)}KB\n- 타입: ${file.type || '알 수 없음'}`
        }
      }

      setExtractedText(text)
      onTextExtracted?.(text)
    } catch (error) {
      console.error('텍스트 추출 오류:', error)
      setExtractedText(`텍스트 추출 중 오류가 발생했습니다: ${error}`)
    } finally {
      setIsExtracting(false)
    }
  }, [file, onTextExtracted])

  const copyToClipboard = () => {
    navigator.clipboard.writeText(extractedText)
      .then(() => alert('텍스트가 클립보드에 복사되었습니다'))
      .catch(() => alert('클립보드 복사에 실패했습니다'))
  }

  const downloadText = () => {
    const blob = new Blob([extractedText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${file.name}_extracted.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const fileType = getFileType()

  return (
    <div className={`document-viewer ${className}`}>
      {/* File Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText 
              className={`${
                fileType === 'pdf' ? 'text-red-600' : 
                fileType === 'json' ? 'text-blue-600' :
                fileType === 'md' ? 'text-purple-600' :
                'text-gray-600'
              }`} 
              size={24} 
            />
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white truncate">
                {file.name}
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {(file.size / 1024).toFixed(1)}KB • {fileType.toUpperCase()}
                {extractedText && (
                  <span className="ml-2">
                    • {extractedText.length.toLocaleString()}자
                  </span>
                )}
              </p>
            </div>
          </div>

          {showTextExtraction && (
            <div className="flex items-center gap-2">
              {!extractedText ? (
                <button
                  onClick={extractText}
                  disabled={isExtracting}
                  className="px-4 py-2 bg-emerald-500 text-white rounded-lg font-medium hover:bg-emerald-600 disabled:bg-gray-400 transition-colors flex items-center gap-2"
                >
                  {isExtracting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      추출 중...
                    </>
                  ) : (
                    <>
                      <FileText size={16} />
                      텍스트 추출
                    </>
                  )}
                </button>
              ) : (
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowTextPreview(!showTextPreview)}
                    className="px-3 py-2 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2"
                  >
                    <Eye size={14} />
                    {showTextPreview ? '미리보기 숨기기' : '미리보기 보기'}
                  </button>
                  <button
                    onClick={copyToClipboard}
                    className="px-3 py-2 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2"
                  >
                    <Copy size={14} />
                    복사
                  </button>
                  <button
                    onClick={downloadText}
                    className="px-3 py-2 text-sm bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center gap-2"
                  >
                    <Download size={14} />
                    다운로드
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Document Viewer */}
      <div className="flex-1">
        {fileType === 'pdf' ? (
          <PDFViewer 
            file={file} 
            className="h-full"
            onTextExtracted={(text, pageIndex) => {
              // PDF 페이지별 텍스트가 추출되면 전체 텍스트에 추가
              if (!extractedText) {
                setExtractedText(text)
                onTextExtracted?.(text)
              }
            }}
          />
        ) : (
          <div className="h-96 overflow-auto bg-gray-50 dark:bg-gray-900 p-4">
            {extractedText ? (
              <pre className="text-sm text-gray-700 dark:text-gray-300 font-mono whitespace-pre-wrap">
                {extractedText}
              </pre>
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <FileText className="mx-auto text-gray-400 mb-4" size={48} />
                  <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
                    {file.name}
                  </h4>
                  <p className="text-gray-500 dark:text-gray-400">
                    '텍스트 추출' 버튼을 클릭하여 파일 내용을 확인하세요
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Text Preview Panel (for non-PDF files) */}
      {showTextPreview && extractedText && fileType !== 'pdf' && (
        <div className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <div className="p-4 max-h-60 overflow-auto">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              추출된 텍스트 미리보기:
            </h4>
            <pre className="text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
              {extractedText.substring(0, 1000)}
              {extractedText.length > 1000 && '...'}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

// 파일 타입 체크 유틸리티
export const getSupportedFileTypes = () => {
  return {
    pdf: { label: 'PDF 문서', extensions: ['.pdf'], mime: ['application/pdf'] },
    txt: { label: '텍스트 파일', extensions: ['.txt'], mime: ['text/plain'] },
    json: { label: 'JSON 파일', extensions: ['.json'], mime: ['application/json'] },
    markdown: { label: '마크다운 문서', extensions: ['.md', '.markdown'], mime: ['text/markdown'] },
    csv: { label: 'CSV 파일', extensions: ['.csv'], mime: ['text/csv'] },
    xml: { label: 'XML 파일', extensions: ['.xml'], mime: ['application/xml', 'text/xml'] }
  }
}

export const isFileSupported = (file: File): boolean => {
  const fileName = file.name.toLowerCase()
  const supportedExtensions = ['.pdf', '.txt', '.json', '.md', '.markdown', '.csv', '.xml']
  return supportedExtensions.some(ext => fileName.endsWith(ext))
}