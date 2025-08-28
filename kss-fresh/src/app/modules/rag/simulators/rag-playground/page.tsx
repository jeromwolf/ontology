'use client'

import Link from 'next/link'
import { useState } from 'react'
import { ArrowLeft, Upload, FileText, Zap, Brain, RefreshCw, CheckCircle } from 'lucide-react'

export default function RAGPlaygroundPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [extractedText, setExtractedText] = useState<string>('')
  const [query, setQuery] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingMessage, setProcessingMessage] = useState('')
  const [results, setResults] = useState<any>(null)
  const [chatHistory, setChatHistory] = useState<Array<{question: string, answer: string, chunks: any[]}>>([])
  const [step, setStep] = useState<'upload' | 'viewer'>('upload')

  const extractTextFromFile = async (file: File): Promise<string> => {
    const fileName = file.name.toLowerCase()
    const fileType = file.type.toLowerCase()
    
    console.log('extractTextFromFile 시작:', fileName, fileType)
    
    try {
      if (fileName.endsWith('.txt') || fileType.includes('text/plain')) {
        // 텍스트 파일 - 실제 내용 읽기
        return await file.text()
      } else if (fileName.endsWith('.json') || fileType.includes('application/json')) {
        // JSON 파일 - 실제 내용 읽기 
        const jsonText = await file.text()
        try {
          const parsed = JSON.parse(jsonText)
          return `JSON 파일 "${file.name}" 내용:\n\n${JSON.stringify(parsed, null, 2)}`
        } catch {
          return `JSON 파일 "${file.name}" 내용:\n\n${jsonText}`
        }
      } else if (fileName.endsWith('.md') || fileName.endsWith('.markdown')) {
        // 마크다운 파일 - 실제 내용 읽기
        return await file.text()
      } else if (fileName.endsWith('.csv')) {
        // CSV 파일 - 실제 내용 읽기
        const csvText = await file.text()
        return `CSV 파일 "${file.name}" 내용:\n\n${csvText}`
      } else if (fileName.endsWith('.xml')) {
        // XML 파일 - 실제 내용 읽기
        return await file.text()
      } else if (fileName.endsWith('.pdf') || fileType.includes('pdf')) {
        console.log('PDF 파일 감지:', fileName)
        // PDF는 서버 사이드에서 처리
        try {
          // 진행 상태 메시지 설정
          setProcessingMessage('PDF를 서버로 전송 중... 📤')
          
          console.log('PDF를 서버로 전송 준비...')
          
          // FormData 생성
          const formData = new FormData()
          formData.append('file', file)
          
          // PDF 처리 API 호출 (새로운 엔드포인트 사용)
          const response = await fetch('/api/pdf-process', {
            method: 'POST',
            body: formData
          })
          
          console.log('API 응답 상태:', response.status, response.statusText)
          
          setProcessingMessage('PDF 텍스트 추출 중... 📄')
          
          const result = await response.json()
          
          if (!response.ok) {
            throw new Error(result.error || 'PDF 처리 실패')
          }
          
          console.log('PDF 추출 성공:', result.pageCount, '페이지')
          
          // 메타데이터 포함한 결과 반환
          return `PDF 파일 "${file.name}"에서 추출된 텍스트:

📄 파일 정보:
- 페이지 수: ${result.pageCount}
- 크기: ${(file.size / 1024).toFixed(1)}KB
${result.metadata.title ? `- 제목: ${result.metadata.title}` : ''}
${result.metadata.author ? `- 작성자: ${result.metadata.author}` : ''}
${result.metadata.textLength ? `- 추출된 텍스트: ${result.metadata.textLength.toLocaleString()}자` : ''}

==================
추출된 내용:
==================

${result.text}`
        } catch (error: any) {
          console.error('PDF 서버 처리 실패:', error)
          return `❌ PDF 파일 "${file.name}" 처리 중 오류가 발생했습니다.

${error.message}

다른 PDF 파일을 시도하거나, TXT/MD/JSON 형식을 사용해주세요.`
        }
      } else {
        // 기타 파일 - 텍스트로 읽기 시도
        try {
          const content = await file.text()
          if (content.length > 0) {
            return content
          }
        } catch (error) {
          console.log('텍스트로 읽기 실패, 기본 메시지 표시')
        }
        
        return `파일 "${file.name}"이 업로드되었습니다.
        
파일 정보:
- 크기: ${(file.size / 1024).toFixed(1)}KB
- 타입: ${file.type || '알 수 없음'}
- 수정일: ${new Date(file.lastModified).toLocaleString('ko-KR')}

지원되는 형식: TXT, JSON, MD, CSV, XML
PDF는 현재 시뮬레이션 모드입니다.`
      }
    } catch (error) {
      console.error('파일 처리 오류:', error)
      return `파일 "${file.name}" 처리 중 오류가 발생했습니다: ${error}`
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      console.log('파일 업로드 시작:', file.name, file.type, file.size)
      setSelectedFile(file)
      setIsProcessing(true)
      
      // 타임아웃 설정 (30초)
      const timeoutId = setTimeout(() => {
        console.error('파일 처리 타임아웃')
        setIsProcessing(false)
        setExtractedText('파일 처리가 너무 오래 걸립니다. 다시 시도해주세요.')
        setStep('viewer')
      }, 30000)
      
      try {
        console.log('텍스트 추출 시작...')
        const text = await extractTextFromFile(file)
        console.log('텍스트 추출 완료:', text.length, '자')
        clearTimeout(timeoutId)
        setExtractedText(text)
        setStep('viewer')
      } catch (error) {
        console.error('파일 처리 오류:', error)
        clearTimeout(timeoutId)
        setExtractedText(`파일 "${file.name}" 처리 중 오류가 발생했습니다.\n\n${error}`)
        setStep('viewer')
      } finally {
        setIsProcessing(false)
      }
    }
  }

  const performSemanticSearch = (text: string, query: string) => {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10)
    const queryLower = query.toLowerCase()
    
    // 간단한 키워드 매칭 기반 유사도 검색 시뮬레이션
    const matches = sentences.map((sentence, index) => {
      const sentenceLower = sentence.toLowerCase()
      let score = 0
      
      // 키워드 매칭 점수 계산
      const queryWords = queryLower.split(/\s+/)
      queryWords.forEach(word => {
        if (word.length > 2 && sentenceLower.includes(word)) {
          score += 1
        }
      })
      
      // 문장 길이와 위치 가중치
      score = score / Math.max(queryWords.length, 1)
      if (sentence.length > 50) score *= 1.2
      if (index < sentences.length / 3) score *= 1.1 // 앞부분 가중치
      
      return {
        id: index + 1,
        content: sentence.trim(),
        score: Math.min(score + Math.random() * 0.1, 1), // 약간의 랜덤 요소
        source: selectedFile?.name || "문서",
        page: Math.floor(index / 3) + 1
      }
    })
    
    // 점수순으로 정렬하고 상위 3개 반환
    return matches
      .filter(match => match.score > 0.1)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
  }

  const generateAnswer = (chunks: any[], query: string) => {
    if (chunks.length === 0) {
      return `"${query}"에 대한 정보를 문서에서 찾을 수 없습니다. 다른 질문을 시도해보세요.`
    }
    
    const topChunk = chunks[0]
    const context = chunks.map(c => c.content).join(' ')
    
    // 간단한 답변 생성 시뮬레이션
    if (query.includes('RAG') || query.includes('rag')) {
      return `${query}에 대한 답변: ${topChunk.content} 업로드하신 문서에 따르면, RAG 시스템은 검색과 생성을 결합하여 더 정확한 답변을 제공합니다.`
    } else if (query.includes('어떻게') || query.includes('방법')) {
      return `${query}에 대한 답변: 문서 내용을 바탕으로 말씀드리면, ${topChunk.content} 이러한 방식으로 구현할 수 있습니다.`
    } else {
      return `${query}에 대한 답변: 문서에서 찾은 관련 정보는 다음과 같습니다: ${topChunk.content} 추가로 필요한 정보가 있으시면 더 구체적으로 질문해주세요.`
    }
  }

  const handleQuery = async () => {
    if (!query.trim() || !extractedText) return
    
    setIsProcessing(true)
    
    // 실제 텍스트에서 검색 수행
    setTimeout(() => {
      const chunks = performSemanticSearch(extractedText, query)
      const answer = generateAnswer(chunks, query)
      
      // 채팅 히스토리에 추가
      const newChat = {
        question: query,
        answer,
        chunks
      }
      
      setChatHistory(prev => [...prev, newChat])
      setQuery('')
      setIsProcessing(false)
    }, 1000 + Math.random() * 500)
  }

  const reset = () => {
    setSelectedFile(null)
    setExtractedText('')
    setQuery('')
    setResults(null)
    setChatHistory([])
    setStep('upload')
    setProcessingMessage('')
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/rag/beginner"
              className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 transition-colors"
            >
              <ArrowLeft size={20} />
              초급 과정으로 돌아가기
            </Link>
            <div className="h-6 border-l border-gray-300 dark:border-gray-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                RAG 플레이그라운드
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                문서를 업로드하고 실시간으로 질문해보세요
              </p>
            </div>
            {selectedFile && (
              <div className="ml-auto">
                <button
                  onClick={reset}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 rounded-lg transition-colors"
                >
                  새 문서
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {step === 'upload' ? (
          /* Upload Screen */
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="max-w-md w-full">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto bg-emerald-100 dark:bg-emerald-900/20 rounded-xl flex items-center justify-center mb-4">
                  {isProcessing ? (
                    <RefreshCw className="text-emerald-600 animate-spin" size={32} />
                  ) : (
                    <FileText className="text-emerald-600" size={32} />
                  )}
                </div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                  {isProcessing ? "문서를 처리하는 중..." : "문서를 업로드하세요"}
                </h2>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  {isProcessing 
                    ? processingMessage || "파일에서 텍스트를 추출하고 있습니다. 잠시만 기다려주세요."
                    : "PDF, TXT, JSON, MD, CSV, XML 파일을 업로드하여 RAG 시스템을 테스트할 수 있습니다"
                  }
                </p>

                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  accept=".txt,.json,.md,.csv,.xml,.pdf"
                  onChange={handleFileUpload}
                  disabled={isProcessing}
                />
                <label
                  htmlFor="file-upload"
                  className={`block w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                    isProcessing 
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-emerald-500 hover:bg-emerald-600 cursor-pointer'
                  } text-white`}
                >
                  {isProcessing ? "처리 중..." : "파일 선택하기"}
                </label>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                  ✅ 지원 형식: PDF, TXT, JSON, MD, CSV, XML<br/>
                  📄 PDF는 서버에서 안전하게 처리됩니다
                </p>
                
                <div className="mt-4 p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg">
                  <p className="text-sm text-emerald-700 dark:text-emerald-300 mb-2">
                    💡 처음이신가요? 샘플 문서로 시작해보세요!
                  </p>
                  <a
                    href="/sample-rag-document.txt"
                    download="RAG-시스템-가이드.txt"
                    className="inline-flex items-center gap-2 text-sm text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    RAG 가이드 샘플 문서 다운로드
                  </a>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Document Viewer + Chat Interface */
          <div className="flex-1 flex h-screen">
            {/* Left Panel - Document Viewer */}
            <div className="w-1/2 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-3">
                  <FileText className="text-emerald-600" size={20} />
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900 dark:text-white truncate">
                      {selectedFile?.name}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {extractedText.length.toLocaleString()}자, {extractedText.split(/\s+/).length.toLocaleString()}단어
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="flex-1 overflow-auto p-4">
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <pre className="whitespace-pre-wrap text-sm leading-relaxed text-gray-700 dark:text-gray-300 font-mono">
                    {extractedText}
                  </pre>
                </div>
              </div>
            </div>

            {/* Right Panel - Chat Interface */}
            <div className="w-1/2 bg-gray-50 dark:bg-gray-900 flex flex-col">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                <div className="flex items-center gap-2">
                  <Brain className="text-emerald-600" size={20} />
                  <h3 className="font-semibold text-gray-900 dark:text-white">RAG 질의응답</h3>
                  <span className="ml-auto text-xs text-gray-500 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                    {chatHistory.length}개 대화
                  </span>
                </div>
              </div>

              {/* Chat History */}
              <div className="flex-1 overflow-auto p-4 space-y-4">
                {chatHistory.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-12 h-12 mx-auto bg-emerald-100 dark:bg-emerald-900/20 rounded-lg flex items-center justify-center mb-4">
                      <Brain className="text-emerald-600" size={24} />
                    </div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">문서에 대해 질문해보세요</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      업로드된 문서의 내용을 바탕으로 정확한 답변을 제공합니다
                    </p>
                  </div>
                ) : (
                  chatHistory.map((chat, index) => (
                    <div key={index} className="space-y-3">
                      {/* User Question */}
                      <div className="flex justify-end">
                        <div className="max-w-xs bg-emerald-500 text-white rounded-lg px-4 py-2">
                          <p className="text-sm">{chat.question}</p>
                        </div>
                      </div>
                      
                      {/* AI Answer */}
                      <div className="flex justify-start">
                        <div className="max-w-sm bg-white dark:bg-gray-800 rounded-lg px-4 py-3 shadow-sm border border-gray-200 dark:border-gray-700">
                          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                            {chat.answer}
                          </p>
                          
                          {chat.chunks.length > 0 && (
                            <div className="space-y-2">
                              <p className="text-xs text-gray-500 font-medium">📄 참조된 내용:</p>
                              {chat.chunks.slice(0, 2).map((chunk, chunkIndex) => (
                                <div key={chunkIndex} className="text-xs bg-gray-50 dark:bg-gray-700 p-2 rounded border-l-2 border-emerald-500">
                                  <p className="text-gray-600 dark:text-gray-400 line-clamp-2">
                                    {chunk.content}
                                  </p>
                                  <span className="text-gray-500 text-xs">
                                    유사도: {(chunk.score * 100).toFixed(0)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Chat Input */}
              <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
                    placeholder="문서에 대해 질문해보세요..."
                    className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    disabled={isProcessing}
                  />
                  <button
                    onClick={handleQuery}
                    disabled={!query.trim() || isProcessing}
                    className="px-4 py-2 bg-emerald-500 text-white rounded-lg font-medium hover:bg-emerald-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                  >
                    {isProcessing ? (
                      <RefreshCw size={16} className="animate-spin" />
                    ) : (
                      <Zap size={16} />
                    )}
                    {isProcessing ? '처리중' : '질문'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}