'use client'

import Link from 'next/link'
import { useState, useCallback } from 'react'
import { ArrowLeft, Upload, FileText, CheckCircle, AlertCircle, Eye, Scissors } from 'lucide-react'

export default function DocumentUploaderPage() {
  const [files, setFiles] = useState<File[]>([])
  const [processing, setProcessing] = useState<{ [key: string]: boolean }>({})
  const [results, setResults] = useState<{ [key: string]: any }>({})
  const [dragActive, setDragActive] = useState(false)

  const supportedFormats = [
    { ext: 'PDF', color: 'bg-red-500', description: 'PDF 문서' },
    { ext: 'TXT', color: 'bg-gray-500', description: '텍스트 파일' },
    { ext: 'DOC', color: 'bg-blue-500', description: 'Word 문서' },
    { ext: 'HTML', color: 'bg-orange-500', description: '웹 페이지' }
  ]

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const newFiles = Array.from(e.dataTransfer.files)
      setFiles(prev => [...prev, ...newFiles])
    }
  }, [])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      setFiles(prev => [...prev, ...newFiles])
    }
  }

  const processFile = async (file: File) => {
    const fileId = file.name + file.size
    setProcessing(prev => ({ ...prev, [fileId]: true }))
    
    // 시뮬레이션: 파일 처리
    setTimeout(() => {
      const mockResult = {
        fileName: file.name,
        fileSize: (file.size / 1024).toFixed(1) + ' KB',
        fileType: file.type || 'unknown',
        extractedText: generateMockExtractedText(file.name),
        metadata: {
          pages: Math.floor(Math.random() * 20) + 1,
          words: Math.floor(Math.random() * 5000) + 500,
          characters: Math.floor(Math.random() * 25000) + 2500,
          createdAt: new Date().toLocaleDateString('ko-KR'),
          language: 'Korean'
        },
        chunks: generateMockChunks(file.name)
      }
      
      setResults(prev => ({ ...prev, [fileId]: mockResult }))
      setProcessing(prev => ({ ...prev, [fileId]: false }))
    }, 2000 + Math.random() * 1000)
  }

  const generateMockExtractedText = (fileName: string) => {
    const samples = [
      `${fileName}에서 추출된 텍스트입니다. RAG(Retrieval-Augmented Generation)는 대규모 언어 모델의 성능을 크게 향상시키는 혁신적인 기술입니다. 이 기술은 외부 지식 베이스에서 관련 정보를 검색하여 생성 과정에 활용합니다...`,
      `문서 내용: 인공지능과 머신러닝의 발전으로 자연어 처리 분야에서 놀라운 성과를 거두고 있습니다. 특히 Transformer 아키텍처 기반의 언어 모델들이 다양한 태스크에서 인간 수준의 성능을 보여주고 있습니다...`,
      `주요 내용: 벡터 데이터베이스는 고차원 벡터 공간에서의 유사도 검색을 효율적으로 수행할 수 있는 특화된 데이터베이스입니다. FAISS, Pinecone, Chroma 등이 대표적인 솔루션으로 널리 사용되고 있습니다...`
    ]
    return samples[Math.floor(Math.random() * samples.length)]
  }

  const generateMockChunks = (fileName: string) => {
    const chunkCount = Math.floor(Math.random() * 5) + 3
    return Array.from({ length: chunkCount }, (_, i) => ({
      id: i + 1,
      size: Math.floor(Math.random() * 800) + 200,
      preview: `청크 ${i + 1}: ${fileName}에서 추출된 내용의 일부입니다. 이 청크는 의미적으로 완결된 단위로 분할되었으며...`
    }))
  }

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
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
                문서 업로더 시뮬레이터
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                다양한 문서 형식의 텍스트 추출과 전처리 과정을 체험하세요
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-8">
        {/* Supported Formats */}
        <div className="mb-8">
          <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">지원 형식</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {supportedFormats.map((format) => (
              <div key={format.ext} className="flex items-center gap-3 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className={`w-8 h-8 ${format.color} rounded flex items-center justify-center text-white text-xs font-bold`}>
                  {format.ext}
                </div>
                <div>
                  <p className="font-medium text-gray-900 dark:text-white text-sm">{format.ext}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{format.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Upload Area */}
        <div className="mb-8">
          <div
            className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
              dragActive
                ? 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20'
                : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="w-16 h-16 mx-auto bg-emerald-100 dark:bg-emerald-900/20 rounded-xl flex items-center justify-center mb-4">
              <Upload className="text-emerald-600" size={32} />
            </div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              파일을 드래그하거나 클릭하여 업로드
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              PDF, TXT, DOC, HTML 파일을 지원합니다 (최대 10MB)
            </p>
            
            <input
              type="file"
              id="file-input"
              className="hidden"
              multiple
              accept=".pdf,.txt,.doc,.docx,.html,.htm"
              onChange={handleFileInput}
            />
            <label
              htmlFor="file-input"
              className="inline-flex items-center gap-2 px-6 py-3 bg-emerald-500 text-white rounded-lg font-medium hover:bg-emerald-600 cursor-pointer transition-colors"
            >
              <Upload size={16} />
              파일 선택
            </label>
          </div>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="space-y-6">
            <h2 className="text-lg font-bold text-gray-900 dark:text-white">
              업로드된 파일 ({files.length})
            </h2>
            
            <div className="grid gap-6">
              {files.map((file, index) => {
                const fileId = file.name + file.size
                const isProcessing = processing[fileId]
                const result = results[fileId]
                
                return (
                  <div key={index} className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
                    {/* File Header */}
                    <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/20 rounded-lg flex items-center justify-center">
                            <FileText className="text-blue-600" size={20} />
                          </div>
                          <div>
                            <h3 className="font-semibold text-gray-900 dark:text-white">{file.name}</h3>
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              {(file.size / 1024).toFixed(1)} KB • {file.type || 'unknown'}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-3">
                          {!result && !isProcessing && (
                            <button
                              onClick={() => processFile(file)}
                              className="px-4 py-2 bg-emerald-500 text-white rounded-lg font-medium hover:bg-emerald-600 transition-colors"
                            >
                              처리하기
                            </button>
                          )}
                          
                          <button
                            onClick={() => removeFile(index)}
                            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                          >
                            ✕
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Processing Status */}
                    {isProcessing && (
                      <div className="p-6 bg-amber-50 dark:bg-amber-900/20 border-b border-gray-200 dark:border-gray-700">
                        <div className="flex items-center gap-3">
                          <div className="w-5 h-5 border-2 border-amber-600 border-t-transparent rounded-full animate-spin"></div>
                          <span className="text-amber-700 dark:text-amber-300 font-medium">
                            문서를 처리하는 중입니다...
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Results */}
                    {result && (
                      <div className="p-6 space-y-6">
                        {/* Success Status */}
                        <div className="flex items-center gap-3 p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-700">
                          <CheckCircle className="text-emerald-600" size={20} />
                          <span className="text-emerald-700 dark:text-emerald-300 font-medium">
                            처리 완료
                          </span>
                        </div>

                        {/* Metadata */}
                        <div className="grid md:grid-cols-2 gap-6">
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">문서 정보</h4>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-600 dark:text-gray-400">페이지 수:</span>
                                <span className="font-medium text-gray-900 dark:text-white">{result.metadata.pages}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600 dark:text-gray-400">단어 수:</span>
                                <span className="font-medium text-gray-900 dark:text-white">{result.metadata.words.toLocaleString()}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600 dark:text-gray-400">문자 수:</span>
                                <span className="font-medium text-gray-900 dark:text-white">{result.metadata.characters.toLocaleString()}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600 dark:text-gray-400">언어:</span>
                                <span className="font-medium text-gray-900 dark:text-white">{result.metadata.language}</span>
                              </div>
                            </div>
                          </div>

                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">청킹 결과</h4>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-600 dark:text-gray-400">청크 수:</span>
                                <span className="font-medium text-gray-900 dark:text-white">{result.chunks.length}개</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600 dark:text-gray-400">평균 크기:</span>
                                <span className="font-medium text-gray-900 dark:text-white">
                                  {Math.round(result.chunks.reduce((a: number, b: any) => a + b.size, 0) / result.chunks.length)}자
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Extracted Text Preview */}
                        <div>
                          <div className="flex items-center gap-2 mb-3">
                            <Eye className="text-gray-600" size={16} />
                            <h4 className="font-semibold text-gray-900 dark:text-white">추출된 텍스트 미리보기</h4>
                          </div>
                          <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
                            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                              {result.extractedText}
                            </p>
                          </div>
                        </div>

                        {/* Chunks Preview */}
                        <div>
                          <div className="flex items-center gap-2 mb-3">
                            <Scissors className="text-gray-600" size={16} />
                            <h4 className="font-semibold text-gray-900 dark:text-white">청크 분할 결과</h4>
                          </div>
                          <div className="space-y-3">
                            {result.chunks.map((chunk: any) => (
                              <div key={chunk.id} className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-700">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                                    청크 #{chunk.id}
                                  </span>
                                  <span className="text-xs text-blue-600 dark:text-blue-300 bg-blue-100 dark:bg-blue-800/30 px-2 py-1 rounded">
                                    {chunk.size}자
                                  </span>
                                </div>
                                <p className="text-sm text-blue-700 dark:text-blue-300">
                                  {chunk.preview}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Help Text */}
        {files.length === 0 && (
          <div className="mt-12 bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-700">
            <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-2">💡 시뮬레이터 사용 방법</h3>
            <ul className="text-blue-700 dark:text-blue-300 text-sm space-y-1">
              <li>1. 위의 업로드 영역에 파일을 드래그하거나 클릭하여 파일을 선택하세요</li>
              <li>2. 업로드된 파일의 "처리하기" 버튼을 클릭하세요</li>
              <li>3. 문서에서 추출된 텍스트와 청킹 결과를 확인해보세요</li>
              <li>4. 여러 파일을 동시에 업로드하여 비교할 수 있습니다</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}