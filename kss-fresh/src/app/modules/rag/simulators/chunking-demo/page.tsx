'use client'

import Link from 'next/link'
import { useState } from 'react'
import { ArrowLeft, Scissors, BarChart3, RefreshCw, Settings, Eye } from 'lucide-react'

export default function ChunkingDemoPage() {
  const [inputText, setInputText] = useState(`인공지능과 머신러닝의 발전이 가속화되면서 자연어 처리 분야에서 놀라운 성과를 거두고 있습니다. 

특히 Transformer 아키텍처를 기반으로 한 대규모 언어 모델들이 다양한 태스크에서 인간 수준의 성능을 보여주고 있습니다. 

RAG(Retrieval-Augmented Generation)는 이러한 언어 모델의 한계를 극복하기 위해 개발된 혁신적인 접근법입니다. 

외부 지식 베이스에서 관련 정보를 검색하여 생성 과정에 활용함으로써 더욱 정확하고 신뢰할 수 있는 답변을 제공합니다.

벡터 데이터베이스는 고차원 벡터 공간에서의 유사도 검색을 효율적으로 수행할 수 있는 특화된 데이터베이스입니다. 

FAISS, Pinecone, Chroma 등이 대표적인 솔루션으로 널리 사용되고 있으며, 각각 고유한 특성과 장단점을 가지고 있습니다.`)

  const [selectedStrategy, setSelectedStrategy] = useState<'fixed' | 'semantic' | 'recursive' | 'paragraph' | 'sentence'>('fixed')
  const [chunkSize, setChunkSize] = useState(200)
  const [overlap, setOverlap] = useState(50)
  const [results, setResults] = useState<any>(null)

  const strategies = {
    fixed: {
      name: '고정 크기 청킹',
      description: '일정한 문자 수로 텍스트를 분할',
      color: 'bg-blue-500',
      icon: BarChart3
    },
    semantic: {
      name: '의미 단위 청킹',
      description: '의미적 유사성을 기반으로 분할',
      color: 'bg-green-500',
      icon: RefreshCw
    },
    recursive: {
      name: '재귀적 청킹',
      description: '계층적 구분자를 사용하여 분할',
      color: 'bg-purple-500',
      icon: Settings
    },
    paragraph: {
      name: '문단 기반 청킹',
      description: '문단 단위로 텍스트를 분할',
      color: 'bg-orange-500',
      icon: Eye
    },
    sentence: {
      name: '문장 기반 청킹',
      description: '문장 단위로 텍스트를 분할',
      color: 'bg-red-500',
      icon: Scissors
    }
  }

  const executeChunking = () => {
    let chunks: any[] = []
    
    switch (selectedStrategy) {
      case 'fixed':
        chunks = fixedSizeChunking(inputText, chunkSize, overlap)
        break
      case 'semantic':
        chunks = semanticChunking(inputText)
        break
      case 'recursive':
        chunks = recursiveChunking(inputText)
        break
      case 'paragraph':
        chunks = paragraphChunking(inputText)
        break
      case 'sentence':
        chunks = sentenceChunking(inputText)
        break
    }

    setResults({
      strategy: selectedStrategy,
      chunks,
      totalChunks: chunks.length,
      avgChunkSize: Math.round(chunks.reduce((sum, chunk) => sum + chunk.content.length, 0) / chunks.length),
      minChunkSize: Math.min(...chunks.map(chunk => chunk.content.length)),
      maxChunkSize: Math.max(...chunks.map(chunk => chunk.content.length))
    })
  }

  const fixedSizeChunking = (text: string, size: number, overlapSize: number) => {
    const chunks = []
    let start = 0
    let chunkId = 1

    while (start < text.length) {
      let end = start + size
      
      // 단어 중간에서 잘리지 않도록 조정
      if (end < text.length) {
        const lastSpace = text.lastIndexOf(' ', end)
        if (lastSpace > start) {
          end = lastSpace
        }
      }

      const content = text.slice(start, end).trim()
      if (content) {
        chunks.push({
          id: chunkId++,
          content,
          start,
          end,
          size: content.length,
          type: 'fixed'
        })
      }

      start = Math.max(start + 1, end - overlapSize)
    }

    return chunks
  }

  const semanticChunking = (text: string) => {
    // 간단한 의미 단위 시뮬레이션 (문단 + 의미적 유사성 고려)
    const paragraphs = text.split('\n\n').filter(p => p.trim())
    const chunks = []
    let chunkId = 1

    for (let i = 0; i < paragraphs.length; i++) {
      const para = paragraphs[i].trim()
      if (para) {
        // 긴 문단은 문장 단위로 분할
        if (para.length > 300) {
          const sentences = para.split(/[.!?]/).filter(s => s.trim())
          let currentChunk = ''
          
          for (const sentence of sentences) {
            if (currentChunk.length + sentence.length > 250 && currentChunk) {
              chunks.push({
                id: chunkId++,
                content: currentChunk.trim() + '.',
                size: currentChunk.length,
                type: 'semantic',
                similarity: Math.random() * 0.3 + 0.7 // 0.7-1.0
              })
              currentChunk = sentence
            } else {
              currentChunk += sentence + '.'
            }
          }
          
          if (currentChunk.trim()) {
            chunks.push({
              id: chunkId++,
              content: currentChunk.trim(),
              size: currentChunk.length,
              type: 'semantic',
              similarity: Math.random() * 0.3 + 0.7
            })
          }
        } else {
          chunks.push({
            id: chunkId++,
            content: para,
            size: para.length,
            type: 'semantic',
            similarity: Math.random() * 0.3 + 0.7
          })
        }
      }
    }

    return chunks
  }

  const recursiveChunking = (text: string) => {
    const separators = ['\n\n', '\n', '. ', ' ', '']
    const chunks = []
    let chunkId = 1

    const splitRecursively = (text: string, separators: string[], targetSize: number = 200): string[] => {
      if (text.length <= targetSize || separators.length === 0) {
        return [text]
      }

      const separator = separators[0]
      const parts = text.split(separator)
      const result: string[] = []

      for (const part of parts) {
        if (part.length <= targetSize) {
          result.push(part)
        } else {
          result.push(...splitRecursively(part, separators.slice(1), targetSize))
        }
      }

      return result
    }

    const splitTexts = splitRecursively(inputText, separators, chunkSize)
    
    splitTexts.forEach((text, index) => {
      if (text.trim()) {
        chunks.push({
          id: chunkId++,
          content: text.trim(),
          size: text.length,
          type: 'recursive',
          level: 0 // 실제로는 재귀 레벨을 추적
        })
      }
    })

    return chunks
  }

  const paragraphChunking = (text: string) => {
    const paragraphs = text.split('\n\n').filter(p => p.trim())
    return paragraphs.map((para, index) => ({
      id: index + 1,
      content: para.trim(),
      size: para.length,
      type: 'paragraph'
    }))
  }

  const sentenceChunking = (text: string) => {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim())
    return sentences.map((sentence, index) => ({
      id: index + 1,
      content: sentence.trim() + (index < sentences.length - 1 ? '.' : ''),
      size: sentence.length,
      type: 'sentence'
    }))
  }

  const getChunkColor = (strategy: string, index: number) => {
    const colors = ['bg-blue-100', 'bg-green-100', 'bg-yellow-100', 'bg-purple-100', 'bg-pink-100', 'bg-indigo-100']
    return colors[index % colors.length]
  }

  const resetText = () => {
    setInputText(`인공지능과 머신러닝의 발전이 가속화되면서 자연어 처리 분야에서 놀라운 성과를 거두고 있습니다. 

특히 Transformer 아키텍처를 기반으로 한 대규모 언어 모델들이 다양한 태스크에서 인간 수준의 성능을 보여주고 있습니다. 

RAG(Retrieval-Augmented Generation)는 이러한 언어 모델의 한계를 극복하기 위해 개발된 혁신적인 접근법입니다. 

외부 지식 베이스에서 관련 정보를 검색하여 생성 과정에 활용함으로써 더욱 정확하고 신뢰할 수 있는 답변을 제공합니다.

벡터 데이터베이스는 고차원 벡터 공간에서의 유사도 검색을 효율적으로 수행할 수 있는 특화된 데이터베이스입니다. 

FAISS, Pinecone, Chroma 등이 대표적인 솔루션으로 널리 사용되고 있으며, 각각 고유한 특성과 장단점을 가지고 있습니다.`)
    setResults(null)
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
                청킹 데모 - 5가지 전략 비교
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                다양한 청킹 전략을 실시간으로 비교하고 결과를 확인하세요
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Panel: Input & Settings */}
          <div className="lg:col-span-1 space-y-6">
            {/* Input Text */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">입력 텍스트</h2>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full h-64 p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm leading-relaxed resize-none"
                placeholder="분할할 텍스트를 입력하세요..."
              />
              <button
                onClick={resetText}
                className="mt-2 text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              >
                예시 텍스트로 리셋
              </button>
            </div>

            {/* Strategy Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">청킹 전략</h2>
              <div className="space-y-3">
                {Object.entries(strategies).map(([key, strategy]) => {
                  const IconComponent = strategy.icon
                  return (
                    <label key={key} className="flex items-center gap-3 cursor-pointer">
                      <input
                        type="radio"
                        name="strategy"
                        value={key}
                        checked={selectedStrategy === key}
                        onChange={(e) => setSelectedStrategy(e.target.value as any)}
                        className="text-emerald-500"
                      />
                      <div className={`w-8 h-8 ${strategy.color} rounded flex items-center justify-center`}>
                        <IconComponent size={16} className="text-white" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white text-sm">{strategy.name}</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">{strategy.description}</p>
                      </div>
                    </label>
                  )
                })}
              </div>
            </div>

            {/* Settings */}
            {(selectedStrategy === 'fixed' || selectedStrategy === 'recursive') && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
                <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">설정</h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      청크 크기: {chunkSize}자
                    </label>
                    <input
                      type="range"
                      min="50"
                      max="500"
                      value={chunkSize}
                      onChange={(e) => setChunkSize(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      중첩 크기: {overlap}자
                    </label>
                    <input
                      type="range"
                      min="0"
                      max={Math.floor(chunkSize * 0.5)}
                      value={overlap}
                      onChange={(e) => setOverlap(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Execute Button */}
            <button
              onClick={executeChunking}
              disabled={!inputText.trim()}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-emerald-500 text-white rounded-lg font-medium hover:bg-emerald-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <Scissors size={16} />
              청킹 실행
            </button>
          </div>

          {/* Right Panel: Results */}
          <div className="lg:col-span-2">
            {results ? (
              <div className="space-y-6">
                {/* Stats */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
                  <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    {strategies[results.strategy as keyof typeof strategies].name} 결과
                  </h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">{results.totalChunks}</div>
                      <div className="text-xs text-blue-600">총 청크 수</div>
                    </div>
                    <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-green-600">{results.avgChunkSize}</div>
                      <div className="text-xs text-green-600">평균 크기</div>
                    </div>
                    <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-orange-600">{results.minChunkSize}</div>
                      <div className="text-xs text-orange-600">최소 크기</div>
                    </div>
                    <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">{results.maxChunkSize}</div>
                      <div className="text-xs text-purple-600">최대 크기</div>
                    </div>
                  </div>
                </div>

                {/* Chunks */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    청크 결과 ({results.totalChunks}개)
                  </h3>
                  <div className="space-y-4 max-h-[600px] overflow-y-auto">
                    {results.chunks.map((chunk: any, index: number) => (
                      <div
                        key={chunk.id}
                        className={`p-4 rounded-lg border ${getChunkColor(results.strategy, index)} border-gray-200 dark:border-gray-600`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-900 dark:text-white">
                            청크 #{chunk.id}
                          </span>
                          <div className="flex items-center gap-3">
                            {chunk.similarity && (
                              <span className="text-xs text-blue-600 bg-blue-100 dark:bg-blue-800/30 px-2 py-1 rounded">
                                유사도: {(chunk.similarity * 100).toFixed(0)}%
                              </span>
                            )}
                            <span className="text-xs text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                              {chunk.size}자
                            </span>
                          </div>
                        </div>
                        <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                          {chunk.content}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-12 border border-gray-200 dark:border-gray-700 text-center">
                <div className="w-16 h-16 mx-auto bg-gray-100 dark:bg-gray-700 rounded-xl flex items-center justify-center mb-4">
                  <Scissors className="text-gray-400" size={32} />
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  청킹 결과가 여기에 표시됩니다
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  왼쪽에서 텍스트를 입력하고 청킹 전략을 선택한 후 "청킹 실행" 버튼을 클릭하세요
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}