'use client'

import { useState, useEffect, useRef } from 'react'
import { Search, RefreshCw, Info } from 'lucide-react'

interface TextEmbedding {
  text: string
  x: number
  y: number
  similarity?: number
}

const sampleTexts = [
  '인공지능은 미래 기술입니다',
  'AI는 혁신적인 기술입니다',
  '머신러닝과 딥러닝',
  '자연어 처리 기술',
  '컴퓨터 비전과 이미지 인식',
  '강아지는 귀여운 동물입니다',
  '고양이도 사랑스럽습니다',
  '오늘 날씨가 좋네요',
  '비가 오는 날씨입니다',
  'LLM과 RAG 시스템'
]

// 간단한 임베딩 시뮬레이션 (실제로는 임베딩 모델 사용)
const generateMockEmbedding = (text: string): [number, number] => {
  // 텍스트 특성에 따라 2D 좌표 생성
  let x = 0, y = 0
  
  // AI/기술 관련 키워드
  if (text.includes('인공지능') || text.includes('AI') || text.includes('머신러닝') || text.includes('딥러닝')) {
    x = Math.random() * 100 - 200
    y = Math.random() * 100 - 200
  }
  // 자연어/컴퓨터비전 관련
  else if (text.includes('자연어') || text.includes('컴퓨터 비전') || text.includes('LLM') || text.includes('RAG')) {
    x = Math.random() * 100 + 100
    y = Math.random() * 100 - 200
  }
  // 동물 관련
  else if (text.includes('강아지') || text.includes('고양이') || text.includes('동물')) {
    x = Math.random() * 100 - 200
    y = Math.random() * 100 + 100
  }
  // 날씨 관련
  else if (text.includes('날씨') || text.includes('비')) {
    x = Math.random() * 100 + 100
    y = Math.random() * 100 + 100
  }
  // 기타
  else {
    x = Math.random() * 400 - 200
    y = Math.random() * 400 - 200
  }
  
  return [x, y]
}

// 코사인 유사도 계산 시뮬레이션
const calculateSimilarity = (text1: string, text2: string): number => {
  const words1 = new Set(text1.split(' '))
  const words2 = new Set(text2.split(' '))
  
  const intersection = new Set(Array.from(words1).filter(x => words2.has(x)))
  const union = new Set([...Array.from(words1), ...Array.from(words2)])
  
  return intersection.size / union.size
}

export default function EmbeddingVisualizer() {
  const [embeddings, setEmbeddings] = useState<TextEmbedding[]>([])
  const [queryText, setQueryText] = useState('')
  const [selectedText, setSelectedText] = useState<string | null>(null)
  const [animationFrame, setAnimationFrame] = useState(0)
  const [hoveredPoint, setHoveredPoint] = useState<string | null>(null)
  const [isAnimating, setIsAnimating] = useState(false)
  const [connectionLines, setConnectionLines] = useState<Array<{from: string, to: string, strength: number}>>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const backgroundCanvasRef = useRef<HTMLCanvasElement>(null)
  const particleCanvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    // 초기 임베딩 생성
    const initialEmbeddings = sampleTexts.map(text => {
      const [x, y] = generateMockEmbedding(text)
      return { text, x, y }
    })
    setEmbeddings(initialEmbeddings)
  }, [])

  // 애니메이션 루프
  useEffect(() => {
    const animate = () => {
      setAnimationFrame(prev => prev + 1)
      drawBackground()
      drawEmbeddings()
      drawParticles()
      animationRef.current = requestAnimationFrame(animate)
    }
    
    animate()
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [embeddings, selectedText, queryText, hoveredPoint, isAnimating])
  
  // 캔버스 마우스 이벤트
  const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const mouseX = event.clientX - rect.left
    const mouseY = event.clientY - rect.top
    
    // 스케일 조정
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const scaledX = mouseX * scaleX
    const scaledY = mouseY * scaleY
    
    let hoveredText = null
    
    embeddings.forEach(embedding => {
      const x = embedding.x + canvas.width / 2
      const y = embedding.y + canvas.height / 2
      const distance = Math.sqrt((scaledX - x) ** 2 + (scaledY - y) ** 2)
      
      if (distance < 20) {
        hoveredText = embedding.text
      }
    })
    
    setHoveredPoint(hoveredText)
  }
  
  const handleCanvasMouseLeave = () => {
    setHoveredPoint(null)
  }
  
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const mouseX = event.clientX - rect.left
    const mouseY = event.clientY - rect.top
    
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const scaledX = mouseX * scaleX
    const scaledY = mouseY * scaleY
    
    embeddings.forEach(embedding => {
      const x = embedding.x + canvas.width / 2
      const y = embedding.y + canvas.height / 2
      const distance = Math.sqrt((scaledX - x) ** 2 + (scaledY - y) ** 2)
      
      if (distance < 20) {
        setSelectedText(embedding.text)
      }
    })
  }

  // 배경 그리드 그리기 (별도 캔버스)
  const drawBackground = () => {
    const canvas = backgroundCanvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const isDarkMode = document.documentElement.classList.contains('dark')
    
    // 동적 배경 그리드
    ctx.strokeStyle = isDarkMode ? '#374151' : '#e5e7eb'
    ctx.lineWidth = 1
    
    // 애니메이션에 따른 그리드 이동
    const offset = (animationFrame * 0.5) % 50
    
    for (let i = -offset; i <= canvas.width + 50; i += 50) {
      ctx.globalAlpha = 0.3 + 0.2 * Math.sin(animationFrame * 0.02 + i * 0.01)
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, canvas.height)
      ctx.stroke()
    }
    
    for (let i = -offset; i <= canvas.height + 50; i += 50) {
      ctx.globalAlpha = 0.3 + 0.2 * Math.sin(animationFrame * 0.02 + i * 0.01)
      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(canvas.width, i)
      ctx.stroke()
    }
    
    ctx.globalAlpha = 1

    // 중심선 (강조된 스타일)
    ctx.strokeStyle = isDarkMode ? '#6b7280' : '#9ca3af'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(canvas.width / 2, 0)
    ctx.lineTo(canvas.width / 2, canvas.height)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(0, canvas.height / 2)
    ctx.lineTo(canvas.width, canvas.height / 2)
    ctx.stroke()
  }
  
  // 파티클 시스템 그리기
  const drawParticles = () => {
    const canvas = particleCanvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    if (!isAnimating && !hoveredPoint) return
    
    // 호버된 점 주변에 파티클 생성
    const hoveredEmbedding = embeddings.find(e => e.text === hoveredPoint)
    if (hoveredEmbedding) {
      const centerX = hoveredEmbedding.x + canvas.width / 2
      const centerY = hoveredEmbedding.y + canvas.height / 2
      
      for (let i = 0; i < 20; i++) {
        const angle = (animationFrame * 0.02 + i * 0.314) % (Math.PI * 2)
        const radius = 30 + 20 * Math.sin(animationFrame * 0.03 + i)
        const x = centerX + Math.cos(angle) * radius
        const y = centerY + Math.sin(angle) * radius
        
        ctx.beginPath()
        ctx.arc(x, y, 2, 0, 2 * Math.PI)
        ctx.fillStyle = `rgba(59, 130, 246, ${0.5 - (radius - 30) / 40})`
        ctx.fill()
      }
    }
  }
  
  // 연결선 그리기
  const drawConnections = (ctx: CanvasRenderingContext2D) => {
    if (!queryText || queryText.length < 2) return
    
    const queryEmbedding = embeddings.find(e => e.text === queryText)
    if (!queryEmbedding) return
    
    const queryX = queryEmbedding.x + ctx.canvas.width / 2
    const queryY = queryEmbedding.y + ctx.canvas.height / 2
    
    embeddings.forEach(embedding => {
      if (embedding.text === queryText) return
      
      const similarity = calculateSimilarity(queryText, embedding.text)
      if (similarity > 0.1) {
        const x = embedding.x + ctx.canvas.width / 2
        const y = embedding.y + ctx.canvas.height / 2
        
        // 연결선 애니메이션
        const alpha = similarity * (0.5 + 0.3 * Math.sin(animationFrame * 0.05))
        
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`
        ctx.lineWidth = 1 + similarity * 3
        ctx.beginPath()
        ctx.moveTo(queryX, queryY)
        ctx.lineTo(x, y)
        ctx.stroke()
        
        // 연결선 위에 파티클 효과
        const progress = (animationFrame * 0.02) % 1
        const particleX = queryX + (x - queryX) * progress
        const particleY = queryY + (y - queryY) * progress
        
        ctx.beginPath()
        ctx.arc(particleX, particleY, 2 * similarity, 0, 2 * Math.PI)
        ctx.fillStyle = `rgba(16, 185, 129, ${similarity})`
        ctx.fill()
      }
    })
  }
  
  const drawEmbeddings = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // 연결선 그리기
    drawConnections(ctx)

    // 임베딩 점들 그리기 (향상된 스타일)
    embeddings.forEach(embedding => {
      const x = embedding.x + canvas.width / 2
      const y = embedding.y + canvas.height / 2
      
      // 유사도에 따른 색상 및 크기
      let color = '#6b7280'
      let size = 6
      let glowIntensity = 0
      
      const isHovered = hoveredPoint === embedding.text
      const isSelected = selectedText === embedding.text
      
      if (queryText && queryText.length > 2) {
        const similarity = calculateSimilarity(queryText, embedding.text)
        if (similarity > 0.3) {
          color = '#10b981'
          size = 8 + similarity * 10
          glowIntensity = similarity
        } else if (similarity > 0.1) {
          color = '#f59e0b'
          size = 6 + similarity * 5
          glowIntensity = similarity * 0.5
        }
      }
      
      if (isSelected) {
        color = '#3b82f6'
        size = 12 + 2 * Math.sin(animationFrame * 0.1)
        glowIntensity = 0.8
      }
      
      if (isHovered) {
        size = size * 1.5
        glowIntensity = Math.max(glowIntensity, 0.6)
      }
      
      // 글로우 효과
      if (glowIntensity > 0) {
        ctx.shadowColor = color
        ctx.shadowBlur = 20 * glowIntensity
      }
      
      // 3D 효과를 위한 그라디언트
      const gradient = ctx.createRadialGradient(x - size/3, y - size/3, 0, x, y, size)
      const rgb = color === '#10b981' ? '16, 185, 129' : 
                  color === '#f59e0b' ? '245, 158, 11' :
                  color === '#3b82f6' ? '59, 130, 246' : '107, 114, 128'
      
      gradient.addColorStop(0, `rgba(${rgb}, 1)`)
      gradient.addColorStop(0.7, `rgba(${rgb}, 0.8)`)
      gradient.addColorStop(1, `rgba(${rgb}, 0.4)`)
      
      // 점 그리기
      ctx.beginPath()
      ctx.arc(x, y, size, 0, 2 * Math.PI)
      ctx.fillStyle = gradient
      ctx.fill()
      
      // 엣지 효과
      ctx.beginPath()
      ctx.arc(x, y, size, 0, 2 * Math.PI)
      ctx.strokeStyle = `rgba(${rgb}, 0.6)`
      ctx.lineWidth = 1
      ctx.stroke()
      
      ctx.shadowBlur = 0
      
      // 동적 텍스트 레이블
      const isDarkMode = document.documentElement.classList.contains('dark')
      ctx.fillStyle = isDarkMode ? '#e5e7eb' : '#374151'
      ctx.font = `${isSelected || isHovered ? 'bold ' : ''}12px Inter`
      
      // 텍스트 배경 (가독성 향상)
      const textWidth = ctx.measureText(embedding.text).width
      ctx.fillStyle = `rgba(${isDarkMode ? '31, 41, 55' : '255, 255, 255'}, 0.8)`
      ctx.fillRect(x + 10, y - 15, textWidth + 6, 16)
      
      ctx.fillStyle = isDarkMode ? '#e5e7eb' : '#374151'
      ctx.fillText(embedding.text, x + 13, y - 5)
    })
  }

  const handleSearch = () => {
    if (!queryText) return
    
    setIsAnimating(true)
    
    // 쿼리 텍스트의 임베딩 생성 및 추가
    const [x, y] = generateMockEmbedding(queryText)
    const newEmbedding: TextEmbedding = { text: queryText, x, y }
    
    // 기존 임베딩과의 유사도 계산
    const embeddingsWithSimilarity = embeddings.map(emb => ({
      ...emb,
      similarity: calculateSimilarity(queryText, emb.text)
    }))
    
    // 연결선 생성
    const connections = embeddingsWithSimilarity
      .filter(emb => emb.similarity && emb.similarity > 0.1)
      .map(emb => ({
        from: queryText,
        to: emb.text,
        strength: emb.similarity || 0
      }))
    
    setConnectionLines(connections)
    setEmbeddings([...embeddingsWithSimilarity, newEmbedding])
    
    // 애니메이션 종료
    setTimeout(() => setIsAnimating(false), 3000)
  }

  const resetVisualization = () => {
    const initialEmbeddings = sampleTexts.map(text => {
      const [x, y] = generateMockEmbedding(text)
      return { text, x, y }
    })
    setEmbeddings(initialEmbeddings)
    setQueryText('')
    setSelectedText(null)
  }

  return (
    <div className="space-y-6">
      {/* Search Input */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="검색할 텍스트를 입력하세요..."
            className="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
          <button
            onClick={handleSearch}
            disabled={isAnimating}
            className={`px-4 py-2 rounded-lg transition-all duration-200 flex items-center gap-2 ${
              isAnimating 
                ? 'bg-yellow-500 text-white animate-pulse' 
                : 'bg-emerald-600 text-white hover:bg-emerald-700 hover:scale-105'
            }`}
          >
            <Search size={16} className={isAnimating ? 'animate-spin' : ''} />
            {isAnimating ? '검색 중...' : '검색'}
          </button>
          <button
            onClick={resetVisualization}
            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors flex items-center gap-2"
          >
            <RefreshCw size={16} />
            초기화
          </button>
        </div>
      </div>

      {/* Visualization Canvas */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
          임베딩 공간 시각화
        </h3>
        <div className="relative">
          <canvas
            ref={backgroundCanvasRef}
            width={600}
            height={400}
            className="absolute top-0 left-0 w-full border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-900"
            style={{ zIndex: 1 }}
          />
          <canvas
            ref={canvasRef}
            width={600}
            height={400}
            className="relative w-full border border-gray-300 dark:border-gray-600 rounded-lg cursor-crosshair"
            style={{ zIndex: 2 }}
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={handleCanvasMouseLeave}
            onClick={handleCanvasClick}
          />
          <canvas
            ref={particleCanvasRef}
            width={600}
            height={400}
            className="absolute top-0 left-0 w-full border border-gray-300 dark:border-gray-600 rounded-lg pointer-events-none"
            style={{ zIndex: 3 }}
          />
          
          {/* 호버 정보 툴팁 */}
          {hoveredPoint && (
            <div className="absolute top-2 right-2 bg-black text-white px-3 py-2 rounded-lg text-sm z-10">
              호버: {hoveredPoint}
              {queryText && (
                <div className="text-xs mt-1">
                  유사도: {(calculateSimilarity(queryText, hoveredPoint) * 100).toFixed(0)}%
                </div>
              )}
            </div>
          )}
        </div>
        
        {/* Legend */}
        <div className="mt-4 flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gray-500"></div>
            <span className="text-gray-600 dark:text-gray-400">기본</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-500"></div>
            <span className="text-gray-600 dark:text-gray-400">낮은 유사도</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-emerald-500"></div>
            <span className="text-gray-600 dark:text-gray-400">높은 유사도</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500"></div>
            <span className="text-gray-600 dark:text-gray-400">선택됨</span>
          </div>
        </div>
      </div>

      {/* Text List */}
      <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6">
        <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
          텍스트 목록 (클릭하여 선택)
        </h3>
        <div className="grid md:grid-cols-2 gap-3">
          {embeddings.map((embedding, index) => (
            <button
              key={index}
              onClick={() => setSelectedText(embedding.text)}
              className={`p-3 rounded-lg text-left transition-all ${
                selectedText === embedding.text
                  ? 'bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-500'
                  : 'bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 hover:border-gray-400'
              }`}
            >
              <div className="font-medium text-gray-900 dark:text-white">
                {embedding.text}
              </div>
              {embedding.similarity !== undefined && (
                <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  유사도: {(embedding.similarity * 100).toFixed(0)}%
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Info Box */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
        <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2 flex items-center gap-2">
          <Info size={20} />
          임베딩의 작동 원리
        </h4>
        <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
          <li>• 의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치합니다</li>
          <li>• 실제 임베딩은 수백~수천 차원이지만, 여기서는 2D로 단순화했습니다</li>
          <li>• 검색 시 쿼리와 가장 가까운 벡터들을 찾아 관련 문서를 반환합니다</li>
          <li>• 거리 계산은 주로 코사인 유사도나 유클리드 거리를 사용합니다</li>
        </ul>
      </div>
    </div>
  )
}