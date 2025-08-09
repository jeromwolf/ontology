'use client'

import { useEffect, useRef, useState } from 'react'

interface Token {
  id: string
  text: string
  x: number
  y: number
  attention: number
}

interface AttentionLine {
  from: string
  to: string
  weight: number
  active: boolean
}

export default function TransformerAttentionSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [tokens, setTokens] = useState<Token[]>([])
  const [attentionLines, setAttentionLines] = useState<AttentionLine[]>([])
  const [currentToken, setCurrentToken] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [animationFrame, setAnimationFrame] = useState(0)
  const animationRef = useRef<number>()

  const sampleTokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']

  // 토큰 초기화
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const width = canvas.width = 400
    const height = canvas.height = 200

    const newTokens: Token[] = sampleTokens.map((text, index) => ({
      id: `token-${index}`,
      text,
      x: 50 + index * 55,
      y: height / 2,
      attention: 0
    }))

    const newAttentionLines: AttentionLine[] = []
    
    // 모든 토큰 간 어텐션 라인 생성
    for (let i = 0; i < sampleTokens.length; i++) {
      for (let j = 0; j < sampleTokens.length; j++) {
        if (i !== j) {
          newAttentionLines.push({
            from: `token-${i}`,
            to: `token-${j}`,
            weight: Math.random(),
            active: false
          })
        }
      }
    }

    setTokens(newTokens)
    setAttentionLines(newAttentionLines)
  }, [])

  // 애니메이션 루프
  useEffect(() => {
    const animate = () => {
      setAnimationFrame(prev => prev + 1)
      
      if (isRunning) {
        // 현재 토큰 순환
        if (animationFrame % 120 === 0) {
          setCurrentToken(prev => (prev + 1) % sampleTokens.length)
        }

        // 어텐션 가중치 업데이트
        setAttentionLines(prevLines => prevLines.map(line => {
          const isFromCurrentToken = line.from === `token-${currentToken}`
          const isToCurrentToken = line.to === `token-${currentToken}`
          
          return {
            ...line,
            active: isFromCurrentToken || isToCurrentToken,
            weight: isFromCurrentToken || isToCurrentToken 
              ? 0.3 + 0.7 * Math.sin(animationFrame * 0.08) 
              : line.weight * 0.95
          }
        }))

        // 토큰 어텐션 값 업데이트
        setTokens(prevTokens => prevTokens.map((token, index) => ({
          ...token,
          attention: index === currentToken 
            ? 1.0 
            : 0.2 + 0.6 * Math.sin(animationFrame * 0.05 + index)
        })))
      }
      
      drawAttention()
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [tokens, attentionLines, isRunning, currentToken, animationFrame])

  const drawAttention = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // 배경
    ctx.fillStyle = '#1e293b'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // 어텐션 라인 그리기
    attentionLines.forEach(line => {
      const fromToken = tokens.find(t => t.id === line.from)
      const toToken = tokens.find(t => t.id === line.to)
      
      if (fromToken && toToken && line.active) {
        const opacity = Math.max(0.1, line.weight)
        
        // 곡선 그리기
        ctx.beginPath()
        ctx.strokeStyle = `rgba(59, 130, 246, ${opacity})`
        ctx.lineWidth = 1 + line.weight * 3
        
        const controlY = fromToken.y - 60
        ctx.moveTo(fromToken.x, fromToken.y - 10)
        ctx.quadraticCurveTo(
          (fromToken.x + toToken.x) / 2, 
          controlY, 
          toToken.x, 
          toToken.y - 10
        )
        
        if (line.weight > 0.7) {
          ctx.shadowColor = '#3b82f6'
          ctx.shadowBlur = 10
        } else {
          ctx.shadowBlur = 0
        }
        
        ctx.stroke()
        
        // 어텐션 플로우 파티클
        if (line.weight > 0.5) {
          const progress = (animationFrame * 0.03) % 1
          const particleX = fromToken.x + (toToken.x - fromToken.x) * progress
          const particleY = fromToken.y - 60 * Math.sin(Math.PI * progress)
          
          ctx.beginPath()
          ctx.arc(particleX, particleY, 2, 0, 2 * Math.PI)
          ctx.fillStyle = `rgba(34, 197, 94, ${line.weight})`
          ctx.fill()
        }
      }
    })

    // 토큰 그리기
    tokens.forEach((token, index) => {
      const isCurrent = index === currentToken
      const intensity = token.attention
      
      // 토큰 배경
      ctx.beginPath()
      ctx.arc(token.x, token.y, 20, 0, 2 * Math.PI)
      
      if (isCurrent) {
        ctx.fillStyle = `rgba(239, 68, 68, ${0.8 + 0.2 * Math.sin(animationFrame * 0.1)})`
        ctx.shadowColor = '#ef4444'
        ctx.shadowBlur = 15
      } else {
        ctx.fillStyle = `rgba(71, 85, 105, ${0.5 + 0.3 * intensity})`
        ctx.shadowBlur = 0
      }
      
      ctx.fill()
      
      // 토큰 테두리
      ctx.beginPath()
      ctx.arc(token.x, token.y, 20, 0, 2 * Math.PI)
      ctx.strokeStyle = isCurrent ? '#fbbf24' : 'rgba(148, 163, 184, 0.8)'
      ctx.lineWidth = isCurrent ? 2 : 1
      ctx.stroke()
      
      // 토큰 텍스트
      ctx.fillStyle = '#ffffff'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(token.text, token.x, token.y)
      
      // 어텐션 값 표시
      if (isRunning && intensity > 0.5) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.9)'
        ctx.font = '10px monospace'
        ctx.fillText(intensity.toFixed(2), token.x, token.y + 35)
      }
    })

    // 헤더 정보
    ctx.fillStyle = 'rgba(148, 163, 184, 0.9)'
    ctx.font = '12px monospace'
    ctx.textAlign = 'left'
    ctx.fillText('Multi-Head Attention', 10, 20)
    
    if (isRunning) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.9)'
      ctx.fillText(`Focus: "${sampleTokens[currentToken]}"`, 10, canvas.height - 15)
    }
  }

  const startAttention = () => {
    setIsRunning(true)
    setCurrentToken(0)
  }

  const stopAttention = () => {
    setIsRunning(false)
    setCurrentToken(0)
    
    // 모든 어텐션 리셋
    setAttentionLines(prev => prev.map(line => ({
      ...line,
      active: false,
      weight: Math.random() * 0.3
    })))
    
    setTokens(prev => prev.map(token => ({
      ...token,
      attention: 0
    })))
  }

  return (
    <div className="relative bg-slate-800 rounded-lg p-4 border border-slate-600">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold text-sm">Transformer</h3>
        <div className="flex gap-2">
          {!isRunning ? (
            <button
              onClick={startAttention}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors"
            >
              Start
            </button>
          ) : (
            <button
              onClick={stopAttention}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      </div>
      
      <canvas
        ref={canvasRef}
        className="w-full border border-slate-500 rounded"
        style={{ maxHeight: '200px' }}
      />
      
      {isRunning && (
        <div className="absolute top-2 right-2 bg-blue-600 text-white px-2 py-1 rounded text-xs animate-pulse">
          Attending...
        </div>
      )}
      
      <div className="mt-2 text-xs text-slate-400">
        Blockchain-based prediction with live trading
      </div>
    </div>
  )
}