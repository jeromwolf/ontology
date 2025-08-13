'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import mermaid from 'mermaid'
import { 
  Download, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Maximize2, 
  Minimize2,
  Image as ImageIcon,
  FileText,
  Copy,
  AlertCircle,
  CheckCircle,
  Eye,
  Move,
  RotateCw
} from 'lucide-react'
import SpaceOptimizedButton, { ButtonGroup } from './SpaceOptimizedButton'
import { cn } from '@/lib/utils'

export interface MermaidPreviewProps {
  code: string
  className?: string
  theme?: 'light' | 'dark' | 'forest' | 'base' | 'neutral'
  onError?: (error: string) => void
  onSuccess?: () => void
  enableZoom?: boolean
  enablePan?: boolean
  autoFit?: boolean
}

interface ViewportState {
  scale: number
  translateX: number
  translateY: number
}

/**
 * 전문급 Mermaid 미리보기 컴포넌트
 * 
 * 특징:
 * ✅ 실시간 렌더링: 코드 변경시 즉시 업데이트
 * ✅ 다중 테마: light, dark, forest, base, neutral
 * ✅ 고급 줌/팬: 마우스 휠, 드래그 지원
 * ✅ 오류 처리: 문법 오류 시각적 표시
 * ✅ 고해상도 내보내기: SVG, PNG 지원
 * ✅ 자동 피팅: 컨테이너 크기에 맞춰 자동 조정
 */
const MermaidPreview: React.FC<MermaidPreviewProps> = ({
  code,
  className,
  theme = 'light',
  onError,
  onSuccess,
  enableZoom = true,
  enablePan = true,
  autoFit = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [viewport, setViewport] = useState<ViewportState>({
    scale: 1,
    translateX: 0,
    translateY: 0,
  })
  const [isPanning, setIsPanning] = useState(false)
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 })

  // Mermaid 초기화
  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: theme as any,
      securityLevel: 'loose',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      fontSize: 14,
      darkMode: theme === 'dark',
      suppressErrorRendering: true, // 오류 렌더링 비활성화
      logLevel: 'error', // 로그 레벨을 error로 설정하여 불필요한 메시지 억제
      themeVariables: {
        primaryColor: '#3b82f6',
        primaryTextColor: '#1f2937',
        primaryBorderColor: '#e5e7eb',
        lineColor: '#6b7280',
        secondaryColor: '#f3f4f6',
        tertiaryColor: '#ffffff',
      },
    })
  }, [theme])

  // 다이어그램 렌더링
  const renderDiagram = useCallback(async () => {
    if (!code.trim() || !containerRef.current) return

    setIsLoading(true)
    setError(null)

    try {
      // 기존 SVG 제거
      const container = containerRef.current
      const existingSvg = container.querySelector('svg')
      if (existingSvg) {
        container.removeChild(existingSvg)
      }

      // 유니크 ID 생성
      const id = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      
      // Mermaid 렌더링
      const { svg } = await mermaid.render(id, code)
      
      // SVG를 DOM에 추가
      container.innerHTML = svg
      
      // SVG 참조 저장
      svgRef.current = container.querySelector('svg') as SVGSVGElement
      
      // 오류 메시지 제거 (Mermaid가 생성하는 오류 텍스트들)
      const errorElements = container.querySelectorAll('text[fill="red"], text[fill="#ff0000"], .error-text, .mermaid-error')
      errorElements.forEach(element => element.remove())
      
      // "Syntax error in text" 메시지 제거
      const textElements = container.querySelectorAll('text')
      textElements.forEach(element => {
        if (element.textContent?.includes('Syntax error') || 
            element.textContent?.includes('mermaid version')) {
          element.remove()
        }
      })
      
      if (svgRef.current && autoFit) {
        // 자동 피팅 적용
        fitToContainer()
      }

      onSuccess?.()
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '다이어그램 렌더링 중 오류가 발생했습니다.'
      setError(errorMessage)
      onError?.(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }, [code, autoFit, onError, onSuccess])

  // 코드 변경시 렌더링
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      renderDiagram()
    }, 300) // 디바운싱

    return () => clearTimeout(timeoutId)
  }, [renderDiagram])

  // 컨테이너에 맞춰 피팅
  const fitToContainer = useCallback(() => {
    if (!svgRef.current || !containerRef.current) return

    const svg = svgRef.current
    const container = containerRef.current
    const svgBox = svg.getBBox()
    const containerBox = container.getBoundingClientRect()

    if (svgBox.width === 0 || svgBox.height === 0) return

    const scaleX = (containerBox.width - 40) / svgBox.width
    const scaleY = (containerBox.height - 40) / svgBox.height
    const scale = Math.min(scaleX, scaleY, 1) // 최대 1배율

    const translateX = (containerBox.width - svgBox.width * scale) / 2
    const translateY = (containerBox.height - svgBox.height * scale) / 2

    setViewport({ scale, translateX, translateY })
  }, [])

  // 줌 기능
  const handleZoom = (delta: number) => {
    if (!enableZoom) return

    setViewport(prev => {
      const newScale = Math.max(0.1, Math.min(5, prev.scale + delta))
      return { ...prev, scale: newScale }
    })
  }

  // 휠 줌
  const handleWheel = (event: React.WheelEvent) => {
    if (!enableZoom) return
    
    event.preventDefault()
    const delta = event.deltaY > 0 ? -0.1 : 0.1
    handleZoom(delta)
  }

  // 패닝 시작
  const handleMouseDown = (event: React.MouseEvent) => {
    if (!enablePan || event.button !== 0) return
    
    setIsPanning(true)
    setLastPanPoint({ x: event.clientX, y: event.clientY })
    event.preventDefault()
  }

  // 패닝
  const handleMouseMove = (event: React.MouseEvent) => {
    if (!isPanning || !enablePan) return

    const deltaX = event.clientX - lastPanPoint.x
    const deltaY = event.clientY - lastPanPoint.y

    setViewport(prev => ({
      ...prev,
      translateX: prev.translateX + deltaX,
      translateY: prev.translateY + deltaY,
    }))

    setLastPanPoint({ x: event.clientX, y: event.clientY })
  }

  // 패닝 종료
  const handleMouseUp = () => {
    setIsPanning(false)
  }

  // 뷰포트 리셋
  const resetViewport = () => {
    if (autoFit) {
      fitToContainer()
    } else {
      setViewport({ scale: 1, translateX: 0, translateY: 0 })
    }
  }

  // SVG 내보내기
  const exportSVG = () => {
    if (!svgRef.current) return

    const svgData = new XMLSerializer().serializeToString(svgRef.current)
    const blob = new Blob([svgData], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    
    const a = document.createElement('a')
    a.href = url
    a.download = 'diagram.svg'
    a.click()
    
    URL.revokeObjectURL(url)
  }

  // PNG 내보내기
  const exportPNG = () => {
    if (!svgRef.current) return

    const svg = svgRef.current
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!
    const img = new Image()
    
    const svgData = new XMLSerializer().serializeToString(svg)
    const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(svgBlob)
    
    img.onload = () => {
      canvas.width = img.width * 2 // 고해상도
      canvas.height = img.height * 2
      ctx.scale(2, 2)
      ctx.drawImage(img, 0, 0)
      
      canvas.toBlob((blob) => {
        if (blob) {
          const pngUrl = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = pngUrl
          a.download = 'diagram.png'
          a.click()
          URL.revokeObjectURL(pngUrl)
        }
      }, 'image/png')
      
      URL.revokeObjectURL(url)
    }
    
    img.src = url
  }

  // 클립보드 복사
  const copyToClipboard = async () => {
    if (!svgRef.current) return

    try {
      const svgData = new XMLSerializer().serializeToString(svgRef.current)
      await navigator.clipboard.writeText(svgData)
    } catch (err) {
      console.error('클립보드 복사 실패:', err)
    }
  }

  return (
    <div
      className={cn(
        'relative flex flex-col h-full border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden',
        isFullscreen && 'fixed inset-0 z-50 bg-white dark:bg-gray-900',
        className
      )}
    >
      {/* 헤더 툴바 */}
      <div className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-gray-600 dark:text-gray-400" />
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            미리보기
          </span>
          {error && (
            <div className="flex items-center gap-1 text-red-600 dark:text-red-400">
              <AlertCircle className="w-3 h-3" />
              <span className="text-xs">오류</span>
            </div>
          )}
          {!error && !isLoading && code.trim() && (
            <div className="flex items-center gap-1 text-green-600 dark:text-green-400">
              <CheckCircle className="w-3 h-3" />
              <span className="text-xs">정상</span>
            </div>
          )}
        </div>

        <ButtonGroup>
          {enableZoom && (
            <>
              <SpaceOptimizedButton
                variant="ghost"
                size="xs"
                icon={<ZoomOut className="w-3 h-3" />}
                tooltip="축소"
                onClick={() => handleZoom(-0.2)}
                disabled={viewport.scale <= 0.1}
              />
              
              <SpaceOptimizedButton
                variant="ghost"
                size="xs"
                tooltip={`${Math.round(viewport.scale * 100)}%`}
                onClick={resetViewport}
              >
                {Math.round(viewport.scale * 100)}%
              </SpaceOptimizedButton>
              
              <SpaceOptimizedButton
                variant="ghost"
                size="xs"
                icon={<ZoomIn className="w-3 h-3" />}
                tooltip="확대"
                onClick={() => handleZoom(0.2)}
                disabled={viewport.scale >= 5}
              />
            </>
          )}
          
          <SpaceOptimizedButton
            variant="ghost"
            size="xs"
            icon={<RotateCcw className="w-3 h-3" />}
            tooltip="뷰포트 리셋"
            onClick={resetViewport}
          />
          
          <SpaceOptimizedButton
            variant="ghost"
            size="xs"
            icon={<Copy className="w-3 h-3" />}
            tooltip="SVG 복사"
            onClick={copyToClipboard}
          />
          
          <SpaceOptimizedButton
            variant="ghost"
            size="xs"
            icon={<FileText className="w-3 h-3" />}
            tooltip="SVG 다운로드"
            onClick={exportSVG}
          />
          
          <SpaceOptimizedButton
            variant="ghost"
            size="xs"
            icon={<ImageIcon className="w-3 h-3" />}
            tooltip="PNG 다운로드"
            onClick={exportPNG}
          />
          
          <SpaceOptimizedButton
            variant="ghost"
            size="xs"
            icon={isFullscreen ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
            tooltip={`${isFullscreen ? '축소' : '전체화면'}`}
            onClick={() => setIsFullscreen(!isFullscreen)}
          />
        </ButtonGroup>
      </div>

      {/* 미리보기 영역 */}
      <div 
        className="flex-1 relative overflow-hidden bg-white dark:bg-gray-900"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isPanning ? 'grabbing' : (enablePan ? 'grab' : 'default') }}
      >
        {/* 로딩 상태 */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-10">
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
              <RotateCw className="w-4 h-4 animate-spin" />
              <span className="text-sm">렌더링 중...</span>
            </div>
          </div>
        )}

        {/* 오류 표시 */}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="max-w-md text-center">
              <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-3" />
              <h3 className="text-lg font-medium text-red-900 dark:text-red-100 mb-2">
                렌더링 오류
              </h3>
              <p className="text-sm text-red-700 dark:text-red-300 mb-4">
                {error}
              </p>
              <SpaceOptimizedButton
                variant="outline"
                size="sm"
                onClick={() => renderDiagram()}
              >
                다시 시도
              </SpaceOptimizedButton>
            </div>
          </div>
        )}

        {/* 빈 상태 */}
        {!code.trim() && !isLoading && !error && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-600">
            <div className="text-center">
              <Eye className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">코드를 입력하면 여기에 다이어그램이 표시됩니다</p>
            </div>
          </div>
        )}

        {/* 다이어그램 컨테이너 */}
        <div
          ref={containerRef}
          className="w-full h-full flex items-center justify-center"
          style={{
            transform: `scale(${viewport.scale}) translate(${viewport.translateX / viewport.scale}px, ${viewport.translateY / viewport.scale}px)`,
            transformOrigin: 'center center',
          }}
        />
      </div>

      {/* 상태 바 */}
      <div className="flex items-center justify-between px-3 py-1 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-600 dark:text-gray-400">
        <div className="flex items-center gap-4">
          <span>테마: {theme}</span>
          {enableZoom && <span>줌: {Math.round(viewport.scale * 100)}%</span>}
          {enablePan && <span>패닝: {enablePan ? '활성' : '비활성'}</span>}
        </div>
        
        <div className="flex items-center gap-2">
          <span>Mermaid v11.9.0</span>
        </div>
      </div>
    </div>
  )
}

export default MermaidPreview