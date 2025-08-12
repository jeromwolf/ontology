import { useEffect, useRef, useCallback } from 'react'

export interface CanvasConfig {
  maintainAspectRatio?: boolean
  aspectRatio?: number
  minWidth?: number
  minHeight?: number
  maxWidth?: number
  maxHeight?: number
  devicePixelRatio?: boolean
}

export interface ResponsiveCanvasHook {
  canvasRef: React.RefObject<HTMLCanvasElement>
  containerRef: React.RefObject<HTMLDivElement>
  dimensions: { width: number; height: number }
  context: CanvasRenderingContext2D | null
  resizeCanvas: () => void
}

/**
 * 완전 반응형 캔버스를 위한 커스텀 훅
 * 기존 문제점: 고정 크기, 저해상도, 반응형 부족
 * 해결: 동적 크기 조정, 고해상도 지원, 컨테이너 크기 추적
 */
export const useResponsiveCanvas = (
  config: CanvasConfig = {}
): ResponsiveCanvasHook => {
  const {
    maintainAspectRatio = false,
    aspectRatio = 16 / 9,
    minWidth = 300,
    minHeight = 200,
    maxWidth = Infinity,
    maxHeight = Infinity,
    devicePixelRatio = true,
  } = config

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const contextRef = useRef<CanvasRenderingContext2D | null>(null)
  const dimensionsRef = useRef({ width: 0, height: 0 })

  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const containerRect = container.getBoundingClientRect()
    let { width, height } = containerRect

    // 최소/최대 크기 제한 적용
    width = Math.max(minWidth, Math.min(maxWidth, width))
    height = Math.max(minHeight, Math.min(maxHeight, height))

    // 종횡비 유지 옵션
    if (maintainAspectRatio) {
      const containerAspectRatio = width / height
      if (containerAspectRatio > aspectRatio) {
        width = height * aspectRatio
      } else {
        height = width / aspectRatio
      }
    }

    // 디바이스 픽셀 비율 적용 (레티나 디스플레이 대응)
    const dpr = devicePixelRatio ? window.devicePixelRatio || 1 : 1

    // 캔버스 실제 해상도 설정 (고해상도)
    canvas.width = width * dpr
    canvas.height = height * dpr

    // CSS 크기 설정 (표시 크기)
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`

    // 컨텍스트 스케일 조정
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.scale(dpr, dpr)
      contextRef.current = ctx
    }

    // 차원 정보 업데이트
    dimensionsRef.current = { width, height }
  }, [
    minWidth,
    minHeight,
    maxWidth,
    maxHeight,
    maintainAspectRatio,
    aspectRatio,
    devicePixelRatio,
  ])

  useEffect(() => {
    // 초기 크기 설정
    resizeCanvas()

    // ResizeObserver를 사용한 정밀한 크기 감지
    let resizeObserver: ResizeObserver | null = null

    if (containerRef.current) {
      resizeObserver = new ResizeObserver(() => {
        resizeCanvas()
      })
      resizeObserver.observe(containerRef.current)
    }

    // 윈도우 리사이즈 이벤트 (백업)
    const handleResize = () => resizeCanvas()
    window.addEventListener('resize', handleResize)

    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
      window.removeEventListener('resize', handleResize)
    }
  }, [resizeCanvas])

  return {
    canvasRef,
    containerRef,
    dimensions: dimensionsRef.current,
    context: contextRef.current,
    resizeCanvas,
  }
}