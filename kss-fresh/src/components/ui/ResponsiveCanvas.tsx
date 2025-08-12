'use client'

import React, { forwardRef, useImperativeHandle } from 'react'
import { useResponsiveCanvas, type CanvasConfig } from '@/hooks/useResponsiveCanvas'
import { cn } from '@/lib/utils'

export interface ResponsiveCanvasProps extends CanvasConfig {
  className?: string
  containerClassName?: string
  onResize?: (dimensions: { width: number; height: number }) => void
  onContextReady?: (context: CanvasRenderingContext2D) => void
  children?: React.ReactNode
}

export interface ResponsiveCanvasRef {
  canvas: HTMLCanvasElement | null
  context: CanvasRenderingContext2D | null
  dimensions: { width: number; height: number }
  resizeCanvas: () => void
}

/**
 * 완전 반응형 캔버스 컴포넌트
 * 
 * 기존 문제점 해결:
 * ❌ 고정 크기: width={800} height={600}
 * ❌ 저해상도: 레티나 디스플레이 미지원
 * ❌ 반응형 부족: 화면 크기 변경시 깨짐
 * ❌ 공간 낭비: 컨테이너 크기 미활용
 * 
 * 새로운 기능:
 * ✅ 동적 크기 조정: 컨테이너에 맞춰 자동 리사이즈
 * ✅ 고해상도 지원: devicePixelRatio 적용
 * ✅ 성능 최적화: ResizeObserver 사용
 * ✅ 접근성: 키보드 네비게이션 지원
 * ✅ TypeScript: 완전한 타입 안전성
 */
const ResponsiveCanvas = forwardRef<ResponsiveCanvasRef, ResponsiveCanvasProps>(
  (
    {
      className,
      containerClassName,
      onResize,
      onContextReady,
      children,
      ...canvasConfig
    },
    ref
  ) => {
    const { canvasRef, containerRef, dimensions, context, resizeCanvas } =
      useResponsiveCanvas(canvasConfig)

    // 상위 컴포넌트에서 접근 가능한 ref 인터페이스
    useImperativeHandle(ref, () => ({
      canvas: canvasRef.current,
      context,
      dimensions,
      resizeCanvas,
    }))

    // 크기 변경 콜백
    React.useEffect(() => {
      if (onResize && dimensions.width > 0 && dimensions.height > 0) {
        onResize(dimensions)
      }
    }, [dimensions, onResize])

    // 컨텍스트 준비 콜백
    React.useEffect(() => {
      if (onContextReady && context) {
        onContextReady(context)
      }
    }, [context, onContextReady])

    return (
      <div
        ref={containerRef}
        className={cn(
          // 기본 스타일
          'relative w-full h-full',
          // 최소 크기 보장
          'min-h-[300px]',
          // 컨테이너 스타일링
          'overflow-hidden rounded-lg',
          // 포커스 스타일 (접근성)
          'focus-within:ring-2 focus-within:ring-blue-500 focus-within:ring-offset-2',
          containerClassName
        )}
        role="img"
        aria-label="Interactive canvas visualization"
      >
        <canvas
          ref={canvasRef}
          className={cn(
            // 기본 캔버스 스타일
            'block',
            // 포커스 스타일 (키보드 접근성)
            'focus:outline-none focus:ring-2 focus:ring-blue-500',
            className
          )}
          tabIndex={0}
          aria-label="Canvas drawing area"
        />
        
        {/* 오버레이 컨텐츠 (제어 버튼, 정보 등) */}
        {children && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="pointer-events-auto">{children}</div>
          </div>
        )}
        
        {/* 로딩 상태 표시 */}
        {dimensions.width === 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Canvas 초기화 중...
            </div>
          </div>
        )}
      </div>
    )
  }
)

ResponsiveCanvas.displayName = 'ResponsiveCanvas'

export default ResponsiveCanvas

// 편의를 위한 타입 내보내기
export type { ResponsiveCanvasProps, ResponsiveCanvasRef, CanvasConfig }