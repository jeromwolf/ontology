'use client'

import React, { useState, useEffect, createContext, useContext } from 'react'
import { Maximize2, Minimize2, PanelRightClose, PanelRightOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

export type LayoutMode = 'balanced' | 'visualization-focused' | 'controls-focused' | 'fullscreen'
export type LayoutOrientation = 'horizontal' | 'vertical' | 'auto'

export interface LayoutConfig {
  mode: LayoutMode
  orientation: LayoutOrientation
  showControls: boolean
  allowModeSwitch: boolean
  showModeToggle: boolean
  minControlsWidth: number
  maxControlsWidth: number
}

export interface AdaptiveLayoutProps {
  children: React.ReactNode
  controls: React.ReactNode
  className?: string
  config?: Partial<LayoutConfig>
  onLayoutChange?: (config: LayoutConfig) => void
}

// 레이아웃 컨텍스트 (자식 컴포넌트에서 접근 가능)
const LayoutContext = createContext<{
  config: LayoutConfig
  updateConfig: (updates: Partial<LayoutConfig>) => void
} | null>(null)

export const useAdaptiveLayout = () => {
  const context = useContext(LayoutContext)
  if (!context) {
    throw new Error('useAdaptiveLayout must be used within AdaptiveLayout')
  }
  return context
}

/**
 * 적응형 레이아웃 컴포넌트
 * 
 * 기존 문제점 해결:
 * ❌ 고정 비율: grid-cols-4 (75:25)
 * ❌ 화면 크기 무시: 모든 디바이스에서 동일
 * ❌ 사용자 선택 없음: 강제 레이아웃
 * ❌ 공간 낭비: 제어판 과다 점유
 * 
 * 새로운 기능:
 * ✅ 동적 비율: 90:10 기본, 필요시 70:30
 * ✅ 모드 전환: 균형/시각화중심/제어중심/전체화면
 * ✅ 반응형: 화면 크기별 자동 최적화
 * ✅ 사용자 제어: 레이아웃 모드 선택 가능
 * ✅ 접근성: 키보드 단축키 지원
 */
const AdaptiveLayout: React.FC<AdaptiveLayoutProps> = ({
  children,
  controls,
  className,
  config: configOverrides = {},
  onLayoutChange,
}) => {
  const [config, setConfig] = useState<LayoutConfig>({
    mode: 'balanced',
    orientation: 'auto',
    showControls: true,
    allowModeSwitch: true,
    showModeToggle: true,
    minControlsWidth: 200,
    maxControlsWidth: 400,
    ...configOverrides,
  })

  const [screenSize, setScreenSize] = useState<'sm' | 'md' | 'lg' | 'xl'>('lg')

  // 화면 크기 감지
  useEffect(() => {
    const updateScreenSize = () => {
      const width = window.innerWidth
      if (width < 768) setScreenSize('sm')
      else if (width < 1024) setScreenSize('md')
      else if (width < 1280) setScreenSize('lg')
      else setScreenSize('xl')
    }

    updateScreenSize()
    window.addEventListener('resize', updateScreenSize)
    return () => window.removeEventListener('resize', updateScreenSize)
  }, [])

  // 화면 크기에 따른 자동 모드 조정
  useEffect(() => {
    if (config.orientation === 'auto') {
      if (screenSize === 'sm' || screenSize === 'md') {
        // 작은 화면: 세로 레이아웃
        updateConfig({ mode: 'visualization-focused' })
      }
    }
  }, [screenSize, config.orientation])

  const updateConfig = (updates: Partial<LayoutConfig>) => {
    const newConfig = { ...config, ...updates }
    setConfig(newConfig)
    onLayoutChange?.(newConfig)
  }

  // 키보드 단축키
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!config.allowModeSwitch) return

      // Alt + 숫자키로 모드 전환
      if (event.altKey) {
        switch (event.key) {
          case '1':
            updateConfig({ mode: 'balanced' })
            break
          case '2':
            updateConfig({ mode: 'visualization-focused' })
            break
          case '3':
            updateConfig({ mode: 'controls-focused' })
            break
          case '4':
            updateConfig({ mode: 'fullscreen' })
            break
          case 'h':
            updateConfig({ showControls: !config.showControls })
            break
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [config])

  // 레이아웃 클래스 계산
  const getLayoutClasses = () => {
    const isVertical = config.orientation === 'vertical' || 
                     (config.orientation === 'auto' && (screenSize === 'sm' || screenSize === 'md'))

    if (config.mode === 'fullscreen') {
      return {
        container: 'grid grid-cols-1 grid-rows-1',
        main: 'col-span-1 row-span-1',
        controls: 'hidden',
      }
    }

    if (isVertical) {
      // 세로 레이아웃 (모바일/태블릿)
      switch (config.mode) {
        case 'visualization-focused':
          return {
            container: 'grid grid-rows-[1fr_auto] gap-2',
            main: 'row-span-1 min-h-[400px]',
            controls: `row-span-1 max-h-48 overflow-y-auto ${!config.showControls ? 'hidden' : ''}`,
          }
        case 'controls-focused':
          return {
            container: 'grid grid-rows-[auto_1fr] gap-2',
            main: 'row-span-1 min-h-[300px]',
            controls: `row-span-1 ${!config.showControls ? 'hidden' : ''}`,
          }
        default: // balanced
          return {
            container: 'grid grid-rows-[2fr_1fr] gap-2',
            main: 'row-span-1 min-h-[350px]',
            controls: `row-span-1 ${!config.showControls ? 'hidden' : ''}`,
          }
      }
    } else {
      // 가로 레이아웃 (데스크톱)
      switch (config.mode) {
        case 'visualization-focused':
          return {
            container: 'grid grid-cols-[1fr_auto] gap-2',
            main: 'col-span-1',
            controls: `col-span-1 w-64 ${!config.showControls ? 'hidden' : ''}`,
          }
        case 'controls-focused':
          return {
            container: 'grid grid-cols-[1fr_400px] gap-2',
            main: 'col-span-1',
            controls: `col-span-1 ${!config.showControls ? 'hidden' : ''}`,
          }
        default: // balanced
          return {
            container: 'grid grid-cols-[1fr_280px] gap-2',
            main: 'col-span-1',
            controls: `col-span-1 ${!config.showControls ? 'hidden' : ''}`,
          }
      }
    }
  }

  const layoutClasses = getLayoutClasses()

  const layoutValue = {
    config,
    updateConfig,
  }

  return (
    <LayoutContext.Provider value={layoutValue}>
      <div
        className={cn(
          'relative w-full h-full p-2 gap-2',
          layoutClasses.container,
          className
        )}
        role="main"
        aria-label="Adaptive layout container"
      >
        {/* 모드 전환 토글 버튼 */}
        {config.showModeToggle && config.allowModeSwitch && (
          <div className="absolute top-2 right-2 z-10 flex gap-1">
            <button
              onClick={() => updateConfig({ showControls: !config.showControls })}
              className="p-2 bg-black/20 hover:bg-black/30 text-white rounded-lg transition-colors"
              title={`제어판 ${config.showControls ? '숨기기' : '보이기'} (Alt+H)`}
              aria-label={`제어판 ${config.showControls ? '숨기기' : '보이기'}`}
            >
              {config.showControls ? (
                <PanelRightClose className="w-4 h-4" />
              ) : (
                <PanelRightOpen className="w-4 h-4" />
              )}
            </button>
            
            <button
              onClick={() => 
                updateConfig({ 
                  mode: config.mode === 'fullscreen' ? 'balanced' : 'fullscreen' 
                })
              }
              className="p-2 bg-black/20 hover:bg-black/30 text-white rounded-lg transition-colors"
              title={`${config.mode === 'fullscreen' ? '일반' : '전체화면'} 모드 (Alt+4)`}
              aria-label={`${config.mode === 'fullscreen' ? '일반' : '전체화면'} 모드`}
            >
              {config.mode === 'fullscreen' ? (
                <Minimize2 className="w-4 h-4" />
              ) : (
                <Maximize2 className="w-4 h-4" />
              )}
            </button>
          </div>
        )}

        {/* 메인 컨텐츠 영역 (시각화) */}
        <div 
          className={cn(
            'relative overflow-hidden bg-white dark:bg-gray-900 rounded-lg',
            layoutClasses.main
          )}
          role="region"
          aria-label="Main visualization area"
        >
          {children}
        </div>

        {/* 제어판 영역 */}
        <div 
          className={cn(
            'relative overflow-auto bg-gray-50 dark:bg-gray-800 rounded-lg',
            layoutClasses.controls
          )}
          role="region"
          aria-label="Control panel"
        >
          {controls}
        </div>

        {/* 키보드 단축키 안내 */}
        {config.allowModeSwitch && (
          <div className="sr-only" aria-live="polite">
            Alt+1: 균형 모드, Alt+2: 시각화 중심, Alt+3: 제어 중심, Alt+4: 전체화면, Alt+H: 제어판 토글
          </div>
        )}
      </div>
    </LayoutContext.Provider>
  )
}

export default AdaptiveLayout