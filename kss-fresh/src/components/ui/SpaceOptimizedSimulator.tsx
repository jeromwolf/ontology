'use client'

import React, { useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, Download, Settings, Palette } from 'lucide-react'
import ResponsiveCanvas, { ResponsiveCanvasRef } from './ResponsiveCanvas'
import AdaptiveLayout from './AdaptiveLayout'
import CollapsibleControls, { createControlSection } from './CollapsibleControls'
import SpaceOptimizedButton, { ButtonGroup, SimulationControls } from './SpaceOptimizedButton'

/**
 * 공간 최적화 시뮬레이터 템플릿
 * 
 * 모든 새로운 시뮬레이터의 기본 템플릿으로 사용
 * 기존 문제점들이 모두 해결된 완전한 예제
 */
const SpaceOptimizedSimulator: React.FC = () => {
  const canvasRef = useRef<ResponsiveCanvasRef>(null)
  const [isRunning, setIsRunning] = React.useState(false)
  const [settings, setSettings] = React.useState({
    speed: 1,
    color: '#3b82f6',
    showGrid: true,
    autoPlay: false,
  })

  // 캔버스 그리기 함수 (예제)
  const drawVisualization = (context: CanvasRenderingContext2D, dimensions: { width: number; height: number }) => {
    const { width, height } = dimensions
    
    // 배경 클리어
    context.clearRect(0, 0, width, height)
    
    // 그리드 그리기 (옵션)
    if (settings.showGrid) {
      context.strokeStyle = '#e5e7eb'
      context.lineWidth = 1
      
      for (let x = 0; x < width; x += 20) {
        context.beginPath()
        context.moveTo(x, 0)
        context.lineTo(x, height)
        context.stroke()
      }
      
      for (let y = 0; y < height; y += 20) {
        context.beginPath()
        context.moveTo(0, y)
        context.lineTo(width, y)
        context.stroke()
      }
    }
    
    // 예제 애니메이션
    const time = Date.now() * 0.001 * settings.speed
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) * 0.3
    
    context.fillStyle = settings.color
    context.beginPath()
    context.arc(
      centerX + Math.cos(time) * radius * 0.5,
      centerY + Math.sin(time) * radius * 0.5,
      20,
      0,
      Math.PI * 2
    )
    context.fill()
  }

  // 애니메이션 루프
  useEffect(() => {
    let animationFrame: number

    const animate = () => {
      if (canvasRef.current && isRunning) {
        const context = canvasRef.current.context
        if (context) {
          const dimensions = canvasRef.current.dimensions
          drawVisualization(context, dimensions)
        }
      }
      
      if (isRunning) {
        animationFrame = requestAnimationFrame(animate)
      }
    }

    if (isRunning) {
      animate()
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
    }
  }, [isRunning, settings])

  // 제어 섹션들 정의
  const controlSections = [
    createControlSection(
      'simulation',
      '시뮬레이션 제어',
      <div className="space-y-3">
        <ButtonGroup>
          <SpaceOptimizedButton
            variant={isRunning ? 'secondary' : 'primary'}
            size="sm"
            compact
            icon={isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            onClick={() => setIsRunning(!isRunning)}
          >
            {isRunning ? '일시정지' : '시작'}
          </SpaceOptimizedButton>
          
          <SpaceOptimizedButton
            variant="outline"
            size="sm"
            compact
            icon={<RotateCcw className="w-4 h-4" />}
            onClick={() => {
              setIsRunning(false)
              // 리셋 로직
            }}
          >
            초기화
          </SpaceOptimizedButton>
        </ButtonGroup>

        <div>
          <label className="block text-sm font-medium mb-1">속도</label>
          <input
            type="range"
            min="0.1"
            max="3"
            step="0.1"
            value={settings.speed}
            onChange={(e) => setSettings(prev => ({ ...prev, speed: parseFloat(e.target.value) }))}
            className="w-full"
          />
          <div className="text-xs text-gray-500 mt-1">
            {settings.speed.toFixed(1)}x
          </div>
        </div>
      </div>,
      { 
        defaultExpanded: true, 
        icon: <Play className="w-4 h-4" />,
        badge: isRunning ? '실행중' : undefined 
      }
    ),

    createControlSection(
      'appearance',
      '외형 설정',
      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium mb-1">색상</label>
          <input
            type="color"
            value={settings.color}
            onChange={(e) => setSettings(prev => ({ ...prev, color: e.target.value }))}
            className="w-full h-8 rounded border border-gray-300"
          />
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="showGrid"
            checked={settings.showGrid}
            onChange={(e) => setSettings(prev => ({ ...prev, showGrid: e.target.checked }))}
            className="rounded"
          />
          <label htmlFor="showGrid" className="text-sm">그리드 표시</label>
        </div>
      </div>,
      { 
        icon: <Palette className="w-4 h-4" /> 
      }
    ),

    createControlSection(
      'export',
      '내보내기',
      <div className="space-y-2">
        <SpaceOptimizedButton
          variant="outline"
          size="sm"
          fullWidth
          icon={<Download className="w-4 h-4" />}
          onClick={() => {
            // PNG 내보내기 로직
            const canvasEl = canvasRef.current?.canvas
            if (canvasEl) {
              const link = document.createElement('a')
              link.download = 'simulation.png'
              link.href = canvasEl.toDataURL()
              link.click()
            }
          }}
        >
          PNG로 저장
        </SpaceOptimizedButton>
      </div>,
      { 
        icon: <Download className="w-4 h-4" /> 
      }
    ),
  ]

  return (
    <div className="w-full h-full">
      <AdaptiveLayout
        controls={
          <CollapsibleControls
            sections={controlSections}
            title="시뮬레이션 제어"
            defaultCollapsed={false}
            onSectionToggle={(sectionId, expanded) => {
              console.log(`Section ${sectionId} ${expanded ? 'expanded' : 'collapsed'}`)
            }}
          />
        }
        config={{
          mode: 'visualization-focused',
          allowModeSwitch: true,
          showModeToggle: true,
        }}
        onLayoutChange={(config) => {
          console.log('Layout changed:', config)
        }}
      >
        <ResponsiveCanvas
          ref={canvasRef}
          onContextReady={(context) => {
            console.log('Canvas ready:', context)
          }}
          onResize={(dimensions) => {
            console.log('Canvas resized:', dimensions)
            // 즉시 다시 그리기
            if (canvasRef.current?.context) {
              drawVisualization(canvasRef.current.context, dimensions)
            }
          }}
          maintainAspectRatio={false}
          minWidth={400}
          minHeight={300}
          className="border border-gray-200 dark:border-gray-700"
        />
      </AdaptiveLayout>
    </div>
  )
}

export default SpaceOptimizedSimulator