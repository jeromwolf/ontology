'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'

export default function PNJunctionSimulator() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [voltage, setVoltage] = useState(0)
  const [mode, setMode] = useState<'forward' | 'reverse'>('forward')
  const [depletionWidth, setDepletionWidth] = useState(50)
  const [current, setCurrent] = useState(0)
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    drawJunction()
  }, [voltage, mode, depletionWidth])

  const toggleFullscreen = () => {
    if (!containerRef.current) return

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  const drawJunction = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // 배경 초기화
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, height)

    // P형 영역 (왼쪽)
    const pRegionGradient = ctx.createLinearGradient(0, 0, width / 2 - depletionWidth / 2, 0)
    pRegionGradient.addColorStop(0, '#ff6b9d')
    pRegionGradient.addColorStop(1, '#c44569')
    ctx.fillStyle = pRegionGradient
    ctx.fillRect(0, 0, width / 2 - depletionWidth / 2, height)

    // N형 영역 (오른쪽)
    const nRegionGradient = ctx.createLinearGradient(width / 2 + depletionWidth / 2, 0, width, 0)
    nRegionGradient.addColorStop(0, '#4a69bd')
    nRegionGradient.addColorStop(1, '#6a89cc')
    ctx.fillStyle = nRegionGradient
    ctx.fillRect(width / 2 + depletionWidth / 2, 0, width / 2 + depletionWidth / 2, height)

    // 공핍 영역 (가운데)
    ctx.fillStyle = '#2d2d44'
    ctx.fillRect(width / 2 - depletionWidth / 2, 0, depletionWidth, height)

    // P형 텍스트
    ctx.fillStyle = 'white'
    ctx.font = 'bold 24px Arial'
    ctx.fillText('P형', 80, height / 2)
    ctx.font = '16px Arial'
    ctx.fillText('정공 과잉', 60, height / 2 + 30)

    // N형 텍스트
    ctx.fillText('N형', width - 120, height / 2)
    ctx.fillText('전자 과잉', width - 140, height / 2 + 30)

    // 공핍 영역 텍스트
    ctx.fillText('공핍 영역', width / 2 - 40, 30)
    ctx.fillText(`${depletionWidth}nm`, width / 2 - 30, 50)

    // 정공과 전자 그리기
    drawCarriers(ctx, width, height)

    // 전압 화살표
    drawVoltageArrows(ctx, width, height)
  }

  const drawCarriers = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // P형 정공 (빨간 원)
    ctx.fillStyle = '#ff6b9d'
    for (let i = 0; i < 15; i++) {
      const x = Math.random() * (width / 2 - depletionWidth / 2 - 40) + 20
      const y = Math.random() * (height - 40) + 20
      ctx.beginPath()
      ctx.arc(x, y, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = 'white'
      ctx.font = 'bold 12px Arial'
      ctx.fillText('+', x - 4, y + 4)
      ctx.fillStyle = '#ff6b9d'
    }

    // N형 전자 (파란 원)
    ctx.fillStyle = '#4a69bd'
    for (let i = 0; i < 15; i++) {
      const x = Math.random() * (width / 2 - depletionWidth / 2 - 40) + width / 2 + depletionWidth / 2 + 20
      const y = Math.random() * (height - 40) + 20
      ctx.beginPath()
      ctx.arc(x, y, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = 'white'
      ctx.font = 'bold 12px Arial'
      ctx.fillText('-', x - 3, y + 4)
      ctx.fillStyle = '#4a69bd'
    }
  }

  const drawVoltageArrows = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    if (voltage === 0) return

    ctx.strokeStyle = voltage > 0 ? '#00d2ff' : '#ff6348'
    ctx.lineWidth = 3
    ctx.setLineDash([5, 5])

    const startX = mode === 'forward' ? 50 : width - 50
    const endX = mode === 'forward' ? width - 50 : 50
    const y = height - 50

    ctx.beginPath()
    ctx.moveTo(startX, y)
    ctx.lineTo(endX, y)
    ctx.stroke()

    // 화살표 머리
    const angle = mode === 'forward' ? 0 : Math.PI
    ctx.save()
    ctx.translate(endX, y)
    ctx.rotate(angle)
    ctx.beginPath()
    ctx.moveTo(0, 0)
    ctx.lineTo(-15, -8)
    ctx.lineTo(-15, 8)
    ctx.closePath()
    ctx.fillStyle = ctx.strokeStyle
    ctx.fill()
    ctx.restore()

    ctx.setLineDash([])
  }

  const applyVoltage = (v: number) => {
    setVoltage(v)

    if (mode === 'forward') {
      // 순방향: 공핍 영역 감소, 전류 증가
      const newWidth = Math.max(10, 50 - v * 5)
      setDepletionWidth(newWidth)
      const newCurrent = v > 0.7 ? Math.exp((v - 0.7) * 5) : 0
      setCurrent(newCurrent)
    } else {
      // 역방향: 공핍 영역 증가, 전류 거의 0
      const newWidth = Math.min(100, 50 + Math.abs(v) * 8)
      setDepletionWidth(newWidth)
      setCurrent(Math.abs(v) * 0.01) // 미세한 누설 전류
    }
  }

  return (
    <div ref={containerRef} className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-900 dark:to-gray-800 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8 relative">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            PN 접합 시뮬레이터
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            순방향/역방향 바이어스를 적용하여 PN 접합의 동작 원리를 학습하세요
          </p>

          {/* 상단 버튼들 */}
          <div className="absolute top-0 right-0 flex gap-2">
            {/* 모듈로 돌아가기 버튼 */}
            <Link
              href="/modules/semiconductor"
              className="p-3 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:shadow-xl transition-all text-gray-700 dark:text-gray-300"
              title="모듈로 돌아가기"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </Link>

            {/* 전체화면 버튼 */}
            <button
              onClick={toggleFullscreen}
              className="p-3 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:shadow-xl transition-all text-gray-700 dark:text-gray-300"
              title={isFullscreen ? "전체화면 종료 (ESC)" : "전체화면"}
            >
              {isFullscreen ? (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
              )}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas 영역 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <canvas
                ref={canvasRef}
                width={800}
                height={400}
                className="w-full border-2 border-gray-300 dark:border-gray-600 rounded-lg"
              />
            </div>

            {/* I-V 특성 곡선 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mt-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                I-V 특성 곡선
              </h3>
              <div className="relative h-64 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center">
                <svg viewBox="0 0 400 200" className="w-full h-full">
                  {/* X축, Y축 */}
                  <line x1="50" y1="100" x2="350" y2="100" stroke="currentColor" strokeWidth="2" className="text-gray-400" />
                  <line x1="200" y1="20" x2="200" y2="180" stroke="currentColor" strokeWidth="2" className="text-gray-400" />

                  {/* I-V 곡선 */}
                  <path
                    d="M 50,100 L 180,100 Q 200,100 210,60 L 350,20"
                    fill="none"
                    stroke="#4a69bd"
                    strokeWidth="3"
                  />

                  {/* 역방향 영역 */}
                  <path
                    d="M 200,100 L 50,105"
                    fill="none"
                    stroke="#ff6b9d"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                  />

                  {/* 현재 동작점 */}
                  <circle
                    cx={mode === 'forward' ? 200 + voltage * 30 : 200 - Math.abs(voltage) * 30}
                    cy={mode === 'forward' ? 100 - current * 10 : 100 + current * 5}
                    r="6"
                    fill="#00d2ff"
                  />

                  {/* 라벨 */}
                  <text x="360" y="105" fontSize="14" fill="currentColor" className="text-gray-600">V</text>
                  <text x="205" y="15" fontSize="14" fill="currentColor" className="text-gray-600">I</text>
                  <text x="205" y="115" fontSize="12" fill="currentColor" className="text-gray-500">0</text>
                  <text x="195" y="95" fontSize="12" fill="currentColor" className="text-gray-500">0.7V</text>
                </svg>
              </div>
            </div>
          </div>

          {/* 제어판 */}
          <div className="space-y-6">
            {/* 바이어스 모드 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                바이어스 모드
              </h3>
              <div className="space-y-3">
                <button
                  onClick={() => {
                    setMode('forward')
                    setVoltage(0)
                    setDepletionWidth(50)
                  }}
                  className={`w-full px-4 py-3 rounded-lg font-medium transition-all ${
                    mode === 'forward'
                      ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  순방향 바이어스
                </button>
                <button
                  onClick={() => {
                    setMode('reverse')
                    setVoltage(0)
                    setDepletionWidth(50)
                  }}
                  className={`w-full px-4 py-3 rounded-lg font-medium transition-all ${
                    mode === 'reverse'
                      ? 'bg-gradient-to-r from-pink-500 to-red-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  역방향 바이어스
                </button>
              </div>
            </div>

            {/* 전압 제어 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                인가 전압: {voltage.toFixed(2)}V
              </h3>
              <input
                type="range"
                min="0"
                max={mode === 'forward' ? '2' : '10'}
                step="0.1"
                value={voltage}
                onChange={(e) => applyVoltage(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mt-2">
                <span>0V</span>
                <span>{mode === 'forward' ? '2V' : '10V'}</span>
              </div>
            </div>

            {/* 측정값 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                측정값
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">공핍 영역 폭</span>
                  <span className="font-mono font-bold text-blue-600 dark:text-blue-400">
                    {depletionWidth.toFixed(0)} nm
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">전류</span>
                  <span className="font-mono font-bold text-purple-600 dark:text-purple-400">
                    {current.toFixed(2)} mA
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">내장 전위</span>
                  <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400">
                    0.7 V
                  </span>
                </div>
              </div>
            </div>

            {/* 설명 */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
              <h3 className="font-bold mb-2">💡 동작 원리</h3>
              <p className="text-sm leading-relaxed">
                {mode === 'forward'
                  ? '순방향 바이어스: P형에 (+), N형에 (-)를 연결하면 공핍 영역이 좁아지고 전류가 흐릅니다.'
                  : '역방향 바이어스: P형에 (-), N형에 (+)를 연결하면 공핍 영역이 넓어지고 전류가 거의 흐르지 않습니다.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
