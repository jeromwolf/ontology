'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'

type ProcessStep = 'coating' | 'exposure' | 'development' | 'etching' | 'complete'

export default function LithographySimulator() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [currentStep, setCurrentStep] = useState<ProcessStep>('coating')
  const [exposureTime, setExposureTime] = useState(0)
  const [wavelength, setWavelength] = useState(193) // EUV: 13.5nm, ArF: 193nm, KrF: 248nm
  const [resolution, setResolution] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)

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

  useEffect(() => {
    drawProcess()
  }, [currentStep, exposureTime, wavelength])

  useEffect(() => {
    // 해상도 계산: k1 * λ / NA (Rayleigh criterion)
    const k1 = 0.5
    const NA = 1.35
    const calculatedResolution = (k1 * wavelength) / NA
    setResolution(calculatedResolution)
  }, [wavelength])

  const drawProcess = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, height)

    const waferY = height - 100
    const layerHeight = 30

    // 웨이퍼 기판
    const waferGradient = ctx.createLinearGradient(0, waferY, 0, height)
    waferGradient.addColorStop(0, '#718093')
    waferGradient.addColorStop(1, '#2f3542')
    ctx.fillStyle = waferGradient
    ctx.fillRect(100, waferY, width - 200, 100)

    // 라벨
    ctx.fillStyle = 'white'
    ctx.font = '14px Arial'
    ctx.fillText('Si 웨이퍼', width / 2 - 30, height - 20)

    if (currentStep === 'coating' || currentStep === 'exposure' || currentStep === 'development' || currentStep === 'etching' || currentStep === 'complete') {
      // 포토레지스트 층
      const resistGradient = ctx.createLinearGradient(0, waferY - layerHeight, 0, waferY)
      resistGradient.addColorStop(0, '#ffd32a')
      resistGradient.addColorStop(1, '#f39c12')
      ctx.fillStyle = resistGradient
      ctx.fillRect(100, waferY - layerHeight, width - 200, layerHeight)

      ctx.fillStyle = 'white'
      ctx.fillText('포토레지스트', width / 2 - 40, waferY - layerHeight / 2 + 5)
    }

    if (currentStep === 'exposure') {
      // 마스크
      const maskY = 100
      ctx.fillStyle = '#34495e'
      ctx.fillRect(150, maskY, width - 300, 40)

      // 마스크 패턴 (간단한 회로 패턴)
      ctx.fillStyle = '#1a1a2e'
      for (let i = 0; i < 5; i++) {
        const x = 200 + i * 100
        ctx.fillRect(x, maskY + 10, 40, 20)
      }

      ctx.fillStyle = 'white'
      ctx.fillText('마스크 (회로 패턴)', width / 2 - 50, maskY - 10)

      // UV 광선
      drawUVRays(ctx, maskY + 40, waferY - layerHeight)
    }

    if (currentStep === 'development') {
      // 현상 후 패턴
      ctx.fillStyle = '#1a1a2e'
      for (let i = 0; i < 5; i++) {
        const x = 200 + i * 100
        ctx.fillRect(x, waferY - layerHeight, 40, layerHeight)
      }
    }

    if (currentStep === 'etching' || currentStep === 'complete') {
      // 에칭 후 패턴 (웨이퍼에 각인)
      ctx.fillStyle = '#34495e'
      for (let i = 0; i < 5; i++) {
        const x = 200 + i * 100
        ctx.fillRect(x, waferY - 10, 40, 10)
      }

      if (currentStep === 'complete') {
        // 레지스트 제거
        ctx.fillStyle = '#2ecc71'
        ctx.fillText('✓ 패턴 완성', width / 2 - 40, waferY - 60)
      }
    }

    // 공정 단계 표시
    drawStepIndicator(ctx, width)
  }

  const drawUVRays = (startY: number, endY: number) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width

    // UV 광선 효과
    ctx.strokeStyle = wavelength === 13.5 ? '#a29bfe' : wavelength === 193 ? '#74b9ff' : '#55efc4'
    ctx.lineWidth = 2
    ctx.globalAlpha = 0.6

    for (let i = 0; i < 10; i++) {
      const x = 200 + i * 40
      ctx.beginPath()
      ctx.moveTo(x, startY)
      ctx.lineTo(x, endY)
      ctx.stroke()

      // 파동 효과
      ctx.beginPath()
      for (let y = startY; y < endY; y += 10) {
        const offset = Math.sin((y - startY) / 10) * 5
        if (y === startY) {
          ctx.moveTo(x + offset, y)
        } else {
          ctx.lineTo(x + offset, y)
        }
      }
      ctx.stroke()
    }

    ctx.globalAlpha = 1.0

    // 파장 정보
    ctx.fillStyle = 'white'
    ctx.font = '14px Arial'
    const wavelengthText = wavelength === 13.5 ? 'EUV 13.5nm' : wavelength === 193 ? 'ArF 193nm' : 'KrF 248nm'
    ctx.fillText(wavelengthText, width / 2 - 40, (startY + endY) / 2)
  }

  const drawStepIndicator = (ctx: CanvasRenderingContext2D, width: number) => {
    const steps: ProcessStep[] = ['coating', 'exposure', 'development', 'etching', 'complete']
    const stepNames = ['코팅', '노광', '현상', '에칭', '완료']
    const stepWidth = 120
    const startX = (width - stepWidth * steps.length) / 2
    const y = 30

    steps.forEach((step, index) => {
      const x = startX + index * stepWidth
      const isActive = currentStep === step
      const isCompleted = steps.indexOf(currentStep) > index

      // 배경
      ctx.fillStyle = isActive ? '#3498db' : isCompleted ? '#2ecc71' : '#34495e'
      ctx.fillRect(x, y, stepWidth - 10, 40)

      // 텍스트
      ctx.fillStyle = 'white'
      ctx.font = 'bold 14px Arial'
      ctx.fillText(`${index + 1}. ${stepNames[index]}`, x + 10, y + 25)

      // 연결선
      if (index < steps.length - 1) {
        ctx.strokeStyle = isCompleted ? '#2ecc71' : '#34495e'
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(x + stepWidth - 10, y + 20)
        ctx.lineTo(x + stepWidth, y + 20)
        ctx.stroke()
      }
    })
  }

  const nextStep = () => {
    const steps: ProcessStep[] = ['coating', 'exposure', 'development', 'etching', 'complete']
    const currentIndex = steps.indexOf(currentStep)
    if (currentIndex < steps.length - 1) {
      setCurrentStep(steps[currentIndex + 1])
    }
  }

  const prevStep = () => {
    const steps: ProcessStep[] = ['coating', 'exposure', 'development', 'etching', 'complete']
    const currentIndex = steps.indexOf(currentStep)
    if (currentIndex > 0) {
      setCurrentStep(steps[currentIndex - 1])
    }
  }

  const reset = () => {
    setCurrentStep('coating')
    setExposureTime(0)
  }

  return (
    <div ref={containerRef} className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-900 dark:to-gray-800 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8 relative">
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
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            포토리소그래피 공정 시뮬레이터
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            반도체 회로 패턴을 형성하는 핵심 공정을 단계별로 학습하세요
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas 영역 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <canvas
                ref={canvasRef}
                width={800}
                height={500}
                className="w-full border-2 border-gray-300 dark:border-gray-600 rounded-lg"
              />
            </div>

            {/* 공정 제어 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mt-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                공정 제어
              </h3>
              <div className="flex gap-4">
                <button
                  onClick={prevStep}
                  disabled={currentStep === 'coating'}
                  className="flex-1 px-6 py-3 bg-gray-500 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors"
                >
                  ← 이전 단계
                </button>
                <button
                  onClick={nextStep}
                  disabled={currentStep === 'complete'}
                  className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all"
                >
                  다음 단계 →
                </button>
                <button
                  onClick={reset}
                  className="px-6 py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 transition-colors"
                >
                  초기화
                </button>
              </div>
            </div>
          </div>

          {/* 제어판 */}
          <div className="space-y-6">
            {/* 광원 선택 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                광원 선택
              </h3>
              <div className="space-y-3">
                <button
                  onClick={() => setWavelength(248)}
                  className={`w-full px-4 py-3 rounded-lg font-medium transition-all ${
                    wavelength === 248
                      ? 'bg-gradient-to-r from-green-500 to-teal-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  KrF (248nm)
                </button>
                <button
                  onClick={() => setWavelength(193)}
                  className={`w-full px-4 py-3 rounded-lg font-medium transition-all ${
                    wavelength === 193
                      ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  ArF (193nm)
                </button>
                <button
                  onClick={() => setWavelength(13.5)}
                  className={`w-full px-4 py-3 rounded-lg font-medium transition-all ${
                    wavelength === 13.5
                      ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  EUV (13.5nm)
                </button>
              </div>
            </div>

            {/* 공정 파라미터 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                공정 파라미터
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    노광 시간: {exposureTime}초
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    step="0.5"
                    value={exposureTime}
                    onChange={(e) => setExposureTime(parseFloat(e.target.value))}
                    className="w-full"
                    disabled={currentStep !== 'exposure'}
                  />
                </div>

                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">최소 해상도</span>
                    <span className="font-mono font-bold text-blue-600 dark:text-blue-400">
                      {resolution.toFixed(1)} nm
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* 현재 단계 설명 */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
              <h3 className="font-bold mb-2">💡 현재 단계</h3>
              <p className="text-sm leading-relaxed">
                {currentStep === 'coating' && '포토레지스트를 웨이퍼에 균일하게 도포합니다.'}
                {currentStep === 'exposure' && 'UV 광선으로 마스크 패턴을 레지스트에 전사합니다.'}
                {currentStep === 'development' && '노광된 부분을 현상액으로 제거합니다.'}
                {currentStep === 'etching' && '패턴이 형성된 영역을 제외하고 에칭합니다.'}
                {currentStep === 'complete' && '레지스트를 제거하고 최종 패턴이 완성됩니다.'}
              </p>
            </div>

            {/* 기술 정보 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                기술 정보
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">공정 노드:</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {wavelength === 13.5 ? '5nm' : wavelength === 193 ? '7nm' : '14nm'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">다중 패터닝:</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {wavelength === 13.5 ? 'Single' : 'LELE/SAQP'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">수율 예상:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {wavelength === 13.5 ? '75%' : wavelength === 193 ? '85%' : '92%'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
