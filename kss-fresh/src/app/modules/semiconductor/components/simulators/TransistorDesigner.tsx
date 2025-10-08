'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'

type TransistorType = 'NMOS' | 'PMOS' | 'FinFET' | 'GAA'

export default function TransistorDesigner() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [transistorType, setTransistorType] = useState<TransistorType>('NMOS')
  const [gateLength, setGateLength] = useState(7) // nm
  const [gateWidth, setGateWidth] = useState(100) // nm
  const [oxideThickness, setOxideThickness] = useState(1.5) // nm
  const [vgs, setVgs] = useState(0) // Gate-Source voltage
  const [vds, setVds] = useState(0) // Drain-Source voltage
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

  // 드레인 전류 계산 (간단한 MOSFET 모델)
  const calculateDrainCurrent = () => {
    const Cox = 3.45e-11 // F/cm^2 (SiO2 capacitance)
    const mu = 400 // cm^2/V·s (electron mobility)
    const Vth = 0.4 // Threshold voltage

    if (vgs < Vth) return 0

    const W_L = gateWidth / gateLength
    const Id = (mu * Cox * W_L * (vgs - Vth) * vds) / 1000 // Convert to mA

    return Math.max(0, Math.min(Id, 100)) // Limit to 100mA
  }

  const calculatePerformance = () => {
    const id = calculateDrainCurrent()
    const ft = (id * 1000) / (2 * Math.PI * gateLength) // Cutoff frequency (simplified)
    const power = id * vds // Power consumption

    return {
      current: id.toFixed(2),
      frequency: (ft / 1e9).toFixed(1), // GHz
      power: power.toFixed(2), // mW
      density: ((gateWidth * gateLength) / 1000).toFixed(2) // µm^2
    }
  }

  const performance = calculatePerformance()

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
            트랜지스터 설계 도구
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            다양한 MOSFET 구조를 설계하고 성능을 분석하세요
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 3D 시각화 영역 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                {transistorType} 구조
              </h3>
              <div className="relative h-96 bg-gradient-to-br from-gray-900 to-gray-800 rounded-lg flex items-center justify-center">
                {/* 간단한 트랜지스터 구조 시각화 */}
                <svg viewBox="0 0 400 300" className="w-full h-full">
                  {/* NMOS 기본 구조 */}
                  {transistorType === 'NMOS' && (
                    <>
                      {/* 기판 (P-type) */}
                      <rect x="50" y="200" width="300" height="80" fill="#ff6b9d" />
                      <text x="180" y="250" fill="white" fontSize="16">P-type 기판</text>

                      {/* N+ Source/Drain */}
                      <rect x="70" y="180" width="60" height="40" fill="#4a69bd" />
                      <rect x="270" y="180" width="60" height="40" fill="#4a69bd" />
                      <text x="85" y="205" fill="white" fontSize="12">Source</text>
                      <text x="285" y="205" fill="white" fontSize="12">Drain</text>

                      {/* Gate Oxide */}
                      <rect x="130" y="170" width="140" height={oxideThickness * 5} fill="#2ecc71" />

                      {/* Gate */}
                      <rect x="130" y={170 - gateLength * 2} width="140" height={gateLength * 2} fill="#f39c12" />
                      <text x="175" y={165 - gateLength * 2} fill="white" fontSize="12">Gate ({gateLength}nm)</text>

                      {/* Channel */}
                      <rect x="130" y={170 + oxideThickness * 5} width="140" height="10" fill={vgs > 0.4 ? '#4a69bd' : 'transparent'} opacity="0.6" />
                    </>
                  )}

                  {/* FinFET 구조 */}
                  {transistorType === 'FinFET' && (
                    <>
                      {/* Fin 구조 */}
                      <rect x="180" y="120" width="40" height="120" fill="#4a69bd" />
                      <text x="160" y="250" fill="white" fontSize="12">Fin</text>

                      {/* Gate (3면 감싸는 형태) */}
                      <path d="M 160,140 L 180,140 L 180,200 L 160,200 Z" fill="#f39c12" opacity="0.8" />
                      <rect x="180" y="140" width="40" height="60" fill="#f39c12" opacity="0.5" />
                      <path d="M 220,140 L 240,140 L 240,200 L 220,200 Z" fill="#f39c12" opacity="0.8" />
                      <text x="160" y="130" fill="white" fontSize="12">Gate (3면)</text>

                      {/* Source/Drain */}
                      <rect x="180" y="100" width="40" height="20" fill="#2ecc71" />
                      <rect x="180" y="220" width="40" height="20" fill="#2ecc71" />
                    </>
                  )}

                  {/* GAA 구조 */}
                  {transistorType === 'GAA' && (
                    <>
                      {/* Nanosheet 층 */}
                      {[0, 1, 2].map((i) => (
                        <g key={i}>
                          <rect x="150" y={140 + i * 40} width="100" height="20" fill="#4a69bd" />
                          <rect x="140" y={135 + i * 40} width="120" height="30" fill="#f39c12" opacity="0.5" />
                        </g>
                      ))}
                      <text x="140" y="120" fill="white" fontSize="12">Gate-All-Around</text>
                      <text x="150" y="260" fill="white" fontSize="12">3-layer Nanosheet</text>
                    </>
                  )}

                  {/* 전압 표시 */}
                  <text x="20" y="30" fill="#00d2ff" fontSize="14">Vgs = {vgs}V</text>
                  <text x="20" y="50" fill="#00d2ff" fontSize="14">Vds = {vds}V</text>
                </svg>
              </div>
            </div>

            {/* I-V 특성 곡선 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mt-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                I-V 특성 곡선
              </h3>
              <div className="relative h-64 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <svg viewBox="0 0 400 200" className="w-full h-full">
                  {/* 축 */}
                  <line x1="50" y1="150" x2="350" y2="150" stroke="currentColor" strokeWidth="2" className="text-gray-400" />
                  <line x1="50" y1="20" x2="50" y2="150" stroke="currentColor" strokeWidth="2" className="text-gray-400" />

                  {/* I-V 곡선 (선형 & 포화 영역) */}
                  {[1.0, 1.5, 2.0, 2.5].map((vg, i) => (
                    <path
                      key={i}
                      d={`M 50,${150 - i * 25} Q 150,${150 - i * 25 - 10} 200,${150 - i * 30} L 350,${150 - i * 30}`}
                      fill="none"
                      stroke={`hsl(${200 + i * 20}, 70%, 60%)`}
                      strokeWidth="2"
                    />
                  ))}

                  {/* 현재 동작점 */}
                  <circle cx={50 + vds * 60} cy={150 - calculateDrainCurrent() * 1.5} r="5" fill="#00d2ff" />

                  {/* 라벨 */}
                  <text x="360" y="155" fontSize="14" fill="currentColor">Vds (V)</text>
                  <text x="30" y="15" fontSize="14" fill="currentColor">Id (mA)</text>
                </svg>
              </div>
            </div>
          </div>

          {/* 제어판 */}
          <div className="space-y-6">
            {/* 트랜지스터 타입 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                트랜지스터 타입
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {(['NMOS', 'PMOS', 'FinFET', 'GAA'] as TransistorType[]).map((type) => (
                  <button
                    key={type}
                    onClick={() => setTransistorType(type)}
                    className={`px-4 py-2 rounded-lg font-medium transition-all text-sm ${
                      transistorType === type
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* 설계 파라미터 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                설계 파라미터
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    게이트 길이: {gateLength}nm
                  </label>
                  <input
                    type="range"
                    min="3"
                    max="14"
                    step="1"
                    value={gateLength}
                    onChange={(e) => setGateLength(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    게이트 폭: {gateWidth}nm
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="500"
                    step="10"
                    value={gateWidth}
                    onChange={(e) => setGateWidth(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    산화막 두께: {oxideThickness}nm
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="3"
                    step="0.1"
                    value={oxideThickness}
                    onChange={(e) => setOxideThickness(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            {/* 바이어스 제어 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                바이어스 제어
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    Vgs: {vgs.toFixed(1)}V
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="3"
                    step="0.1"
                    value={vgs}
                    onChange={(e) => setVgs(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    Vds: {vds.toFixed(1)}V
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="3"
                    step="0.1"
                    value={vds}
                    onChange={(e) => setVds(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            {/* 성능 지표 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                성능 지표
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">드레인 전류</span>
                  <span className="font-mono font-bold text-blue-600 dark:text-blue-400">
                    {performance.current} mA
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">차단 주파수</span>
                  <span className="font-mono font-bold text-purple-600 dark:text-purple-400">
                    {performance.frequency} GHz
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">소비 전력</span>
                  <span className="font-mono font-bold text-red-600 dark:text-red-400">
                    {performance.power} mW
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">집적 밀도</span>
                  <span className="font-mono font-bold text-green-600 dark:text-green-400">
                    {performance.density} µm²
                  </span>
                </div>
              </div>
            </div>

            {/* 기술 비교 */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
              <h3 className="font-bold mb-2">💡 기술 특징</h3>
              <p className="text-sm leading-relaxed">
                {transistorType === 'NMOS' && '평면 MOSFET - 가장 기본적인 구조로 이해하기 쉽습니다.'}
                {transistorType === 'PMOS' && '정공을 캐리어로 사용하는 PMOS - CMOS 회로의 필수 요소입니다.'}
                {transistorType === 'FinFET' && '3면 게이트로 누설 전류 감소 - 7nm 이하 공정의 핵심 기술입니다.'}
                {transistorType === 'GAA' && 'Gate-All-Around - 3nm 이하 공정의 차세대 기술입니다.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
