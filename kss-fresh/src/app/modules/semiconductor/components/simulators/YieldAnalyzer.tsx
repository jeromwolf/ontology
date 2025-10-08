'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'

type DefectType = 'particle' | 'scratch' | 'pattern' | 'doping'

interface Defect {
  id: number
  x: number
  y: number
  type: DefectType
  severity: number
}

export default function YieldAnalyzer() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [waferSize, setWaferSize] = useState(300) // mm
  const [dieSize, setDieSize] = useState(10) // mm
  const [defectDensity, setDefectDensity] = useState(0.5) // defects/cm²
  const [defects, setDefects] = useState<Defect[]>([])
  const [goodDies, setGoodDies] = useState(0)
  const [totalDies, setTotalDies] = useState(0)
  const [yieldPercentage, setYieldPercentage] = useState(0)
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
    calculateYield()
  }, [waferSize, dieSize, defectDensity])

  const calculateYield = () => {
    const radius = waferSize / 2
    const area = Math.PI * Math.pow(radius / 10, 2) // cm²
    const dieArea = Math.pow(dieSize / 10, 2) // cm²
    const totalDefects = Math.floor(area * defectDensity)

    // 웨이퍼에 들어가는 다이 개수 계산
    const diesPerRow = Math.floor((waferSize - 20) / dieSize)
    const calculatedTotalDies = Math.floor(Math.PI * Math.pow(diesPerRow / 2, 2))
    setTotalDies(calculatedTotalDies)

    // 결함 생성
    const newDefects: Defect[] = []
    for (let i = 0; i < totalDefects; i++) {
      const angle = Math.random() * 2 * Math.PI
      const r = Math.sqrt(Math.random()) * radius
      newDefects.push({
        id: i,
        x: r * Math.cos(angle),
        y: r * Math.sin(angle),
        type: ['particle', 'scratch', 'pattern', 'doping'][Math.floor(Math.random() * 4)] as DefectType,
        severity: Math.random()
      })
    }
    setDefects(newDefects)

    // 수율 계산 (Murphy 모델)
    const D0 = defectDensity
    const A = dieArea
    const Y = Math.pow((1 - Math.exp(-D0 * A)) / (D0 * A), 2) * 100

    setYieldPercentage(Y)
    setGoodDies(Math.floor(calculatedTotalDies * Y / 100))
  }

  const getDefectColor = (type: DefectType) => {
    switch (type) {
      case 'particle': return '#e74c3c'
      case 'scratch': return '#f39c12'
      case 'pattern': return '#9b59b6'
      case 'doping': return '#3498db'
    }
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
            수율 분석기
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            반도체 제조 공정의 수율을 분석하고 결함을 시각화하세요
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 웨이퍼 시각화 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                {waferSize}mm 웨이퍼 맵
              </h3>
              <div className="relative bg-gray-900 rounded-lg p-8 flex items-center justify-center" style={{ height: '500px' }}>
                <svg viewBox="-200 -200 400 400" className="w-full h-full">
                  {/* 웨이퍼 원 */}
                  <circle cx="0" cy="0" r={waferSize / 2} fill="#718093" stroke="#2c3e50" strokeWidth="2" />

                  {/* 웨이퍼 flat (방향 표시) */}
                  <line x1={-waferSize / 4} y1={waferSize / 2} x2={waferSize / 4} y2={waferSize / 2} stroke="#2c3e50" strokeWidth="3" />

                  {/* 다이 그리드 */}
                  {Array.from({ length: Math.floor(waferSize / dieSize) + 1 }, (_, i) => {
                    const offset = -waferSize / 2 + i * dieSize
                    return (
                      <g key={i}>
                        {/* 세로선 */}
                        <line
                          x1={offset}
                          y1={-waferSize / 2}
                          x2={offset}
                          y2={waferSize / 2}
                          stroke="#34495e"
                          strokeWidth="0.5"
                          opacity="0.3"
                        />
                        {/* 가로선 */}
                        <line
                          x1={-waferSize / 2}
                          y1={offset}
                          x2={waferSize / 2}
                          y2={offset}
                          stroke="#34495e"
                          strokeWidth="0.5"
                          opacity="0.3"
                        />
                      </g>
                    )
                  })}

                  {/* 결함 표시 */}
                  {defects.map((defect) => (
                    <circle
                      key={defect.id}
                      cx={defect.x}
                      cy={defect.y}
                      r={3 + defect.severity * 3}
                      fill={getDefectColor(defect.type)}
                      opacity="0.8"
                    />
                  ))}
                </svg>
              </div>
            </div>

            {/* 수율 트렌드 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mt-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                수율 트렌드
              </h3>
              <div className="relative h-64 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <svg viewBox="0 0 400 200" className="w-full h-full">
                  {/* 축 */}
                  <line x1="50" y1="150" x2="350" y2="150" stroke="currentColor" strokeWidth="2" className="text-gray-400" />
                  <line x1="50" y1="20" x2="50" y2="150" stroke="currentColor" strokeWidth="2" className="text-gray-400" />

                  {/* 수율 곡선 (결함 밀도에 따른) */}
                  <path
                    d={`M 50,${150 - yieldPercentage * 1.2} L ${50 + defectDensity * 200},${150 - yieldPercentage * 1.2}`}
                    fill="none"
                    stroke="#2ecc71"
                    strokeWidth="3"
                  />

                  {/* 목표선 */}
                  <line x1="50" y1="60" x2="350" y2="60" stroke="#e74c3c" strokeWidth="2" strokeDasharray="5,5" />
                  <text x="360" y="65" fontSize="12" fill="#e74c3c">목표 (75%)</text>

                  {/* 현재 수율 포인트 */}
                  <circle cx={50 + defectDensity * 200} cy={150 - yieldPercentage * 1.2} r="6" fill="#00d2ff" />

                  {/* 라벨 */}
                  <text x="150" y="170" fontSize="14" fill="currentColor" className="text-gray-600">결함 밀도 (defects/cm²)</text>
                  <text x="10" y="100" fontSize="14" fill="currentColor" className="text-gray-600" transform="rotate(-90 10 100)">수율 (%)</text>
                </svg>
              </div>
            </div>
          </div>

          {/* 제어판 */}
          <div className="space-y-6">
            {/* 웨이퍼 설정 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                웨이퍼 설정
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    웨이퍼 크기: {waferSize}mm
                  </label>
                  <div className="flex gap-2">
                    {[200, 300, 450].map((size) => (
                      <button
                        key={size}
                        onClick={() => setWaferSize(size)}
                        className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium ${
                          waferSize === size
                            ? 'bg-blue-500 text-white'
                            : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                        }`}
                      >
                        {size}mm
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    다이 크기: {dieSize}mm
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="20"
                    step="1"
                    value={dieSize}
                    onChange={(e) => setDieSize(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    결함 밀도: {defectDensity.toFixed(2)} defects/cm²
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="2"
                    step="0.1"
                    value={defectDensity}
                    onChange={(e) => setDefectDensity(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            {/* 수율 분석 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                수율 분석
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">전체 다이</span>
                  <span className="font-mono font-bold text-blue-600 dark:text-blue-400">
                    {totalDies}개
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">양품 다이</span>
                  <span className="font-mono font-bold text-green-600 dark:text-green-400">
                    {goodDies}개
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <span className="text-sm text-gray-600 dark:text-gray-400">불량 다이</span>
                  <span className="font-mono font-bold text-red-600 dark:text-red-400">
                    {totalDies - goodDies}개
                  </span>
                </div>
                <div className="p-4 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg">
                  <div className="text-white text-center">
                    <div className="text-sm opacity-90 mb-1">수율</div>
                    <div className="text-3xl font-bold">{yieldPercentage.toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            </div>

            {/* 결함 유형 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                결함 유형
              </h3>
              <div className="space-y-2">
                {[
                  { type: 'particle', label: '파티클', color: '#e74c3c' },
                  { type: 'scratch', label: '스크래치', color: '#f39c12' },
                  { type: 'pattern', label: '패턴 오류', color: '#9b59b6' },
                  { type: 'doping', label: '도핑 불량', color: '#3498db' }
                ].map((item) => {
                  const count = defects.filter((d) => d.type === item.type).length
                  return (
                    <div key={item.type} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded" style={{ backgroundColor: item.color }} />
                        <span className="text-sm text-gray-700 dark:text-gray-300">{item.label}</span>
                      </div>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">{count}개</span>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* 개선 제안 */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
              <h3 className="font-bold mb-2">💡 개선 제안</h3>
              <p className="text-sm leading-relaxed">
                {yieldPercentage > 80 && '수율이 우수합니다. 현재 공정을 유지하세요.'}
                {yieldPercentage >= 60 && yieldPercentage <= 80 && '결함 밀도를 낮추기 위해 클린룸 관리를 강화하세요.'}
                {yieldPercentage < 60 && '수율이 낮습니다. 공정 파라미터를 점검하고 결함 원인을 분석하세요.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
