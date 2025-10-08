'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'

export default function WaferFab() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [currentProcess, setCurrentProcess] = useState(0)
  const [temperature, setTemperature] = useState(1100) // Celsius
  const [pressure, setPressure] = useState(1) // atm
  const [gasFlow, setGasFlow] = useState(50) // sccm
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

  const processes = [
    {
      name: 'CZ 결정 성장',
      description: '단결정 실리콘 잉곳을 제조합니다',
      temp: '1414°C (Si 융점)',
      time: '24-48시간',
      color: '#e74c3c'
    },
    {
      name: '잉곳 슬라이싱',
      description: '다이아몬드 와이어로 웨이퍼를 절단합니다',
      temp: '상온',
      time: '8-12시간',
      color: '#3498db'
    },
    {
      name: '연마 (Lapping)',
      description: '웨이퍼 표면을 평탄하게 만듭니다',
      temp: '상온',
      time: '4-6시간',
      color: '#2ecc71'
    },
    {
      name: '에칭',
      description: '화학적으로 손상층을 제거합니다',
      temp: '80-100°C',
      time: '2-3시간',
      color: '#f39c12'
    },
    {
      name: 'CMP',
      description: '화학적 기계적 연마로 거울면을 만듭니다',
      temp: '상온',
      time: '3-5시간',
      color: '#9b59b6'
    },
    {
      name: '검사 & 포장',
      description: '품질 검사 후 클린룸 포장합니다',
      temp: '상온',
      time: '2-4시간',
      color: '#1abc9c'
    }
  ]

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
            웨이퍼 제조 시뮬레이터
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            단결정 실리콘 웨이퍼 제조 공정을 단계별로 학습하세요
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 공정 시각화 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                {processes[currentProcess].name}
              </h3>

              <div className="relative bg-gray-900 rounded-lg p-8 h-96 flex items-center justify-center">
                <svg viewBox="0 0 400 300" className="w-full h-full">
                  {/* CZ 결정 성장 */}
                  {currentProcess === 0 && (
                    <>
                      {/* 도가니 */}
                      <ellipse cx="200" cy="250" rx="80" ry="20" fill="#ff6b6b" />
                      <rect x="120" y="180" width="160" height="70" fill="#ff6b6b" opacity="0.8" />
                      <ellipse cx="200" cy="180" rx="80" ry="20" fill="#ff8787" />

                      {/* 용융 실리콘 */}
                      <ellipse cx="200" cy="220" rx="70" ry="15" fill="#ffd93d" />

                      {/* 시드 크리스탈 */}
                      <rect x="195" y="50" width="10" height="130" fill="#718093" />
                      <rect x="190" y="180" width="20" height="40" fill="#95a5a6" />

                      {/* 회전 화살표 */}
                      <path d="M 220 100 Q 240 100 240 120" stroke="#00d2ff" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />

                      <text x="120" y="30" fill="white" fontSize="14">온도: {temperature}°C</text>
                      <text x="120" y="50" fill="white" fontSize="14">회전: 10 rpm</text>
                    </>
                  )}

                  {/* 슬라이싱 */}
                  {currentProcess === 1 && (
                    <>
                      {/* 실리콘 잉곳 */}
                      <rect x="100" y="100" width="60" height="150" fill="#718093" rx="30" />

                      {/* 다이아몬드 와이어 */}
                      <line x1="80" y1="150" x2="180" y2="150" stroke="#00d2ff" strokeWidth="2" strokeDasharray="5,5" />
                      <line x1="80" y1="180" x2="180" y2="180" stroke="#00d2ff" strokeWidth="2" strokeDasharray="5,5" />
                      <line x1="80" y1="210" x2="180" y2="210" stroke="#00d2ff" strokeWidth="2" strokeDasharray="5,5" />

                      {/* 절단된 웨이퍼들 */}
                      {[0, 1, 2, 3, 4].map((i) => (
                        <ellipse key={i} cx={250 + i * 25} cy="180" rx="20" ry="3" fill="#95a5a6" />
                      ))}

                      <text x="220" y="230" fill="white" fontSize="14">웨이퍼 두께: 775µm</text>
                    </>
                  )}

                  {/* 연마 */}
                  {currentProcess === 2 && (
                    <>
                      {/* 웨이퍼 */}
                      <ellipse cx="200" cy="150" rx="80" ry="10" fill="#718093" />

                      {/* 연마 패드 */}
                      <ellipse cx="200" cy="100" rx="90" ry="15" fill="#2ecc71" opacity="0.6" />

                      {/* 연마 입자 */}
                      {Array.from({ length: 20 }).map((_, i) => (
                        <circle
                          key={i}
                          cx={150 + Math.random() * 100}
                          cy={120 + Math.random() * 30}
                          r="2"
                          fill="#27ae60"
                        />
                      ))}

                      {/* 압력 화살표 */}
                      <path d="M 200 60 L 200 90" stroke="#00d2ff" strokeWidth="3" markerEnd="url(#arrow)" />

                      <text x="120" y="200" fill="white" fontSize="14">압력: {pressure} atm</text>
                      <text x="120" y="220" fill="white" fontSize="14">슬러리: Al₂O₃</text>
                    </>
                  )}

                  {/* 에칭 */}
                  {currentProcess === 3 && (
                    <>
                      {/* 에칭 탱크 */}
                      <rect x="100" y="80" width="200" height="180" fill="#34495e" rx="10" />

                      {/* 용액 */}
                      <rect x="110" y="150" width="180" height="100" fill="#f39c12" opacity="0.6" />

                      {/* 웨이퍼 */}
                      <ellipse cx="200" cy="180" rx="70" ry="8" fill="#95a5a6" />

                      {/* 기포 */}
                      {Array.from({ length: 15 }).map((_, i) => (
                        <circle
                          key={i}
                          cx={130 + Math.random() * 140}
                          cy={160 + Math.random() * 80}
                          r={2 + Math.random() * 3}
                          fill="#ffd93d"
                          opacity="0.6"
                        />
                      ))}

                      <text x="120" y="120" fill="white" fontSize="14">HF:HNO₃:CH₃COOH</text>
                      <text x="120" y="140" fill="white" fontSize="14">온도: {temperature}°C</text>
                    </>
                  )}

                  {/* CMP */}
                  {currentProcess === 4 && (
                    <>
                      {/* CMP 패드 */}
                      <ellipse cx="200" cy="100" rx="100" ry="20" fill="#9b59b6" />

                      {/* 웨이퍼 */}
                      <ellipse cx="200" cy="150" rx="80" ry="10" fill="#ecf0f1" />

                      {/* 슬러리 */}
                      {Array.from({ length: 30 }).map((_, i) => (
                        <circle
                          key={i}
                          cx={120 + Math.random() * 160}
                          cy={120 + Math.random() * 40}
                          r="1.5"
                          fill="#8e44ad"
                          opacity="0.6"
                        />
                      ))}

                      {/* 회전 표시 */}
                      <text x="250" y="110" fill="white" fontSize="16">⟲</text>

                      <text x="120" y="200" fill="white" fontSize="14">슬러리: SiO₂ + KOH</text>
                      <text x="120" y="220" fill="white" fontSize="14">회전: 70 rpm</text>
                      <text x="120" y="240" fill="white" fontSize="14">표면 거칠기: &lt; 0.5nm</text>
                    </>
                  )}

                  {/* 검사 */}
                  {currentProcess === 5 && (
                    <>
                      {/* 검사 장비 */}
                      <rect x="100" y="80" width="200" height="40" fill="#34495e" rx="5" />
                      <circle cx="200" cy="100" r="15" fill="#00d2ff" opacity="0.6" />

                      {/* 레이저 빔 */}
                      <line x1="200" y1="120" x2="200" y2="160" stroke="#00d2ff" strokeWidth="2" />

                      {/* 웨이퍼 */}
                      <ellipse cx="200" cy="180" rx="80" ry="10" fill="#ecf0f1" />

                      {/* 합격 표시 */}
                      <circle cx="200" cy="180" r="30" stroke="#2ecc71" strokeWidth="3" fill="none" />
                      <path d="M 180 180 L 195 195 L 220 165" stroke="#2ecc71" strokeWidth="4" fill="none" />

                      <text x="120" y="240" fill="white" fontSize="14">✓ TTV &lt; 1µm</text>
                      <text x="120" y="260" fill="white" fontSize="14">✓ Bow/Warp 검사 완료</text>
                    </>
                  )}

                  {/* Arrow marker definition */}
                  <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                      <polygon points="0 0, 10 3, 0 6" fill="#00d2ff" />
                    </marker>
                  </defs>
                </svg>
              </div>

              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>설명:</strong> {processes[currentProcess].description}
                </p>
                <div className="grid grid-cols-2 gap-4 mt-3 text-sm">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">온도:</span>{' '}
                    <span className="font-medium text-gray-900 dark:text-white">{processes[currentProcess].temp}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">소요시간:</span>{' '}
                    <span className="font-medium text-gray-900 dark:text-white">{processes[currentProcess].time}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* 공정 단계 네비게이션 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mt-6">
              <div className="flex gap-2 overflow-x-auto">
                {processes.map((process, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentProcess(index)}
                    className={`flex-1 min-w-[120px] px-4 py-3 rounded-lg font-medium transition-all text-sm ${
                      currentProcess === index
                        ? 'text-white shadow-lg'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                    style={{
                      backgroundColor: currentProcess === index ? process.color : undefined
                    }}
                  >
                    {index + 1}. {process.name}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* 제어판 */}
          <div className="space-y-6">
            {/* 공정 파라미터 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                공정 파라미터
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    온도: {temperature}°C
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="1500"
                    step="10"
                    value={temperature}
                    onChange={(e) => setTemperature(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    압력: {pressure} atm
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="5"
                    step="0.1"
                    value={pressure}
                    onChange={(e) => setPressure(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-600 dark:text-gray-400 mb-2 block">
                    가스 유량: {gasFlow} sccm
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="200"
                    step="5"
                    value={gasFlow}
                    onChange={(e) => setGasFlow(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            {/* 웨이퍼 사양 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                웨이퍼 사양
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-600 dark:text-gray-400">직경</span>
                  <span className="font-medium text-gray-900 dark:text-white">300mm (12")</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-600 dark:text-gray-400">두께</span>
                  <span className="font-medium text-gray-900 dark:text-white">775µm</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-600 dark:text-gray-400">결정 방향</span>
                  <span className="font-medium text-gray-900 dark:text-white">&lt;100&gt;</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-600 dark:text-gray-400">도핑 타입</span>
                  <span className="font-medium text-gray-900 dark:text-white">P-type (Boron)</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-600 dark:text-gray-400">저항률</span>
                  <span className="font-medium text-gray-900 dark:text-white">1-10 Ω·cm</span>
                </div>
              </div>
            </div>

            {/* 품질 지표 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                품질 지표
              </h3>
              <div className="space-y-3">
                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">TTV</span>
                    <span className="font-mono font-bold text-green-600 dark:text-green-400">0.8 µm</span>
                  </div>
                </div>
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Flatness</span>
                    <span className="font-mono font-bold text-blue-600 dark:text-blue-400">0.5 µm</span>
                  </div>
                </div>
                <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">표면 거칠기</span>
                    <span className="font-mono font-bold text-purple-600 dark:text-purple-400">0.3 nm</span>
                  </div>
                </div>
              </div>
            </div>

            {/* 정보 */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
              <h3 className="font-bold mb-2">💡 공정 핵심</h3>
              <p className="text-sm leading-relaxed">
                웨이퍼 제조는 고순도 단결정 실리콘을 만드는 과정입니다.
                CZ법으로 성장시킨 잉곳을 슬라이싱하고 연마하여 거울처럼 매끄러운 표면을 만듭니다.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
