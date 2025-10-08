'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'

type ArchType = 'CPU' | 'GPU' | 'NPU' | 'SoC'

interface Block {
  id: string
  name: string
  x: number
  y: number
  width: number
  height: number
  color: string
}

export default function ChipArchitecture() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [archType, setArchType] = useState<ArchType>('CPU')
  const [showDataflow, setShowDataflow] = useState(false)
  const [hoveredBlock, setHoveredBlock] = useState<string | null>(null)
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

  const architectures: Record<ArchType, Block[]> = {
    CPU: [
      { id: 'cores', name: 'CPU 코어 (8개)', x: 50, y: 50, width: 200, height: 150, color: '#3498db' },
      { id: 'cache', name: 'L3 캐시 (32MB)', x: 280, y: 50, width: 100, height: 150, color: '#9b59b6' },
      { id: 'memory', name: '메모리 컨트롤러', x: 50, y: 230, width: 150, height: 80, color: '#2ecc71' },
      { id: 'pcie', name: 'PCIe 컨트롤러', x: 230, y: 230, width: 150, height: 80, color: '#f39c12' }
    ],
    GPU: [
      { id: 'sm', name: 'SM (128개)', x: 50, y: 50, width: 180, height: 120, color: '#2ecc71' },
      { id: 'tensor', name: 'Tensor Core', x: 250, y: 50, width: 130, height: 120, color: '#e74c3c' },
      { id: 'memory', name: 'HBM 메모리', x: 50, y: 190, width: 150, height: 80, color: '#9b59b6' },
      { id: 'nvlink', name: 'NVLink', x: 230, y: 190, width: 150, height: 80, color: '#f39c12' }
    ],
    NPU: [
      { id: 'mac', name: 'MAC 어레이 (512x512)', x: 50, y: 50, width: 200, height: 150, color: '#e74c3c' },
      { id: 'buffer', name: '가중치 버퍼', x: 280, y: 50, width: 100, height: 70, color: '#9b59b6' },
      { id: 'activation', name: '활성화 메모리', x: 280, y: 140, width: 100, height: 60, color: '#3498db' },
      { id: 'controller', name: '시스톨릭 컨트롤러', x: 50, y: 230, width: 330, height: 80, color: '#2ecc71' }
    ],
    SoC: [
      { id: 'cpu', name: 'CPU (4+4)', x: 50, y: 50, width: 120, height: 80, color: '#3498db' },
      { id: 'gpu', name: 'GPU', x: 190, y: 50, width: 120, height: 80, color: '#2ecc71' },
      { id: 'npu', name: 'NPU', x: 330, y: 50, width: 80, height: 80, color: '#e74c3c' },
      { id: 'modem', name: '5G 모뎀', x: 50, y: 150, width: 100, height: 60, color: '#f39c12' },
      { id: 'isp', name: 'ISP', x: 170, y: 150, width: 100, height: 60, color: '#9b59b6' },
      { id: 'memory', name: '메모리', x: 290, y: 150, width: 120, height: 60, color: '#34495e' },
      { id: 'security', name: '보안 엔진', x: 50, y: 230, width: 180, height: 60, color: '#1abc9c' },
      { id: 'interconnect', name: 'NoC 인터커넥트', x: 250, y: 230, width: 160, height: 60, color: '#95a5a6' }
    ]
  }

  const getBlockInfo = (id: string) => {
    const info: Record<string, string> = {
      cores: 'x86-64 아키텍처, 5GHz 부스트 클럭',
      cache: '공유 L3 캐시, 64-way set associative',
      memory: 'DDR5-5600, 듀얼 채널 지원',
      pcie: 'PCIe 5.0 x16, 64GT/s',
      sm: 'CUDA 코어, FP32/FP16 연산',
      tensor: 'AI 가속, FP16/INT8/INT4',
      nvlink: 'GPU 간 고속 연결, 900GB/s',
      mac: 'INT8 곱셈-누적 연산기',
      buffer: 'On-chip SRAM, 10MB',
      activation: 'Feature map 저장',
      controller: '데이터 플로우 제어',
      cpu: 'big.LITTLE 구조',
      gpu: '모바일 그래픽 가속',
      npu: '온디바이스 AI 추론',
      modem: 'Sub-6GHz + mmWave',
      isp: '이미지 신호 처리',
      security: 'TEE, 암호화 엔진',
      interconnect: '블록 간 통신 버스'
    }
    return info[id] || '정보 없음'
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
            칩 아키텍처 시각화
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            다양한 프로세서 아키텍처의 블록 다이어그램을 살펴보세요
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 아키텍처 다이어그램 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  {archType} 아키텍처
                </h3>
                <button
                  onClick={() => setShowDataflow(!showDataflow)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all text-sm ${
                    showDataflow
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  데이터 흐름 {showDataflow ? 'ON' : 'OFF'}
                </button>
              </div>

              <div className="relative bg-gray-900 rounded-lg p-8">
                <svg viewBox="0 0 450 350" className="w-full h-full">
                  {/* 데이터 흐름 표시 */}
                  {showDataflow && (
                    <g>
                      <defs>
                        <marker
                          id="arrowhead"
                          markerWidth="10"
                          markerHeight="10"
                          refX="9"
                          refY="3"
                          orient="auto"
                        >
                          <polygon points="0 0, 10 3, 0 6" fill="#00d2ff" />
                        </marker>
                      </defs>
                      {architectures[archType].map((block, i) => {
                        if (i < architectures[archType].length - 1) {
                          const nextBlock = architectures[archType][i + 1]
                          return (
                            <line
                              key={`flow-${i}`}
                              x1={block.x + block.width / 2}
                              y1={block.y + block.height}
                              x2={nextBlock.x + nextBlock.width / 2}
                              y2={nextBlock.y}
                              stroke="#00d2ff"
                              strokeWidth="2"
                              markerEnd="url(#arrowhead)"
                              opacity="0.6"
                            />
                          )
                        }
                        return null
                      })}
                    </g>
                  )}

                  {/* 아키텍처 블록 */}
                  {architectures[archType].map((block) => (
                    <g
                      key={block.id}
                      onMouseEnter={() => setHoveredBlock(block.id)}
                      onMouseLeave={() => setHoveredBlock(null)}
                      style={{ cursor: 'pointer' }}
                    >
                      <rect
                        x={block.x}
                        y={block.y}
                        width={block.width}
                        height={block.height}
                        fill={block.color}
                        opacity={hoveredBlock === block.id ? 1 : 0.8}
                        rx="8"
                        stroke={hoveredBlock === block.id ? '#00d2ff' : 'white'}
                        strokeWidth={hoveredBlock === block.id ? '3' : '2'}
                      />
                      <text
                        x={block.x + block.width / 2}
                        y={block.y + block.height / 2}
                        fill="white"
                        fontSize="14"
                        fontWeight="bold"
                        textAnchor="middle"
                        dominantBaseline="middle"
                      >
                        {block.name}
                      </text>
                    </g>
                  ))}
                </svg>

                {/* Hover 정보 */}
                {hoveredBlock && (
                  <div className="absolute bottom-4 left-4 bg-black/80 text-white px-4 py-2 rounded-lg text-sm">
                    {getBlockInfo(hoveredBlock)}
                  </div>
                )}
              </div>
            </div>

            {/* 사양 비교 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mt-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                사양 비교
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left p-2 text-gray-600 dark:text-gray-400">항목</th>
                      <th className="text-left p-2 text-gray-900 dark:text-white">CPU</th>
                      <th className="text-left p-2 text-gray-900 dark:text-white">GPU</th>
                      <th className="text-left p-2 text-gray-900 dark:text-white">NPU</th>
                      <th className="text-left p-2 text-gray-900 dark:text-white">SoC</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="p-2 text-gray-600 dark:text-gray-400">트랜지스터</td>
                      <td className="p-2">25B</td>
                      <td className="p-2">80B</td>
                      <td className="p-2">40B</td>
                      <td className="p-2">15B</td>
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="p-2 text-gray-600 dark:text-gray-400">다이 크기</td>
                      <td className="p-2">350mm²</td>
                      <td className="p-2">600mm²</td>
                      <td className="p-2">400mm²</td>
                      <td className="p-2">120mm²</td>
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="p-2 text-gray-600 dark:text-gray-400">공정</td>
                      <td className="p-2">5nm</td>
                      <td className="p-2">4nm</td>
                      <td className="p-2">5nm</td>
                      <td className="p-2">4nm</td>
                    </tr>
                    <tr>
                      <td className="p-2 text-gray-600 dark:text-gray-400">TDP</td>
                      <td className="p-2">125W</td>
                      <td className="p-2">350W</td>
                      <td className="p-2">75W</td>
                      <td className="p-2">5W</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* 제어판 */}
          <div className="space-y-6">
            {/* 아키텍처 선택 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                아키텍처 선택
              </h3>
              <div className="space-y-3">
                {(['CPU', 'GPU', 'NPU', 'SoC'] as ArchType[]).map((type) => (
                  <button
                    key={type}
                    onClick={() => setArchType(type)}
                    className={`w-full px-4 py-3 rounded-lg font-medium transition-all ${
                      archType === type
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* 성능 지표 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                성능 지표
              </h3>
              <div className="space-y-3">
                {archType === 'CPU' && (
                  <>
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">싱글코어</span>
                        <span className="font-mono font-bold text-blue-600 dark:text-blue-400">2100</span>
                      </div>
                    </div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">멀티코어</span>
                        <span className="font-mono font-bold text-purple-600 dark:text-purple-400">18500</span>
                      </div>
                    </div>
                  </>
                )}
                {archType === 'GPU' && (
                  <>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">FP32 TFLOPS</span>
                        <span className="font-mono font-bold text-green-600 dark:text-green-400">82</span>
                      </div>
                    </div>
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Tensor TFLOPS</span>
                        <span className="font-mono font-bold text-red-600 dark:text-red-400">660</span>
                      </div>
                    </div>
                  </>
                )}
                {archType === 'NPU' && (
                  <>
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">INT8 TOPS</span>
                        <span className="font-mono font-bold text-red-600 dark:text-red-400">450</span>
                      </div>
                    </div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">효율성</span>
                        <span className="font-mono font-bold text-purple-600 dark:text-purple-400">6 TOPS/W</span>
                      </div>
                    </div>
                  </>
                )}
                {archType === 'SoC' && (
                  <>
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Geekbench</span>
                        <span className="font-mono font-bold text-blue-600 dark:text-blue-400">5800</span>
                      </div>
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">AI 성능</span>
                        <span className="font-mono font-bold text-green-600 dark:text-green-400">35 TOPS</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* 특징 */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
              <h3 className="font-bold mb-2">💡 주요 특징</h3>
              <p className="text-sm leading-relaxed">
                {archType === 'CPU' && '범용 연산에 최적화된 구조로 복잡한 제어 흐름과 순차 처리에 강점'}
                {archType === 'GPU' && '대규모 병렬 처리에 특화되어 그래픽과 AI 학습에 이상적'}
                {archType === 'NPU' && 'AI 추론 전용 가속기로 낮은 전력으로 고효율 연산 가능'}
                {archType === 'SoC' && '모든 기능을 하나의 칩에 통합하여 모바일 기기에 최적화'}
              </p>
            </div>

            {/* 응용 분야 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                주요 응용 분야
              </h3>
              <div className="space-y-2 text-sm">
                {archType === 'CPU' && (
                  <>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">데스크톱/서버</div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">운영체제</div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">범용 컴퓨팅</div>
                  </>
                )}
                {archType === 'GPU' && (
                  <>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">딥러닝 학습</div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">그래픽 렌더링</div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">과학 시뮬레이션</div>
                  </>
                )}
                {archType === 'NPU' && (
                  <>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">온디바이스 AI</div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">실시간 추론</div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">엣지 컴퓨팅</div>
                  </>
                )}
                {archType === 'SoC' && (
                  <>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">스마트폰</div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">태블릿</div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">웨어러블</div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
