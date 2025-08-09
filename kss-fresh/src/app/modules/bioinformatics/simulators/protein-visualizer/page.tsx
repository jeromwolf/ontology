'use client'

import React, { useState, useRef, useEffect, useMemo } from 'react'
import Link from 'next/link'
import { ArrowLeft, Upload, RotateCcw, ZoomIn, ZoomOut, Play, Pause, Download, Info, Settings, Eye, Layers } from 'lucide-react'

interface Atom {
  id: number
  element: string
  x: number
  y: number
  z: number
  residue: string
  resNum: number
  chain: string
}

interface Bond {
  from: number
  to: number
}

interface SecondaryStructure {
  type: 'helix' | 'sheet' | 'loop'
  start: number
  end: number
  chain: string
}

interface ProteinStructure {
  atoms: Atom[]
  bonds: Bond[]
  secondaryStructure: SecondaryStructure[]
}

// 샘플 단백질 구조 데이터 (PDB 형식 간소화)
const SAMPLE_PROTEINS: Record<string, ProteinStructure> = {
  'Lysozyme (1LYZ)': {
    atoms: [
      { id: 1, element: 'N', x: 20, y: 30, z: 40, residue: 'LYS', resNum: 1, chain: 'A' },
      { id: 2, element: 'C', x: 22, y: 28, z: 39, residue: 'LYS', resNum: 1, chain: 'A' },
      { id: 3, element: 'C', x: 24, y: 26, z: 38, residue: 'LYS', resNum: 1, chain: 'A' },
      { id: 4, element: 'O', x: 25, y: 24, z: 37, residue: 'LYS', resNum: 1, chain: 'A' },
      { id: 5, element: 'N', x: 26, y: 28, z: 36, residue: 'VAL', resNum: 2, chain: 'A' },
      { id: 6, element: 'C', x: 28, y: 30, z: 35, residue: 'VAL', resNum: 2, chain: 'A' },
      { id: 7, element: 'C', x: 30, y: 32, z: 34, residue: 'VAL', resNum: 2, chain: 'A' },
      { id: 8, element: 'O', x: 31, y: 34, z: 33, residue: 'VAL', resNum: 2, chain: 'A' },
      { id: 9, element: 'N', x: 32, y: 30, z: 32, residue: 'PHE', resNum: 3, chain: 'A' },
      { id: 10, element: 'C', x: 34, y: 28, z: 31, residue: 'PHE', resNum: 3, chain: 'A' },
      // 추가 원자들...
    ],
    bonds: [
      { from: 1, to: 2 }, { from: 2, to: 3 }, { from: 3, to: 4 },
      { from: 5, to: 6 }, { from: 6, to: 7 }, { from: 7, to: 8 },
      { from: 9, to: 10 }, { from: 2, to: 5 }, { from: 6, to: 9 }
    ],
    secondaryStructure: [
      { type: 'helix' as const, start: 1, end: 10, chain: 'A' },
      { type: 'sheet' as const, start: 15, end: 25, chain: 'A' },
      { type: 'loop' as const, start: 11, end: 14, chain: 'A' }
    ]
  },
  'Insulin (1INS)': {
    atoms: [
      { id: 1, element: 'N', x: 15, y: 25, z: 35, residue: 'GLY', resNum: 1, chain: 'A' },
      { id: 2, element: 'C', x: 17, y: 23, z: 34, residue: 'GLY', resNum: 1, chain: 'A' },
      { id: 3, element: 'C', x: 19, y: 21, z: 33, residue: 'GLY', resNum: 1, chain: 'A' },
      { id: 4, element: 'O', x: 20, y: 19, z: 32, residue: 'GLY', resNum: 1, chain: 'A' },
      { id: 5, element: 'N', x: 21, y: 23, z: 31, residue: 'ILE', resNum: 2, chain: 'A' },
      { id: 6, element: 'C', x: 23, y: 25, z: 30, residue: 'ILE', resNum: 2, chain: 'A' },
      { id: 7, element: 'C', x: 25, y: 27, z: 29, residue: 'ILE', resNum: 2, chain: 'A' },
      { id: 8, element: 'O', x: 26, y: 29, z: 28, residue: 'ILE', resNum: 2, chain: 'A' },
      // 체인 B 원자들
      { id: 9, element: 'N', x: 45, y: 25, z: 35, residue: 'PHE', resNum: 1, chain: 'B' },
      { id: 10, element: 'C', x: 47, y: 23, z: 34, residue: 'PHE', resNum: 1, chain: 'B' },
    ],
    bonds: [
      { from: 1, to: 2 }, { from: 2, to: 3 }, { from: 3, to: 4 },
      { from: 5, to: 6 }, { from: 6, to: 7 }, { from: 7, to: 8 },
      { from: 2, to: 5 }, { from: 9, to: 10 }
    ],
    secondaryStructure: [
      { type: 'helix' as const, start: 1, end: 8, chain: 'A' },
      { type: 'helix' as const, start: 1, end: 5, chain: 'B' }
    ]
  },
  'Hemoglobin (1HHB)': {
    atoms: [
      { id: 1, element: 'N', x: 10, y: 20, z: 30, residue: 'VAL', resNum: 1, chain: 'A' },
      { id: 2, element: 'C', x: 12, y: 18, z: 29, residue: 'VAL', resNum: 1, chain: 'A' },
      { id: 3, element: 'C', x: 14, y: 16, z: 28, residue: 'VAL', resNum: 1, chain: 'A' },
      { id: 4, element: 'O', x: 15, y: 14, z: 27, residue: 'VAL', resNum: 1, chain: 'A' },
      { id: 5, element: 'Fe', x: 25, y: 25, z: 25, residue: 'HEM', resNum: 147, chain: 'A' },
      { id: 6, element: 'N', x: 23, y: 23, z: 23, residue: 'HEM', resNum: 147, chain: 'A' },
      { id: 7, element: 'N', x: 27, y: 27, z: 27, residue: 'HEM', resNum: 147, chain: 'A' },
      { id: 8, element: 'N', x: 25, y: 27, z: 23, residue: 'HEM', resNum: 147, chain: 'A' },
      { id: 9, element: 'N', x: 25, y: 23, z: 27, residue: 'HEM', resNum: 147, chain: 'A' },
    ],
    bonds: [
      { from: 1, to: 2 }, { from: 2, to: 3 }, { from: 3, to: 4 },
      { from: 5, to: 6 }, { from: 5, to: 7 }, { from: 5, to: 8 }, { from: 5, to: 9 }
    ],
    secondaryStructure: [
      { type: 'helix' as const, start: 1, end: 20, chain: 'A' },
      { type: 'helix' as const, start: 25, end: 45, chain: 'A' },
      { type: 'sheet' as const, start: 50, end: 60, chain: 'A' }
    ]
  }
}

// 원소별 색상
const ELEMENT_COLORS: { [key: string]: string } = {
  'C': '#909090',  // 탄소 - 회색
  'N': '#3050F8',  // 질소 - 파랑
  'O': '#FF0D0D',  // 산소 - 빨강
  'S': '#FFFF30',  // 황 - 노랑
  'P': '#FF8000',  // 인 - 주황
  'Fe': '#E06633', // 철 - 갈색
  'Zn': '#7D80B0', // 아연 - 보라
  'Ca': '#3DFF00', // 칼슘 - 녹색
  'Mg': '#8AFF00', // 마그네슘 - 연두
  'H': '#FFFFFF'   // 수소 - 흰색
}

// 원소별 크기 (반데르발스 반지름)
const ELEMENT_SIZES: { [key: string]: number } = {
  'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'P': 1.8,
  'Fe': 2.0, 'Zn': 1.39, 'Ca': 2.31, 'Mg': 1.73, 'H': 1.2
}

// 이차 구조 색상
const SECONDARY_STRUCTURE_COLORS: { [key: string]: string } = {
  'helix': '#FF6B6B',  // 나선 - 빨강
  'sheet': '#4ECDC4',  // 베타시트 - 청록
  'loop': '#45B7D1'    // 루프 - 파랑
}

export default function ProteinVisualizerPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedProtein, setSelectedProtein] = useState('')
  const [protein, setProtein] = useState<ProteinStructure | null>(null)
  const [viewMode, setViewMode] = useState<'ball-stick' | 'space-fill' | 'cartoon' | 'ribbon'>('ball-stick')
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 })
  const [zoom, setZoom] = useState(1)
  const [isRotating, setIsRotating] = useState(false)
  const [showHydrogens, setShowHydrogens] = useState(false)
  const [showLabels, setShowLabels] = useState(false)
  const [selectedChain, setSelectedChain] = useState('all')
  const [animationSpeed, setAnimationSpeed] = useState(1)

  // 자동 회전 애니메이션
  useEffect(() => {
    let animationFrame: number
    if (isRotating && protein) {
      const animate = () => {
        setRotation(prev => ({
          ...prev,
          y: prev.y + animationSpeed
        }))
        animationFrame = requestAnimationFrame(animate)
      }
      animationFrame = requestAnimationFrame(animate)
    }
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
    }
  }, [isRotating, animationSpeed, protein])

  // 샘플 단백질 로드
  const loadSampleProtein = (proteinName: string) => {
    const proteinData = SAMPLE_PROTEINS[proteinName as keyof typeof SAMPLE_PROTEINS]
    if (proteinData) {
      setProtein(proteinData)
      setSelectedProtein(proteinName)
      setRotation({ x: 0, y: 0, z: 0 })
      setZoom(1)
    }
  }

  // PDB 파일 파싱 (간소화된 버전)
  const parsePDBFile = (content: string): ProteinStructure => {
    const lines = content.split('\n')
    const atoms: Atom[] = []
    const bonds: Bond[] = []
    const secondaryStructure: SecondaryStructure[] = []

    lines.forEach(line => {
      if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
        const id = parseInt(line.substring(6, 11).trim())
        const element = line.substring(76, 78).trim() || line.substring(12, 16).trim().charAt(0)
        const x = parseFloat(line.substring(30, 38))
        const y = parseFloat(line.substring(38, 46))
        const z = parseFloat(line.substring(46, 54))
        const residue = line.substring(17, 20).trim()
        const resNum = parseInt(line.substring(22, 26).trim())
        const chain = line.substring(21, 22).trim()

        atoms.push({ id, element, x, y, z, residue, resNum, chain })
      }
    })

    // 간단한 결합 추론 (실제로는 더 복잡한 알고리즘 필요)
    for (let i = 0; i < atoms.length - 1; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const atom1 = atoms[i]
        const atom2 = atoms[j]
        const distance = Math.sqrt(
          Math.pow(atom1.x - atom2.x, 2) +
          Math.pow(atom1.y - atom2.y, 2) +
          Math.pow(atom1.z - atom2.z, 2)
        )
        
        // 결합 거리 기준 (간소화)
        if (distance < 2.0 && atom1.resNum === atom2.resNum) {
          bonds.push({ from: atom1.id, to: atom2.id })
        }
      }
    }

    return { atoms, bonds, secondaryStructure }
  }

  // 캔버스에 단백질 렌더링
  const renderProtein = () => {
    if (!protein || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.save()

    // 화면 중앙으로 이동
    ctx.translate(canvas.width / 2, canvas.height / 2)
    ctx.scale(zoom, zoom)

    // 3D 회전을 위한 변환 행렬 (간소화)
    const cosX = Math.cos(rotation.x * Math.PI / 180)
    const sinX = Math.sin(rotation.x * Math.PI / 180)
    const cosY = Math.cos(rotation.y * Math.PI / 180)
    const sinY = Math.sin(rotation.y * Math.PI / 180)

    const transform3D = (x: number, y: number, z: number) => {
      // Y축 회전
      const x1 = x * cosY - z * sinY
      const z1 = x * sinY + z * cosY
      
      // X축 회전
      const y2 = y * cosX - z1 * sinX
      const z2 = y * sinX + z1 * cosX
      
      return { x: x1, y: y2, z: z2 }
    }

    // 필터링된 원자들
    const filteredAtoms = protein.atoms.filter(atom => {
      if (!showHydrogens && atom.element === 'H') return false
      if (selectedChain !== 'all' && atom.chain !== selectedChain) return false
      return true
    })

    // 깊이별 정렬
    const sortedAtoms = [...filteredAtoms].sort((a, b) => {
      const aTransformed = transform3D(a.x, a.y, a.z)
      const bTransformed = transform3D(b.x, b.y, b.z)
      return bTransformed.z - aTransformed.z
    })

    // 결합 렌더링
    if (viewMode === 'ball-stick' || viewMode === 'cartoon') {
      ctx.strokeStyle = '#666666'
      ctx.lineWidth = 2
      
      protein.bonds.forEach(bond => {
        const atom1 = protein.atoms.find(a => a.id === bond.from)
        const atom2 = protein.atoms.find(a => a.id === bond.to)
        
        if (!atom1 || !atom2) return
        if (selectedChain !== 'all' && (atom1.chain !== selectedChain || atom2.chain !== selectedChain)) return
        
        const pos1 = transform3D(atom1.x, atom1.y, atom1.z)
        const pos2 = transform3D(atom2.x, atom2.y, atom2.z)
        
        ctx.beginPath()
        ctx.moveTo(pos1.x * 2, pos1.y * 2)
        ctx.lineTo(pos2.x * 2, pos2.y * 2)
        ctx.stroke()
      })
    }

    // 원자 렌더링
    sortedAtoms.forEach(atom => {
      const pos = transform3D(atom.x, atom.y, atom.z)
      const radius = viewMode === 'space-fill' 
        ? (ELEMENT_SIZES[atom.element] || 1.5) * 8
        : viewMode === 'ball-stick' 
        ? (ELEMENT_SIZES[atom.element] || 1.5) * 4
        : 3

      // 원자 색상
      const color = ELEMENT_COLORS[atom.element] || '#808080'
      
      // 깊이에 따른 음영 효과
      const depthFactor = Math.max(0.3, 1 - (pos.z + 50) / 100)
      
      ctx.fillStyle = color
      ctx.globalAlpha = depthFactor
      
      ctx.beginPath()
      ctx.arc(pos.x * 2, pos.y * 2, radius, 0, 2 * Math.PI)
      ctx.fill()
      
      // 원자 테두리
      if (viewMode === 'ball-stick') {
        ctx.strokeStyle = '#333333'
        ctx.lineWidth = 1
        ctx.stroke()
      }
      
      // 라벨 표시
      if (showLabels) {
        ctx.globalAlpha = 1
        ctx.fillStyle = '#000000'
        ctx.font = '10px Arial'
        ctx.textAlign = 'center'
        ctx.fillText(
          `${atom.element}${atom.resNum}`,
          pos.x * 2,
          pos.y * 2 - radius - 5
        )
      }
    })

    // 리본/카툰 모드의 이차 구조 렌더링
    if (viewMode === 'ribbon' || viewMode === 'cartoon') {
      protein.secondaryStructure.forEach(structure => {
        if (selectedChain !== 'all' && structure.chain !== selectedChain) return
        
        const color = SECONDARY_STRUCTURE_COLORS[structure.type]
        ctx.strokeStyle = color
        ctx.lineWidth = structure.type === 'helix' ? 8 : structure.type === 'sheet' ? 6 : 4
        
        // 이차 구조를 나타내는 선 그리기 (간소화)
        const structureAtoms = filteredAtoms.filter(atom => 
          atom.resNum >= structure.start && 
          atom.resNum <= structure.end && 
          atom.chain === structure.chain
        )
        
        if (structureAtoms.length > 1) {
          ctx.beginPath()
          structureAtoms.forEach((atom, index) => {
            const pos = transform3D(atom.x, atom.y, atom.z)
            if (index === 0) {
              ctx.moveTo(pos.x * 2, pos.y * 2)
            } else {
              ctx.lineTo(pos.x * 2, pos.y * 2)
            }
          })
          ctx.stroke()
        }
      })
    }

    ctx.restore()
  }

  // 렌더링 실행
  useEffect(() => {
    renderProtein()
  }, [protein, viewMode, rotation, zoom, showHydrogens, showLabels, selectedChain])

  // 파일 업로드 처리
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const content = e.target?.result as string
        try {
          const parsedProtein = parsePDBFile(content)
          setProtein(parsedProtein)
          setSelectedProtein(file.name)
        } catch (error) {
          alert('PDB 파일 파싱 중 오류가 발생했습니다.')
        }
      }
      reader.readAsText(file)
    }
  }

  // 체인 목록 추출
  const availableChains = useMemo(() => {
    if (!protein) return []
    const chains = Array.from(new Set(protein.atoms.map(atom => atom.chain)))
    return chains.sort()
  }, [protein])

  return (
    <div className="min-h-screen bg-gradient-to-b from-cyan-50 to-blue-50 dark:from-gray-900 dark:to-cyan-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link 
            href="/modules/bioinformatics"
            className="inline-flex items-center text-cyan-600 dark:text-cyan-400 hover:text-cyan-700 dark:hover:text-cyan-300 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            생물정보학 모듈로 돌아가기
          </Link>
          
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            3D 단백질 구조 뷰어
          </h1>
          
          <p className="text-lg text-gray-600 dark:text-gray-300">
            PDB 파일을 시각화하고 AlphaFold 예측 결과와 비교할 수 있는 인터랙티브 3D 뷰어
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 3D 뷰어 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  3D 구조 뷰어
                </h2>
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsRotating(!isRotating)}
                    className={`p-2 rounded-lg transition-colors ${
                      isRotating ? 'bg-cyan-600 text-white' : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  >
                    {isRotating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </button>
                  <button
                    onClick={() => setZoom(Math.min(3, zoom + 0.2))}
                    className="p-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setZoom(Math.max(0.5, zoom - 0.2))}
                    className="p-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    <ZoomOut className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => {
                      setRotation({ x: 0, y: 0, z: 0 })
                      setZoom(1)
                    }}
                    className="p-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>
              
              <div className="relative border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden">
                <canvas
                  ref={canvasRef}
                  width={800}
                  height={600}
                  className="w-full h-auto bg-black cursor-grab active:cursor-grabbing"
                  onMouseDown={(e) => {
                    const startX = e.clientX
                    const startY = e.clientY
                    const startRotation = { ...rotation }
                    
                    const handleMouseMove = (e: MouseEvent) => {
                      const deltaX = e.clientX - startX
                      const deltaY = e.clientY - startY
                      
                      setRotation({
                        x: startRotation.x + deltaY * 0.5,
                        y: startRotation.y + deltaX * 0.5,
                        z: startRotation.z
                      })
                    }
                    
                    const handleMouseUp = () => {
                      document.removeEventListener('mousemove', handleMouseMove)
                      document.removeEventListener('mouseup', handleMouseUp)
                    }
                    
                    document.addEventListener('mousemove', handleMouseMove)
                    document.addEventListener('mouseup', handleMouseUp)
                  }}
                />
                
                {!protein && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
                    <div className="text-center text-white">
                      <Upload className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p className="text-lg mb-2">단백질 구조를 로드하세요</p>
                      <p className="text-sm opacity-75">샘플을 선택하거나 PDB 파일을 업로드하세요</p>
                    </div>
                  </div>
                )}
              </div>
              
              {protein && (
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="text-center p-2 bg-gray-100 dark:bg-gray-700 rounded">
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {protein.atoms.length}
                    </div>
                    <div className="text-gray-600 dark:text-gray-400">원자</div>
                  </div>
                  <div className="text-center p-2 bg-gray-100 dark:bg-gray-700 rounded">
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {protein.bonds.length}
                    </div>
                    <div className="text-gray-600 dark:text-gray-400">결합</div>
                  </div>
                  <div className="text-center p-2 bg-gray-100 dark:bg-gray-700 rounded">
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {availableChains.length}
                    </div>
                    <div className="text-gray-600 dark:text-gray-400">체인</div>
                  </div>
                  <div className="text-center p-2 bg-gray-100 dark:bg-gray-700 rounded">
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {protein.secondaryStructure.length}
                    </div>
                    <div className="text-gray-600 dark:text-gray-400">이차구조</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 컨트롤 패널 */}
          <div className="space-y-6">
            {/* 샘플 로드 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                샘플 단백질
              </h3>
              <div className="space-y-2">
                {Object.keys(SAMPLE_PROTEINS).map(proteinName => (
                  <button
                    key={proteinName}
                    onClick={() => loadSampleProtein(proteinName)}
                    className={`w-full p-3 text-left rounded-lg border transition-all ${
                      selectedProtein === proteinName
                        ? 'border-cyan-500 bg-cyan-50 dark:bg-cyan-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-cyan-300'
                    }`}
                  >
                    <div className="font-medium text-gray-900 dark:text-white">
                      {proteinName.split(' ')[0]}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {proteinName.split(' ')[1]}
                    </div>
                  </button>
                ))}
              </div>
              
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  PDB 파일 업로드
                </label>
                <input
                  type="file"
                  accept=".pdb"
                  onChange={handleFileUpload}
                  className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-cyan-50 file:text-cyan-700 hover:file:bg-cyan-100"
                />
              </div>
            </div>

            {/* 시각화 옵션 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                시각화 설정
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    렌더링 모드
                  </label>
                  <select
                    value={viewMode}
                    onChange={(e) => setViewMode(e.target.value as any)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                  >
                    <option value="ball-stick">Ball & Stick</option>
                    <option value="space-fill">Space Filling</option>
                    <option value="cartoon">Cartoon</option>
                    <option value="ribbon">Ribbon</option>
                  </select>
                </div>
                
                {availableChains.length > 1 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      체인 선택
                    </label>
                    <select
                      value={selectedChain}
                      onChange={(e) => setSelectedChain(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                    >
                      <option value="all">모든 체인</option>
                      {availableChains.map(chain => (
                        <option key={chain} value={chain}>체인 {chain}</option>
                      ))}
                    </select>
                  </div>
                )}
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    회전 속도
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={animationSpeed}
                    onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {animationSpeed.toFixed(1)}x
                  </div>
                </div>
                
                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showHydrogens}
                      onChange={(e) => setShowHydrogens(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">수소 원자 표시</span>
                  </label>
                  
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showLabels}
                      onChange={(e) => setShowLabels(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">원자 라벨 표시</span>
                  </label>
                </div>
              </div>
            </div>

            {/* 색상 범례 */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Layers className="w-5 h-5" />
                색상 범례
              </h3>
              
              <div className="space-y-3">
                <div>
                  <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">원소</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {Object.entries(ELEMENT_COLORS).map(([element, color]) => (
                      <div key={element} className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full border border-gray-300"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-gray-600 dark:text-gray-400">{element}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">이차구조</div>
                  <div className="space-y-1 text-xs">
                    {Object.entries(SECONDARY_STRUCTURE_COLORS).map(([structure, color]) => (
                      <div key={structure} className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded border border-gray-300"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-gray-600 dark:text-gray-400">
                          {structure === 'helix' ? '알파 나선' : 
                           structure === 'sheet' ? '베타 시트' : '루프'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* 내보내기 */}
            {protein && (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  내보내기
                </h3>
                <button
                  onClick={() => {
                    const canvas = canvasRef.current
                    if (canvas) {
                      const link = document.createElement('a')
                      link.download = `protein_${selectedProtein}_screenshot.png`
                      link.href = canvas.toDataURL()
                      link.click()
                    }
                  }}
                  className="w-full px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  스크린샷 저장
                </button>
              </div>
            )}
          </div>
        </div>

        {/* 도움말 */}
        <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                사용 방법
              </h3>
              <ul className="space-y-1 text-sm text-blue-800 dark:text-blue-200">
                <li>• 마우스를 드래그하여 단백질을 회전시킬 수 있습니다</li>
                <li>• 줌 버튼이나 마우스 휠로 확대/축소가 가능합니다</li>
                <li>• 다양한 렌더링 모드로 구조를 관찰하세요</li>
                <li>• PDB 파일을 업로드하여 실제 단백질 구조를 시각화하세요</li>
                <li>• 자동 회전 기능으로 모든 각도에서 구조를 확인할 수 있습니다</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}