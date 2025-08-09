'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { 
  Beaker,
  Dna,
  Target,
  TrendingUp,
  Play,
  Pause,
  RotateCcw,
  ArrowLeft,
  ChevronRight,
  AlertCircle,
  CheckCircle,
  Clock,
  DollarSign,
  Activity,
  Zap,
  FlaskConical,
  Microscope
} from 'lucide-react'

interface Molecule {
  id: string
  name: string
  smiles: string
  bindingAffinity: number
  drugLikeness: number
  toxicity: number
  synthesizability: number
  stage: 'discovery' | 'lead' | 'optimization' | 'preclinical' | 'clinical'
}

interface ProteinTarget {
  id: string
  name: string
  type: string
  diseaseArea: string
  druggability: number
}

export default function DrugDiscoverySimulator() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [currentStage, setCurrentStage] = useState<'target' | 'screening' | 'optimization' | 'testing'>('target')
  const [selectedTarget, setSelectedTarget] = useState<ProteinTarget | null>(null)
  const [molecules, setMolecules] = useState<Molecule[]>([])
  const [selectedMolecule, setSelectedMolecule] = useState<Molecule | null>(null)
  const [progress, setProgress] = useState(0)
  const [stats, setStats] = useState({
    screened: 0,
    hits: 0,
    leads: 0,
    optimized: 0,
    timeElapsed: 0,
    cost: 0
  })

  const proteinTargets: ProteinTarget[] = [
    { id: 'ace2', name: 'ACE2 Receptor', type: 'Enzyme', diseaseArea: 'COVID-19', druggability: 0.85 },
    { id: 'braf', name: 'BRAF Kinase', type: 'Kinase', diseaseArea: 'Cancer', druggability: 0.92 },
    { id: 'app', name: 'Amyloid Precursor', type: 'Protein', diseaseArea: "Alzheimer's", druggability: 0.68 },
    { id: 'tnf', name: 'TNF-α', type: 'Cytokine', diseaseArea: 'Inflammation', druggability: 0.78 }
  ]

  const generateMolecule = (): Molecule => {
    const names = ['Compound', 'Molecule', 'Drug', 'Agent', 'Inhibitor']
    const id = Math.random().toString(36).substr(2, 9)
    return {
      id,
      name: `${names[Math.floor(Math.random() * names.length)]}-${id.toUpperCase()}`,
      smiles: generateRandomSMILES(),
      bindingAffinity: Math.random() * 0.8 + 0.2,
      drugLikeness: Math.random() * 0.7 + 0.3,
      toxicity: Math.random() * 0.6,
      synthesizability: Math.random() * 0.8 + 0.2,
      stage: 'discovery'
    }
  }

  const generateRandomSMILES = () => {
    const fragments = ['C', 'CC', 'CCC', 'O', 'N', 'S', 'F', 'Cl', 'Br', '=', '#', '@', '(', ')']
    let smiles = ''
    const length = Math.floor(Math.random() * 20) + 10
    for (let i = 0; i < length; i++) {
      smiles += fragments[Math.floor(Math.random() * fragments.length)]
    }
    return smiles
  }

  const runVirtualScreening = () => {
    if (!selectedTarget) return
    
    setIsRunning(true)
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsRunning(false)
          setCurrentStage('optimization')
          return 100
        }
        
        // Generate molecules during screening
        if (prev % 10 === 0) {
          const newMolecule = generateMolecule()
          const isHit = newMolecule.bindingAffinity > 0.6 && newMolecule.drugLikeness > 0.5
          
          if (isHit) {
            newMolecule.stage = 'lead'
            setMolecules(prev => [...prev, newMolecule])
            setStats(prev => ({ ...prev, hits: prev.hits + 1 }))
          }
          
          setStats(prev => ({
            ...prev,
            screened: prev.screened + Math.floor(Math.random() * 1000) + 500,
            timeElapsed: prev.timeElapsed + 1,
            cost: prev.cost + Math.floor(Math.random() * 10000) + 5000
          }))
        }
        
        return prev + 2
      })
    }, 100)
  }

  const optimizeMolecule = (molecule: Molecule) => {
    const optimized = { ...molecule }
    optimized.bindingAffinity = Math.min(0.99, molecule.bindingAffinity * 1.1)
    optimized.drugLikeness = Math.min(0.99, molecule.drugLikeness * 1.15)
    optimized.toxicity = Math.max(0.1, molecule.toxicity * 0.8)
    optimized.synthesizability = Math.min(0.99, molecule.synthesizability * 1.05)
    optimized.stage = 'optimization'
    optimized.name = `${molecule.name}-OPT`
    
    setMolecules(prev => prev.map(m => m.id === molecule.id ? optimized : m))
    setStats(prev => ({ ...prev, optimized: prev.optimized + 1 }))
    setSelectedMolecule(optimized)
  }

  const drawMoleculeStructure = (ctx: CanvasRenderingContext2D, molecule: Molecule) => {
    ctx.clearRect(0, 0, 300, 300)
    
    // Simple molecular visualization
    const centerX = 150
    const centerY = 150
    const atoms = Math.floor(Math.random() * 8) + 5
    const radius = 60
    
    ctx.strokeStyle = '#4B5563'
    ctx.lineWidth = 2
    
    // Draw bonds
    for (let i = 0; i < atoms; i++) {
      const angle1 = (i * 2 * Math.PI) / atoms
      const x1 = centerX + radius * Math.cos(angle1)
      const y1 = centerY + radius * Math.sin(angle1)
      
      const nextI = (i + 1) % atoms
      const angle2 = (nextI * 2 * Math.PI) / atoms
      const x2 = centerX + radius * Math.cos(angle2)
      const y2 = centerY + radius * Math.sin(angle2)
      
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()
      
      // Random double bonds
      if (Math.random() > 0.7) {
        ctx.beginPath()
        ctx.moveTo(x1 + 3, y1 + 3)
        ctx.lineTo(x2 + 3, y2 + 3)
        ctx.stroke()
      }
    }
    
    // Draw atoms
    const atomColors = ['#EF4444', '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
    for (let i = 0; i < atoms; i++) {
      const angle = (i * 2 * Math.PI) / atoms
      const x = centerX + radius * Math.cos(angle)
      const y = centerY + radius * Math.sin(angle)
      
      ctx.fillStyle = atomColors[i % atomColors.length]
      ctx.beginPath()
      ctx.arc(x, y, 12, 0, 2 * Math.PI)
      ctx.fill()
      
      ctx.fillStyle = 'white'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      const atomTypes = ['C', 'N', 'O', 'S', 'F']
      ctx.fillText(atomTypes[i % atomTypes.length], x, y)
    }
  }

  useEffect(() => {
    if (selectedMolecule && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      if (ctx) {
        drawMoleculeStructure(ctx, selectedMolecule)
      }
    }
  }, [selectedMolecule])

  const reset = () => {
    setIsRunning(false)
    setCurrentStage('target')
    setSelectedTarget(null)
    setMolecules([])
    setSelectedMolecule(null)
    setProgress(0)
    setStats({
      screened: 0,
      hits: 0,
      leads: 0,
      optimized: 0,
      timeElapsed: 0,
      cost: 0
    })
  }

  const getScoreColor = (score: number) => {
    if (score > 0.8) return 'text-green-600 dark:text-green-400'
    if (score > 0.6) return 'text-yellow-600 dark:text-yellow-400'
    if (score > 0.4) return 'text-orange-600 dark:text-orange-400'
    return 'text-red-600 dark:text-red-400'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/medical-ai"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>Medical AI로 돌아가기</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                AI 신약 개발 시뮬레이터
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={reset}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                초기화
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {['타겟 선택', '가상 스크리닝', '최적화', '전임상 평가'].map((step, index) => {
              const stages = ['target', 'screening', 'optimization', 'testing']
              const isActive = stages.indexOf(currentStage) >= index
              const isComplete = stages.indexOf(currentStage) > index
              
              return (
                <div key={step} className="flex-1 relative">
                  <div className="flex items-center">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold transition-colors ${
                      isActive 
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white' 
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500'
                    }`}>
                      {isComplete ? <CheckCircle className="w-5 h-5" /> : index + 1}
                    </div>
                    {index < 3 && (
                      <div className={`flex-1 h-1 ml-2 transition-colors ${
                        isComplete ? 'bg-purple-500' : 'bg-gray-200 dark:bg-gray-700'
                      }`}></div>
                    )}
                  </div>
                  <div className="mt-2">
                    <p className={`text-sm font-medium ${
                      isActive ? 'text-gray-900 dark:text-white' : 'text-gray-400 dark:text-gray-500'
                    }`}>
                      {step}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Panel - Target & Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Target Selection */}
            {currentStage === 'target' && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                  <Target className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                  단백질 타겟 선택
                </h3>
                <div className="space-y-3">
                  {proteinTargets.map((target) => (
                    <button
                      key={target.id}
                      onClick={() => {
                        setSelectedTarget(target)
                        setCurrentStage('screening')
                      }}
                      className="w-full text-left p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-colors"
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <h4 className="font-semibold text-gray-900 dark:text-white">
                            {target.name}
                          </h4>
                          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                            {target.type} • {target.diseaseArea}
                          </p>
                        </div>
                        <div className="text-right">
                          <div className="text-xs text-gray-500 dark:text-gray-400">Druggability</div>
                          <div className={`text-sm font-semibold ${getScoreColor(target.druggability)}`}>
                            {(target.druggability * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Selected Target Info */}
            {selectedTarget && currentStage !== 'target' && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
                  선택된 타겟
                </h3>
                <div className="space-y-3">
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">이름</p>
                    <p className="font-semibold text-gray-900 dark:text-white">
                      {selectedTarget.name}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">질병 영역</p>
                    <p className="font-semibold text-gray-900 dark:text-white">
                      {selectedTarget.diseaseArea}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Druggability Score</p>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-purple-500 to-pink-500"
                          style={{ width: `${selectedTarget.druggability * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Screening Controls */}
            {currentStage === 'screening' && (
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                  <Microscope className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  가상 스크리닝
                </h3>
                <button
                  onClick={runVirtualScreening}
                  disabled={isRunning}
                  className={`w-full py-3 rounded-lg font-medium transition-all ${
                    isRunning
                      ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-lg'
                  }`}
                >
                  {isRunning ? (
                    <span className="flex items-center justify-center gap-2">
                      <Pause className="w-5 h-5" />
                      스크리닝 중... {progress}%
                    </span>
                  ) : (
                    <span className="flex items-center justify-center gap-2">
                      <Play className="w-5 h-5" />
                      스크리닝 시작
                    </span>
                  )}
                </button>
                
                {progress > 0 && (
                  <div className="mt-4">
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Statistics */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
                통계
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400">스크리닝 화합물</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {stats.screened.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400">히트 화합물</span>
                  <span className="font-semibold text-green-600 dark:text-green-400">
                    {stats.hits}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400">최적화 완료</span>
                  <span className="font-semibold text-blue-600 dark:text-blue-400">
                    {stats.optimized}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    소요 시간
                  </span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {stats.timeElapsed} 개월
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-1">
                    <DollarSign className="w-4 h-4" />
                    비용
                  </span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    ${stats.cost.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Middle Panel - Molecules List */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <FlaskConical className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                후보 화합물
              </h3>
              
              {molecules.length === 0 ? (
                <div className="text-center py-12">
                  <Beaker className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-500 dark:text-gray-400">
                    스크리닝을 시작하여 화합물을 발견하세요
                  </p>
                </div>
              ) : (
                <div className="space-y-3 max-h-[600px] overflow-y-auto">
                  {molecules.map((molecule) => (
                    <div
                      key={molecule.id}
                      onClick={() => setSelectedMolecule(molecule)}
                      className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedMolecule?.id === molecule.id
                          ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                          : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold text-gray-900 dark:text-white">
                          {molecule.name}
                        </h4>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          molecule.stage === 'optimization' 
                            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                            : molecule.stage === 'lead'
                            ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                            : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                        }`}>
                          {molecule.stage === 'optimization' ? '최적화' : 
                           molecule.stage === 'lead' ? '선도물질' : '발견'}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">결합력</span>
                          <div className="flex items-center gap-1 mt-1">
                            <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                              <div
                                className="h-1.5 rounded-full bg-purple-500"
                                style={{ width: `${molecule.bindingAffinity * 100}%` }}
                              ></div>
                            </div>
                            <span className={`font-medium ${getScoreColor(molecule.bindingAffinity)}`}>
                              {(molecule.bindingAffinity * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">약물성</span>
                          <div className="flex items-center gap-1 mt-1">
                            <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                              <div
                                className="h-1.5 rounded-full bg-blue-500"
                                style={{ width: `${molecule.drugLikeness * 100}%` }}
                              ></div>
                            </div>
                            <span className={`font-medium ${getScoreColor(molecule.drugLikeness)}`}>
                              {(molecule.drugLikeness * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Molecule Details */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <Dna className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                화합물 상세 정보
              </h3>
              
              {selectedMolecule ? (
                <div className="space-y-4">
                  {/* Molecule Structure */}
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                      분자 구조
                    </h4>
                    <canvas
                      ref={canvasRef}
                      width={300}
                      height={300}
                      className="w-full bg-white dark:bg-gray-800 rounded"
                    />
                  </div>
                  
                  {/* Properties */}
                  <div className="space-y-3">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-gray-500 dark:text-gray-400">결합 친화도</span>
                        <span className={`text-sm font-semibold ${getScoreColor(selectedMolecule.bindingAffinity)}`}>
                          {(selectedMolecule.bindingAffinity * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-purple-500 to-pink-500"
                          style={{ width: `${selectedMolecule.bindingAffinity * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-gray-500 dark:text-gray-400">약물 유사성</span>
                        <span className={`text-sm font-semibold ${getScoreColor(selectedMolecule.drugLikeness)}`}>
                          {(selectedMolecule.drugLikeness * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                          style={{ width: `${selectedMolecule.drugLikeness * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-gray-500 dark:text-gray-400">독성 위험</span>
                        <span className={`text-sm font-semibold ${getScoreColor(1 - selectedMolecule.toxicity)}`}>
                          {(selectedMolecule.toxicity * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-red-500 to-orange-500"
                          style={{ width: `${selectedMolecule.toxicity * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-gray-500 dark:text-gray-400">합성 가능성</span>
                        <span className={`text-sm font-semibold ${getScoreColor(selectedMolecule.synthesizability)}`}>
                          {(selectedMolecule.synthesizability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-green-500 to-emerald-500"
                          style={{ width: `${selectedMolecule.synthesizability * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  
                  {/* SMILES */}
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">SMILES</p>
                    <p className="text-xs font-mono text-gray-700 dark:text-gray-300 break-all">
                      {selectedMolecule.smiles}
                    </p>
                  </div>
                  
                  {/* Actions */}
                  {currentStage === 'optimization' && selectedMolecule.stage !== 'optimization' && (
                    <button
                      onClick={() => optimizeMolecule(selectedMolecule)}
                      className="w-full py-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:shadow-lg transition-all"
                    >
                      <span className="flex items-center justify-center gap-2">
                        <Zap className="w-4 h-4" />
                        화합물 최적화
                      </span>
                    </button>
                  )}
                  
                  {selectedMolecule.stage === 'optimization' && (
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="flex items-start gap-2">
                        <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
                        <div className="text-sm text-green-800 dark:text-green-200">
                          <p className="font-semibold">최적화 완료</p>
                          <p>이 화합물은 전임상 시험 준비가 완료되었습니다.</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Dna className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-500 dark:text-gray-400">
                    화합물을 선택하여 상세 정보를 확인하세요
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Info Box */}
        <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
            <div className="text-sm text-blue-800 dark:text-blue-200">
              <p className="font-semibold mb-1">시뮬레이터 정보</p>
              <p>이 시뮬레이터는 AI 기반 신약 개발 과정을 단순화하여 보여줍니다. 
              실제 신약 개발은 수년간의 연구와 수십억 원의 투자가 필요한 복잡한 과정입니다.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}