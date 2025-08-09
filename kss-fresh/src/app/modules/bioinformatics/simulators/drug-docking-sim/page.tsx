'use client'

import { useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, Beaker, Target, Activity, TrendingUp, AlertCircle, CheckCircle, Download, Play } from 'lucide-react'

interface Ligand {
  id: string
  name: string
  smiles: string
  molecularWeight: number
  logP: number
  hDonors: number
  hAcceptors: number
}

interface DockingResult {
  ligand: Ligand
  bindingAffinity: number
  kiValue: number
  interactions: string[]
  drugLikeness: boolean
  lipinskiViolations: number
}

export default function DrugDockingSimPage() {
  const [selectedTarget, setSelectedTarget] = useState('')
  const [selectedLigands, setSelectedLigands] = useState<string[]>([])
  const [isDocking, setIsDocking] = useState(false)
  const [dockingResults, setDockingResults] = useState<DockingResult[]>([])
  const [activeTab, setActiveTab] = useState<'setup' | 'results'>('setup')

  // Sample protein targets
  const proteinTargets = [
    { id: 'covid-mpro', name: 'SARS-CoV-2 Main Protease', pdbId: '6LU7' },
    { id: 'egfr', name: 'EGFR Kinase', pdbId: '4HJO' },
    { id: 'bcr-abl', name: 'BCR-ABL Kinase', pdbId: '3OXZ' },
    { id: 'hdac', name: 'HDAC2', pdbId: '4LXZ' }
  ]

  // Sample ligand library
  const ligandLibrary: Ligand[] = [
    {
      id: 'remdesivir',
      name: 'Remdesivir',
      smiles: 'CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=NC3=C(N=CN=C32)N)O)O)OC4=CC=CC=C4',
      molecularWeight: 602.6,
      logP: 2.8,
      hDonors: 4,
      hAcceptors: 12
    },
    {
      id: 'imatinib',
      name: 'Imatinib',
      smiles: 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CC=N5',
      molecularWeight: 493.6,
      logP: 3.0,
      hDonors: 2,
      hAcceptors: 7
    },
    {
      id: 'aspirin',
      name: 'Aspirin',
      smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
      molecularWeight: 180.2,
      logP: 1.2,
      hDonors: 1,
      hAcceptors: 4
    },
    {
      id: 'metformin',
      name: 'Metformin',
      smiles: 'CN(C)C(=N)NC(=N)N',
      molecularWeight: 129.2,
      logP: -1.3,
      hDonors: 3,
      hAcceptors: 2
    },
    {
      id: 'paracetamol',
      name: 'Paracetamol',
      smiles: 'CC(=O)NC1=CC=C(C=C1)O',
      molecularWeight: 151.2,
      logP: 0.5,
      hDonors: 2,
      hAcceptors: 2
    }
  ]

  const runDocking = () => {
    if (!selectedTarget || selectedLigands.length === 0) return
    
    setIsDocking(true)
    
    // Simulate docking process
    setTimeout(() => {
      const results: DockingResult[] = selectedLigands.map(ligandId => {
        const ligand = ligandLibrary.find(l => l.id === ligandId)!
        
        // Simulate binding affinity calculation
        const bindingAffinity = -Math.random() * 8 - 4 // -4 to -12 kcal/mol
        const kiValue = Math.exp(-bindingAffinity * 1.36) // Approximate Ki from ΔG
        
        // Check Lipinski's Rule of Five
        let violations = 0
        if (ligand.molecularWeight > 500) violations++
        if (ligand.logP > 5) violations++
        if (ligand.hDonors > 5) violations++
        if (ligand.hAcceptors > 10) violations++
        
        // Simulate interactions
        const possibleInteractions = [
          'Hydrogen bond with ASP189',
          'Hydrophobic interaction with VAL216',
          'π-π stacking with TYR161',
          'Salt bridge with LYS192',
          'Van der Waals with LEU167'
        ]
        
        const interactions = possibleInteractions
          .sort(() => Math.random() - 0.5)
          .slice(0, Math.floor(Math.random() * 3) + 2)
        
        return {
          ligand,
          bindingAffinity,
          kiValue,
          interactions,
          drugLikeness: violations <= 1,
          lipinskiViolations: violations
        }
      })
      
      // Sort by binding affinity
      results.sort((a, b) => a.bindingAffinity - b.bindingAffinity)
      
      setDockingResults(results)
      setActiveTab('results')
      setIsDocking(false)
    }, 2000)
  }

  const exportResults = () => {
    if (dockingResults.length === 0) return
    
    const csv = [
      'Ligand,Binding Affinity (kcal/mol),Ki (nM),Drug-likeness,Lipinski Violations,Interactions',
      ...dockingResults.map(r => 
        `${r.ligand.name},${r.bindingAffinity.toFixed(2)},${r.kiValue.toFixed(2)},${r.drugLikeness ? 'Yes' : 'No'},${r.lipinskiViolations},"${r.interactions.join('; ')}"`
      )
    ].join('\n')
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `docking_results_${Date.now()}.csv`
    a.click()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50 dark:from-gray-900 dark:via-emerald-950 dark:to-teal-950 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link
              href="/modules/bioinformatics"
              className="p-2 hover:bg-white/50 dark:hover:bg-gray-800/50 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">
                분자 도킹 시뮬레이터
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                리간드-수용체 도킹, 결합 친화도 계산
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Beaker className="w-8 h-8 text-emerald-600" />
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab('setup')}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'setup'
                ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            도킹 설정
          </button>
          <button
            onClick={() => setActiveTab('results')}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'results'
                ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            disabled={dockingResults.length === 0}
          >
            결과 ({dockingResults.length})
          </button>
        </div>

        {activeTab === 'setup' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Target Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-emerald-600" />
                타겟 단백질 선택
              </h3>
              <div className="space-y-2">
                {proteinTargets.map(target => (
                  <label
                    key={target.id}
                    className="flex items-center p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                  >
                    <input
                      type="radio"
                      name="target"
                      value={target.id}
                      checked={selectedTarget === target.id}
                      onChange={(e) => setSelectedTarget(e.target.value)}
                      className="mr-3"
                    />
                    <div>
                      <div className="font-medium">{target.name}</div>
                      <div className="text-sm text-gray-500">PDB: {target.pdbId}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Ligand Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-emerald-600" />
                리간드 라이브러리
              </h3>
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {ligandLibrary.map(ligand => (
                  <label
                    key={ligand.id}
                    className="flex items-start p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                  >
                    <input
                      type="checkbox"
                      value={ligand.id}
                      checked={selectedLigands.includes(ligand.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedLigands([...selectedLigands, ligand.id])
                        } else {
                          setSelectedLigands(selectedLigands.filter(id => id !== ligand.id))
                        }
                      }}
                      className="mr-3 mt-1"
                    />
                    <div className="flex-1">
                      <div className="font-medium">{ligand.name}</div>
                      <div className="text-xs text-gray-500 mt-1">
                        MW: {ligand.molecularWeight} | LogP: {ligand.logP}
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Run Docking */}
            <div className="lg:col-span-2">
              <button
                onClick={runDocking}
                disabled={!selectedTarget || selectedLigands.length === 0 || isDocking}
                className="w-full py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg font-semibold hover:from-emerald-600 hover:to-teal-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isDocking ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
                    도킹 시뮬레이션 진행 중...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    도킹 시작
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'results' && dockingResults.length > 0 && (
          <div className="space-y-6">
            {/* Results Summary */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                  도킹 결과
                </h3>
                <button
                  onClick={exportResults}
                  className="px-4 py-2 bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 rounded-lg hover:bg-emerald-200 dark:hover:bg-emerald-800/50 transition-colors flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  CSV 내보내기
                </button>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left p-2">순위</th>
                      <th className="text-left p-2">리간드</th>
                      <th className="text-left p-2">결합 친화도</th>
                      <th className="text-left p-2">Ki (nM)</th>
                      <th className="text-left p-2">Drug-likeness</th>
                      <th className="text-left p-2">주요 상호작용</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dockingResults.map((result, index) => (
                      <tr key={result.ligand.id} className="border-b border-gray-100 dark:border-gray-700">
                        <td className="p-2">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                            index === 0 ? 'bg-gradient-to-r from-yellow-400 to-amber-500' :
                            index === 1 ? 'bg-gradient-to-r from-gray-300 to-gray-400' :
                            index === 2 ? 'bg-gradient-to-r from-orange-400 to-orange-500' :
                            'bg-gray-500'
                          }`}>
                            {index + 1}
                          </div>
                        </td>
                        <td className="p-2">
                          <div className="font-medium">{result.ligand.name}</div>
                          <div className="text-xs text-gray-500">MW: {result.ligand.molecularWeight}</div>
                        </td>
                        <td className="p-2">
                          <div className="font-mono text-sm">{result.bindingAffinity.toFixed(2)} kcal/mol</div>
                        </td>
                        <td className="p-2">
                          <div className="font-mono text-sm">{result.kiValue.toFixed(2)}</div>
                        </td>
                        <td className="p-2">
                          {result.drugLikeness ? (
                            <div className="flex items-center gap-1 text-green-600">
                              <CheckCircle className="w-4 h-4" />
                              <span className="text-sm">Pass</span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1 text-red-600">
                              <AlertCircle className="w-4 h-4" />
                              <span className="text-sm">{result.lipinskiViolations} violations</span>
                            </div>
                          )}
                        </td>
                        <td className="p-2">
                          <div className="text-sm">
                            {result.interactions.slice(0, 2).join(', ')}
                            {result.interactions.length > 2 && ` (+${result.interactions.length - 2})`}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Best Candidate Details */}
            {dockingResults[0] && (
              <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-emerald-600" />
                  최적 후보 물질
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">
                      {dockingResults[0].ligand.name}
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">결합 친화도</span>
                        <span className="font-medium">{dockingResults[0].bindingAffinity.toFixed(2)} kcal/mol</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Ki 값</span>
                        <span className="font-medium">{dockingResults[0].kiValue.toFixed(2)} nM</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">LogP</span>
                        <span className="font-medium">{dockingResults[0].ligand.logP}</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">
                      분자 상호작용
                    </h4>
                    <ul className="space-y-1 text-sm">
                      {dockingResults[0].interactions.map((interaction, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="text-emerald-500 mt-1">•</span>
                          <span className="text-gray-600 dark:text-gray-400">{interaction}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}