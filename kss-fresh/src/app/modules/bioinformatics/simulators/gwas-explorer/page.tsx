'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, Upload, BarChart3, Activity, TrendingUp, AlertCircle, Download, Filter } from 'lucide-react'

interface SNP {
  id: string
  chromosome: number
  position: number
  alleles: string
  pValue: number
  effectSize: number
  maf: number // Minor Allele Frequency
  gene?: string
}

interface GWASResult {
  phenotype: string
  totalSNPs: number
  significantSNPs: number
  genomicInflationFactor: number
  heritability: number
}

export default function GWASExplorerPage() {
  const [gwasData, setGwasData] = useState<SNP[]>([])
  const [filteredData, setFilteredData] = useState<SNP[]>([])
  const [gwasResult, setGwasResult] = useState<GWASResult | null>(null)
  const [pValueThreshold, setPValueThreshold] = useState(5e-8)
  const [selectedChromosome, setSelectedChromosome] = useState<number | 'all'>('all')
  const [activeView, setActiveView] = useState<'manhattan' | 'qq' | 'table'>('manhattan')

  // Generate sample GWAS data
  useEffect(() => {
    generateSampleData()
  }, [])

  const generateSampleData = () => {
    const sampleSNPs: SNP[] = []
    const genes = ['APOE', 'BRCA1', 'TP53', 'CFTR', 'HBB', 'LDLR', 'PCSK9', 'FTO', 'MC4R', 'TCF7L2']
    
    // Generate random SNPs across 22 chromosomes
    for (let chr = 1; chr <= 22; chr++) {
      const snpCount = Math.floor(Math.random() * 50000) + 30000
      for (let i = 0; i < snpCount; i += 1000) { // Sample every 1000th SNP for visualization
        const pValue = Math.pow(10, -Math.random() * 12) // p-values from 1 to 10^-12
        const isSignificant = pValue < 5e-8
        
        sampleSNPs.push({
          id: `rs${chr}${i}`,
          chromosome: chr,
          position: i * 1000 + Math.floor(Math.random() * 1000),
          alleles: ['A/G', 'C/T', 'G/C', 'T/A'][Math.floor(Math.random() * 4)],
          pValue: pValue,
          effectSize: isSignificant ? Math.random() * 0.5 + 0.1 : Math.random() * 0.1,
          maf: Math.random() * 0.5,
          gene: isSignificant && Math.random() > 0.5 ? genes[Math.floor(Math.random() * genes.length)] : undefined
        })
      }
    }
    
    setGwasData(sampleSNPs)
    setFilteredData(sampleSNPs)
    
    // Calculate GWAS statistics
    const significantSNPs = sampleSNPs.filter(snp => snp.pValue < 5e-8)
    setGwasResult({
      phenotype: 'Type 2 Diabetes',
      totalSNPs: sampleSNPs.length,
      significantSNPs: significantSNPs.length,
      genomicInflationFactor: 1.05 + Math.random() * 0.1,
      heritability: 0.3 + Math.random() * 0.2
    })
  }

  useEffect(() => {
    // Filter data based on selections
    let filtered = gwasData
    
    if (selectedChromosome !== 'all') {
      filtered = filtered.filter(snp => snp.chromosome === selectedChromosome)
    }
    
    setFilteredData(filtered)
  }, [selectedChromosome, gwasData])

  const getManhattanPlotData = () => {
    // Group SNPs by chromosome for Manhattan plot
    const chromosomeData: { [key: number]: SNP[] } = {}
    filteredData.forEach(snp => {
      if (!chromosomeData[snp.chromosome]) {
        chromosomeData[snp.chromosome] = []
      }
      chromosomeData[snp.chromosome].push(snp)
    })
    return chromosomeData
  }

  const getQQPlotData = () => {
    // Sort p-values for QQ plot
    const sortedPValues = filteredData
      .map(snp => snp.pValue)
      .sort((a, b) => a - b)
    
    const n = sortedPValues.length
    const expectedPValues = sortedPValues.map((_, i) => (i + 0.5) / n)
    
    return sortedPValues.map((observed, i) => ({
      observed: -Math.log10(observed),
      expected: -Math.log10(expectedPValues[i])
    }))
  }

  const calculatePRS = () => {
    // Calculate Polygenic Risk Score
    const significantSNPs = filteredData.filter(snp => snp.pValue < pValueThreshold)
    const prs = significantSNPs.reduce((score, snp) => {
      return score + snp.effectSize * (2 * snp.maf) // Simplified PRS calculation
    }, 0)
    return prs.toFixed(4)
  }

  const exportResults = () => {
    const csv = [
      'SNP,Chromosome,Position,Alleles,P-value,Effect Size,MAF,Gene',
      ...filteredData
        .filter(snp => snp.pValue < pValueThreshold)
        .map(snp => 
          `${snp.id},${snp.chromosome},${snp.position},${snp.alleles},${snp.pValue.toExponential()},${snp.effectSize.toFixed(4)},${snp.maf.toFixed(3)},${snp.gene || ''}`
        )
    ].join('\n')
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `gwas_results_${Date.now()}.csv`
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
                GWAS 데이터 탐색기
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Manhattan Plot, QQ Plot, PRS 계산기
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="w-8 h-8 text-emerald-600" />
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex flex-wrap items-center gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                염색체 필터
              </label>
              <select
                value={selectedChromosome}
                onChange={(e) => setSelectedChromosome(e.target.value === 'all' ? 'all' : parseInt(e.target.value))}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
              >
                <option value="all">전체</option>
                {Array.from({ length: 22 }, (_, i) => i + 1).map(chr => (
                  <option key={chr} value={chr}>Chr {chr}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                P-value 임계값
              </label>
              <select
                value={pValueThreshold}
                onChange={(e) => setPValueThreshold(parseFloat(e.target.value))}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
              >
                <option value={5e-8}>5×10⁻⁸ (GWAS)</option>
                <option value={1e-5}>1×10⁻⁵</option>
                <option value={0.05}>0.05</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                시각화 유형
              </label>
              <div className="flex gap-2">
                <button
                  onClick={() => setActiveView('manhattan')}
                  className={`px-3 py-2 rounded-lg ${
                    activeView === 'manhattan'
                      ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  Manhattan
                </button>
                <button
                  onClick={() => setActiveView('qq')}
                  className={`px-3 py-2 rounded-lg ${
                    activeView === 'qq'
                      ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  QQ Plot
                </button>
                <button
                  onClick={() => setActiveView('table')}
                  className={`px-3 py-2 rounded-lg ${
                    activeView === 'table'
                      ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  Table
                </button>
              </div>
            </div>
            
            <button
              onClick={exportResults}
              className="ml-auto px-4 py-2 bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 rounded-lg hover:bg-emerald-200 dark:hover:bg-emerald-800/50 transition-colors flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              내보내기
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Main Visualization */}
          <div className="lg:col-span-3">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              {activeView === 'manhattan' && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                    Manhattan Plot
                  </h3>
                  <div className="h-[400px] bg-gray-50 dark:bg-gray-900 rounded-lg p-4 relative">
                    {/* Simplified Manhattan Plot Visualization */}
                    <div className="absolute top-4 left-4 right-4">
                      <div className="border-b-2 border-red-500 opacity-50" style={{ top: '20%', position: 'absolute', width: '100%' }}>
                        <span className="text-xs text-red-500 absolute -top-5 right-0">p = 5×10⁻⁸</span>
                      </div>
                    </div>
                    <div className="flex items-end justify-around h-full pb-8">
                      {Object.entries(getManhattanPlotData()).slice(0, 10).map(([chr, snps]) => {
                        const maxHeight = Math.max(...snps.map(s => -Math.log10(s.pValue)))
                        return (
                          <div key={chr} className="flex-1 flex flex-col items-center">
                            <div className="w-full flex flex-col justify-end h-full">
                              {snps.slice(0, 50).map((snp, i) => {
                                const height = (-Math.log10(snp.pValue) / 12) * 100
                                const color = snp.pValue < pValueThreshold ? 'bg-red-500' : 
                                             parseInt(chr) % 2 === 0 ? 'bg-blue-400' : 'bg-gray-400'
                                return (
                                  <div
                                    key={i}
                                    className={`w-1 ${color} opacity-70 mx-auto`}
                                    style={{ height: `${height}%`, marginBottom: '1px' }}
                                  />
                                )
                              })}
                            </div>
                            <span className="text-xs mt-2">{chr}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              )}
              
              {activeView === 'qq' && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                    QQ Plot
                  </h3>
                  <div className="h-[400px] bg-gray-50 dark:bg-gray-900 rounded-lg p-8 relative">
                    {/* Simplified QQ Plot */}
                    <div className="absolute inset-8">
                      {/* Diagonal reference line */}
                      <div className="absolute inset-0">
                        <svg className="w-full h-full">
                          <line
                            x1="0"
                            y1="100%"
                            x2="100%"
                            y2="0"
                            stroke="gray"
                            strokeWidth="2"
                            strokeDasharray="5,5"
                            opacity="0.5"
                          />
                        </svg>
                      </div>
                      {/* Points */}
                      <div className="absolute inset-0">
                        {getQQPlotData().slice(0, 100).map((point, i) => {
                          const x = (point.expected / 12) * 100
                          const y = 100 - (point.observed / 12) * 100
                          return (
                            <div
                              key={i}
                              className="absolute w-2 h-2 bg-emerald-500 rounded-full opacity-70"
                              style={{ left: `${x}%`, top: `${y}%` }}
                            />
                          )
                        })}
                      </div>
                    </div>
                    <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2">
                      <span className="text-xs text-gray-600 dark:text-gray-400">Expected -log₁₀(p)</span>
                    </div>
                    <div className="absolute left-0 top-1/2 transform -rotate-90 -translate-y-1/2">
                      <span className="text-xs text-gray-600 dark:text-gray-400">Observed -log₁₀(p)</span>
                    </div>
                  </div>
                </div>
              )}
              
              {activeView === 'table' && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
                    Significant SNPs
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-200 dark:border-gray-700">
                          <th className="text-left p-2">SNP ID</th>
                          <th className="text-left p-2">Chr</th>
                          <th className="text-left p-2">Position</th>
                          <th className="text-left p-2">P-value</th>
                          <th className="text-left p-2">Effect</th>
                          <th className="text-left p-2">Gene</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredData
                          .filter(snp => snp.pValue < pValueThreshold)
                          .sort((a, b) => a.pValue - b.pValue)
                          .slice(0, 20)
                          .map(snp => (
                            <tr key={snp.id} className="border-b border-gray-100 dark:border-gray-700">
                              <td className="p-2 font-mono text-sm">{snp.id}</td>
                              <td className="p-2">{snp.chromosome}</td>
                              <td className="p-2">{snp.position.toLocaleString()}</td>
                              <td className="p-2 font-mono text-sm">{snp.pValue.toExponential(2)}</td>
                              <td className="p-2">{snp.effectSize.toFixed(3)}</td>
                              <td className="p-2">
                                {snp.gene ? (
                                  <span className="px-2 py-1 bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 rounded text-xs">
                                    {snp.gene}
                                  </span>
                                ) : '-'}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            {/* GWAS Statistics */}
            {gwasResult && (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-emerald-600" />
                  GWAS 통계
                </h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">표현형</div>
                    <div className="font-semibold">{gwasResult.phenotype}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">전체 SNPs</div>
                    <div className="font-semibold">{gwasResult.totalSNPs.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">유의미한 SNPs</div>
                    <div className="font-semibold text-emerald-600">{gwasResult.significantSNPs}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Genomic λ</div>
                    <div className="font-semibold">{gwasResult.genomicInflationFactor.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">유전율 (h²)</div>
                    <div className="font-semibold">{(gwasResult.heritability * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            )}

            {/* PRS Calculator */}
            <div className="bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-emerald-600" />
                PRS 계산
              </h3>
              <div className="text-3xl font-bold text-emerald-700 dark:text-emerald-300 mb-2">
                {calculatePRS()}
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Polygenic Risk Score
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                {filteredData.filter(snp => snp.pValue < pValueThreshold).length}개 SNP 기반
              </p>
            </div>

            {/* Info */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-blue-600" />
                GWAS 가이드
              </h3>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li className="flex items-start gap-2">
                  <span className="text-blue-500 mt-1">•</span>
                  <span>Manhattan plot에서 빨간선 위의 점들이 유의미한 SNP입니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500 mt-1">•</span>
                  <span>QQ plot은 p-value 분포의 정상성을 확인합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500 mt-1">•</span>
                  <span>PRS는 질병 위험도를 나타내는 종합 점수입니다</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}