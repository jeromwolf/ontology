'use client'

import { useState, useEffect } from 'react'
import { Calculator, TrendingUp, AlertCircle, DollarSign, Download, ChevronRight } from 'lucide-react'

interface DcfInputs {
  // 현재 재무 데이터
  currentRevenue: number
  currentEbitda: number
  currentCapex: number
  currentWorkingCapital: number
  currentDebt: number
  currentCash: number
  sharesOutstanding: number
  
  // 성장률 가정
  revenueGrowthRate1to5: number
  revenueGrowthRate6to10: number
  ebitdaMargin: number
  capexToRevenue: number
  workingCapitalToRevenue: number
  
  // 할인율 요소
  riskFreeRate: number
  equityRiskPremium: number
  beta: number
  terminalGrowthRate: number
  taxRate: number
}

interface DcfResults {
  wacc: number
  projectedCashFlows: number[]
  terminalValue: number
  enterpriseValue: number
  equityValue: number
  fairValuePerShare: number
  currentPrice: number
  upside: number
}

export default function DcfValuationModel() {
  const [inputs, setInputs] = useState<DcfInputs>({
    currentRevenue: 1000000000, // 10억원
    currentEbitda: 200000000, // 2억원
    currentCapex: 50000000, // 5천만원
    currentWorkingCapital: 100000000, // 1억원
    currentDebt: 300000000, // 3억원
    currentCash: 150000000, // 1.5억원
    sharesOutstanding: 10000000, // 1천만주
    
    revenueGrowthRate1to5: 15, // %
    revenueGrowthRate6to10: 8, // %
    ebitdaMargin: 20, // %
    capexToRevenue: 5, // %
    workingCapitalToRevenue: 10, // %
    
    riskFreeRate: 3.5, // %
    equityRiskPremium: 6.5, // %
    beta: 1.2,
    terminalGrowthRate: 3, // %
    taxRate: 25 // %
  })

  const [results, setResults] = useState<DcfResults | null>(null)
  const [activeTab, setActiveTab] = useState<'inputs' | 'results'>('inputs')

  // DCF 계산 함수
  const calculateDCF = (inputData: DcfInputs): DcfResults => {
    // WACC 계산 (간단화된 버전 - 100% 자기자본 가정)
    const wacc = inputData.riskFreeRate + inputData.beta * inputData.equityRiskPremium
    
    // 10년간 현금흐름 예측
    const projectedCashFlows: number[] = []
    let revenue = inputData.currentRevenue
    
    for (let year = 1; year <= 10; year++) {
      // 매출 성장
      const growthRate = year <= 5 
        ? inputData.revenueGrowthRate1to5 / 100 
        : inputData.revenueGrowthRate6to10 / 100
      revenue = revenue * (1 + growthRate)
      
      // EBITDA
      const ebitda = revenue * (inputData.ebitdaMargin / 100)
      
      // 세후 영업이익 (간단화: EBITDA에서 세금만 차감)
      const nopat = ebitda * (1 - inputData.taxRate / 100)
      
      // CapEx
      const capex = revenue * (inputData.capexToRevenue / 100)
      
      // 운전자본 변화
      const workingCapitalChange = revenue * (inputData.workingCapitalToRevenue / 100) * growthRate
      
      // Free Cash Flow
      const fcf = nopat - capex - workingCapitalChange
      projectedCashFlows.push(fcf)
    }
    
    // Terminal Value 계산
    const terminalFCF = projectedCashFlows[9] * (1 + inputData.terminalGrowthRate / 100)
    const terminalValue = terminalFCF / ((wacc / 100) - (inputData.terminalGrowthRate / 100))
    
    // 현재가치로 할인
    let enterpriseValue = 0
    projectedCashFlows.forEach((cf, index) => {
      enterpriseValue += cf / Math.pow(1 + wacc / 100, index + 1)
    })
    enterpriseValue += terminalValue / Math.pow(1 + wacc / 100, 10)
    
    // Equity Value 계산
    const equityValue = enterpriseValue - inputData.currentDebt + inputData.currentCash
    
    // 주당 가치
    const fairValuePerShare = equityValue / inputData.sharesOutstanding
    
    // 현재가 (예시)
    const currentPrice = 50000 // 50,000원
    const upside = ((fairValuePerShare - currentPrice) / currentPrice) * 100
    
    return {
      wacc,
      projectedCashFlows,
      terminalValue,
      enterpriseValue,
      equityValue,
      fairValuePerShare,
      currentPrice,
      upside
    }
  }

  // 입력값 변경 시 자동 계산
  useEffect(() => {
    const dcfResults = calculateDCF(inputs)
    setResults(dcfResults)
  }, [inputs])

  const handleInputChange = (field: keyof DcfInputs, value: string) => {
    const numValue = parseFloat(value) || 0
    setInputs(prev => ({ ...prev, [field]: numValue }))
  }

  const formatKRW = (value: number): string => {
    if (value >= 1e12) return `₩${(value / 1e12).toFixed(1)}조`
    if (value >= 1e8) return `₩${(value / 1e8).toFixed(1)}억`
    if (value >= 1e4) return `₩${(value / 1e4).toFixed(0)}만`
    return `₩${value.toFixed(0)}`
  }

  const runAnalysis = () => {
    setActiveTab('results')
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
          <Calculator className="w-6 h-6" />
          DCF 가치평가 모델
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('inputs')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'inputs'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            입력값
          </button>
          <button
            onClick={() => setActiveTab('results')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'results'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            분석 결과
          </button>
        </div>
      </div>

      {activeTab === 'inputs' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 현재 재무 데이터 */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-gray-700 dark:text-gray-300">현재 재무 데이터</h4>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  매출액 (원)
                </label>
                <input
                  type="number"
                  value={inputs.currentRevenue || ''}
                  onChange={(e) => handleInputChange('currentRevenue', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  placeholder="현재 매출액"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  EBITDA (원)
                </label>
                <input
                  type="number"
                  value={inputs.currentEbitda || ''}
                  onChange={(e) => handleInputChange('currentEbitda', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  placeholder="현재 EBITDA"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    부채 (원)
                  </label>
                  <input
                    type="number"
                    value={inputs.currentDebt || ''}
                    onChange={(e) => handleInputChange('currentDebt', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="총 부채"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    현금 (원)
                  </label>
                  <input
                    type="number"
                    value={inputs.currentCash || ''}
                    onChange={(e) => handleInputChange('currentCash', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="현금성 자산"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  발행주식수
                </label>
                <input
                  type="number"
                  value={inputs.sharesOutstanding || ''}
                  onChange={(e) => handleInputChange('sharesOutstanding', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  placeholder="총 발행주식수"
                />
              </div>
            </div>
          </div>

          {/* 성장률 및 할인율 */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-gray-700 dark:text-gray-300">성장률 및 할인율 가정</h4>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    1-5년 성장률 (%)
                  </label>
                  <input
                    type="number"
                    value={inputs.revenueGrowthRate1to5 || ''}
                    onChange={(e) => handleInputChange('revenueGrowthRate1to5', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    6-10년 성장률 (%)
                  </label>
                  <input
                    type="number"
                    value={inputs.revenueGrowthRate6to10 || ''}
                    onChange={(e) => handleInputChange('revenueGrowthRate6to10', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  EBITDA 마진 (%)
                </label>
                <input
                  type="number"
                  value={inputs.ebitdaMargin || ''}
                  onChange={(e) => handleInputChange('ebitdaMargin', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    무위험이자율 (%)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={inputs.riskFreeRate || ''}
                    onChange={(e) => handleInputChange('riskFreeRate', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    베타
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={inputs.beta || ''}
                    onChange={(e) => handleInputChange('beta', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  영구성장률 (%)
                </label>
                <input
                  type="number"
                  value={inputs.terminalGrowthRate || ''}
                  onChange={(e) => handleInputChange('terminalGrowthRate', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  법인세율 (%)
                </label>
                <input
                  type="number"
                  value={inputs.taxRate || ''}
                  onChange={(e) => handleInputChange('taxRate', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
            </div>

            <button
              onClick={runAnalysis}
              className="w-full mt-6 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg transition-colors font-medium flex items-center justify-center gap-2"
            >
              DCF 분석 실행
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* 평가 결과 */}
          {results && (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">기업가치</p>
                  <p className="text-xl font-bold text-purple-600 dark:text-purple-400">
                    {formatKRW(results.enterpriseValue)}
                  </p>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">주주가치</p>
                  <p className="text-xl font-bold text-green-600 dark:text-green-400">
                    {formatKRW(results.equityValue)}
                  </p>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">적정주가</p>
                  <p className="text-xl font-bold text-blue-600 dark:text-blue-400">
                    {formatKRW(results.fairValuePerShare)}
                  </p>
                </div>
                <div className={`p-4 rounded-lg ${results.upside > 0 ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'}`}>
                  <p className="text-sm text-gray-600 dark:text-gray-400">상승여력</p>
                  <p className={`text-xl font-bold ${results.upside > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {results.upside > 0 ? '+' : ''}{results.upside.toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* WACC 정보 */}
              <div className="bg-amber-50 dark:bg-amber-900/20 p-4 rounded-lg flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5" />
                <div className="text-sm">
                  <p className="font-semibold text-amber-700 dark:text-amber-300">할인율 (WACC)</p>
                  <p className="text-amber-600 dark:text-amber-400 text-lg font-bold">{results.wacc.toFixed(2)}%</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    무위험이자율 {inputs.riskFreeRate}% + 베타 {inputs.beta} × 시장위험프리미엄 {inputs.equityRiskPremium}%
                  </p>
                </div>
              </div>

              {/* 현금흐름 예측 테이블 */}
              <div>
                <h4 className="text-lg font-semibold mb-4 text-gray-700 dark:text-gray-300 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  10년간 현금흐름 예측
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200 dark:border-gray-700">
                        <th className="text-left py-2 px-4 font-medium text-gray-700 dark:text-gray-300">연도</th>
                        {[...Array(10)].map((_, i) => (
                          <th key={i} className="text-right py-2 px-4 font-medium text-gray-700 dark:text-gray-300">
                            {i + 1}년
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-gray-200 dark:border-gray-700">
                        <td className="py-2 px-4 font-medium text-gray-600 dark:text-gray-400">FCF</td>
                        {results.projectedCashFlows.map((cf, index) => (
                          <td key={index} className="text-right py-2 px-4 text-gray-700 dark:text-gray-300">
                            {formatKRW(cf)}
                          </td>
                        ))}
                      </tr>
                      <tr>
                        <td className="py-2 px-4 font-medium text-gray-600 dark:text-gray-400">현재가치</td>
                        {results.projectedCashFlows.map((cf, index) => (
                          <td key={index} className="text-right py-2 px-4 text-gray-700 dark:text-gray-300">
                            {formatKRW(cf / Math.pow(1 + results.wacc / 100, index + 1))}
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 가치 구성 */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                  <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">기업가치 구성</h5>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">운영가치 (1-10년 FCF)</span>
                      <span className="font-medium text-gray-700 dark:text-gray-300">
                        {formatKRW(results.enterpriseValue - results.terminalValue / Math.pow(1 + results.wacc / 100, 10))}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">터미널 가치</span>
                      <span className="font-medium text-gray-700 dark:text-gray-300">
                        {formatKRW(results.terminalValue / Math.pow(1 + results.wacc / 100, 10))}
                      </span>
                    </div>
                    <div className="pt-2 mt-2 border-t border-gray-200 dark:border-gray-600">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-gray-700 dark:text-gray-300">총 기업가치</span>
                        <span className="font-bold text-purple-600 dark:text-purple-400">
                          {formatKRW(results.enterpriseValue)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                  <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">주주가치 계산</h5>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">기업가치</span>
                      <span className="font-medium text-gray-700 dark:text-gray-300">
                        {formatKRW(results.enterpriseValue)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">(-) 순부채</span>
                      <span className="font-medium text-red-600 dark:text-red-400">
                        -{formatKRW(inputs.currentDebt - inputs.currentCash)}
                      </span>
                    </div>
                    <div className="pt-2 mt-2 border-t border-gray-200 dark:border-gray-600">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-gray-700 dark:text-gray-300">주주가치</span>
                        <span className="font-bold text-green-600 dark:text-green-400">
                          {formatKRW(results.equityValue)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 투자 의견 */}
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <h5 className="font-semibold text-blue-800 dark:text-blue-200 mb-2 flex items-center gap-2">
                  <DollarSign className="w-5 h-5" />
                  DCF 분석 요약
                </h5>
                <div className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                  <p>• 현재 주가 {formatKRW(results.currentPrice)} 대비 적정가치는 {formatKRW(results.fairValuePerShare)}입니다.</p>
                  <p>• {results.upside > 0 ? '저평가' : '고평가'} 상태로 {Math.abs(results.upside).toFixed(1)}%의 {results.upside > 0 ? '상승' : '하락'} 여력이 있습니다.</p>
                  <p>• 터미널 가치가 전체 기업가치의 {((results.terminalValue / Math.pow(1 + results.wacc / 100, 10)) / results.enterpriseValue * 100).toFixed(0)}%를 차지합니다.</p>
                  <p>• 할인율 1%p 변화 시 기업가치는 약 {(results.enterpriseValue * 0.1 / results.wacc).toFixed(0)}% 변동합니다.</p>
                </div>
              </div>

              {/* 다운로드 버튼 */}
              <div className="flex justify-end">
                <button
                  onClick={() => {
                    // Excel 다운로드 기능 (구현 예정)
                    console.log('DCF 분석 결과 다운로드')
                  }}
                  className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Excel 다운로드
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}