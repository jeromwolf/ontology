'use client'

import { useState, useEffect } from 'react'
import { Calculator, TrendingUp, TrendingDown, Minus, Loader2, RefreshCw } from 'lucide-react'
import { stockDataService, KOREAN_STOCKS } from '@/lib/services/stock-data.service'

interface FinancialData {
  stockPrice: number
  eps: number
  bps: number
  roe: number
  revenue: number
  operatingIncome: number
  netIncome: number
  totalAssets: number
  totalDebt: number
  equity: number
}

interface AnalysisResult {
  per: number
  pbr: number
  roic: number
  debtRatio: number
  operatingMargin: number
  netMargin: number
  recommendation: 'buy' | 'hold' | 'sell'
  score: number
}

export default function FinancialAnalyzerWithAPI() {
  const [selectedStock, setSelectedStock] = useState<string>('005930') // 삼성전자
  const [financialData, setFinancialData] = useState<FinancialData>({
    stockPrice: 0,
    eps: 0,
    bps: 0,
    roe: 0,
    revenue: 0,
    operatingIncome: 0,
    netIncome: 0,
    totalAssets: 0,
    totalDebt: 0,
    equity: 0
  })
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [useRealData, setUseRealData] = useState(true)

  // 실시간 데이터 가져오기
  const fetchRealTimeData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // 주식 가격 데이터
      const stockData = await stockDataService.getStockData(selectedStock)
      
      // 재무 데이터
      const financialInfo = await stockDataService.getFinancialData(selectedStock)
      
      setFinancialData({
        stockPrice: stockData.currentPrice,
        eps: financialInfo.eps,
        bps: financialInfo.bps,
        roe: financialInfo.roe,
        revenue: financialInfo.revenue,
        operatingIncome: financialInfo.operatingIncome,
        netIncome: financialInfo.netIncome,
        totalAssets: financialInfo.revenue * 1.5, // 추정치
        totalDebt: financialInfo.revenue * financialInfo.debtRatio / 100,
        equity: financialInfo.bps * 1000000 // 추정치
      })
      
      // 자동으로 분석 실행
      const result = calculateAnalysis({
        stockPrice: stockData.currentPrice,
        eps: financialInfo.eps,
        bps: financialInfo.bps,
        roe: financialInfo.roe,
        revenue: financialInfo.revenue,
        operatingIncome: financialInfo.operatingIncome,
        netIncome: financialInfo.netIncome,
        totalAssets: financialInfo.revenue * 1.5,
        totalDebt: financialInfo.revenue * financialInfo.debtRatio / 100,
        equity: financialInfo.bps * 1000000
      })
      
      setAnalysis(result)
    } catch (err) {
      setError('실시간 데이터를 가져오는데 실패했습니다. 목업 데이터를 사용합니다.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (useRealData) {
      fetchRealTimeData()
    }
  }, [selectedStock, useRealData])

  const handleInputChange = (field: keyof FinancialData, value: string) => {
    setFinancialData(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }))
  }

  const calculateAnalysis = (data: FinancialData = financialData): AnalysisResult => {
    const per = data.eps > 0 ? data.stockPrice / data.eps : 0
    const pbr = data.bps > 0 ? data.stockPrice / data.bps : 0
    const roic = data.totalDebt + data.equity > 0 
      ? (data.operatingIncome * 0.75) / (data.totalDebt + data.equity) * 100 
      : 0
    const debtRatio = data.totalAssets > 0 
      ? (data.totalDebt / data.totalAssets) * 100 
      : 0
    const operatingMargin = data.revenue > 0 
      ? (data.operatingIncome / data.revenue) * 100 
      : 0
    const netMargin = data.revenue > 0 
      ? (data.netIncome / data.revenue) * 100 
      : 0

    // 투자 점수 계산 (100점 만점)
    let score = 50 // 기본 점수

    // PER 평가 (20점)
    if (per > 0 && per < 10) score += 20
    else if (per < 15) score += 15
    else if (per < 20) score += 10
    else if (per < 25) score += 5
    else if (per > 30) score -= 10

    // PBR 평가 (15점)
    if (pbr > 0 && pbr < 1) score += 15
    else if (pbr < 1.5) score += 10
    else if (pbr < 2) score += 5
    else if (pbr > 3) score -= 5

    // ROE 평가 (20점)
    if (data.roe > 20) score += 20
    else if (data.roe > 15) score += 15
    else if (data.roe > 10) score += 10
    else if (data.roe > 5) score += 5
    else score -= 10

    // 영업이익률 평가 (15점)
    if (operatingMargin > 20) score += 15
    else if (operatingMargin > 15) score += 12
    else if (operatingMargin > 10) score += 8
    else if (operatingMargin > 5) score += 4
    else score -= 5

    // 부채비율 평가 (10점)
    if (debtRatio < 30) score += 10
    else if (debtRatio < 50) score += 5
    else if (debtRatio < 70) score += 0
    else score -= 10

    score = Math.max(0, Math.min(100, score))

    let recommendation: 'buy' | 'hold' | 'sell'
    if (score >= 70) recommendation = 'buy'
    else if (score >= 40) recommendation = 'hold'
    else recommendation = 'sell'

    return {
      per,
      pbr,
      roic,
      debtRatio,
      operatingMargin,
      netMargin,
      recommendation,
      score
    }
  }

  const runAnalysis = () => {
    const result = calculateAnalysis()
    setAnalysis(result)
  }

  const formatNumber = (num: number): string => {
    if (num >= 1e12) return `${(num / 1e12).toFixed(2)}조`
    if (num >= 1e8) return `${(num / 1e8).toFixed(2)}억`
    if (num >= 1e4) return `${(num / 1e4).toFixed(2)}만`
    return num.toFixed(2)
  }

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'buy': return 'text-green-600 dark:text-green-400'
      case 'hold': return 'text-yellow-600 dark:text-yellow-400'
      case 'sell': return 'text-red-600 dark:text-red-400'
      default: return ''
    }
  }

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation) {
      case 'buy': return <TrendingUp className="w-5 h-5" />
      case 'hold': return <Minus className="w-5 h-5" />
      case 'sell': return <TrendingDown className="w-5 h-5" />
      default: return null
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-2xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
          <Calculator className="w-6 h-6" />
          재무 분석기
        </h3>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={useRealData}
              onChange={(e) => setUseRealData(e.target.checked)}
              className="rounded text-blue-600"
            />
            <span className="text-sm text-gray-600 dark:text-gray-400">실시간 데이터</span>
          </label>
          {useRealData && (
            <button
              onClick={fetchRealTimeData}
              disabled={loading}
              className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>

      {useRealData && (
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            종목 선택
          </label>
          <select
            value={selectedStock}
            onChange={(e) => setSelectedStock(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            {Object.entries(KOREAN_STOCKS).map(([name, code]) => (
              <option key={code} value={code}>
                {name} ({code})
              </option>
            ))}
          </select>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300 rounded-lg text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold mb-4 text-gray-700 dark:text-gray-300">재무 데이터 입력</h4>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                주가 (원)
              </label>
              <input
                type="number"
                value={financialData.stockPrice || ''}
                onChange={(e) => handleInputChange('stockPrice', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                disabled={useRealData && loading}
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  EPS (원)
                </label>
                <input
                  type="number"
                  value={financialData.eps || ''}
                  onChange={(e) => handleInputChange('eps', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  disabled={useRealData && loading}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  BPS (원)
                </label>
                <input
                  type="number"
                  value={financialData.bps || ''}
                  onChange={(e) => handleInputChange('bps', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  disabled={useRealData && loading}
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                ROE (%)
              </label>
              <input
                type="number"
                value={financialData.roe || ''}
                onChange={(e) => handleInputChange('roe', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                disabled={useRealData && loading}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                매출액 (원)
              </label>
              <input
                type="number"
                value={financialData.revenue || ''}
                onChange={(e) => handleInputChange('revenue', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                disabled={useRealData && loading}
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  영업이익 (원)
                </label>
                <input
                  type="number"
                  value={financialData.operatingIncome || ''}
                  onChange={(e) => handleInputChange('operatingIncome', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  disabled={useRealData && loading}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  순이익 (원)
                </label>
                <input
                  type="number"
                  value={financialData.netIncome || ''}
                  onChange={(e) => handleInputChange('netIncome', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  disabled={useRealData && loading}
                />
              </div>
            </div>
          </div>

          {!useRealData && (
            <button
              onClick={runAnalysis}
              className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors font-medium"
            >
              분석 실행
            </button>
          )}
        </div>

        <div>
          <h4 className="text-lg font-semibold mb-4 text-gray-700 dark:text-gray-300">분석 결과</h4>
          {analysis ? (
            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-semibold text-gray-700 dark:text-gray-300">투자 추천</span>
                  <span className={`font-bold text-lg flex items-center gap-2 ${getRecommendationColor(analysis.recommendation)}`}>
                    {getRecommendationIcon(analysis.recommendation)}
                    {analysis.recommendation.toUpperCase()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600 dark:text-gray-400">투자 점수</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          analysis.score >= 70 ? 'bg-green-500' :
                          analysis.score >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${analysis.score}%` }}
                      />
                    </div>
                    <span className="font-semibold text-gray-700 dark:text-gray-300">
                      {analysis.score.toFixed(0)}/100
                    </span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">PER</div>
                  <div className="text-lg font-semibold text-gray-800 dark:text-white">
                    {analysis.per.toFixed(2)}
                  </div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">PBR</div>
                  <div className="text-lg font-semibold text-gray-800 dark:text-white">
                    {analysis.pbr.toFixed(2)}
                  </div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">ROIC</div>
                  <div className="text-lg font-semibold text-gray-800 dark:text-white">
                    {analysis.roic.toFixed(2)}%
                  </div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">부채비율</div>
                  <div className="text-lg font-semibold text-gray-800 dark:text-white">
                    {analysis.debtRatio.toFixed(2)}%
                  </div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">영업이익률</div>
                  <div className="text-lg font-semibold text-gray-800 dark:text-white">
                    {analysis.operatingMargin.toFixed(2)}%
                  </div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">순이익률</div>
                  <div className="text-lg font-semibold text-gray-800 dark:text-white">
                    {analysis.netMargin.toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
                <h5 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">분석 요약</h5>
                <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                  <li>• PER {analysis.per.toFixed(1)}배로 {analysis.per < 15 ? '저평가' : analysis.per > 25 ? '고평가' : '적정 수준'}</li>
                  <li>• ROE {financialData.roe.toFixed(1)}%로 {financialData.roe > 15 ? '우수한' : financialData.roe > 10 ? '양호한' : '낮은'} 수익성</li>
                  <li>• 부채비율 {analysis.debtRatio.toFixed(1)}%로 {analysis.debtRatio < 50 ? '안정적' : analysis.debtRatio < 100 ? '보통' : '위험'}</li>
                  <li>• 영업이익률 {analysis.operatingMargin.toFixed(1)}%로 {analysis.operatingMargin > 15 ? '높은' : analysis.operatingMargin > 10 ? '적정' : '낮은'} 수준</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-400 dark:text-gray-600">
              {loading ? (
                <div className="text-center">
                  <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                  <p>실시간 데이터를 불러오는 중...</p>
                </div>
              ) : (
                <p>재무 데이터를 입력하고 분석을 실행하세요</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}