'use client'

import { useState } from 'react'
import { FlaskConical, AlertCircle, CheckCircle, XCircle, BarChart3 } from 'lucide-react'

interface TestResult {
  testStatistic: number
  pValue: number
  criticalValue: number
  rejectNull: boolean
  conclusion: string
}

export default function HypothesisTester() {
  const [testType, setTestType] = useState('one-sample-t')
  const [inputData, setInputData] = useState('')
  const [hypothesisValue, setHypothesisValue] = useState(0)
  const [alternativeType, setAlternativeType] = useState<'two-sided' | 'less' | 'greater'>('two-sided')
  const [significanceLevel, setSignificanceLevel] = useState(0.05)
  const [testResult, setTestResult] = useState<TestResult | null>(null)
  
  // 두 그룹 데이터
  const [inputGroup1, setInputGroup1] = useState('')
  const [inputGroup2, setInputGroup2] = useState('')

  const parseData = (input: string): number[] => {
    return input
      .split(/[,\s]+/)
      .filter(s => s.trim())
      .map(s => parseFloat(s))
      .filter(n => !isNaN(n))
  }

  const getTCriticalValue = (alpha: number): number => {
    if (alpha === 0.05) return 1.96
    if (alpha === 0.01) return 2.576
    return 1.645
  }

  const calculatePValue = (t: number): number => {
    // 간단한 p-value 근사
    const absT = Math.abs(t)
    if (absT > 2.576) return 0.01
    if (absT > 1.96) return 0.05
    if (absT > 1.645) return 0.1
    return 0.2
  }

  const oneSampleTTest = () => {
    const data = parseData(inputData)
    if (data.length < 2) {
      alert('최소 2개 이상의 데이터가 필요합니다.')
      return
    }

    const n = data.length
    const mean = data.reduce((a, b) => a + b, 0) / n
    const variance = data.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / (n - 1)
    const se = Math.sqrt(variance) / Math.sqrt(n)
    const t = (mean - hypothesisValue) / se
    const criticalValue = getTCriticalValue(significanceLevel)
    const pValue = calculatePValue(t)
    const rejectNull = pValue < significanceLevel

    setTestResult({
      testStatistic: t,
      pValue,
      criticalValue,
      rejectNull,
      conclusion: rejectNull 
        ? '귀무가설을 기각합니다. 통계적으로 유의한 차이가 있습니다.'
        : '귀무가설을 기각할 수 없습니다. 통계적으로 유의한 차이가 없습니다.'
    })
  }

  const independentTTest = () => {
    const g1 = parseData(inputGroup1)
    const g2 = parseData(inputGroup2)
    
    if (g1.length < 2 || g2.length < 2) {
      alert('각 그룹에 최소 2개 이상의 데이터가 필요합니다.')
      return
    }

    const n1 = g1.length, n2 = g2.length
    const mean1 = g1.reduce((a, b) => a + b, 0) / n1
    const mean2 = g2.reduce((a, b) => a + b, 0) / n2
    const var1 = g1.reduce((sum, x) => sum + Math.pow(x - mean1, 2), 0) / (n1 - 1)
    const var2 = g2.reduce((sum, x) => sum + Math.pow(x - mean2, 2), 0) / (n2 - 1)
    
    const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    const se = Math.sqrt(pooledVar * (1/n1 + 1/n2))
    const t = (mean1 - mean2) / se
    const criticalValue = getTCriticalValue(significanceLevel)
    const pValue = calculatePValue(t)
    const rejectNull = pValue < significanceLevel

    setTestResult({
      testStatistic: t,
      pValue,
      criticalValue,
      rejectNull,
      conclusion: rejectNull 
        ? '귀무가설을 기각합니다. 두 그룹의 평균이 유의하게 다릅니다.'
        : '귀무가설을 기각할 수 없습니다. 두 그룹의 평균이 유의하게 다르지 않습니다.'
    })
  }

  const runTest = () => {
    switch (testType) {
      case 'one-sample-t':
        oneSampleTTest()
        break
      case 'independent-t':
        independentTTest()
        break
      default:
        oneSampleTTest()
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 mb-6">
        <FlaskConical className="w-6 h-6 text-purple-600" />
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">가설검정 시뮬레이터</h2>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* 설정 패널 */}
        <div className="space-y-4">
          {/* 검정 유형 선택 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">검정 유형</label>
            <select
              value={testType}
              onChange={(e) => setTestType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="one-sample-t">단일표본 t-검정</option>
              <option value="independent-t">독립표본 t-검정</option>
            </select>
          </div>

          {/* 데이터 입력 */}
          {testType === 'one-sample-t' ? (
            <>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">데이터 입력</label>
                <textarea
                  value={inputData}
                  onChange={(e) => setInputData(e.target.value)}
                  placeholder="숫자를 쉼표 또는 공백으로 구분하여 입력하세요"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 h-24 text-gray-900 dark:text-white"
                />
              </div>
              
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">귀무가설 값</label>
                <input
                  type="number"
                  value={hypothesisValue}
                  onChange={(e) => setHypothesisValue(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
            </>
          ) : (
            <>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">그룹 1 데이터</label>
                <textarea
                  value={inputGroup1}
                  onChange={(e) => setInputGroup1(e.target.value)}
                  placeholder="그룹 1의 데이터를 입력하세요"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 h-20 text-gray-900 dark:text-white"
                />
              </div>
              
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">그룹 2 데이터</label>
                <textarea
                  value={inputGroup2}
                  onChange={(e) => setInputGroup2(e.target.value)}
                  placeholder="그룹 2의 데이터를 입력하세요"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 h-20 text-gray-900 dark:text-white"
                />
              </div>
            </>
          )}

          {/* 검정 설정 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">대립가설</label>
            <select
              value={alternativeType}
              onChange={(e) => setAlternativeType(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="two-sided">양측검정 (≠)</option>
              <option value="less">좌측검정 (&lt;)</option>
              <option value="greater">우측검정 (&gt;)</option>
            </select>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">유의수준 (α)</label>
            <select
              value={significanceLevel}
              onChange={(e) => setSignificanceLevel(parseFloat(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value={0.01}>0.01</option>
              <option value={0.05}>0.05</option>
              <option value={0.1}>0.10</option>
            </select>
          </div>

          <button
            onClick={runTest}
            className="w-full px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium"
          >
            검정 실행
          </button>
        </div>

        {/* 결과 패널 */}
        <div className="space-y-4">
          {testResult && (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-4">
                {testResult.rejectNull ? (
                  <XCircle className="w-6 h-6 text-red-600" />
                ) : (
                  <CheckCircle className="w-6 h-6 text-green-600" />
                )}
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">검정 결과</h3>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                  <div className="text-lg font-bold text-gray-900 dark:text-white">
                    {testResult.testStatistic.toFixed(4)}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">검정통계량</div>
                </div>
                
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                  <div className="text-lg font-bold text-gray-900 dark:text-white">
                    {testResult.pValue.toFixed(4)}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">p-값</div>
                </div>
              </div>

              <div className={`p-4 rounded-lg ${testResult.rejectNull 
                ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
                : 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
              }`}>
                <p className={`font-medium ${testResult.rejectNull 
                  ? 'text-red-800 dark:text-red-200' 
                  : 'text-green-800 dark:text-green-200'
                }`}>
                  {testResult.conclusion}
                </p>
              </div>
            </div>
          )}

          {/* 도움말 */}
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
            <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">가설검정이란?</h4>
            <p className="text-sm text-blue-700 dark:text-blue-300">
              통계적 가설검정은 표본 데이터를 바탕으로 모집단에 대한 가설을 검증하는 방법입니다. 
              귀무가설(H₀)과 대립가설(H₁)을 설정하고, 검정통계량과 p-값을 계산하여 가설의 타당성을 판단합니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}