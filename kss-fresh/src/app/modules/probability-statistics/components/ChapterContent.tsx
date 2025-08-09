'use client'

import { useState } from 'react'
import { 
  BarChart3, Calculator, Brain, TrendingUp, 
  Zap, AlertCircle, CheckCircle, Lightbulb,
  Code, Play, FlaskConical, Dice1, Activity,
  Target, Shuffle, PieChart, LineChart, Gauge
} from 'lucide-react'

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const renderContent = () => {
    switch (chapterId) {
      case 'probability-basics':
        return <ProbabilityBasicsContent />
      case 'distributions':
        return <DistributionsContent />
      case 'descriptive-statistics':
        return <DescriptiveStatisticsContent />
      case 'inferential-statistics':
        return <InferentialStatisticsContent />
      case 'bayesian-statistics':
        return <BayesianStatisticsContent />
      case 'regression-analysis':
        return <RegressionAnalysisContent />
      case 'time-series':
        return <TimeSeriesContent />
      case 'ml-statistics':
        return <MLStatisticsContent />
      default:
        return <div>챕터 콘텐츠를 찾을 수 없습니다.</div>
    }
  }

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {renderContent()}
    </div>
  )
}

// Chapter 1: 확률의 기초
function ProbabilityBasicsContent() {
  const [diceResult, setDiceResult] = useState<number | null>(null)
  const [coinFlips, setCoinFlips] = useState<string[]>([])

  const rollDice = () => {
    const result = Math.floor(Math.random() * 6) + 1
    setDiceResult(result)
  }

  const flipCoins = () => {
    const flips = Array.from({ length: 10 }, () => 
      Math.random() < 0.5 ? 'H' : 'T'
    )
    setCoinFlips(flips)
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">확률의 기초</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          확률론은 불확실성을 수학적으로 다루는 분야입니다. 
          일상생활의 많은 현상들을 확률적으로 모델링할 수 있습니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Dice1 className="text-rose-500" />
          핵심 개념
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-rose-600 dark:text-rose-400 mb-2">표본공간 (Sample Space)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              모든 가능한 결과들의 집합. 주사위의 경우 S = {'{'} 1, 2, 3, 4, 5, 6 {'}'}
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-rose-600 dark:text-rose-400 mb-2">사건 (Event)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              표본공간의 부분집합. 예: "짝수가 나오는 사건" = {'{'} 2, 4, 6 {'}'}
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-rose-600 dark:text-rose-400 mb-2">확률 (Probability)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              사건이 발생할 가능성을 0과 1 사이의 수로 표현
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-rose-600 dark:text-rose-400 mb-2">조건부 확률</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              P(A|B) = 사건 B가 일어났을 때 사건 A가 일어날 확률
            </p>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">확률의 공리</h3>
        <ol className="list-decimal list-inside space-y-3">
          <li className="text-gray-700 dark:text-gray-300">
            <strong>비음성:</strong> 모든 사건 A에 대해 P(A) ≥ 0
          </li>
          <li className="text-gray-700 dark:text-gray-300">
            <strong>정규화:</strong> 전체 표본공간의 확률은 1, P(S) = 1
          </li>
          <li className="text-gray-700 dark:text-gray-300">
            <strong>가산가법성:</strong> 서로소인 사건들의 합사건의 확률은 각 사건의 확률의 합
          </li>
        </ol>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Play className="text-green-500" />
          인터랙티브 실험
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-3">주사위 던지기</h4>
            <button 
              onClick={rollDice}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors mb-3"
            >
              주사위 던지기
            </button>
            {diceResult && (
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600 dark:text-green-400">
                  {diceResult}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  P(X = {diceResult}) = 1/6 ≈ 0.167
                </p>
              </div>
            )}
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-3">동전 던지기 (10회)</h4>
            <button 
              onClick={flipCoins}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors mb-3"
            >
              동전 10개 던지기
            </button>
            {coinFlips.length > 0 && (
              <div>
                <div className="flex gap-1 flex-wrap mb-2">
                  {coinFlips.map((flip, idx) => (
                    <span key={idx} className={`w-8 h-8 flex items-center justify-center rounded ${
                      flip === 'H' ? 'bg-blue-500 text-white' : 'bg-gray-500 text-white'
                    }`}>
                      {flip}
                    </span>
                  ))}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  앞면: {coinFlips.filter(f => f === 'H').length}개, 
                  뒷면: {coinFlips.filter(f => f === 'T').length}개
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">베이즈 정리</h3>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
          <p className="text-lg font-mono text-center mb-4">
            P(A|B) = P(B|A) × P(A) / P(B)
          </p>
          <p className="text-gray-700 dark:text-gray-300">
            베이즈 정리는 사전 확률을 이용해 사후 확률을 계산하는 강력한 도구입니다.
            머신러닝, 의료 진단, 스팸 필터링 등 다양한 분야에서 활용됩니다.
          </p>
        </div>
        
        <div className="bg-purple-100 dark:bg-purple-800/30 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">예제: 질병 진단</h4>
          <ul className="list-disc list-inside space-y-1 text-sm text-gray-700 dark:text-gray-300">
            <li>질병 유병률: 1% (P(질병) = 0.01)</li>
            <li>검사 정확도: 99% (P(양성|질병) = 0.99)</li>
            <li>오진율: 5% (P(양성|건강) = 0.05)</li>
          </ul>
          <p className="mt-3 text-sm font-semibold text-purple-700 dark:text-purple-300">
            양성 판정시 실제 질병일 확률 ≈ 16.7%
          </p>
        </div>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Lightbulb className="text-yellow-500" />
          실생활 응용
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">보험</h4>
            <p className="text-sm">사고 발생 확률을 계산하여 보험료 책정</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">투자</h4>
            <p className="text-sm">포트폴리오 리스크 관리와 수익률 예측</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">의료</h4>
            <p className="text-sm">검사 결과 해석과 치료 효과 예측</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">AI/ML</h4>
            <p className="text-sm">불확실성 정량화와 예측 모델링</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 2: 확률 분포
function DistributionsContent() {
  const [normalMean, setNormalMean] = useState(0)
  const [normalStd, setNormalStd] = useState(1)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">확률 분포</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          확률 분포는 확률 변수가 가질 수 있는 값들과 그 값들이 나타날 확률을 나타냅니다.
          연속형과 이산형으로 나뉘며, 각각 다양한 특성을 가집니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6">주요 확률 분포</h3>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Activity className="text-blue-500" />
              정규 분포 (Normal Distribution)
            </h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  가장 중요한 연속 확률 분포로, 자연과 사회의 많은 현상이 정규 분포를 따릅니다.
                </p>
                <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
                  <p className="font-mono text-sm">f(x) = (1/σ√(2π)) × e^(-½((x-μ)/σ)²)</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                    μ: 평균, σ: 표준편차
                  </p>
                </div>
              </div>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-semibold mb-1">평균 (μ): {normalMean}</label>
                  <input 
                    type="range" 
                    min="-5" 
                    max="5" 
                    value={normalMean}
                    onChange={(e) => setNormalMean(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold mb-1">표준편차 (σ): {normalStd}</label>
                  <input 
                    type="range" 
                    min="0.5" 
                    max="3" 
                    step="0.1"
                    value={normalStd}
                    onChange={(e) => setNormalStd(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
              <div className="bg-blue-100 dark:bg-blue-900/30 p-2 rounded text-center">
                <p className="font-semibold">68%</p>
                <p className="text-xs">μ ± σ</p>
              </div>
              <div className="bg-blue-200 dark:bg-blue-800/30 p-2 rounded text-center">
                <p className="font-semibold">95%</p>
                <p className="text-xs">μ ± 2σ</p>
              </div>
              <div className="bg-blue-300 dark:bg-blue-700/30 p-2 rounded text-center">
                <p className="font-semibold">99.7%</p>
                <p className="text-xs">μ ± 3σ</p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-4">이항 분포 (Binomial Distribution)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              n번의 독립적인 베르누이 시행에서 성공 횟수의 분포입니다.
            </p>
            <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg mb-4">
              <p className="font-mono text-sm">P(X = k) = C(n,k) × p^k × (1-p)^(n-k)</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                n: 시행 횟수, p: 성공 확률, k: 성공 횟수
              </p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <h5 className="font-semibold mb-2">예제: 동전 던지기</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                공정한 동전을 10번 던질 때 앞면이 정확히 5번 나올 확률:
              </p>
              <p className="text-sm font-mono mt-2">P(X = 5) = C(10,5) × 0.5^5 × 0.5^5 ≈ 0.246</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h4 className="text-xl font-semibold mb-4">포아송 분포 (Poisson Distribution)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              일정 시간 또는 공간에서 발생하는 사건의 횟수를 모델링합니다.
            </p>
            <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg mb-4">
              <p className="font-mono text-sm">P(X = k) = (λ^k × e^(-λ)) / k!</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                λ: 평균 발생률, k: 발생 횟수
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
                <h5 className="font-semibold text-sm mb-1">활용 예시</h5>
                <ul className="text-xs space-y-1">
                  <li>• 콜센터 전화 수</li>
                  <li>• 웹사이트 방문자 수</li>
                  <li>• 교통사고 발생 건수</li>
                </ul>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
                <h5 className="font-semibold text-sm mb-1">특징</h5>
                <ul className="text-xs space-y-1">
                  <li>• 평균 = 분산 = λ</li>
                  <li>• 희귀 사건 모델링</li>
                  <li>• n→∞일 때 이항분포 근사</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">기타 중요 분포</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">지수 분포</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              사건 간 대기 시간 모델링 (예: 고장 시간)
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">감마 분포</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              대기 시간의 합, 신뢰성 분석
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">베타 분포</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              확률의 확률, 베이지안 분석
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">카이제곱 분포</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              적합도 검정, 분산 분석
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 3: 기술 통계
function DescriptiveStatisticsContent() {
  const [dataSet] = useState([23, 25, 27, 29, 31, 33, 35, 37, 39, 41])
  
  const mean = dataSet.reduce((a, b) => a + b) / dataSet.length
  const median = dataSet[Math.floor(dataSet.length / 2)]
  const variance = dataSet.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / dataSet.length
  const stdDev = Math.sqrt(variance)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">기술 통계</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          기술 통계는 데이터의 특성을 요약하고 시각화하는 방법을 다룹니다.
          중심 경향성, 산포도, 분포의 형태 등을 측정합니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <BarChart3 className="text-green-500" />
          중심 경향성 측정
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">평균 (Mean)</h4>
            <p className="text-2xl font-bold mb-2">{mean.toFixed(1)}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              모든 값의 합을 개수로 나눈 값
            </p>
            <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs font-mono">Σx / n</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">중앙값 (Median)</h4>
            <p className="text-2xl font-bold mb-2">{median}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              정렬된 데이터의 중간 위치 값
            </p>
            <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs">이상치에 강건함</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">최빈값 (Mode)</h4>
            <p className="text-2xl font-bold mb-2">-</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              가장 자주 나타나는 값
            </p>
            <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs">범주형 데이터에 유용</p>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-green-100 dark:bg-green-800/30 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">샘플 데이터</h4>
          <div className="flex gap-2 flex-wrap">
            {dataSet.map((val, idx) => (
              <span key={idx} className="px-3 py-1 bg-white dark:bg-gray-700 rounded">
                {val}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Gauge className="text-blue-500" />
          산포도 측정
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">분산 (Variance)</h4>
            <p className="text-2xl font-bold mb-2">{variance.toFixed(2)}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              평균으로부터의 편차 제곱의 평균
            </p>
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs font-mono">σ² = Σ(x - μ)² / n</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">표준편차 (Std Dev)</h4>
            <p className="text-2xl font-bold mb-2">{stdDev.toFixed(2)}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              분산의 제곱근, 원래 단위와 동일
            </p>
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs font-mono">σ = √(σ²)</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">범위 (Range)</h4>
            <p className="text-2xl font-bold mb-2">{Math.max(...dataSet) - Math.min(...dataSet)}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              최댓값과 최솟값의 차이
            </p>
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs">Max - Min</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">사분위수 범위 (IQR)</h4>
            <p className="text-2xl font-bold mb-2">12</p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Q3 - Q1, 중간 50% 데이터의 범위
            </p>
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-xs">이상치에 강건함</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">분포의 형태</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">왜도 (Skewness)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              분포의 비대칭성을 측정
            </p>
            <ul className="text-xs space-y-1">
              <li>• 양의 왜도: 오른쪽 꼬리가 긴 분포</li>
              <li>• 음의 왜도: 왼쪽 꼬리가 긴 분포</li>
              <li>• 0에 가까움: 대칭 분포</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">첨도 (Kurtosis)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              분포의 뾰족함 정도를 측정
            </p>
            <ul className="text-xs space-y-1">
              <li>• 첨도 &gt; 0: 정규분포보다 뾰족함</li>
              <li>• 첨도 &lt; 0: 정규분포보다 평평함</li>
              <li>• 첨도 = 0: 정규분포와 유사</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <PieChart className="text-yellow-500" />
          데이터 시각화
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">히스토그램</h4>
            <p className="text-sm">연속형 데이터의 분포 표현</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">박스플롯</h4>
            <p className="text-sm">5수 요약과 이상치 표시</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">산점도</h4>
            <p className="text-sm">두 변수 간 관계 시각화</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 4: 추론 통계
function InferentialStatisticsContent() {
  const [sampleSize, setSampleSize] = useState(30)
  const [confidenceLevel, setConfidenceLevel] = useState(95)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">추론 통계</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          추론 통계는 표본 데이터를 사용하여 모집단에 대한 결론을 도출하는 방법입니다.
          가설 검정, 신뢰구간, p-값 등의 개념을 다룹니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Target className="text-indigo-500" />
          가설 검정
        </h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg mb-6">
          <h4 className="text-xl font-semibold mb-4">가설 검정의 단계</h4>
          <ol className="list-decimal list-inside space-y-3">
            <li className="text-gray-700 dark:text-gray-300">
              <strong>가설 설정:</strong> 귀무가설(H₀)과 대립가설(H₁) 설정
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              <strong>유의수준 결정:</strong> 일반적으로 α = 0.05 또는 0.01
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              <strong>검정통계량 계산:</strong> t, z, χ² 등 적절한 통계량 선택
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              <strong>p-값 계산:</strong> 관측된 결과가 나올 확률
            </li>
            <li className="text-gray-700 dark:text-gray-300">
              <strong>결론:</strong> p &lt; α면 귀무가설 기각
            </li>
          </ol>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-indigo-100 dark:bg-indigo-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-indigo-700 dark:text-indigo-300 mb-2">제1종 오류 (α)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              귀무가설이 참인데 기각하는 오류
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              "거짓 양성" - 없는 효과를 있다고 판단
            </p>
          </div>
          <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">제2종 오류 (β)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              귀무가설이 거짓인데 기각하지 못하는 오류
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              "거짓 음성" - 있는 효과를 없다고 판단
            </p>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6">신뢰구간</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg mb-4">
          <h4 className="font-semibold mb-4">신뢰구간 계산기</h4>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-semibold mb-2">
                표본 크기 (n): {sampleSize}
              </label>
              <input
                type="range"
                min="10"
                max="100"
                value={sampleSize}
                onChange={(e) => setSampleSize(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-semibold mb-2">
                신뢰수준: {confidenceLevel}%
              </label>
              <div className="flex gap-2">
                {[90, 95, 99].map(level => (
                  <button
                    key={level}
                    onClick={() => setConfidenceLevel(level)}
                    className={`px-3 py-1 rounded ${
                      confidenceLevel === level 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  >
                    {level}%
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <p className="text-sm font-semibold mb-2">해석</p>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              "{confidenceLevel}% 신뢰구간"은 동일한 방법으로 반복 표본추출시 
              {confidenceLevel}%의 구간이 모수를 포함한다는 의미입니다.
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">평균의 신뢰구간</h4>
            <p className="text-sm font-mono mb-2">x̄ ± t × (s/√n)</p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              모분산을 모를 때 t-분포 사용
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">비율의 신뢰구간</h4>
            <p className="text-sm font-mono mb-2">p̂ ± z × √(p̂(1-p̂)/n)</p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              대표본에서 정규 근사 사용
            </p>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">주요 통계 검정</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">t-검정</h4>
            <ul className="text-sm space-y-1">
              <li>• 단일표본 t-검정</li>
              <li>• 독립표본 t-검정</li>
              <li>• 대응표본 t-검정</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">분산분석 (ANOVA)</h4>
            <ul className="text-sm space-y-1">
              <li>• 일원 분산분석</li>
              <li>• 이원 분산분석</li>
              <li>• 반복측정 ANOVA</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">카이제곱 검정</h4>
            <ul className="text-sm space-y-1">
              <li>• 적합도 검정</li>
              <li>• 독립성 검정</li>
              <li>• 동질성 검정</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">비모수 검정</h4>
            <ul className="text-sm space-y-1">
              <li>• Mann-Whitney U</li>
              <li>• Wilcoxon 검정</li>
              <li>• Kruskal-Wallis</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <AlertCircle className="text-red-500" />
          p-값의 올바른 해석
        </h3>
        <div className="space-y-3">
          <div className="flex items-start gap-2">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
            <p className="text-sm text-gray-700 dark:text-gray-300">
              p-값은 귀무가설이 참일 때 관측된 결과 이상으로 극단적인 결과가 나올 확률
            </p>
          </div>
          <div className="flex items-start gap-2">
            <AlertCircle className="text-red-500 mt-1 flex-shrink-0" size={20} />
            <p className="text-sm text-gray-700 dark:text-gray-300">
              p-값은 귀무가설이 참일 확률이 아님
            </p>
          </div>
          <div className="flex items-start gap-2">
            <AlertCircle className="text-red-500 mt-1 flex-shrink-0" size={20} />
            <p className="text-sm text-gray-700 dark:text-gray-300">
              p-값이 작다고 효과가 크다는 의미는 아님
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 5: 베이지안 통계
function BayesianStatisticsContent() {
  const [prior, setPrior] = useState(0.1)
  const [likelihood, setLikelihood] = useState(0.8)
  
  const evidence = prior * likelihood + (1 - prior) * 0.1
  const posterior = (likelihood * prior) / evidence

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">베이지안 통계</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          베이지안 통계는 사전 지식을 활용하여 불확실성을 정량화하고 
          새로운 증거를 통해 믿음을 업데이트하는 방법론입니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Brain className="text-purple-500" />
          베이즈 정리 시뮬레이터
        </h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-semibold mb-2">
                사전 확률 P(H): {(prior * 100).toFixed(1)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={prior}
                onChange={(e) => setPrior(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-semibold mb-2">
                우도 P(E|H): {(likelihood * 100).toFixed(1)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={likelihood}
                onChange={(e) => setLikelihood(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div className="pt-4 border-t">
              <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">계산 결과</h4>
                <p className="text-sm mb-1">
                  증거 P(E) = {(evidence * 100).toFixed(2)}%
                </p>
                <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                  사후 확률 P(H|E) = {(posterior * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
            <p className="text-sm font-mono text-center">
              P(H|E) = P(E|H) × P(H) / P(E)
            </p>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">빈도주의 vs 베이지안</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">빈도주의 접근</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>확률은 장기적 빈도</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>모수는 고정된 값</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>p-값과 신뢰구간 사용</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>객관적이고 반복 가능</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">베이지안 접근</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <span>확률은 믿음의 정도</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <span>모수는 확률 분포</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <span>사후 분포와 신용구간</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <span>사전 지식 활용 가능</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">베이지안 추론 과정</h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex items-start gap-4">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">1</div>
            <div>
              <h4 className="font-semibold mb-1">사전 분포 선택</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                과거 경험이나 전문가 의견을 반영한 초기 믿음
              </p>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex items-start gap-4">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">2</div>
            <div>
              <h4 className="font-semibold mb-1">데이터 수집</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                실험이나 관찰을 통한 새로운 증거 확보
              </p>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex items-start gap-4">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">3</div>
            <div>
              <h4 className="font-semibold mb-1">우도 계산</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                주어진 모수 값에서 데이터가 관측될 확률
              </p>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg flex items-start gap-4">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">4</div>
            <div>
              <h4 className="font-semibold mb-1">사후 분포 도출</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                베이즈 정리를 통해 업데이트된 믿음 계산
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">베이지안 방법의 장점</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              불확실성 정량화
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              모든 모수에 대한 전체 확률 분포 제공
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              작은 표본 크기
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              사전 정보 활용으로 소규모 데이터에서도 유용
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              순차적 업데이트
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              새 데이터가 들어올 때마다 점진적 업데이트
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              의사결정 지원
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              손실 함수와 결합하여 최적 의사결정
            </p>
          </div>
        </div>
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">실제 응용 사례</h3>
        
        <div className="space-y-3">
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">의료 진단</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              증상과 검사 결과를 종합한 질병 확률 계산
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">A/B 테스팅</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              전환율 차이의 확률적 평가와 조기 종료 결정
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-1">추천 시스템</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              사용자 선호도의 불확실성을 고려한 개인화
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 6: 회귀 분석
function RegressionAnalysisContent() {
  const [slope] = useState(2.5)
  const [intercept] = useState(10)
  const [rSquared] = useState(0.85)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">회귀 분석</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          회귀 분석은 변수들 간의 관계를 모델링하고 예측하는 통계적 방법입니다.
          독립변수와 종속변수 간의 함수 관계를 추정합니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <LineChart className="text-cyan-500" />
          단순 선형 회귀
        </h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <div className="mb-4">
            <p className="text-lg font-mono text-center mb-4">
              Y = β₀ + β₁X + ε
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">모델 파라미터</h4>
                <ul className="space-y-2 text-sm">
                  <li>• β₀ (절편): {intercept}</li>
                  <li>• β₁ (기울기): {slope}</li>
                  <li>• R² (결정계수): {rSquared}</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">해석</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  X가 1단위 증가할 때 Y는 평균적으로 {slope}단위 증가합니다.
                  모델이 전체 변동의 {(rSquared * 100).toFixed(0)}%를 설명합니다.
                </p>
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-cyan-100 dark:bg-cyan-900/30 rounded-lg">
            <h4 className="font-semibold mb-2">가정 사항</h4>
            <ul className="text-sm space-y-1">
              <li>• 선형성: X와 Y의 관계가 선형</li>
              <li>• 독립성: 오차항들이 서로 독립</li>
              <li>• 등분산성: 오차의 분산이 일정</li>
              <li>• 정규성: 오차가 정규분포를 따름</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">다중 회귀 분석</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg mb-4">
          <p className="text-lg font-mono text-center mb-4">
            Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
          </p>
          <p className="text-sm text-gray-700 dark:text-gray-300 text-center">
            여러 독립변수를 사용하여 종속변수를 예측
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
              다중공선성
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              독립변수들 간의 높은 상관관계
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              VIF &gt; 10이면 문제 있음
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
              변수 선택
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              중요한 변수만 포함시키기
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              전진, 후진, 단계적 선택법
            </p>
          </div>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">회귀 진단</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">
              잔차 분석
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 잔차 플롯으로 패턴 확인</li>
              <li>• Q-Q 플롯으로 정규성 검정</li>
              <li>• 표준화 잔차 확인</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">
              영향력 진단
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 레버리지 (leverage)</li>
              <li>• Cook's distance</li>
              <li>• DFFITS, DFBETAS</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">고급 회귀 기법</h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              다항 회귀 (Polynomial Regression)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              비선형 관계를 모델링할 때 사용
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              로지스틱 회귀 (Logistic Regression)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              log(p/(1-p)) = β₀ + β₁X₁ + ... + βₚXₚ
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              이진 분류 문제에 사용
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
              정규화 회귀 (Regularization)
            </h4>
            <div className="grid grid-cols-3 gap-2 mt-2">
              <div className="text-center">
                <p className="text-xs font-semibold">Ridge (L2)</p>
                <p className="text-xs">계수 축소</p>
              </div>
              <div className="text-center">
                <p className="text-xs font-semibold">Lasso (L1)</p>
                <p className="text-xs">변수 선택</p>
              </div>
              <div className="text-center">
                <p className="text-xs font-semibold">Elastic Net</p>
                <p className="text-xs">L1 + L2</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Calculator className="text-yellow-500" />
          회귀 분석 체크리스트
        </h3>
        
        <div className="space-y-2">
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">데이터 탐색 및 시각화</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">변수 변환 필요성 검토</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">모델 적합 및 계수 해석</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">회귀 가정 검토</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">이상치 및 영향점 확인</span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-1" />
            <span className="text-sm">모델 검증 (교차 검증)</span>
          </label>
        </div>
      </div>
    </div>
  )
}

// Chapter 7: 시계열 분석
function TimeSeriesContent() {
  const [trendType, setTrendType] = useState('linear')
  const [seasonalPeriod] = useState(12)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">시계열 분석</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          시계열 분석은 시간에 따라 순차적으로 관측된 데이터를 분석하고 
          미래 값을 예측하는 통계적 방법입니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <TrendingUp className="text-teal-500" />
          시계열의 구성 요소
        </h3>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
              추세 (Trend)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              장기적인 방향성
            </p>
            <select 
              value={trendType}
              onChange={(e) => setTrendType(e.target.value)}
              className="mt-2 text-xs w-full p-1 rounded border dark:bg-gray-700"
            >
              <option value="linear">선형</option>
              <option value="exponential">지수</option>
              <option value="polynomial">다항</option>
            </select>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
              계절성 (Seasonal)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              주기적 패턴
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              주기: {seasonalPeriod}개월
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
              순환 (Cyclic)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              불규칙 장기 변동
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              경기 순환 등
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
              불규칙 (Irregular)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              무작위 변동
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              예측 불가능
            </p>
          </div>
        </div>
        
        <div className="mt-6 bg-teal-100 dark:bg-teal-900/30 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">분해 모델</h4>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <p className="font-semibold">가법 모델</p>
              <p className="font-mono text-xs">Y = Trend + Seasonal + Irregular</p>
            </div>
            <div>
              <p className="font-semibold">승법 모델</p>
              <p className="font-mono text-xs">Y = Trend × Seasonal × Irregular</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">정상성 (Stationarity)</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg mb-4">
          <h4 className="font-semibold mb-3">정상 시계열의 조건</h4>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-blue-500">1.</span>
              <span className="text-sm">평균이 시간에 따라 일정</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500">2.</span>
              <span className="text-sm">분산이 시간에 따라 일정</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500">3.</span>
              <span className="text-sm">자기공분산이 시차에만 의존</span>
            </li>
          </ul>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
              정상성 검정
            </h4>
            <ul className="text-sm space-y-1">
              <li>• ADF 검정</li>
              <li>• KPSS 검정</li>
              <li>• Phillips-Perron 검정</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
              정상화 방법
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 차분 (Differencing)</li>
              <li>• 로그 변환</li>
              <li>• 추세 제거</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">ARIMA 모델</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <p className="text-lg font-mono text-center mb-4">
            ARIMA(p, d, q)
          </p>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400">AR(p)</h4>
              <p className="text-sm mt-1">자기회귀</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                과거 p개 시점의 값 사용
              </p>
            </div>
            
            <div className="text-center">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400">I(d)</h4>
              <p className="text-sm mt-1">차분</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                d번 차분으로 정상화
              </p>
            </div>
            
            <div className="text-center">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400">MA(q)</h4>
              <p className="text-sm mt-1">이동평균</p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                과거 q개 오차항 사용
              </p>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
            <h4 className="font-semibold mb-2">모델 선택 과정</h4>
            <ol className="text-sm space-y-1 list-decimal list-inside">
              <li>ACF/PACF 플롯 확인</li>
              <li>Box-Jenkins 방법론 적용</li>
              <li>AIC/BIC 기준으로 모델 비교</li>
              <li>잔차 진단</li>
            </ol>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">예측 기법</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
              전통적 방법
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 단순 이동평균</li>
              <li>• 지수 평활법</li>
              <li>• Holt-Winters</li>
              <li>• X-13ARIMA-SEATS</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
              현대적 방법
            </h4>
            <ul className="text-sm space-y-1">
              <li>• Prophet (Facebook)</li>
              <li>• LSTM/GRU</li>
              <li>• Transformer</li>
              <li>• 앙상블 방법</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">예측 성능 평가</h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">MAE</h4>
            <p className="text-xs text-gray-600 dark:text-gray-400">Mean Absolute Error</p>
            <p className="text-sm font-mono mt-2">Σ|실제-예측|/n</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">RMSE</h4>
            <p className="text-xs text-gray-600 dark:text-gray-400">Root Mean Square Error</p>
            <p className="text-sm font-mono mt-2">√(Σ(실제-예측)²/n)</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">MAPE</h4>
            <p className="text-xs text-gray-600 dark:text-gray-400">Mean Absolute Percentage Error</p>
            <p className="text-sm font-mono mt-2">Σ|실제-예측|/실제 × 100</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Chapter 8: ML을 위한 통계
function MLStatisticsContent() {
  const [regularizationStrength, setRegularizationStrength] = useState(0.1)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">머신러닝을 위한 통계</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          머신러닝의 이론적 기반이 되는 통계적 개념들을 살펴봅니다.
          모델의 성능 평가, 과적합 방지, 검증 방법 등을 다룹니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Zap className="text-violet-500" />
          편향-분산 트레이드오프
        </h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <div className="text-center mb-4">
            <p className="text-lg font-mono">
              총 오차 = 편향² + 분산 + 노이즈
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-violet-100 dark:bg-violet-900/30 p-4 rounded-lg">
              <h4 className="font-semibold text-violet-700 dark:text-violet-300 mb-2">
                높은 편향 (Underfitting)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 모델이 너무 단순함</li>
                <li>• 훈련/테스트 오차 모두 높음</li>
                <li>• 중요한 패턴을 놓침</li>
              </ul>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                해결: 복잡한 모델, 더 많은 특성
              </p>
            </div>
            
            <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
                높은 분산 (Overfitting)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 모델이 너무 복잡함</li>
                <li>• 훈련 오차는 낮지만 테스트 오차 높음</li>
                <li>• 노이즈까지 학습</li>
              </ul>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                해결: 정규화, 더 많은 데이터
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">교차 검증 (Cross-Validation)</h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
              K-Fold Cross-Validation
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              데이터를 K개 폴드로 나누어 K번 학습/검증
            </p>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map(fold => (
                <div 
                  key={fold} 
                  className={`flex-1 h-8 flex items-center justify-center text-xs font-semibold rounded ${
                    fold === 3 ? 'bg-blue-500 text-white' : 'bg-gray-300 dark:bg-gray-600'
                  }`}
                >
                  {fold === 3 ? 'Test' : 'Train'}
                </div>
              ))}
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                LOOCV
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                Leave-One-Out: n개 샘플에 대해 n번 검증
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                작은 데이터셋에 유용
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                Stratified K-Fold
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                각 폴드의 클래스 비율 유지
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                불균형 데이터에 필수
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">정규화 (Regularization)</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <div className="mb-4">
            <label className="block text-sm font-semibold mb-2">
              정규화 강도 (λ): {regularizationStrength.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={regularizationStrength}
              onChange={(e) => setRegularizationStrength(Number(e.target.value))}
              className="w-full"
            />
          </div>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-1">L1 (Lasso)</h4>
              <p className="text-xs">계수를 0으로 만들어 변수 선택</p>
              <p className="text-xs font-mono mt-1">Σ|βᵢ|</p>
            </div>
            
            <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-1">L2 (Ridge)</h4>
              <p className="text-xs">계수를 축소하여 과적합 방지</p>
              <p className="text-xs font-mono mt-1">Σβᵢ²</p>
            </div>
            
            <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-1">Elastic Net</h4>
              <p className="text-xs">L1과 L2의 장점 결합</p>
              <p className="text-xs font-mono mt-1">αΣ|βᵢ| + (1-α)Σβᵢ²</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">성능 평가 지표</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">분류 지표</h4>
            <div className="space-y-2">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">정확도 (Accuracy)</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">전체 예측 중 맞춘 비율</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">정밀도 (Precision)</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">양성 예측 중 실제 양성 비율</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">재현율 (Recall)</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">실제 양성 중 양성 예측 비율</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">F1 Score</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">정밀도와 재현율의 조화평균</p>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">회귀 지표</h4>
            <div className="space-y-2">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">MAE</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">평균 절대 오차</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">MSE/RMSE</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">평균 제곱 오차</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">R² Score</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">결정 계수</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">MAPE</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">평균 절대 백분율 오차</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">특성 공학과 선택</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">
              특성 공학
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 다항 특성 생성</li>
              <li>• 교호작용 항</li>
              <li>• 로그/지수 변환</li>
              <li>• 구간화 (Binning)</li>
              <li>• 원-핫 인코딩</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">
              특성 선택
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 필터 방법 (상관계수, χ²)</li>
              <li>• 래퍼 방법 (RFE)</li>
              <li>• 임베디드 방법 (Lasso)</li>
              <li>• 순열 중요도</li>
              <li>• SHAP 값</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <FlaskConical className="text-yellow-500" />
          A/B 테스팅과 실험 설계
        </h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">표본 크기 계산</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              최소 탐지 효과(MDE), 검정력, 유의수준을 고려
            </p>
            <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
              n = 2 × (Z_α/2 + Z_β)² × σ² / δ²
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">
                다중 검정 보정
              </h4>
              <ul className="text-sm space-y-1">
                <li>• Bonferroni 보정</li>
                <li>• Benjamini-Hochberg</li>
                <li>• False Discovery Rate</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">
                실험 설계 원칙
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 무작위 배정</li>
                <li>• 통제 변수</li>
                <li>• 실험 기간 설정</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}