'use client'

import { useState, useMemo } from 'react'
import SimulatorNav from './SimulatorNav'
import { AlertTriangle, BarChart3, TrendingUp, AlertCircle } from 'lucide-react'

type Dataset = 'credit' | 'hiring' | 'healthcare' | 'criminal'
type Demographic = 'gender' | 'race' | 'age'

interface GroupMetrics {
  group: string
  total: number
  positive: number
  truePositive: number
  falsePositive: number
  trueNegative: number
  falseNegative: number
  accuracy: number
  tpr: number // True Positive Rate
  fpr: number // False Positive Rate
  ppv: number // Positive Predictive Value
}

const DATASETS = {
  credit: {
    name: '신용 평가',
    description: '대출 승인 예측 모델',
    groups: {
      gender: [
        { group: 'Male', total: 5000, positive: 3500, tp: 3200, fp: 300, tn: 1300, fn: 200 },
        { group: 'Female', total: 5000, positive: 2800, tp: 2400, fp: 400, tn: 1700, fn: 500 }
      ],
      race: [
        { group: 'White', total: 4000, positive: 3200, tp: 2900, fp: 300, tn: 700, fn: 100 },
        { group: 'Black', total: 3000, positive: 1800, tp: 1500, fp: 300, tn: 1000, fn: 200 },
        { group: 'Asian', total: 3000, positive: 2400, tp: 2200, fp: 200, tn: 500, fn: 100 }
      ],
      age: [
        { group: '18-30', total: 3000, positive: 2100, tp: 1900, fp: 200, tn: 800, fn: 100 },
        { group: '31-50', total: 4000, positive: 3200, tp: 3000, fp: 200, tn: 700, fn: 100 },
        { group: '51+', total: 3000, positive: 1800, tp: 1600, fp: 200, tn: 1100, fn: 100 }
      ]
    }
  },
  hiring: {
    name: '채용 심사',
    description: '면접 통과 예측 모델',
    groups: {
      gender: [
        { group: 'Male', total: 3000, positive: 1800, tp: 1600, fp: 200, tn: 1100, fn: 100 },
        { group: 'Female', total: 3000, positive: 1200, tp: 1000, fp: 200, tn: 1700, fn: 100 }
      ],
      race: [
        { group: 'White', total: 2500, positive: 1750, tp: 1600, fp: 150, tn: 700, fn: 50 },
        { group: 'Black', total: 2000, positive: 1000, tp: 850, fp: 150, tn: 950, fn: 50 },
        { group: 'Hispanic', total: 1500, positive: 750, tp: 650, fp: 100, tn: 700, fn: 50 }
      ],
      age: [
        { group: '22-30', total: 2500, positive: 1750, tp: 1600, fp: 150, tn: 700, fn: 50 },
        { group: '31-45', total: 2500, positive: 1500, tp: 1350, fp: 150, tn: 950, fn: 50 },
        { group: '46+', total: 1000, positive: 400, tp: 350, fp: 50, tn: 550, fn: 50 }
      ]
    }
  },
  healthcare: {
    name: '의료 진단',
    description: '질병 위험도 예측 모델',
    groups: {
      gender: [
        { group: 'Male', total: 4000, positive: 2400, tp: 2200, fp: 200, tn: 1500, fn: 100 },
        { group: 'Female', total: 4000, positive: 2000, tp: 1800, fp: 200, tn: 1900, fn: 100 }
      ],
      race: [
        { group: 'White', total: 3500, positive: 2100, tp: 1950, fp: 150, tn: 1350, fn: 50 },
        { group: 'Black', total: 2500, positive: 1250, tp: 1100, fp: 150, tn: 1200, fn: 50 },
        { group: 'Hispanic', total: 2000, positive: 1000, tp: 900, fp: 100, tn: 950, fn: 50 }
      ],
      age: [
        { group: '0-18', total: 2000, positive: 600, tp: 550, fp: 50, tn: 1400, fn: 0 },
        { group: '19-60', total: 4000, positive: 2000, tp: 1850, fp: 150, tn: 1950, fn: 50 },
        { group: '61+', total: 2000, positive: 1400, tp: 1300, fp: 100, tn: 550, fn: 50 }
      ]
    }
  },
  criminal: {
    name: '형사 사법',
    description: '재범 위험도 예측 모델 (COMPAS-style)',
    groups: {
      gender: [
        { group: 'Male', total: 6000, positive: 3600, tp: 3200, fp: 400, tn: 2200, fn: 200 },
        { group: 'Female', total: 2000, positive: 800, tp: 600, fp: 200, tn: 1100, fn: 100 }
      ],
      race: [
        { group: 'White', total: 4000, positive: 2000, tp: 1700, fp: 300, tn: 1900, fn: 100 },
        { group: 'Black', total: 4000, positive: 2400, tp: 2000, fp: 400, tn: 1500, fn: 100 }
      ],
      age: [
        { group: '18-25', total: 3000, positive: 2100, tp: 1900, fp: 200, tn: 800, fn: 100 },
        { group: '26-40', total: 3500, positive: 2100, tp: 1850, fp: 250, tn: 1300, fn: 100 },
        { group: '41+', total: 1500, positive: 600, tp: 500, fp: 100, tn: 850, fn: 50 }
      ]
    }
  }
}

export default function BiasDetector() {
  const [dataset, setDataset] = useState<Dataset>('credit')
  const [demographic, setDemographic] = useState<Demographic>('gender')
  const [showMetrics, setShowMetrics] = useState(false)

  const currentData = DATASETS[dataset]
  const groups = currentData.groups[demographic]

  const metrics: GroupMetrics[] = useMemo(() => {
    return groups.map(g => {
      const accuracy = ((g.tp + g.tn) / g.total) * 100
      const tpr = g.tp / (g.tp + g.fn)
      const fpr = g.fp / (g.fp + g.tn)
      const ppv = g.tp / (g.tp + g.fp)

      return {
        group: g.group,
        total: g.total,
        positive: g.positive,
        truePositive: g.tp,
        falsePositive: g.fp,
        trueNegative: g.tn,
        falseNegative: g.fn,
        accuracy,
        tpr,
        fpr,
        ppv
      }
    })
  }, [groups])

  // Statistical Parity
  const positiveRates = metrics.map(m => ({
    group: m.group,
    rate: (m.positive / m.total) * 100
  }))
  const avgPositiveRate = positiveRates.reduce((sum, r) => sum + r.rate, 0) / positiveRates.length
  const maxDiff = Math.max(...positiveRates.map(r => Math.abs(r.rate - avgPositiveRate)))

  // Disparate Impact (80% rule)
  const minRate = Math.min(...positiveRates.map(r => r.rate))
  const maxRate = Math.max(...positiveRates.map(r => r.rate))
  const disparateImpact = (minRate / maxRate) * 100
  const passes80Rule = disparateImpact >= 80

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50 to-pink-50 dark:from-gray-900 dark:to-rose-950">
      <SimulatorNav />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-rose-500 to-pink-600 text-white rounded-xl">
              <AlertTriangle className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Bias Detector
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                AI 모델의 편향성을 탐지하고 시각화합니다
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                데이터셋 선택
              </label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value as Dataset)}
                className="w-full px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-rose-500 dark:text-white"
              >
                <option value="credit">신용 평가</option>
                <option value="hiring">채용 심사</option>
                <option value="healthcare">의료 진단</option>
                <option value="criminal">형사 사법</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                인구통계학적 속성
              </label>
              <select
                value={demographic}
                onChange={(e) => setDemographic(e.target.value as Demographic)}
                className="w-full px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-rose-500 dark:text-white"
              >
                <option value="gender">성별</option>
                <option value="race">인종</option>
                <option value="age">연령대</option>
              </select>
            </div>

            <div className="flex items-end">
              <button
                onClick={() => setShowMetrics(!showMetrics)}
                className="w-full px-6 py-2 bg-gradient-to-r from-rose-500 to-pink-600 text-white rounded-lg hover:from-rose-600 hover:to-pink-700 transition-all shadow-lg"
              >
                {showMetrics ? '메트릭 숨기기' : '상세 메트릭 보기'}
              </button>
            </div>
          </div>
        </div>

        {/* Statistical Parity */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <BarChart3 className="w-6 h-6 text-rose-600 dark:text-rose-400" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Statistical Parity (통계적 동등성)
            </h2>
          </div>

          <div className="space-y-4">
            {positiveRates.map((item, idx) => {
              const diff = item.rate - avgPositiveRate
              const isDifferent = Math.abs(diff) > 5
              return (
                <div key={idx}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-700 dark:text-gray-300">
                      {item.group}
                    </span>
                    <span className={`font-semibold ${isDifferent ? 'text-rose-600 dark:text-rose-400' : 'text-green-600 dark:text-green-400'}`}>
                      {item.rate.toFixed(1)}%
                      {isDifferent && (
                        <span className="text-sm ml-2">
                          ({diff > 0 ? '+' : ''}{diff.toFixed(1)}%)
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all ${isDifferent ? 'bg-gradient-to-r from-rose-500 to-pink-600' : 'bg-gradient-to-r from-green-500 to-emerald-600'}`}
                      style={{ width: `${item.rate}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-6 p-4 bg-rose-50 dark:bg-rose-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400 mt-0.5" />
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">
                  최대 편차: {maxDiff.toFixed(1)}%
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  {maxDiff > 10 ? '⚠️ 상당한 편향이 감지되었습니다' : '✓ 비교적 균등한 분포입니다'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Disparate Impact */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <TrendingUp className="w-6 h-6 text-rose-600 dark:text-rose-400" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Disparate Impact (차별적 영향)
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">최소 승인률</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {minRate.toFixed(1)}%
              </p>
            </div>

            <div className="p-4 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">최대 승인률</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {maxRate.toFixed(1)}%
              </p>
            </div>

            <div className={`p-4 rounded-lg ${passes80Rule ? 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20' : 'bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20'}`}>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Disparate Impact Ratio
              </p>
              <p className={`text-2xl font-bold ${passes80Rule ? 'text-green-600 dark:text-green-400' : 'text-rose-600 dark:text-rose-400'}`}>
                {disparateImpact.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                {passes80Rule ? '✓ 80% 규칙 통과' : '⚠️ 80% 규칙 미달'}
              </p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>80% 규칙:</strong> 미국 고용기회평등위원회(EEOC)의 기준으로,
              소수 집단의 선택 비율이 다수 집단의 80% 이상이어야 차별적이지 않다고 간주합니다.
            </p>
          </div>
        </div>

        {/* Fairness Metrics */}
        {showMetrics && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              Fairlearn-style Metrics
            </h2>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                      그룹
                    </th>
                    <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                      Accuracy
                    </th>
                    <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                      TPR
                    </th>
                    <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                      FPR
                    </th>
                    <th className="text-right py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                      PPV
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m, idx) => (
                    <tr key={idx} className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">
                        {m.group}
                      </td>
                      <td className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">
                        {m.accuracy.toFixed(1)}%
                      </td>
                      <td className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">
                        {(m.tpr * 100).toFixed(1)}%
                      </td>
                      <td className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">
                        {(m.fpr * 100).toFixed(1)}%
                      </td>
                      <td className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">
                        {(m.ppv * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <strong className="text-gray-900 dark:text-white">TPR (True Positive Rate):</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  실제 양성 중 올바르게 예측한 비율 (재현율/민감도)
                </p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <strong className="text-gray-900 dark:text-white">FPR (False Positive Rate):</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  실제 음성 중 잘못 양성으로 예측한 비율
                </p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <strong className="text-gray-900 dark:text-white">PPV (Positive Predictive Value):</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  양성으로 예측한 것 중 실제 양성인 비율 (정밀도)
                </p>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <strong className="text-gray-900 dark:text-white">Accuracy:</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  전체 예측 중 올바른 예측의 비율
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
