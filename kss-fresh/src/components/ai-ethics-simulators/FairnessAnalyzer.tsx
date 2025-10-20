'use client'

import { useState, useMemo } from 'react'
import SimulatorNav from './SimulatorNav'
import { Scale, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react'

type FairnessMetric = 'statistical-parity' | 'equal-opportunity' | 'predictive-parity'

interface ModelMetrics {
  name: string
  accuracy: number
  groups: {
    name: string
    tpr: number
    fpr: number
    ppv: number
    selectionRate: number
    count: number
  }[]
}

const BASELINE_MODEL: ModelMetrics = {
  name: 'Baseline',
  accuracy: 82.5,
  groups: [
    { name: 'Group A', tpr: 0.85, fpr: 0.15, ppv: 0.83, selectionRate: 0.68, count: 5000 },
    { name: 'Group B', tpr: 0.72, fpr: 0.22, ppv: 0.75, selectionRate: 0.52, count: 5000 }
  ]
}

export default function FairnessAnalyzer() {
  const [selectedMetric, setSelectedMetric] = useState<FairnessMetric>('statistical-parity')
  const [fairnessConstraint, setFairnessConstraint] = useState(0.8)
  const [showComparison, setShowComparison] = useState(false)

  // Mitigated model based on fairness constraint
  const mitigatedModel: ModelMetrics = useMemo(() => {
    const baselineA = BASELINE_MODEL.groups[0]
    const baselineB = BASELINE_MODEL.groups[1]

    let groupA, groupB

    switch (selectedMetric) {
      case 'statistical-parity':
        // Adjust selection rates to be closer
        const avgSelectionRate = (baselineA.selectionRate + baselineB.selectionRate) / 2
        const adjustmentA = (avgSelectionRate - baselineA.selectionRate) * (1 - fairnessConstraint)
        const adjustmentB = (avgSelectionRate - baselineB.selectionRate) * (1 - fairnessConstraint)

        groupA = {
          ...baselineA,
          selectionRate: baselineA.selectionRate + adjustmentA,
          tpr: baselineA.tpr - 0.03,
        }
        groupB = {
          ...baselineB,
          selectionRate: baselineB.selectionRate + adjustmentB,
          tpr: baselineB.tpr + 0.05,
        }
        break

      case 'equal-opportunity':
        // Adjust TPR to be closer
        const avgTPR = (baselineA.tpr + baselineB.tpr) / 2
        const tprAdjustmentA = (avgTPR - baselineA.tpr) * (1 - fairnessConstraint)
        const tprAdjustmentB = (avgTPR - baselineB.tpr) * (1 - fairnessConstraint)

        groupA = {
          ...baselineA,
          tpr: baselineA.tpr + tprAdjustmentA,
          selectionRate: baselineA.selectionRate - 0.02,
        }
        groupB = {
          ...baselineB,
          tpr: baselineB.tpr + tprAdjustmentB,
          selectionRate: baselineB.selectionRate + 0.04,
        }
        break

      case 'predictive-parity':
        // Adjust PPV to be closer
        const avgPPV = (baselineA.ppv + baselineB.ppv) / 2
        const ppvAdjustmentA = (avgPPV - baselineA.ppv) * (1 - fairnessConstraint)
        const ppvAdjustmentB = (avgPPV - baselineB.ppv) * (1 - fairnessConstraint)

        groupA = {
          ...baselineA,
          ppv: baselineA.ppv + ppvAdjustmentA,
          fpr: baselineA.fpr + 0.02,
        }
        groupB = {
          ...baselineB,
          ppv: baselineB.ppv + ppvAdjustmentB,
          fpr: baselineB.fpr - 0.03,
        }
        break
    }

    const newAccuracy = 82.5 - (1 - fairnessConstraint) * 3.5

    return {
      name: 'Mitigated',
      accuracy: newAccuracy,
      groups: [groupA, groupB]
    }
  }, [selectedMetric, fairnessConstraint])

  const fairnessScore = useMemo(() => {
    const groupA = mitigatedModel.groups[0]
    const groupB = mitigatedModel.groups[1]

    let score: number
    switch (selectedMetric) {
      case 'statistical-parity':
        const srRatio = Math.min(groupA.selectionRate, groupB.selectionRate) /
                       Math.max(groupA.selectionRate, groupB.selectionRate)
        score = srRatio * 100
        break
      case 'equal-opportunity':
        const tprRatio = Math.min(groupA.tpr, groupB.tpr) / Math.max(groupA.tpr, groupB.tpr)
        score = tprRatio * 100
        break
      case 'predictive-parity':
        const ppvRatio = Math.min(groupA.ppv, groupB.ppv) / Math.max(groupA.ppv, groupB.ppv)
        score = ppvRatio * 100
        break
    }

    return score
  }, [mitigatedModel, selectedMetric])

  const metricDescriptions = {
    'statistical-parity': {
      title: 'Statistical Parity (통계적 동등성)',
      description: '모든 그룹의 긍정적 결과 비율이 동일해야 함',
      formula: 'P(Ŷ=1|A=a) = P(Ŷ=1|A=b)',
      metric: 'Selection Rate'
    },
    'equal-opportunity': {
      title: 'Equal Opportunity (기회 균등)',
      description: '실제 긍정인 경우 모든 그룹의 예측이 동일해야 함',
      formula: 'P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)',
      metric: 'True Positive Rate'
    },
    'predictive-parity': {
      title: 'Predictive Parity (예측 동등성)',
      description: '긍정으로 예측된 경우 실제 긍정 비율이 동일해야 함',
      formula: 'P(Y=1|Ŷ=1,A=a) = P(Y=1|Ŷ=1,A=b)',
      metric: 'Positive Predictive Value'
    }
  }

  const currentDescription = metricDescriptions[selectedMetric]

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50 to-pink-50 dark:from-gray-900 dark:to-rose-950">
      <SimulatorNav />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-rose-500 to-pink-600 text-white rounded-xl">
              <Scale className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Fairness Analyzer
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                다양한 공정성 메트릭을 분석하고 모델을 비교합니다
              </p>
            </div>
          </div>

          {/* Metric Selection */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            {(Object.keys(metricDescriptions) as FairnessMetric[]).map((metric) => {
              const desc = metricDescriptions[metric]
              const isSelected = selectedMetric === metric
              return (
                <button
                  key={metric}
                  onClick={() => setSelectedMetric(metric)}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    isSelected
                      ? 'border-rose-500 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-rose-300 dark:hover:border-rose-700'
                  }`}
                >
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    {desc.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {desc.description}
                  </p>
                </button>
              )
            })}
          </div>

          {/* Current Metric Info */}
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>수식:</strong> <code className="bg-white dark:bg-gray-800 px-2 py-1 rounded">
                {currentDescription.formula}
              </code>
            </p>
            <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
              <strong>측정 지표:</strong> {currentDescription.metric}
            </p>
          </div>
        </div>

        {/* Fairness Constraint Slider */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Fairness Constraint 조정
          </h2>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-700 dark:text-gray-300">공정성 제약 강도</span>
              <span className="font-semibold text-rose-600 dark:text-rose-400">
                {(fairnessConstraint * 100).toFixed(0)}%
              </span>
            </div>

            <input
              type="range"
              min="0.5"
              max="1"
              step="0.05"
              value={fairnessConstraint}
              onChange={(e) => setFairnessConstraint(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />

            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
              <span>낮음 (정확도 우선)</span>
              <span>높음 (공정성 우선)</span>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">모델 정확도</p>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                {mitigatedModel.accuracy.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                기준선 대비: {(mitigatedModel.accuracy - BASELINE_MODEL.accuracy).toFixed(1)}%
              </p>
            </div>

            <div className={`p-4 rounded-lg ${
              fairnessScore >= 90
                ? 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20'
                : 'bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20'
            }`}>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">공정성 점수</p>
              <p className={`text-3xl font-bold ${
                fairnessScore >= 90
                  ? 'text-green-600 dark:text-green-400'
                  : 'text-yellow-600 dark:text-yellow-400'
              }`}>
                {fairnessScore.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {fairnessScore >= 90 ? '✓ 공정성 달성' : '⚠️ 개선 필요'}
              </p>
            </div>
          </div>
        </div>

        {/* Model Comparison */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              모델 비교
            </h2>
            <button
              onClick={() => setShowComparison(!showComparison)}
              className="px-4 py-2 bg-gradient-to-r from-rose-500 to-pink-600 text-white rounded-lg hover:from-rose-600 hover:to-pink-700 transition-all"
            >
              {showComparison ? '간단히 보기' : '상세히 보기'}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Baseline Model */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-gray-400"></div>
                Baseline Model
              </h3>

              {BASELINE_MODEL.groups.map((group, idx) => (
                <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                  <p className="font-semibold text-gray-900 dark:text-white mb-3">
                    {group.name}
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Selection Rate:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {(group.selectionRate * 100).toFixed(1)}%
                      </span>
                    </div>
                    {showComparison && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">TPR:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {(group.tpr * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">FPR:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {(group.fpr * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">PPV:</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {(group.ppv * 100).toFixed(1)}%
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Mitigated Model */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-rose-500 to-pink-600"></div>
                Mitigated Model
              </h3>

              {mitigatedModel.groups.map((group, idx) => (
                <div key={idx} className="p-4 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
                  <p className="font-semibold text-gray-900 dark:text-white mb-3">
                    {group.name}
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Selection Rate:</span>
                      <span className="font-medium text-rose-600 dark:text-rose-400">
                        {(group.selectionRate * 100).toFixed(1)}%
                      </span>
                    </div>
                    {showComparison && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">TPR:</span>
                          <span className="font-medium text-rose-600 dark:text-rose-400">
                            {(group.tpr * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">FPR:</span>
                          <span className="font-medium text-rose-600 dark:text-rose-400">
                            {(group.fpr * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600 dark:text-gray-400">PPV:</span>
                          <span className="font-medium text-rose-600 dark:text-rose-400">
                            {(group.ppv * 100).toFixed(1)}%
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Mitigation Strategies */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
          <div className="flex items-center gap-3 mb-6">
            <TrendingUp className="w-6 h-6 text-rose-600 dark:text-rose-400" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              완화 전략 제안
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Pre-processing
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    데이터 재샘플링, 레이블 재할당, 보호 속성 변환
                  </p>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    In-processing
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    공정성 제약 조건, 적대적 학습, 다중 목적 최적화
                  </p>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Post-processing
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    임계값 최적화, 예측 재조정, 거부 옵션
                  </p>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    Trade-off 고려
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    정확도와 공정성 간의 균형, 비즈니스 영향 평가
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
