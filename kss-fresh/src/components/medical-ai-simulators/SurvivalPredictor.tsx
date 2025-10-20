'use client'

import React, { useState, useEffect, useRef } from 'react'

type RiskLevel = 'Low' | 'Medium' | 'High'
type ModelType = 'Cox' | 'Random Forest' | 'XGBoost'

interface PatientFeatures {
  age: number
  stage: number
  tumorSize: number
  lymphNodes: number
  grade: number
  biomarkerCA199: number
  biomarkerCEA: number
  performanceStatus: number
}

interface SurvivalResult {
  model: ModelType
  survivalProbability: number
  riskScore: number
  riskLevel: RiskLevel
  cIndex: number
  medianSurvival: number
}

interface FeatureImportance {
  feature: string
  importance: number
}

export default function SurvivalPredictor() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [features, setFeatures] = useState<PatientFeatures>({
    age: 58,
    stage: 2,
    tumorSize: 3.5,
    lymphNodes: 2,
    grade: 2,
    biomarkerCA199: 150,
    biomarkerCEA: 8,
    performanceStatus: 1
  })
  const [results, setResults] = useState<SurvivalResult[]>([])
  const [isPredicting, setIsPredicting] = useState(false)
  const [selectedModel, setSelectedModel] = useState<ModelType>('Cox')
  const [timeHorizon, setTimeHorizon] = useState(5)
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([])

  useEffect(() => {
    if (results.length > 0) {
      drawKaplanMeier()
    }
  }, [results, timeHorizon])

  const calculateRiskScore = (features: PatientFeatures, model: ModelType): number => {
    // Cox Proportional Hazards coefficients (simplified)
    const coxCoefficients = {
      age: 0.03,
      stage: 0.85,
      tumorSize: 0.25,
      lymphNodes: 0.15,
      grade: 0.40,
      biomarkerCA199: 0.002,
      biomarkerCEA: 0.05,
      performanceStatus: 0.30
    }

    // Random Forest importance weights
    const rfWeights = {
      age: 0.12,
      stage: 0.28,
      tumorSize: 0.18,
      lymphNodes: 0.15,
      grade: 0.10,
      biomarkerCA199: 0.08,
      biomarkerCEA: 0.05,
      performanceStatus: 0.04
    }

    // XGBoost weights
    const xgbWeights = {
      age: 0.10,
      stage: 0.32,
      tumorSize: 0.20,
      lymphNodes: 0.18,
      grade: 0.08,
      biomarkerCA199: 0.06,
      biomarkerCEA: 0.04,
      performanceStatus: 0.02
    }

    let riskScore = 0

    if (model === 'Cox') {
      riskScore =
        coxCoefficients.age * features.age +
        coxCoefficients.stage * features.stage +
        coxCoefficients.tumorSize * features.tumorSize +
        coxCoefficients.lymphNodes * features.lymphNodes +
        coxCoefficients.grade * features.grade +
        coxCoefficients.biomarkerCA199 * features.biomarkerCA199 +
        coxCoefficients.biomarkerCEA * features.biomarkerCEA +
        coxCoefficients.performanceStatus * features.performanceStatus
    } else if (model === 'Random Forest') {
      // Normalize and weight features
      riskScore =
        rfWeights.age * (features.age / 100) +
        rfWeights.stage * (features.stage / 4) +
        rfWeights.tumorSize * (features.tumorSize / 10) +
        rfWeights.lymphNodes * (features.lymphNodes / 20) +
        rfWeights.grade * (features.grade / 3) +
        rfWeights.biomarkerCA199 * Math.min(features.biomarkerCA199 / 500, 1) +
        rfWeights.biomarkerCEA * Math.min(features.biomarkerCEA / 50, 1) +
        rfWeights.performanceStatus * (features.performanceStatus / 4)

      riskScore *= 5 // Scale to similar range
    } else {
      // XGBoost
      riskScore =
        xgbWeights.age * (features.age / 100) +
        xgbWeights.stage * (features.stage / 4) +
        xgbWeights.tumorSize * (features.tumorSize / 10) +
        xgbWeights.lymphNodes * (features.lymphNodes / 20) +
        xgbWeights.grade * (features.grade / 3) +
        xgbWeights.biomarkerCA199 * Math.min(features.biomarkerCA199 / 500, 1) +
        xgbWeights.biomarkerCEA * Math.min(features.biomarkerCEA / 50, 1) +
        xgbWeights.performanceStatus * (features.performanceStatus / 4)

      riskScore *= 5.5 // Scale
    }

    return riskScore
  }

  const predictSurvival = () => {
    setIsPredicting(true)

    setTimeout(() => {
      const models: ModelType[] = ['Cox', 'Random Forest', 'XGBoost']
      const newResults: SurvivalResult[] = []

      models.forEach(model => {
        const riskScore = calculateRiskScore(features, model)

        // Calculate survival probability using exponential decay
        const baselineSurvival = 0.85
        const hazardRatio = Math.exp(riskScore - 3)
        const survivalProbability = Math.pow(baselineSurvival, hazardRatio * (timeHorizon / 5))

        // Determine risk level
        let riskLevel: RiskLevel
        if (riskScore < 2.5) riskLevel = 'Low'
        else if (riskScore < 4.5) riskLevel = 'Medium'
        else riskLevel = 'High'

        // Calculate C-index (concordance index)
        const cIndex = model === 'Cox' ? 0.78 + Math.random() * 0.05 :
                       model === 'Random Forest' ? 0.81 + Math.random() * 0.04 :
                       0.84 + Math.random() * 0.03

        // Median survival in months
        const medianSurvival = riskLevel === 'Low' ? 48 + Math.random() * 12 :
                               riskLevel === 'Medium' ? 24 + Math.random() * 12 :
                               12 + Math.random() * 8

        newResults.push({
          model,
          survivalProbability,
          riskScore,
          riskLevel,
          cIndex,
          medianSurvival
        })
      })

      setResults(newResults)
      calculateFeatureImportance()
      setIsPredicting(false)
    }, 2000)
  }

  const calculateFeatureImportance = () => {
    const importance: FeatureImportance[] = [
      { feature: 'Cancer Stage', importance: 0.32 },
      { feature: 'Tumor Size', importance: 0.20 },
      { feature: 'Lymph Nodes', importance: 0.18 },
      { feature: 'Age', importance: 0.12 },
      { feature: 'Tumor Grade', importance: 0.10 },
      { feature: 'CA19-9', importance: 0.06 },
      { feature: 'CEA', importance: 0.04 },
      { feature: 'Performance Status', importance: 0.03 }
    ]

    setFeatureImportance(importance)
  }

  const drawKaplanMeier = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#f8fafc'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = '#e2e8f0'
    ctx.lineWidth = 1

    for (let i = 0; i <= 10; i++) {
      const y = 40 + (height - 80) * (i / 10)
      ctx.beginPath()
      ctx.moveTo(60, y)
      ctx.lineTo(width - 20, y)
      ctx.stroke()

      // Y-axis labels
      ctx.fillStyle = '#64748b'
      ctx.font = '12px Inter'
      ctx.textAlign = 'right'
      ctx.fillText(`${100 - i * 10}%`, 50, y + 4)
    }

    for (let i = 0; i <= timeHorizon; i++) {
      const x = 60 + (width - 80) * (i / timeHorizon)
      ctx.beginPath()
      ctx.moveTo(x, 40)
      ctx.lineTo(x, height - 40)
      ctx.stroke()

      // X-axis labels
      ctx.fillStyle = '#64748b'
      ctx.textAlign = 'center'
      ctx.fillText(`${i}y`, x, height - 20)
    }

    // Draw axes
    ctx.strokeStyle = '#334155'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(60, 40)
    ctx.lineTo(60, height - 40)
    ctx.lineTo(width - 20, height - 40)
    ctx.stroke()

    // Draw survival curves for each model
    const colors = {
      'Cox': '#ef4444',
      'Random Forest': '#3b82f6',
      'XGBoost': '#10b981'
    }

    results.forEach(result => {
      ctx.strokeStyle = colors[result.model]
      ctx.lineWidth = 3
      ctx.beginPath()

      for (let t = 0; t <= timeHorizon * 12; t++) {
        const months = t
        const years = months / 12
        const x = 60 + (width - 80) * (years / timeHorizon)

        // Calculate survival at time t using exponential decay
        const baselineSurvival = 0.85
        const hazardRatio = Math.exp(result.riskScore - 3)
        const survivalAtT = Math.pow(baselineSurvival, hazardRatio * (months / 60))

        const y = 40 + (height - 80) * (1 - survivalAtT)

        if (t === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }

      ctx.stroke()

      // Draw label
      ctx.fillStyle = colors[result.model]
      ctx.font = 'bold 12px Inter'
      ctx.textAlign = 'left'
      const labelY = 40 + (height - 80) * (1 - result.survivalProbability)
      ctx.fillText(result.model, width - 150, labelY)
    })

    // Draw title
    ctx.fillStyle = '#1e293b'
    ctx.font = 'bold 16px Inter'
    ctx.textAlign = 'center'
    ctx.fillText('Kaplan-Meier Survival Curves', width / 2, 25)

    // Y-axis label
    ctx.save()
    ctx.translate(20, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillStyle = '#475569'
    ctx.font = 'bold 14px Inter'
    ctx.textAlign = 'center'
    ctx.fillText('Survival Probability', 0, 0)
    ctx.restore()

    // X-axis label
    ctx.fillStyle = '#475569'
    ctx.font = 'bold 14px Inter'
    ctx.textAlign = 'center'
    ctx.fillText('Time (years)', width / 2, height - 5)
  }

  const getRiskColor = (level: RiskLevel): string => {
    return level === 'Low' ? '#10b981' : level === 'Medium' ? '#f59e0b' : '#ef4444'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-500 via-pink-600 to-red-500 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6">
          <h1 className="text-4xl font-bold text-white mb-2">Survival Predictor</h1>
          <p className="text-white/80">Cox regression and ML ensemble for cancer survival prediction and risk stratification</p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Models</div>
              <div className="text-white font-bold">Cox / RF / XGBoost</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">C-Index</div>
              <div className="text-white font-bold">0.75 - 0.85</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Cohort</div>
              <div className="text-white font-bold">SEER Database</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Time Horizon</div>
              <div className="text-white font-bold">1-10 Years</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Patient Features</h3>

              <div className="space-y-4">
                <div>
                  <label className="text-white text-sm block mb-1">Age: {features.age} years</label>
                  <input
                    type="range"
                    min="30"
                    max="90"
                    value={features.age}
                    onChange={(e) => setFeatures({ ...features, age: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-white text-sm block mb-1">Cancer Stage: {features.stage}</label>
                  <input
                    type="range"
                    min="1"
                    max="4"
                    value={features.stage}
                    onChange={(e) => setFeatures({ ...features, stage: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-white text-sm block mb-1">Tumor Size: {features.tumorSize.toFixed(1)} cm</label>
                  <input
                    type="range"
                    min="0.5"
                    max="10"
                    step="0.5"
                    value={features.tumorSize}
                    onChange={(e) => setFeatures({ ...features, tumorSize: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-white text-sm block mb-1">Positive Lymph Nodes: {features.lymphNodes}</label>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    value={features.lymphNodes}
                    onChange={(e) => setFeatures({ ...features, lymphNodes: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-white text-sm block mb-1">Tumor Grade: {features.grade}</label>
                  <input
                    type="range"
                    min="1"
                    max="3"
                    value={features.grade}
                    onChange={(e) => setFeatures({ ...features, grade: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-white text-sm block mb-1">CA19-9: {features.biomarkerCA199} U/mL</label>
                  <input
                    type="range"
                    min="0"
                    max="500"
                    step="10"
                    value={features.biomarkerCA199}
                    onChange={(e) => setFeatures({ ...features, biomarkerCA199: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-white text-sm block mb-1">CEA: {features.biomarkerCEA} ng/mL</label>
                  <input
                    type="range"
                    min="0"
                    max="50"
                    step="1"
                    value={features.biomarkerCEA}
                    onChange={(e) => setFeatures({ ...features, biomarkerCEA: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-white text-sm block mb-1">Performance Status (ECOG): {features.performanceStatus}</label>
                  <input
                    type="range"
                    min="0"
                    max="4"
                    value={features.performanceStatus}
                    onChange={(e) => setFeatures({ ...features, performanceStatus: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>
              </div>

              <div className="mt-4">
                <label className="text-white text-sm block mb-1">Time Horizon: {timeHorizon} years</label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={timeHorizon}
                  onChange={(e) => setTimeHorizon(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <button
                onClick={predictSurvival}
                disabled={isPredicting}
                className={`w-full mt-6 py-3 rounded-lg font-bold transition-all ${
                  isPredicting ? 'bg-gray-400 cursor-not-allowed' : 'bg-white text-pink-600 hover:bg-gray-100'
                }`}
              >
                {isPredicting ? 'Predicting...' : 'Predict Survival'}
              </button>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Feature Importance</h3>
              <div className="space-y-2">
                {featureImportance.map(item => (
                  <div key={item.feature}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-white text-sm">{item.feature}</span>
                      <span className="text-white font-bold text-sm">{(item.importance * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-pink-500 to-red-500"
                        style={{ width: `${item.importance * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h2 className="text-2xl font-bold text-white mb-4">Kaplan-Meier Curves</h2>
              <canvas
                ref={canvasRef}
                width={700}
                height={450}
                className="w-full rounded-lg"
              />
            </div>

            {results.length > 0 && (
              <>
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                  <h3 className="text-xl font-bold text-white mb-4">Model Comparison</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {results.map(result => (
                      <div key={result.model} className="bg-white/10 rounded-lg p-4">
                        <div className="font-bold text-white mb-3">{result.model}</div>

                        <div className="space-y-2 text-sm">
                          <div>
                            <div className="text-white/60">Survival ({timeHorizon}y)</div>
                            <div className="text-2xl font-bold text-white">{(result.survivalProbability * 100).toFixed(1)}%</div>
                          </div>

                          <div>
                            <div className="text-white/60">Risk Level</div>
                            <div
                              className="inline-block px-2 py-1 rounded font-bold text-white text-sm"
                              style={{ backgroundColor: getRiskColor(result.riskLevel) }}
                            >
                              {result.riskLevel}
                            </div>
                          </div>

                          <div>
                            <div className="text-white/60">C-Index</div>
                            <div className="text-lg font-bold text-white">{result.cIndex.toFixed(3)}</div>
                          </div>

                          <div>
                            <div className="text-white/60">Median Survival</div>
                            <div className="text-lg font-bold text-white">{result.medianSurvival.toFixed(0)} mo</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                  <h3 className="text-xl font-bold text-white mb-3">Clinical Interpretation</h3>
                  <div className="space-y-2 text-sm text-white/80">
                    <p><strong className="text-white">C-Index:</strong> Measures model discrimination (0.5 = random, 1.0 = perfect)</p>
                    <p><strong className="text-white">Risk Score:</strong> Higher values indicate worse prognosis</p>
                    <p><strong className="text-white">Median Survival:</strong> Time at which 50% of patients survive</p>
                    <p><strong className="text-white">XGBoost:</strong> Often outperforms Cox in complex interactions</p>
                    <p className="mt-2 text-white/60 italic">Note: This is a simulation for educational purposes. Real clinical decisions require validated models and expert consultation.</p>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
