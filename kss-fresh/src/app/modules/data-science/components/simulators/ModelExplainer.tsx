'use client'

import { useState, useEffect, useRef } from 'react'
import { Eye, BarChart3, Lightbulb, Target, Zap, Info, ChevronDown, Shield } from 'lucide-react'

interface Feature {
  name: string
  value: number
  importance: number
  contribution: number
  category: string
}

interface Prediction {
  value: number
  confidence: number
  baselineValue: number
}

interface ModelInfo {
  type: string
  features: Feature[]
  accuracy: number
  samples: number
}

export default function ModelExplainer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const waterfallRef = useRef<HTMLCanvasElement>(null)
  const [selectedModel, setSelectedModel] = useState<'credit' | 'churn' | 'house' | 'medical'>('credit')
  const [explainMethod, setExplainMethod] = useState<'shap' | 'lime' | 'permutation' | 'pdp'>('shap')
  const [selectedInstance, setSelectedInstance] = useState(0)
  const [showGlobalExplanation, setShowGlobalExplanation] = useState(false)
  
  // 샘플 모델들
  const models: { [key: string]: ModelInfo } = {
    credit: {
      type: '신용 대출 승인 모델',
      features: [
        { name: '연소득', value: 65000000, importance: 0.25, contribution: 0.12, category: '재무' },
        { name: '신용점수', value: 720, importance: 0.30, contribution: 0.18, category: '신용' },
        { name: '부채비율', value: 0.35, importance: 0.20, contribution: -0.08, category: '재무' },
        { name: '고용기간', value: 5, importance: 0.10, contribution: 0.05, category: '고용' },
        { name: '연체횟수', value: 1, importance: 0.15, contribution: -0.10, category: '신용' }
      ],
      accuracy: 0.89,
      samples: 10000
    },
    churn: {
      type: '고객 이탈 예측 모델',
      features: [
        { name: '사용기간', value: 24, importance: 0.22, contribution: -0.15, category: '활동' },
        { name: '월평균사용량', value: 450, importance: 0.28, contribution: -0.12, category: '활동' },
        { name: '고객지원문의', value: 8, importance: 0.18, contribution: 0.20, category: '서비스' },
        { name: '요금제등급', value: 3, importance: 0.12, contribution: 0.05, category: '상품' },
        { name: '결제지연횟수', value: 2, importance: 0.20, contribution: 0.15, category: '결제' }
      ],
      accuracy: 0.85,
      samples: 50000
    },
    house: {
      type: '주택 가격 예측 모델',
      features: [
        { name: '면적(평)', value: 32, importance: 0.35, contribution: 120000000, category: '규모' },
        { name: '방개수', value: 3, importance: 0.15, contribution: 30000000, category: '규모' },
        { name: '역까지거리', value: 500, importance: 0.20, contribution: -20000000, category: '위치' },
        { name: '건축연도', value: 2015, importance: 0.10, contribution: 15000000, category: '상태' },
        { name: '층수', value: 12, importance: 0.20, contribution: 25000000, category: '위치' }
      ],
      accuracy: 0.91,
      samples: 8000
    },
    medical: {
      type: '질병 위험도 예측 모델',
      features: [
        { name: '나이', value: 45, importance: 0.20, contribution: 0.08, category: '기본정보' },
        { name: 'BMI', value: 26.5, importance: 0.25, contribution: 0.12, category: '건강지표' },
        { name: '혈압', value: 130, importance: 0.30, contribution: 0.15, category: '건강지표' },
        { name: '운동빈도', value: 2, importance: 0.15, contribution: -0.10, category: '생활습관' },
        { name: '흡연여부', value: 0, importance: 0.10, contribution: -0.05, category: '생활습관' }
      ],
      accuracy: 0.87,
      samples: 20000
    }
  }
  
  const [prediction, setPrediction] = useState<Prediction>({
    value: 0.75,
    confidence: 0.82,
    baselineValue: 0.5
  })
  
  // 예측값 계산
  const calculatePrediction = () => {
    const model = models[selectedModel]
    const baseValue = selectedModel === 'house' ? 400000000 : 0.5
    
    let totalContribution = 0
    if (selectedModel === 'house') {
      totalContribution = model.features.reduce((sum, f) => sum + f.contribution, 0)
    } else {
      totalContribution = model.features.reduce((sum, f) => sum + f.contribution, 0)
    }
    
    const finalValue = selectedModel === 'house' 
      ? baseValue + totalContribution
      : Math.max(0, Math.min(1, baseValue + totalContribution))
    
    setPrediction({
      value: finalValue,
      confidence: 0.75 + Math.random() * 0.15,
      baselineValue: baseValue
    })
  }
  
  // SHAP 값 시각화
  const drawShapValues = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const width = 600
    const height = 400
    const padding = 60
    
    ctx.clearRect(0, 0, width, height)
    
    const model = models[selectedModel]
    const features = [...model.features].sort((a, b) => 
      Math.abs(b.contribution) - Math.abs(a.contribution)
    )
    
    const maxContrib = Math.max(...features.map(f => Math.abs(f.contribution)))
    const barHeight = (height - 2 * padding) / features.length - 10
    
    // 배경 그리드
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    
    // 중앙선
    const centerX = width / 2
    ctx.strokeStyle = '#6b7280'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(centerX, padding)
    ctx.lineTo(centerX, height - padding)
    ctx.stroke()
    
    // 피처별 기여도
    features.forEach((feature, index) => {
      const y = padding + index * (barHeight + 10) + barHeight / 2
      const barWidth = (Math.abs(feature.contribution) / maxContrib) * (width / 2 - padding - 20)
      
      // 막대
      ctx.fillStyle = feature.contribution > 0 ? '#3b82f6' : '#ef4444'
      if (feature.contribution > 0) {
        ctx.fillRect(centerX, y - barHeight / 2, barWidth, barHeight)
      } else {
        ctx.fillRect(centerX - barWidth, y - barHeight / 2, barWidth, barHeight)
      }
      
      // 피처명
      ctx.fillStyle = '#374151'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      ctx.fillText(feature.name, padding - 10, y)
      
      // 기여도 값
      ctx.textAlign = 'left'
      const contribText = selectedModel === 'house' 
        ? `${(feature.contribution / 100000000).toFixed(1)}억`
        : feature.contribution.toFixed(3)
      
      if (feature.contribution > 0) {
        ctx.fillText(contribText, centerX + barWidth + 5, y)
      } else {
        ctx.textAlign = 'right'
        ctx.fillText(contribText, centerX - barWidth - 5, y)
      }
    })
    
    // 제목
    ctx.fillStyle = '#111827'
    ctx.font = 'bold 14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('SHAP Values - 피처별 기여도', width / 2, 30)
  }
  
  // Waterfall 차트
  const drawWaterfallChart = () => {
    const canvas = waterfallRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const width = 600
    const height = 300
    const padding = 40
    
    ctx.clearRect(0, 0, width, height)
    
    const model = models[selectedModel]
    const features = [...model.features].sort((a, b) => 
      Math.abs(b.contribution) - Math.abs(a.contribution)
    )
    
    // 누적 계산
    let cumulative = prediction.baselineValue
    const steps = [{
      name: '기준값',
      value: prediction.baselineValue,
      cumulative: cumulative,
      contribution: 0
    }]
    
    features.forEach(feature => {
      if (selectedModel === 'house') {
        cumulative += feature.contribution
        steps.push({
          name: feature.name,
          value: feature.contribution,
          cumulative: cumulative,
          contribution: feature.contribution
        })
      } else {
        const oldCumulative = cumulative
        cumulative = Math.max(0, Math.min(1, cumulative + feature.contribution))
        steps.push({
          name: feature.name,
          value: feature.contribution,
          cumulative: cumulative,
          contribution: cumulative - oldCumulative
        })
      }
    })
    
    const barWidth = (width - 2 * padding) / (steps.length + 1) - 10
    const maxValue = selectedModel === 'house' 
      ? Math.max(...steps.map(s => s.cumulative))
      : 1
    const minValue = selectedModel === 'house'
      ? Math.min(...steps.map(s => s.cumulative))
      : 0
    
    // Y축 스케일
    const yScale = (value: number) => {
      return height - padding - ((value - minValue) / (maxValue - minValue)) * (height - 2 * padding)
    }
    
    // 막대 그리기
    steps.forEach((step, index) => {
      const x = padding + index * (barWidth + 10)
      
      if (index === 0) {
        // 기준값
        ctx.fillStyle = '#6b7280'
        const barHeight = Math.abs(yScale(0) - yScale(step.value))
        ctx.fillRect(x, yScale(step.value), barWidth, barHeight)
      } else {
        // 변화량
        const prevCumulative = steps[index - 1].cumulative
        const y1 = yScale(prevCumulative)
        const y2 = yScale(step.cumulative)
        
        ctx.fillStyle = step.contribution > 0 ? '#10b981' : '#ef4444'
        ctx.fillRect(x, Math.min(y1, y2), barWidth, Math.abs(y2 - y1))
        
        // 연결선
        if (index < steps.length - 1) {
          ctx.strokeStyle = '#9ca3af'
          ctx.lineWidth = 1
          ctx.setLineDash([2, 2])
          ctx.beginPath()
          ctx.moveTo(x + barWidth, y2)
          ctx.lineTo(x + barWidth + 10, y2)
          ctx.stroke()
          ctx.setLineDash([])
        }
      }
      
      // 레이블
      ctx.fillStyle = '#374151'
      ctx.font = '10px sans-serif'
      ctx.save()
      ctx.translate(x + barWidth / 2, height - padding + 5)
      ctx.rotate(-Math.PI / 4)
      ctx.textAlign = 'right'
      ctx.fillText(step.name, 0, 0)
      ctx.restore()
    })
    
    // 최종값
    const finalX = padding + steps.length * (barWidth + 10)
    const finalY = yScale(prediction.value)
    ctx.fillStyle = '#7c3aed'
    ctx.fillRect(finalX, finalY, barWidth, yScale(0) - finalY)
    
    ctx.fillStyle = '#374151'
    ctx.font = '10px sans-serif'
    ctx.save()
    ctx.translate(finalX + barWidth / 2, height - padding + 5)
    ctx.rotate(-Math.PI / 4)
    ctx.textAlign = 'right'
    ctx.fillText('최종예측', 0, 0)
    ctx.restore()
  }
  
  // Permutation Importance
  const drawPermutationImportance = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 600, 400)
    
    const model = models[selectedModel]
    const features = [...model.features].sort((a, b) => b.importance - a.importance)
    
    const barWidth = 400
    const barHeight = 30
    const startY = 80
    
    features.forEach((feature, index) => {
      const y = startY + index * (barHeight + 15)
      
      // 배경
      ctx.fillStyle = '#f3f4f6'
      ctx.fillRect(100, y, barWidth, barHeight)
      
      // 중요도 막대
      ctx.fillStyle = '#8b5cf6'
      ctx.fillRect(100, y, barWidth * feature.importance, barHeight)
      
      // 피처명
      ctx.fillStyle = '#374151'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      ctx.fillText(feature.name, 90, y + barHeight / 2)
      
      // 중요도 값
      ctx.textAlign = 'left'
      ctx.fillText(`${(feature.importance * 100).toFixed(1)}%`, 510, y + barHeight / 2)
      
      // 에러바 (시뮬레이션)
      const errorMargin = feature.importance * 0.1
      ctx.strokeStyle = '#374151'
      ctx.lineWidth = 2
      const errorX = 100 + barWidth * feature.importance
      ctx.beginPath()
      ctx.moveTo(errorX - barWidth * errorMargin, y + barHeight / 2)
      ctx.lineTo(errorX + barWidth * errorMargin, y + barHeight / 2)
      ctx.stroke()
      
      // 에러바 끝
      ctx.beginPath()
      ctx.moveTo(errorX - barWidth * errorMargin, y + 5)
      ctx.lineTo(errorX - barWidth * errorMargin, y + barHeight - 5)
      ctx.moveTo(errorX + barWidth * errorMargin, y + 5)
      ctx.lineTo(errorX + barWidth * errorMargin, y + barHeight - 5)
      ctx.stroke()
    })
    
    // 제목
    ctx.fillStyle = '#111827'
    ctx.font = 'bold 14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('Permutation Feature Importance', 300, 30)
    ctx.font = '12px sans-serif'
    ctx.fillStyle = '#6b7280'
    ctx.fillText('모델 성능에 대한 각 피처의 중요도', 300, 50)
  }
  
  // Partial Dependence Plot
  const drawPartialDependence = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 600, 400)
    
    const model = models[selectedModel]
    const feature = model.features[0] // 첫 번째 피처에 대한 PDP
    
    // 시뮬레이션 데이터 생성
    const points = 50
    const data: { x: number; y: number }[] = []
    
    for (let i = 0; i < points; i++) {
      const x = i / (points - 1)
      // 비선형 관계 시뮬레이션
      const y = 0.5 + 0.3 * Math.sin(x * Math.PI * 2) + 0.2 * x
      data.push({ x, y })
    }
    
    const padding = 60
    const width = 600
    const height = 400
    const plotWidth = width - 2 * padding
    const plotHeight = height - 2 * padding
    
    // 축
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    ctx.strokeRect(padding, padding, plotWidth, plotHeight)
    
    // 그리드
    for (let i = 0; i <= 5; i++) {
      const x = padding + (i / 5) * plotWidth
      const y = padding + (i / 5) * plotHeight
      
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, height - padding)
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }
    
    // PDP 곡선
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 3
    ctx.beginPath()
    
    data.forEach((point, index) => {
      const x = padding + point.x * plotWidth
      const y = height - padding - point.y * plotHeight
      
      if (index === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
    
    // 신뢰구간
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
    ctx.beginPath()
    data.forEach((point, index) => {
      const x = padding + point.x * plotWidth
      const yUpper = height - padding - (point.y + 0.1) * plotHeight
      
      if (index === 0) ctx.moveTo(x, yUpper)
      else ctx.lineTo(x, yUpper)
    })
    
    for (let i = data.length - 1; i >= 0; i--) {
      const point = data[i]
      const x = padding + point.x * plotWidth
      const yLower = height - padding - (point.y - 0.1) * plotHeight
      ctx.lineTo(x, yLower)
    }
    ctx.closePath()
    ctx.fill()
    
    // 현재값 표시
    const currentX = feature.value / 100000000 // 정규화
    const currentIndex = Math.floor(currentX * (points - 1))
    const currentY = data[Math.min(currentIndex, points - 1)].y
    
    ctx.fillStyle = '#ef4444'
    ctx.beginPath()
    ctx.arc(
      padding + currentX * plotWidth,
      height - padding - currentY * plotHeight,
      5,
      0,
      Math.PI * 2
    )
    ctx.fill()
    
    // 라벨
    ctx.fillStyle = '#374151'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText(feature.name, width / 2, height - 20)
    
    ctx.save()
    ctx.translate(20, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('예측값', 0, 0)
    ctx.restore()
    
    // 제목
    ctx.fillStyle = '#111827'
    ctx.font = 'bold 14px sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText(`Partial Dependence Plot - ${feature.name}`, width / 2, 20)
  }
  
  useEffect(() => {
    calculatePrediction()
  }, [selectedModel, selectedInstance])
  
  useEffect(() => {
    switch (explainMethod) {
      case 'shap':
        drawShapValues()
        break
      case 'permutation':
        drawPermutationImportance()
        break
      case 'pdp':
        drawPartialDependence()
        break
    }
    
    if (explainMethod === 'shap' || explainMethod === 'lime') {
      drawWaterfallChart()
    }
  }, [selectedModel, explainMethod, prediction])
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">모델 설명 도구 (Model Explainer)</h2>
        
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 메인 영역 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 모델 선택 및 예측 결과 */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold mb-2">예측 결과</h3>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value as any)}
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
                  >
                    <option value="credit">신용 대출 승인 모델</option>
                    <option value="churn">고객 이탈 예측 모델</option>
                    <option value="house">주택 가격 예측 모델</option>
                    <option value="medical">질병 위험도 예측 모델</option>
                  </select>
                </div>
                
                <div className="ml-6 text-center">
                  <div className="text-3xl font-bold text-blue-600">
                    {selectedModel === 'house' 
                      ? `${(prediction.value / 100000000).toFixed(1)}억원`
                      : selectedModel === 'credit'
                      ? `${(prediction.value * 100).toFixed(0)}% 승인`
                      : selectedModel === 'churn'
                      ? `${(prediction.value * 100).toFixed(0)}% 이탈`
                      : `${(prediction.value * 100).toFixed(0)}% 위험`
                    }
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    신뢰도: {(prediction.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
              
              {/* 샘플 인스턴스 선택 */}
              <div className="mt-4">
                <label className="text-sm font-medium">테스트 샘플 선택</label>
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="range"
                    min="0"
                    max="9"
                    value={selectedInstance}
                    onChange={(e) => setSelectedInstance(parseInt(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    샘플 #{selectedInstance + 1}
                  </span>
                </div>
              </div>
            </div>
            
            {/* 설명 방법 선택 */}
            <div className="flex gap-2">
              {[
                { id: 'shap', name: 'SHAP', icon: <BarChart3 className="w-4 h-4" /> },
                { id: 'lime', name: 'LIME', icon: <Target className="w-4 h-4" /> },
                { id: 'permutation', name: 'Permutation', icon: <Zap className="w-4 h-4" /> },
                { id: 'pdp', name: 'PDP', icon: <Eye className="w-4 h-4" /> }
              ].map(method => (
                <button
                  key={method.id}
                  onClick={() => setExplainMethod(method.id as any)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                    explainMethod === method.id
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {method.icon}
                  {method.name}
                </button>
              ))}
            </div>
            
            {/* 메인 시각화 */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <canvas
                ref={canvasRef}
                width={600}
                height={400}
                className="w-full border border-gray-300 dark:border-gray-600 rounded"
              />
            </div>
            
            {/* Waterfall 차트 (SHAP/LIME) */}
            {(explainMethod === 'shap' || explainMethod === 'lime') && (
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <h3 className="font-semibold mb-3">예측값 구성 (Waterfall Chart)</h3>
                <canvas
                  ref={waterfallRef}
                  width={600}
                  height={300}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded"
                />
              </div>
            )}
          </div>
          
          {/* 사이드바 */}
          <div className="space-y-6">
            {/* 피처 정보 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Info className="w-5 h-5" />
                입력 피처
              </h3>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 space-y-3">
                {models[selectedModel].features.map((feature, index) => (
                  <div key={index} className="border-b border-gray-200 dark:border-gray-700 pb-2 last:border-0">
                    <div className="flex justify-between items-start">
                      <span className="text-sm font-medium">{feature.name}</span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {selectedModel === 'house' && feature.name === '면적(평)' ? `${feature.value}평` :
                         selectedModel === 'house' && feature.name === '역까지거리' ? `${feature.value}m` :
                         selectedModel === 'house' && feature.name === '건축연도' ? feature.value :
                         selectedModel === 'credit' && feature.name === '연소득' ? `${(feature.value / 10000).toFixed(0)}만원` :
                         selectedModel === 'credit' && feature.name === '부채비율' ? `${(feature.value * 100).toFixed(0)}%` :
                         feature.value}
                      </span>
                    </div>
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${feature.importance * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* 모델 정보 */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                모델 정보
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">정확도:</span>
                  <span className="font-medium">{(models[selectedModel].accuracy * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">학습 샘플:</span>
                  <span className="font-medium">{models[selectedModel].samples.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">모델 타입:</span>
                  <span className="font-medium">XGBoost</span>
                </div>
              </div>
            </div>
            
            {/* 설명 방법 설명 */}
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Lightbulb className="w-4 h-4" />
                {explainMethod.toUpperCase()} 설명
              </h4>
              <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
                {explainMethod === 'shap' && (
                  <>
                    <p>SHAP(SHapley Additive exPlanations)는 게임 이론을 기반으로 각 피처의 기여도를 계산합니다.</p>
                    <p>• 파란색: 긍정적 기여</p>
                    <p>• 빨간색: 부정적 기여</p>
                  </>
                )}
                {explainMethod === 'lime' && (
                  <>
                    <p>LIME은 복잡한 모델을 국소적으로 해석 가능한 모델로 근사합니다.</p>
                    <p>• 개별 예측에 대한 설명 제공</p>
                    <p>• 주변 데이터를 샘플링하여 분석</p>
                  </>
                )}
                {explainMethod === 'permutation' && (
                  <>
                    <p>Permutation Importance는 피처 값을 무작위로 섞어 모델 성능 변화를 측정합니다.</p>
                    <p>• 전역적 피처 중요도</p>
                    <p>• 에러바는 불확실성 표시</p>
                  </>
                )}
                {explainMethod === 'pdp' && (
                  <>
                    <p>PDP는 특정 피처가 예측에 미치는 평균적인 영향을 보여줍니다.</p>
                    <p>• 비선형 관계 파악 가능</p>
                    <p>• 음영은 신뢰구간</p>
                  </>
                )}
              </div>
            </div>
            
            {/* 글로벌 설명 토글 */}
            <button
              onClick={() => setShowGlobalExplanation(!showGlobalExplanation)}
              className="w-full px-4 py-2 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors flex items-center justify-center gap-2"
            >
              {showGlobalExplanation ? '로컬 설명 보기' : '글로벌 설명 보기'}
              <ChevronDown className={`w-4 h-4 transition-transform ${showGlobalExplanation ? 'rotate-180' : ''}`} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}