'use client'

import { useState, useEffect, useRef } from 'react'
import { 
  Wand2, Brain, BarChart3, Target, Zap, GitBranch, 
  ChevronRight, Check, AlertCircle, Download, Info,
  TrendingUp, Filter, Layers, ArrowUpDown, Hash,
  Calendar, Type, Binary, PieChart, Shuffle
} from 'lucide-react'

interface Feature {
  name: string
  type: 'numeric' | 'categorical' | 'datetime' | 'text'
  importance?: number
  category?: string
  description?: string
  stats?: {
    mean?: number
    std?: number
    min?: number
    max?: number
    missing?: number
    unique?: number
  }
}

interface EngineeredFeature extends Feature {
  method: string
  originalFeatures: string[]
  formula?: string
}

interface Dataset {
  name: string
  description: string
  features: Feature[]
  targetVariable: string
  sampleRows: number
}

const SAMPLE_DATASETS: Dataset[] = [
  {
    name: '전자상거래 고객 데이터',
    description: '구매 행동 예측을 위한 고객 특성',
    targetVariable: 'purchase_amount',
    sampleRows: 10000,
    features: [
      { name: 'age', type: 'numeric', stats: { mean: 35, std: 12, min: 18, max: 75 } },
      { name: 'income', type: 'numeric', stats: { mean: 50000, std: 25000, min: 15000, max: 200000 } },
      { name: 'membership_days', type: 'numeric', stats: { mean: 365, std: 300, min: 1, max: 2000 } },
      { name: 'total_spent', type: 'numeric', stats: { mean: 1200, std: 800, min: 0, max: 10000 } },
      { name: 'num_purchases', type: 'numeric', stats: { mean: 15, std: 10, min: 0, max: 100 } },
      { name: 'category', type: 'categorical', stats: { unique: 5 } },
      { name: 'device_type', type: 'categorical', stats: { unique: 3 } },
      { name: 'last_purchase_date', type: 'datetime' },
      { name: 'registration_date', type: 'datetime' }
    ]
  },
  {
    name: '부동산 가격 데이터',
    description: '주택 가격 예측을 위한 특성',
    targetVariable: 'price',
    sampleRows: 5000,
    features: [
      { name: 'sqft', type: 'numeric', stats: { mean: 1500, std: 600, min: 500, max: 5000 } },
      { name: 'bedrooms', type: 'numeric', stats: { mean: 3, std: 1, min: 1, max: 6 } },
      { name: 'bathrooms', type: 'numeric', stats: { mean: 2, std: 0.5, min: 1, max: 4 } },
      { name: 'age', type: 'numeric', stats: { mean: 20, std: 15, min: 0, max: 100 } },
      { name: 'lot_size', type: 'numeric', stats: { mean: 7000, std: 3000, min: 2000, max: 20000 } },
      { name: 'neighborhood', type: 'categorical', stats: { unique: 20 } },
      { name: 'property_type', type: 'categorical', stats: { unique: 4 } },
      { name: 'built_year', type: 'datetime' }
    ]
  },
  {
    name: '신용카드 사기 탐지',
    description: '거래 패턴 기반 사기 탐지',
    targetVariable: 'is_fraud',
    sampleRows: 50000,
    features: [
      { name: 'amount', type: 'numeric', stats: { mean: 88, std: 250, min: 0.01, max: 25000 } },
      { name: 'merchant_risk_score', type: 'numeric', stats: { mean: 0.3, std: 0.2, min: 0, max: 1 } },
      { name: 'days_since_last_transaction', type: 'numeric', stats: { mean: 5, std: 10, min: 0, max: 365 } },
      { name: 'merchant_category', type: 'categorical', stats: { unique: 15 } },
      { name: 'transaction_type', type: 'categorical', stats: { unique: 3 } },
      { name: 'country', type: 'categorical', stats: { unique: 50 } },
      { name: 'transaction_time', type: 'datetime' },
      { name: 'card_issue_date', type: 'datetime' }
    ]
  }
]

const FEATURE_ENGINEERING_METHODS = [
  {
    category: '수치형 변환',
    icon: <Hash className="w-5 h-5" />,
    methods: [
      { name: '다항식 특성', description: 'x², x³, x₁×x₂ 등 다항식 조합' },
      { name: '로그 변환', description: 'log(x) - 왜도 감소' },
      { name: '제곱근 변환', description: '√x - 분포 정규화' },
      { name: '역수 변환', description: '1/x - 극단값 처리' },
      { name: '정규화', description: 'Min-Max, Z-score 스케일링' }
    ]
  },
  {
    category: '범주형 인코딩',
    icon: <Type className="w-5 h-5" />,
    methods: [
      { name: 'Target Encoding', description: '타겟 평균으로 인코딩' },
      { name: 'Frequency Encoding', description: '빈도수로 인코딩' },
      { name: 'Binary Encoding', description: '이진수 표현' },
      { name: 'Hashing', description: '해시 함수 활용' },
      { name: 'Embedding', description: '저차원 벡터 표현' }
    ]
  },
  {
    category: '시간 특성',
    icon: <Calendar className="w-5 h-5" />,
    methods: [
      { name: '날짜 분해', description: '년/월/일/요일 추출' },
      { name: '주기성', description: 'sin/cos 변환으로 주기 표현' },
      { name: '시간차', description: '이벤트 간 시간 간격' },
      { name: '이동 평균', description: '시간 윈도우 통계' },
      { name: '계절성', description: '계절 패턴 추출' }
    ]
  },
  {
    category: '상호작용',
    icon: <GitBranch className="w-5 h-5" />,
    methods: [
      { name: '곱셈 상호작용', description: 'x₁ × x₂' },
      { name: '나눗셈 비율', description: 'x₁ / x₂' },
      { name: '차이', description: 'x₁ - x₂' },
      { name: '조건부 특성', description: 'if-then 규칙 기반' },
      { name: '그룹 통계', description: '그룹별 평균/표준편차' }
    ]
  },
  {
    category: '집계 특성',
    icon: <PieChart className="w-5 h-5" />,
    methods: [
      { name: '카운트', description: '범주별 개수' },
      { name: '비율', description: '전체 대비 비율' },
      { name: '순위', description: '값의 순위 변환' },
      { name: '구간화', description: '연속값을 구간으로' },
      { name: '클러스터링', description: '유사 그룹 라벨' }
    ]
  }
]

export default function FeatureEngineeringLabPyCaret() {
  const [selectedDataset, setSelectedDataset] = useState<Dataset>(SAMPLE_DATASETS[0])
  const [currentStep, setCurrentStep] = useState<'data' | 'analyze' | 'engineer' | 'evaluate'>('data')
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
  const [engineeredFeatures, setEngineeredFeatures] = useState<EngineeredFeature[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [featureImportance, setFeatureImportance] = useState<Map<string, number>>(new Map())
  
  // 자동 피처 엔지니어링 시뮬레이션
  const autoEngineerFeatures = () => {
    setIsProcessing(true)
    
    setTimeout(() => {
      const newFeatures: EngineeredFeature[] = []
      
      // 수치형 특성 변환
      selectedDataset.features
        .filter(f => f.type === 'numeric' && selectedFeatures.includes(f.name))
        .forEach(feature => {
          // 로그 변환
          if (feature.stats?.min && feature.stats.min > 0) {
            newFeatures.push({
              name: `log_${feature.name}`,
              type: 'numeric',
              method: '로그 변환',
              originalFeatures: [feature.name],
              formula: `log(${feature.name})`,
              category: '수치형 변환',
              description: `${feature.name}의 로그 변환으로 왜도 감소`,
              importance: Math.random() * 0.5 + 0.3
            })
          }
          
          // 제곱 변환
          newFeatures.push({
            name: `${feature.name}_squared`,
            type: 'numeric',
            method: '다항식 특성',
            originalFeatures: [feature.name],
            formula: `${feature.name}²`,
            category: '수치형 변환',
            description: `${feature.name}의 제곱값`,
            importance: Math.random() * 0.4 + 0.2
          })
        })
      
      // 상호작용 특성
      const numericFeatures = selectedDataset.features
        .filter(f => f.type === 'numeric' && selectedFeatures.includes(f.name))
      
      for (let i = 0; i < numericFeatures.length - 1; i++) {
        for (let j = i + 1; j < numericFeatures.length; j++) {
          const f1 = numericFeatures[i]
          const f2 = numericFeatures[j]
          
          // 곱셈 상호작용
          newFeatures.push({
            name: `${f1.name}_x_${f2.name}`,
            type: 'numeric',
            method: '곱셈 상호작용',
            originalFeatures: [f1.name, f2.name],
            formula: `${f1.name} × ${f2.name}`,
            category: '상호작용',
            description: `${f1.name}와 ${f2.name}의 상호작용`,
            importance: Math.random() * 0.6 + 0.1
          })
          
          // 비율 (0으로 나누기 방지)
          if (f2.stats?.min && f2.stats.min > 0) {
            newFeatures.push({
              name: `${f1.name}_per_${f2.name}`,
              type: 'numeric',
              method: '나눗셈 비율',
              originalFeatures: [f1.name, f2.name],
              formula: `${f1.name} / ${f2.name}`,
              category: '상호작용',
              description: `${f1.name} 대비 ${f2.name} 비율`,
              importance: Math.random() * 0.5 + 0.2
            })
          }
        }
      }
      
      // 시간 특성
      selectedDataset.features
        .filter(f => f.type === 'datetime' && selectedFeatures.includes(f.name))
        .forEach(feature => {
          ['year', 'month', 'day', 'weekday', 'hour'].forEach(part => {
            newFeatures.push({
              name: `${feature.name}_${part}`,
              type: 'numeric',
              method: '날짜 분해',
              originalFeatures: [feature.name],
              category: '시간 특성',
              description: `${feature.name}에서 추출한 ${part}`,
              importance: Math.random() * 0.4 + 0.1
            })
          })
          
          // 주기성
          newFeatures.push({
            name: `${feature.name}_sin_day`,
            type: 'numeric',
            method: '주기성',
            originalFeatures: [feature.name],
            formula: 'sin(2π × day_of_year / 365)',
            category: '시간 특성',
            description: `${feature.name}의 연간 주기성 (sin)`,
            importance: Math.random() * 0.3 + 0.1
          })
        })
      
      // 범주형 인코딩
      selectedDataset.features
        .filter(f => f.type === 'categorical' && selectedFeatures.includes(f.name))
        .forEach(feature => {
          newFeatures.push({
            name: `${feature.name}_target_encoded`,
            type: 'numeric',
            method: 'Target Encoding',
            originalFeatures: [feature.name],
            category: '범주형 인코딩',
            description: `${feature.name}의 타겟 평균 인코딩`,
            importance: Math.random() * 0.7 + 0.2
          })
          
          newFeatures.push({
            name: `${feature.name}_frequency`,
            type: 'numeric',
            method: 'Frequency Encoding',
            originalFeatures: [feature.name],
            category: '범주형 인코딩',
            description: `${feature.name}의 빈도수 인코딩`,
            importance: Math.random() * 0.4 + 0.1
          })
        })
      
      // 중요도 정렬
      newFeatures.sort((a, b) => (b.importance || 0) - (a.importance || 0))
      
      setEngineeredFeatures(newFeatures)
      
      // 전체 특성 중요도 업데이트
      const importanceMap = new Map<string, number>()
      selectedFeatures.forEach(f => {
        importanceMap.set(f, Math.random() * 0.3 + 0.1)
      })
      newFeatures.forEach(f => {
        importanceMap.set(f.name, f.importance || 0)
      })
      setFeatureImportance(importanceMap)
      
      setIsProcessing(false)
      setCurrentStep('evaluate')
    }, 2000)
  }
  
  // 중요도 시각화
  const drawImportanceChart = (canvasRef: React.RefObject<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 400, 300)
    
    // 상위 10개 특성만 표시
    const sortedFeatures = Array.from(featureImportance.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
    
    if (sortedFeatures.length === 0) return
    
    const maxImportance = sortedFeatures[0][1]
    const barHeight = 25
    const barGap = 5
    
    sortedFeatures.forEach(([ name, importance], i) => {
      const y = i * (barHeight + barGap) + 20
      const width = (importance / maxImportance) * 300
      
      // 막대
      ctx.fillStyle = engineeredFeatures.find(f => f.name === name) 
        ? '#8b5cf6' // 엔지니어링된 특성
        : '#3b82f6' // 원본 특성
      ctx.fillRect(50, y, width, barHeight)
      
      // 라벨
      ctx.fillStyle = '#374151'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'right'
      ctx.fillText(name.substring(0, 15) + (name.length > 15 ? '...' : ''), 45, y + barHeight/2 + 4)
      
      // 중요도 값
      ctx.textAlign = 'left'
      ctx.fillText(importance.toFixed(3), width + 55, y + barHeight/2 + 4)
    })
  }
  
  const importanceCanvasRef = useRef<HTMLCanvasElement>(null)
  
  useEffect(() => {
    if (featureImportance.size > 0) {
      drawImportanceChart(importanceCanvasRef)
    }
  }, [featureImportance])
  
  // CSV 다운로드
  const downloadEngineeredFeatures = () => {
    let csv = 'Feature Name,Type,Method,Original Features,Formula,Category,Importance\n'
    
    engineeredFeatures.forEach(feature => {
      csv += `"${feature.name}","${feature.type}","${feature.method}",`
      csv += `"${feature.originalFeatures.join(', ')}",`
      csv += `"${feature.formula || ''}","${feature.category || ''}",`
      csv += `${feature.importance?.toFixed(3) || ''}\n`
    })
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `engineered_features_${selectedDataset.name.replace(/\s+/g, '_')}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <div className="mb-6">
          <h2 className="text-2xl font-bold mb-2">피처 엔지니어링 with PyCaret</h2>
          <p className="text-gray-600 dark:text-gray-400">
            자동으로 의미 있는 특성을 생성하고 중요도를 평가합니다
          </p>
        </div>
        
        {/* 진행 단계 */}
        <div className="flex items-center justify-between mb-8">
          {[
            { id: 'data', label: '데이터 선택', icon: <BarChart3 className="w-5 h-5" /> },
            { id: 'analyze', label: '특성 분석', icon: <Brain className="w-5 h-5" /> },
            { id: 'engineer', label: '특성 생성', icon: <Wand2 className="w-5 h-5" /> },
            { id: 'evaluate', label: '평가', icon: <Target className="w-5 h-5" /> }
          ].map((step, idx) => (
            <div key={step.id} className="flex items-center flex-1">
              <div className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                currentStep === step.id
                  ? 'bg-purple-500 text-white'
                  : currentStep > step.id
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}>
                {currentStep > step.id ? <Check className="w-5 h-5" /> : step.icon}
                <span className="font-medium">{step.label}</span>
              </div>
              {idx < 3 && (
                <ChevronRight className="w-5 h-5 mx-2 text-gray-400" />
              )}
            </div>
          ))}
        </div>
        
        {/* 단계별 컨텐츠 */}
        {currentStep === 'data' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">데이터셋 선택</h3>
            <div className="grid md:grid-cols-3 gap-4">
              {SAMPLE_DATASETS.map((dataset) => (
                <button
                  key={dataset.name}
                  onClick={() => {
                    setSelectedDataset(dataset)
                    setSelectedFeatures([])
                    setEngineeredFeatures([])
                  }}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    selectedDataset.name === dataset.name
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                >
                  <h4 className="font-semibold mb-1">{dataset.name}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    {dataset.description}
                  </p>
                  <div className="text-xs text-gray-500 space-y-1">
                    <div>특성: {dataset.features.length}개</div>
                    <div>샘플: {dataset.sampleRows.toLocaleString()}개</div>
                    <div>타겟: {dataset.targetVariable}</div>
                  </div>
                </button>
              ))}
            </div>
            
            <button
              onClick={() => setCurrentStep('analyze')}
              className="px-6 py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors"
            >
              다음 단계
            </button>
          </div>
        )}
        
        {currentStep === 'analyze' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">특성 분석 및 선택</h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">사용 가능한 특성</h4>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {selectedDataset.features.map((feature) => (
                    <label
                      key={feature.name}
                      className="flex items-center p-3 rounded-lg border cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                    >
                      <input
                        type="checkbox"
                        checked={selectedFeatures.includes(feature.name)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedFeatures([...selectedFeatures, feature.name])
                          } else {
                            setSelectedFeatures(selectedFeatures.filter(f => f !== feature.name))
                          }
                        }}
                        className="mr-3"
                      />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{feature.name}</span>
                          <span className={`text-xs px-2 py-1 rounded ${
                            feature.type === 'numeric' ? 'bg-blue-100 text-blue-700' :
                            feature.type === 'categorical' ? 'bg-green-100 text-green-700' :
                            feature.type === 'datetime' ? 'bg-purple-100 text-purple-700' :
                            'bg-gray-100 text-gray-700'
                          }`}>
                            {feature.type}
                          </span>
                        </div>
                        {feature.stats && (
                          <div className="text-xs text-gray-500 mt-1">
                            {feature.type === 'numeric' && (
                              <>평균: {feature.stats.mean}, 표준편차: {feature.stats.std}</>
                            )}
                            {feature.type === 'categorical' && (
                              <>고유값: {feature.stats.unique}개</>
                            )}
                          </div>
                        )}
                      </div>
                    </label>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-3">선택된 특성: {selectedFeatures.length}개</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  {selectedFeatures.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {selectedFeatures.map((feature) => (
                        <span
                          key={feature}
                          className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-sm"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-sm">특성을 선택해주세요</p>
                  )}
                </div>
                
                <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Info className="w-4 h-4" />
                    PyCaret 자동 분석
                  </h5>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 결측값 자동 처리</li>
                    <li>• 이상치 탐지 및 처리</li>
                    <li>• 데이터 타입 자동 추론</li>
                    <li>• 상관관계 분석</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="flex gap-4">
              <button
                onClick={() => setCurrentStep('data')}
                className="px-6 py-3 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors"
              >
                이전
              </button>
              <button
                onClick={() => setCurrentStep('engineer')}
                disabled={selectedFeatures.length === 0}
                className="px-6 py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                특성 엔지니어링 시작
              </button>
            </div>
          </div>
        )}
        
        {currentStep === 'engineer' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">자동 특성 생성</h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">PyCaret 엔지니어링 방법</h4>
                <div className="space-y-3">
                  {FEATURE_ENGINEERING_METHODS.map((category) => (
                    <div key={category.category} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        {category.icon}
                        <span className="font-medium">{category.category}</span>
                      </div>
                      <div className="space-y-1">
                        {category.methods.map((method) => (
                          <div key={method.name} className="text-sm">
                            <span className="font-medium">{method.name}</span>
                            <span className="text-gray-500 ml-2">- {method.description}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-3">선택된 특성 요약</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 mb-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">수치형:</span>
                      <span className="ml-2 font-semibold">
                        {selectedFeatures.filter(f => 
                          selectedDataset.features.find(df => df.name === f)?.type === 'numeric'
                        ).length}개
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">범주형:</span>
                      <span className="ml-2 font-semibold">
                        {selectedFeatures.filter(f => 
                          selectedDataset.features.find(df => df.name === f)?.type === 'categorical'
                        ).length}개
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">날짜/시간:</span>
                      <span className="ml-2 font-semibold">
                        {selectedFeatures.filter(f => 
                          selectedDataset.features.find(df => df.name === f)?.type === 'datetime'
                        ).length}개
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">총 특성:</span>
                      <span className="ml-2 font-semibold">{selectedFeatures.length}개</span>
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={autoEngineerFeatures}
                  disabled={isProcessing}
                  className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-medium hover:from-purple-600 hover:to-pink-600 transition-all disabled:opacity-50"
                >
                  {isProcessing ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
                      특성 생성 중...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      PyCaret으로 자동 특성 생성
                    </>
                  )}
                </button>
                
                <div className="mt-4 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <h5 className="font-medium mb-2">예상 결과</h5>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 다항식 특성: ~{Math.floor(selectedFeatures.length * (selectedFeatures.length - 1) / 2)}개</li>
                    <li>• 변환 특성: ~{selectedFeatures.filter(f => 
                      selectedDataset.features.find(df => df.name === f)?.type === 'numeric'
                    ).length * 3}개</li>
                    <li>• 시간 특성: ~{selectedFeatures.filter(f => 
                      selectedDataset.features.find(df => df.name === f)?.type === 'datetime'
                    ).length * 6}개</li>
                    <li>• 인코딩 특성: ~{selectedFeatures.filter(f => 
                      selectedDataset.features.find(df => df.name === f)?.type === 'categorical'
                    ).length * 2}개</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {currentStep === 'evaluate' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">특성 평가 결과</h3>
              <button
                onClick={downloadEngineeredFeatures}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                <Download className="w-4 h-4" />
                CSV 다운로드
              </button>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">생성된 특성 ({engineeredFeatures.length}개)</h4>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {engineeredFeatures.map((feature, idx) => (
                    <div
                      key={idx}
                      className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium">{feature.name}</span>
                        <span className="text-sm px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded">
                          {(feature.importance || 0).toFixed(3)}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        <div>{feature.description}</div>
                        {feature.formula && (
                          <div className="font-mono text-xs mt-1">{feature.formula}</div>
                        )}
                      </div>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded">
                          {feature.method}
                        </span>
                        <span className="text-xs text-gray-500">
                          원본: {feature.originalFeatures.join(', ')}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-3">특성 중요도</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <canvas
                    ref={importanceCanvasRef}
                    width={400}
                    height={300}
                    className="w-full"
                  />
                </div>
                
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                    <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                      {engineeredFeatures.filter(f => (f.importance || 0) > 0.5).length}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      고중요도 특성
                    </div>
                  </div>
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {((engineeredFeatures.length / selectedFeatures.length) * 100).toFixed(0)}%
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      특성 증가율
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    권장사항
                  </h5>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 중요도 0.5 이상인 특성 우선 사용</li>
                    <li>• 상관관계가 높은 특성은 제거 고려</li>
                    <li>• 교차 검증으로 과적합 방지</li>
                    <li>• 도메인 지식으로 특성 검증</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="flex gap-4">
              <button
                onClick={() => {
                  setCurrentStep('data')
                  setSelectedFeatures([])
                  setEngineeredFeatures([])
                  setFeatureImportance(new Map())
                }}
                className="px-6 py-3 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors"
              >
                새로운 데이터셋
              </button>
              <button
                onClick={() => setCurrentStep('engineer')}
                className="px-6 py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors"
              >
                다시 생성
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}