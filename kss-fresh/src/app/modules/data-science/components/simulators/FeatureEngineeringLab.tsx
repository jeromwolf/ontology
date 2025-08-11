'use client'

import { useState, useEffect, useRef } from 'react'
import { Settings, Wand2, Zap, BarChart3, Database, Filter, Shuffle, Download } from 'lucide-react'

interface Feature {
  name: string
  type: 'numeric' | 'categorical' | 'datetime' | 'text'
  values: any[]
  stats?: {
    mean?: number
    std?: number
    min?: number
    max?: number
    unique?: number
    missing?: number
  }
}

interface TransformationResult {
  name: string
  type: string
  preview: any[]
  importance?: number
}

export default function FeatureEngineeringLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [rawData, setRawData] = useState<any[]>([])
  const [features, setFeatures] = useState<Feature[]>([])
  const [selectedFeature, setSelectedFeature] = useState<string>('')
  const [transformationType, setTransformationType] = useState<string>('scaling')
  const [engineeredFeatures, setEngineeredFeatures] = useState<TransformationResult[]>([])
  const [correlationMatrix, setCorrelationMatrix] = useState<number[][]>([])
  const [showImportance, setShowImportance] = useState(false)
  
  // 샘플 데이터 생성
  const generateSampleData = () => {
    const sampleSize = 100
    const data: any[] = []
    
    for (let i = 0; i < sampleSize; i++) {
      data.push({
        age: Math.floor(Math.random() * 50) + 20,
        income: Math.floor(Math.random() * 100000) + 30000,
        experience: Math.floor(Math.random() * 20),
        education: ['고졸', '학사', '석사', '박사'][Math.floor(Math.random() * 4)],
        department: ['영업', '개발', '마케팅', '인사', '재무'][Math.floor(Math.random() * 5)],
        performance: Math.random() * 100,
        hire_date: new Date(2015 + Math.floor(Math.random() * 8), Math.floor(Math.random() * 12), 1),
        satisfaction: Math.floor(Math.random() * 5) + 1,
        is_promoted: Math.random() > 0.7 ? 1 : 0
      })
    }
    
    setRawData(data)
    extractFeatures(data)
  }
  
  // 피처 추출
  const extractFeatures = (data: any[]) => {
    if (data.length === 0) return
    
    const featureList: Feature[] = []
    const firstRow = data[0]
    
    Object.keys(firstRow).forEach(key => {
      const values = data.map(row => row[key])
      const type = detectFeatureType(values[0])
      
      const feature: Feature = {
        name: key,
        type,
        values,
        stats: calculateStats(values, type)
      }
      
      featureList.push(feature)
    })
    
    setFeatures(featureList)
    calculateCorrelations(featureList.filter(f => f.type === 'numeric'))
  }
  
  // 피처 타입 감지
  const detectFeatureType = (value: any): Feature['type'] => {
    if (typeof value === 'number') return 'numeric'
    if (value instanceof Date) return 'datetime'
    if (typeof value === 'string') {
      if (value.length > 50) return 'text'
      return 'categorical'
    }
    return 'categorical'
  }
  
  // 통계 계산
  const calculateStats = (values: any[], type: Feature['type']) => {
    const stats: Feature['stats'] = {
      missing: values.filter(v => v === null || v === undefined).length
    }
    
    if (type === 'numeric') {
      const numericValues = values.filter(v => typeof v === 'number')
      stats.mean = numericValues.reduce((sum, v) => sum + v, 0) / numericValues.length
      stats.std = Math.sqrt(
        numericValues.reduce((sum, v) => sum + Math.pow(v - stats.mean!, 2), 0) / numericValues.length
      )
      stats.min = Math.min(...numericValues)
      stats.max = Math.max(...numericValues)
    } else if (type === 'categorical') {
      stats.unique = new Set(values).size
    }
    
    return stats
  }
  
  // 상관관계 계산
  const calculateCorrelations = (numericFeatures: Feature[]) => {
    const n = numericFeatures.length
    const matrix: number[][] = Array(n).fill(0).map(() => Array(n).fill(0))
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          matrix[i][j] = 1
        } else {
          matrix[i][j] = calculateCorrelation(
            numericFeatures[i].values as number[],
            numericFeatures[j].values as number[]
          )
        }
      }
    }
    
    setCorrelationMatrix(matrix)
  }
  
  // 피어슨 상관계수
  const calculateCorrelation = (x: number[], y: number[]): number => {
    const n = x.length
    const sumX = x.reduce((a, b) => a + b, 0)
    const sumY = y.reduce((a, b) => a + b, 0)
    const sumXY = x.reduce((total, xi, i) => total + xi * y[i], 0)
    const sumX2 = x.reduce((total, xi) => total + xi * xi, 0)
    const sumY2 = y.reduce((total, yi) => total + yi * yi, 0)
    
    const num = n * sumXY - sumX * sumY
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
    
    return den === 0 ? 0 : num / den
  }
  
  // 피처 변환
  const transformFeature = () => {
    if (!selectedFeature) return
    
    const feature = features.find(f => f.name === selectedFeature)
    if (!feature) return
    
    let result: TransformationResult | null = null
    
    switch (transformationType) {
      case 'scaling':
        result = applyScaling(feature)
        break
      case 'encoding':
        result = applyEncoding(feature)
        break
      case 'binning':
        result = applyBinning(feature)
        break
      case 'polynomial':
        result = applyPolynomial(feature)
        break
      case 'log':
        result = applyLog(feature)
        break
      case 'interaction':
        result = applyInteraction(feature)
        break
    }
    
    if (result) {
      setEngineeredFeatures([...engineeredFeatures, result])
    }
  }
  
  // 스케일링
  const applyScaling = (feature: Feature): TransformationResult => {
    if (feature.type !== 'numeric') {
      return {
        name: `${feature.name}_scaled`,
        type: 'StandardScaler',
        preview: ['N/A - 숫자형 피처만 가능']
      }
    }
    
    const values = feature.values as number[]
    const mean = feature.stats?.mean || 0
    const std = feature.stats?.std || 1
    
    const scaled = values.map(v => (v - mean) / std)
    
    return {
      name: `${feature.name}_scaled`,
      type: 'StandardScaler',
      preview: scaled.slice(0, 5).map(v => v.toFixed(3)),
      importance: Math.random() * 0.8 + 0.2
    }
  }
  
  // 인코딩
  const applyEncoding = (feature: Feature): TransformationResult => {
    if (feature.type !== 'categorical') {
      return {
        name: `${feature.name}_encoded`,
        type: 'LabelEncoder',
        preview: ['N/A - 범주형 피처만 가능']
      }
    }
    
    const uniqueValues = Array.from(new Set(feature.values))
    const encoding = Object.fromEntries(uniqueValues.map((v, i) => [v, i]))
    const encoded = feature.values.map(v => encoding[v])
    
    return {
      name: `${feature.name}_encoded`,
      type: 'LabelEncoder',
      preview: encoded.slice(0, 5),
      importance: Math.random() * 0.6 + 0.1
    }
  }
  
  // 구간화
  const applyBinning = (feature: Feature): TransformationResult => {
    if (feature.type !== 'numeric') {
      return {
        name: `${feature.name}_binned`,
        type: 'Binning',
        preview: ['N/A - 숫자형 피처만 가능']
      }
    }
    
    const values = feature.values as number[]
    const min = feature.stats?.min || 0
    const max = feature.stats?.max || 100
    const binSize = (max - min) / 5
    
    const binned = values.map(v => {
      const bin = Math.floor((v - min) / binSize)
      return `Bin${Math.min(bin, 4) + 1}`
    })
    
    return {
      name: `${feature.name}_binned`,
      type: 'Equal-width Binning',
      preview: binned.slice(0, 5),
      importance: Math.random() * 0.5 + 0.3
    }
  }
  
  // 다항식 피처
  const applyPolynomial = (feature: Feature): TransformationResult => {
    if (feature.type !== 'numeric') {
      return {
        name: `${feature.name}_poly`,
        type: 'Polynomial',
        preview: ['N/A - 숫자형 피처만 가능']
      }
    }
    
    const values = feature.values as number[]
    const squared = values.map(v => v * v)
    
    return {
      name: `${feature.name}_squared`,
      type: 'Polynomial (degree=2)',
      preview: squared.slice(0, 5).map(v => v.toFixed(0)),
      importance: Math.random() * 0.7 + 0.2
    }
  }
  
  // 로그 변환
  const applyLog = (feature: Feature): TransformationResult => {
    if (feature.type !== 'numeric') {
      return {
        name: `${feature.name}_log`,
        type: 'Log Transform',
        preview: ['N/A - 숫자형 피처만 가능']
      }
    }
    
    const values = feature.values as number[]
    const logged = values.map(v => Math.log1p(Math.max(0, v))) // log(1+x) to handle 0
    
    return {
      name: `${feature.name}_log`,
      type: 'Log Transform',
      preview: logged.slice(0, 5).map(v => v.toFixed(3)),
      importance: Math.random() * 0.6 + 0.2
    }
  }
  
  // 상호작용 피처
  const applyInteraction = (feature: Feature): TransformationResult => {
    const numericFeatures = features.filter(f => f.type === 'numeric' && f.name !== feature.name)
    
    if (feature.type !== 'numeric' || numericFeatures.length === 0) {
      return {
        name: `${feature.name}_interaction`,
        type: 'Interaction',
        preview: ['N/A - 숫자형 피처 필요']
      }
    }
    
    // 랜덤하게 다른 피처 선택
    const otherFeature = numericFeatures[Math.floor(Math.random() * numericFeatures.length)]
    const values1 = feature.values as number[]
    const values2 = otherFeature.values as number[]
    
    const interaction = values1.map((v, i) => v * values2[i])
    
    return {
      name: `${feature.name}_x_${otherFeature.name}`,
      type: 'Feature Interaction',
      preview: interaction.slice(0, 5).map(v => v.toFixed(0)),
      importance: Math.random() * 0.9 + 0.1
    }
  }
  
  // 상관관계 히트맵 그리기
  const drawCorrelationHeatmap = () => {
    const canvas = canvasRef.current
    if (!canvas || correlationMatrix.length === 0) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const size = 400
    const numericFeatures = features.filter(f => f.type === 'numeric')
    const n = numericFeatures.length
    const cellSize = size / n
    
    ctx.clearRect(0, 0, size, size)
    
    // 히트맵 그리기
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const value = correlationMatrix[i][j]
        const intensity = Math.abs(value)
        
        if (value > 0) {
          ctx.fillStyle = `rgba(59, 130, 246, ${intensity})`
        } else {
          ctx.fillStyle = `rgba(239, 68, 68, ${intensity})`
        }
        
        ctx.fillRect(i * cellSize, j * cellSize, cellSize - 1, cellSize - 1)
        
        // 대각선에 피처 이름 표시
        if (i === j) {
          ctx.fillStyle = '#fff'
          ctx.font = '10px sans-serif'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(
            numericFeatures[i].name.substring(0, 8),
            i * cellSize + cellSize / 2,
            j * cellSize + cellSize / 2
          )
        }
      }
    }
    
    // 레이블
    ctx.fillStyle = '#666'
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    
    numericFeatures.forEach((feature, i) => {
      ctx.save()
      ctx.translate(i * cellSize + cellSize / 2, size + 10)
      ctx.rotate(-Math.PI / 4)
      ctx.fillText(feature.name, 0, 0)
      ctx.restore()
    })
  }
  
  // 피처 중요도 그래프
  const drawFeatureImportance = () => {
    const canvas = document.getElementById('importance-canvas') as HTMLCanvasElement
    if (!canvas || engineeredFeatures.length === 0) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const width = 600
    const height = 300
    const padding = 40
    
    ctx.clearRect(0, 0, width, height)
    
    // 중요도가 있는 피처만 필터링
    const featuresWithImportance = engineeredFeatures
      .filter(f => f.importance !== undefined)
      .sort((a, b) => (b.importance || 0) - (a.importance || 0))
      .slice(0, 10)
    
    if (featuresWithImportance.length === 0) return
    
    const barWidth = (width - 2 * padding) / featuresWithImportance.length
    const maxImportance = Math.max(...featuresWithImportance.map(f => f.importance || 0))
    
    // 막대 그래프
    featuresWithImportance.forEach((feature, i) => {
      const importance = feature.importance || 0
      const barHeight = (importance / maxImportance) * (height - 2 * padding)
      const x = padding + i * barWidth + barWidth * 0.1
      const y = height - padding - barHeight
      
      // 막대
      ctx.fillStyle = '#3b82f6'
      ctx.fillRect(x, y, barWidth * 0.8, barHeight)
      
      // 값 표시
      ctx.fillStyle = '#333'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(
        importance.toFixed(2),
        x + barWidth * 0.4,
        y - 5
      )
      
      // 피처 이름
      ctx.save()
      ctx.translate(x + barWidth * 0.4, height - padding + 5)
      ctx.rotate(-Math.PI / 4)
      ctx.textAlign = 'right'
      ctx.fillText(feature.name.substring(0, 15), 0, 0)
      ctx.restore()
    })
    
    // Y축
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padding, padding)
    ctx.lineTo(padding, height - padding)
    ctx.stroke()
  }
  
  useEffect(() => {
    generateSampleData()
  }, [])
  
  useEffect(() => {
    drawCorrelationHeatmap()
  }, [correlationMatrix])
  
  useEffect(() => {
    if (showImportance) {
      setTimeout(drawFeatureImportance, 100)
    }
  }, [showImportance, engineeredFeatures])
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">피처 엔지니어링 실험실</h2>
        
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 데이터 및 피처 정보 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 원본 피처 목록 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Database className="w-5 h-5" />
                원본 피처 ({features.length}개)
              </h3>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 max-h-64 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-2">피처명</th>
                      <th className="text-left py-2">타입</th>
                      <th className="text-left py-2">통계</th>
                      <th className="text-center py-2">선택</th>
                    </tr>
                  </thead>
                  <tbody>
                    {features.map((feature, index) => (
                      <tr key={index} className="border-b border-gray-100 dark:border-gray-800">
                        <td className="py-2 font-medium">{feature.name}</td>
                        <td className="py-2">
                          <span className={`px-2 py-1 rounded text-xs ${
                            feature.type === 'numeric' ? 'bg-blue-100 text-blue-700' :
                            feature.type === 'categorical' ? 'bg-green-100 text-green-700' :
                            feature.type === 'datetime' ? 'bg-purple-100 text-purple-700' :
                            'bg-gray-100 text-gray-700'
                          }`}>
                            {feature.type}
                          </span>
                        </td>
                        <td className="py-2 text-xs text-gray-600 dark:text-gray-400">
                          {feature.type === 'numeric' && feature.stats ? (
                            <span>μ={feature.stats.mean?.toFixed(1)}, σ={feature.stats.std?.toFixed(1)}</span>
                          ) : feature.type === 'categorical' && feature.stats ? (
                            <span>{feature.stats.unique} unique</span>
                          ) : '-'}
                        </td>
                        <td className="py-2 text-center">
                          <input
                            type="radio"
                            name="selectedFeature"
                            checked={selectedFeature === feature.name}
                            onChange={() => setSelectedFeature(feature.name)}
                            className="cursor-pointer"
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            
            {/* 상관관계 히트맵 */}
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  상관관계 히트맵
                </h3>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <canvas
                    ref={canvasRef}
                    width={400}
                    height={400}
                    className="w-full max-w-sm mx-auto"
                  />
                  <div className="mt-2 text-xs text-gray-500 text-center">
                    <span className="inline-block w-3 h-3 bg-blue-500 mr-1"></span>양의 상관관계
                    <span className="inline-block w-3 h-3 bg-red-500 ml-3 mr-1"></span>음의 상관관계
                  </div>
                </div>
              </div>
              
              {/* 변환된 피처 목록 */}
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  생성된 피처 ({engineeredFeatures.length}개)
                </h3>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 max-h-96 overflow-y-auto">
                  {engineeredFeatures.length > 0 ? (
                    <div className="space-y-3">
                      {engineeredFeatures.map((feature, index) => (
                        <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-3">
                          <div className="flex justify-between items-start mb-2">
                            <span className="font-medium text-sm">{feature.name}</span>
                            <span className="text-xs text-gray-500">{feature.type}</span>
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            미리보기: [{feature.preview.slice(0, 3).join(', ')}...]
                          </div>
                          {feature.importance && (
                            <div className="mt-2">
                              <div className="flex justify-between text-xs mb-1">
                                <span>중요도</span>
                                <span>{(feature.importance * 100).toFixed(0)}%</span>
                              </div>
                              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                                <div
                                  className="bg-blue-500 h-1.5 rounded-full"
                                  style={{ width: `${feature.importance * 100}%` }}
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-center py-8">
                      피처를 선택하고 변환을 적용하세요
                    </p>
                  )}
                </div>
              </div>
            </div>
            
            {/* 피처 중요도 차트 */}
            {showImportance && engineeredFeatures.some(f => f.importance) && (
              <div>
                <h3 className="text-lg font-semibold mb-3">피처 중요도 분석</h3>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <canvas
                    id="importance-canvas"
                    width={600}
                    height={300}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </div>
          
          {/* 변환 도구 패널 */}
          <div className="space-y-6">
            {/* 변환 유형 선택 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Wand2 className="w-5 h-5" />
                피처 변환
              </h3>
              
              <div className="space-y-3">
                <select
                  value={transformationType}
                  onChange={(e) => setTransformationType(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
                >
                  <optgroup label="스케일링">
                    <option value="scaling">표준화 (StandardScaler)</option>
                    <option value="minmax">정규화 (MinMaxScaler)</option>
                  </optgroup>
                  <optgroup label="인코딩">
                    <option value="encoding">레이블 인코딩</option>
                    <option value="onehot">원-핫 인코딩</option>
                  </optgroup>
                  <optgroup label="변환">
                    <option value="binning">구간화 (Binning)</option>
                    <option value="polynomial">다항식 피처</option>
                    <option value="log">로그 변환</option>
                    <option value="sqrt">제곱근 변환</option>
                  </optgroup>
                  <optgroup label="고급">
                    <option value="interaction">피처 상호작용</option>
                    <option value="pca">주성분 분석 (PCA)</option>
                  </optgroup>
                </select>
                
                <button
                  onClick={transformFeature}
                  disabled={!selectedFeature}
                  className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  변환 적용
                </button>
              </div>
            </div>
            
            {/* 변환 설명 */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Settings className="w-4 h-4" />
                변환 설명
              </h4>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {transformationType === 'scaling' && (
                  <p>평균 0, 표준편차 1로 정규화합니다. 거리 기반 알고리즘에 유용합니다.</p>
                )}
                {transformationType === 'encoding' && (
                  <p>범주형 변수를 숫자로 변환합니다. 머신러닝 모델 입력에 필요합니다.</p>
                )}
                {transformationType === 'binning' && (
                  <p>연속형 변수를 구간으로 나눕니다. 비선형 관계를 포착할 수 있습니다.</p>
                )}
                {transformationType === 'polynomial' && (
                  <p>변수의 거듭제곱을 생성합니다. 비선형 패턴을 학습할 수 있습니다.</p>
                )}
                {transformationType === 'log' && (
                  <p>로그 변환으로 치우친 분포를 정규분포에 가깝게 만듭니다.</p>
                )}
                {transformationType === 'interaction' && (
                  <p>두 피처의 곱을 생성합니다. 피처 간 상호작용을 모델링합니다.</p>
                )}
              </div>
            </div>
            
            {/* 도구 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Filter className="w-5 h-5" />
                도구
              </h3>
              
              <div className="space-y-2">
                <button
                  onClick={() => setShowImportance(!showImportance)}
                  className="w-full px-4 py-2 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors"
                >
                  피처 중요도 {showImportance ? '숨기기' : '보기'}
                </button>
                
                <button
                  onClick={generateSampleData}
                  className="w-full px-4 py-2 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors"
                >
                  <Shuffle className="w-4 h-4 inline mr-2" />
                  데이터 재생성
                </button>
                
                <button
                  onClick={() => {
                    const data = {
                      original: features.map(f => ({
                        name: f.name,
                        type: f.type,
                        stats: f.stats
                      })),
                      engineered: engineeredFeatures
                    }
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = `features-${Date.now()}.json`
                    a.click()
                    URL.revokeObjectURL(url)
                  }}
                  className="w-full px-4 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors"
                >
                  <Download className="w-4 h-4 inline mr-2" />
                  피처 내보내기
                </button>
              </div>
            </div>
            
            {/* 팁 */}
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
              <h4 className="font-semibold mb-2">💡 피처 엔지니어링 팁</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 도메인 지식을 활용하여 의미있는 피처를 생성하세요</li>
                <li>• 과적합을 피하기 위해 피처 수를 적절히 유지하세요</li>
                <li>• 타겟 변수와의 상관관계를 확인하세요</li>
                <li>• 교차 검증으로 피처의 유용성을 검증하세요</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}