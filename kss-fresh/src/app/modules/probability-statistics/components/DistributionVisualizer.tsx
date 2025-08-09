'use client'

import { useState, useEffect, useRef } from 'react'
import { BarChart3, RefreshCw, Download } from 'lucide-react'

interface Distribution {
  name: string
  type: 'continuous' | 'discrete'
  params: { [key: string]: number }
  formula: string
  description: string
}

export default function DistributionVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedDist, setSelectedDist] = useState('normal')
  const [params, setParams] = useState<{ [key: string]: number }>({
    mean: 0,
    std: 1,
    n: 10,
    p: 0.5,
    lambda: 3,
    alpha: 2,
    beta: 2
  })
  const [sampleSize, setSampleSize] = useState(1000)
  const [showPDF, setShowPDF] = useState(true)
  const [showCDF, setShowCDF] = useState(false)
  const [samples, setSamples] = useState<number[]>([])

  const distributions: { [key: string]: Distribution } = {
    normal: {
      name: '정규 분포',
      type: 'continuous',
      params: { mean: 0, std: 1 },
      formula: 'f(x) = (1/σ√(2π)) × e^(-½((x-μ)/σ)²)',
      description: '가장 중요한 연속 확률 분포'
    },
    binomial: {
      name: '이항 분포',
      type: 'discrete',
      params: { n: 10, p: 0.5 },
      formula: 'P(X=k) = C(n,k) × p^k × (1-p)^(n-k)',
      description: 'n번의 베르누이 시행에서 성공 횟수'
    },
    poisson: {
      name: '포아송 분포',
      type: 'discrete',
      params: { lambda: 3 },
      formula: 'P(X=k) = (λ^k × e^(-λ)) / k!',
      description: '단위 시간당 발생 횟수'
    },
    exponential: {
      name: '지수 분포',
      type: 'continuous',
      params: { lambda: 1 },
      formula: 'f(x) = λe^(-λx)',
      description: '사건 발생 시간 간격'
    },
    uniform: {
      name: '균등 분포',
      type: 'continuous',
      params: { a: 0, b: 1 },
      formula: 'f(x) = 1/(b-a) for a ≤ x ≤ b',
      description: '모든 값이 동일한 확률'
    },
    beta: {
      name: '베타 분포',
      type: 'continuous',
      params: { alpha: 2, beta: 2 },
      formula: 'f(x) = x^(α-1) × (1-x)^(β-1) / B(α,β)',
      description: '0과 1 사이의 확률 모델링'
    }
  }

  // 샘플 생성
  const generateSamples = () => {
    let newSamples: number[] = []
    
    switch (selectedDist) {
      case 'normal':
        // Box-Muller 변환
        for (let i = 0; i < sampleSize; i++) {
          const u1 = Math.random()
          const u2 = Math.random()
          const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
          newSamples.push(z0 * params.std + params.mean)
        }
        break
        
      case 'binomial':
        for (let i = 0; i < sampleSize; i++) {
          let successes = 0
          for (let j = 0; j < params.n; j++) {
            if (Math.random() < params.p) successes++
          }
          newSamples.push(successes)
        }
        break
        
      case 'poisson':
        for (let i = 0; i < sampleSize; i++) {
          let k = 0
          let p = 1
          const L = Math.exp(-params.lambda)
          do {
            k++
            p *= Math.random()
          } while (p > L)
          newSamples.push(k - 1)
        }
        break
        
      case 'exponential':
        for (let i = 0; i < sampleSize; i++) {
          newSamples.push(-Math.log(1 - Math.random()) / params.lambda)
        }
        break
        
      case 'uniform':
        for (let i = 0; i < sampleSize; i++) {
          newSamples.push(params.a + Math.random() * (params.b - params.a))
        }
        break
        
      case 'beta':
        // 간단한 근사 (실제로는 더 복잡한 알고리즘 필요)
        for (let i = 0; i < sampleSize; i++) {
          let x = 0
          for (let j = 0; j < params.alpha; j++) {
            x += Math.random()
          }
          let y = 0
          for (let j = 0; j < params.beta; j++) {
            y += Math.random()
          }
          newSamples.push(x / (x + y))
        }
        break
    }
    
    setSamples(newSamples)
  }

  // 그래프 그리기
  useEffect(() => {
    if (!canvasRef.current || samples.length === 0) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Canvas 크기 설정
    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)
    
    // 배경 클리어
    ctx.fillStyle = '#f9fafb'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // 히스토그램 그리기
    const min = Math.min(...samples)
    const max = Math.max(...samples)
    const range = max - min
    const binCount = distributions[selectedDist].type === 'discrete' 
      ? Math.floor(max - min + 1)
      : 30
    const binWidth = range / binCount
    
    // 빈도 계산
    const bins = new Array(binCount).fill(0)
    samples.forEach(sample => {
      const binIndex = Math.min(
        Math.floor((sample - min) / binWidth),
        binCount - 1
      )
      bins[binIndex]++
    })
    
    const maxFreq = Math.max(...bins)
    const width = canvas.offsetWidth
    const height = canvas.offsetHeight
    const padding = 40
    const graphWidth = width - 2 * padding
    const graphHeight = height - 2 * padding
    
    // 축 그리기
    ctx.strokeStyle = '#374151'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padding, padding)
    ctx.lineTo(padding, height - padding)
    ctx.lineTo(width - padding, height - padding)
    ctx.stroke()
    
    // 히스토그램 막대 그리기
    ctx.fillStyle = 'rgba(147, 51, 234, 0.6)'
    bins.forEach((freq, i) => {
      const x = padding + (i * graphWidth) / binCount
      const barHeight = (freq / maxFreq) * graphHeight
      const y = height - padding - barHeight
      
      ctx.fillRect(
        x + 1,
        y,
        (graphWidth / binCount) - 2,
        barHeight
      )
    })
    
    // 이론적 PDF 그리기 (연속 분포만)
    if (showPDF && distributions[selectedDist].type === 'continuous') {
      ctx.strokeStyle = '#dc2626'
      ctx.lineWidth = 2
      ctx.beginPath()
      
      for (let i = 0; i <= 100; i++) {
        const x = min + (range * i) / 100
        let y = 0
        
        switch (selectedDist) {
          case 'normal':
            y = (1 / (params.std * Math.sqrt(2 * Math.PI))) * 
                Math.exp(-0.5 * Math.pow((x - params.mean) / params.std, 2))
            break
          case 'exponential':
            if (x >= 0) {
              y = params.lambda * Math.exp(-params.lambda * x)
            }
            break
          case 'uniform':
            if (x >= params.a && x <= params.b) {
              y = 1 / (params.b - params.a)
            }
            break
        }
        
        const px = padding + (i * graphWidth) / 100
        const py = height - padding - (y * graphHeight * binWidth * sampleSize / maxFreq)
        
        if (i === 0) {
          ctx.moveTo(px, py)
        } else {
          ctx.lineTo(px, py)
        }
      }
      ctx.stroke()
    }
    
    // 라벨 그리기
    ctx.fillStyle = '#374151'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'center'
    
    // X축 라벨
    for (let i = 0; i <= 5; i++) {
      const value = min + (range * i) / 5
      const x = padding + (i * graphWidth) / 5
      ctx.fillText(value.toFixed(2), x, height - padding + 20)
    }
    
    // Y축 라벨
    ctx.textAlign = 'right'
    for (let i = 0; i <= 5; i++) {
      const value = (maxFreq * i) / 5
      const y = height - padding - (i * graphHeight) / 5
      ctx.fillText(value.toFixed(0), padding - 10, y + 5)
    }
  }, [samples, selectedDist, showPDF, params])

  // 초기 샘플 생성
  useEffect(() => {
    generateSamples()
  }, [selectedDist, params, sampleSize])

  const updateParam = (param: string, value: number) => {
    setParams({ ...params, [param]: value })
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <BarChart3 className="w-6 h-6" />
          확률 분포 시각화
        </h2>
        <div className="flex gap-2">
          <button
            onClick={generateSamples}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center gap-2 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            재생성
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 설정 패널 */}
        <div className="lg:col-span-1 space-y-4">
          {/* 분포 선택 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <label className="block text-sm font-medium mb-2">확률 분포</label>
            <select
              value={selectedDist}
              onChange={(e) => setSelectedDist(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700"
            >
              {Object.entries(distributions).map(([key, dist]) => (
                <option key={key} value={key}>{dist.name}</option>
              ))}
            </select>
          </div>

          {/* 파라미터 설정 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">파라미터</h3>
            {selectedDist === 'normal' && (
              <>
                <div className="mb-3">
                  <label className="block text-sm mb-1">평균 (μ)</label>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={params.mean}
                    onChange={(e) => updateParam('mean', Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm">{params.mean.toFixed(1)}</div>
                </div>
                <div>
                  <label className="block text-sm mb-1">표준편차 (σ)</label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={params.std}
                    onChange={(e) => updateParam('std', Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm">{params.std.toFixed(1)}</div>
                </div>
              </>
            )}
            
            {selectedDist === 'binomial' && (
              <>
                <div className="mb-3">
                  <label className="block text-sm mb-1">시행 횟수 (n)</label>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={params.n}
                    onChange={(e) => updateParam('n', Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm">{params.n}</div>
                </div>
                <div>
                  <label className="block text-sm mb-1">성공 확률 (p)</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={params.p}
                    onChange={(e) => updateParam('p', Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm">{params.p.toFixed(2)}</div>
                </div>
              </>
            )}
            
            {(selectedDist === 'poisson' || selectedDist === 'exponential') && (
              <div>
                <label className="block text-sm mb-1">λ (람다)</label>
                <input
                  type="range"
                  min="0.1"
                  max="10"
                  step="0.1"
                  value={params.lambda}
                  onChange={(e) => updateParam('lambda', Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-center text-sm">{params.lambda.toFixed(1)}</div>
              </div>
            )}
            
            {selectedDist === 'uniform' && (
              <>
                <div className="mb-3">
                  <label className="block text-sm mb-1">최솟값 (a)</label>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={params.a}
                    onChange={(e) => updateParam('a', Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm">{params.a.toFixed(1)}</div>
                </div>
                <div>
                  <label className="block text-sm mb-1">최댓값 (b)</label>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={params.b}
                    onChange={(e) => updateParam('b', Math.max(params.a + 0.1, Number(e.target.value)))}
                    className="w-full"
                  />
                  <div className="text-center text-sm">{params.b.toFixed(1)}</div>
                </div>
              </>
            )}
            
            {selectedDist === 'beta' && (
              <>
                <div className="mb-3">
                  <label className="block text-sm mb-1">α (알파)</label>
                  <input
                    type="range"
                    min="0.1"
                    max="10"
                    step="0.1"
                    value={params.alpha}
                    onChange={(e) => updateParam('alpha', Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm">{params.alpha.toFixed(1)}</div>
                </div>
                <div>
                  <label className="block text-sm mb-1">β (베타)</label>
                  <input
                    type="range"
                    min="0.1"
                    max="10"
                    step="0.1"
                    value={params.beta}
                    onChange={(e) => updateParam('beta', Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm">{params.beta.toFixed(1)}</div>
                </div>
              </>
            )}
          </div>

          {/* 샘플 크기 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <label className="block text-sm font-medium mb-2">샘플 크기</label>
            <input
              type="range"
              min="100"
              max="10000"
              step="100"
              value={sampleSize}
              onChange={(e) => setSampleSize(Number(e.target.value))}
              className="w-full"
            />
            <div className="text-center text-sm">{sampleSize.toLocaleString()}</div>
          </div>

          {/* 표시 옵션 */}
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-3">표시 옵션</h3>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showPDF}
                onChange={(e) => setShowPDF(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">이론적 PDF 표시</span>
            </label>
          </div>

          {/* 분포 정보 */}
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h3 className="font-semibold mb-2">{distributions[selectedDist].name}</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              {distributions[selectedDist].description}
            </p>
            <div className="bg-white dark:bg-gray-800 p-2 rounded font-mono text-xs">
              {distributions[selectedDist].formula}
            </div>
          </div>
        </div>

        {/* 그래프 영역 */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
            <canvas
              ref={canvasRef}
              className="w-full h-96"
              style={{ width: '100%', height: '384px' }}
            />
          </div>

          {/* 통계 정보 */}
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-center">
              <div className="text-sm text-gray-600 dark:text-gray-400">평균</div>
              <div className="font-semibold">
                {samples.length > 0 
                  ? (samples.reduce((a, b) => a + b, 0) / samples.length).toFixed(3)
                  : '-'}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-center">
              <div className="text-sm text-gray-600 dark:text-gray-400">표준편차</div>
              <div className="font-semibold">
                {samples.length > 0 
                  ? Math.sqrt(
                      samples.reduce((sum, x) => {
                        const mean = samples.reduce((a, b) => a + b, 0) / samples.length
                        return sum + Math.pow(x - mean, 2)
                      }, 0) / samples.length
                    ).toFixed(3)
                  : '-'}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-center">
              <div className="text-sm text-gray-600 dark:text-gray-400">최솟값</div>
              <div className="font-semibold">
                {samples.length > 0 ? Math.min(...samples).toFixed(3) : '-'}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg text-center">
              <div className="text-sm text-gray-600 dark:text-gray-400">최댓값</div>
              <div className="font-semibold">
                {samples.length > 0 ? Math.max(...samples).toFixed(3) : '-'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}