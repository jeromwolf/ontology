'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { ArrowLeft, Play, Pause, RotateCcw, BarChart3, TrendingUp, AlertTriangle, CheckCircle, Sigma, Target, Activity, Download, Settings, Sparkles } from 'lucide-react'

interface ProcessData {
  timestamp: Date
  value: number
  subgroup: number
  outOfControl: boolean
  rule: string | null
}

interface ControlLimits {
  ucl: number  // Upper Control Limit
  lcl: number  // Lower Control Limit
  cl: number   // Center Line
  usl: number  // Upper Specification Limit
  lsl: number  // Lower Specification Limit
}

interface ProcessCapability {
  cp: number
  cpk: number
  pp: number
  ppk: number
  sigma: number
  dpmo: number
  yield: number
}

interface WesternElectricRule {
  id: string
  name: string
  description: string
  check: (data: number[], limits: ControlLimits) => boolean
}

export default function SPCControlSystemPage() {
  const searchParams = useSearchParams()
  const backUrl = searchParams.get('from') || '/modules/smart-factory'
  const [isRunning, setIsRunning] = useState(false)
  const [selectedChart, setSelectedChart] = useState<'xbar-r' | 'xbar-s' | 'p-chart' | 'c-chart'>('xbar-r')
  const [processType, setProcessType] = useState<'stable' | 'shift' | 'trend' | 'cyclic'>('stable')
  const [subgroupSize, setSubgroupSize] = useState(5)
  const [showWesternRules, setShowWesternRules] = useState(true)
  
  const xbarChartRef = useRef<HTMLCanvasElement>(null)
  const rChartRef = useRef<HTMLCanvasElement>(null)
  const histogramRef = useRef<HTMLCanvasElement>(null)
  
  const [processData, setProcessData] = useState<ProcessData[]>([])
  const [controlLimits, setControlLimits] = useState<ControlLimits>({
    ucl: 10.5,
    lcl: 9.5,
    cl: 10.0,
    usl: 11.0,
    lsl: 9.0
  })
  const [capability, setCapability] = useState<ProcessCapability>({
    cp: 1.33,
    cpk: 1.25,
    pp: 1.28,
    ppk: 1.20,
    sigma: 3.8,
    dpmo: 12500,
    yield: 98.75
  })

  const westernElectricRules: WesternElectricRule[] = [
    {
      id: 'rule1',
      name: 'Rule 1',
      description: '1개 점이 3σ 한계선 밖',
      check: (data, limits) => {
        const last = data[data.length - 1]
        return last > limits.ucl || last < limits.lcl
      }
    },
    {
      id: 'rule2',
      name: 'Rule 2',
      description: '연속 9개 점이 중심선 한쪽',
      check: (data, limits) => {
        if (data.length < 9) return false
        const last9 = data.slice(-9)
        return last9.every(d => d > limits.cl) || last9.every(d => d < limits.cl)
      }
    },
    {
      id: 'rule3',
      name: 'Rule 3',
      description: '연속 6개 점이 증가/감소',
      check: (data, limits) => {
        if (data.length < 6) return false
        const last6 = data.slice(-6)
        let increasing = true
        let decreasing = true
        for (let i = 1; i < 6; i++) {
          if (last6[i] <= last6[i-1]) increasing = false
          if (last6[i] >= last6[i-1]) decreasing = false
        }
        return increasing || decreasing
      }
    },
    {
      id: 'rule4',
      name: 'Rule 4',
      description: '연속 14개 점이 교대로 상승/하강',
      check: (data, limits) => {
        if (data.length < 14) return false
        const last14 = data.slice(-14)
        for (let i = 2; i < 14; i++) {
          if ((last14[i] - last14[i-1]) * (last14[i-1] - last14[i-2]) >= 0) return false
        }
        return true
      }
    }
  ]

  // X-bar 차트 그리기
  useEffect(() => {
    if (!xbarChartRef.current) return
    const canvas = xbarChartRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = 250

    const drawChart = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 배경 영역
      ctx.fillStyle = 'rgba(34, 197, 94, 0.1)'
      ctx.fillRect(0, (canvas.height - (controlLimits.cl - controlLimits.lcl) * 100) / 2, 
                   canvas.width, (controlLimits.ucl - controlLimits.lcl) * 100)

      // 그리드
      ctx.strokeStyle = '#E5E7EB'
      ctx.lineWidth = 1
      for (let i = 0; i <= 10; i++) {
        const y = (canvas.height / 10) * i
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      // 관리 한계선
      const drawLine = (y: number, color: string, label: string, dash?: number[]) => {
        const yPos = canvas.height - ((y - 8) * 50)
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        if (dash) ctx.setLineDash(dash)
        ctx.beginPath()
        ctx.moveTo(0, yPos)
        ctx.lineTo(canvas.width, yPos)
        ctx.stroke()
        ctx.setLineDash([])
        
        ctx.fillStyle = color
        ctx.font = '12px Arial'
        ctx.fillText(label + ': ' + y.toFixed(1), 10, yPos - 5)
      }

      drawLine(controlLimits.ucl, '#EF4444', 'UCL')
      drawLine(controlLimits.cl, '#3B82F6', 'CL', [5, 5])
      drawLine(controlLimits.lcl, '#EF4444', 'LCL')
      drawLine(controlLimits.usl, '#F59E0B', 'USL', [10, 5])
      drawLine(controlLimits.lsl, '#F59E0B', 'LSL', [10, 5])

      // 데이터 점 그리기
      if (processData.length > 0) {
        ctx.strokeStyle = '#3B82F6'
        ctx.lineWidth = 2
        ctx.beginPath()
        
        const pointsToShow = Math.min(processData.length, 50)
        const startIndex = Math.max(0, processData.length - pointsToShow)
        
        processData.slice(startIndex).forEach((data, index) => {
          const x = (canvas.width / pointsToShow) * index
          const y = canvas.height - ((data.value - 8) * 50)
          
          if (index === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
          
          // 데이터 점
          ctx.save()
          ctx.fillStyle = data.outOfControl ? '#EF4444' : '#3B82F6'
          ctx.beginPath()
          ctx.arc(x, y, 4, 0, Math.PI * 2)
          ctx.fill()
          
          // 관리이탈 표시
          if (data.outOfControl) {
            ctx.strokeStyle = '#EF4444'
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.arc(x, y, 8, 0, Math.PI * 2)
            ctx.stroke()
            
            // 룰 표시
            if (data.rule) {
              ctx.fillStyle = '#EF4444'
              ctx.font = '10px Arial'
              ctx.fillText(data.rule, x - 10, y - 15)
            }
          }
          ctx.restore()
        })
        ctx.stroke()
      }

      // 현재 값 표시
      if (processData.length > 0) {
        const lastData = processData[processData.length - 1]
        ctx.fillStyle = '#000'
        ctx.font = 'bold 14px Arial'
        ctx.fillText(`현재값: ${lastData.value.toFixed(2)}`, canvas.width - 100, 20)
      }
    }

    const interval = setInterval(drawChart, 100)
    return () => clearInterval(interval)
  }, [processData, controlLimits])

  // R 차트 그리기
  useEffect(() => {
    if (!rChartRef.current) return
    const canvas = rChartRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = 200

    const drawRChart = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // R 차트 배경
      ctx.fillStyle = 'rgba(34, 197, 94, 0.1)'
      ctx.fillRect(0, 0, canvas.width, canvas.height * 0.7)

      // 그리드
      ctx.strokeStyle = '#E5E7EB'
      ctx.lineWidth = 1
      for (let i = 0; i <= 5; i++) {
        const y = (canvas.height / 5) * i
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      // R 차트 한계선
      const rUCL = 2.114 * 0.5 // D4 * R-bar
      const rCL = 0.5
      const rLCL = 0

      ctx.strokeStyle = '#EF4444'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(0, canvas.height * 0.2)
      ctx.lineTo(canvas.width, canvas.height * 0.2)
      ctx.stroke()
      
      ctx.strokeStyle = '#3B82F6'
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(0, canvas.height * 0.6)
      ctx.lineTo(canvas.width, canvas.height * 0.6)
      ctx.stroke()
      ctx.setLineDash([])

      // R 데이터
      if (processData.length > subgroupSize) {
        ctx.strokeStyle = '#10B981'
        ctx.lineWidth = 2
        ctx.beginPath()
        
        const ranges: number[] = []
        for (let i = 0; i < processData.length - subgroupSize + 1; i += subgroupSize) {
          const subgroup = processData.slice(i, i + subgroupSize)
          const max = Math.max(...subgroup.map(d => d.value))
          const min = Math.min(...subgroup.map(d => d.value))
          ranges.push(max - min)
        }
        
        const pointsToShow = Math.min(ranges.length, 50)
        ranges.slice(-pointsToShow).forEach((range, index) => {
          const x = (canvas.width / pointsToShow) * index
          const y = canvas.height - (range / 3) * canvas.height
          
          if (index === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
          
          ctx.save()
          ctx.fillStyle = '#10B981'
          ctx.beginPath()
          ctx.arc(x, y, 3, 0, Math.PI * 2)
          ctx.fill()
          ctx.restore()
        })
        ctx.stroke()
      }

      // 라벨
      ctx.fillStyle = '#000'
      ctx.font = '12px Arial'
      ctx.fillText('R-UCL: ' + rUCL.toFixed(2), 10, canvas.height * 0.2 - 5)
      ctx.fillText('R-CL: ' + rCL.toFixed(2), 10, canvas.height * 0.6 - 5)
    }

    const interval = setInterval(drawRChart, 100)
    return () => clearInterval(interval)
  }, [processData, subgroupSize])

  // 히스토그램 그리기
  useEffect(() => {
    if (!histogramRef.current) return
    const canvas = histogramRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = 250

    const drawHistogram = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (processData.length < 20) return

      // 데이터 분포 계산
      const values = processData.map(d => d.value)
      const min = Math.min(...values)
      const max = Math.max(...values)
      const bins = 20
      const binWidth = (max - min) / bins
      const histogram = new Array(bins).fill(0)

      values.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1)
        histogram[binIndex]++
      })

      const maxCount = Math.max(...histogram)

      // 히스토그램 그리기
      histogram.forEach((count, index) => {
        const x = (canvas.width / bins) * index
        const barWidth = canvas.width / bins - 2
        const barHeight = (count / maxCount) * (canvas.height - 40)
        
        ctx.fillStyle = '#3B82F6'
        ctx.fillRect(x, canvas.height - barHeight - 20, barWidth, barHeight)
      })

      // 정규분포 곡선
      ctx.strokeStyle = '#EF4444'
      ctx.lineWidth = 2
      ctx.beginPath()
      
      const mean = values.reduce((a, b) => a + b) / values.length
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length
      const stdDev = Math.sqrt(variance)
      
      for (let i = 0; i < canvas.width; i++) {
        const x = min + (max - min) * (i / canvas.width)
        const y = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * 
                 Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2))
        const yPos = canvas.height - 20 - (y * values.length * binWidth) / maxCount * (canvas.height - 40)
        
        if (i === 0) ctx.moveTo(i, yPos)
        else ctx.lineTo(i, yPos)
      }
      ctx.stroke()

      // 규격 한계선
      const drawSpecLine = (value: number, color: string, label: string) => {
        const x = ((value - min) / (max - min)) * canvas.width
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, canvas.height - 20)
        ctx.stroke()
        ctx.setLineDash([])
        
        ctx.fillStyle = color
        ctx.font = '10px Arial'
        ctx.save()
        ctx.translate(x, 10)
        ctx.rotate(-Math.PI / 2)
        ctx.fillText(label, 0, 0)
        ctx.restore()
      }

      drawSpecLine(controlLimits.lsl, '#F59E0B', 'LSL')
      drawSpecLine(controlLimits.usl, '#F59E0B', 'USL')

      // 통계 정보
      ctx.fillStyle = '#000'
      ctx.font = '12px Arial'
      ctx.fillText(`평균: ${mean.toFixed(2)}`, 10, canvas.height - 5)
      ctx.fillText(`표준편차: ${stdDev.toFixed(3)}`, 100, canvas.height - 5)
    }

    const interval = setInterval(drawHistogram, 500)
    return () => clearInterval(interval)
  }, [processData, controlLimits])

  // 데이터 생성
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      const newData: ProcessData = {
        timestamp: new Date(),
        value: 0,
        subgroup: Math.floor(processData.length / subgroupSize),
        outOfControl: false,
        rule: null
      }

      // 프로세스 타입에 따른 데이터 생성
      const time = processData.length
      let baseValue = 10.0
      
      switch (processType) {
        case 'stable':
          baseValue = 10.0 + (Math.random() - 0.5) * 0.8
          break
        case 'shift':
          if (time > 30) baseValue = 10.3 + (Math.random() - 0.5) * 0.8
          else baseValue = 10.0 + (Math.random() - 0.5) * 0.8
          break
        case 'trend':
          baseValue = 10.0 + (time * 0.01) + (Math.random() - 0.5) * 0.8
          break
        case 'cyclic':
          baseValue = 10.0 + Math.sin(time * 0.2) * 0.5 + (Math.random() - 0.5) * 0.8
          break
      }

      newData.value = baseValue

      // Western Electric Rules 검사
      if (showWesternRules && processData.length > 0) {
        const recentValues = [...processData.slice(-20).map(d => d.value), newData.value]
        
        for (const rule of westernElectricRules) {
          if (rule.check(recentValues, controlLimits)) {
            newData.outOfControl = true
            newData.rule = rule.name
            break
          }
        }
      }

      setProcessData(prev => [...prev.slice(-199), newData])

      // 공정능력 업데이트
      if (processData.length > 30) {
        const values = processData.slice(-30).map(d => d.value)
        const mean = values.reduce((a, b) => a + b) / values.length
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length
        const stdDev = Math.sqrt(variance)
        
        const cp = (controlLimits.usl - controlLimits.lsl) / (6 * stdDev)
        const cpu = (controlLimits.usl - mean) / (3 * stdDev)
        const cpl = (mean - controlLimits.lsl) / (3 * stdDev)
        const cpk = Math.min(cpu, cpl)
        
        const sigma = 3 + (cpk - 1) * 1.5
        const dpmo = Math.round(1000000 * (1 - normalCDF(sigma)))
        
        setCapability({
          cp: cp,
          cpk: cpk,
          pp: cp * 0.95,
          ppk: cpk * 0.95,
          sigma: sigma,
          dpmo: dpmo,
          yield: 100 - (dpmo / 10000)
        })
      }
    }, 500)

    return () => clearInterval(interval)
  }, [isRunning, processType, processData, controlLimits, showWesternRules, subgroupSize])

  // 정규분포 누적분포함수
  const normalCDF = (x: number) => {
    return 0.5 * (1 + erf(x / Math.sqrt(2)))
  }

  const erf = (x: number) => {
    const a1 = 0.254829592
    const a2 = -0.284496736
    const a3 = 1.421413741
    const a4 = -1.453152027
    const a5 = 1.061405429
    const p = 0.3275911
    const sign = x < 0 ? -1 : 1
    x = Math.abs(x)
    const t = 1.0 / (1.0 + p * x)
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)
    return sign * y
  }

  const getCapabilityColor = (cpk: number) => {
    if (cpk > 1.33) return 'text-green-500'
    if (cpk > 1.0) return 'text-yellow-500'
    return 'text-red-500'
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link 
                href={backUrl}
                className="flex items-center gap-2 text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>학습 페이지로 돌아가기</span>
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowWesternRules(!showWesternRules)}
                className={`px-3 py-1 rounded text-sm ${
                  showWesternRules 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                }`}
              >
                <AlertTriangle className="w-4 h-4 inline mr-1" />
                Western Rules
              </button>
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium ${
                  isRunning 
                    ? 'bg-red-600 text-white hover:bg-red-700' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '모니터링 중지' : '모니터링 시작'}
              </button>
              <button
                onClick={() => {
                  setIsRunning(false)
                  setProcessData([])
                  setCapability({
                    cp: 1.33,
                    cpk: 1.25,
                    pp: 1.28,
                    ppk: 1.20,
                    sigma: 3.8,
                    dpmo: 12500,
                    yield: 98.75
                  })
                }}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                <RotateCcw className="w-4 h-4" />
                리셋
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Title */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                SPC 통계 공정 관리 시스템
                <Sparkles className="w-6 h-6 text-yellow-500" />
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">실시간 품질 모니터링과 공정 능력 분석</p>
            </div>
          </div>
        </div>

        {/* Process Capability Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-5 h-5 text-blue-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Cp</span>
            </div>
            <div className={`text-2xl font-bold ${getCapabilityColor(capability.cp)}`}>
              {capability.cp.toFixed(2)}
            </div>
          </div>

          <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
            capability.cpk < 1.0 ? 'border-red-500 ring-2 ring-red-500 ring-opacity-50' : 'border-gray-200 dark:border-gray-700'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-5 h-5 text-green-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Cpk</span>
            </div>
            <div className={`text-2xl font-bold ${getCapabilityColor(capability.cpk)}`}>
              {capability.cpk.toFixed(2)}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <Sigma className="w-5 h-5 text-purple-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Sigma</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {capability.sigma.toFixed(1)}σ
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-orange-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">DPMO</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {capability.dpmo.toLocaleString()}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">수율</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {capability.yield.toFixed(2)}%
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-5 h-5 text-blue-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Pp</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {capability.pp.toFixed(2)}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-5 h-5 text-indigo-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Ppk</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {capability.ppk.toFixed(2)}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Control Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* X-bar Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">X̄ 관리도</h2>
              <canvas 
                ref={xbarChartRef}
                className="w-full"
                style={{ height: '250px' }}
              />
            </div>

            {/* R Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">R 관리도</h2>
              <canvas 
                ref={rChartRef}
                className="w-full"
                style={{ height: '200px' }}
              />
            </div>

            {/* Histogram */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">공정 분포</h2>
              <canvas 
                ref={histogramRef}
                className="w-full"
                style={{ height: '250px' }}
              />
            </div>
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            {/* Process Type */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">공정 상태</h3>
              
              <div className="space-y-3">
                <button
                  onClick={() => setProcessType('stable')}
                  className={`w-full p-3 rounded-lg border text-left ${
                    processType === 'stable'
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white">안정 상태</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">정상적인 변동만 존재</div>
                </button>
                
                <button
                  onClick={() => setProcessType('shift')}
                  className={`w-full p-3 rounded-lg border text-left ${
                    processType === 'shift'
                      ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white">평균 이동</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">공정 평균이 변화</div>
                </button>
                
                <button
                  onClick={() => setProcessType('trend')}
                  className={`w-full p-3 rounded-lg border text-left ${
                    processType === 'trend'
                      ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white">추세 발생</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">지속적인 증가/감소</div>
                </button>
                
                <button
                  onClick={() => setProcessType('cyclic')}
                  className={`w-full p-3 rounded-lg border text-left ${
                    processType === 'cyclic'
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white">주기적 변동</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">반복적인 패턴</div>
                </button>
              </div>
            </div>

            {/* Western Electric Rules */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Western Electric Rules</h3>
              
              <div className="space-y-3">
                {westernElectricRules.map((rule) => (
                  <div key={rule.id} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="font-medium text-gray-900 dark:text-white">{rule.name}</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">{rule.description}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">설정</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    부분군 크기
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={subgroupSize}
                    onChange={(e) => setSubgroupSize(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                  />
                </div>
                
                <button className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm font-medium">
                  <Settings className="w-4 h-4 inline mr-2" />
                  관리한계 재계산
                </button>
                
                <button className="w-full p-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors text-sm font-medium">
                  <Download className="w-4 h-4 inline mr-2" />
                  데이터 내보내기
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}