'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, Info, Zap, Thermometer, Activity, AlertTriangle, CheckCircle, Settings, HelpCircle, Gauge, Package } from 'lucide-react'

interface InteractiveDigitalTwinProps {
  backUrl?: string
}

export default function InteractiveDigitalTwin({ backUrl = '/modules/smart-factory' }: InteractiveDigitalTwinProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [showHelp, setShowHelp] = useState(true)
  
  // 사용자가 조정 가능한 파라미터 (초기값을 더 합리적으로 설정)
  const [productionSpeed, setProductionSpeed] = useState(50) // 생산 속도 (0-100)
  const [coolingPower, setCoolingPower] = useState(60) // 냉각 파워 (0-100)
  const [qualityControl, setQualityControl] = useState(80) // 품질 관리 수준 (0-100)
  
  // 실시간 데이터
  const [temperature, setTemperature] = useState(25)
  const [production, setProduction] = useState(0)
  const [machineHealth, setMachineHealth] = useState(100)
  const [defects, setDefects] = useState(0)
  const [totalCost, setTotalCost] = useState(0)
  const [productAnimations, setProductAnimations] = useState<number[]>([])

  // 시뮬레이션 로직
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      // 온도 계산 (생산 속도가 높으면 온도 상승, 냉각이 강하면 온도 하강)
      setTemperature(prev => {
        const heatGeneration = productionSpeed * 0.25 // 생산 속도에 비례한 열 발생 (감소)
        const cooling = coolingPower * 0.3 // 냉각 효과 (증가)
        const ambientTemp = 25 // 주변 온도
        
        const newTemp = prev + (heatGeneration - cooling) / 12 + (ambientTemp - prev) * 0.02
        return Math.min(80, Math.max(20, newTemp))
      })

      // 생산량 증가 (속도에 비례)
      if (Math.random() < productionSpeed / 100) {
        setProduction(prev => prev + 1)
        setProductAnimations(prev => [...prev, Date.now()])
      }

      // 기계 상태 (온도와 속도에 영향)
      setMachineHealth(prev => {
        if (temperature > 65) return Math.max(0, prev - 1.5)
        if (temperature > 50 && productionSpeed > 80) return Math.max(0, prev - 0.8)
        if (productionSpeed < 30) return Math.min(100, prev + 0.2) // 천천히 하면 회복
        return Math.max(0, prev - 0.1)
      })

      // 불량품 발생 (품질 관리 수준과 온도에 영향) - 더 명확한 변화
      const baseDefectChance = (100 - qualityControl) / 400 // 더 높은 기본 불량률로 변화를 명확히
      const temperatureDefectChance = temperature > 65 ? 0.1 : 0 // 온도 영향 증가
      const totalDefectChance = baseDefectChance + temperatureDefectChance
      if (Math.random() < totalDefectChance && productionSpeed > 0) {
        setDefects(prev => prev + 1)
      }

      // 비용 계산 (전력 + 냉각 + 품질 관리)
      const powerCost = productionSpeed * 0.5
      const coolingCost = coolingPower * 0.3
      const qualityCost = qualityControl * 0.2
      setTotalCost(prev => prev + (powerCost + coolingCost + qualityCost) / 100)
    }, 500)

    return () => clearInterval(interval)
  }, [isRunning, productionSpeed, coolingPower, qualityControl, temperature])

  // 애니메이션 정리
  useEffect(() => {
    const cleanup = setInterval(() => {
      setProductAnimations(prev => prev.filter(time => Date.now() - time < 3000))
    }, 1000)
    return () => clearInterval(cleanup)
  }, [])

  // 성능 평가
  const getPerformanceScore = () => {
    if (production === 0) return { 
      score: 0, 
      grade: 'N/A', 
      color: 'gray',
      efficiency: 0,
      quality: 0,
      healthScore: 0
    }
    
    // 각 항목 계산 (0~1 범위로 정규화) - 더 관대한 계산
    const efficiency = Math.min(1, production / Math.max(1, totalCost) / 3) // 비용 대비 생산 효율 (더 관대)
    const quality = production > 0 ? (production - defects) / production : 1 // 품질률 (0~1) - 생산 전에는 100%로 가정
    const healthScore = machineHealth / 100 // 기계 건강도 (0~1)
    
    // 총점 계산 (가중치: 품질 50%, 효율 30%, 건강 20%)
    // 품질에 더 관대한 기준 적용
    let qualityScore = quality
    if (quality < 0.9) qualityScore = quality * 0.9   // 90% 미만시 약간 패널티
    if (quality < 0.8) qualityScore = quality * 0.8   // 80% 미만시 패널티
    if (quality < 0.7) qualityScore = quality * 0.6   // 70% 미만시 강한 패널티
    
    // 기계 건강도가 낮으면 추가 패널티
    let adjustedHealthScore = healthScore
    if (healthScore < 0.5) adjustedHealthScore = healthScore * 0.5  // 50% 미만시 강한 패널티
    
    // 온도가 너무 높으면 추가 패널티 - 더 관대한 기준
    let temperaturePenalty = 1
    if (temperature > 75) temperaturePenalty = 0.8
    if (temperature > 65) temperaturePenalty = 0.9
    
    const totalScore = Math.min(100, (efficiency * 0.3 + qualityScore * 0.5 + adjustedHealthScore * 0.2) * 100 * temperaturePenalty)
    
    if (totalScore > 80) return { 
      score: totalScore, 
      grade: 'S', 
      color: 'purple',
      efficiency: efficiency * 100,
      quality: quality * 100,
      healthScore: healthScore * 100
    }
    if (totalScore > 70) return { 
      score: totalScore, 
      grade: 'A', 
      color: 'green',
      efficiency: efficiency * 100,
      quality: quality * 100,
      healthScore: healthScore * 100
    }
    if (totalScore > 60) return { 
      score: totalScore, 
      grade: 'B', 
      color: 'blue',
      efficiency: efficiency * 100,
      quality: quality * 100,
      healthScore: healthScore * 100
    }
    if (totalScore > 50) return { 
      score: totalScore, 
      grade: 'C', 
      color: 'yellow',
      efficiency: efficiency * 100,
      quality: quality * 100,
      healthScore: healthScore * 100
    }
    return { 
      score: totalScore, 
      grade: 'D', 
      color: 'red',
      efficiency: efficiency * 100,
      quality: quality * 100,
      healthScore: healthScore * 100
    }
  }

  const performance = getPerformanceScore()

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4">
      {/* 헤더 */}
      <div className="max-w-7xl mx-auto mb-4">
        <Link 
          href={backUrl}
          className="inline-flex items-center gap-2 text-amber-600 hover:text-amber-700 mb-3"
        >
          <ArrowLeft className="w-5 h-5" />
          학습 페이지로 돌아가기
        </Link>
        
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              인터랙티브 디지털 트윈
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              직접 조작해보며 최적의 공장 운영 방법을 찾아보세요!
            </p>
          </div>
          <button
            onClick={() => setShowHelp(!showHelp)}
            className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400"
          >
            <HelpCircle className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* 도움말 */}
      {showHelp && (
        <div className="max-w-7xl mx-auto mb-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-xl border border-blue-200 dark:border-blue-800">
          <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
            🎮 조작 방법
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-blue-800 dark:text-blue-200">
            <div>
              <strong>🚀 생산 속도</strong>: 빠르게 하면 많이 생산하지만 온도가 올라갑니다
            </div>
            <div>
              <strong>❄️ 냉각 파워</strong>: 기계를 시원하게 유지하지만 비용이 듭니다
            </div>
            <div>
              <strong>✅ 품질 관리</strong>: 불량품을 줄이지만 생산 비용이 증가합니다
            </div>
          </div>
          <p className="text-sm text-blue-700 dark:text-blue-300 mt-3">
            💡 목표: 적은 비용으로 많은 제품을 불량 없이 생산하세요! S등급에 도전해보세요!
          </p>
          <button
            onClick={() => setShowHelp(false)}
            className="mt-2 text-sm text-blue-600 hover:text-blue-700"
          >
            닫기
          </button>
        </div>
      )}

      {/* 성능 점수 */}
      <div className="max-w-7xl mx-auto mb-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold">📊 실시간 성능</h3>
            <div className={`text-4xl font-bold text-${performance.color}-600`}>
              {performance.grade} 등급
            </div>
          </div>
          
          {/* 등급 기준 설명 */}
          <div className="mb-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <div className="font-semibold text-gray-700 dark:text-gray-300 mb-1">💰 비용 효율성 (30%)</div>
                <div className="text-lg font-bold">{performance.efficiency.toFixed(0)}%</div>
                <div className="text-xs text-gray-500">생산량 ÷ 비용</div>
              </div>
              <div>
                <div className="font-semibold text-gray-700 dark:text-gray-300 mb-1">✅ 품질률 (50%)</div>
                <div className="text-lg font-bold">{performance.quality.toFixed(0)}%</div>
                <div className="text-xs text-gray-500">정상품 ÷ 총생산</div>
              </div>
              <div>
                <div className="font-semibold text-gray-700 dark:text-gray-300 mb-1">🔧 기계 건강 (20%)</div>
                <div className="text-lg font-bold">{performance.healthScore.toFixed(0)}%</div>
                <div className="text-xs text-gray-500">현재 상태</div>
              </div>
            </div>
            <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">총점:</span>
                <span className="text-lg font-bold">{performance.score.toFixed(1)}점 / 100점</span>
              </div>
            </div>
          </div>
          
          {/* 등급표 */}
          <div className="grid grid-cols-5 gap-2 text-center text-xs">
            <div className={`p-2 rounded ${performance.grade === 'S' ? 'bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 font-bold' : 'bg-gray-100 dark:bg-gray-700 text-gray-500'}`}>
              <div className="font-semibold">S</div>
              <div>80점+</div>
            </div>
            <div className={`p-2 rounded ${performance.grade === 'A' ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 font-bold' : 'bg-gray-100 dark:bg-gray-700 text-gray-500'}`}>
              <div className="font-semibold">A</div>
              <div>70-79</div>
            </div>
            <div className={`p-2 rounded ${performance.grade === 'B' ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 font-bold' : 'bg-gray-100 dark:bg-gray-700 text-gray-500'}`}>
              <div className="font-semibold">B</div>
              <div>60-69</div>
            </div>
            <div className={`p-2 rounded ${performance.grade === 'C' ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300 font-bold' : 'bg-gray-100 dark:bg-gray-700 text-gray-500'}`}>
              <div className="font-semibold">C</div>
              <div>50-59</div>
            </div>
            <div className={`p-2 rounded ${performance.grade === 'D' ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 font-bold' : 'bg-gray-100 dark:bg-gray-700 text-gray-500'}`}>
              <div className="font-semibold">D</div>
              <div>50점-</div>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mt-3">
            <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <Thermometer className={`w-6 h-6 mx-auto mb-1 ${
                temperature > 60 ? 'text-red-500' : 
                temperature > 45 ? 'text-yellow-500' : 'text-green-500'
              }`} />
              <div className="text-lg font-bold">{temperature.toFixed(1)}°C</div>
              <div className="text-xs text-gray-600 dark:text-gray-400">온도</div>
            </div>
            
            <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <Package className="w-6 h-6 mx-auto mb-1 text-blue-500" />
              <div className="text-lg font-bold">{production}</div>
              <div className="text-xs text-gray-600 dark:text-gray-400">생산량</div>
            </div>
            
            <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <AlertTriangle className="w-6 h-6 mx-auto mb-1 text-red-500" />
              <div className="text-lg font-bold text-red-600">{defects}</div>
              <div className="text-xs text-gray-600 dark:text-gray-400">불량품</div>
            </div>
            
            <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <Gauge className={`w-6 h-6 mx-auto mb-1 ${
                machineHealth > 60 ? 'text-green-500' : 
                machineHealth > 30 ? 'text-yellow-500' : 'text-red-500'
              }`} />
              <div className="text-lg font-bold">{machineHealth.toFixed(0)}%</div>
              <div className="text-xs text-gray-600 dark:text-gray-400">기계 상태</div>
            </div>
            
            <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <span className="text-2xl mb-1">💰</span>
              <div className="text-lg font-bold">${totalCost.toFixed(0)}</div>
              <div className="text-xs text-gray-600 dark:text-gray-400">총 비용</div>
            </div>
            
            <div className="text-center p-2 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <span className="text-2xl mb-1">📈</span>
              <div className="text-lg font-bold">{production > 0 ? ((production - defects) / production * 100).toFixed(0) : 0}%</div>
              <div className="text-xs text-gray-600 dark:text-gray-400">품질률</div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* 왼쪽: 조작 패널 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-lg">
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-purple-600" />
            컨트롤 패널
          </h2>

          <div className="space-y-6">
            {/* 생산 속도 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="font-medium flex items-center gap-2">
                  <span className="text-xl">🚀</span> 생산 속도
                </label>
                <span className="text-lg font-bold text-blue-600">{productionSpeed}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={productionSpeed}
                onChange={(e) => setProductionSpeed(Number(e.target.value))}
                className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${productionSpeed}%, #E5E7EB ${productionSpeed}%, #E5E7EB 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>정지</span>
                <span>최대 속도</span>
              </div>
            </div>

            {/* 냉각 파워 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="font-medium flex items-center gap-2">
                  <span className="text-xl">❄️</span> 냉각 파워
                </label>
                <span className="text-lg font-bold text-cyan-600">{coolingPower}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={coolingPower}
                onChange={(e) => setCoolingPower(Number(e.target.value))}
                className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #06B6D4 0%, #06B6D4 ${coolingPower}%, #E5E7EB ${coolingPower}%, #E5E7EB 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>끄기</span>
                <span>최대 냉각</span>
              </div>
            </div>

            {/* 품질 관리 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="font-medium flex items-center gap-2">
                  <span className="text-xl">✅</span> 품질 관리
                </label>
                <span className="text-lg font-bold text-green-600">{qualityControl}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={qualityControl}
                onChange={(e) => setQualityControl(Number(e.target.value))}
                className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #10B981 0%, #10B981 ${qualityControl}%, #E5E7EB ${qualityControl}%, #E5E7EB 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>낮음</span>
                <span>엄격함</span>
              </div>
            </div>
          </div>

          <div className="mt-6 space-y-3">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition-colors ${
                isRunning 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              {isRunning ? '일시정지' : '시뮬레이션 시작'}
            </button>
            
            <button
              onClick={() => {
                setIsRunning(false)
                setTemperature(25)
                setProduction(0)
                setMachineHealth(100)
                setDefects(0)
                setTotalCost(0)
                setProductionSpeed(50)
                setCoolingPower(60)
                setQualityControl(80)
                setProductAnimations([])
              }}
              className="w-full flex items-center justify-center gap-2 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium"
            >
              <RotateCcw className="w-5 h-5" />
              리셋
            </button>
          </div>

          {/* 팁 */}
          <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
            <p className="text-xs text-amber-800 dark:text-amber-200">
              💡 <strong>팁:</strong> 
              {temperature > 60 ? '온도가 너무 높습니다! 냉각을 강화하거나 속도를 줄이세요.' :
               machineHealth < 50 ? '기계 상태가 좋지 않습니다. 속도를 늑춰서 회복시키세요.' :
               production > 0 && defects / production > 0.1 ? '불량률이 너무 높습니다. 품질 관리를 강화하세요.' :
               '온도가 50°C를 넘지 않도록 유지하면서 생산 속도를 최대한 높여보세요!'}
            </p>
          </div>
        </div>

        {/* 오른쪽: 가상 공장 시각화 */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl p-5 shadow-lg">
          <h2 className="text-lg font-bold mb-4">🏭 가상 공장</h2>
          
          <div className="relative h-96 bg-gradient-to-b from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl overflow-hidden">
            {/* 온도 게이지 */}
            <div className="absolute top-4 right-4 w-20 h-48 bg-white dark:bg-gray-900 rounded-lg p-2">
              <div className="text-center text-xs font-semibold mb-1">온도</div>
              <div className="relative h-32 w-8 mx-auto bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className={`absolute bottom-0 w-full transition-all duration-500 ${
                    temperature > 60 ? 'bg-red-500' : 
                    temperature > 45 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ height: `${(temperature / 80) * 100}%` }}
                />
              </div>
              <div className="text-center text-sm font-bold mt-1">
                {temperature.toFixed(0)}°C
              </div>
            </div>

            {/* 공장 기계 */}
            <div className="absolute top-20 left-1/3 transform -translate-x-1/2">
              <div className={`
                w-40 h-32 rounded-lg flex flex-col items-center justify-center
                transition-all duration-500 relative shadow-lg
                ${machineHealth < 30 ? 'bg-red-500' : 
                  machineHealth < 60 ? 'bg-yellow-500' : 'bg-green-500'}
              `}>
                <span className="text-5xl">🏭</span>
                <div className="absolute -top-2 -right-2 bg-white dark:bg-gray-900 rounded-full px-2 py-1 text-xs font-bold">
                  {machineHealth.toFixed(0)}%
                </div>
                
                {/* 작동 애니메이션 */}
                {isRunning && productionSpeed > 0 && (
                  <>
                    <div className={`absolute -top-8 ${productionSpeed > 70 ? 'animate-spin' : 'animate-pulse'}`}>
                      <span className="text-3xl">⚙️</span>
                    </div>
                    {temperature > 50 && (
                      <div className="absolute -top-12 animate-bounce">
                        <span className="text-2xl">💨</span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* 컨베이어 벨트 */}
            <div className="absolute bottom-24 left-0 right-0 h-20 bg-gray-400 dark:bg-gray-600">
              <div className="h-full relative">
                {/* 움직이는 제품들 */}
                {productAnimations.map((startTime, index) => {
                  const progress = ((Date.now() - startTime) / 3000) * 100
                  return (
                    <div
                      key={startTime}
                      className="absolute h-16 w-16 flex items-center justify-center text-3xl"
                      style={{ 
                        left: `${progress}%`,
                        transform: 'translateX(-50%)',
                        transition: 'left 3s linear'
                      }}
                    >
                      📦
                    </div>
                  )
                })}
              </div>
              <div className="text-center text-xs text-gray-600 dark:text-gray-400 mt-1">
                컨베이어 벨트 (속도: {productionSpeed}%)
              </div>
            </div>

            {/* 품질 검사 스테이션 */}
            <div className="absolute bottom-24 right-1/4">
              <div className="w-20 h-20 bg-blue-500 rounded-lg flex flex-col items-center justify-center text-white">
                <span className="text-2xl">🔍</span>
                <span className="text-xs">QC {qualityControl}%</span>
              </div>
            </div>

            {/* 냉각 시스템 */}
            {coolingPower > 0 && (
              <div className="absolute top-20 right-1/3">
                <div className={`text-3xl ${coolingPower > 50 ? 'animate-pulse' : ''}`}>
                  ❄️
                </div>
                <div className="text-xs text-center mt-1 bg-white dark:bg-gray-900 rounded px-1">
                  {coolingPower}%
                </div>
              </div>
            )}
          </div>

          {/* 실시간 인사이트 */}
          <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg">
            <h3 className="font-semibold mb-2">💡 실시간 분석</h3>
            <div className="space-y-1 text-sm">
              {temperature > 60 && (
                <p className="text-red-600 dark:text-red-400">
                  🔥 온도가 너무 높습니다! 냉각을 강화하거나 속도를 줄이세요.
                </p>
              )}
              {machineHealth < 50 && (
                <p className="text-yellow-600 dark:text-yellow-400">
                  ⚠️ 기계 상태가 좋지 않습니다. 속도를 줄여 기계를 보호하세요.
                </p>
              )}
              {production > 0 && defects / production > 0.1 && (
                <p className="text-orange-600 dark:text-orange-400">
                  📊 불량률이 10%를 초과했습니다. 품질 관리를 강화하세요.
                </p>
              )}
              {production > 0 && defects / production > 0.05 && defects / production <= 0.1 && (
                <p className="text-yellow-600 dark:text-yellow-400">
                  ⚠️ 불량률이 5%를 초과했습니다. 주의가 필요합니다.
                </p>
              )}
              {productionSpeed > 80 && coolingPower < 50 && (
                <p className="text-red-600 dark:text-red-400">
                  ⚡ 고속 생산 중인데 냉각이 부족합니다!
                </p>
              )}
              {productionSpeed < 30 && production > 10 && (
                <p className="text-blue-600 dark:text-blue-400">
                  🐌 생산 속도가 너무 느립니다. 효율성을 고려해보세요.
                </p>
              )}
              {performance.grade === 'S' && production > 0 && defects / production < 0.05 && (
                <p className="text-purple-600 dark:text-purple-400 font-bold">
                  🏆 완벽한 균형입니다! S등급 달성!
                </p>
              )}
              {production === 0 && isRunning && (
                <p className="text-gray-600 dark:text-gray-400">
                  🏭 생산 속도를 높여 제품을 생산해보세요.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          background: white;
          border: 2px solid currentColor;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        input[type="range"]::-moz-range-thumb {
          width: 20px;
          height: 20px;
          background: white;
          border: 2px solid currentColor;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
      `}</style>
    </div>
  )
}