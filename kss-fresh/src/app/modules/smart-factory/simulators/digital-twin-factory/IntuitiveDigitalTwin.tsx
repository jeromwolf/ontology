'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, RotateCcw, Info, Zap, Thermometer, Activity, AlertTriangle, CheckCircle, Settings, HelpCircle } from 'lucide-react'

interface IntuitiveDigitalTwinProps {
  backUrl?: string
}

export default function IntuitiveDigitalTwin({ backUrl = '/modules/smart-factory' }: IntuitiveDigitalTwinProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [showHelp, setShowHelp] = useState(true)
  const [selectedScenario, setSelectedScenario] = useState('normal')
  
  // 실시간 데이터
  const [temperature, setTemperature] = useState(25)
  const [production, setProduction] = useState(0)
  const [machineHealth, setMachineHealth] = useState(100)
  const [defects, setDefects] = useState(0)
  const [productPosition, setProductPosition] = useState(0)

  // 시나리오별 설정
  const scenarios = {
    normal: {
      name: '정상 운영',
      description: '일반적인 속도로 안전하게 운영',
      tempIncrease: 0.5,
      productionRate: 3,
      defectChance: 0.02
    },
    rush: {
      name: '긴급 주문',
      description: '빠른 생산 (기계가 빨리 뜨거워짐!)',
      tempIncrease: 1.5,
      productionRate: 8,
      defectChance: 0.15
    },
    maintenance: {
      name: '정비 모드',
      description: '느리게 운영하며 기계 점검',
      tempIncrease: -0.3,
      productionRate: 1,
      defectChance: 0  // 정비 모드에서는 불량품 없음
    }
  }

  // 시뮬레이션 로직
  useEffect(() => {
    if (!isRunning) return

    const scenario = scenarios[selectedScenario as keyof typeof scenarios]
    
    const interval = setInterval(() => {
      // 온도 변화
      setTemperature(prev => {
        const newTemp = prev + scenario.tempIncrease + (Math.random() - 0.5)
        return Math.min(80, Math.max(20, newTemp))
      })

      // 생산량 증가
      setProduction(prev => prev + Math.floor(Math.random() * scenario.productionRate + 1))

      // 기계 상태 (온도가 높으면 상태 악화)
      setMachineHealth(prev => {
        if (temperature > 60) return Math.max(0, prev - 2)
        if (temperature > 45) return Math.max(0, prev - 0.5)
        return Math.min(100, prev + 0.1)
      })

      // 불량품 발생 (정비 모드가 아닐 때만)
      if (selectedScenario !== 'maintenance') {
        if (Math.random() < scenario.defectChance || (temperature > 65 && Math.random() < 0.3)) {
          setDefects(prev => prev + 1)
        }
      }

      // 제품 이동 애니메이션
      setProductPosition(prev => (prev + 10) % 100)
    }, 500)

    return () => clearInterval(interval)
  }, [isRunning, temperature, selectedScenario])

  // 장비 상태 판단
  const getMachineStatus = () => {
    if (machineHealth < 30) return { color: 'red', text: '위험! 즉시 정비 필요', emoji: '🚨' }
    if (machineHealth < 60) return { color: 'yellow', text: '주의 필요', emoji: '⚠️' }
    if (temperature > 65) return { color: 'red', text: '과열 위험!', emoji: '🔥' }
    if (temperature > 50) return { color: 'yellow', text: '온도 상승 중', emoji: '🌡️' }
    return { color: 'green', text: '정상 작동', emoji: '✅' }
  }

  const status = getMachineStatus()

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
              디지털 트윈 시뮬레이터
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              가상 공장에서 미리 테스트해보고 최적의 운영 방법을 찾아보세요
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
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                🤔 디지털 트윈이란?
              </h4>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                실제 공장의 <strong>가상 복제본</strong>입니다. 
                실제로 해보기 전에 여기서 먼저 테스트해볼 수 있어요!
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                📊 장비 상태란?
              </h4>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                기계의 <strong>건강 상태</strong>를 %로 표시합니다. 
                온도가 높으면 기계가 손상되어 상태가 나빠져요.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                🔮 What-if 분석이란?
              </h4>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>"만약 ~하면 어떻게 될까?"</strong>를 미리 테스트하는 것입니다. 
                시나리오를 바꿔가며 결과를 예측해보세요!
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowHelp(false)}
            className="mt-3 text-sm text-blue-600 hover:text-blue-700"
          >
            닫기
          </button>
        </div>
      )}

      {/* 실시간 모니터링 대시보드 - 최상단 배치 */}
      <div className="max-w-7xl mx-auto mb-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
          <h3 className="text-lg font-semibold mb-3">📊 실시간 공장 현황</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {/* 장비 상태 */}
            <div className={`p-3 rounded-lg border-2 ${
              status.color === 'red' ? 'border-red-500 bg-red-50 dark:bg-red-900/20' :
              status.color === 'yellow' ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' :
              'border-green-500 bg-green-50 dark:bg-green-900/20'
            }`}>
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">장비 상태</div>
              <div className="text-xl font-bold">{status.emoji} {status.text}</div>
            </div>

            {/* 기계 건강도 */}
            <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">기계 건강도</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${
                      machineHealth > 60 ? 'bg-green-500' : 
                      machineHealth > 30 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${machineHealth}%` }}
                  />
                </div>
                <span className="text-lg font-bold">{machineHealth.toFixed(0)}%</span>
              </div>
            </div>

            {/* 온도 */}
            <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">온도</div>
              <div className="flex items-center gap-2">
                <Thermometer className={`w-5 h-5 ${
                  temperature > 60 ? 'text-red-500' : 
                  temperature > 45 ? 'text-yellow-500' : 'text-green-500'
                }`} />
                <span className="text-lg font-bold">{temperature.toFixed(1)}°C</span>
              </div>
            </div>

            {/* 생산량 */}
            <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">생산량</div>
              <div className="text-lg font-bold text-blue-600">{production} 개</div>
            </div>

            {/* 불량률 */}
            <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">불량품</div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold text-red-600">{defects} 개</span>
                {production > 0 && (
                  <span className="text-sm text-red-500">
                    ({((defects / production) * 100).toFixed(1)}%)
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* 왼쪽: What-if 시나리오 제어 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-lg">
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-purple-600" />
            What-if 시나리오
          </h2>
          
          <div className="mb-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
            <p className="text-sm text-purple-800 dark:text-purple-200">
              💡 <strong>만약 이렇게 운영하면?</strong><br/>
              다양한 상황을 미리 테스트해보세요
            </p>
          </div>

          <div className="space-y-3">
            {Object.entries(scenarios).map(([key, scenario]) => (
              <button
                key={key}
                onClick={() => setSelectedScenario(key)}
                className={`w-full p-3 rounded-lg text-left transition-all ${
                  selectedScenario === key
                    ? 'bg-purple-100 dark:bg-purple-900 border-2 border-purple-500'
                    : 'bg-gray-50 dark:bg-gray-700 border-2 border-transparent hover:border-gray-300'
                }`}
              >
                <div className="font-medium">{scenario.name}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  {scenario.description}
                </div>
              </button>
            ))}
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
                setProductPosition(0)
              }}
              className="w-full flex items-center justify-center gap-2 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium"
            >
              <RotateCcw className="w-5 h-5" />
              리셋
            </button>
          </div>
        </div>

        {/* 중앙: 가상 공장 시각화 */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl p-5 shadow-lg">
          <h2 className="text-lg font-bold mb-4">🏭 가상 공장 (디지털 트윈)</h2>
          
          <div className="relative h-96 bg-gradient-to-b from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl overflow-hidden">
            {/* 공장 기계 */}
            <div className="absolute top-20 left-1/4 transform -translate-x-1/2">
              <div className={`
                w-32 h-24 rounded-lg flex flex-col items-center justify-center
                transition-all duration-500 relative
                ${machineHealth < 30 ? 'bg-red-500' : 
                  machineHealth < 60 ? 'bg-yellow-500' : 'bg-green-500'}
              `}>
                <span className="text-3xl">🏭</span>
                <span className="text-xs font-bold text-white mt-1">
                  {machineHealth.toFixed(0)}%
                </span>
                
                {/* 작동 표시 */}
                {isRunning && (
                  <>
                    <div className="absolute -top-6 animate-spin">
                      <span className="text-2xl">⚙️</span>
                    </div>
                    {temperature > 50 && (
                      <div className="absolute -top-10 animate-bounce">
                        <span className="text-xl">💨</span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* 컨베이어 벨트 */}
            <div className="absolute bottom-20 left-0 right-0 h-16 bg-gray-400 dark:bg-gray-600">
              <div className="h-full flex items-center">
                {/* 움직이는 제품 */}
                {isRunning && (
                  <div
                    className="absolute h-12 w-12 bg-blue-500 rounded flex items-center justify-center text-xl transition-all duration-500"
                    style={{ left: `${productPosition}%` }}
                  >
                    📦
                  </div>
                )}
              </div>
              <div className="text-center text-xs text-gray-600 dark:text-gray-400 mt-1">
                컨베이어 벨트
              </div>
            </div>

            {/* 완성품 저장소 */}
            <div className="absolute top-20 right-1/4 transform translate-x-1/2">
              <div className="w-24 h-24 bg-gray-300 dark:bg-gray-600 rounded-lg flex flex-col items-center justify-center">
                <span className="text-2xl">📦</span>
                <span className="text-sm font-bold">{production}</span>
                <span className="text-xs">완성품</span>
              </div>
            </div>

            {/* 불량품 */}
            {defects > 0 && (
              <div className="absolute bottom-20 right-10">
                <div className="bg-red-500 text-white px-3 py-2 rounded-lg flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  <span className="text-sm">불량 {defects}개</span>
                </div>
              </div>
            )}
          </div>

          {/* 분석 결과 */}
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h3 className="font-semibold mb-2">📈 시뮬레이션 분석</h3>
            {isRunning ? (
              <div className="space-y-2 text-sm">
                {selectedScenario === 'rush' && temperature > 50 && (
                  <p className="text-red-600 dark:text-red-400">
                    ⚠️ 빠른 생산으로 온도가 급상승! 기계 수명이 단축될 수 있습니다.
                  </p>
                )}
                {machineHealth < 50 && (
                  <p className="text-yellow-600 dark:text-yellow-400">
                    🔧 기계 상태가 나빠지고 있습니다. 정비가 필요할 것 같습니다.
                  </p>
                )}
                {defects > production * 0.1 && production > 10 && (
                  <p className="text-red-600 dark:text-red-400">
                    📊 불량률이 10%를 초과했습니다! 품질 관리가 필요합니다.
                  </p>
                )}
                {selectedScenario === 'maintenance' && machineHealth > 90 && (
                  <p className="text-green-600 dark:text-green-400">
                    ✅ 정비 모드가 효과적입니다. 기계 상태가 회복되고 있습니다.
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                시뮬레이션을 시작하면 분석 결과가 여기에 표시됩니다.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}