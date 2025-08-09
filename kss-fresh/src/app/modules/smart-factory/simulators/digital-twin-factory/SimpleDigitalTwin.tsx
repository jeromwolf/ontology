'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Pause, Info, Zap, Thermometer, Activity, AlertTriangle, CheckCircle } from 'lucide-react'

interface SimpleDigitalTwinProps {
  backUrl?: string
}

export default function SimpleDigitalTwin({ backUrl = '/modules/smart-factory' }: SimpleDigitalTwinProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [showTutorial, setShowTutorial] = useState(true)
  const [temperature, setTemperature] = useState(25)
  const [production, setProduction] = useState(0)
  const [efficiency, setEfficiency] = useState(100)
  const [defects, setDefects] = useState(0)

  // 시뮬레이션 실행
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      // 온도 상승
      setTemperature(prev => {
        const newTemp = prev + (Math.random() * 2 - 0.5)
        return Math.min(80, Math.max(20, newTemp))
      })

      // 생산량 증가
      setProduction(prev => prev + Math.floor(Math.random() * 5 + 3))

      // 효율성 변화 (온도가 높으면 효율 감소)
      setEfficiency(prev => {
        if (temperature > 50) return Math.max(60, prev - 1)
        return Math.min(100, prev + 0.5)
      })

      // 불량품 (온도가 높으면 불량 증가)
      if (temperature > 60 && Math.random() > 0.5) {
        setDefects(prev => prev + 1)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [isRunning, temperature])

  const getMachineStatus = () => {
    if (temperature > 60) return { status: 'danger', text: '과열! 냉각 필요 🔥' }
    if (temperature > 45) return { status: 'warning', text: '주의 필요 ⚠️' }
    if (efficiency < 70) return { status: 'warning', text: '효율 저하 📉' }
    return { status: 'good', text: '정상 작동 중 ✅' }
  }

  const status = getMachineStatus()

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      {/* 헤더 */}
      <div className="max-w-7xl mx-auto mb-6">
        <Link 
          href={backUrl}
          className="inline-flex items-center gap-2 text-amber-600 hover:text-amber-700 mb-4"
        >
          <ArrowLeft className="w-5 h-5" />
          학습 페이지로 돌아가기
        </Link>
        
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          디지털 트윈 체험하기
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          실제 공장과 똑같은 가상 공장을 만들어 미리 테스트해보세요!
        </p>
      </div>

      {/* 실시간 상태 표시 - 상단으로 이동 */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-500" />
            실시간 공장 상태
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <Thermometer className={`w-8 h-8 mx-auto mb-1 ${temperature > 60 ? 'text-red-500' : temperature > 45 ? 'text-yellow-500' : 'text-green-500'}`} />
              <div className="text-2xl font-bold">{temperature.toFixed(1)}°C</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">온도</div>
            </div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <Activity className="w-8 h-8 mx-auto mb-1 text-blue-500" />
              <div className="text-2xl font-bold">{production}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">생산량</div>
            </div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <Zap className={`w-8 h-8 mx-auto mb-1 ${efficiency > 80 ? 'text-green-500' : efficiency > 60 ? 'text-yellow-500' : 'text-red-500'}`} />
              <div className="text-2xl font-bold">{efficiency.toFixed(0)}%</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">효율성</div>
            </div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <AlertTriangle className="w-8 h-8 mx-auto mb-1 text-red-500" />
              <div className="text-2xl font-bold text-red-600">{defects}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">불량품</div>
            </div>
          </div>
        </div>
      </div>

      {/* 튜토리얼 */}
      {showTutorial && (
        <div className="max-w-6xl mx-auto mb-8 bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-800">
          <div className="flex items-start gap-3">
            <Info className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                🎯 디지털 트윈이란?
              </h3>
              <p className="text-blue-800 dark:text-blue-200 mb-3">
                실제 공장 기계와 똑같이 움직이는 <strong>가상의 쌍둥이</strong>입니다!
                실제로 만들기 전에 가상으로 테스트해서 문제를 미리 발견할 수 있어요.
              </p>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <p className="font-medium mb-2">👉 이렇게 해보세요:</p>
                <ol className="list-decimal list-inside space-y-1 text-sm">
                  <li>시작 버튼을 눌러 기계를 작동시키세요</li>
                  <li>온도가 올라가는 것을 확인하세요</li>
                  <li>온도가 60도를 넘으면 불량품이 나오기 시작합니다!</li>
                  <li>일시정지를 눌러 기계를 식혀보세요</li>
                </ol>
              </div>
              <button
                onClick={() => setShowTutorial(false)}
                className="mt-3 text-sm text-blue-600 hover:text-blue-700"
              >
                닫기
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* 왼쪽: 가상 공장 (디지털 트윈) */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <span className="text-blue-600">🏭 가상 공장</span>
            <span className="text-sm font-normal text-gray-500">(디지털 트윈)</span>
          </h2>

          {/* 시각적 표현 - 더 명확한 애니메이션 */}
          <div className="relative h-80 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl mb-6 overflow-hidden">
            {/* 컨베이어 벨트 */}
            <div className="absolute bottom-0 left-0 right-0 h-24 bg-gray-300 dark:bg-gray-600">
              <div className="h-full flex items-center overflow-hidden">
                {[...Array(10)].map((_, i) => (
                  <div
                    key={i}
                    className={`flex-shrink-0 w-16 h-16 mx-4 rounded-lg flex items-center justify-center text-2xl
                      ${isRunning ? 'animate-slide' : ''}
                    `}
                    style={{
                      backgroundColor: i % 3 === 0 && temperature > 60 ? '#ef4444' : '#3b82f6',
                      animationDelay: `${i * 0.5}s`
                    }}
                  >
                    📦
                  </div>
                ))}
              </div>
            </div>

            {/* 기계 본체 - 온도에 따라 색상 변화 */}
            <div className="absolute top-10 left-1/2 transform -translate-x-1/2">
              <div className={`
                w-48 h-32 rounded-lg flex items-center justify-center relative
                transition-all duration-1000
                ${status.status === 'danger' ? 'bg-red-500' : 
                  status.status === 'warning' ? 'bg-yellow-500' : 'bg-green-500'}
              `}>
                <span className="text-4xl">🏭</span>
                
                {/* 작동 중 표시 - 기어 회전 */}
                {isRunning && (
                  <div className="absolute -top-8 right-0">
                    <div className="animate-spin text-3xl">⚙️</div>
                  </div>
                )}
                
                {/* 연기 효과 (온도 높을 때) */}
                {temperature > 50 && isRunning && (
                  <div className="absolute -top-10 left-1/2 transform -translate-x-1/2">
                    <div className="animate-bounce text-2xl opacity-70">💨</div>
                  </div>
                )}
              </div>
            </div>

            {/* 상태 표시 */}
            <div className="absolute top-4 left-4 right-4 flex justify-between">
              <div className="bg-white dark:bg-gray-900 px-3 py-2 rounded-lg shadow">
                <div className="text-sm font-medium">{status.text}</div>
              </div>
              {isRunning && (
                <div className="bg-green-500 px-3 py-2 rounded-lg shadow animate-pulse">
                  <div className="text-sm font-medium text-white">작동 중</div>
                </div>
              )}
            </div>
          </div>

          <style jsx>{`
            @keyframes slide {
              0% { transform: translateX(-100px); }
              100% { transform: translateX(800px); }
            }
            .animate-slide {
              animation: slide 8s linear infinite;
            }
          `}</style>

          {/* 컨트롤 */}
          <div className="flex gap-3 justify-center">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                isRunning 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              {isRunning ? '일시정지' : '시작'}
            </button>
            <button
              onClick={() => {
                setIsRunning(false)
                setTemperature(25)
                setProduction(0)
                setEfficiency(100)
                setDefects(0)
              }}
              className="flex items-center gap-2 px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium"
            >
              리셋
            </button>
          </div>
        </div>

        {/* 오른쪽: 실시간 데이터 */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
          <h2 className="text-xl font-bold mb-4">📊 실시간 모니터링</h2>

          <div className="space-y-4">
            {/* 온도 */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="flex items-center gap-2">
                  <Thermometer className="w-5 h-5 text-orange-500" />
                  온도
                </span>
                <span className="font-bold text-lg">{temperature.toFixed(1)}°C</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-3">
                <div 
                  className={`h-3 rounded-full transition-all duration-500 ${
                    temperature > 60 ? 'bg-red-500' : 
                    temperature > 45 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${(temperature / 80) * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>20°C</span>
                <span>80°C</span>
              </div>
            </div>

            {/* 생산량 */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-500" />
                  총 생산량
                </span>
                <span className="font-bold text-lg">{production} 개</span>
              </div>
            </div>

            {/* 효율성 */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-green-500" />
                  효율성
                </span>
                <span className="font-bold text-lg">{efficiency.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-3">
                <div 
                  className={`h-3 rounded-full transition-all duration-500 ${
                    efficiency > 80 ? 'bg-green-500' : 
                    efficiency > 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${efficiency}%` }}
                />
              </div>
            </div>

            {/* 불량품 */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                  불량품
                </span>
                <span className="font-bold text-lg text-red-600">{defects} 개</span>
              </div>
              {defects > 0 && (
                <p className="text-sm text-red-600 mt-2">
                  ⚠️ 온도가 너무 높아 불량품이 발생했습니다!
                </p>
              )}
            </div>
          </div>

          {/* 인사이트 */}
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h3 className="font-semibold mb-2">💡 디지털 트윈의 가치</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              지금처럼 가상으로 테스트해보니, <strong>온도가 60도를 넘으면 불량품이 발생</strong>한다는 것을 발견했네요! 
              실제 공장에서는 온도 관리 시스템을 강화해야겠습니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}