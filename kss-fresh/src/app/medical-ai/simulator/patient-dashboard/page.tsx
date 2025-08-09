'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  Heart,
  Activity,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  ArrowLeft,
  Bell,
  User,
  Calendar,
  Clock,
  Thermometer,
  Droplet,
  Wind,
  Brain,
  Battery,
  Pill,
  FileText,
  ChevronRight,
  AlertCircle,
  CheckCircle
} from 'lucide-react'

interface VitalData {
  time: string
  heartRate: number
  bloodPressureSys: number
  bloodPressureDia: number
  temperature: number
  respRate: number
  spo2: number
}

interface Alert {
  id: string
  type: 'critical' | 'warning' | 'info'
  message: string
  time: string
  parameter?: string
}

interface Medication {
  name: string
  dose: string
  frequency: string
  taken: boolean
  nextDose: string
}

export default function PatientDashboardSimulator() {
  const [currentTime, setCurrentTime] = useState(new Date())
  const [vitalData, setVitalData] = useState<VitalData[]>([])
  const [currentVitals, setCurrentVitals] = useState({
    heartRate: 72,
    bloodPressureSys: 120,
    bloodPressureDia: 80,
    temperature: 36.5,
    respRate: 16,
    spo2: 98
  })
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [riskScore, setRiskScore] = useState(15)
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('6h')
  const [isMonitoring, setIsMonitoring] = useState(true)

  const patientInfo = {
    name: '홍길동',
    age: 65,
    id: 'P-2024-0123',
    diagnosis: '급성 심근경색',
    admissionDate: '2024-01-15',
    room: 'ICU-03',
    doctor: '김철수 교수'
  }

  const medications: Medication[] = [
    { name: '아스피린', dose: '100mg', frequency: '1일 1회', taken: true, nextDose: '09:00' },
    { name: '메토프롤롤', dose: '50mg', frequency: '1일 2회', taken: false, nextDose: '14:00' },
    { name: '리시노프릴', dose: '10mg', frequency: '1일 1회', taken: true, nextDose: '21:00' },
    { name: '아토르바스타틴', dose: '40mg', frequency: '1일 1회', taken: true, nextDose: '22:00' }
  ]

  // Simulate real-time vital signs
  useEffect(() => {
    if (!isMonitoring) return

    const interval = setInterval(() => {
      setCurrentTime(new Date())
      
      // Generate new vital signs with realistic variations
      const newVitals = {
        heartRate: Math.round(currentVitals.heartRate + (Math.random() - 0.5) * 10),
        bloodPressureSys: Math.round(currentVitals.bloodPressureSys + (Math.random() - 0.5) * 8),
        bloodPressureDia: Math.round(currentVitals.bloodPressureDia + (Math.random() - 0.5) * 5),
        temperature: Number((currentVitals.temperature + (Math.random() - 0.5) * 0.2).toFixed(1)),
        respRate: Math.round(currentVitals.respRate + (Math.random() - 0.5) * 3),
        spo2: Math.round(currentVitals.spo2 + (Math.random() - 0.5) * 2)
      }
      
      // Ensure values stay within reasonable ranges
      newVitals.heartRate = Math.max(50, Math.min(150, newVitals.heartRate))
      newVitals.bloodPressureSys = Math.max(80, Math.min(180, newVitals.bloodPressureSys))
      newVitals.bloodPressureDia = Math.max(50, Math.min(110, newVitals.bloodPressureDia))
      newVitals.temperature = Math.max(35, Math.min(40, newVitals.temperature))
      newVitals.respRate = Math.max(10, Math.min(30, newVitals.respRate))
      newVitals.spo2 = Math.max(85, Math.min(100, newVitals.spo2))
      
      setCurrentVitals(newVitals)
      
      // Add to history
      const newDataPoint: VitalData = {
        time: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
        ...newVitals
      }
      
      setVitalData(prev => [...prev.slice(-50), newDataPoint])
      
      // Check for alerts
      checkForAlerts(newVitals)
      
      // Update risk score
      updateRiskScore(newVitals)
    }, 3000) // Update every 3 seconds

    return () => clearInterval(interval)
  }, [isMonitoring, currentVitals])

  const checkForAlerts = (vitals: typeof currentVitals) => {
    const newAlerts: Alert[] = []
    
    if (vitals.heartRate > 100) {
      newAlerts.push({
        id: `hr-${Date.now()}`,
        type: vitals.heartRate > 120 ? 'critical' : 'warning',
        message: `빈맥 감지: ${vitals.heartRate} bpm`,
        time: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
        parameter: 'heartRate'
      })
    }
    
    if (vitals.bloodPressureSys > 140) {
      newAlerts.push({
        id: `bp-${Date.now()}`,
        type: vitals.bloodPressureSys > 160 ? 'critical' : 'warning',
        message: `고혈압: ${vitals.bloodPressureSys}/${vitals.bloodPressureDia} mmHg`,
        time: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
        parameter: 'bloodPressure'
      })
    }
    
    if (vitals.spo2 < 92) {
      newAlerts.push({
        id: `spo2-${Date.now()}`,
        type: vitals.spo2 < 90 ? 'critical' : 'warning',
        message: `저산소증: SpO2 ${vitals.spo2}%`,
        time: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
        parameter: 'spo2'
      })
    }
    
    if (vitals.temperature > 38) {
      newAlerts.push({
        id: `temp-${Date.now()}`,
        type: vitals.temperature > 39 ? 'critical' : 'warning',
        message: `발열: ${vitals.temperature}°C`,
        time: new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
        parameter: 'temperature'
      })
    }
    
    if (newAlerts.length > 0) {
      setAlerts(prev => [...newAlerts, ...prev].slice(0, 10))
    }
  }

  const updateRiskScore = (vitals: typeof currentVitals) => {
    let score = 0
    
    // Heart rate scoring
    if (vitals.heartRate < 60 || vitals.heartRate > 100) score += 10
    if (vitals.heartRate < 50 || vitals.heartRate > 120) score += 20
    
    // Blood pressure scoring
    if (vitals.bloodPressureSys > 140 || vitals.bloodPressureSys < 90) score += 15
    if (vitals.bloodPressureDia > 90 || vitals.bloodPressureDia < 60) score += 10
    
    // SpO2 scoring
    if (vitals.spo2 < 95) score += 10
    if (vitals.spo2 < 90) score += 25
    
    // Temperature scoring
    if (vitals.temperature > 38 || vitals.temperature < 36) score += 10
    
    // Respiration rate scoring
    if (vitals.respRate > 20 || vitals.respRate < 12) score += 10
    
    setRiskScore(Math.min(100, score))
  }

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical': return <AlertTriangle className="w-4 h-4 text-red-500" />
      case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-500" />
      default: return <Bell className="w-4 h-4 text-blue-500" />
    }
  }

  const getRiskColor = (score: number) => {
    if (score < 30) return 'text-green-600 dark:text-green-400'
    if (score < 60) return 'text-yellow-600 dark:text-yellow-400'
    if (score < 80) return 'text-orange-600 dark:text-orange-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getRiskLevel = (score: number) => {
    if (score < 30) return '낮음'
    if (score < 60) return '보통'
    if (score < 80) return '높음'
    return '위급'
  }

  const getVitalStatus = (param: string, value: number) => {
    switch (param) {
      case 'heartRate':
        if (value < 60 || value > 100) return value < 50 || value > 120 ? 'critical' : 'warning'
        return 'normal'
      case 'bloodPressureSys':
        if (value < 90 || value > 140) return value < 80 || value > 160 ? 'critical' : 'warning'
        return 'normal'
      case 'spo2':
        if (value < 95) return value < 90 ? 'critical' : 'warning'
        return 'normal'
      case 'temperature':
        if (value < 36 || value > 37.5) return value < 35 || value > 39 ? 'critical' : 'warning'
        return 'normal'
      default:
        return 'normal'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'critical': return 'text-red-600 dark:text-red-400'
      case 'warning': return 'text-yellow-600 dark:text-yellow-400'
      default: return 'text-green-600 dark:text-green-400'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/medical-ai"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>Medical AI로 돌아가기</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                AI 환자 모니터링 대시보드
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsMonitoring(!isMonitoring)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  isMonitoring
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                }`}
              >
                {isMonitoring ? '모니터링 중' : '일시정지'}
              </button>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                {currentTime.toLocaleString('ko-KR')}
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Patient Info Bar */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 mb-6 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                  <User className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h2 className="font-semibold text-gray-900 dark:text-white">{patientInfo.name}</h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {patientInfo.age}세 • {patientInfo.id}
                  </p>
                </div>
              </div>
              <div className="h-10 w-px bg-gray-200 dark:bg-gray-700"></div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">진단명</p>
                <p className="font-medium text-gray-900 dark:text-white">{patientInfo.diagnosis}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">병실</p>
                <p className="font-medium text-gray-900 dark:text-white">{patientInfo.room}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">담당의</p>
                <p className="font-medium text-gray-900 dark:text-white">{patientInfo.doctor}</p>
              </div>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400">AI 위험도 점수</p>
              <p className={`text-3xl font-bold ${getRiskColor(riskScore)}`}>
                {riskScore}
              </p>
              <p className={`text-sm font-medium ${getRiskColor(riskScore)}`}>
                {getRiskLevel(riskScore)}
              </p>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Vital Signs */}
          <div className="lg:col-span-2 space-y-6">
            {/* Real-time Vitals */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                실시간 생체 신호
              </h3>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <Heart className="w-5 h-5 text-red-500" />
                    <span className={`text-xs font-medium ${getStatusColor(getVitalStatus('heartRate', currentVitals.heartRate))}`}>
                      {getVitalStatus('heartRate', currentVitals.heartRate) === 'normal' ? '정상' : 
                       getVitalStatus('heartRate', currentVitals.heartRate) === 'warning' ? '주의' : '위험'}
                    </span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {currentVitals.heartRate}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">bpm</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <Droplet className="w-5 h-5 text-blue-500" />
                    <span className={`text-xs font-medium ${getStatusColor(getVitalStatus('bloodPressureSys', currentVitals.bloodPressureSys))}`}>
                      {getVitalStatus('bloodPressureSys', currentVitals.bloodPressureSys) === 'normal' ? '정상' : 
                       getVitalStatus('bloodPressureSys', currentVitals.bloodPressureSys) === 'warning' ? '주의' : '위험'}
                    </span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {currentVitals.bloodPressureSys}/{currentVitals.bloodPressureDia}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">mmHg</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <Thermometer className="w-5 h-5 text-orange-500" />
                    <span className={`text-xs font-medium ${getStatusColor(getVitalStatus('temperature', currentVitals.temperature))}`}>
                      {getVitalStatus('temperature', currentVitals.temperature) === 'normal' ? '정상' : 
                       getVitalStatus('temperature', currentVitals.temperature) === 'warning' ? '주의' : '위험'}
                    </span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {currentVitals.temperature}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">°C</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <Wind className="w-5 h-5 text-cyan-500" />
                    <span className="text-xs font-medium text-green-600 dark:text-green-400">정상</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {currentVitals.respRate}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">/min</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <Activity className="w-5 h-5 text-purple-500" />
                    <span className={`text-xs font-medium ${getStatusColor(getVitalStatus('spo2', currentVitals.spo2))}`}>
                      {getVitalStatus('spo2', currentVitals.spo2) === 'normal' ? '정상' : 
                       getVitalStatus('spo2', currentVitals.spo2) === 'warning' ? '주의' : '위험'}
                    </span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {currentVitals.spo2}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">%</div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <Brain className="w-5 h-5 text-indigo-500" />
                    <span className="text-xs font-medium text-green-600 dark:text-green-400">명료</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    15
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">GCS</div>
                </div>
              </div>
            </div>

            {/* Trend Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center gap-2 text-gray-900 dark:text-white">
                  <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
                  트렌드 분석
                </h3>
                <div className="flex gap-2">
                  {(['1h', '6h', '24h', '7d'] as const).map((range) => (
                    <button
                      key={range}
                      onClick={() => setSelectedTimeRange(range)}
                      className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                        selectedTimeRange === range
                          ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                      }`}
                    >
                      {range}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Simple trend visualization */}
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">심박수 추세</span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      평균 {Math.round(vitalData.reduce((acc, d) => acc + d.heartRate, 0) / Math.max(vitalData.length, 1))} bpm
                    </span>
                  </div>
                  <div className="h-20 bg-gray-50 dark:bg-gray-700 rounded-lg p-2 flex items-end gap-1">
                    {vitalData.slice(-20).map((data, index) => (
                      <div
                        key={index}
                        className="flex-1 bg-gradient-to-t from-red-500 to-pink-500 rounded-sm"
                        style={{ height: `${(data.heartRate / 150) * 100}%` }}
                      ></div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">산소포화도 추세</span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      평균 {Math.round(vitalData.reduce((acc, d) => acc + d.spo2, 0) / Math.max(vitalData.length, 1))}%
                    </span>
                  </div>
                  <div className="h-20 bg-gray-50 dark:bg-gray-700 rounded-lg p-2 flex items-end gap-1">
                    {vitalData.slice(-20).map((data, index) => (
                      <div
                        key={index}
                        className="flex-1 bg-gradient-to-t from-blue-500 to-cyan-500 rounded-sm"
                        style={{ height: `${data.spo2}%` }}
                      ></div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Medications */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <Pill className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                투약 관리
              </h3>
              
              <div className="space-y-3">
                {medications.map((med, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                        med.taken 
                          ? 'bg-green-100 dark:bg-green-900/30' 
                          : 'bg-yellow-100 dark:bg-yellow-900/30'
                      }`}>
                        {med.taken ? (
                          <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                        ) : (
                          <Clock className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
                        )}
                      </div>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">{med.name}</p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          {med.dose} • {med.frequency}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-500 dark:text-gray-400">다음 투약</p>
                      <p className="font-medium text-gray-900 dark:text-white">{med.nextDose}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Panel - Alerts & AI Insights */}
          <div className="lg:col-span-1 space-y-6">
            {/* Alerts */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <Bell className="w-5 h-5 text-red-600 dark:text-red-400" />
                실시간 알림
              </h3>
              
              {alerts.length === 0 ? (
                <div className="text-center py-8">
                  <Bell className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
                  <p className="text-gray-500 dark:text-gray-400">현재 알림이 없습니다</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {alerts.map((alert) => (
                    <div
                      key={alert.id}
                      className={`p-3 rounded-lg border ${
                        alert.type === 'critical' 
                          ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                          : alert.type === 'warning'
                          ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
                          : 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        {getAlertIcon(alert.type)}
                        <div className="flex-1">
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            {alert.message}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            {alert.time}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* AI Predictions */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <Brain className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                AI 예측 분석
              </h3>
              
              <div className="space-y-4">
                <div className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                    24시간 예측
                  </h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">악화 가능성</span>
                      <span className="text-sm font-medium text-orange-600 dark:text-orange-400">32%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">재입원 위험</span>
                      <span className="text-sm font-medium text-green-600 dark:text-green-400">12%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">합병증 위험</span>
                      <span className="text-sm font-medium text-yellow-600 dark:text-yellow-400">18%</span>
                    </div>
                  </div>
                </div>
                
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                    권장 조치
                  </h4>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <ChevronRight className="w-4 h-4 mt-0.5 text-blue-500" />
                      <span>심전도 모니터링 강화</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ChevronRight className="w-4 h-4 mt-0.5 text-blue-500" />
                      <span>30분마다 활력징후 체크</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ChevronRight className="w-4 h-4 mt-0.5 text-blue-500" />
                      <span>약물 용량 조절 검토</span>
                    </li>
                  </ul>
                </div>
                
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                    긍정적 지표
                  </h4>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 mt-0.5 text-green-500" />
                      <span>산소포화도 안정적</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 mt-0.5 text-green-500" />
                      <span>의식 수준 명료</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 mt-0.5 text-green-500" />
                      <span>약물 순응도 우수</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Clinical Notes */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <FileText className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                임상 기록
              </h3>
              
              <div className="space-y-3">
                <div className="pb-3 border-b border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-900 dark:text-white">회진 기록</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">08:30</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    환자 상태 안정. 흉통 호전. 심전도 정상 소견.
                  </p>
                </div>
                <div className="pb-3 border-b border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-900 dark:text-white">간호 기록</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">10:15</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    투약 완료. 활력징후 측정. 특이사항 없음.
                  </p>
                </div>
                <div className="pb-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-900 dark:text-white">검사 결과</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">11:00</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Troponin I: 0.02 ng/mL (정상)
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