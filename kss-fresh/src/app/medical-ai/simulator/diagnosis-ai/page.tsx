'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  Brain,
  Heart,
  Activity,
  AlertTriangle,
  ChevronRight,
  ArrowLeft,
  Play,
  RotateCcw,
  Info,
  TrendingUp,
  FileText,
  Stethoscope,
  Thermometer,
  Clock,
  User
} from 'lucide-react'

interface Symptom {
  id: string
  name: string
  severity: number
  duration: string
}

interface VitalSign {
  name: string
  value: number
  unit: string
  normal: string
  status: 'normal' | 'warning' | 'critical'
}

interface Diagnosis {
  disease: string
  probability: number
  confidence: number
  evidence: string[]
  recommendations: string[]
  urgency: 'low' | 'medium' | 'high' | 'critical'
}

export default function DiagnosisAISimulator() {
  const [selectedSymptoms, setSelectedSymptoms] = useState<Symptom[]>([])
  const [vitalSigns, setVitalSigns] = useState<VitalSign[]>([
    { name: '체온', value: 36.5, unit: '°C', normal: '36.0-37.5', status: 'normal' },
    { name: '심박수', value: 72, unit: 'bpm', normal: '60-100', status: 'normal' },
    { name: '혈압(수축기)', value: 120, unit: 'mmHg', normal: '90-140', status: 'normal' },
    { name: '혈압(이완기)', value: 80, unit: 'mmHg', normal: '60-90', status: 'normal' },
    { name: '호흡수', value: 16, unit: '/min', normal: '12-20', status: 'normal' },
    { name: '산소포화도', value: 98, unit: '%', normal: '95-100', status: 'normal' }
  ])
  const [patientInfo, setPatientInfo] = useState({
    age: 35,
    gender: 'male',
    history: [] as string[]
  })
  const [diagnoses, setDiagnoses] = useState<Diagnosis[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showDetails, setShowDetails] = useState<string | null>(null)

  const symptomOptions = [
    { id: 'fever', name: '발열', category: '일반' },
    { id: 'cough', name: '기침', category: '호흡기' },
    { id: 'headache', name: '두통', category: '신경계' },
    { id: 'fatigue', name: '피로', category: '일반' },
    { id: 'chest_pain', name: '흉통', category: '심혈관' },
    { id: 'shortness_breath', name: '호흡곤란', category: '호흡기' },
    { id: 'nausea', name: '구토/메스꺼움', category: '소화기' },
    { id: 'dizziness', name: '어지러움', category: '신경계' },
    { id: 'abdominal_pain', name: '복통', category: '소화기' },
    { id: 'joint_pain', name: '관절통', category: '근골격계' }
  ]

  const medicalHistory = [
    '고혈압',
    '당뇨병',
    '천식',
    '알레르기',
    '심장질환',
    '암 병력',
    '수술 이력',
    '가족력'
  ]

  const addSymptom = (symptomId: string) => {
    const symptom = symptomOptions.find(s => s.id === symptomId)
    if (symptom && !selectedSymptoms.find(s => s.id === symptomId)) {
      setSelectedSymptoms([...selectedSymptoms, {
        id: symptomId,
        name: symptom.name,
        severity: 5,
        duration: '1일'
      }])
    }
  }

  const removeSymptom = (symptomId: string) => {
    setSelectedSymptoms(selectedSymptoms.filter(s => s.id !== symptomId))
  }

  const updateSymptomSeverity = (symptomId: string, severity: number) => {
    setSelectedSymptoms(selectedSymptoms.map(s => 
      s.id === symptomId ? { ...s, severity } : s
    ))
  }

  const updateVitalSign = (index: number, value: number) => {
    const newVitalSigns = [...vitalSigns]
    const vital = newVitalSigns[index]
    vital.value = value
    
    // Update status based on normal range
    const [min, max] = vital.normal.split('-').map(v => parseFloat(v))
    if (value < min || value > max) {
      vital.status = Math.abs(value - min) > 10 || Math.abs(value - max) > 10 ? 'critical' : 'warning'
    } else {
      vital.status = 'normal'
    }
    
    setVitalSigns(newVitalSigns)
  }

  const runDiagnosis = () => {
    setIsAnalyzing(true)
    
    // Simulate AI analysis
    setTimeout(() => {
      const mockDiagnoses: Diagnosis[] = []
      
      // Rule-based diagnosis simulation
      if (selectedSymptoms.find(s => s.id === 'fever') && selectedSymptoms.find(s => s.id === 'cough')) {
        mockDiagnoses.push({
          disease: '급성 상기도 감염',
          probability: 0.75,
          confidence: 0.82,
          evidence: ['발열', '기침', '정상 범위 생체 신호'],
          recommendations: [
            '충분한 휴식과 수분 섭취',
            '해열제 복용 고려',
            '증상 악화 시 의료진 상담'
          ],
          urgency: 'low'
        })
        
        if (selectedSymptoms.find(s => s.id === 'shortness_breath')) {
          mockDiagnoses.push({
            disease: '폐렴',
            probability: 0.45,
            confidence: 0.68,
            evidence: ['발열', '기침', '호흡곤란'],
            recommendations: [
              '흉부 X-ray 검사 권장',
              '혈액 검사 필요',
              '즉시 의료기관 방문'
            ],
            urgency: 'high'
          })
        }
      }
      
      if (selectedSymptoms.find(s => s.id === 'chest_pain')) {
        const urgency = vitalSigns.find(v => v.name === '혈압(수축기)')?.value! > 140 ? 'critical' : 'high'
        mockDiagnoses.push({
          disease: '급성 관상동맥 증후군',
          probability: 0.35,
          confidence: 0.72,
          evidence: ['흉통', patientInfo.age > 40 ? '연령' : '', patientInfo.history.includes('심장질환') ? '심장질환 병력' : ''].filter(Boolean),
          recommendations: [
            '즉시 응급실 방문',
            'ECG 검사 필수',
            '심장 효소 검사',
            '니트로글리세린 투여 고려'
          ],
          urgency
        })
      }
      
      if (selectedSymptoms.find(s => s.id === 'headache') && selectedSymptoms.find(s => s.id === 'dizziness')) {
        mockDiagnoses.push({
          disease: '편두통',
          probability: 0.60,
          confidence: 0.75,
          evidence: ['두통', '어지러움', '정상 혈압'],
          recommendations: [
            '조용하고 어두운 곳에서 휴식',
            '진통제 복용',
            '유발 요인 파악',
            '신경과 상담 고려'
          ],
          urgency: 'medium'
        })
      }
      
      // Default if no specific patterns
      if (mockDiagnoses.length === 0 && selectedSymptoms.length > 0) {
        mockDiagnoses.push({
          disease: '추가 검사 필요',
          probability: 0.00,
          confidence: 0.50,
          evidence: selectedSymptoms.map(s => s.name),
          recommendations: [
            '종합 건강 검진 권장',
            '증상 일지 작성',
            '1주일 후 재평가'
          ],
          urgency: 'low'
        })
      }
      
      setDiagnoses(mockDiagnoses.sort((a, b) => b.probability - a.probability))
      setIsAnalyzing(false)
    }, 2000)
  }

  const reset = () => {
    setSelectedSymptoms([])
    setVitalSigns([
      { name: '체온', value: 36.5, unit: '°C', normal: '36.0-37.5', status: 'normal' },
      { name: '심박수', value: 72, unit: 'bpm', normal: '60-100', status: 'normal' },
      { name: '혈압(수축기)', value: 120, unit: 'mmHg', normal: '90-140', status: 'normal' },
      { name: '혈압(이완기)', value: 80, unit: 'mmHg', normal: '60-90', status: 'normal' },
      { name: '호흡수', value: 16, unit: '/min', normal: '12-20', status: 'normal' },
      { name: '산소포화도', value: 98, unit: '%', normal: '95-100', status: 'normal' }
    ])
    setPatientInfo({ age: 35, gender: 'male', history: [] })
    setDiagnoses([])
  }

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical': return 'text-red-600 dark:text-red-400'
      case 'high': return 'text-orange-600 dark:text-orange-400'
      case 'medium': return 'text-yellow-600 dark:text-yellow-400'
      case 'low': return 'text-green-600 dark:text-green-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'critical': return 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
      case 'warning': return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400'
      case 'normal': return 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
      default: return 'bg-gray-100 dark:bg-gray-900/30 text-gray-600 dark:text-gray-400'
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
                AI 진단 보조 시뮬레이터
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={reset}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                초기화
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Panel - Patient Info & Symptoms */}
          <div className="lg:col-span-1 space-y-6">
            {/* Patient Information */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <User className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                환자 정보
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    나이
                  </label>
                  <input
                    type="number"
                    value={patientInfo.age}
                    onChange={(e) => setPatientInfo({ ...patientInfo, age: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    성별
                  </label>
                  <select
                    value={patientInfo.gender}
                    onChange={(e) => setPatientInfo({ ...patientInfo, gender: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="male">남성</option>
                    <option value="female">여성</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    병력
                  </label>
                  <div className="space-y-2">
                    {medicalHistory.map((history) => (
                      <label key={history} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={patientInfo.history.includes(history)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setPatientInfo({ ...patientInfo, history: [...patientInfo.history, history] })
                            } else {
                              setPatientInfo({ ...patientInfo, history: patientInfo.history.filter(h => h !== history) })
                            }
                          }}
                          className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-600 dark:text-gray-400">{history}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Symptom Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <Stethoscope className="w-5 h-5 text-green-600 dark:text-green-400" />
                증상 선택
              </h3>
              <div className="space-y-2">
                {symptomOptions.map((symptom) => (
                  <button
                    key={symptom.id}
                    onClick={() => addSymptom(symptom.id)}
                    disabled={selectedSymptoms.find(s => s.id === symptom.id) !== undefined}
                    className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                      selectedSymptoms.find(s => s.id === symptom.id)
                        ? 'bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                        : 'bg-gray-50 dark:bg-gray-700 hover:bg-blue-50 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm">{symptom.name}</span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">{symptom.category}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Middle Panel - Selected Symptoms & Vital Signs */}
          <div className="lg:col-span-1 space-y-6">
            {/* Selected Symptoms */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <Activity className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                선택된 증상
              </h3>
              {selectedSymptoms.length === 0 ? (
                <p className="text-sm text-gray-500 dark:text-gray-400">증상을 선택해주세요</p>
              ) : (
                <div className="space-y-3">
                  {selectedSymptoms.map((symptom) => (
                    <div key={symptom.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-gray-900 dark:text-white">{symptom.name}</span>
                        <button
                          onClick={() => removeSymptom(symptom.id)}
                          className="text-red-500 hover:text-red-600 text-sm"
                        >
                          제거
                        </button>
                      </div>
                      <div className="space-y-2">
                        <div>
                          <label className="text-xs text-gray-500 dark:text-gray-400">강도</label>
                          <input
                            type="range"
                            min="1"
                            max="10"
                            value={symptom.severity}
                            onChange={(e) => updateSymptomSeverity(symptom.id, parseInt(e.target.value))}
                            className="w-full"
                          />
                          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                            <span>약함</span>
                            <span>{symptom.severity}/10</span>
                            <span>심함</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Vital Signs */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <Thermometer className="w-5 h-5 text-red-600 dark:text-red-400" />
                생체 신호
              </h3>
              <div className="space-y-3">
                {vitalSigns.map((vital, index) => (
                  <div key={vital.name} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {vital.name}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded-full ${getStatusColor(vital.status)}`}>
                        {vital.status === 'normal' ? '정상' : vital.status === 'warning' ? '주의' : '위험'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        value={vital.value}
                        onChange={(e) => updateVitalSign(index, parseFloat(e.target.value) || 0)}
                        className="flex-1 px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-sm"
                        step="0.1"
                      />
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        {vital.unit}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      정상: {vital.normal}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Analyze Button */}
            <button
              onClick={runDiagnosis}
              disabled={selectedSymptoms.length === 0 || isAnalyzing}
              className={`w-full py-3 rounded-lg font-medium transition-all ${
                selectedSymptoms.length === 0 || isAnalyzing
                  ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-lg'
              }`}
            >
              {isAnalyzing ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  AI 분석 중...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <Brain className="w-5 h-5" />
                  AI 진단 실행
                </span>
              )}
            </button>
          </div>

          {/* Right Panel - Diagnosis Results */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
                <FileText className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                진단 결과
              </h3>
              
              {diagnoses.length === 0 ? (
                <div className="text-center py-12">
                  <Brain className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-500 dark:text-gray-400">
                    증상을 입력하고 AI 진단을 실행하세요
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {diagnoses.map((diagnosis, index) => (
                    <div
                      key={index}
                      className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
                    >
                      <div
                        className="p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                        onClick={() => setShowDetails(showDetails === diagnosis.disease ? null : diagnosis.disease)}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <h4 className="font-semibold text-gray-900 dark:text-white">
                            {diagnosis.disease}
                          </h4>
                          <span className={`text-sm font-medium ${getUrgencyColor(diagnosis.urgency)}`}>
                            {diagnosis.urgency === 'critical' ? '위급' :
                             diagnosis.urgency === 'high' ? '높음' :
                             diagnosis.urgency === 'medium' ? '보통' : '낮음'}
                          </span>
                        </div>
                        
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-500 dark:text-gray-400">가능성</span>
                            <span className="text-sm font-medium text-gray-900 dark:text-white">
                              {(diagnosis.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500"
                              style={{ width: `${diagnosis.probability * 100}%` }}
                            ></div>
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between mt-2">
                          <span className="text-sm text-gray-500 dark:text-gray-400">신뢰도</span>
                          <span className="text-sm text-gray-900 dark:text-white">
                            {(diagnosis.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      
                      {showDetails === diagnosis.disease && (
                        <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-700/50">
                          <div className="space-y-3">
                            <div>
                              <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                근거
                              </h5>
                              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                                {diagnosis.evidence.map((ev, i) => (
                                  <li key={i} className="flex items-start gap-2">
                                    <ChevronRight className="w-4 h-4 mt-0.5" />
                                    <span>{ev}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                            
                            <div>
                              <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                권장사항
                              </h5>
                              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                                {diagnosis.recommendations.map((rec, i) => (
                                  <li key={i} className="flex items-start gap-2">
                                    <Info className="w-4 h-4 mt-0.5 text-blue-500" />
                                    <span>{rec}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                  
                  <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                      <div className="text-sm text-yellow-800 dark:text-yellow-200">
                        <p className="font-semibold mb-1">주의사항</p>
                        <p>이 결과는 AI 시뮬레이션이며 실제 의학적 진단이 아닙니다. 
                        실제 증상이 있으시면 반드시 의료 전문가와 상담하세요.</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}