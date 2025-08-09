'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Activity,
  Heart,
  TrendingUp,
  AlertTriangle,
  Monitor,
  Smartphone,
  Wifi,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  BarChart3,
  Users,
  Clock,
  Shield,
  Zap,
  Eye,
  Watch,
  Bell,
  Database
} from 'lucide-react'

export default function PatientMonitoringPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', title: '개요', icon: Monitor },
    { id: 'technologies', title: '모니터링 기술', icon: Activity },
    { id: 'predictive', title: '예측 분석', icon: TrendingUp },
    { id: 'implementation', title: '구현 사례', icon: BarChart3 },
    { id: 'challenges', title: '도전과 미래', icon: AlertTriangle }
  ]

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
                <span>목록으로</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Chapter 6: 환자 모니터링
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-full text-sm font-medium">
                중급
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar Navigation */}
          <aside className="lg:col-span-1">
            <div className="sticky top-24 space-y-2">
              {sections.map((section) => {
                const Icon = section.icon
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                      activeSection === section.id
                        ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-lg'
                        : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{section.title}</span>
                  </button>
                )
              })}
            </div>
          </aside>

          {/* Main Content */}
          <main className="lg:col-span-3 space-y-8">
            {/* Overview Section */}
            {activeSection === 'overview' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    AI 기반 환자 모니터링 시스템
                  </h2>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      AI 환자 모니터링은 실시간 생체 신호 분석, 이상 징후 조기 감지, 
                      악화 예측을 통해 환자 안전을 향상시키고 의료진의 업무 효율을 높입니다.
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 my-8">
                      <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                        <Heart className="w-10 h-10 text-green-600 dark:text-green-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          모니터링 대상
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>바이탈 사인 (HR, BP, SpO2)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>심전도 (ECG/EKG)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>호흡 패턴</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>혈당 수치</span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Activity className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          AI 기능
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>패턴 인식</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>이상 감지</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>예측 분석</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>알림 최적화</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 my-8">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        임상적 효과
                      </h3>
                      <div className="grid grid-cols-3 gap-4 mt-6">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-green-600 dark:text-green-400">35%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">급성 악화 감소</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">6시간</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">조기 감지</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">50%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">알람 피로도 감소</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Technologies Section */}
            {activeSection === 'technologies' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    모니터링 기술
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Wearable Devices */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white flex items-center gap-2">
                        <Watch className="w-6 h-6 text-green-600 dark:text-green-400" />
                        웨어러블 디바이스
                      </h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2">스마트워치</h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• 심박수 변이도 (HRV)</li>
                            <li>• 심전도 (ECG)</li>
                            <li>• 산소포화도 (SpO2)</li>
                            <li>• 활동량 및 수면 패턴</li>
                          </ul>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2">패치형 센서</h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• 연속 혈당 모니터링 (CGM)</li>
                            <li>• 체온 모니터링</li>
                            <li>• 호흡수 측정</li>
                            <li>• 자세 및 낙상 감지</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Remote Monitoring */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white flex items-center gap-2">
                        <Wifi className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                        원격 모니터링 시스템
                      </h3>
                      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                        <div className="space-y-4">
                          <div className="flex items-start gap-3">
                            <Smartphone className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-1" />
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-white">모바일 헬스</h4>
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                스마트폰 앱을 통한 증상 기록, 약물 복용 관리, 의료진과 실시간 소통
                              </p>
                            </div>
                          </div>
                          <div className="flex items-start gap-3">
                            <Monitor className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-1" />
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-white">텔레메트리</h4>
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                병원 내 무선 모니터링, 중앙 모니터링 스테이션, 실시간 알람 시스템
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* AI Models */}
                    <div>
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        AI 모델 아키텍처
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python text-gray-700 dark:text-gray-300">{`# 실시간 환자 모니터링 AI 모델
import numpy as np
from tensorflow.keras import layers, models

class PatientMonitoringAI:
    def __init__(self):
        self.model = self.build_lstm_model()
        self.alert_threshold = 0.8
        
    def build_lstm_model(self):
        """시계열 예측을 위한 LSTM 모델"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(None, 7)),  # 7개 바이탈 사인
            layers.LSTM(64, return_sequences=True),
            layers.Attention(),
            layers.LSTM(32),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 위험도 점수
        ])
        return model
    
    def predict_deterioration(self, vital_signs):
        """환자 악화 예측"""
        # 데이터 전처리
        normalized = self.normalize_vitals(vital_signs)
        
        # 예측 수행
        risk_score = self.model.predict(normalized)
        
        if risk_score > self.alert_threshold:
            return {
                'alert': True,
                'risk_level': 'HIGH',
                'score': float(risk_score),
                'action': 'Immediate medical attention required'
            }
        return {'alert': False, 'risk_level': 'LOW'}`}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Predictive Analytics Section */}
            {activeSection === 'predictive' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    예측 분석과 조기 경보
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Early Warning Systems */}
                    <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                        <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
                        조기 경보 시스템
                      </h3>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="bg-white dark:bg-gray-800 rounded p-4">
                          <h4 className="font-medium text-red-600 dark:text-red-400 mb-2">
                            패혈증 예측
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            6-12시간 전 예측<br/>
                            정확도: 92%
                          </p>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded p-4">
                          <h4 className="font-medium text-orange-600 dark:text-orange-400 mb-2">
                            심정지 예측
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            4-6시간 전 예측<br/>
                            민감도: 86%
                          </p>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded p-4">
                          <h4 className="font-medium text-yellow-600 dark:text-yellow-400 mb-2">
                            호흡부전 예측
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            8-12시간 전 예측<br/>
                            특이도: 94%
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Alert Management */}
                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white flex items-center gap-2">
                        <Bell className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                        스마트 알림 관리
                      </h3>
                      <div className="space-y-3">
                        <div className="flex items-center gap-3">
                          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                          <span className="text-gray-700 dark:text-gray-300">
                            <strong>긴급</strong>: 즉시 대응 필요 (생명 위협)
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                          <span className="text-gray-700 dark:text-gray-300">
                            <strong>높음</strong>: 30분 내 확인 필요
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                          <span className="text-gray-700 dark:text-gray-300">
                            <strong>중간</strong>: 1시간 내 검토
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                          <span className="text-gray-700 dark:text-gray-300">
                            <strong>낮음</strong>: 정기 확인
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Predictive Models */}
                    <div>
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        예측 모델 성능
                      </h3>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-gray-200 dark:border-gray-700">
                              <th className="text-left py-3 px-4">예측 대상</th>
                              <th className="text-center py-3 px-4">예측 시간</th>
                              <th className="text-center py-3 px-4">AUC</th>
                              <th className="text-center py-3 px-4">민감도</th>
                              <th className="text-center py-3 px-4">특이도</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="border-b border-gray-100 dark:border-gray-800">
                              <td className="py-3 px-4">급성 신부전</td>
                              <td className="text-center py-3 px-4">48시간</td>
                              <td className="text-center py-3 px-4 font-mono text-green-600">0.92</td>
                              <td className="text-center py-3 px-4 font-mono">87%</td>
                              <td className="text-center py-3 px-4 font-mono">91%</td>
                            </tr>
                            <tr className="border-b border-gray-100 dark:border-gray-800">
                              <td className="py-3 px-4">재입원 위험</td>
                              <td className="text-center py-3 px-4">30일</td>
                              <td className="text-center py-3 px-4 font-mono text-blue-600">0.88</td>
                              <td className="text-center py-3 px-4 font-mono">82%</td>
                              <td className="text-center py-3 px-4 font-mono">85%</td>
                            </tr>
                            <tr className="border-b border-gray-100 dark:border-gray-800">
                              <td className="py-3 px-4">낙상 위험</td>
                              <td className="text-center py-3 px-4">24시간</td>
                              <td className="text-center py-3 px-4 font-mono text-purple-600">0.85</td>
                              <td className="text-center py-3 px-4 font-mono">79%</td>
                              <td className="text-center py-3 px-4 font-mono">88%</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Implementation Section */}
            {activeSection === 'implementation' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    구현 사례
                  </h2>
                  
                  <div className="space-y-6">
                    {/* Case 1 */}
                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
                          <Heart className="w-8 h-8 text-green-600 dark:text-green-400" />
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                            Philips eICU - 원격 중환자실
                          </h3>
                          <p className="text-gray-600 dark:text-gray-400 mb-4">
                            AI 기반 24시간 원격 중환자 모니터링 시스템
                          </p>
                          <div className="grid md:grid-cols-3 gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded">
                            <div>
                              <div className="text-2xl font-bold text-green-600">35%</div>
                              <div className="text-sm text-gray-500">ICU 사망률 감소</div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-blue-600">20%</div>
                              <div className="text-sm text-gray-500">재원 기간 단축</div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-purple-600">400+</div>
                              <div className="text-sm text-gray-500">병원 도입</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Case 2 */}
                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                          <Activity className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                            Current Health - 심부전 관리
                          </h3>
                          <p className="text-gray-600 dark:text-gray-400 mb-4">
                            웨어러블 기반 만성 심부전 환자 원격 모니터링
                          </p>
                          <div className="grid md:grid-cols-3 gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded">
                            <div>
                              <div className="text-2xl font-bold text-blue-600">87%</div>
                              <div className="text-sm text-gray-500">악화 조기 감지</div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-green-600">10일</div>
                              <div className="text-sm text-gray-500">평균 조기 감지</div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-orange-600">58%</div>
                              <div className="text-sm text-gray-500">입원 감소</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Challenges Section */}
            {activeSection === 'challenges' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    도전 과제와 미래
                  </h2>
                  
                  <div className="space-y-6">
                    {/* Challenges */}
                    <div>
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        현재 도전 과제
                      </h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                          <h4 className="font-medium text-red-600 dark:text-red-400 mb-2 flex items-center gap-2">
                            <AlertCircle className="w-5 h-5" />
                            기술적 과제
                          </h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• 알람 피로도 (Alert Fatigue)</li>
                            <li>• 거짓 양성 알람 최소화</li>
                            <li>• 데이터 통합 및 상호운용성</li>
                            <li>• 실시간 처리 지연</li>
                          </ul>
                        </div>
                        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                          <h4 className="font-medium text-yellow-600 dark:text-yellow-400 mb-2 flex items-center gap-2">
                            <Shield className="w-5 h-5" />
                            운영적 과제
                          </h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• 의료진 교육 및 적응</li>
                            <li>• 워크플로우 통합</li>
                            <li>• 비용 효과성 입증</li>
                            <li>• 환자 프라이버시</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Future Directions */}
                    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        미래 발전 방향
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-start gap-3">
                          <Zap className="w-5 h-5 text-indigo-600 dark:text-indigo-400 mt-1" />
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">
                              멀티모달 AI
                            </h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              영상, 텍스트, 시계열 데이터를 통합한 종합적 분석
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Eye className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-1" />
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">
                              설명 가능한 AI
                            </h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              의료진이 이해할 수 있는 예측 근거 제시
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Database className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-1" />
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">
                              연합 학습
                            </h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              환자 데이터 프라이버시를 보호하면서 모델 개선
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Navigation */}
            <div className="flex justify-between items-center pt-8">
              <Link
                href="/medical-ai/chapter/genomics"
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                이전 챕터
              </Link>
              <Link
                href="/medical-ai/chapter/medical-data"
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-pink-600 text-white rounded-lg hover:shadow-lg transition-all"
              >
                다음 챕터
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}