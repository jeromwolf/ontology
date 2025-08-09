'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Brain,
  Activity,
  Stethoscope,
  FileText,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  Database,
  Cpu,
  BarChart3,
  Target,
  Lightbulb,
  Users,
  Shield,
  Zap,
  GitBranch,
  Layers,
  Clock,
  Heart
} from 'lucide-react'

export default function DiagnosisAssistantPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', title: '개요', icon: Brain },
    { id: 'clinical-decision', title: '임상 의사결정', icon: GitBranch },
    { id: 'ml-models', title: 'ML 모델', icon: Cpu },
    { id: 'integration', title: '시스템 통합', icon: Database },
    { id: 'cases', title: '사례 연구', icon: FileText }
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
                Chapter 3: 진단 보조 시스템
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-full text-sm font-medium">
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
                        ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg'
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
                    AI 기반 진단 보조 시스템
                  </h2>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      진단 보조 AI는 환자의 증상, 병력, 검사 결과를 종합적으로 분석하여
                      의료진의 진단 과정을 지원하는 시스템입니다. 머신러닝과 자연어 처리 기술을
                      활용하여 더 정확하고 신속한 진단을 가능하게 합니다.
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 my-8">
                      <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                        <Stethoscope className="w-10 h-10 text-purple-600 dark:text-purple-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          주요 기능
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>증상 기반 감별 진단</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>위험도 평가 및 예측</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>치료 권고안 제시</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>약물 상호작용 검토</span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Activity className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          핵심 기술
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>의료 지식 그래프</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>임상 NLP</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>예측 모델링</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>설명 가능한 AI</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 my-8">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        진단 보조 AI의 임상적 가치
                      </h3>
                      <div className="grid grid-cols-3 gap-4 mt-6">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">30%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">진단 시간 단축</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">25%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">오진율 감소</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-green-600 dark:text-green-400">40%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">의료비 절감</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Clinical Decision Section */}
            {activeSection === 'clinical-decision' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    임상 의사결정 지원
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Decision Tree */}
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                        <GitBranch className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                        의사결정 트리 프로세스
                      </h3>
                      
                      <div className="space-y-4">
                        <div className="flex items-start gap-4">
                          <div className="w-10 h-10 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">
                            1
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                              데이터 수집
                            </h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              환자 증상, 병력, 검사 결과, 가족력 등 포괄적 데이터 수집
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start gap-4">
                          <div className="w-10 h-10 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">
                            2
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                              패턴 인식
                            </h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              머신러닝을 통한 증상 패턴 분석 및 유사 사례 매칭
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start gap-4">
                          <div className="w-10 h-10 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">
                            3
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                              확률 계산
                            </h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              베이지안 추론을 통한 각 질병별 확률 계산
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start gap-4">
                          <div className="w-10 h-10 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">
                            4
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                              권고안 생성
                            </h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              가이드라인 기반 치료 권고안 및 추가 검사 제안
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Risk Assessment */}
                    <div className="border-l-4 border-red-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        위험도 평가 시스템
                      </h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                            고위험군 식별
                          </h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• 급성 심근경색 위험도</li>
                            <li>• 뇌졸중 발생 예측</li>
                            <li>• 패혈증 조기 감지</li>
                            <li>• 급성 신부전 위험</li>
                          </ul>
                        </div>
                        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                            예후 예측
                          </h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• 30일 재입원율</li>
                            <li>• 치료 반응 예측</li>
                            <li>• 합병증 발생 가능성</li>
                            <li>• 생존율 추정</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Clinical Guidelines */}
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        임상 가이드라인 통합
                      </h3>
                      <div className="space-y-3">
                        <div className="flex items-center gap-3">
                          <Shield className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                          <span className="text-gray-700 dark:text-gray-300">
                            WHO, NIH, ACC/AHA 등 국제 가이드라인 반영
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <Database className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                          <span className="text-gray-700 dark:text-gray-300">
                            최신 임상 연구 및 메타분석 결과 실시간 업데이트
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <Users className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                          <span className="text-gray-700 dark:text-gray-300">
                            다학제 팀 접근법 지원 (MDT 회의)
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ML Models Section */}
            {activeSection === 'ml-models' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    머신러닝 모델 아키텍처
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Model Types */}
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6">
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                          <Cpu className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                          분류 모델
                        </h4>
                        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                          <li>• Random Forest: 다중 질병 분류</li>
                          <li>• XGBoost: 위험도 스코어링</li>
                          <li>• Neural Networks: 복잡한 패턴 인식</li>
                          <li>• SVM: 이진 분류 (양성/음성)</li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                          <Brain className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                          예측 모델
                        </h4>
                        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                          <li>• LSTM: 시계열 예후 예측</li>
                          <li>• Cox Regression: 생존 분석</li>
                          <li>• Bayesian Networks: 인과 추론</li>
                          <li>• GNN: 약물 상호작용 예측</li>
                        </ul>
                      </div>
                    </div>

                    {/* Model Performance */}
                    <div>
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        주요 모델 성능 지표
                      </h3>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-gray-200 dark:border-gray-700">
                              <th className="text-left py-3 px-4 text-gray-900 dark:text-white">모델</th>
                              <th className="text-center py-3 px-4 text-gray-900 dark:text-white">정확도</th>
                              <th className="text-center py-3 px-4 text-gray-900 dark:text-white">민감도</th>
                              <th className="text-center py-3 px-4 text-gray-900 dark:text-white">특이도</th>
                              <th className="text-center py-3 px-4 text-gray-900 dark:text-white">F1-Score</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="border-b border-gray-100 dark:border-gray-800">
                              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">심혈관 질환 예측</td>
                              <td className="text-center py-3 px-4 font-mono text-green-600 dark:text-green-400">92.3%</td>
                              <td className="text-center py-3 px-4 font-mono text-blue-600 dark:text-blue-400">89.7%</td>
                              <td className="text-center py-3 px-4 font-mono text-purple-600 dark:text-purple-400">94.1%</td>
                              <td className="text-center py-3 px-4 font-mono text-orange-600 dark:text-orange-400">0.917</td>
                            </tr>
                            <tr className="border-b border-gray-100 dark:border-gray-800">
                              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">당뇨병 합병증</td>
                              <td className="text-center py-3 px-4 font-mono text-green-600 dark:text-green-400">88.5%</td>
                              <td className="text-center py-3 px-4 font-mono text-blue-600 dark:text-blue-400">85.2%</td>
                              <td className="text-center py-3 px-4 font-mono text-purple-600 dark:text-purple-400">91.3%</td>
                              <td className="text-center py-3 px-4 font-mono text-orange-600 dark:text-orange-400">0.882</td>
                            </tr>
                            <tr className="border-b border-gray-100 dark:border-gray-800">
                              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">패혈증 조기 감지</td>
                              <td className="text-center py-3 px-4 font-mono text-green-600 dark:text-green-400">94.7%</td>
                              <td className="text-center py-3 px-4 font-mono text-blue-600 dark:text-blue-400">96.3%</td>
                              <td className="text-center py-3 px-4 font-mono text-purple-600 dark:text-purple-400">93.2%</td>
                              <td className="text-center py-3 px-4 font-mono text-orange-600 dark:text-orange-400">0.948</td>
                            </tr>
                            <tr>
                              <td className="py-3 px-4 text-gray-700 dark:text-gray-300">재입원 위험 예측</td>
                              <td className="text-center py-3 px-4 font-mono text-green-600 dark:text-green-400">86.2%</td>
                              <td className="text-center py-3 px-4 font-mono text-blue-600 dark:text-blue-400">82.5%</td>
                              <td className="text-center py-3 px-4 font-mono text-purple-600 dark:text-purple-400">89.7%</td>
                              <td className="text-center py-3 px-4 font-mono text-orange-600 dark:text-orange-400">0.859</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Code Example */}
                    <div>
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        진단 예측 모델 예제
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python text-gray-700 dark:text-gray-300">{`# 심혈관 질환 위험도 예측 모델
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class CardiovascularRiskPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def preprocess_features(self, patient_data):
        """환자 데이터 전처리"""
        features = [
            patient_data['age'],
            patient_data['systolic_bp'],
            patient_data['cholesterol'],
            patient_data['hdl'],
            patient_data['smoking'],
            patient_data['diabetes'],
            patient_data['bmi']
        ]
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, patient_data):
        """10년 내 심혈관 질환 발생 위험도 예측"""
        features = self.preprocess_features(patient_data)
        features_scaled = self.scaler.transform(features)
        
        # 확률 예측
        risk_probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # 위험도 레벨 분류
        if risk_probability < 0.1:
            risk_level = "Low"
        elif risk_probability < 0.2:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return {
            'probability': risk_probability,
            'risk_level': risk_level,
            'recommendations': self.get_recommendations(risk_level)
        }`}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Integration Section */}
            {activeSection === 'integration' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 시스템 통합
                  </h2>
                  
                  <div className="space-y-6">
                    {/* EHR Integration */}
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                        <Database className="w-6 h-6 text-green-600 dark:text-green-400" />
                        EHR/EMR 시스템 연동
                      </h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                            데이터 수집
                          </h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• HL7 FHIR API 연동</li>
                            <li>• 실시간 데이터 스트리밍</li>
                            <li>• 구조화/비구조화 데이터 처리</li>
                            <li>• 의료 영상 PACS 연동</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                            결과 피드백
                          </h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• CDS Hooks 구현</li>
                            <li>• SMART on FHIR 앱</li>
                            <li>• 실시간 알림 시스템</li>
                            <li>• 임상 대시보드 통합</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Workflow Integration */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        임상 워크플로우 통합
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-start gap-3">
                          <Clock className="w-5 h-5 text-blue-500 mt-1" />
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">외래 진료</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              진료 중 실시간 진단 제안, 처방 검토, 검사 권고
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Heart className="w-5 h-5 text-red-500 mt-1" />
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">응급실</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              중증도 분류, 신속 진단, 위험 환자 조기 감지
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Activity className="w-5 h-5 text-purple-500 mt-1" />
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">중환자실</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              24시간 모니터링, 악화 예측, 치료 최적화
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Security & Compliance */}
                    <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                        <Shield className="w-6 h-6 text-red-600 dark:text-red-400" />
                        보안 및 규제 준수
                      </h3>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2 text-sm">
                            데이터 보안
                          </h4>
                          <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• 엔드투엔드 암호화</li>
                            <li>• 역할 기반 접근 제어</li>
                            <li>• 감사 로그 관리</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2 text-sm">
                            규제 준수
                          </h4>
                          <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• HIPAA 준수</li>
                            <li>• GDPR 준수</li>
                            <li>• FDA 의료기기 인증</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white mb-2 text-sm">
                            윤리적 AI
                          </h4>
                          <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• 편향성 모니터링</li>
                            <li>• 설명 가능성 보장</li>
                            <li>• 인간 중심 설계</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Case Studies Section */}
            {activeSection === 'cases' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    실제 적용 사례
                  </h2>
                  
                  <div className="space-y-6">
                    {/* Case 1 */}
                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                          <Stethoscope className="w-8 h-8 text-purple-600 dark:text-purple-400" />
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                            Mayo Clinic - AI 기반 심장병 진단
                          </h3>
                          <p className="text-gray-600 dark:text-gray-400 mb-4">
                            ECG 데이터와 임상 정보를 결합한 심방세동 조기 진단 시스템
                          </p>
                          <div className="grid md:grid-cols-3 gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded">
                            <div>
                              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                                97.8%
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                진단 정확도
                              </div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                83%
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                조기 발견율 향상
                              </div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                                2분
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                평균 진단 시간
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Case 2 */}
                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                          <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                            IBM Watson for Oncology - 암 치료 결정 지원
                          </h3>
                          <p className="text-gray-600 dark:text-gray-400 mb-4">
                            개인 맞춤형 암 치료 계획 수립 및 항암제 선택 지원
                          </p>
                          <div className="grid md:grid-cols-3 gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded">
                            <div>
                              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                96%
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                가이드라인 일치율
                              </div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                                13개국
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                도입 국가
                              </div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                                230개
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                협력 병원
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Case 3 */}
                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
                          <AlertTriangle className="w-8 h-8 text-green-600 dark:text-green-400" />
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                            Johns Hopkins - 패혈증 조기 경보 시스템
                          </h3>
                          <p className="text-gray-600 dark:text-gray-400 mb-4">
                            실시간 환자 데이터 분석을 통한 패혈증 위험 예측
                          </p>
                          <div className="grid md:grid-cols-3 gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded">
                            <div>
                              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                                6시간
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                조기 감지 시간
                              </div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                                18%
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                사망률 감소
                              </div>
                            </div>
                            <div>
                              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                $1.4M
                              </div>
                              <div className="text-sm text-gray-500 dark:text-gray-500">
                                연간 의료비 절감
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Success Factors */}
                    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        성공 요인 분석
                      </h3>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                            <Lightbulb className="w-5 h-5 text-yellow-500" />
                            기술적 요인
                          </h4>
                          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                            <li className="flex items-start gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                              <span>고품질 데이터셋 확보</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                              <span>지속적인 모델 업데이트</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                              <span>설명 가능한 AI 구현</span>
                            </li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                            <Users className="w-5 h-5 text-blue-500" />
                            조직적 요인
                          </h4>
                          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                            <li className="flex items-start gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                              <span>의료진 교육 및 훈련</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                              <span>다학제 팀 협업</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                              <span>단계적 도입 전략</span>
                            </li>
                          </ul>
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
                href="/medical-ai/chapter/medical-imaging"
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                이전 챕터
              </Link>
              <Link
                href="/medical-ai/chapter/drug-discovery"
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