'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Brain, 
  Heart,
  TrendingUp,
  Users,
  Target,
  Lightbulb,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  BookOpen,
  Globe,
  Microscope,
  Activity
} from 'lucide-react'

export default function IntroductionPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', title: '개요', icon: Brain },
    { id: 'history', title: '발전 역사', icon: TrendingUp },
    { id: 'applications', title: '응용 분야', icon: Target },
    { id: 'challenges', title: '도전 과제', icon: AlertCircle },
    { id: 'future', title: '미래 전망', icon: Lightbulb }
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
                Chapter 1: Medical AI 개요
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-full text-sm font-medium">
                기초
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
                        ? 'bg-gradient-to-r from-red-500 to-pink-600 text-white shadow-lg'
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
                    Medical AI란?
                  </h2>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      Medical AI(의료 인공지능)는 의료 분야에 인공지능 기술을 적용하여 진단, 치료, 
                      예방, 관리 등 의료 서비스 전반의 품질을 향상시키고 효율성을 높이는 기술입니다.
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 my-8">
                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Heart className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          주요 목표
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>진단 정확도 향상</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>치료 효과 최적화</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>의료 비용 절감</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>의료 접근성 개선</span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                        <Brain className="w-10 h-10 text-purple-600 dark:text-purple-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          핵심 기술
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                            <span>딥러닝 (Deep Learning)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                            <span>컴퓨터 비전 (Computer Vision)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                            <span>자연어 처리 (NLP)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                            <span>예측 모델링</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 my-8">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        의료 AI의 중요성
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        전 세계적으로 고령화가 진행되고 만성질환이 증가하면서 의료 수요는 급증하고 있지만, 
                        의료 인력과 자원은 제한적입니다. Medical AI는 이러한 문제를 해결할 수 있는 핵심 기술로 주목받고 있습니다.
                      </p>
                      <div className="grid grid-cols-3 gap-4 mt-6">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-red-600 dark:text-red-400">95%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">진단 정확도</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">50%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">시간 단축</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-green-600 dark:text-green-400">30%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">비용 절감</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* History Section */}
            {activeSection === 'history' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    Medical AI의 발전 역사
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Timeline */}
                    <div className="relative">
                      <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-red-500 to-pink-600"></div>
                      
                      <div className="space-y-8">
                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-pink-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg">
                            1960s
                          </div>
                          <div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              초기 전문가 시스템
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300">
                              MYCIN, DENDRAL 등 규칙 기반 의료 전문가 시스템 개발
                            </p>
                          </div>
                        </div>

                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg">
                            1980s
                          </div>
                          <div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              의료 영상 처리
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300">
                              CT, MRI 영상 분석을 위한 컴퓨터 비전 기술 도입
                            </p>
                          </div>
                        </div>

                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg">
                            2000s
                          </div>
                          <div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              머신러닝 혁명
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300">
                              SVM, Random Forest 등 머신러닝 알고리즘의 의료 적용
                            </p>
                          </div>
                        </div>

                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg">
                            2010s
                          </div>
                          <div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              딥러닝 시대
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300">
                              CNN, RNN 등 딥러닝 기술로 의료 영상 진단 정확도 혁신
                            </p>
                          </div>
                        </div>

                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-orange-500 to-red-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg">
                            2020s
                          </div>
                          <div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              LLM & 멀티모달 AI
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300">
                              GPT, BERT 등 대규모 언어모델과 멀티모달 AI의 의료 활용
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Applications Section */}
            {activeSection === 'applications' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    주요 응용 분야
                  </h2>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                      <Microscope className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        진단 보조
                      </h3>
                      <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                        <li>• 의료 영상 판독 (X-ray, CT, MRI)</li>
                        <li>• 병리 슬라이드 분석</li>
                        <li>• 피부 질환 진단</li>
                        <li>• 안과 질환 스크리닝</li>
                      </ul>
                    </div>

                    <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <Activity className="w-10 h-10 text-purple-600 dark:text-purple-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        치료 최적화
                      </h3>
                      <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                        <li>• 개인 맞춤형 치료 계획</li>
                        <li>• 약물 용량 최적화</li>
                        <li>• 수술 로봇 지원</li>
                        <li>• 방사선 치료 계획</li>
                      </ul>
                    </div>

                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                      <Globe className="w-10 h-10 text-green-600 dark:text-green-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        공중보건
                      </h3>
                      <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                        <li>• 전염병 예측 및 추적</li>
                        <li>• 인구 건강 관리</li>
                        <li>• 의료 자원 배분</li>
                        <li>• 건강 정책 수립</li>
                      </ul>
                    </div>

                    <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-6">
                      <Users className="w-10 h-10 text-orange-600 dark:text-orange-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        환자 관리
                      </h3>
                      <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                        <li>• 원격 환자 모니터링</li>
                        <li>• 재입원 위험 예측</li>
                        <li>• 복약 순응도 관리</li>
                        <li>• 건강 상태 추적</li>
                      </ul>
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
                    도전 과제
                  </h2>
                  
                  <div className="space-y-6">
                    <div className="border-l-4 border-red-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        데이터 품질과 편향
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        의료 데이터의 불균형과 편향은 AI 모델의 일반화 성능을 저하시킬 수 있습니다. 
                        특정 인종, 성별, 연령대에 편향된 데이터는 공정한 의료 서비스 제공을 어렵게 만듭니다.
                      </p>
                    </div>

                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        규제와 인증
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        의료 AI 제품은 엄격한 규제 승인을 받아야 합니다. FDA, CE 마크 등 각국의 
                        의료기기 인증 절차는 복잡하고 시간이 오래 걸립니다.
                      </p>
                    </div>

                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        설명 가능성
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        딥러닝 모델의 블랙박스 특성은 의료 현장에서 신뢰를 얻기 어렵게 만듭니다. 
                        의사와 환자가 이해할 수 있는 설명 가능한 AI가 필요합니다.
                      </p>
                    </div>

                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        개인정보 보호
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        민감한 의료 정보를 다루는 만큼 HIPAA, GDPR 등 개인정보 보호 규정을 
                        준수하면서도 AI 학습에 필요한 데이터를 확보하는 것이 도전 과제입니다.
                      </p>
                    </div>

                    <div className="border-l-4 border-orange-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        의료진 수용성
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        새로운 기술에 대한 의료진의 거부감과 학습 곡선은 Medical AI 도입의 
                        주요 장벽입니다. 사용자 친화적인 인터페이스와 교육이 필요합니다.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Future Section */}
            {activeSection === 'future' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    미래 전망
                  </h2>
                  
                  <div className="space-y-8">
                    <div className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <h3 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                        2025-2030 단기 전망
                      </h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="flex items-start gap-3">
                          <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">FDA 승인 AI 증가</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              연간 100개 이상의 의료 AI 제품 승인 예상
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">원격의료 통합</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              AI 기반 원격 진단 및 모니터링 일반화
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">정밀의료 확산</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              유전체 기반 맞춤형 치료 대중화
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">예방의학 강화</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              AI 기반 질병 예측 및 예방 시스템 구축
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gradient-to-r from-blue-100 to-cyan-100 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                      <h3 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                        2030-2040 장기 전망
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-start gap-3">
                          <Lightbulb className="w-6 h-6 text-yellow-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">AI 의사 보조에서 파트너로</h4>
                            <p className="text-gray-600 dark:text-gray-400">
                              AI가 의료진과 협업하여 복잡한 의사결정을 공동으로 수행
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Lightbulb className="w-6 h-6 text-yellow-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">디지털 트윈 의료</h4>
                            <p className="text-gray-600 dark:text-gray-400">
                              개인별 디지털 트윈을 통한 질병 시뮬레이션과 치료 최적화
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Lightbulb className="w-6 h-6 text-yellow-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">나노 의료 로봇</h4>
                            <p className="text-gray-600 dark:text-gray-400">
                              AI 제어 나노로봇을 통한 체내 직접 치료
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Lightbulb className="w-6 h-6 text-yellow-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">뇌-컴퓨터 인터페이스</h4>
                            <p className="text-gray-600 dark:text-gray-400">
                              신경 질환 치료를 위한 AI 기반 BCI 기술 상용화
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        시장 전망
                      </h3>
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <div className="text-3xl font-bold text-red-600 dark:text-red-400">$188B</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">2030년 시장 규모</div>
                        </div>
                        <div>
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">37%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">연평균 성장률</div>
                        </div>
                        <div>
                          <div className="text-3xl font-bold text-green-600 dark:text-green-400">70%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">병원 AI 도입률</div>
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
                href="/medical-ai"
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                목록으로
              </Link>
              <Link
                href="/medical-ai/chapter/medical-imaging"
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