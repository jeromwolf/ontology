'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Brain, 
  Activity, 
  Microscope, 
  Pill, 
  Scan, 
  Heart,
  Database,
  Dna,
  ArrowRight,
  Sparkles,
  TrendingUp,
  Users,
  Shield,
  Clock,
  Award,
  ChevronRight,
  Stethoscope,
  FileText,
  Zap,
  BookOpen,
  Code,
  Beaker as BeakerIcon
} from 'lucide-react'

const features = [
  {
    icon: Scan,
    title: '의료 영상 분석',
    description: 'X-Ray, CT, MRI 영상을 AI로 분석하여 질병을 조기 발견',
    color: 'from-blue-500 to-cyan-600'
  },
  {
    icon: Brain,
    title: '진단 보조 AI',
    description: '증상과 검사 결과를 종합하여 정확한 진단 지원',
    color: 'from-purple-500 to-pink-600'
  },
  {
    icon: Pill,
    title: '신약 개발',
    description: 'AI를 활용한 신약 후보 물질 발굴과 임상 시험 최적화',
    color: 'from-green-500 to-emerald-600'
  },
  {
    icon: Dna,
    title: '유전체 분석',
    description: '개인 맞춤형 정밀 의료를 위한 유전체 데이터 분석',
    color: 'from-orange-500 to-red-600'
  }
]

const stats = [
  { label: '학습 모듈', value: '8개', icon: BookOpen },
  { label: '시뮬레이터', value: '4개', icon: BeakerIcon },
  { label: '실습 예제', value: '20+', icon: Code },
  { label: '학습 시간', value: '15시간', icon: Clock }
]

const chapters = [
  {
    id: 1,
    title: 'Medical AI 개요',
    description: '의료 AI의 기본 개념과 응용 분야',
    icon: Brain,
    topics: ['의료 AI 정의', '발전 역사', '주요 응용 분야', '미래 전망']
  },
  {
    id: 2,
    title: '의료 영상 분석',
    description: 'CNN을 활용한 의료 영상 진단',
    icon: Scan,
    topics: ['X-Ray 분석', 'CT/MRI 판독', '종양 검출', '영상 전처리']
  },
  {
    id: 3,
    title: '진단 보조 시스템',
    description: '머신러닝 기반 진단 지원',
    icon: Activity,
    topics: ['증상 분석', '질병 예측', '위험도 평가', '치료 추천']
  },
  {
    id: 4,
    title: '신약 개발 AI',
    description: 'AI 기반 신약 발굴과 개발',
    icon: Pill,
    topics: ['분자 설계', '약물 상호작용', '임상시험 최적화', '부작용 예측']
  },
  {
    id: 5,
    title: '유전체 분석',
    description: '정밀의료를 위한 유전체 데이터 분석',
    icon: Dna,
    topics: ['유전자 시퀀싱', '변이 분석', '질병 연관성', '맞춤 치료']
  },
  {
    id: 6,
    title: '환자 모니터링',
    description: '실시간 환자 상태 모니터링과 예측',
    icon: Heart,
    topics: ['바이탈 모니터링', '이상 징후 감지', '예후 예측', 'ICU 관리']
  },
  {
    id: 7,
    title: '의료 데이터 관리',
    description: 'EHR과 의료 빅데이터 처리',
    icon: Database,
    topics: ['EHR 시스템', '데이터 표준화', 'FHIR 프로토콜', '데이터 보안']
  },
  {
    id: 8,
    title: '윤리와 규제',
    description: '의료 AI의 윤리적 고려사항과 규제',
    icon: Shield,
    topics: ['개인정보 보호', 'FDA 승인', '의료 윤리', '책임 소재']
  }
]

const simulators = [
  {
    title: 'X-Ray Analyzer',
    description: '흉부 X-Ray 영상에서 폐렴, 결핵 등 질병 검출',
    icon: Scan,
    gradient: 'from-blue-500 to-cyan-600',
    features: ['실시간 분석', 'Heatmap 시각화', '정확도 95%+']
  },
  {
    title: 'Diagnosis AI',
    description: '증상 기반 질병 진단 및 치료 추천 시스템',
    icon: Brain,
    gradient: 'from-purple-500 to-pink-600',
    features: ['다중 증상 분석', '확률 기반 진단', '치료 가이드']
  },
  {
    title: 'Drug Discovery',
    description: '분자 구조 기반 신약 후보 물질 스크리닝',
    icon: BeakerIcon,
    gradient: 'from-green-500 to-emerald-600',
    features: ['분자 도킹', 'ADMET 예측', '타겟 단백질 분석']
  },
  {
    title: 'Patient Dashboard',
    description: '실시간 환자 모니터링 및 위험도 평가',
    icon: Activity,
    gradient: 'from-orange-500 to-red-600',
    features: ['실시간 모니터링', '예측 알림', '트렌드 분석']
  }
]

export default function MedicalAIPage() {
  const [activeTab, setActiveTab] = useState<'chapters' | 'simulators'>('chapters')

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-red-50 via-pink-50 to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        <div className="relative max-w-7xl mx-auto px-6 py-24">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-full text-sm font-medium mb-6">
              <Sparkles className="w-4 h-4" />
              AI-Powered Healthcare Innovation
            </div>
            
            <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-red-600 via-pink-600 to-purple-600 bg-clip-text text-transparent">
              Medical AI
            </h1>
            
            <p className="text-xl text-gray-600 dark:text-gray-300 mb-12 leading-relaxed">
              의료 분야에 AI를 적용하여 진단 정확도를 높이고, 치료 효과를 개선하며,
              <br />신약 개발을 가속화하는 혁신적인 기술을 학습합니다
            </p>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
              {stats.map((stat, idx) => {
                const Icon = stat.icon
                return (
                  <div key={idx} className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                    <Icon className="w-8 h-8 text-red-500 mb-3 mx-auto" />
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">{stat.value}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</div>
                  </div>
                )
              })}
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/medical-ai/chapter/introduction"
                className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-red-600 to-pink-600 text-white rounded-xl font-semibold hover:shadow-xl hover:scale-105 transition-all"
              >
                <BookOpen className="w-5 h-5" />
                학습 시작하기
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                href="/medical-ai/simulator/xray-analyzer"
                className="inline-flex items-center gap-2 px-8 py-4 bg-white dark:bg-gray-800 text-gray-900 dark:text-white border-2 border-gray-200 dark:border-gray-700 rounded-xl font-semibold hover:shadow-xl hover:scale-105 transition-all"
              >
                <BeakerIcon className="w-5 h-5" />
                시뮬레이터 체험
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-6 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-900 dark:text-white">
            주요 학습 영역
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, idx) => {
              const Icon = feature.icon
              return (
                <div key={idx} className="group relative bg-gray-50 dark:bg-gray-800 rounded-xl p-6 hover:shadow-xl transition-all hover:scale-105">
                  <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${feature.color} mb-4`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 text-sm">
                    {feature.description}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Content Tabs */}
      <section className="py-20 px-6 bg-gray-50 dark:bg-gray-800">
        <div className="max-w-7xl mx-auto">
          {/* Tab Navigation */}
          <div className="flex justify-center mb-12">
            <div className="inline-flex bg-white dark:bg-gray-900 rounded-xl p-1 shadow-lg">
              <button
                onClick={() => setActiveTab('chapters')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  activeTab === 'chapters'
                    ? 'bg-gradient-to-r from-red-600 to-pink-600 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                <BookOpen className="w-5 h-5 inline mr-2" />
                학습 콘텐츠
              </button>
              <button
                onClick={() => setActiveTab('simulators')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  activeTab === 'simulators'
                    ? 'bg-gradient-to-r from-red-600 to-pink-600 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                <BeakerIcon className="w-5 h-5 inline mr-2" />
                시뮬레이터
              </button>
            </div>
          </div>

          {/* Chapters Content */}
          {activeTab === 'chapters' && (
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {chapters.map((chapter) => {
                const Icon = chapter.icon
                return (
                  <Link
                    key={chapter.id}
                    href={`/medical-ai/chapter/${chapter.title.toLowerCase().replace(/\s+/g, '-')}`}
                    className="group bg-white dark:bg-gray-900 rounded-xl p-6 hover:shadow-xl transition-all hover:scale-105"
                  >
                    <div className="flex items-start gap-4 mb-4">
                      <div className="p-3 bg-gradient-to-br from-red-100 to-pink-100 dark:from-red-900/30 dark:to-pink-900/30 rounded-xl">
                        <Icon className="w-6 h-6 text-red-600 dark:text-red-400" />
                      </div>
                      <div className="flex-1">
                        <div className="text-sm font-medium text-red-600 dark:text-red-400 mb-1">
                          Chapter {chapter.id}
                        </div>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                          {chapter.title}
                        </h3>
                      </div>
                    </div>
                    <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                      {chapter.description}
                    </p>
                    <div className="space-y-2">
                      {chapter.topics.map((topic, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-500">
                          <ChevronRight className="w-3 h-3" />
                          <span>{topic}</span>
                        </div>
                      ))}
                    </div>
                  </Link>
                )
              })}
            </div>
          )}

          {/* Simulators Content */}
          {activeTab === 'simulators' && (
            <div className="grid md:grid-cols-2 gap-8">
              {simulators.map((sim, idx) => {
                const Icon = sim.icon
                return (
                  <Link
                    key={idx}
                    href={`/medical-ai/simulator/${sim.title.toLowerCase().replace(/\s+/g, '-')}`}
                    className="group relative bg-white dark:bg-gray-900 rounded-xl overflow-hidden hover:shadow-2xl transition-all hover:scale-105"
                  >
                    <div className={`h-2 bg-gradient-to-r ${sim.gradient}`}></div>
                    <div className="p-8">
                      <div className="flex items-start gap-4 mb-6">
                        <div className={`p-4 rounded-xl bg-gradient-to-br ${sim.gradient}`}>
                          <Icon className="w-8 h-8 text-white" />
                        </div>
                        <div className="flex-1">
                          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                            {sim.title}
                          </h3>
                          <p className="text-gray-600 dark:text-gray-400">
                            {sim.description}
                          </p>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {sim.features.map((feature, fIdx) => (
                          <span
                            key={fIdx}
                            className="px-3 py-1 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-full text-sm"
                          >
                            {feature}
                          </span>
                        ))}
                      </div>
                      <div className="mt-6 flex items-center gap-2 text-red-600 dark:text-red-400 font-medium group-hover:gap-4 transition-all">
                        체험하기
                        <ArrowRight className="w-5 h-5" />
                      </div>
                    </div>
                  </Link>
                )
              })}
            </div>
          )}
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-20 px-6 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-900 dark:text-white">
            실제 활용 사례
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-full flex items-center justify-center">
                <Stethoscope className="w-10 h-10 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                조기 진단
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                암, 당뇨병 망막증 등 질병의 조기 발견으로 생존율 40% 향상
              </p>
            </div>
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center">
                <FileText className="w-10 h-10 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                의료 효율성
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                의료진 업무 부담 30% 감소, 진단 시간 50% 단축
              </p>
            </div>
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center">
                <Zap className="w-10 h-10 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                신약 개발
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                개발 기간 10년→5년 단축, 비용 50% 절감
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}