'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Grid3x3, 
  Move, 
  Maximize2, 
  GitBranch, 
  Layers, 
  TrendingUp,
  BarChart3,
  Activity,
  ArrowRight,
  Sparkles,
  Sigma,
  Binary,
  Axis3d,
  ChevronRight,
  BookOpen,
  Code,
  Beaker,
  Clock,
  Award,
  Zap,
  Brain
} from 'lucide-react'

const features = [
  {
    icon: Move,
    title: '벡터 연산',
    description: '벡터의 덧셈, 내적, 외적 등 기본 연산 마스터',
    color: 'from-blue-500 to-cyan-600'
  },
  {
    icon: Grid3x3,
    title: '행렬 변환',
    description: '회전, 스케일, 전단 등 다양한 행렬 변환 이해',
    color: 'from-purple-500 to-pink-600'
  },
  {
    icon: GitBranch,
    title: '고유값 분해',
    description: 'PCA, 차원축소의 핵심 개념인 고유값/고유벡터',
    color: 'from-green-500 to-emerald-600'
  },
  {
    icon: Layers,
    title: 'SVD 분해',
    description: '추천시스템, 이미지 압축에 활용되는 특이값 분해',
    color: 'from-orange-500 to-red-600'
  }
]

const stats = [
  { label: '학습 모듈', value: '8개', icon: BookOpen },
  { label: '시뮬레이터', value: '4개', icon: Beaker },
  { label: '실습 예제', value: '50+', icon: Code },
  { label: '학습 시간', value: '20시간', icon: Clock }
]

const chapters = [
  {
    id: 1,
    title: '벡터와 벡터공간',
    description: '벡터의 기본 개념과 연산, 벡터공간의 성질',
    icon: Move,
    topics: ['벡터 정의', '내적과 외적', '벡터 투영', '기저와 차원']
  },
  {
    id: 2,
    title: '행렬과 행렬연산',
    description: '행렬의 기본 연산과 특수 행렬',
    icon: Grid3x3,
    topics: ['행렬 곱셈', '역행렬', '행렬식', '전치행렬']
  },
  {
    id: 3,
    title: '선형변환',
    description: '선형변환의 기하학적 의미와 응용',
    icon: Maximize2,
    topics: ['변환 행렬', '회전과 스케일링', '투영 변환', '좌표계 변환']
  },
  {
    id: 4,
    title: '고유값과 고유벡터',
    description: '행렬 분해의 핵심 개념',
    icon: GitBranch,
    topics: ['특성방정식', '대각화', 'PCA 원리', '스펙트럼 정리']
  },
  {
    id: 5,
    title: '직교성과 정규화',
    description: '벡터의 직교성과 그람-슈미트 과정',
    icon: Axis3d,
    topics: ['직교 벡터', 'QR 분해', '정규직교 기저', '최소제곱법']
  },
  {
    id: 6,
    title: 'SVD와 차원축소',
    description: '특이값 분해와 데이터 압축',
    icon: Layers,
    topics: ['SVD 이론', '저계수 근사', '이미지 압축', '추천시스템']
  },
  {
    id: 7,
    title: '선형시스템',
    description: '연립방정식의 해법과 응용',
    icon: Binary,
    topics: ['가우스 소거법', 'LU 분해', '반복법', '조건수']
  },
  {
    id: 8,
    title: 'AI/ML 응용',
    description: '머신러닝에서의 선형대수 활용',
    icon: Sparkles,
    topics: ['신경망 행렬', '역전파 알고리즘', '최적화', '텐서 연산']
  }
]

const simulators = [
  {
    title: 'Vector Visualizer',
    description: '2D/3D 벡터 연산을 실시간으로 시각화',
    icon: Move,
    gradient: 'from-blue-500 to-cyan-600',
    features: ['벡터 덧셈/뺄셈', '내적/외적 계산', '투영 시각화']
  },
  {
    title: 'Matrix Calculator',
    description: '행렬 연산과 변환을 대화형으로 탐색',
    icon: Grid3x3,
    gradient: 'from-purple-500 to-pink-600',
    features: ['행렬 곱셈', '역행렬 계산', '변환 애니메이션']
  },
  {
    title: 'Eigenvalue Explorer',
    description: '고유값과 고유벡터를 시각적으로 이해',
    icon: GitBranch,
    gradient: 'from-green-500 to-emerald-600',
    features: ['고유값 계산', '고유벡터 시각화', 'PCA 데모']
  },
  {
    title: 'SVD Decomposer',
    description: 'SVD를 통한 이미지 압축과 복원',
    icon: Layers,
    gradient: 'from-orange-500 to-red-600',
    features: ['SVD 분해', '저계수 근사', '압축률 조절']
  }
]

const applications = [
  {
    title: '컴퓨터 그래픽스',
    icon: Maximize2,
    description: '3D 렌더링, 게임 엔진에서의 변환 행렬'
  },
  {
    title: '머신러닝',
    icon: Brain,
    description: '신경망, PCA, 추천시스템의 수학적 기반'
  },
  {
    title: '데이터 사이언스',
    icon: BarChart3,
    description: '차원축소, 특징추출, 데이터 압축'
  },
  {
    title: '컴퓨터 비전',
    icon: Activity,
    description: '이미지 변환, 필터링, 특징점 검출'
  }
]

export default function LinearAlgebraPage() {
  const [activeTab, setActiveTab] = useState<'chapters' | 'simulators'>('chapters')

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        <div className="relative max-w-7xl mx-auto px-6 py-24">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded-full text-sm font-medium mb-6">
              <Sparkles className="w-4 h-4" />
              AI/ML의 수학적 기초
            </div>
            
            <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
              Linear Algebra
            </h1>
            
            <p className="text-xl text-gray-600 dark:text-gray-300 mb-12 leading-relaxed">
              벡터, 행렬, 선형변환의 개념을 시각적으로 이해하고
              <br />AI와 머신러닝의 수학적 기초를 탄탄히 다집니다
            </p>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
              {stats.map((stat, idx) => {
                const Icon = stat.icon
                return (
                  <div key={idx} className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
                    <Icon className="w-8 h-8 text-indigo-500 mb-3 mx-auto" />
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">{stat.value}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</div>
                  </div>
                )
              })}
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/linear-algebra/chapter/vectors"
                className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold hover:shadow-xl hover:scale-105 transition-all"
              >
                <BookOpen className="w-5 h-5" />
                학습 시작하기
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                href="/linear-algebra/simulator/vector-visualizer"
                className="inline-flex items-center gap-2 px-8 py-4 bg-white dark:bg-gray-800 text-gray-900 dark:text-white border-2 border-gray-200 dark:border-gray-700 rounded-xl font-semibold hover:shadow-xl hover:scale-105 transition-all"
              >
                <Beaker className="w-5 h-5" />
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
            핵심 학습 영역
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
                    ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
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
                    ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                <Beaker className="w-5 h-5 inline mr-2" />
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
                    href={`/linear-algebra/chapter/${chapter.title.toLowerCase().replace(/\s+/g, '-')}`}
                    className="group bg-white dark:bg-gray-900 rounded-xl p-6 hover:shadow-xl transition-all hover:scale-105"
                  >
                    <div className="flex items-start gap-4 mb-4">
                      <div className="p-3 bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-indigo-900/30 dark:to-purple-900/30 rounded-xl">
                        <Icon className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                      </div>
                      <div className="flex-1">
                        <div className="text-sm font-medium text-indigo-600 dark:text-indigo-400 mb-1">
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
                    href={`/linear-algebra/simulator/${sim.title.toLowerCase().replace(/\s+/g, '-')}`}
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
                      <div className="mt-6 flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-medium group-hover:gap-4 transition-all">
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

      {/* Applications */}
      <section className="py-20 px-6 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-900 dark:text-white">
            실제 응용 분야
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {applications.map((app, idx) => {
              const Icon = app.icon
              return (
                <div key={idx} className="text-center">
                  <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center">
                    <Icon className="w-10 h-10 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                    {app.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    {app.description}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Why Learn */}
      <section className="py-20 px-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-gray-800 dark:to-gray-900">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
            왜 선형대수를 배워야 할까요?
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 leading-relaxed">
            선형대수는 AI, 머신러닝, 컴퓨터 그래픽스, 데이터 사이언스의 핵심 기초입니다.
            벡터와 행렬 연산을 이해하면 복잡한 알고리즘의 작동 원리를 명확히 파악할 수 있으며,
            더 나은 모델을 설계하고 최적화할 수 있습니다.
          </p>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <Zap className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">
                효율적인 계산
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                벡터화된 연산으로 대규모 데이터를 빠르게 처리
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <Award className="w-12 h-12 text-green-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">
                깊은 이해
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                AI 모델의 수학적 원리를 완벽히 이해
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <TrendingUp className="w-12 h-12 text-blue-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">
                커리어 성장
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                데이터 사이언티스트, ML 엔지니어 필수 역량
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}