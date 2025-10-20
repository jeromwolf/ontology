'use client'

import { useState } from 'react'
import SimulatorNav from './SimulatorNav'
import { TrendingUp, Download, Users, Briefcase, Globe, Brain, CheckCircle, AlertTriangle } from 'lucide-react'

type AISystemType = 'llm' | 'computer-vision' | 'recommender' | 'autonomous'
type StakeholderType = 'users' | 'workers' | 'society' | 'environment'
type ImpactCategory = 'economic' | 'social' | 'environmental' | 'psychological'

interface Impact {
  id: string
  stakeholder: StakeholderType
  category: ImpactCategory
  description: string
  severity: number // 1-5
  likelihood: number // 1-5
  mitigation: string
}

const AI_SYSTEMS = {
  llm: {
    name: '대규모 언어 모델 (LLM)',
    icon: Brain,
    color: 'from-purple-500 to-indigo-600',
    examples: ['ChatGPT', 'Claude', 'Gemini']
  },
  'computer-vision': {
    name: '컴퓨터 비전',
    icon: Globe,
    color: 'from-blue-500 to-cyan-600',
    examples: ['얼굴 인식', '객체 탐지', '의료 영상 분석']
  },
  recommender: {
    name: '추천 시스템',
    icon: Users,
    color: 'from-green-500 to-emerald-600',
    examples: ['Netflix', 'YouTube', '전자상거래']
  },
  autonomous: {
    name: '자율 시스템',
    icon: Briefcase,
    color: 'from-orange-500 to-red-600',
    examples: ['자율주행차', '드론', '로봇']
  }
}

const STAKEHOLDERS = {
  users: { name: '사용자', icon: Users, color: 'text-blue-600 dark:text-blue-400' },
  workers: { name: '근로자', icon: Briefcase, color: 'text-green-600 dark:text-green-400' },
  society: { name: '사회', icon: Globe, color: 'text-purple-600 dark:text-purple-400' },
  environment: { name: '환경', icon: Globe, color: 'text-emerald-600 dark:text-emerald-400' }
}

const CATEGORIES = {
  economic: { name: '경제적', color: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300' },
  social: { name: '사회적', color: 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300' },
  environmental: { name: '환경적', color: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300' },
  psychological: { name: '심리적', color: 'bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300' }
}

const DPIA_CHECKLIST = [
  { id: 'd1', question: '개인정보를 새롭고 독특한 방식으로 사용하나요?', checked: false },
  { id: 'd2', question: '대규모 개인정보를 처리하나요?', checked: false },
  { id: 'd3', question: '민감한 개인정보를 처리하나요?', checked: false },
  { id: 'd4', question: '자동화된 의사결정을 수행하나요?', checked: false },
  { id: 'd5', question: '프로파일링 또는 특수 카테고리 데이터를 사용하나요?', checked: false },
  { id: 'd6', question: '개인의 권리 행사를 방해할 가능성이 있나요?', checked: false },
  { id: 'd7', question: '공공장소를 체계적으로 모니터링하나요?', checked: false },
  { id: 'd8', question: '취약 계층(어린이, 노인 등)의 데이터를 처리하나요?', checked: false }
]

export default function ImpactAssessment() {
  const [systemType, setSystemType] = useState<AISystemType>('llm')
  const [impacts, setImpacts] = useState<Impact[]>([
    {
      id: '1',
      stakeholder: 'users',
      category: 'psychological',
      description: '과도한 의존으로 인한 비판적 사고 능력 저하',
      severity: 3,
      likelihood: 4,
      mitigation: '사용자 교육 프로그램 제공, 사용 시간 제한 기능'
    },
    {
      id: '2',
      stakeholder: 'workers',
      category: 'economic',
      description: '콘텐츠 크리에이터 일자리 감소',
      severity: 4,
      likelihood: 3,
      mitigation: '재교육 프로그램, 공정한 보상 체계 구축'
    }
  ])
  const [dpiaChecklist, setDpiaChecklist] = useState(DPIA_CHECKLIST)
  const [showAddImpact, setShowAddImpact] = useState(false)
  const [newImpact, setNewImpact] = useState<Partial<Impact>>({
    stakeholder: 'users',
    category: 'economic',
    severity: 3,
    likelihood: 3
  })

  const addImpact = () => {
    if (!newImpact.description || !newImpact.mitigation) return

    const impact: Impact = {
      id: Date.now().toString(),
      stakeholder: newImpact.stakeholder!,
      category: newImpact.category!,
      description: newImpact.description,
      severity: newImpact.severity!,
      likelihood: newImpact.likelihood!,
      mitigation: newImpact.mitigation
    }

    setImpacts([...impacts, impact])
    setNewImpact({
      stakeholder: 'users',
      category: 'economic',
      severity: 3,
      likelihood: 3
    })
    setShowAddImpact(false)
  }

  const removeImpact = (id: string) => {
    setImpacts(impacts.filter(i => i.id !== id))
  }

  const toggleDPIA = (id: string) => {
    setDpiaChecklist(prev =>
      prev.map(item =>
        item.id === id ? { ...item, checked: !item.checked } : item
      )
    )
  }

  const sortedImpacts = [...impacts].sort((a, b) => {
    const scoreA = a.severity * a.likelihood
    const scoreB = b.severity * b.likelihood
    return scoreB - scoreA
  })

  const dpiaScore = dpiaChecklist.filter(item => item.checked).length
  const dpiaRequired = dpiaScore >= 2

  const exportReport = () => {
    const report = {
      systemType: AI_SYSTEMS[systemType].name,
      timestamp: new Date().toISOString(),
      impacts: impacts.map(i => ({
        stakeholder: STAKEHOLDERS[i.stakeholder].name,
        category: CATEGORIES[i.category].name,
        description: i.description,
        severity: i.severity,
        likelihood: i.likelihood,
        riskScore: i.severity * i.likelihood,
        mitigation: i.mitigation
      })),
      dpia: {
        required: dpiaRequired,
        score: dpiaScore,
        checklist: dpiaChecklist.filter(item => item.checked).map(item => item.question)
      },
      summary: {
        totalImpacts: impacts.length,
        highRiskImpacts: impacts.filter(i => i.severity * i.likelihood >= 12).length,
        averageRiskScore: impacts.reduce((sum, i) => sum + i.severity * i.likelihood, 0) / impacts.length
      }
    }

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `impact-assessment-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50 to-pink-50 dark:from-gray-900 dark:to-rose-950">
      <SimulatorNav />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-gradient-to-br from-rose-500 to-pink-600 text-white rounded-xl">
                <TrendingUp className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Impact Assessment
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  AI 시스템의 사회적 영향을 평가하고 관리합니다
                </p>
              </div>
            </div>

            <button
              onClick={exportReport}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-rose-500 to-pink-600 text-white rounded-lg hover:from-rose-600 hover:to-pink-700 transition-all shadow-lg"
            >
              <Download className="w-5 h-5" />
              <span>보고서 내보내기</span>
            </button>
          </div>

          {/* System Type Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
            {(Object.keys(AI_SYSTEMS) as AISystemType[]).map((type) => {
              const system = AI_SYSTEMS[type]
              const Icon = system.icon
              const isSelected = systemType === type
              return (
                <button
                  key={type}
                  onClick={() => setSystemType(type)}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    isSelected
                      ? 'border-rose-500 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-rose-300 dark:hover:border-rose-700'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-2 bg-gradient-to-br ${system.color} text-white rounded-lg`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-gray-900 dark:text-white text-sm">
                        {system.name}
                      </h3>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {system.examples[0]}
                      </p>
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Impact Matrix */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              영향 매트릭스
            </h2>
            <button
              onClick={() => setShowAddImpact(!showAddImpact)}
              className="px-4 py-2 bg-gradient-to-r from-rose-500 to-pink-600 text-white rounded-lg hover:from-rose-600 hover:to-pink-700 transition-all"
            >
              {showAddImpact ? '취소' : '+ 영향 추가'}
            </button>
          </div>

          {/* Add Impact Form */}
          {showAddImpact && (
            <div className="mb-6 p-6 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    이해관계자
                  </label>
                  <select
                    value={newImpact.stakeholder}
                    onChange={(e) => setNewImpact({ ...newImpact, stakeholder: e.target.value as StakeholderType })}
                    className="w-full px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-rose-500 dark:text-white"
                  >
                    {Object.entries(STAKEHOLDERS).map(([key, value]) => (
                      <option key={key} value={key}>{value.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    영향 카테고리
                  </label>
                  <select
                    value={newImpact.category}
                    onChange={(e) => setNewImpact({ ...newImpact, category: e.target.value as ImpactCategory })}
                    className="w-full px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-rose-500 dark:text-white"
                  >
                    {Object.entries(CATEGORIES).map(([key, value]) => (
                      <option key={key} value={key}>{value.name}</option>
                    ))}
                  </select>
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    영향 설명
                  </label>
                  <input
                    type="text"
                    value={newImpact.description || ''}
                    onChange={(e) => setNewImpact({ ...newImpact, description: e.target.value })}
                    placeholder="구체적인 영향을 설명하세요"
                    className="w-full px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-rose-500 dark:text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    심각도: {newImpact.severity}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={newImpact.severity}
                    onChange={(e) => setNewImpact({ ...newImpact, severity: parseInt(e.target.value) })}
                    className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    가능성: {newImpact.likelihood}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={newImpact.likelihood}
                    onChange={(e) => setNewImpact({ ...newImpact, likelihood: parseInt(e.target.value) })}
                    className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    완화 전략
                  </label>
                  <input
                    type="text"
                    value={newImpact.mitigation || ''}
                    onChange={(e) => setNewImpact({ ...newImpact, mitigation: e.target.value })}
                    placeholder="이 영향을 완화하기 위한 전략"
                    className="w-full px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-rose-500 dark:text-white"
                  />
                </div>
              </div>

              <button
                onClick={addImpact}
                className="w-full px-6 py-3 bg-gradient-to-r from-rose-500 to-pink-600 text-white rounded-lg hover:from-rose-600 hover:to-pink-700 transition-all font-semibold"
              >
                영향 추가
              </button>
            </div>
          )}

          {/* Impact List */}
          <div className="space-y-4">
            {sortedImpacts.map((impact) => {
              const riskScore = impact.severity * impact.likelihood
              const Icon = STAKEHOLDERS[impact.stakeholder].icon
              let riskColor = 'from-green-500 to-emerald-600'
              let riskLabel = 'Low'
              if (riskScore >= 15) {
                riskColor = 'from-red-500 to-rose-600'
                riskLabel = 'Critical'
              } else if (riskScore >= 10) {
                riskColor = 'from-orange-500 to-amber-600'
                riskLabel = 'High'
              } else if (riskScore >= 6) {
                riskColor = 'from-yellow-500 to-orange-500'
                riskLabel = 'Medium'
              }

              return (
                <div
                  key={impact.id}
                  className="p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-700/50 dark:to-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start gap-3 flex-1">
                      <Icon className={`w-5 h-5 mt-0.5 ${STAKEHOLDERS[impact.stakeholder].color}`} />
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-semibold ${CATEGORIES[impact.category].color}`}>
                            {CATEGORIES[impact.category].name}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {STAKEHOLDERS[impact.stakeholder].name}
                          </span>
                        </div>
                        <p className="font-semibold text-gray-900 dark:text-white">
                          {impact.description}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className={`px-3 py-1 bg-gradient-to-r ${riskColor} text-white rounded-lg font-bold text-sm`}>
                        {riskScore} - {riskLabel}
                      </div>
                      <button
                        onClick={() => removeImpact(impact.id)}
                        className="text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                      >
                        ✕
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">심각도</p>
                      <div className="flex gap-1">
                        {[1, 2, 3, 4, 5].map(i => (
                          <div
                            key={i}
                            className={`h-2 flex-1 rounded ${
                              i <= impact.severity
                                ? 'bg-gradient-to-r from-rose-500 to-pink-600'
                                : 'bg-gray-200 dark:bg-gray-600'
                            }`}
                          />
                        ))}
                      </div>
                    </div>

                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">가능성</p>
                      <div className="flex gap-1">
                        {[1, 2, 3, 4, 5].map(i => (
                          <div
                            key={i}
                            className={`h-2 flex-1 rounded ${
                              i <= impact.likelihood
                                ? 'bg-gradient-to-r from-rose-500 to-pink-600'
                                : 'bg-gray-200 dark:bg-gray-600'
                            }`}
                          />
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                      <strong>완화 전략:</strong>
                    </p>
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {impact.mitigation}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* DPIA Checklist */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              DPIA 체크리스트
            </h2>

            <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
              데이터 보호 영향 평가(Data Protection Impact Assessment)가 필요한지 확인하세요
            </p>

            <div className="space-y-3 mb-6">
              {dpiaChecklist.map((item) => (
                <button
                  key={item.id}
                  onClick={() => toggleDPIA(item.id)}
                  className={`w-full p-3 rounded-lg border text-left transition-all ${
                    item.checked
                      ? 'border-rose-500 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-rose-300 dark:hover:border-rose-700'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5">
                      {item.checked ? (
                        <CheckCircle className="w-5 h-5 text-rose-600 dark:text-rose-400" />
                      ) : (
                        <div className="w-5 h-5 rounded-full border-2 border-gray-300 dark:border-gray-600" />
                      )}
                    </div>
                    <p className="text-sm text-gray-900 dark:text-white flex-1">
                      {item.question}
                    </p>
                  </div>
                </button>
              ))}
            </div>

            <div className={`p-4 rounded-lg border ${
              dpiaRequired
                ? 'bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 border-rose-500'
                : 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-green-500'
            }`}>
              <div className="flex items-start gap-3">
                {dpiaRequired ? (
                  <AlertTriangle className="w-5 h-5 text-rose-600 dark:text-rose-400 mt-0.5" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
                )}
                <div>
                  <p className="font-semibold text-gray-900 dark:text-white">
                    {dpiaRequired ? 'DPIA 필요' : 'DPIA 불필요'}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    체크된 항목: {dpiaScore}/8
                    {dpiaRequired && ' (2개 이상 체크 시 DPIA 권장)'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Summary Statistics */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              영향 요약
            </h2>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="p-4 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">총 영향 수</p>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">
                  {impacts.length}
                </p>
              </div>

              <div className="p-4 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">고위험 영향</p>
                <p className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                  {impacts.filter(i => i.severity * i.likelihood >= 12).length}
                </p>
              </div>

              <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg col-span-2">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">평균 위험 점수</p>
                <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                  {impacts.length > 0
                    ? (impacts.reduce((sum, i) => sum + i.severity * i.likelihood, 0) / impacts.length).toFixed(1)
                    : '0.0'
                  }
                </p>
              </div>
            </div>

            {/* Stakeholder Breakdown */}
            <div className="space-y-3">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                이해관계자별 영향
              </h3>
              {(Object.keys(STAKEHOLDERS) as StakeholderType[]).map((stakeholder) => {
                const count = impacts.filter(i => i.stakeholder === stakeholder).length
                const Icon = STAKEHOLDERS[stakeholder].icon
                return (
                  <div key={stakeholder} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Icon className={`w-4 h-4 ${STAKEHOLDERS[stakeholder].color}`} />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {STAKEHOLDERS[stakeholder].name}
                      </span>
                    </div>
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {count}개
                    </span>
                  </div>
                )
              })}
            </div>

            {/* Category Breakdown */}
            <div className="space-y-3 mt-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                카테고리별 영향
              </h3>
              {(Object.keys(CATEGORIES) as ImpactCategory[]).map((category) => {
                const count = impacts.filter(i => i.category === category).length
                return (
                  <div key={category} className="flex items-center justify-between">
                    <span className={`px-2 py-1 rounded-full text-xs font-semibold ${CATEGORIES[category].color}`}>
                      {CATEGORIES[category].name}
                    </span>
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {count}개
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
