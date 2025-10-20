'use client'

import { useState } from 'react'
import SimulatorNav from './SimulatorNav'
import { FileCheck, Download, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'

type RiskLevel = 'prohibited' | 'high-risk' | 'limited-risk' | 'minimal'
type EthicalPrinciple = 'transparency' | 'accountability' | 'fairness' | 'privacy' | 'safety' | 'human-agency' | 'societal-wellbeing'

interface ComplianceItem {
  id: string
  category: string
  requirement: string
  checked: boolean
}

const RISK_LEVELS = {
  prohibited: {
    name: '금지',
    color: 'from-red-500 to-rose-600',
    bgColor: 'from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20',
    borderColor: 'border-red-500',
    description: '사회적으로 용납할 수 없는 위험',
    examples: ['사회 신용 점수 시스템', '무차별 감시', '잠재의식 조작']
  },
  'high-risk': {
    name: '고위험',
    color: 'from-orange-500 to-amber-600',
    bgColor: 'from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20',
    borderColor: 'border-orange-500',
    description: '개인의 권리와 안전에 중대한 영향',
    examples: ['채용/평가 시스템', '신용 평가', '의료 진단', '법 집행']
  },
  'limited-risk': {
    name: '제한적 위험',
    color: 'from-yellow-500 to-orange-500',
    bgColor: 'from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20',
    borderColor: 'border-yellow-500',
    description: '투명성 의무 필요',
    examples: ['챗봇', '추천 시스템', '감정 인식', 'Deepfake']
  },
  minimal: {
    name: '최소 위험',
    color: 'from-green-500 to-emerald-600',
    bgColor: 'from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20',
    borderColor: 'border-green-500',
    description: '자유로운 사용 가능',
    examples: ['스팸 필터', '게임 AI', '이미지 편집', '텍스트 자동완성']
  }
}

const ETHICAL_PRINCIPLES = {
  transparency: {
    name: '투명성',
    icon: '👁️',
    description: 'AI 시스템의 작동 방식과 결정 과정을 이해할 수 있어야 함'
  },
  accountability: {
    name: '책임성',
    icon: '⚖️',
    description: 'AI 시스템의 결과에 대한 명확한 책임 소재가 있어야 함'
  },
  fairness: {
    name: '공정성',
    icon: '🤝',
    description: '편향 없이 모든 사용자를 공정하게 대우해야 함'
  },
  privacy: {
    name: '프라이버시',
    icon: '🔒',
    description: '개인정보 보호 및 데이터 주권을 존중해야 함'
  },
  safety: {
    name: '안전성',
    icon: '🛡️',
    description: '물리적, 정신적 피해를 방지해야 함'
  },
  'human-agency': {
    name: '인간 주체성',
    icon: '🙋',
    description: '인간의 자율성과 통제권을 보장해야 함'
  },
  'societal-wellbeing': {
    name: '사회적 복지',
    icon: '🌍',
    description: '사회 전체의 이익과 지속가능성을 고려해야 함'
  }
}

const KOREAN_AI_ACT_CHECKLIST: ComplianceItem[] = [
  { id: 'k1', category: '신뢰성', requirement: 'AI 시스템의 안정성 및 견고성 확보', checked: false },
  { id: 'k2', category: '신뢰성', requirement: '오작동 및 오류 발생 시 대응 방안 수립', checked: false },
  { id: 'k3', category: '투명성', requirement: 'AI 활용 사실 고지 의무', checked: false },
  { id: 'k4', category: '투명성', requirement: '자동화된 결정에 대한 설명 제공', checked: false },
  { id: 'k5', category: '다양성', requirement: '데이터 편향성 점검 및 완화', checked: false },
  { id: 'k6', category: '다양성', requirement: '소수자 및 취약 계층 보호', checked: false },
  { id: 'k7', category: '침해금지', requirement: '인간의 존엄성 및 기본권 침해 방지', checked: false },
  { id: 'k8', category: '침해금지', requirement: '차별 및 편견 방지 조치', checked: false },
  { id: 'k9', category: '책임성', requirement: '개발·제공자의 손해배상 책임 명시', checked: false },
  { id: 'k10', category: '책임성', requirement: 'AI 영향평가 수행 및 문서화', checked: false },
  { id: 'k11', category: '개인정보', requirement: '개인정보보호법 준수', checked: false },
  { id: 'k12', category: '개인정보', requirement: '데이터 최소 수집 및 익명화', checked: false }
]

export default function EthicsFramework() {
  const [riskLevel, setRiskLevel] = useState<RiskLevel>('high-risk')
  const [selectedPrinciples, setSelectedPrinciples] = useState<EthicalPrinciple[]>([
    'transparency',
    'accountability',
    'fairness'
  ])
  const [complianceChecklist, setComplianceChecklist] = useState<ComplianceItem[]>(KOREAN_AI_ACT_CHECKLIST)
  const [likelihood, setLikelihood] = useState(3)
  const [impact, setImpact] = useState(4)

  const togglePrinciple = (principle: EthicalPrinciple) => {
    setSelectedPrinciples(prev =>
      prev.includes(principle)
        ? prev.filter(p => p !== principle)
        : [...prev, principle]
    )
  }

  const toggleCompliance = (id: string) => {
    setComplianceChecklist(prev =>
      prev.map(item =>
        item.id === id ? { ...item, checked: !item.checked } : item
      )
    )
  }

  const complianceScore = Math.round(
    (complianceChecklist.filter(item => item.checked).length / complianceChecklist.length) * 100
  )

  const riskScore = likelihood * impact
  const getRiskCategory = (score: number) => {
    if (score >= 15) return { level: 'Critical', color: 'text-red-600 dark:text-red-400' }
    if (score >= 10) return { level: 'High', color: 'text-orange-600 dark:text-orange-400' }
    if (score >= 6) return { level: 'Medium', color: 'text-yellow-600 dark:text-yellow-400' }
    return { level: 'Low', color: 'text-green-600 dark:text-green-400' }
  }

  const riskCategory = getRiskCategory(riskScore)

  const exportFramework = () => {
    const framework = {
      riskLevel,
      ethicalPrinciples: selectedPrinciples.map(p => ETHICAL_PRINCIPLES[p].name),
      complianceScore,
      complianceChecklist: complianceChecklist.filter(item => item.checked).map(item => ({
        category: item.category,
        requirement: item.requirement
      })),
      riskAssessment: {
        likelihood,
        impact,
        score: riskScore,
        category: riskCategory.level
      },
      timestamp: new Date().toISOString()
    }

    const blob = new Blob([JSON.stringify(framework, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ethics-framework-${Date.now()}.json`
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
                <FileCheck className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Ethics Framework Builder
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  맞춤형 AI 윤리 프레임워크를 구축하세요
                </p>
              </div>
            </div>

            <button
              onClick={exportFramework}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-rose-500 to-pink-600 text-white rounded-lg hover:from-rose-600 hover:to-pink-700 transition-all shadow-lg"
            >
              <Download className="w-5 h-5" />
              <span>프레임워크 내보내기</span>
            </button>
          </div>
        </div>

        {/* EU AI Act Risk Classification */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            EU AI Act 위험 분류
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {(Object.keys(RISK_LEVELS) as RiskLevel[]).map((level) => {
              const risk = RISK_LEVELS[level]
              const isSelected = riskLevel === level
              return (
                <button
                  key={level}
                  onClick={() => setRiskLevel(level)}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    isSelected
                      ? `border-2 ${risk.borderColor} bg-gradient-to-br ${risk.bgColor}`
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                >
                  <h3 className={`font-bold mb-2 ${isSelected ? `bg-gradient-to-r ${risk.color} bg-clip-text text-transparent` : 'text-gray-900 dark:text-white'}`}>
                    {risk.name}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    {risk.description}
                  </p>
                  <div className="text-xs text-gray-500 dark:text-gray-500">
                    <strong>예시:</strong>
                    <ul className="list-disc list-inside mt-1 space-y-0.5">
                      {risk.examples.slice(0, 2).map((ex, idx) => (
                        <li key={idx}>{ex}</li>
                      ))}
                    </ul>
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Ethical Principles */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            윤리적 원칙 선택
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {(Object.keys(ETHICAL_PRINCIPLES) as EthicalPrinciple[]).map((principle) => {
              const p = ETHICAL_PRINCIPLES[principle]
              const isSelected = selectedPrinciples.includes(principle)
              return (
                <button
                  key={principle}
                  onClick={() => togglePrinciple(principle)}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    isSelected
                      ? 'border-rose-500 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-rose-300 dark:hover:border-rose-700'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">{p.icon}</span>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                        {p.name}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {p.description}
                      </p>
                    </div>
                    {isSelected && (
                      <CheckCircle className="w-5 h-5 text-rose-600 dark:text-rose-400" />
                    )}
                  </div>
                </button>
              )
            })}
          </div>

          <div className="mt-6 p-4 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
            <p className="font-semibold text-gray-900 dark:text-white">
              선택된 원칙: {selectedPrinciples.length}개
            </p>
            <div className="flex flex-wrap gap-2 mt-2">
              {selectedPrinciples.map(p => (
                <span
                  key={p}
                  className="px-3 py-1 bg-white dark:bg-gray-800 rounded-full text-sm font-medium text-rose-600 dark:text-rose-400"
                >
                  {ETHICAL_PRINCIPLES[p].name}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Risk Assessment Matrix */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
              위험 평가 매트릭스
            </h2>

            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  발생 가능성 (Likelihood): {likelihood}
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={likelihood}
                  onChange={(e) => setLikelihood(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span>1 (희박)</span>
                  <span>5 (확실)</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  영향도 (Impact): {impact}
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={impact}
                  onChange={(e) => setImpact(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span>1 (경미)</span>
                  <span>5 (심각)</span>
                </div>
              </div>

              <div className="p-6 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
                <div className="text-center">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">위험 점수</p>
                  <p className={`text-5xl font-bold ${riskCategory.color}`}>
                    {riskScore}
                  </p>
                  <p className={`text-lg font-semibold mt-2 ${riskCategory.color}`}>
                    {riskCategory.level} Risk
                  </p>
                </div>
              </div>

              {/* 5x5 Risk Matrix Visual */}
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div className="grid grid-cols-6 gap-1 text-xs">
                  <div></div>
                  {[1, 2, 3, 4, 5].map(i => (
                    <div key={i} className="text-center font-semibold text-gray-600 dark:text-gray-400">
                      {i}
                    </div>
                  ))}
                  {[5, 4, 3, 2, 1].map(l => (
                    <div key={l} className="contents">
                      <div className="font-semibold text-gray-600 dark:text-gray-400 flex items-center justify-center">
                        {l}
                      </div>
                      {[1, 2, 3, 4, 5].map(i => {
                        const score = l * i
                        const isCurrentCell = l === likelihood && i === impact
                        let bgColor = 'bg-green-200 dark:bg-green-900/40'
                        if (score >= 15) bgColor = 'bg-red-200 dark:bg-red-900/40'
                        else if (score >= 10) bgColor = 'bg-orange-200 dark:bg-orange-900/40'
                        else if (score >= 6) bgColor = 'bg-yellow-200 dark:bg-yellow-900/40'

                        return (
                          <div
                            key={i}
                            className={`aspect-square flex items-center justify-center rounded ${bgColor} ${
                              isCurrentCell ? 'ring-4 ring-rose-500 font-bold' : ''
                            }`}
                          >
                            {isCurrentCell && '●'}
                          </div>
                        )
                      })}
                    </div>
                  ))}
                </div>
                <div className="mt-2 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Impact (영향도) →
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    ↑ Likelihood (가능성)
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Korean AI Act Compliance */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                한국 AI 기본법 준수
              </h2>
              <div className="text-right">
                <p className="text-3xl font-bold text-rose-600 dark:text-rose-400">
                  {complianceScore}%
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">준수율</p>
              </div>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
              {complianceChecklist.map((item) => (
                <button
                  key={item.id}
                  onClick={() => toggleCompliance(item.id)}
                  className={`w-full p-3 rounded-lg border text-left transition-all ${
                    item.checked
                      ? 'border-green-500 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-rose-300 dark:hover:border-rose-700'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5">
                      {item.checked ? (
                        <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                      ) : (
                        <div className="w-5 h-5 rounded-full border-2 border-gray-300 dark:border-gray-600" />
                      )}
                    </div>
                    <div className="flex-1">
                      <span className="text-xs font-semibold text-rose-600 dark:text-rose-400">
                        {item.category}
                      </span>
                      <p className="text-sm text-gray-900 dark:text-white mt-1">
                        {item.requirement}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>

            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <strong>권장 준수율:</strong> 고위험 AI는 90% 이상, 제한적 위험은 70% 이상
              </p>
            </div>
          </div>
        </div>

        {/* Mitigation Plan */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            완화 계획 생성
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {riskScore >= 10 && (
              <div className="p-4 bg-gradient-to-br from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20 rounded-lg border border-red-200 dark:border-red-800">
                <div className="flex items-start gap-3">
                  <XCircle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                      즉각적 조치 필요
                    </h3>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• 위험 평가 재실시</li>
                      <li>• 전문가 검토 요청</li>
                      <li>• 시스템 배포 중단 고려</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}

            <div className="p-4 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    기술적 통제
                  </h3>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 모델 모니터링 강화</li>
                    <li>• 편향 탐지 시스템 구축</li>
                    <li>• 설명 가능성 향상</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-start gap-3">
                <FileCheck className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    정책/프로세스
                  </h3>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 윤리 가이드라인 수립</li>
                    <li>• 정기 감사 실시</li>
                    <li>• 이해관계자 참여</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
