'use client'

import { notFound } from 'next/navigation'
import { Shield } from 'lucide-react'
import { aiSecurityMetadata } from '../../metadata'
import AdversarialAttackLab from '../../components/AdversarialAttackLab'
import ModelExtractionSimulator from '../../components/ModelExtractionSimulator'
import PrivacyAttackSimulator from '../../components/PrivacyAttackSimulator'
import DefenseMechanismsTester from '../../components/DefenseMechanismsTester'
import SecurityAuditTool from '../../components/SecurityAuditTool'

const simulatorComponents: Record<string, React.ComponentType> = {
  'adversarial-attack-lab': AdversarialAttackLab,
  'model-extraction': ModelExtractionSimulator,
  'privacy-attack': PrivacyAttackSimulator,
  'defense-mechanisms': DefenseMechanismsTester,
  'security-audit': SecurityAuditTool
}

export default function SimulatorPage({ params }: { params: { simulatorId: string } }) {
  const simulator = aiSecurityMetadata.simulators?.find(s => s.id === params.simulatorId)
  
  if (!simulator) {
    notFound()
  }

  const SimulatorComponent = simulatorComponents[params.simulatorId]
  
  if (!SimulatorComponent) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center py-16">
            <Shield className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              시뮬레이터 개발 중
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              {simulator.title} 시뮬레이터는 현재 개발 중입니다.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center gap-3 mb-2">
            <Shield className="w-8 h-8 text-red-600" />
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              {simulator.title}
            </h1>
          </div>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            {simulator.description}
          </p>
          <div className="flex items-center gap-4 mt-4">
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              simulator.difficulty === 'beginner' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
              simulator.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
              simulator.difficulty === 'advanced' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
              'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
            }`}>
              {simulator.difficulty === 'beginner' ? '초급' :
               simulator.difficulty === 'intermediate' ? '중급' :
               simulator.difficulty === 'advanced' ? '고급' : '전문가'}
            </span>
          </div>
        </div>
      </div>

      {/* Simulator Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <SimulatorComponent />
      </div>
    </div>
  )
}