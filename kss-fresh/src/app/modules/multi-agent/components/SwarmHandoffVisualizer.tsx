'use client'

import React, { useState } from 'react'
import { Play, RefreshCw, ArrowRight, User, Bot } from 'lucide-react'

interface Agent {
  id: string
  name: string
  role: string
  color: string
}

interface HandoffStep {
  from: string
  to: string
  reason: string
  context: Record<string, unknown>
  timestamp: number
}

interface Scenario {
  name: string
  description: string
  initialAgent: string
  initialMessage: string
}

const agents: Agent[] = [
  { id: 'triage', name: 'Triage Agent', role: '요청 분류', color: 'from-purple-500 to-purple-700' },
  { id: 'flight', name: 'Flight Agent', role: '항공편 조회/변경', color: 'from-blue-500 to-blue-700' },
  { id: 'refund', name: 'Refund Agent', role: '환불 처리', color: 'from-green-500 to-green-700' },
  { id: 'billing', name: 'Billing Agent', role: '결제/청구', color: 'from-amber-500 to-amber-700' },
  { id: 'human', name: 'Human Agent', role: '상담원 연결', color: 'from-red-500 to-red-700' }
]

const scenarios: Scenario[] = [
  {
    name: '항공권 취소 & 환불',
    description: '고객이 항공권을 취소하고 환불을 요청하는 시나리오',
    initialAgent: 'triage',
    initialMessage: '내일 비행기 취소하고 환불받고 싶어요'
  },
  {
    name: '결제 오류',
    description: '이중 결제된 항공권의 환불 처리',
    initialAgent: 'triage',
    initialMessage: '결제가 두 번 되었어요'
  },
  {
    name: '복잡한 일정 변경',
    description: '다구간 항공권의 일정 변경',
    initialAgent: 'triage',
    initialMessage: '출장 일정이 변경되어 3구간 모두 변경해야 해요'
  }
]

export default function SwarmHandoffVisualizer() {
  const [handoffHistory, setHandoffHistory] = useState<HandoffStep[]>([])
  const [currentAgent, setCurrentAgent] = useState<string>('triage')
  const [isRunning, setIsRunning] = useState(false)
  const [selectedScenario, setSelectedScenario] = useState<Scenario>(scenarios[0])
  const [contextVariables, setContextVariables] = useState<Record<string, unknown>>({
    booking_id: 'KE1234',
    customer_tier: 'Gold',
    total_amount: 850000
  })

  const executeScenario = async () => {
    setIsRunning(true)
    setHandoffHistory([])
    setCurrentAgent('triage')

    await new Promise(resolve => setTimeout(resolve, 1000))

    // Scenario 1: 항공권 취소 & 환불
    if (selectedScenario.name === '항공권 취소 & 환불') {
      const steps: HandoffStep[] = [
        {
          from: 'triage',
          to: 'flight',
          reason: '항공권 취소 요청 감지',
          context: { ...contextVariables, intent: 'cancel_flight' },
          timestamp: Date.now()
        },
        {
          from: 'flight',
          to: 'refund',
          reason: '취소 완료, 환불 처리 필요',
          context: { ...contextVariables, flight_status: 'cancelled', refund_amount: 850000 },
          timestamp: Date.now() + 2000
        },
        {
          from: 'refund',
          to: 'end',
          reason: '환불 승인 완료',
          context: { ...contextVariables, refund_status: 'approved', expected_days: 7 },
          timestamp: Date.now() + 4000
        }
      ]

      for (const step of steps) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        setHandoffHistory(prev => [...prev, step])
        setCurrentAgent(step.to)
        setContextVariables(step.context)
      }
    }
    // Scenario 2: 결제 오류
    else if (selectedScenario.name === '결제 오류') {
      const steps: HandoffStep[] = [
        {
          from: 'triage',
          to: 'billing',
          reason: '이중 결제 이슈 감지',
          context: { ...contextVariables, issue: 'double_charge', duplicate_payment: true },
          timestamp: Date.now()
        },
        {
          from: 'billing',
          to: 'refund',
          reason: '중복 결제 확인, 환불 필요',
          context: { ...contextVariables, verified: true, refund_amount: 850000 },
          timestamp: Date.now() + 2000
        },
        {
          from: 'refund',
          to: 'end',
          reason: '환불 처리 완료',
          context: { ...contextVariables, refund_status: 'processed', ref_number: 'REF12345' },
          timestamp: Date.now() + 4000
        }
      ]

      for (const step of steps) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        setHandoffHistory(prev => [...prev, step])
        setCurrentAgent(step.to)
        setContextVariables(step.context)
      }
    }
    // Scenario 3: 복잡한 일정 변경
    else {
      const steps: HandoffStep[] = [
        {
          from: 'triage',
          to: 'flight',
          reason: '다구간 일정 변경 요청',
          context: { ...contextVariables, segments: 3, complexity: 'high' },
          timestamp: Date.now()
        },
        {
          from: 'flight',
          to: 'human',
          reason: '복잡도가 높아 상담원 지원 필요',
          context: { ...contextVariables, reason: 'complex_multi_leg', agent_notes: '3구간 동시 변경' },
          timestamp: Date.now() + 2000
        }
      ]

      for (const step of steps) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        setHandoffHistory(prev => [...prev, step])
        setCurrentAgent(step.to)
        setContextVariables(step.context)
      }
    }

    setIsRunning(false)
  }

  const reset = () => {
    setHandoffHistory([])
    setCurrentAgent('triage')
    setContextVariables({
      booking_id: 'KE1234',
      customer_tier: 'Gold',
      total_amount: 850000
    })
  }

  const getAgentById = (id: string) => agents.find(a => a.id === id)

  return (
    <div className="w-full bg-gradient-to-br from-orange-900 via-red-900 to-orange-900 rounded-xl p-6 text-white">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">Swarm Handoff Visualizer</h3>
        <p className="text-orange-200">에이전트 간 작업 이관(Handoff) 과정을 실시간으로 시각화합니다</p>
      </div>

      {/* Scenario Selection */}
      <div className="mb-6">
        <h4 className="text-sm font-semibold text-orange-200 mb-2">시나리오 선택</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          {scenarios.map(scenario => (
            <button
              key={scenario.name}
              onClick={() => { setSelectedScenario(scenario); reset(); }}
              className={`p-3 rounded-lg text-left transition-colors ${
                selectedScenario.name === scenario.name
                  ? 'bg-orange-600'
                  : 'bg-orange-800 hover:bg-orange-700'
              }`}
            >
              <div className="font-semibold text-sm">{scenario.name}</div>
              <div className="text-xs text-orange-200 mt-1">{scenario.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={executeScenario}
          disabled={isRunning}
          className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 rounded-lg flex items-center gap-2 transition-colors"
        >
          <Play className="w-4 h-4" />
          실행
        </button>
        <button
          onClick={reset}
          disabled={isRunning}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 rounded-lg flex items-center gap-2 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          초기화
        </button>
      </div>

      {/* Agent Status */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        {agents.map(agent => (
          <div
            key={agent.id}
            className={`p-3 rounded-lg border-2 transition-all ${
              currentAgent === agent.id
                ? `bg-gradient-to-br ${agent.color} border-white shadow-lg scale-105`
                : 'bg-orange-800/30 border-orange-700'
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              {agent.id === 'human' ? (
                <User className="w-4 h-4" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
              <span className="font-semibold text-sm">{agent.name}</span>
            </div>
            <div className="text-xs text-orange-200">{agent.role}</div>
            {currentAgent === agent.id && (
              <div className="mt-2 text-xs bg-white/20 rounded px-2 py-1">현재 활성</div>
            )}
          </div>
        ))}
      </div>

      {/* Handoff History */}
      <div className="bg-orange-800/30 rounded-lg p-4 mb-4">
        <h4 className="text-lg font-semibold mb-3">Handoff 기록</h4>
        {handoffHistory.length === 0 ? (
          <p className="text-orange-200 text-sm">시나리오를 실행하여 handoff 과정을 확인하세요</p>
        ) : (
          <div className="space-y-3">
            {handoffHistory.map((step, index) => {
              const fromAgent = getAgentById(step.from)
              const toAgent = getAgentById(step.to === 'end' ? 'refund' : step.to)

              return (
                <div key={index} className="flex items-start gap-3 p-3 bg-orange-900/50 rounded-lg">
                  <div className="flex-shrink-0 w-6 h-6 bg-orange-600 rounded-full flex items-center justify-center text-xs font-bold">
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold">{fromAgent?.name}</span>
                      <ArrowRight className="w-4 h-4 text-orange-400" />
                      <span className="font-semibold">{step.to === 'end' ? '완료' : toAgent?.name}</span>
                    </div>
                    <p className="text-sm text-orange-200">{step.reason}</p>
                    <div className="mt-2 text-xs bg-black/20 rounded p-2 font-mono">
                      Context: {JSON.stringify(Object.keys(step.context).slice(0, 3))}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Context Variables */}
      <div className="bg-orange-800/30 rounded-lg p-4">
        <h4 className="text-lg font-semibold mb-3">Context Variables</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {Object.entries(contextVariables).map(([key, value]) => (
            <div key={key} className="bg-orange-900/50 rounded p-2">
              <div className="text-xs text-orange-300 mb-1">{key}</div>
              <div className="text-sm font-mono">{String(value)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 bg-gray-900 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-orange-200 mb-2">Swarm Code Example:</h4>
        <pre className="text-xs text-gray-300 overflow-x-auto">
{`from swarm import Swarm, Agent

def transfer_to_flight():
    return flight_agent

def transfer_to_refund():
    return refund_agent

triage_agent = Agent(
    name="Triage Agent",
    instructions="요청을 분류하고 적절한 에이전트로 연결",
    functions=[transfer_to_flight, transfer_to_refund]
)

client = Swarm()
response = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "${selectedScenario.initialMessage}"}],
    context_variables={"booking_id": "KE1234"}
)`}
        </pre>
      </div>
    </div>
  )
}
