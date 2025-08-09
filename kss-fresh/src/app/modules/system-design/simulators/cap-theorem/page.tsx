'use client'

import { useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, Server, Network, Shield, AlertTriangle, CheckCircle, XCircle, Info, Zap } from 'lucide-react'

type SystemMode = 'CP' | 'AP' | 'CA'
type NodeStatus = 'active' | 'partitioned' | 'inconsistent' | 'unavailable'

interface Node {
  id: number
  name: string
  status: NodeStatus
  data: string
  version: number
  lastUpdate: number
}

interface NetworkPartition {
  active: boolean
  nodes: number[]
}

export default function CAPTheoremSimulator() {
  const [mode, setMode] = useState<SystemMode>('CP')
  const [nodes, setNodes] = useState<Node[]>([
    { id: 1, name: 'Node A', status: 'active', data: 'Value: 100', version: 1, lastUpdate: Date.now() },
    { id: 2, name: 'Node B', status: 'active', data: 'Value: 100', version: 1, lastUpdate: Date.now() },
    { id: 3, name: 'Node C', status: 'active', data: 'Value: 100', version: 1, lastUpdate: Date.now() }
  ])
  
  const [partition, setPartition] = useState<NetworkPartition>({ active: false, nodes: [] })
  const [writeAttempts, setWriteAttempts] = useState(0)
  const [readAttempts, setReadAttempts] = useState(0)
  const [selectedNode, setSelectedNode] = useState<number | null>(null)

  // 네트워크 파티션 시뮬레이션
  const createPartition = () => {
    const partitionedNodes = [2, 3] // Node B와 C를 파티션
    setPartition({ active: true, nodes: partitionedNodes })
    
    setNodes(prev => prev.map(node => {
      if (partitionedNodes.includes(node.id)) {
        return { ...node, status: 'partitioned' }
      }
      return node
    }))
  }

  const healPartition = () => {
    setPartition({ active: false, nodes: [] })
    
    // 모드에 따라 다르게 처리
    if (mode === 'AP') {
      // AP 모드: 파티션 해제 후 최종 일관성 달성
      const latestVersion = Math.max(...nodes.map(n => n.version))
      const latestData = nodes.find(n => n.version === latestVersion)?.data || 'Value: 100'
      
      setNodes(prev => prev.map(node => ({
        ...node,
        status: 'active',
        data: latestData,
        version: latestVersion,
        lastUpdate: Date.now()
      })))
    } else {
      setNodes(prev => prev.map(node => ({
        ...node,
        status: 'active'
      })))
    }
  }

  // 쓰기 작업 시뮬레이션
  const performWrite = (nodeId: number) => {
    setWriteAttempts(prev => prev + 1)
    const newValue = `Value: ${Math.floor(Math.random() * 900) + 100}`
    
    if (partition.active) {
      // 파티션 상황에서의 처리
      if (mode === 'CP') {
        // CP: 일관성 우선 - 파티션된 노드에는 쓰기 불가
        if (partition.nodes.includes(nodeId)) {
          alert('쓰기 실패: 네트워크 파티션으로 인해 이 노드는 사용 불가능합니다.')
          return
        }
        
        // 파티션되지 않은 노드들만 업데이트
        setNodes(prev => prev.map(node => {
          if (!partition.nodes.includes(node.id)) {
            return {
              ...node,
              data: newValue,
              version: node.version + 1,
              lastUpdate: Date.now()
            }
          }
          return { ...node, status: 'unavailable' }
        }))
      } else if (mode === 'AP') {
        // AP: 가용성 우선 - 모든 노드에 쓰기 가능 (일관성 희생)
        setNodes(prev => prev.map(node => {
          if (node.id === nodeId || (!partition.nodes.includes(node.id) && !partition.nodes.includes(nodeId))) {
            return {
              ...node,
              data: newValue,
              version: node.version + 1,
              lastUpdate: Date.now(),
              status: partition.nodes.includes(node.id) ? 'inconsistent' : node.status
            }
          }
          return node
        }))
      }
    } else {
      // 정상 상황
      setNodes(prev => prev.map(node => ({
        ...node,
        data: newValue,
        version: node.version + 1,
        lastUpdate: Date.now()
      })))
    }
  }

  // 읽기 작업 시뮬레이션
  const performRead = (nodeId: number) => {
    setReadAttempts(prev => prev + 1)
    const node = nodes.find(n => n.id === nodeId)
    
    if (!node) return
    
    if (partition.active && mode === 'CP' && partition.nodes.includes(nodeId)) {
      alert(`읽기 실패: Node ${node.name}는 네트워크 파티션으로 인해 사용 불가능합니다.`)
    } else {
      alert(`읽기 성공: ${node.name}에서 "${node.data}" (Version ${node.version})`)
    }
  }

  const getModeDescription = () => {
    switch (mode) {
      case 'CP':
        return {
          title: 'Consistency + Partition Tolerance',
          description: '일관성과 분할 내성을 보장하지만 가용성을 희생합니다.',
          tradeoff: '네트워크 파티션 시 일부 노드가 사용 불가능해집니다.',
          examples: 'MongoDB, HBase, Redis',
          color: 'blue'
        }
      case 'AP':
        return {
          title: 'Availability + Partition Tolerance',
          description: '가용성과 분할 내성을 보장하지만 일관성을 희생합니다.',
          tradeoff: '네트워크 파티션 시 노드 간 데이터 불일치가 발생할 수 있습니다.',
          examples: 'Cassandra, CouchDB, DynamoDB',
          color: 'green'
        }
      case 'CA':
        return {
          title: 'Consistency + Availability',
          description: '일관성과 가용성을 보장하지만 분할 내성을 희생합니다.',
          tradeoff: '네트워크 파티션이 발생하면 시스템이 중단됩니다.',
          examples: 'Traditional RDBMS (단일 노드)',
          color: 'purple'
        }
    }
  }

  const modeInfo = getModeDescription()

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/system-design"
          className="inline-flex items-center text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          System Design 모듈로 돌아가기
        </Link>
        
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            CAP 이론 시각화
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Consistency, Availability, Partition Tolerance의 트레이드오프를 체험합니다
          </p>
        </div>
      </div>

      {/* Mode Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          시스템 모드 선택
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(['CP', 'AP', 'CA'] as SystemMode[]).map((m) => (
            <button
              key={m}
              onClick={() => {
                setMode(m)
                healPartition()
              }}
              className={`p-4 rounded-lg border-2 transition-all ${
                mode === m
                  ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/20'
                  : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
              }`}
            >
              <div className="font-semibold text-gray-900 dark:text-white mb-1">
                {m}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {m === 'CP' && 'Consistency + Partition'}
                {m === 'AP' && 'Availability + Partition'}
                {m === 'CA' && 'Consistency + Availability'}
              </div>
            </button>
          ))}
        </div>
        
        <div className={`mt-6 p-4 rounded-lg bg-${modeInfo.color}-50 dark:bg-${modeInfo.color}-950/20`}>
          <h4 className={`font-semibold text-${modeInfo.color}-900 dark:text-${modeInfo.color}-300 mb-2 flex items-center gap-2`}>
            <Info className="w-5 h-5" />
            {modeInfo.title}
          </h4>
          <p className={`text-sm text-${modeInfo.color}-700 dark:text-${modeInfo.color}-400 mb-2`}>
            {modeInfo.description}
          </p>
          <p className={`text-sm text-${modeInfo.color}-600 dark:text-${modeInfo.color}-500`}>
            <strong>트레이드오프:</strong> {modeInfo.tradeoff}
          </p>
          <p className={`text-sm text-${modeInfo.color}-600 dark:text-${modeInfo.color}-500 mt-1`}>
            <strong>예시:</strong> {modeInfo.examples}
          </p>
        </div>
      </div>

      {/* Network Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          네트워크 제어
        </h3>
        <div className="flex gap-4">
          <button
            onClick={createPartition}
            disabled={partition.active}
            className={`px-6 py-3 rounded-lg font-semibold transition-colors flex items-center gap-2 ${
              partition.active
                ? 'bg-gray-300 cursor-not-allowed text-gray-500'
                : 'bg-red-500 hover:bg-red-600 text-white'
            }`}
          >
            <AlertTriangle className="w-5 h-5" />
            네트워크 파티션 발생
          </button>
          
          <button
            onClick={healPartition}
            disabled={!partition.active}
            className={`px-6 py-3 rounded-lg font-semibold transition-colors flex items-center gap-2 ${
              !partition.active
                ? 'bg-gray-300 cursor-not-allowed text-gray-500'
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            <CheckCircle className="w-5 h-5" />
            파티션 복구
          </button>
        </div>
        
        {partition.active && (
          <div className="mt-4 p-3 bg-red-50 dark:bg-red-950/20 rounded-lg">
            <p className="text-sm text-red-700 dark:text-red-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              네트워크 파티션 활성화: Node A와 Node B, C 간 통신 불가
            </p>
          </div>
        )}
      </div>

      {/* Nodes Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          분산 노드 상태
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {nodes.map((node) => (
            <div
              key={node.id}
              className={`relative p-6 rounded-lg border-2 transition-all ${
                node.status === 'active' 
                  ? 'border-green-500 bg-green-50 dark:bg-green-950/20'
                  : node.status === 'partitioned'
                  ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20'
                  : node.status === 'inconsistent'
                  ? 'border-orange-500 bg-orange-50 dark:bg-orange-950/20'
                  : 'border-red-500 bg-red-50 dark:bg-red-950/20'
              }`}
            >
              {/* Network Connection Indicator */}
              {partition.active && (
                <div className="absolute -top-3 -right-3">
                  {partition.nodes.includes(node.id) ? (
                    <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center">
                      <Network className="w-4 h-4 text-white" />
                    </div>
                  ) : (
                    <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                      <CheckCircle className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
              )}
              
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Server className={`w-6 h-6 ${
                    node.status === 'active' ? 'text-green-600 dark:text-green-400' :
                    node.status === 'partitioned' ? 'text-yellow-600 dark:text-yellow-400' :
                    node.status === 'inconsistent' ? 'text-orange-600 dark:text-orange-400' :
                    'text-red-600 dark:text-red-400'
                  }`} />
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {node.name}
                  </span>
                </div>
                <span className={`text-xs px-2 py-1 rounded-full ${
                  node.status === 'active' 
                    ? 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200'
                    : node.status === 'partitioned'
                    ? 'bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200'
                    : node.status === 'inconsistent'
                    ? 'bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200'
                    : 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200'
                }`}>
                  {node.status === 'active' && '정상'}
                  {node.status === 'partitioned' && '파티션됨'}
                  {node.status === 'inconsistent' && '불일치'}
                  {node.status === 'unavailable' && '사용불가'}
                </span>
              </div>
              
              <div className="mb-4">
                <div className="font-mono text-lg text-gray-900 dark:text-white mb-1">
                  {node.data}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Version: {node.version} | 
                  Updated: {new Date(node.lastUpdate).toLocaleTimeString()}
                </div>
              </div>
              
              <div className="flex gap-2">
                <button
                  onClick={() => performWrite(node.id)}
                  disabled={mode === 'CP' && partition.active && partition.nodes.includes(node.id)}
                  className={`flex-1 px-3 py-2 rounded text-sm font-semibold transition-colors ${
                    mode === 'CP' && partition.active && partition.nodes.includes(node.id)
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-500 hover:bg-blue-600 text-white'
                  }`}
                >
                  쓰기
                </button>
                <button
                  onClick={() => performRead(node.id)}
                  disabled={mode === 'CP' && partition.active && partition.nodes.includes(node.id)}
                  className={`flex-1 px-3 py-2 rounded text-sm font-semibold transition-colors ${
                    mode === 'CP' && partition.active && partition.nodes.includes(node.id)
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-green-500 hover:bg-green-600 text-white'
                  }`}
                >
                  읽기
                </button>
              </div>
            </div>
          ))}
        </div>
        
        {/* Connection Lines */}
        {!partition.active && (
          <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-500" />
              모든 노드가 정상적으로 연결되어 있습니다
            </p>
          </div>
        )}
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            작업 통계
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 dark:text-gray-400">쓰기 시도</span>
              <span className="font-semibold text-gray-900 dark:text-white">{writeAttempts}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 dark:text-gray-400">읽기 시도</span>
              <span className="font-semibold text-gray-900 dark:text-white">{readAttempts}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 dark:text-gray-400">데이터 일관성</span>
              <span className="font-semibold text-gray-900 dark:text-white">
                {nodes.every(n => n.data === nodes[0].data) ? (
                  <span className="text-green-600 dark:text-green-400">일치</span>
                ) : (
                  <span className="text-red-600 dark:text-red-400">불일치</span>
                )}
              </span>
            </div>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            CAP 속성 상태
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                Consistency
              </span>
              {(mode === 'CP' || mode === 'CA' || !partition.active) ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Availability
              </span>
              {(mode === 'AP' || mode === 'CA' || !partition.active) ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600 dark:text-gray-400 flex items-center gap-2">
                <Network className="w-4 h-4" />
                Partition Tolerance
              </span>
              {(mode === 'CP' || mode === 'AP') ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}