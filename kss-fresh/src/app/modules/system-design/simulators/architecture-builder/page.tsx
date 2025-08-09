'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { 
  ArrowLeft, Server, Database, Cloud, Shield, Globe, 
  HardDrive, Network, Users, Lock, Activity, GitBranch,
  Cpu, Wifi, Save, Download, Trash2, MousePointer
} from 'lucide-react'

interface Component {
  id: string
  type: string
  name: string
  x: number
  y: number
  icon: any
  color: string
}

interface Connection {
  id: string
  from: string
  to: string
  label?: string
}

const componentTypes = [
  { type: 'client', name: 'Client', icon: Users, color: 'blue' },
  { type: 'loadbalancer', name: 'Load Balancer', icon: Network, color: 'purple' },
  { type: 'webserver', name: 'Web Server', icon: Server, color: 'green' },
  { type: 'appserver', name: 'App Server', icon: Cpu, color: 'indigo' },
  { type: 'database', name: 'Database', icon: Database, color: 'yellow' },
  { type: 'cache', name: 'Cache', icon: HardDrive, color: 'red' },
  { type: 'queue', name: 'Message Queue', icon: GitBranch, color: 'pink' },
  { type: 'cdn', name: 'CDN', icon: Globe, color: 'cyan' },
  { type: 'storage', name: 'Storage', icon: Cloud, color: 'gray' },
  { type: 'firewall', name: 'Firewall', icon: Shield, color: 'orange' }
]

export default function ArchitectureBuilder() {
  const searchParams = useSearchParams()
  const backUrl = searchParams.get('from') || '/modules/system-design'
  const canvasRef = useRef<HTMLDivElement>(null)
  const [components, setComponents] = useState<Component[]>([])
  const [connections, setConnections] = useState<Connection[]>([])
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null)
  const [selectedType, setSelectedType] = useState<string>('client')
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectingFrom, setConnectingFrom] = useState<string | null>(null)
  const [draggedComponent, setDraggedComponent] = useState<Component | null>(null)
  const [offset, setOffset] = useState({ x: 0, y: 0 })

  // Add component to canvas
  const addComponent = (type: string) => {
    const componentType = componentTypes.find(ct => ct.type === type)
    if (!componentType) return

    const newComponent: Component = {
      id: `${type}_${Date.now()}`,
      type,
      name: componentType.name,
      x: Math.random() * 400 + 50,
      y: Math.random() * 300 + 50,
      icon: componentType.icon,
      color: componentType.color
    }

    setComponents(prev => [...prev, newComponent])
  }

  // Handle component drag
  const handleMouseDown = (e: React.MouseEvent, component: Component) => {
    if (isConnecting) {
      // Handle connection
      if (!connectingFrom) {
        setConnectingFrom(component.id)
      } else if (connectingFrom !== component.id) {
        const newConnection: Connection = {
          id: `conn_${Date.now()}`,
          from: connectingFrom,
          to: component.id
        }
        setConnections(prev => [...prev, newConnection])
        setConnectingFrom(null)
        setIsConnecting(false)
      }
    } else {
      // Handle drag
      const rect = canvasRef.current?.getBoundingClientRect()
      if (rect) {
        setOffset({
          x: e.clientX - rect.left - component.x,
          y: e.clientY - rect.top - component.y
        })
        setDraggedComponent(component)
      }
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (draggedComponent && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect()
      const newX = e.clientX - rect.left - offset.x
      const newY = e.clientY - rect.top - offset.y

      setComponents(prev => prev.map(comp => 
        comp.id === draggedComponent.id 
          ? { ...comp, x: newX, y: newY }
          : comp
      ))
    }
  }

  const handleMouseUp = () => {
    setDraggedComponent(null)
  }

  // Delete component
  const deleteComponent = (id: string) => {
    setComponents(prev => prev.filter(comp => comp.id !== id))
    setConnections(prev => prev.filter(conn => conn.from !== id && conn.to !== id))
    setSelectedComponent(null)
  }

  // Clear all
  const clearCanvas = () => {
    setComponents([])
    setConnections([])
    setSelectedComponent(null)
    setIsConnecting(false)
    setConnectingFrom(null)
  }

  // Export architecture
  const exportArchitecture = () => {
    const data = {
      components,
      connections,
      timestamp: new Date().toISOString()
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'architecture.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  // Load sample architecture
  const loadSample = () => {
    const sampleComponents: Component[] = [
      { id: 'client_1', type: 'client', name: 'Client', x: 50, y: 200, icon: Users, color: 'blue' },
      { id: 'cdn_1', type: 'cdn', name: 'CDN', x: 200, y: 100, icon: Globe, color: 'cyan' },
      { id: 'lb_1', type: 'loadbalancer', name: 'Load Balancer', x: 350, y: 200, icon: Network, color: 'purple' },
      { id: 'web_1', type: 'webserver', name: 'Web Server', x: 500, y: 100, icon: Server, color: 'green' },
      { id: 'web_2', type: 'webserver', name: 'Web Server', x: 500, y: 200, icon: Server, color: 'green' },
      { id: 'web_3', type: 'webserver', name: 'Web Server', x: 500, y: 300, icon: Server, color: 'green' },
      { id: 'cache_1', type: 'cache', name: 'Redis Cache', x: 650, y: 100, icon: HardDrive, color: 'red' },
      { id: 'db_1', type: 'database', name: 'Primary DB', x: 650, y: 250, icon: Database, color: 'yellow' },
      { id: 'db_2', type: 'database', name: 'Replica DB', x: 650, y: 350, icon: Database, color: 'yellow' }
    ]

    const sampleConnections: Connection[] = [
      { id: 'conn_1', from: 'client_1', to: 'cdn_1' },
      { id: 'conn_2', from: 'client_1', to: 'lb_1' },
      { id: 'conn_3', from: 'cdn_1', to: 'lb_1' },
      { id: 'conn_4', from: 'lb_1', to: 'web_1' },
      { id: 'conn_5', from: 'lb_1', to: 'web_2' },
      { id: 'conn_6', from: 'lb_1', to: 'web_3' },
      { id: 'conn_7', from: 'web_1', to: 'cache_1' },
      { id: 'conn_8', from: 'web_2', to: 'cache_1' },
      { id: 'conn_9', from: 'web_3', to: 'cache_1' },
      { id: 'conn_10', from: 'web_1', to: 'db_1' },
      { id: 'conn_11', from: 'web_2', to: 'db_1' },
      { id: 'conn_12', from: 'web_3', to: 'db_1' },
      { id: 'conn_13', from: 'db_1', to: 'db_2', label: 'Replication' }
    ]

    setComponents(sampleComponents)
    setConnections(sampleConnections)
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <Link
          href={backUrl}
          className="inline-flex items-center text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          학습 페이지로 돌아가기
        </Link>
        
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            아키텍처 빌더
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            드래그 앤 드롭으로 시스템 아키텍처를 설계합니다
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Component Palette */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              컴포넌트
            </h3>
            
            <div className="space-y-2">
              {componentTypes.map((type) => {
                const Icon = type.icon
                return (
                  <button
                    key={type.type}
                    onClick={() => addComponent(type.type)}
                    className={`w-full p-3 rounded-lg border-2 transition-all flex items-center gap-3 ${
                      selectedType === type.type
                        ? `border-${type.color}-500 bg-${type.color}-50 dark:bg-${type.color}-950/20`
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                    }`}
                  >
                    <Icon className={`w-5 h-5 text-${type.color}-600 dark:text-${type.color}-400`} />
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {type.name}
                    </span>
                  </button>
                )
              })}
            </div>
            
            <div className="mt-6 space-y-2">
              <button
                onClick={() => {
                  setIsConnecting(!isConnecting)
                  setConnectingFrom(null)
                }}
                className={`w-full px-4 py-2 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                  isConnecting
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Wifi className="w-5 h-5" />
                {isConnecting ? '연결 모드' : '연결하기'}
              </button>
              
              <button
                onClick={loadSample}
                className="w-full px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" />
                샘플 불러오기
              </button>
              
              <button
                onClick={exportArchitecture}
                disabled={components.length === 0}
                className="w-full px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Save className="w-5 h-5" />
                내보내기
              </button>
              
              <button
                onClick={clearCanvas}
                className="w-full px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
              >
                <Trash2 className="w-5 h-5" />
                전체 삭제
              </button>
            </div>
          </div>
          
          {/* Instructions */}
          <div className="mt-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
              사용 방법
            </h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li className="flex items-start gap-2">
                <MousePointer className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>컴포넌트를 클릭하여 캔버스에 추가</span>
              </li>
              <li className="flex items-start gap-2">
                <MousePointer className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>드래그로 컴포넌트 이동</span>
              </li>
              <li className="flex items-start gap-2">
                <Wifi className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>연결 모드에서 컴포넌트 클릭으로 연결</span>
              </li>
              <li className="flex items-start gap-2">
                <Trash2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>컴포넌트 선택 후 Delete 키로 삭제</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Canvas */}
        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                아키텍처 캔버스
              </h3>
              {isConnecting && (
                <div className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-sm">
                  연결할 컴포넌트를 선택하세요
                </div>
              )}
            </div>
            
            <div
              ref={canvasRef}
              className="relative w-full h-[600px] bg-gray-50 dark:bg-gray-900 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 overflow-hidden"
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              {/* Connections */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                {connections.map((conn) => {
                  const fromComp = components.find(c => c.id === conn.from)
                  const toComp = components.find(c => c.id === conn.to)
                  
                  if (!fromComp || !toComp) return null
                  
                  return (
                    <g key={conn.id}>
                      <line
                        x1={fromComp.x + 40}
                        y1={fromComp.y + 40}
                        x2={toComp.x + 40}
                        y2={toComp.y + 40}
                        stroke="currentColor"
                        strokeWidth="2"
                        className="text-gray-400 dark:text-gray-600"
                        markerEnd="url(#arrowhead)"
                      />
                      {conn.label && (
                        <text
                          x={(fromComp.x + toComp.x) / 2 + 40}
                          y={(fromComp.y + toComp.y) / 2 + 40}
                          className="fill-gray-600 dark:fill-gray-400 text-xs"
                          textAnchor="middle"
                        >
                          {conn.label}
                        </text>
                      )}
                    </g>
                  )
                })}
                <defs>
                  <marker
                    id="arrowhead"
                    markerWidth="10"
                    markerHeight="10"
                    refX="9"
                    refY="3"
                    orient="auto"
                  >
                    <polygon
                      points="0 0, 10 3, 0 6"
                      className="fill-gray-400 dark:fill-gray-600"
                    />
                  </marker>
                </defs>
              </svg>
              
              {/* Components */}
              {components.map((component) => {
                const Icon = component.icon
                return (
                  <div
                    key={component.id}
                    className={`absolute w-20 h-20 flex flex-col items-center justify-center rounded-lg border-2 cursor-move transition-all ${
                      selectedComponent?.id === component.id
                        ? `border-${component.color}-500 bg-${component.color}-100 dark:bg-${component.color}-900/30 shadow-lg`
                        : connectingFrom === component.id
                        ? `border-purple-500 bg-purple-100 dark:bg-purple-900/30`
                        : `border-${component.color}-400 bg-white dark:bg-gray-800 hover:shadow-md`
                    }`}
                    style={{ left: component.x, top: component.y }}
                    onMouseDown={(e) => handleMouseDown(e, component)}
                    onClick={() => !isConnecting && setSelectedComponent(component)}
                  >
                    <Icon className={`w-8 h-8 text-${component.color}-600 dark:text-${component.color}-400`} />
                    <span className="text-xs font-medium text-gray-700 dark:text-gray-300 mt-1">
                      {component.name}
                    </span>
                  </div>
                )
              })}
              
              {components.length === 0 && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <p className="text-gray-400 dark:text-gray-500">
                    컴포넌트를 추가하여 아키텍처를 설계하세요
                  </p>
                </div>
              )}
            </div>
          </div>
          
          {/* Selected Component Info */}
          {selectedComponent && (
            <div className="mt-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-gray-900 dark:text-white">
                  컴포넌트 정보
                </h4>
                <button
                  onClick={() => deleteComponent(selectedComponent.id)}
                  className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded text-sm"
                >
                  삭제
                </button>
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">타입:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    {selectedComponent.name}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">ID:</span>
                  <span className="ml-2 font-mono text-xs text-gray-900 dark:text-white">
                    {selectedComponent.id}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">위치:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    ({Math.round(selectedComponent.x)}, {Math.round(selectedComponent.y)})
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">연결:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    {connections.filter(c => c.from === selectedComponent.id || c.to === selectedComponent.id).length}개
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}