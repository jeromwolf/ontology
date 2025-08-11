'use client'

import { useState, useEffect, useRef } from 'react'
import { Wifi, Radio, Car, Building, AlertTriangle, Activity, Play, Pause, Settings } from 'lucide-react'

interface V2XNode {
  id: string
  type: 'vehicle' | 'infrastructure' | 'pedestrian'
  x: number
  y: number
  vx?: number
  vy?: number
  signalStrength: number
  messages: V2XMessage[]
  color: string
}

interface V2XMessage {
  id: string
  from: string
  to: string
  type: 'BSM' | 'SPAT' | 'MAP' | 'PSM' | 'Emergency'
  content: any
  timestamp: number
  latency: number
}

interface NetworkMetrics {
  totalMessages: number
  avgLatency: number
  packetLoss: number
  throughput: number
  activeConnections: number
}

export default function V2XNetworkSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [isRunning, setIsRunning] = useState(false)
  const [scenario, setScenario] = useState<'intersection' | 'highway' | 'emergency'>('intersection')
  const [messageRate, setMessageRate] = useState(10) // Hz
  const [signalRange, setSignalRange] = useState(300) // meters
  const [showMessages, setShowMessages] = useState(true)
  const [showSignalRange, setShowSignalRange] = useState(true)
  const [nodes, setNodes] = useState<V2XNode[]>([])
  const [messages, setMessages] = useState<V2XMessage[]>([])
  const [metrics, setMetrics] = useState<NetworkMetrics>({
    totalMessages: 0,
    avgLatency: 0,
    packetLoss: 0,
    throughput: 0,
    activeConnections: 0
  })
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  const messageQueueRef = useRef<V2XMessage[]>([])
  const metricsRef = useRef({
    totalMessages: 0,
    totalLatency: 0,
    lostPackets: 0,
    sentPackets: 0
  })

  // 시나리오 초기화
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const initNodes: V2XNode[] = []

    switch (scenario) {
      case 'intersection':
        // 교차로 시나리오
        // RSU (Road Side Unit)
        initNodes.push({
          id: 'RSU-1',
          type: 'infrastructure',
          x: canvas.width / 2,
          y: canvas.height / 2,
          signalStrength: 100,
          messages: [],
          color: '#10B981'
        })

        // 차량들
        for (let i = 0; i < 8; i++) {
          const angle = (i / 8) * Math.PI * 2
          const radius = 200
          initNodes.push({
            id: `Vehicle-${i + 1}`,
            type: 'vehicle',
            x: canvas.width / 2 + Math.cos(angle) * radius,
            y: canvas.height / 2 + Math.sin(angle) * radius,
            vx: -Math.cos(angle) * 2,
            vy: -Math.sin(angle) * 2,
            signalStrength: 80 + Math.random() * 20,
            messages: [],
            color: '#3B82F6'
          })
        }

        // 보행자
        for (let i = 0; i < 4; i++) {
          initNodes.push({
            id: `Pedestrian-${i + 1}`,
            type: 'pedestrian',
            x: 100 + Math.random() * (canvas.width - 200),
            y: 100 + Math.random() * (canvas.height - 200),
            vx: (Math.random() - 0.5) * 1,
            vy: (Math.random() - 0.5) * 1,
            signalStrength: 50,
            messages: [],
            color: '#F59E0B'
          })
        }
        break

      case 'highway':
        // 고속도로 시나리오
        // RSU 여러 개
        for (let i = 0; i < 3; i++) {
          initNodes.push({
            id: `RSU-${i + 1}`,
            type: 'infrastructure',
            x: 200 + i * 300,
            y: 50,
            signalStrength: 100,
            messages: [],
            color: '#10B981'
          })
        }

        // 차량 군집
        for (let i = 0; i < 12; i++) {
          initNodes.push({
            id: `Vehicle-${i + 1}`,
            type: 'vehicle',
            x: 50 + (i % 3) * 100,
            y: 150 + Math.floor(i / 3) * 80,
            vx: 3 + Math.random() * 2,
            vy: (Math.random() - 0.5) * 0.5,
            signalStrength: 70 + Math.random() * 30,
            messages: [],
            color: '#3B82F6'
          })
        }
        break

      case 'emergency':
        // 응급 상황 시나리오
        // 응급 차량
        initNodes.push({
          id: 'Emergency-1',
          type: 'vehicle',
          x: 100,
          y: canvas.height / 2,
          vx: 5,
          vy: 0,
          signalStrength: 100,
          messages: [],
          color: '#EF4444'
        })

        // 일반 차량
        for (let i = 0; i < 10; i++) {
          initNodes.push({
            id: `Vehicle-${i + 1}`,
            type: 'vehicle',
            x: 200 + Math.random() * 400,
            y: 100 + Math.random() * (canvas.height - 200),
            vx: Math.random() * 2,
            vy: (Math.random() - 0.5) * 2,
            signalStrength: 60 + Math.random() * 40,
            messages: [],
            color: '#3B82F6'
          })
        }
        break
    }

    setNodes(initNodes)
  }, [scenario])

  // 메시지 생성 및 전송
  const generateMessage = (from: V2XNode, to: V2XNode, type: V2XMessage['type']) => {
    const distance = Math.sqrt((from.x - to.x) ** 2 + (from.y - to.y) ** 2)
    
    // 신호 범위 체크
    if (distance > signalRange) return null

    // 신호 강도에 따른 패킷 손실
    const signalQuality = Math.max(0, 1 - distance / signalRange) * from.signalStrength / 100
    if (Math.random() > signalQuality) {
      metricsRef.current.lostPackets++
      return null
    }

    // 지연 시간 계산 (거리와 네트워크 혼잡도에 따라)
    const baseLatency = 5 // ms
    const distanceLatency = distance * 0.01
    const congestionLatency = nodes.length * 0.5
    const totalLatency = baseLatency + distanceLatency + congestionLatency + Math.random() * 10

    const message: V2XMessage = {
      id: `msg-${Date.now()}-${Math.random()}`,
      from: from.id,
      to: to.id,
      type,
      content: generateMessageContent(type, from),
      timestamp: Date.now(),
      latency: totalLatency
    }

    metricsRef.current.sentPackets++
    metricsRef.current.totalMessages++
    metricsRef.current.totalLatency += totalLatency

    return message
  }

  const generateMessageContent = (type: V2XMessage['type'], node: V2XNode) => {
    switch (type) {
      case 'BSM': // Basic Safety Message
        return {
          position: { x: node.x, y: node.y },
          velocity: { vx: node.vx || 0, vy: node.vy || 0 },
          heading: Math.atan2(node.vy || 0, node.vx || 0),
          acceleration: 0,
          brakeStatus: false
        }
      case 'SPAT': // Signal Phase and Timing
        return {
          intersection: 'INT-001',
          phases: [
            { direction: 'N-S', state: 'green', timeRemaining: 15 },
            { direction: 'E-W', state: 'red', timeRemaining: 30 }
          ]
        }
      case 'MAP': // Map Data
        return {
          lanes: 4,
          speedLimit: 60,
          geometry: []
        }
      case 'PSM': // Personal Safety Message
        return {
          pedestrianType: 'walking',
          position: { x: node.x, y: node.y },
          heading: Math.random() * Math.PI * 2
        }
      case 'Emergency':
        return {
          emergencyType: 'ambulance',
          priority: 'high',
          eta: 120,
          route: []
        }
    }
  }

  // 애니메이션 및 시뮬레이션
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Canvas 크기 조정
    const resizeCanvas = () => {
      const container = canvas.parentElement
      if (container) {
        canvas.width = container.clientWidth
        canvas.height = container.clientHeight
      }
    }
    
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    let lastMessageTime = 0
    const messageInterval = 1000 / messageRate // ms

    const animate = (timestamp: number) => {
      if (!isRunning) return

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 배경
      ctx.fillStyle = '#1F2937'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // 그리드
      ctx.strokeStyle = 'rgba(75, 85, 99, 0.3)'
      ctx.lineWidth = 1
      for (let x = 0; x < canvas.width; x += 50) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, canvas.height)
        ctx.stroke()
      }
      for (let y = 0; y < canvas.height; y += 50) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      // 노드 업데이트
      const updatedNodes = nodes.map(node => {
        if (node.type === 'vehicle' || node.type === 'pedestrian') {
          let newX = node.x + (node.vx || 0)
          let newY = node.y + (node.vy || 0)

          // 경계 체크
          if (newX < 50 || newX > canvas.width - 50) {
            node.vx = -(node.vx || 0)
            newX = node.x + (node.vx || 0)
          }
          if (newY < 50 || newY > canvas.height - 50) {
            node.vy = -(node.vy || 0)
            newY = node.y + (node.vy || 0)
          }

          return { ...node, x: newX, y: newY }
        }
        return node
      })

      // 메시지 생성
      if (timestamp - lastMessageTime > messageInterval) {
        const newMessages: V2XMessage[] = []
        
        updatedNodes.forEach(fromNode => {
          updatedNodes.forEach(toNode => {
            if (fromNode.id === toNode.id) return

            // 메시지 타입 결정
            let messageType: V2XMessage['type'] = 'BSM'
            
            if (fromNode.type === 'infrastructure' && toNode.type === 'vehicle') {
              messageType = Math.random() > 0.5 ? 'SPAT' : 'MAP'
            } else if (fromNode.type === 'pedestrian') {
              messageType = 'PSM'
            } else if (fromNode.id.includes('Emergency')) {
              messageType = 'Emergency'
            }

            // 확률적으로 메시지 전송
            if (Math.random() < 0.3) {
              const msg = generateMessage(fromNode, toNode, messageType)
              if (msg) {
                newMessages.push(msg)
                messageQueueRef.current.push(msg)
              }
            }
          })
        })

        lastMessageTime = timestamp
      }

      // 신호 범위 그리기
      if (showSignalRange) {
        updatedNodes.forEach(node => {
          const effectiveRange = signalRange * (node.signalStrength / 100)
          
          ctx.strokeStyle = `rgba(${
            node.type === 'infrastructure' ? '16, 185, 129' : 
            node.type === 'vehicle' ? '59, 130, 246' : '245, 158, 11'
          }, 0.1)`
          ctx.fillStyle = `rgba(${
            node.type === 'infrastructure' ? '16, 185, 129' : 
            node.type === 'vehicle' ? '59, 130, 246' : '245, 158, 11'
          }, 0.05)`
          
          ctx.beginPath()
          ctx.arc(node.x, node.y, effectiveRange, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()
        })
      }

      // 메시지 시각화
      if (showMessages) {
        // 활성 메시지 필터링 (3초 이내)
        const activeMessages = messageQueueRef.current.filter(
          msg => Date.now() - msg.timestamp < 3000
        )
        messageQueueRef.current = activeMessages

        activeMessages.forEach(msg => {
          const fromNode = updatedNodes.find(n => n.id === msg.from)
          const toNode = updatedNodes.find(n => n.id === msg.to)
          
          if (fromNode && toNode) {
            const progress = (Date.now() - msg.timestamp) / 3000
            const x = fromNode.x + (toNode.x - fromNode.x) * progress
            const y = fromNode.y + (toNode.y - fromNode.y) * progress

            // 메시지 경로
            ctx.strokeStyle = `rgba(${
              msg.type === 'Emergency' ? '239, 68, 68' :
              msg.type === 'BSM' ? '59, 130, 246' :
              msg.type === 'SPAT' ? '16, 185, 129' :
              msg.type === 'PSM' ? '245, 158, 11' : '156, 163, 175'
            }, ${0.6 * (1 - progress)})`
            ctx.lineWidth = 2
            ctx.setLineDash([5, 5])
            ctx.beginPath()
            ctx.moveTo(fromNode.x, fromNode.y)
            ctx.lineTo(toNode.x, toNode.y)
            ctx.stroke()
            ctx.setLineDash([])

            // 메시지 아이콘
            ctx.fillStyle = ctx.strokeStyle
            ctx.beginPath()
            ctx.arc(x, y, 3, 0, Math.PI * 2)
            ctx.fill()
          }
        })
      }

      // 노드 그리기
      updatedNodes.forEach(node => {
        // 선택된 노드 하이라이트
        if (node.id === selectedNode) {
          ctx.strokeStyle = '#FBBF24'
          ctx.lineWidth = 3
          ctx.beginPath()
          ctx.arc(node.x, node.y, 25, 0, Math.PI * 2)
          ctx.stroke()
        }

        // 노드 아이콘
        ctx.save()
        ctx.translate(node.x, node.y)
        
        if (node.type === 'infrastructure') {
          // RSU
          ctx.fillStyle = '#10B981'
          ctx.fillRect(-15, -15, 30, 30)
          ctx.fillStyle = '#065F46'
          ctx.fillRect(-10, -10, 20, 20)
          
          // 안테나
          ctx.strokeStyle = '#10B981'
          ctx.lineWidth = 2
          for (let i = -1; i <= 1; i++) {
            ctx.beginPath()
            ctx.moveTo(i * 8, -15)
            ctx.lineTo(i * 8, -25)
            ctx.stroke()
            ctx.beginPath()
            ctx.arc(i * 8, -25, 3, 0, Math.PI * 2)
            ctx.stroke()
          }
        } else if (node.type === 'vehicle') {
          // 차량
          ctx.rotate(Math.atan2(node.vy || 0, node.vx || 0))
          ctx.fillStyle = node.color
          ctx.fillRect(-20, -10, 40, 20)
          ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
          ctx.fillRect(-15, -7, 30, 14)
          
          // 응급차량 표시
          if (node.id.includes('Emergency')) {
            ctx.fillStyle = '#FFF'
            ctx.font = 'bold 12px monospace'
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillText('🚨', 0, 0)
          }
        } else if (node.type === 'pedestrian') {
          // 보행자
          ctx.fillStyle = node.color
          ctx.beginPath()
          ctx.arc(0, 0, 8, 0, Math.PI * 2)
          ctx.fill()
          ctx.fillStyle = '#FFF'
          ctx.beginPath()
          ctx.arc(0, 0, 5, 0, Math.PI * 2)
          ctx.fill()
        }
        
        ctx.restore()

        // 노드 ID
        ctx.fillStyle = '#FFF'
        ctx.font = '10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(node.id, node.x, node.y + 25)
        
        // 신호 강도
        ctx.fillStyle = `rgba(255, 255, 255, 0.7)`
        ctx.font = '9px monospace'
        ctx.fillText(`${node.signalStrength}%`, node.x, node.y + 35)
      })

      // 메트릭 업데이트
      const activeConnections = messageQueueRef.current.filter(
        msg => Date.now() - msg.timestamp < 1000
      ).length

      setMetrics({
        totalMessages: metricsRef.current.totalMessages,
        avgLatency: metricsRef.current.totalMessages > 0 
          ? metricsRef.current.totalLatency / metricsRef.current.totalMessages 
          : 0,
        packetLoss: metricsRef.current.sentPackets > 0
          ? (metricsRef.current.lostPackets / metricsRef.current.sentPackets) * 100
          : 0,
        throughput: activeConnections * messageRate,
        activeConnections
      })

      setNodes(updatedNodes)
      animationRef.current = requestAnimationFrame(animate)
    }

    if (isRunning) {
      animate(0)
    }

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, messageRate, signalRange, showMessages, showSignalRange, nodes, selectedNode])

  // 마우스 클릭 핸들러
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    // 클릭한 노드 찾기
    const clickedNode = nodes.find(node => {
      const dist = Math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
      return dist < 30
    })
    
    setSelectedNode(clickedNode?.id || null)
  }

  const reset = () => {
    setIsRunning(false)
    messageQueueRef.current = []
    metricsRef.current = {
      totalMessages: 0,
      totalLatency: 0,
      lostPackets: 0,
      sentPackets: 0
    }
    setMetrics({
      totalMessages: 0,
      avgLatency: 0,
      packetLoss: 0,
      throughput: 0,
      activeConnections: 0
    })
  }

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-700 text-white p-4">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Radio className="w-6 h-6" />
          V2X 네트워크 시뮬레이터
        </h2>
        <p className="text-purple-100 mt-1">차량-사물 통신 네트워크 시각화 및 분석</p>
      </div>

      {/* 컨트롤 패널 */}
      <div className="bg-gray-800 p-4 border-b border-gray-700">
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-6 py-2 rounded-lg flex items-center gap-2 font-medium ${
              isRunning 
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            {isRunning ? '정지' : '시작'}
          </button>

          <button
            onClick={reset}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg"
          >
            리셋
          </button>

          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-gray-300">시나리오:</label>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value as any)}
              className="px-3 py-1 bg-gray-700 border border-gray-600 rounded-lg text-white"
            >
              <option value="intersection">교차로</option>
              <option value="highway">고속도로</option>
              <option value="emergency">응급상황</option>
            </select>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-gray-300">메시지 전송률:</label>
            <input
              type="range"
              min="1"
              max="30"
              value={messageRate}
              onChange={(e) => setMessageRate(parseInt(e.target.value))}
              className="w-24"
            />
            <span className="text-sm font-mono text-gray-300">{messageRate} Hz</span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-gray-300">신호 범위:</label>
            <input
              type="range"
              min="100"
              max="500"
              step="50"
              value={signalRange}
              onChange={(e) => setSignalRange(parseInt(e.target.value))}
              className="w-24"
            />
            <span className="text-sm font-mono text-gray-300">{signalRange}m</span>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-gray-300">
              <input
                type="checkbox"
                checked={showMessages}
                onChange={(e) => setShowMessages(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">메시지 표시</span>
            </label>
            
            <label className="flex items-center gap-2 text-gray-300">
              <input
                type="checkbox"
                checked={showSignalRange}
                onChange={(e) => setShowSignalRange(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">신호 범위 표시</span>
            </label>
          </div>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-5 gap-4 p-4">
        {/* Canvas - 4/5 공간 */}
        <div className="lg:col-span-4 bg-gray-800 rounded-lg overflow-hidden">
          <canvas
            ref={canvasRef}
            className="w-full h-full cursor-pointer"
            onClick={handleCanvasClick}
          />
        </div>

        {/* 사이드바 - 1/5 공간 */}
        <div className="space-y-4">
          {/* 네트워크 메트릭 */}
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              네트워크 상태
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">총 메시지</span>
                <span className="font-mono font-bold text-white">{metrics.totalMessages}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">평균 지연</span>
                <span className="font-mono font-bold text-white">{metrics.avgLatency.toFixed(1)}ms</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">패킷 손실</span>
                <span className={`font-mono font-bold ${
                  metrics.packetLoss > 10 ? 'text-red-400' : 
                  metrics.packetLoss > 5 ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {metrics.packetLoss.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">처리량</span>
                <span className="font-mono font-bold text-white">{metrics.throughput} msg/s</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">활성 연결</span>
                <span className="font-mono font-bold text-white">{metrics.activeConnections}</span>
              </div>
            </div>
          </div>

          {/* 선택된 노드 정보 */}
          {selectedNode && (
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                <Wifi className="w-5 h-5" />
                노드 정보
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">ID</span>
                  <span className="font-mono text-white">{selectedNode}</span>
                </div>
                {nodes.find(n => n.id === selectedNode) && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-400">타입</span>
                      <span className="font-mono text-white">
                        {nodes.find(n => n.id === selectedNode)?.type}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">신호 강도</span>
                      <span className="font-mono text-white">
                        {nodes.find(n => n.id === selectedNode)?.signalStrength}%
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* 메시지 타입 범례 */}
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-3">메시지 타입</h3>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded"></div>
                <span className="text-xs text-gray-300">BSM (기본 안전)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded"></div>
                <span className="text-xs text-gray-300">SPAT (신호 정보)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-500 rounded"></div>
                <span className="text-xs text-gray-300">PSM (보행자 안전)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded"></div>
                <span className="text-xs text-gray-300">Emergency (응급)</span>
              </div>
            </div>
          </div>

          {/* V2X 표준 정보 */}
          <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-800">
            <h4 className="font-semibold text-blue-300 mb-2 flex items-center gap-2">
              <Settings className="w-4 h-4" />
              V2X 표준
            </h4>
            <ul className="text-xs space-y-1 text-blue-200">
              <li>• IEEE 802.11p WAVE</li>
              <li>• SAE J2735 메시지 세트</li>
              <li>• ETSI ITS-G5 (유럽)</li>
              <li>• C-V2X (셀룰러 기반)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}