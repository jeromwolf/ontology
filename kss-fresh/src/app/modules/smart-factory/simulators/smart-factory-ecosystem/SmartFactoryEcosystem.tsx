'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, Info, Zap, Database, Cpu, Cloud, Users, Settings } from 'lucide-react'

interface SmartFactoryEcosystemProps {
  backUrl?: string
}

interface EcosystemNode {
  id: string
  name: string
  category: 'equipment' | 'sensor' | 'system' | 'ai' | 'communication' | 'people'
  icon: string
  x: number
  y: number
  description: string
  chapterLink?: string
  connections: string[]
  color: string
  size: number
}

interface DataFlow {
  from: string
  to: string
  label: string
  active: boolean
  color: string
}

export default function SmartFactoryEcosystem({ backUrl = '/modules/smart-factory' }: SmartFactoryEcosystemProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationFrameRef = useRef(0)
  const [selectedNode, setSelectedNode] = useState<EcosystemNode | null>(null)
  const [dataFlowActive, setDataFlowActive] = useState(true)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [animationSpeed, setAnimationSpeed] = useState(50) // 기본 50ms (2배속)
  const [currentScenario, setCurrentScenario] = useState<string | null>(null)
  const [scenarioStep, setScenarioStep] = useState(0)
  const [currentActiveNode, setCurrentActiveNode] = useState<string | null>(null)
  const [completedNodes, setCompletedNodes] = useState<Set<string>>(new Set())

  // 클릭 사운드 함수
  const playClickSound = () => {
    try {
      // 웹 오디오 API를 사용한 간단한 클릭 사운드 생성
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      const oscillator = audioContext.createOscillator()
      const gainNode = audioContext.createGain()
      
      oscillator.connect(gainNode)
      gainNode.connect(audioContext.destination)
      
      oscillator.frequency.value = 800 // 800Hz 톤
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime) // 볼륨 10%
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1)
      
      oscillator.start(audioContext.currentTime)
      oscillator.stop(audioContext.currentTime + 0.1)
    } catch (error) {
      // 사운드 재생이 실패해도 기능에는 영향 없음
      console.log('Sound not available')
    }
  }

  // 시나리오 정의
  const scenarios = {
    'equipment-failure': {
      name: '장비 고장 시뮬레이션',
      description: 'CNC 가공기에 베어링 고장이 발생하여 전체 생산라인에 미치는 영향을 시뮬레이션합니다.',
      steps: [
        { nodeId: 'cnc-machines', effect: 'failure', message: '🔧 CNC 가공기 베어링 고장 발생!' },
        { nodeId: 'vibration-sensors', effect: 'alert', message: '📳 진동센서가 이상 진동 감지' },
        { nodeId: 'predictive-ai', effect: 'active', message: '🔮 예측정비 AI가 고장 원인 분석 중' },
        { nodeId: 'production-equipment', effect: 'slowdown', message: '⚙️ 생산설비 가동률 50% 감소' },
        { nodeId: 'operators', effect: 'alert', message: '👨‍🔧 긴급 정비팀 출동' },
        { nodeId: 'mes-system', effect: 'update', message: '📊 MES가 생산 일정 재조정' }
      ]
    },
    'ai-optimization': {
      name: 'AI 최적화 시뮬레이션',
      description: 'AI 시스템이 에너지 효율과 생산성을 동시에 개선하는 과정을 보여줍니다.',
      steps: [
        { nodeId: 'iot-sensors', effect: 'active', message: '📡 IoT 센서들이 데이터 수집 강화' },
        { nodeId: 'ai-brain', effect: 'processing', message: '🧠 AI가 빅데이터 분석 시작' },
        { nodeId: 'optimization-ai', effect: 'active', message: '⚡ 최적화 AI가 개선안 도출' },
        { nodeId: 'production-equipment', effect: 'optimize', message: '⚙️ 생산 효율 20% 향상!' },
        { nodeId: 'edge-cloud', effect: 'active', message: '☁️ 클라우드에서 추가 분석' },
        { nodeId: 'operators', effect: 'notification', message: '👨‍🔧 운영자에게 최적화 결과 알림' }
      ]
    },
    'quality-crisis': {
      name: '품질 위기 대응',
      description: '불량률 급증 상황에서 AI 품질관리 시스템이 원인을 찾고 해결하는 과정입니다.',
      steps: [
        { nodeId: 'vision-sensors', effect: 'alert', message: '👁️ 비전센서가 불량품 급증 감지' },
        { nodeId: 'quality-ai', effect: 'alert', message: '✅ 품질관리 AI 비상모드 가동' },
        { nodeId: 'production-equipment', effect: 'pause', message: '⚙️ 생산라인 일시정지' },
        { nodeId: 'temp-sensors', effect: 'alert', message: '🌡️ 온도센서가 이상 온도 발견' },
        { nodeId: 'operators', effect: 'action', message: '👨‍🔧 품질관리팀 긴급 투입' },
        { nodeId: 'mes-system', effect: 'update', message: '📊 불량품 격리 및 재작업 지시' }
      ]
    }
  }

  // 시나리오 실행 함수
  const startScenario = (scenarioId: string) => {
    console.log('시나리오 시작:', scenarioId)
    setDataFlowActive(false) // 시나리오 시작 시 데이터 플로우 정지
    setCurrentScenario(scenarioId)
    setScenarioStep(0)
    setCurrentActiveNode(null)
    setCompletedNodes(new Set())
    
    // 첫 번째 단계 실행
    executeScenarioStep(scenarioId, 0)
  }

  const executeScenarioStep = (scenarioId: string, step: number) => {
    const scenario = scenarios[scenarioId as keyof typeof scenarios]
    if (!scenario || step >= scenario.steps.length) return

    const currentStep = scenario.steps[step]
    console.log('시나리오 스텝 실행:', step, currentStep.nodeId, currentStep.effect)
    
    // 이전 활성 노드를 완료 상태로 변경
    if (currentActiveNode) {
      setCompletedNodes(prev => new Set([...prev, currentActiveNode]))
    }
    
    // 현재 단계의 노드를 활성 상태로 설정
    setCurrentActiveNode(currentStep.nodeId)
  }
  
  // 다음 단계로 수동 진행
  const nextScenarioStep = () => {
    if (!currentScenario) return
    
    const scenario = scenarios[currentScenario as keyof typeof scenarios]
    if (scenarioStep < scenario.steps.length - 1) {
      const nextStep = scenarioStep + 1
      setScenarioStep(nextStep)
      executeScenarioStep(currentScenario, nextStep)
    }
  }
  
  // 이전 단계로 돌아가기
  const prevScenarioStep = () => {
    if (!currentScenario || scenarioStep <= 0) return
    
    const prevStep = scenarioStep - 1
    setScenarioStep(prevStep)
    
    // 완료된 노드에서 현재 노드 제거
    const newCompletedNodes = new Set(completedNodes)
    newCompletedNodes.delete(currentActiveNode!)
    setCompletedNodes(newCompletedNodes)
    
    // 이전 단계 노드 활성화
    const scenario = scenarios[currentScenario as keyof typeof scenarios]
    setCurrentActiveNode(scenario.steps[prevStep].nodeId)
  }

  const stopScenario = () => {
    setCurrentScenario(null)
    setScenarioStep(0)
    setCurrentActiveNode(null)
    setCompletedNodes(new Set())
    setDataFlowActive(true) // 시나리오 종료 시 데이터 플로우 재시작
  }

  // 노드 효과 색상 결정 - 현재 활성 노드만 빨간색, 완료된 노드는 회색
  const getNodeEffectColor = (nodeId: string) => {
    if (!currentScenario) return null
    
    // 현재 활성 노드는 빨간색
    if (nodeId === currentActiveNode) {
      return '#DC2626' // 빨간색
    }
    
    // 완료된 노드는 회색
    if (completedNodes.has(nodeId)) {
      return '#9CA3AF' // 회색
    }
    
    return null
  }

  // 스마트팩토리 생태계 노드들
  const nodes: EcosystemNode[] = [
    // 중앙 핵심: 스마트팩토리
    {
      id: 'smart-factory-core',
      name: '스마트팩토리\n통합 플랫폼',
      category: 'system',
      icon: '🏭',
      x: 400,
      y: 300,
      description: '스마트팩토리의 두뇌 역할을 하는 통합 플랫폼입니다. IoT 센서, AI 시스템, MES/ERP, 생산설비를 연결하여 실시간 데이터 수집, 분석, 의사결정, 제어 명령을 통합 관리합니다.',
      color: '#8B5CF6',
      size: 80,
      connections: ['production-equipment', 'iot-sensors', 'ai-brain', 'mes-system', 'erp-system', 'operators']
    },
    
    // 생산 설비 (왼쪽 영역)
    {
      id: 'production-equipment',
      name: '생산설비',
      category: 'equipment',
      icon: '⚙️',
      x: 150,
      y: 200,
      description: '6축 산업로봇, CNC 가공기, 자동 조립라인 등 실제 제품을 만드는 물리적 장비들입니다. 중앙 플랫폼의 명령을 받아 정밀한 생산 작업을 수행하며, 센서를 통해 상태 정보를 실시간 전송합니다.',
      color: '#F59E0B',
      size: 60,
      connections: ['robots', 'cnc-machines', 'conveyor']
    },
    {
      id: 'robots',
      name: '산업로봇',
      category: 'equipment',
      icon: '🤖',
      x: 80,
      y: 120,
      description: '용접, 도장, 조립 작업을 수행하는 6축 산업로봇과 사람과 협업하는 협동로봇(Cobot)입니다. 밀리미터 단위 정밀도로 24시간 무정지 작업이 가능하며, AI 비전 시스템을 통해 제품 품질을 실시간 검사합니다.',
      color: '#F59E0B',
      size: 45,
      connections: ['production-equipment']
    },
    {
      id: 'cnc-machines',
      name: 'CNC 가공기',
      category: 'equipment',
      icon: '🔧',
      x: 80,
      y: 200,
      description: 'Computer Numerical Control 가공기계로 프로그램된 경로에 따라 금속, 플라스틱 등을 정밀하게 절삭, 드릴링, 밀링 가공합니다. 0.01mm 정밀도로 복잡한 3D 형상도 자동으로 가공 가능합니다.',
      color: '#F59E0B',
      size: 45,
      connections: ['production-equipment']
    },
    {
      id: 'conveyor',
      name: '컨베이어',
      category: 'equipment',
      icon: '📦',
      x: 80,
      y: 280,
      description: '벨트, 롤러, 체인 등으로 구성된 자동 물류 이송 시스템입니다. RFID 태그로 제품을 추적하며, 센서를 통해 적재량과 이송 속도를 실시간 모니터링합니다. AGV(무인운반차)와 연동되어 스마트 물류를 구현합니다.',
      color: '#F59E0B',
      size: 45,
      connections: ['production-equipment']
    },

    // 센서 네트워크 (상단 영역)
    {
      id: 'iot-sensors',
      name: 'IoT 센서\n네트워크',
      category: 'sensor',
      icon: '📡',
      x: 400,
      y: 150,
      description: '공장 전체에 설치된 수백 개의 센서들이 온도, 압력, 진동, 습도, 전력 사용량 등을 24시간 실시간으로 측정합니다. 이 데이터는 5G 네트워크를 통해 중앙 플랫폼으로 전송되어 AI 분석의 기반이 됩니다.',
      color: '#10B981',
      size: 60,
      connections: ['temp-sensors', 'vibration-sensors', 'vision-sensors']
    },
    {
      id: 'temp-sensors',
      name: '온도센서',
      category: 'sensor',
      icon: '🌡️',
      x: 320,
      y: 80,
      description: '생산 환경의 온도를 0.1도 단위로 정밀 측정하는 디지털 온도센서입니다. 과열로 인한 장비 손상을 방지하고, 온도에 민감한 제품의 품질을 보장합니다. 실시간 알람으로 이상 온도 감지 시 즉시 대응 가능합니다.',
      color: '#10B981',
      size: 35,
      connections: ['iot-sensors']
    },
    {
      id: 'vibration-sensors',
      name: '진동센서',
      category: 'sensor',
      icon: '📳',
      x: 400,
      y: 80,
      description: '회전 장비의 진동을 측정하여 베어링 마모, 불균형, 정렬 불량 등을 조기 감지하는 가속도계입니다. FFT 분석을 통해 주파수별 진동 패턴을 분석하여 예측정비 시점을 정확하게 판단합니다.',
      color: '#10B981',
      size: 35,
      connections: ['iot-sensors']
    },
    {
      id: 'vision-sensors',
      name: '비전센서',
      category: 'sensor',
      icon: '👁️',
      x: 480,
      y: 80,
      description: '고해상도 카메라와 머신비전 AI가 결합된 자동 품질 검사 시스템입니다. 제품의 크기, 색상, 표면 결함, 조립 상태를 마이크로미터 단위로 검사하여 사람의 눈으로는 발견하기 어려운 미세한 불량도 감지합니다.',
      color: '#10B981',
      size: 35,
      connections: ['iot-sensors']
    },

    // AI 두뇌 (우상단)
    {
      id: 'ai-brain',
      name: 'AI 분석\n엔진',
      category: 'ai',
      icon: '🧠',
      x: 650,
      y: 150,
      description: '머신러닝과 딥러닝을 활용한 인공지능 시스템입니다. 장비 고장을 미리 예측하고, 생산 공정을 최적화하며, 불량품을 자동으로 감지합니다. 수집된 빅데이터를 실시간으로 분석하여 최적의 운영 방안을 제시합니다.',
      color: '#A855F7', // AI 시스템을 보라색으로 변경
      size: 60,
      connections: ['predictive-ai', 'quality-ai', 'optimization-ai']
    },
    {
      id: 'predictive-ai',
      name: '예측정비AI',
      category: 'ai',
      icon: '🔮',
      x: 580,
      y: 80,
      description: 'RNN과 LSTM 딥러닝 모델을 활용하여 장비의 RUL(Remaining Useful Life)을 예측합니다. 진동, 온도, 전류 패턴을 분석하여 고장 발생 2-4주 전에 미리 알려주어 계획된 정비로 다운타임을 최소화합니다.',
      color: '#A855F7', // AI 시스템을 보라색으로 변경
      size: 40,
      connections: ['ai-brain']
    },
    {
      id: 'quality-ai',
      name: '품질관리AI',
      category: 'ai',
      icon: '✅',
      x: 650,
      y: 80,
      description: 'CNN 컴퓨터 비전과 SVM 머신러닝이 결합된 실시간 품질 검사 AI입니다. 99.9% 이상의 정확도로 스크래치, 변색, 크랙, 변형 등을 자동 감지하며, 불량 유형별 자동 분류로 원인 분석을 지원합니다.',
      color: '#A855F7', // AI 시스템을 보라색으로 변경
      size: 40,
      connections: ['ai-brain']
    },
    {
      id: 'optimization-ai',
      name: '생산최적화AI',
      category: 'ai',
      icon: '⚡',
      x: 720,
      y: 80,
      description: '유전 알고리즘과 시뮬레이션을 통해 생산 스케줄링, 에너지 사용량, 재고 수준을 최적화하는 AI입니다. 수요 예측과 제약 조건을 고려하여 생산 효율 15-20% 향상과 에너지 비용 절감을 실현합니다.',
      color: '#A855F7', // AI 시스템을 보라색으로 변경
      size: 40,
      connections: ['ai-brain']
    },

    // 시스템 (하단)
    {
      id: 'mes-system',
      name: 'MES\n시스템',
      category: 'system',
      icon: '📊',
      x: 300,
      y: 450,
      description: 'Manufacturing Execution System - 생산 주문부터 완제품 출하까지 모든 제조 과정을 실시간 추적하고 관리합니다. 작업 지시, 품질 검사, 재고 관리, 생산 일정 등을 통합적으로 제어하여 효율적인 생산을 지원합니다.',
      color: '#3B82F6',
      size: 55,
      connections: ['smart-factory-core']
    },
    {
      id: 'erp-system',
      name: 'ERP\n시스템',
      category: 'system',
      icon: '💼',
      x: 500,
      y: 450,
      description: 'Enterprise Resource Planning - 회사의 모든 자원(인력, 자금, 자재)을 통합 관리하는 경영 시스템입니다. 주문 관리, 구매, 회계, 인사, 재무 등 기업 전체의 업무 프로세스를 연결하여 효율적인 경영을 지원합니다.',
      color: '#3B82F6',
      size: 55,
      connections: ['smart-factory-core']
    },

    // 통신 인프라 (우하단)
    {
      id: 'edge-cloud',
      name: '엣지/클라우드',
      category: 'communication',
      icon: '☁️',
      x: 650,
      y: 350,
      description: '현장의 엣지 컴퓨팅과 클라우드를 연결하는 핵심 인프라입니다. 중요한 데이터는 현장에서 즉시 처리하고, 대용량 분석은 클라우드에서 수행합니다. 5G 네트워크를 통해 초저지연 통신과 무제한 확장성을 제공합니다.',
      color: '#06B6D4',
      size: 50,
      connections: ['5g-network', 'data-lake']
    },
    {
      id: '5g-network',
      name: '5G 네트워크',
      category: 'communication',
      icon: '📶',
      x: 720,
      y: 280,
      description: '1ms 이하의 초저지연과 Gbps급 대용량 통신을 제공하는 5세대 이동통신 네트워크입니다. 수천 개의 IoT 기기를 동시 연결하고, AR/VR 원격 제어, 자율주행 AGV 등 실시간 제어가 필요한 응용을 지원합니다.',
      color: '#06B6D4',
      size: 40,
      connections: ['edge-cloud']
    },
    {
      id: 'data-lake',
      name: '데이터레이크',
      category: 'communication',
      icon: '🗄️',
      x: 720,
      y: 420,
      description: '센서 데이터, 생산 이력, 품질 정보 등 모든 제조 데이터를 저장하는 대용량 분산 데이터베이스입니다. 하둡, 스파크 등의 빅데이터 기술로 페타바이트급 데이터를 효율적으로 저장하고 실시간 분석을 지원합니다.',
      color: '#06B6D4',
      size: 40,
      connections: ['edge-cloud']
    },

    // 사람 (좌하단)
    {
      id: 'operators',
      name: '운영자',
      category: 'people',
      icon: '👨‍🔧',
      x: 150,
      y: 400,
      description: '스마트팩토리 운영의 핵심 인력들입니다. 숙련 기술자는 현장 장비를 직접 운영하고, 공정 엔지니어는 생산 최적화를 담당하며, 관리자는 전체 운영 상황을 모니터링합니다. HMI를 통해 AI 시스템과 협업합니다.',
      color: '#8B5CF6',
      size: 50,
      connections: ['hmi-interface']
    },
    {
      id: 'hmi-interface',
      name: 'HMI',
      category: 'system',
      icon: '📱',
      x: 80,
      y: 350,
      description: 'Human Machine Interface - 터치스크린, 대시보드, 모바일 앱 등을 통해 운영자가 기계와 소통하는 인터페이스. 생산 상태 모니터링, 장비 제어, 알람 확인 등이 가능합니다.',
      color: '#8B5CF6',
      size: 35,
      connections: ['operators']
    }
  ]

  // 데이터 흐름 화살표들
  const dataFlows: DataFlow[] = [
    { from: 'iot-sensors', to: 'smart-factory-core', label: '실시간 데이터', active: true, color: '#10B981' },
    { from: 'smart-factory-core', to: 'ai-brain', label: '분석 요청', active: true, color: '#EF4444' },
    { from: 'ai-brain', to: 'smart-factory-core', label: '인사이트', active: true, color: '#EF4444' },
    { from: 'smart-factory-core', to: 'production-equipment', label: '제어 신호', active: true, color: '#F59E0B' },
    { from: 'smart-factory-core', to: 'mes-system', label: '생산 데이터', active: true, color: '#3B82F6' },
    { from: 'mes-system', to: 'erp-system', label: '실적 연동', active: true, color: '#3B82F6' },
    { from: 'smart-factory-core', to: 'edge-cloud', label: '클라우드 백업', active: true, color: '#06B6D4' },
    { from: 'operators', to: 'smart-factory-core', label: '모니터링', active: true, color: '#8B5CF6' }
  ]

  // 캔버스 그리기
  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // 캔버스 크기 설정
    canvas.width = 800
    canvas.height = 600

    // 배경
    ctx.fillStyle = '#F8FAFC'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // 연결선 그리기
    nodes.forEach(node => {
      node.connections.forEach(connId => {
        const targetNode = nodes.find(n => n.id === connId)
        if (!targetNode) return

        ctx.strokeStyle = '#E2E8F0'
        ctx.lineWidth = 2
        ctx.setLineDash([])
        
        ctx.beginPath()
        ctx.moveTo(node.x, node.y)
        ctx.lineTo(targetNode.x, targetNode.y)
        ctx.stroke()
      })
    })

    // 데이터 흐름 화살표 그리기
    if (dataFlowActive) {
      dataFlows.forEach((flow, index) => {
        const fromNode = nodes.find(n => n.id === flow.from)
        const toNode = nodes.find(n => n.id === flow.to)
        if (!fromNode || !toNode) return

        // 애니메이션 오프셋 - 속도 설정에 따라 조절
        const speedMultiplier = 100 / animationSpeed // 빠름(100ms)을 1배속 기준으로 설정
        const offset = (animationFrameRef.current * speedMultiplier + index * 30) % 120
        const progress = offset / 120

        // 화살표 위치 계산
        const startX = fromNode.x + (toNode.x - fromNode.x) * 0.1
        const startY = fromNode.y + (toNode.y - fromNode.y) * 0.1
        const endX = fromNode.x + (toNode.x - fromNode.x) * 0.9
        const endY = fromNode.y + (toNode.y - fromNode.y) * 0.9
        
        const currentX = startX + (endX - startX) * progress
        const currentY = startY + (endY - startY) * progress

        // 데이터 패킷 그리기 - 더 부드러운 색상과 작은 크기
        ctx.fillStyle = flow.color + '80' // 투명도 추가
        ctx.beginPath()
        ctx.arc(currentX, currentY, 3, 0, Math.PI * 2) // 4 → 3으로 작게
        ctx.fill()
        
        // 부드러운 글로우 효과
        ctx.fillStyle = flow.color + '40'
        ctx.beginPath()
        ctx.arc(currentX, currentY, 6, 0, Math.PI * 2)
        ctx.fill()

        // 레이블 - 더 적게 표시해서 눈의 피로 줄이기
        if (progress > 0.45 && progress < 0.55 && index % 2 === 0) { // 절반만 표시
          ctx.fillStyle = '#374151'
          ctx.font = '9px Inter'
          ctx.textAlign = 'center'
          ctx.fillText(flow.label, currentX, currentY - 8)
        }
      })
    }

    // 노드 그리기
    nodes.forEach(node => {
      const isHovered = hoveredNode === node.id
      const isSelected = selectedNode?.id === node.id
      const effectColor = getNodeEffectColor(node.id)

      // 시나리오 효과가 있는 노드는 특별한 표시
      if (effectColor) {
        // 시나리오 효과 - 외부 링 그리기
        ctx.fillStyle = effectColor
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.size / 2 + 8, 0, Math.PI * 2)
        ctx.fill()
        
        // 펄스 효과를 위한 반투명 외부 링
        ctx.globalAlpha = 0.3
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.size / 2 + 12, 0, Math.PI * 2)
        ctx.fill()
        ctx.globalAlpha = 1.0
        
      }

      // 노드 배경 (기본 색상 유지)
      ctx.fillStyle = node.color
      ctx.beginPath()
      ctx.arc(node.x, node.y, node.size / 2, 0, Math.PI * 2)
      ctx.fill()

      // 노드 테두리
      ctx.strokeStyle = effectColor ? effectColor : '#FFFFFF'
      ctx.lineWidth = effectColor ? 4 : 3
      ctx.stroke()

      // 아이콘 (텍스트로 대체)
      ctx.fillStyle = '#FFFFFF'
      ctx.font = `${node.size / 3}px Arial`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(node.icon, node.x, node.y)

      // 노드 이름
      ctx.fillStyle = '#374151'
      ctx.font = '12px Inter'
      ctx.textAlign = 'center'
      const lines = node.name.split('\n')
      lines.forEach((line, index) => {
        ctx.fillText(line, node.x, node.y + node.size / 2 + 20 + (index * 14))
      })
    })

    // 범례
    const categories = [
      { name: '생산설비', color: '#F59E0B', icon: '⚙️' },
      { name: 'IoT센서', color: '#10B981', icon: '📡' },
      { name: 'AI시스템', color: '#A855F7', icon: '🧠' },
      { name: '정보시스템', color: '#3B82F6', icon: '📊' },
      { name: '통신인프라', color: '#06B6D4', icon: '☁️' },
      { name: '운영인력', color: '#8B5CF6', icon: '👨‍🔧' }
    ]

    categories.forEach((cat, index) => {
      const x = 50 + (index % 3) * 120
      const y = 550 + Math.floor(index / 3) * 25
      
      ctx.fillStyle = cat.color
      ctx.beginPath()
      ctx.arc(x, y, 8, 0, Math.PI * 2)
      ctx.fill()
      
      ctx.fillStyle = '#374151'
      ctx.font = '11px Inter'
      ctx.textAlign = 'left'
      ctx.fillText(`${cat.icon} ${cat.name}`, x + 15, y + 4)
    })

  }

  // 통합 캔버스 렌더링 - 애니메이션과 상태 변경을 하나로 통합
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null

    const render = () => {
      drawCanvas()
    }

    // 시나리오 진행 중일 때는 애니메이션을 멈춰서 집중할 수 있게 함
    if (dataFlowActive && !currentScenario) {
      intervalId = setInterval(() => {
        animationFrameRef.current += 1
        render()
      }, animationSpeed)
    } else {
      // 애니메이션 정지하고 한 번만 렌더링
      render()
    }

    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [dataFlowActive, animationSpeed, hoveredNode, selectedNode, currentScenario, currentActiveNode, completedNodes])

  // 마우스 이벤트 처리 - 더 정확한 좌표 계산
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    // 이벤트 전파를 막아서 다른 클릭에 영향을 주지 않도록
    event.stopPropagation()
    
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    // 캔버스의 실제 크기와 표시 크기의 비율을 고려
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    
    const x = (event.clientX - rect.left) * scaleX
    const y = (event.clientY - rect.top) * scaleY

    // 클릭된 노드 찾기 - 클릭 영역을 더 크게
    const clickedNode = nodes.find(node => {
      const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2)
      const clickRadius = Math.max(node.size / 2, 30) // 최소 30px 반경
      return distance <= clickRadius
    })

    if (clickedNode) {
      playClickSound() // 클릭 사운드 재생
      setSelectedNode(clickedNode)
    } else {
      setSelectedNode(null)
    }
  }

  const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    
    const x = (event.clientX - rect.left) * scaleX
    const y = (event.clientY - rect.top) * scaleY

    // 호버된 노드 찾기 - 호버 영역도 더 크게
    const hoveredNodeFound = nodes.find(node => {
      const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2)
      const hoverRadius = Math.max(node.size / 2, 25) // 최소 25px 반경
      return distance <= hoverRadius
    })

    setHoveredNode(hoveredNodeFound?.id || null)
    canvas.style.cursor = hoveredNodeFound ? 'pointer' : 'default'
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4">
      {/* 헤더 */}
      <div className="max-w-7xl mx-auto mb-4 relative z-40">
        <Link 
          href={backUrl}
          className="inline-flex items-center gap-2 text-purple-600 hover:text-purple-700 mb-3 relative z-50"
          onClick={(e) => {
            console.log('돌아가기 링크 클릭됨:', backUrl)
            // 기본 링크 동작은 유지하고 이벤트만 로깅
          }}
        >
          <ArrowLeft className="w-5 h-5" />
          스마트팩토리 모듈로 돌아가기
        </Link>
        
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              스마트팩토리 생태계 맵 🏭
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              전체 시스템 구성요소와 실시간 데이터 흐름을 시각화
            </p>
          </div>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                setDataFlowActive(!dataFlowActive)
              }}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                dataFlowActive 
                  ? 'bg-purple-600 text-white hover:bg-purple-700'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {dataFlowActive ? '데이터 흐름 정지' : '데이터 흐름 시작'}
            </button>
            
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">속도:</span>
              <select
                value={animationSpeed}
                onChange={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  setAnimationSpeed(Number(e.target.value))
                }}
                className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm"
              >
                <option value={50}>2배속</option>
                <option value={100}>1배속</option>
                <option value={200}>0.5배속</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* 시나리오 모드 컨트롤 - 더 컴팩트하게 */}
      <div className="max-w-7xl mx-auto mb-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-3 border border-gray-200 dark:border-gray-700">
          <h3 className="text-md font-semibold text-gray-900 dark:text-white mb-2">
            🎯 시나리오 모드
          </h3>
          
          {!currentScenario ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              {Object.entries(scenarios).map(([scenarioId, scenario]) => (
                <button
                  key={scenarioId}
                  onClick={() => startScenario(scenarioId)}
                  className="p-2 text-left bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-purple-50 dark:hover:bg-purple-900/20 border border-gray-200 dark:border-gray-600 transition-colors"
                >
                  <div className="font-medium text-sm text-gray-900 dark:text-white">
                    {scenario.name}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                    {scenario.description}
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-sm text-gray-900 dark:text-white">
                  {scenarios[currentScenario as keyof typeof scenarios]?.name}
                </h4>
                <button
                  onClick={stopScenario}
                  className="px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors"
                >
                  종료
                </button>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-2">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    단계 {scenarioStep + 1} / {scenarios[currentScenario as keyof typeof scenarios]?.steps.length}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-500">
                    🔴 현재 | ⚪ 완료
                  </div>
                </div>
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  {scenarios[currentScenario as keyof typeof scenarios]?.steps[scenarioStep]?.message}
                </div>
              </div>
              
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-1 mb-2">
                <div 
                  className="bg-purple-600 h-1 rounded-full transition-all duration-500"
                  style={{ width: `${((scenarioStep + 1) / scenarios[currentScenario as keyof typeof scenarios]?.steps.length) * 100}%` }}
                />
              </div>
              
              <div className="flex items-center justify-between gap-2">
                <button
                  onClick={prevScenarioStep}
                  disabled={scenarioStep === 0}
                  className={`px-3 py-1 text-xs rounded font-medium transition-all ${
                    scenarioStep === 0
                      ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                      : 'bg-gray-300 hover:bg-gray-400 text-gray-700'
                  }`}
                >
                  ← 이전
                </button>
                
                {scenarioStep === scenarios[currentScenario as keyof typeof scenarios]?.steps.length - 1 ? (
                  <button
                    onClick={stopScenario}
                    className="px-3 py-1 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded font-medium transition-colors"
                  >
                    🎉 완료
                  </button>
                ) : (
                  <button
                    onClick={nextScenarioStep}
                    className="px-3 py-1 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded font-medium transition-all"
                  >
                    다음 →
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* 캔버스 영역 */}
        <div className="lg:col-span-3 bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
          <canvas
            ref={canvasRef}
            className="w-full border border-gray-200 dark:border-gray-700 rounded-lg"
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMouseMove}
            style={{ pointerEvents: 'auto' }}
          />
        </div>

        {/* 상세 정보 패널 */}
        <div className="space-y-4">
          {selectedNode ? (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
              <div className="flex items-center gap-3 mb-3">
                <div 
                  className="w-12 h-12 rounded-full flex items-center justify-center text-2xl text-white"
                  style={{ backgroundColor: selectedNode.color }}
                >
                  {selectedNode.icon}
                </div>
                <div>
                  <h3 className="font-bold text-gray-900 dark:text-white">
                    {selectedNode.name.replace('\n', ' ')}
                  </h3>
                  <span className="text-sm text-gray-500 dark:text-gray-400 capitalize">
                    {selectedNode.category}
                  </span>
                </div>
              </div>
              
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                {selectedNode.description}
              </p>

              {selectedNode.chapterLink && (
                <Link
                  href={selectedNode.chapterLink}
                  className="inline-flex items-center gap-2 px-3 py-2 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded-lg text-sm hover:bg-purple-200 dark:hover:bg-purple-800 transition-colors"
                >
                  <Info className="w-4 h-4" />
                  자세히 학습하기
                </Link>
              )}

              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <h4 className="font-semibold text-sm mb-2">연결된 구성요소:</h4>
                <div className="space-y-1">
                  {selectedNode.connections.map(connId => {
                    const connNode = nodes.find(n => n.id === connId)
                    return connNode ? (
                      <div key={connId} className="text-xs text-gray-500 dark:text-gray-400">
                        • {connNode.name.replace('\n', ' ')}
                      </div>
                    ) : null
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
              <div className="text-center text-gray-500 dark:text-gray-400">
                <Info className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">노드를 클릭하면<br />상세 정보를 볼 수 있습니다</p>
              </div>
            </div>
          )}

          {/* 도움말 */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
            <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center gap-2">
              <Info className="w-4 h-4" />
              사용법
            </h4>
            <div className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
              <p>• <strong>노드 클릭</strong>: 구성요소 상세 정보 보기</p>
              <p>• <strong>색상 구분</strong>: 시스템 유형별 분류</p>
              <p>• <strong>데이터 흐름</strong>: 실시간 정보 이동 경로</p>
              <p>• <strong>연결선</strong>: 물리적/논리적 연결 관계</p>
            </div>
          </div>

          {/* 주요 인사이트 */}
          <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-4">
            <h4 className="font-semibold text-amber-900 dark:text-amber-100 mb-2">
              💡 핵심 인사이트
            </h4>
            <div className="space-y-2 text-sm text-amber-800 dark:text-amber-200">
              <p>• 모든 구성요소가 <strong>중앙 플랫폼</strong>에 연결</p>
              <p>• <strong>실시간 데이터</strong>가 지속적으로 순환</p>
              <p>• <strong>AI가 핵심</strong> 의사결정 지원</p>
              <p>• <strong>사람과 기계</strong>의 협업이 필수</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}