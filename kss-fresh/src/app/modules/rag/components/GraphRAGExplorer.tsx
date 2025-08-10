'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { 
  Network, FileText, Search, Brain, 
  Sparkles, Play, Pause, RefreshCw,
  ChevronRight, Database, GitBranch,
  Users, Building, MapPin, Calendar,
  Maximize2, Minimize2, Settings,
  ZoomIn, ZoomOut, Move, Eye
} from 'lucide-react'

interface Entity {
  id: string
  name: string
  type: 'person' | 'organization' | 'location' | 'event' | 'concept'
  properties: Record<string, any>
}

interface Relation {
  id: string
  source: string
  target: string
  type: string
  properties?: Record<string, any>
}

interface Community {
  id: string
  entities: string[]
  summary: string
  color: string
}

interface GraphNode {
  id: string
  label: string
  type: string
  x: number
  y: number
  vx: number
  vy: number
  community?: string
  color?: string
  size?: number
  connections?: number
}

interface GraphEdge {
  source: string
  target: string
  type: string
  strength?: number
}

interface GraphSettings {
  nodeSize: number
  linkDistance: number
  linkStrength: number
  chargeStrength: number
  showLabels: boolean
  showEdgeLabels: boolean
  highlightDepth: number
  particleEffect: boolean
}

const SAMPLE_TEXT = `
애플은 1976년 스티브 잡스, 스티브 워즈니악, 로널드 웨인이 설립한 미국의 기술 회사입니다. 
본사는 캘리포니아 쿠퍼티노에 위치하며, 팀 쿡이 현재 CEO를 맡고 있습니다.
애플은 아이폰, 맥북, 아이패드 등의 제품을 생산하며, 삼성전자와 경쟁 관계에 있습니다.
최근 애플은 AI 기술에 투자를 늘리고 있으며, 구글, 마이크로소프트와 함께 AI 분야를 선도하고 있습니다.
`

export default function GraphRAGExplorer() {
  const [text, setText] = useState(SAMPLE_TEXT)
  const [entities, setEntities] = useState<Entity[]>([])
  const [relations, setRelations] = useState<Relation[]>([])
  const [communities, setCommunities] = useState<Community[]>([])
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([])
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [queryResult, setQueryResult] = useState<string>('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [isDraggingNode, setIsDraggingNode] = useState<string | null>(null)
  const [draggedNodeIndex, setDraggedNodeIndex] = useState<number | null>(null)
  
  const [settings, setSettings] = useState<GraphSettings>({
    nodeSize: 6,
    linkDistance: 100,
    linkStrength: 0.1,
    chargeStrength: -100,
    showLabels: true,
    showEdgeLabels: false,
    highlightDepth: 2,
    particleEffect: true
  })
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const particlesRef = useRef<Array<{x: number, y: number, vx: number, vy: number, life: number}>>([])
  
  // 엔티티 타입별 색상 (옵시디언 스타일)
  const entityColors = {
    person: '#4ade80',      // 밝은 초록
    organization: '#60a5fa', // 밝은 파랑
    location: '#fbbf24',    // 노랑
    event: '#f87171',       // 빨강
    concept: '#c084fc'      // 보라
  }
  
  // 텍스트에서 엔티티와 관계 추출 (실제 텍스트 분석)
  const extractEntitiesAndRelations = () => {
    setIsProcessing(true)
    
    // 입력된 텍스트에서 실제 엔티티 추출
    setTimeout(() => {
      const extractedEntities = extractEntitiesFromText(text)
      const extractedRelations = extractRelationsFromText(text, extractedEntities)
      
      setEntities(extractedEntities)
      setRelations(extractedRelations)
      
      // 커뮤니티 감지
      detectCommunities(extractedEntities, extractedRelations)
      
      // 그래프 노드와 엣지 생성
      createGraphVisualization(extractedEntities, extractedRelations)
      
      setIsProcessing(false)
    }, 1500)
  }

  // 텍스트에서 엔티티 추출하는 실제 함수
  const extractEntitiesFromText = (inputText: string): Entity[] => {
    const entities: Entity[] = []
    let entityId = 1

    // 조직/회사명 패턴
    const organizationPatterns = [
      /KSS/g, /Knowledge Space Simulator/g, /애플/g, /Apple/g, /삼성전자/g, /Samsung/g, 
      /구글/g, /Google/g, /마이크로소프트/g, /Microsoft/g, /네이버/g, /Naver/g,
      /카카오/g, /Kakao/g, /테슬라/g, /Tesla/g, /OpenAI/g, /오픈AI/g,
      /스탠포드/g, /Stanford/g, /MIT/g, /하버드/g, /Harvard/g
    ]

    // 인물명 패턴  
    const personPatterns = [
      /스티브 잡스/g, /Steve Jobs/g, /팀 쿡/g, /Tim Cook/g, /일론 머스크/g, /Elon Musk/g,
      /젠슨 황/g, /Jensen Huang/g, /샘 알트만/g, /Sam Altman/g, /마크 저커버그/g,
      /제프 베조스/g, /Jeff Bezos/g, /빌 게이츠/g, /Bill Gates/g
    ]

    // 위치명 패턴
    const locationPatterns = [
      /쿠퍼티노/g, /Cupertino/g, /실리콘밸리/g, /Silicon Valley/g, /서울/g, /Seoul/g,
      /한국/g, /Korea/g, /미국/g, /USA/g, /캘리포니아/g, /California/g,
      /워싱턴/g, /Washington/g, /뉴욕/g, /New York/g, /도쿄/g, /Tokyo/g
    ]

    // 기술/개념 패턴
    const conceptPatterns = [
      /AI/g, /인공지능/g, /머신러닝/g, /Machine Learning/g, /딥러닝/g, /Deep Learning/g,
      /온톨로지/g, /Ontology/g, /시뮬레이터/g, /Simulator/g, /교육/g, /Education/g,
      /플랫폼/g, /Platform/g, /VR/g, /AR/g, /메타버스/g, /Metaverse/g,
      /블록체인/g, /Blockchain/g, /NFT/g, /Web3/g, /양자컴퓨팅/g, /Quantum Computing/g,
      /아이폰/g, /iPhone/g, /맥북/g, /MacBook/g, /아이패드/g, /iPad/g
    ]

    // 조직 엔티티 추출
    organizationPatterns.forEach(pattern => {
      const matches = [...inputText.matchAll(pattern)]
      matches.forEach(match => {
        if (match[0] && !entities.find(e => e.name === match[0])) {
          entities.push({
            id: `e${entityId++}`,
            name: match[0],
            type: 'organization',
            properties: { category: '회사/기관' }
          })
        }
      })
    })

    // 인물 엔티티 추출
    personPatterns.forEach(pattern => {
      const matches = [...inputText.matchAll(pattern)]
      matches.forEach(match => {
        if (match[0] && !entities.find(e => e.name === match[0])) {
          entities.push({
            id: `e${entityId++}`,
            name: match[0],
            type: 'person',
            properties: { category: '인물' }
          })
        }
      })
    })

    // 위치 엔티티 추출
    locationPatterns.forEach(pattern => {
      const matches = [...inputText.matchAll(pattern)]
      matches.forEach(match => {
        if (match[0] && !entities.find(e => e.name === match[0])) {
          entities.push({
            id: `e${entityId++}`,
            name: match[0],
            type: 'location',
            properties: { category: '장소' }
          })
        }
      })
    })

    // 개념 엔티티 추출
    conceptPatterns.forEach(pattern => {
      const matches = [...inputText.matchAll(pattern)]
      matches.forEach(match => {
        if (match[0] && !entities.find(e => e.name === match[0])) {
          entities.push({
            id: `e${entityId++}`,
            name: match[0],
            type: 'concept',
            properties: { category: '기술/개념' }
          })
        }
      })
    })

    // 최소 3개 이상의 엔티티가 없으면 기본 엔티티 추가
    if (entities.length < 3) {
      const defaultEntities = [
        { id: 'e_default1', name: '문서 분석', type: 'concept' as const, properties: { category: '분석' } },
        { id: 'e_default2', name: '지식 그래프', type: 'concept' as const, properties: { category: '시각화' } },
        { id: 'e_default3', name: 'GraphRAG', type: 'concept' as const, properties: { category: '기술' } }
      ]
      entities.push(...defaultEntities)
    }

    return entities.slice(0, 15) // 최대 15개로 제한
  }

  // 텍스트에서 관계 추출하는 함수
  const extractRelationsFromText = (inputText: string, entities: Entity[]): Relation[] => {
    const relations: Relation[] = []
    let relationId = 1

    // 관계 패턴들
    const relationPatterns = [
      { pattern: /(.+?)(는|은|이|가)\s*(.+?)(를|을|에게|에)\s*(개발|제작|생산|설립|창립|투자|인수|출시)/g, type: '개발' },
      { pattern: /(.+?)(와|과)\s*(.+?)(의|와|과)\s*(협력|파트너십|제휴|경쟁)/g, type: '관계' },
      { pattern: /(.+?)(의|는|은)\s*(.+?)(CEO|대표|창업자|설립자)/g, type: '리더십' },
      { pattern: /(.+?)(는|은|이|가)\s*(.+?)(에|에서)\s*(위치|본사|거주)/g, type: '위치' },
      { pattern: /(.+?)(와|과)\s*(.+?)(는|은|이|가)\s*(경쟁|라이벌)/g, type: '경쟁' }
    ]

    // 패턴 기반 관계 추출
    relationPatterns.forEach(({ pattern, type }) => {
      const matches = [...inputText.matchAll(pattern)]
      matches.forEach(match => {
        const source = entities.find(e => match[1] && e.name.includes(match[1].trim()))
        const target = entities.find(e => match[3] && e.name.includes(match[3].trim()))
        
        if (source && target && source.id !== target.id) {
          relations.push({
            id: `r${relationId++}`,
            source: source.id,
            target: target.id,
            type: type
          })
        }
      })
    })

    // 기본 관계 생성 (인접한 엔티티들 연결)
    for (let i = 0; i < entities.length - 1; i++) {
      for (let j = i + 1; j < Math.min(entities.length, i + 3); j++) {
        if (Math.random() > 0.3) { // 70% 확률로 관계 생성
          relations.push({
            id: `r${relationId++}`,
            source: entities[i].id,
            target: entities[j].id,
            type: '연관'
          })
        }
      }
    }

    return relations.slice(0, 20) // 최대 20개로 제한
  }
  
  // 커뮤니티 감지 (동적 분석)
  const detectCommunities = (entities: Entity[], relations: Relation[]) => {
    const communities: Community[] = []
    const visitedEntities = new Set<string>()
    
    // 엔티티 타입별로 커뮤니티 그룹 생성
    const typeGroups: Record<string, Entity[]> = {
      organization: [],
      person: [],
      location: [],
      event: [],
      concept: []
    }
    
    entities.forEach(entity => {
      typeGroups[entity.type].push(entity)
    })
    
    let communityId = 1
    const colors = ['#60a5fa', '#c084fc', '#4ade80', '#fbbf24', '#f87171']
    
    // 각 타입별로 커뮤니티 생성 (엔티티가 2개 이상인 경우)
    Object.entries(typeGroups).forEach(([type, typeEntities], index) => {
      if (typeEntities.length >= 2) {
        const entityIds = typeEntities.map(e => e.id)
        communities.push({
          id: `c${communityId++}`,
          entities: entityIds,
          summary: getTypeSummary(type, typeEntities.length),
          color: colors[index % colors.length]
        })
        
        entityIds.forEach(id => visitedEntities.add(id))
      }
    })
    
    // 연결이 많은 엔티티들로 추가 커뮤니티 생성
    const connectionCounts: Record<string, number> = {}
    relations.forEach(rel => {
      connectionCounts[rel.source] = (connectionCounts[rel.source] || 0) + 1
      connectionCounts[rel.target] = (connectionCounts[rel.target] || 0) + 1
    })
    
    const highlyConnected = entities
      .filter(e => !visitedEntities.has(e.id) && connectionCounts[e.id] >= 2)
      .slice(0, 5)
    
    if (highlyConnected.length >= 2) {
      communities.push({
        id: `c${communityId++}`,
        entities: highlyConnected.map(e => e.id),
        summary: '핵심 연결 노드',
        color: colors[communities.length % colors.length]
      })
    }
    
    // 최소 1개 커뮤니티 보장
    if (communities.length === 0 && entities.length > 0) {
      communities.push({
        id: 'c1',
        entities: entities.slice(0, Math.min(5, entities.length)).map(e => e.id),
        summary: '주요 엔티티',
        color: '#60a5fa'
      })
    }
    
    setCommunities(communities)
  }
  
  // 엔티티 타입별 요약 생성
  const getTypeSummary = (type: string, count: number): string => {
    const summaries: Record<string, string> = {
      organization: `조직/기업 (${count}개)`,
      person: `인물 (${count}명)`, 
      location: `장소 (${count}곳)`,
      event: `이벤트 (${count}건)`,
      concept: `개념/기술 (${count}개)`
    }
    return summaries[type] || `${type} (${count}개)`
  }
  
  // 그래프 시각화 데이터 생성
  const createGraphVisualization = (entities: Entity[], relations: Relation[]) => {
    // 각 노드의 연결 수 계산
    const connectionCount: Record<string, number> = {}
    relations.forEach(rel => {
      connectionCount[rel.source] = (connectionCount[rel.source] || 0) + 1
      connectionCount[rel.target] = (connectionCount[rel.target] || 0) + 1
    })
    
    const nodes: GraphNode[] = entities.map((entity, index) => {
      const angle = (index / entities.length) * Math.PI * 2
      const radius = 200
      const connections = connectionCount[entity.id] || 0
      
      return {
        id: entity.id,
        label: entity.name,
        type: entity.type,
        x: 400 + Math.cos(angle) * radius + (Math.random() - 0.5) * 50,
        y: 300 + Math.sin(angle) * radius + (Math.random() - 0.5) * 50,
        vx: 0,
        vy: 0,
        color: entityColors[entity.type],
        size: Math.max(6, Math.min(20, 6 + connections * 2)),
        connections
      }
    })
    
    const edges: GraphEdge[] = relations.map(rel => ({
      source: rel.source,
      target: rel.target,
      type: rel.type,
      strength: 1
    }))
    
    setGraphNodes(nodes)
    setGraphEdges(edges)
  }
  
  // Force-directed 레이아웃 시뮬레이션
  const updateLayout = useCallback(() => {
    if (!isAnimating && !isDraggingNode) return
    
    const nodes = [...graphNodes]
    const edges = graphEdges
    
    // 반발력
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x
        const dy = nodes[j].y - nodes[i].y
        const distance = Math.sqrt(dx * dx + dy * dy)
        
        if (distance > 0 && distance < 150) {
          const force = settings.chargeStrength / (distance * distance) * 0.1
          nodes[i].vx -= (dx / distance) * force
          nodes[i].vy -= (dy / distance) * force
          nodes[j].vx += (dx / distance) * force
          nodes[j].vy += (dy / distance) * force
        }
      }
    }
    
    // 인력 (엣지)
    edges.forEach(edge => {
      const source = nodes.find(n => n.id === edge.source)
      const target = nodes.find(n => n.id === edge.target)
      
      if (source && target) {
        const dx = target.x - source.x
        const dy = target.y - source.y
        const distance = Math.sqrt(dx * dx + dy * dy)
        
        if (distance > 0) {
          const force = (distance - settings.linkDistance) * settings.linkStrength * 0.005
          source.vx += (dx / distance) * force
          source.vy += (dy / distance) * force
          target.vx -= (dx / distance) * force
          target.vy -= (dy / distance) * force
        }
      }
    })
    
    // 중심 인력
    nodes.forEach(node => {
      const dx = 400 - node.x
      const dy = 300 - node.y
      node.vx += dx * 0.0005
      node.vy += dy * 0.0005
    })
    
    // 속도 적용 및 감쇠
    nodes.forEach((node, index) => {
      // 드래그 중인 노드는 물리 시뮬레이션 제외
      if (isDraggingNode === node.id) {
        node.vx = 0
        node.vy = 0
        return
      }
      
      // 속도 제한
      const maxVelocity = 2
      node.vx = Math.max(-maxVelocity, Math.min(maxVelocity, node.vx))
      node.vy = Math.max(-maxVelocity, Math.min(maxVelocity, node.vy))
      
      node.x += node.vx
      node.y += node.vy
      node.vx *= 0.92
      node.vy *= 0.92
    })
    
    setGraphNodes(nodes)
  }, [graphNodes, graphEdges, isAnimating, isDraggingNode, settings])
  
  // 파티클 업데이트
  const updateParticles = useCallback(() => {
    if (!settings.particleEffect) {
      particlesRef.current = []
      return
    }
    
    // 새 파티클 생성
    if (Math.random() < 0.1 && particlesRef.current.length < 50) {
      const edge = graphEdges[Math.floor(Math.random() * graphEdges.length)]
      if (edge) {
        const source = graphNodes.find(n => n.id === edge.source)
        if (source) {
          particlesRef.current.push({
            x: source.x,
            y: source.y,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            life: 1
          })
        }
      }
    }
    
    // 파티클 업데이트
    particlesRef.current = particlesRef.current.filter(p => {
      p.x += p.vx
      p.y += p.vy
      p.life -= 0.02
      return p.life > 0
    })
  }, [graphNodes, graphEdges, settings.particleEffect])
  
  // 그래프 렌더링
  const renderGraph = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // 캔버스 초기화
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // 배경
    ctx.fillStyle = isFullscreen ? '#0a0a0a' : '#111827'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // 변환 적용
    ctx.save()
    ctx.translate(pan.x, pan.y)
    ctx.scale(zoom, zoom)
    
    // 연결된 노드 찾기
    const connectedNodes = new Set<string>()
    if (hoveredNode || selectedNode) {
      const activeNode = hoveredNode || selectedNode
      connectedNodes.add(activeNode!)
      
      // 깊이에 따른 연결 찾기
      for (let depth = 0; depth < settings.highlightDepth; depth++) {
        const currentNodes = [...connectedNodes]
        currentNodes.forEach(nodeId => {
          graphEdges.forEach(edge => {
            if (edge.source === nodeId) connectedNodes.add(edge.target)
            if (edge.target === nodeId) connectedNodes.add(edge.source)
          })
        })
      }
    }
    
    // 엣지 그리기
    graphEdges.forEach(edge => {
      const source = graphNodes.find(n => n.id === edge.source)
      const target = graphNodes.find(n => n.id === edge.target)
      
      if (source && target) {
        const isHighlighted = connectedNodes.size > 0 && 
          connectedNodes.has(edge.source) && connectedNodes.has(edge.target)
        
        ctx.strokeStyle = isHighlighted ? '#60a5fa80' : '#374151'
        ctx.lineWidth = isHighlighted ? 2 : 1
        
        // 곡선 엣지
        const dx = target.x - source.x
        const dy = target.y - source.y
        const distance = Math.sqrt(dx * dx + dy * dy)
        const curvature = 0.2
        
        const midX = (source.x + target.x) / 2
        const midY = (source.y + target.y) / 2
        const offsetX = -dy * curvature
        const offsetY = dx * curvature
        
        ctx.beginPath()
        ctx.moveTo(source.x, source.y)
        ctx.quadraticCurveTo(
          midX + offsetX,
          midY + offsetY,
          target.x,
          target.y
        )
        ctx.stroke()
        
        // 엣지 레이블
        if (settings.showEdgeLabels && zoom > 0.7) {
          ctx.fillStyle = '#9ca3af'
          ctx.font = '10px Inter'
          ctx.textAlign = 'center'
          ctx.fillText(edge.type, midX, midY)
        }
      }
    })
    
    // 파티클 그리기
    particlesRef.current.forEach(particle => {
      ctx.fillStyle = `rgba(96, 165, 250, ${particle.life * 0.5})`
      ctx.beginPath()
      ctx.arc(particle.x, particle.y, 2, 0, Math.PI * 2)
      ctx.fill()
    })
    
    // 노드 그리기
    graphNodes.forEach(node => {
      const isHighlighted = connectedNodes.size === 0 || connectedNodes.has(node.id)
      const isActive = node.id === hoveredNode || node.id === selectedNode
      
      // 노드 광선 효과
      if (isActive) {
        const gradient = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, node.size! * 3
        )
        gradient.addColorStop(0, node.color + '40')
        gradient.addColorStop(1, 'transparent')
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.size! * 3, 0, Math.PI * 2)
        ctx.fill()
      }
      
      // 노드 원
      ctx.fillStyle = isHighlighted ? node.color || '#60a5fa' : '#374151'
      ctx.strokeStyle = isActive ? '#fff' : 'transparent'
      ctx.lineWidth = 2
      
      const nodeSize = (node.size || settings.nodeSize) * (isActive ? 1.2 : 1)
      
      ctx.beginPath()
      ctx.arc(node.x, node.y, nodeSize, 0, Math.PI * 2)
      ctx.fill()
      if (isActive) ctx.stroke()
      
      // 노드 레이블
      if (settings.showLabels && (zoom > 0.5 || isActive)) {
        ctx.fillStyle = isHighlighted ? '#e5e7eb' : '#6b7280'
        ctx.font = isActive ? 'bold 12px Inter' : '11px Inter'
        ctx.textAlign = 'center'
        ctx.fillText(node.label, node.x, node.y + nodeSize + 15)
      }
    })
    
    ctx.restore()
    
    // 미니맵 (우측 하단)
    if (!isFullscreen) {
      const minimapSize = 120
      const minimapX = canvas.width - minimapSize - 20
      const minimapY = canvas.height - minimapSize - 20
      
      // 미니맵 배경
      ctx.fillStyle = 'rgba(17, 24, 39, 0.8)'
      ctx.strokeStyle = '#374151'
      ctx.fillRect(minimapX, minimapY, minimapSize, minimapSize)
      ctx.strokeRect(minimapX, minimapY, minimapSize, minimapSize)
      
      // 미니맵 노드
      graphNodes.forEach(node => {
        const x = minimapX + (node.x / 800) * minimapSize
        const y = minimapY + (node.y / 600) * minimapSize
        
        ctx.fillStyle = node.color || '#60a5fa'
        ctx.beginPath()
        ctx.arc(x, y, 2, 0, Math.PI * 2)
        ctx.fill()
      })
      
      // 뷰포트 표시
      const viewX = minimapX + (-pan.x / zoom / 800) * minimapSize
      const viewY = minimapY + (-pan.y / zoom / 600) * minimapSize
      const viewW = (canvas.width / zoom / 800) * minimapSize
      const viewH = (canvas.height / zoom / 600) * minimapSize
      
      ctx.strokeStyle = '#60a5fa'
      ctx.lineWidth = 1
      ctx.strokeRect(viewX, viewY, viewW, viewH)
    }
  }, [graphNodes, graphEdges, hoveredNode, selectedNode, settings, zoom, pan, isFullscreen])
  
  // 쿼리 처리 (동적 분석)
  const processQuery = () => {
    if (!query.trim()) return
    
    const lowerQuery = query.toLowerCase()
    
    // 추출된 엔티티에서 쿼리와 관련된 정보 찾기
    const relevantEntities = entities.filter(entity => 
      entity.name.toLowerCase().includes(lowerQuery) || 
      lowerQuery.includes(entity.name.toLowerCase())
    )
    
    if (relevantEntities.length > 0) {
      const entity = relevantEntities[0]
      
      // 관련 관계 찾기
      const relatedRelations = relations.filter(rel => 
        rel.source === entity.id || rel.target === entity.id
      )
      
      const connectedEntities = relatedRelations.map(rel => {
        const connectedId = rel.source === entity.id ? rel.target : rel.source
        return entities.find(e => e.id === connectedId)
      }).filter(e => e !== undefined)
      
      // 결과 생성
      let result = `"${entity.name}"에 대한 정보를 찾았습니다.\n\n`
      result += `• 타입: ${getTypeDescription(entity.type)}\n`
      
      if (connectedEntities.length > 0) {
        result += `• 연관된 엔티티: ${connectedEntities.map(e => e!.name).slice(0, 3).join(', ')}\n`
        
        const relationTypes = [...new Set(relatedRelations.map(r => r.type))]
        result += `• 관계: ${relationTypes.slice(0, 3).join(', ')}`
      } else {
        result += `• 단독으로 존재하는 엔티티입니다.`
      }
      
      setQueryResult(result)
      setSelectedNode(entity.id)
    } else {
      // 관계 패턴 검색
      if (lowerQuery.includes('관계') || lowerQuery.includes('연결') || lowerQuery.includes('연관')) {
        const allRelationTypes = [...new Set(relations.map(r => r.type))]
        setQueryResult(`이 그래프에서 발견된 관계 유형들: ${allRelationTypes.join(', ')}`)
      } else if (lowerQuery.includes('개수') || lowerQuery.includes('수')) {
        setQueryResult(`총 ${entities.length}개의 엔티티와 ${relations.length}개의 관계가 발견되었습니다.`)
      } else if (lowerQuery.includes('타입') || lowerQuery.includes('종류')) {
        const entityTypes = entities.reduce((acc, entity) => {
          acc[entity.type] = (acc[entity.type] || 0) + 1
          return acc
        }, {} as Record<string, number>)
        
        const typeDescription = Object.entries(entityTypes)
          .map(([type, count]) => `${getTypeDescription(type)}: ${count}개`)
          .join(', ')
        
        setQueryResult(`엔티티 타입별 분포: ${typeDescription}`)
      } else {
        setQueryResult(`"${query}"와 관련된 정보를 찾을 수 없습니다. 추출된 엔티티: ${entities.map(e => e.name).slice(0, 5).join(', ')}`)
      }
    }
  }
  
  // 엔티티 타입 설명
  const getTypeDescription = (type: string): string => {
    const descriptions: Record<string, string> = {
      organization: '조직/기업',
      person: '인물',
      location: '장소',
      event: '이벤트',
      concept: '개념/기술'
    }
    return descriptions[type] || type
  }
  
  // 애니메이션 루프
  useEffect(() => {
    const animate = () => {
      updateLayout()
      updateParticles()
      renderGraph()
      animationRef.current = requestAnimationFrame(animate)
    }
    animate()
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [updateLayout, updateParticles, renderGraph])
  
  // 캔버스 이벤트 핸들러
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    
    const x = (e.clientX - rect.left - pan.x) / zoom
    const y = (e.clientY - rect.top - pan.y) / zoom
    
    // 노드 클릭 확인
    const clickedNodeIndex = graphNodes.findIndex(node => {
      const dx = node.x - x
      const dy = node.y - y
      return Math.sqrt(dx * dx + dy * dy) < (node.size || settings.nodeSize) + 5
    })
    
    if (clickedNodeIndex !== -1) {
      const clickedNode = graphNodes[clickedNodeIndex]
      setSelectedNode(clickedNode.id)
      setIsDraggingNode(clickedNode.id)
      setDraggedNodeIndex(clickedNodeIndex)
    } else {
      setSelectedNode(null)
      setIsDragging(true)
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
    }
  }
  
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    
    const x = (e.clientX - rect.left - pan.x) / zoom
    const y = (e.clientY - rect.top - pan.y) / zoom
    
    if (isDraggingNode && draggedNodeIndex !== null) {
      // 노드 드래그
      const newNodes = [...graphNodes]
      if (newNodes[draggedNodeIndex]) {
        newNodes[draggedNodeIndex].x = x
        newNodes[draggedNodeIndex].y = y
        setGraphNodes(newNodes)
      }
      if (canvasRef.current) {
        canvasRef.current.style.cursor = 'grabbing'
      }
    } else if (isDragging) {
      // 캔버스 팬
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
      if (canvasRef.current) {
        canvasRef.current.style.cursor = 'grabbing'
      }
    } else {
      // 호버 노드 찾기
      const hoveredNode = graphNodes.find(node => {
        const dx = node.x - x
        const dy = node.y - y
        return Math.sqrt(dx * dx + dy * dy) < (node.size || settings.nodeSize) + 5
      })
      
      setHoveredNode(hoveredNode?.id || null)
      if (canvasRef.current) {
        canvasRef.current.style.cursor = hoveredNode ? 'pointer' : 'grab'
      }
    }
  }
  
  const handleCanvasMouseUp = () => {
    setIsDragging(false)
    setIsDraggingNode(null)
    setDraggedNodeIndex(null)
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'grab'
    }
  }
  
  const handleCanvasWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    const newZoom = Math.min(Math.max(zoom * delta, 0.1), 5)
    setZoom(newZoom)
  }
  
  // 전체화면 토글
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }
  
  // 줌 리셋
  const resetView = () => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }
  
  return (
    <div className={`${isFullscreen ? 'fixed inset-0 z-50' : ''} bg-gray-900`}>
      <div className={`${isFullscreen ? 'h-full' : 'space-y-6'}`}>
        {/* 입력 섹션 (전체화면이 아닐 때만) */}
        {!isFullscreen && (
          <div className="bg-gray-800 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white">
              <FileText className="w-5 h-5 text-purple-400" />
              문서 입력
            </h3>
            
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full h-32 p-3 border border-gray-600 rounded-lg 
                       bg-gray-900 text-gray-100
                       focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              placeholder="분석할 텍스트를 입력하세요..."
            />
            
            <button
              onClick={extractEntitiesAndRelations}
              disabled={isProcessing || !text.trim()}
              className="mt-4 px-6 py-2 bg-purple-600 text-white rounded-lg 
                       hover:bg-purple-700 transition-colors disabled:opacity-50
                       flex items-center gap-2"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  처리 중...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  그래프 생성
                </>
              )}
            </button>
          </div>
        )}
        
        {/* 그래프 시각화 */}
        <div className={`${isFullscreen ? 'h-full' : 'bg-gray-800 rounded-lg shadow-sm'} relative`}>
          {/* 상단 툴바 */}
          <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10 pointer-events-none">
            <div className="flex items-center gap-2 pointer-events-auto">
              <button
                onClick={() => setIsAnimating(!isAnimating)}
                className="px-3 py-1.5 bg-gray-700 text-white rounded-lg 
                         hover:bg-gray-600 transition-colors flex items-center gap-2 text-sm"
              >
                {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isAnimating ? '정지' : '시작'}
              </button>
              
              <button
                onClick={resetView}
                className="p-1.5 bg-gray-700 text-white rounded-lg 
                         hover:bg-gray-600 transition-colors"
                title="뷰 리셋"
              >
                <Move className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => setZoom(zoom * 1.2)}
                className="p-1.5 bg-gray-700 text-white rounded-lg 
                         hover:bg-gray-600 transition-colors"
                title="확대"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => setZoom(zoom * 0.8)}
                className="p-1.5 bg-gray-700 text-white rounded-lg 
                         hover:bg-gray-600 transition-colors"
                title="축소"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-1.5 bg-gray-700 text-white rounded-lg 
                         hover:bg-gray-600 transition-colors"
                title="설정"
              >
                <Settings className="w-4 h-4" />
              </button>
            </div>
            
            <div className="flex items-center gap-2 pointer-events-auto">
              <button
                onClick={toggleFullscreen}
                className="p-1.5 bg-gray-700 text-white rounded-lg 
                         hover:bg-gray-600 transition-colors"
                title={isFullscreen ? "전체화면 종료" : "전체화면"}
              >
                {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
              </button>
            </div>
          </div>
          
          {/* 설정 패널 */}
          {showSettings && (
            <div className="absolute top-16 left-4 bg-gray-800 rounded-lg p-4 shadow-lg z-20 w-64">
              <h4 className="font-semibold text-white mb-3">그래프 설정</h4>
              
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-gray-400">노드 크기</label>
                  <input
                    type="range"
                    min="4"
                    max="20"
                    value={settings.nodeSize}
                    onChange={(e) => setSettings({...settings, nodeSize: Number(e.target.value)})}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="text-sm text-gray-400">링크 거리</label>
                  <input
                    type="range"
                    min="50"
                    max="200"
                    value={settings.linkDistance}
                    onChange={(e) => setSettings({...settings, linkDistance: Number(e.target.value)})}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="text-sm text-gray-400">반발력</label>
                  <input
                    type="range"
                    min="-500"
                    max="-50"
                    value={settings.chargeStrength}
                    onChange={(e) => setSettings({...settings, chargeStrength: Number(e.target.value)})}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="text-sm text-gray-400">링크 강도</label>
                  <input
                    type="range"
                    min="0.05"
                    max="0.5"
                    step="0.05"
                    value={settings.linkStrength}
                    onChange={(e) => setSettings({...settings, linkStrength: Number(e.target.value)})}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="text-sm text-gray-400 flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.showLabels}
                      onChange={(e) => setSettings({...settings, showLabels: e.target.checked})}
                    />
                    노드 레이블 표시
                  </label>
                </div>
                
                <div>
                  <label className="text-sm text-gray-400 flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.showEdgeLabels}
                      onChange={(e) => setSettings({...settings, showEdgeLabels: e.target.checked})}
                    />
                    엣지 레이블 표시
                  </label>
                </div>
                
                <div>
                  <label className="text-sm text-gray-400 flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.particleEffect}
                      onChange={(e) => setSettings({...settings, particleEffect: e.target.checked})}
                    />
                    파티클 효과
                  </label>
                </div>
              </div>
            </div>
          )}
          
          {/* 캔버스 */}
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseUp}
            onWheel={handleCanvasWheel}
            className={`${isFullscreen ? 'w-full h-full' : 'w-full'} bg-gray-900 cursor-grab`}
          />
          
          {/* 선택된 노드 정보 */}
          {selectedNode && (
            <div className="absolute bottom-4 left-4 bg-gray-800 rounded-lg p-4 shadow-lg max-w-sm">
              {(() => {
                const node = entities.find(e => e.id === selectedNode)
                if (!node) return null
                
                return (
                  <>
                    <h4 className="font-semibold text-white mb-2">{node.name}</h4>
                    <p className="text-sm text-gray-400 mb-2">
                      타입: {node.type}
                    </p>
                    {Object.entries(node.properties).map(([key, value]) => (
                      <p key={key} className="text-sm text-gray-400">
                        {key}: {value}
                      </p>
                    ))}
                  </>
                )
              })()}
            </div>
          )}
        </div>
        
        {/* 쿼리 섹션 (전체화면이 아닐 때만) */}
        {!isFullscreen && entities.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white">
              <Search className="w-5 h-5 text-purple-400" />
              GraphRAG 쿼리
            </h3>
            
            <div className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && processQuery()}
                className="flex-1 px-4 py-2 border border-gray-600 rounded-lg 
                         bg-gray-900 text-gray-100
                         focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="그래프에 대해 질문하세요... (예: 애플의 경쟁사는?)"
              />
              
              <button
                onClick={processQuery}
                className="px-6 py-2 bg-purple-600 text-white rounded-lg 
                         hover:bg-purple-700 transition-colors flex items-center gap-2"
              >
                <ChevronRight className="w-4 h-4" />
                검색
              </button>
            </div>
            
            {queryResult && (
              <div className="mt-4 p-4 bg-purple-900/20 rounded-lg">
                <p className="text-gray-300">{queryResult}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}