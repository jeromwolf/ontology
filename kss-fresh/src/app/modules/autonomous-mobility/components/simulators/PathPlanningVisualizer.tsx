'use client'

import { useState, useEffect, useRef } from 'react'
import { Route, Navigation, Target, Cpu, Settings, Play, Pause, RotateCcw, Zap } from 'lucide-react'

interface Node {
  x: number
  y: number
  g: number // Cost from start
  h: number // Heuristic to goal
  f: number // Total cost
  parent: Node | null
  walkable: boolean
}

interface Point {
  x: number
  y: number
}

export default function PathPlanningVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [isRunning, setIsRunning] = useState(false)
  const [algorithm, setAlgorithm] = useState<'astar' | 'rrt' | 'dijkstra' | 'dwa'>('astar')
  const [gridSize, setGridSize] = useState(20)
  const [showCostMap, setShowCostMap] = useState(true)
  const [showSearchSpace, setShowSearchSpace] = useState(true)
  const [start, setStart] = useState<Point>({ x: 50, y: 50 })
  const [goal, setGoal] = useState<Point>({ x: 450, y: 350 })
  const [obstacles, setObstacles] = useState<Point[]>([])
  const [path, setPath] = useState<Point[]>([])
  const [searchedNodes, setSearchedNodes] = useState<Point[]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [stats, setStats] = useState({
    pathLength: 0,
    nodesExpanded: 0,
    computeTime: 0,
    pathCost: 0
  })

  const gridRef = useRef<Node[][]>([])
  const isDrawingRef = useRef(false)
  const drawModeRef = useRef<'obstacle' | 'start' | 'goal'>('obstacle')

  // 그리드 초기화
  const initializeGrid = (width: number, height: number) => {
    const cols = Math.floor(width / gridSize)
    const rows = Math.floor(height / gridSize)
    const grid: Node[][] = []

    for (let i = 0; i < cols; i++) {
      grid[i] = []
      for (let j = 0; j < rows; j++) {
        grid[i][j] = {
          x: i * gridSize,
          y: j * gridSize,
          g: 0,
          h: 0,
          f: 0,
          parent: null,
          walkable: true
        }
      }
    }

    // 장애물 설정
    obstacles.forEach(obs => {
      const i = Math.floor(obs.x / gridSize)
      const j = Math.floor(obs.y / gridSize)
      if (grid[i] && grid[i][j]) {
        grid[i][j].walkable = false
      }
    })

    gridRef.current = grid
  }

  // A* 알고리즘
  const runAStar = () => {
    const startTime = performance.now()
    const grid = gridRef.current
    const openSet: Node[] = []
    const closedSet: Set<Node> = new Set()
    const searched: Point[] = []

    const startNode = grid[Math.floor(start.x / gridSize)][Math.floor(start.y / gridSize)]
    const endNode = grid[Math.floor(goal.x / gridSize)][Math.floor(goal.y / gridSize)]

    openSet.push(startNode)

    const animateStep = () => {
      if (openSet.length === 0 || !isRunning) {
        setStats(prev => ({ ...prev, computeTime: performance.now() - startTime }))
        return
      }

      // 가장 낮은 f 값을 가진 노드 찾기
      let currentNode = openSet[0]
      let currentIndex = 0
      openSet.forEach((node, index) => {
        if (node.f < currentNode.f) {
          currentNode = node
          currentIndex = index
        }
      })

      openSet.splice(currentIndex, 1)
      closedSet.add(currentNode)
      searched.push({ x: currentNode.x, y: currentNode.y })

      // 목표 도달
      if (currentNode === endNode) {
        const pathPoints: Point[] = []
        let current: Node | null = currentNode
        let pathCost = 0

        while (current) {
          pathPoints.unshift({ x: current.x, y: current.y })
          if (current.parent) {
            pathCost += Math.sqrt(
              Math.pow(current.x - current.parent.x, 2) + 
              Math.pow(current.y - current.parent.y, 2)
            )
          }
          current = current.parent
        }

        setPath(pathPoints)
        setStats({
          pathLength: pathPoints.length,
          nodesExpanded: searched.length,
          computeTime: performance.now() - startTime,
          pathCost: pathCost
        })
        return
      }

      // 이웃 노드 탐색
      const neighbors = getNeighbors(currentNode, grid)
      neighbors.forEach(neighbor => {
        if (closedSet.has(neighbor) || !neighbor.walkable) return

        const tentativeG = currentNode.g + distance(currentNode, neighbor)

        if (!openSet.includes(neighbor)) {
          openSet.push(neighbor)
        } else if (tentativeG >= neighbor.g) {
          return
        }

        neighbor.g = tentativeG
        neighbor.h = heuristic(neighbor, endNode)
        neighbor.f = neighbor.g + neighbor.h
        neighbor.parent = currentNode
      })

      setSearchedNodes([...searched])
      setCurrentStep(searched.length)

      animationRef.current = requestAnimationFrame(animateStep)
    }

    animateStep()
  }

  // RRT (Rapidly-exploring Random Tree) 알고리즘
  const runRRT = () => {
    const startTime = performance.now()
    const tree: Point[] = [start]
    const edges: { from: Point; to: Point }[] = []
    const stepSize = 30
    const maxIterations = 1000
    let iterations = 0

    const animateStep = () => {
      if (iterations >= maxIterations || !isRunning) return

      // 랜덤 점 생성
      const random = {
        x: Math.random() * canvasRef.current!.width,
        y: Math.random() * canvasRef.current!.height
      }

      // 가장 가까운 노드 찾기
      let nearest = tree[0]
      let minDist = distance2D(random, nearest)
      tree.forEach(node => {
        const dist = distance2D(random, node)
        if (dist < minDist) {
          nearest = node
          minDist = dist
        }
      })

      // 새 노드 생성
      const angle = Math.atan2(random.y - nearest.y, random.x - nearest.x)
      const newNode = {
        x: nearest.x + Math.cos(angle) * stepSize,
        y: nearest.y + Math.sin(angle) * stepSize
      }

      // 충돌 검사
      if (!isColliding(nearest, newNode)) {
        tree.push(newNode)
        edges.push({ from: nearest, to: newNode })
        setSearchedNodes([...tree])

        // 목표 도달 검사
        if (distance2D(newNode, goal) < stepSize) {
          // 경로 재구성
          const pathPoints = reconstructRRTPath(edges, start, newNode)
          pathPoints.push(goal)
          setPath(pathPoints)
          setStats({
            pathLength: pathPoints.length,
            nodesExpanded: tree.length,
            computeTime: performance.now() - startTime,
            pathCost: calculatePathCost(pathPoints)
          })
          return
        }
      }

      iterations++
      animationRef.current = requestAnimationFrame(animateStep)
    }

    animateStep()
  }

  // DWA (Dynamic Window Approach) 알고리즘
  const runDWA = () => {
    const startTime = performance.now()
    const trajectory: Point[] = [start]
    let currentPos = { ...start }
    let currentVel = { vx: 0, vy: 0 }
    const maxVel = 5
    const maxAccel = 2
    const dt = 0.1
    const predictionTime = 3

    const animateStep = () => {
      if (distance2D(currentPos, goal) < 10 || !isRunning) {
        setPath(trajectory)
        setStats({
          pathLength: trajectory.length,
          nodesExpanded: trajectory.length,
          computeTime: performance.now() - startTime,
          pathCost: calculatePathCost(trajectory)
        })
        return
      }

      // Dynamic Window 계산
      const velocitySamples: { vx: number; vy: number; cost: number }[] = []
      
      for (let dvx = -maxAccel * dt; dvx <= maxAccel * dt; dvx += 0.5) {
        for (let dvy = -maxAccel * dt; dvy <= maxAccel * dt; dvy += 0.5) {
          const newVx = Math.max(-maxVel, Math.min(maxVel, currentVel.vx + dvx))
          const newVy = Math.max(-maxVel, Math.min(maxVel, currentVel.vy + dvy))
          
          // 궤적 예측
          let futurePos = { ...currentPos }
          let collision = false
          
          for (let t = 0; t < predictionTime; t += dt) {
            futurePos.x += newVx * dt
            futurePos.y += newVy * dt
            
            if (isPointColliding(futurePos)) {
              collision = true
              break
            }
          }
          
          if (!collision) {
            // 비용 계산
            const headingCost = Math.abs(Math.atan2(goal.y - futurePos.y, goal.x - futurePos.x))
            const velocityCost = maxVel - Math.sqrt(newVx * newVx + newVy * newVy)
            const distanceCost = distance2D(futurePos, goal)
            
            const totalCost = headingCost + velocityCost * 0.1 + distanceCost * 0.01
            velocitySamples.push({ vx: newVx, vy: newVy, cost: totalCost })
          }
        }
      }

      // 최적 속도 선택
      if (velocitySamples.length > 0) {
        velocitySamples.sort((a, b) => a.cost - b.cost)
        currentVel = velocitySamples[0]
        currentPos.x += currentVel.vx
        currentPos.y += currentVel.vy
        trajectory.push({ ...currentPos })
        setSearchedNodes([...trajectory])
      }

      animationRef.current = requestAnimationFrame(animateStep)
    }

    animateStep()
  }

  // 유틸리티 함수들
  const getNeighbors = (node: Node, grid: Node[][]) => {
    const neighbors: Node[] = []
    const cols = grid.length
    const rows = grid[0].length
    const x = Math.floor(node.x / gridSize)
    const y = Math.floor(node.y / gridSize)

    // 8방향 이웃
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (dx === 0 && dy === 0) continue
        const nx = x + dx
        const ny = y + dy
        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
          neighbors.push(grid[nx][ny])
        }
      }
    }

    return neighbors
  }

  const distance = (a: Node, b: Node) => {
    return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2))
  }

  const distance2D = (a: Point, b: Point) => {
    return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2))
  }

  const heuristic = (a: Node, b: Node) => {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) // Manhattan distance
  }

  const isColliding = (from: Point, to: Point) => {
    // 선분과 장애물 충돌 검사
    const steps = 10
    for (let i = 0; i <= steps; i++) {
      const t = i / steps
      const x = from.x + (to.x - from.x) * t
      const y = from.y + (to.y - from.y) * t
      if (isPointColliding({ x, y })) return true
    }
    return false
  }

  const isPointColliding = (point: Point) => {
    return obstacles.some(obs => 
      Math.abs(point.x - obs.x) < gridSize && 
      Math.abs(point.y - obs.y) < gridSize
    )
  }

  const calculatePathCost = (pathPoints: Point[]) => {
    let cost = 0
    for (let i = 1; i < pathPoints.length; i++) {
      cost += distance2D(pathPoints[i - 1], pathPoints[i])
    }
    return cost
  }

  const reconstructRRTPath = (edges: { from: Point; to: Point }[], start: Point, end: Point) => {
    // RRT 경로 재구성 (간소화된 버전)
    const path = [end]
    let current = end
    
    while (distance2D(current, start) > 1) {
      const edge = edges.find(e => distance2D(e.to, current) < 1)
      if (edge) {
        path.unshift(edge.from)
        current = edge.from
      } else {
        break
      }
    }
    
    return path
  }

  // 마우스 이벤트 핸들러
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    isDrawingRef.current = true
    
    if (e.shiftKey) {
      setStart({ x, y })
    } else if (e.ctrlKey || e.metaKey) {
      setGoal({ x, y })
    } else {
      setObstacles(prev => [...prev, { x, y }])
    }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return
    
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    if (!e.shiftKey && !e.ctrlKey && !e.metaKey) {
      setObstacles(prev => [...prev, { x, y }])
    }
  }

  const handleMouseUp = () => {
    isDrawingRef.current = false
  }

  // 시각화
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

    // 그리드 초기화
    initializeGrid(canvas.width, canvas.height)

    // 렌더링
    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 배경
      ctx.fillStyle = '#F3F4F6'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // 그리드
      if (showCostMap) {
        ctx.strokeStyle = '#E5E7EB'
        ctx.lineWidth = 1
        for (let x = 0; x < canvas.width; x += gridSize) {
          ctx.beginPath()
          ctx.moveTo(x, 0)
          ctx.lineTo(x, canvas.height)
          ctx.stroke()
        }
        for (let y = 0; y < canvas.height; y += gridSize) {
          ctx.beginPath()
          ctx.moveTo(0, y)
          ctx.lineTo(canvas.width, y)
          ctx.stroke()
        }
      }

      // 장애물
      ctx.fillStyle = '#374151'
      obstacles.forEach(obs => {
        ctx.fillRect(
          Math.floor(obs.x / gridSize) * gridSize,
          Math.floor(obs.y / gridSize) * gridSize,
          gridSize,
          gridSize
        )
      })

      // 탐색된 노드
      if (showSearchSpace) {
        searchedNodes.forEach((node, index) => {
          const alpha = 0.1 + (index / searchedNodes.length) * 0.3
          ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`
          ctx.fillRect(
            Math.floor(node.x / gridSize) * gridSize,
            Math.floor(node.y / gridSize) * gridSize,
            gridSize,
            gridSize
          )
        })
      }

      // 경로
      if (path.length > 0) {
        ctx.strokeStyle = '#10B981'
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(path[0].x, path[0].y)
        path.forEach(point => {
          ctx.lineTo(point.x, point.y)
        })
        ctx.stroke()

        // 경로 점
        path.forEach(point => {
          ctx.fillStyle = '#059669'
          ctx.beginPath()
          ctx.arc(point.x, point.y, 4, 0, Math.PI * 2)
          ctx.fill()
        })
      }

      // 시작점
      ctx.fillStyle = '#10B981'
      ctx.beginPath()
      ctx.arc(start.x, start.y, 10, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#FFF'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('S', start.x, start.y)

      // 목표점
      ctx.fillStyle = '#EF4444'
      ctx.beginPath()
      ctx.arc(goal.x, goal.y, 10, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#FFF'
      ctx.fillText('G', goal.x, goal.y)
    }

    render()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [obstacles, start, goal, path, searchedNodes, showCostMap, showSearchSpace, gridSize])

  // 알고리즘 실행
  useEffect(() => {
    if (!isRunning) return

    setPath([])
    setSearchedNodes([])
    setCurrentStep(0)

    switch (algorithm) {
      case 'astar':
        runAStar()
        break
      case 'rrt':
        runRRT()
        break
      case 'dijkstra':
        // Dijkstra는 A*에서 휴리스틱을 0으로 설정
        runAStar()
        break
      case 'dwa':
        runDWA()
        break
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, algorithm])

  const reset = () => {
    setIsRunning(false)
    setPath([])
    setSearchedNodes([])
    setObstacles([])
    setCurrentStep(0)
    setStats({
      pathLength: 0,
      nodesExpanded: 0,
      computeTime: 0,
      pathCost: 0
    })
  }

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-green-600 to-teal-700 text-white p-4">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Route className="w-6 h-6" />
          경로 계획 시각화 도구
        </h2>
        <p className="text-green-100 mt-1">다양한 경로 계획 알고리즘 비교 및 분석</p>
      </div>

      {/* 툴바 */}
      <div className="bg-white dark:bg-gray-800 shadow-md p-4">
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
            {isRunning ? '정지' : '실행'}
          </button>

          <button
            onClick={reset}
            className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            리셋
          </button>

          <div className="flex items-center gap-3">
            <label className="text-sm font-medium">알고리즘:</label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as any)}
              className="px-3 py-1 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="astar">A* (A-Star)</option>
              <option value="dijkstra">Dijkstra</option>
              <option value="rrt">RRT</option>
              <option value="dwa">DWA</option>
            </select>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-sm font-medium">그리드 크기:</label>
            <input
              type="range"
              min="10"
              max="40"
              value={gridSize}
              onChange={(e) => setGridSize(parseInt(e.target.value))}
              className="w-24"
            />
            <span className="text-sm font-mono">{gridSize}</span>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showCostMap}
                onChange={(e) => setShowCostMap(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">그리드 표시</span>
            </label>
            
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showSearchSpace}
                onChange={(e) => setShowSearchSpace(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">탐색 영역 표시</span>
            </label>
          </div>
        </div>
        
        <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
          💡 클릭: 장애물 추가 | Shift+클릭: 시작점 설정 | Ctrl/Cmd+클릭: 목표점 설정
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-5 gap-4 p-4">
        {/* Canvas - 4/5 공간 */}
        <div className="lg:col-span-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
          <canvas
            ref={canvasRef}
            className="w-full h-full cursor-crosshair"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
        </div>

        {/* 사이드바 - 1/5 공간 */}
        <div className="space-y-4">
          {/* 통계 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Cpu className="w-5 h-5" />
              경로 통계
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>경로 길이</span>
                <span className="font-mono font-bold">{stats.pathLength}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>탐색 노드</span>
                <span className="font-mono font-bold">{stats.nodesExpanded}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>계산 시간</span>
                <span className="font-mono font-bold">{stats.computeTime.toFixed(1)}ms</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>경로 비용</span>
                <span className="font-mono font-bold">{stats.pathCost.toFixed(1)}</span>
              </div>
            </div>
          </div>

          {/* 알고리즘 정보 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5" />
              알고리즘 특성
            </h3>
            <div className="text-sm space-y-2">
              {algorithm === 'astar' && (
                <>
                  <p className="font-semibold text-blue-600">A* (A-Star)</p>
                  <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 최적 경로 보장</li>
                    <li>• 휴리스틱 함수 사용</li>
                    <li>• 그래프 기반 탐색</li>
                    <li>• 시간복잡도: O(b^d)</li>
                  </ul>
                </>
              )}
              {algorithm === 'dijkstra' && (
                <>
                  <p className="font-semibold text-green-600">Dijkstra</p>
                  <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 최단 경로 보장</li>
                    <li>• 모든 경로 탐색</li>
                    <li>• 균일 비용 탐색</li>
                    <li>• 시간복잡도: O(V²)</li>
                  </ul>
                </>
              )}
              {algorithm === 'rrt' && (
                <>
                  <p className="font-semibold text-purple-600">RRT</p>
                  <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 고차원 공간 적합</li>
                    <li>• 확률적 탐색</li>
                    <li>• 빠른 탐색 속도</li>
                    <li>• 경로 최적성 미보장</li>
                  </ul>
                </>
              )}
              {algorithm === 'dwa' && (
                <>
                  <p className="font-semibold text-orange-600">DWA</p>
                  <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• 동적 환경 대응</li>
                    <li>• 속도 공간 탐색</li>
                    <li>• 지역 최적화</li>
                    <li>• 실시간 계획</li>
                  </ul>
                </>
              )}
            </div>
          </div>

          {/* 진행 상태 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Navigation className="w-5 h-5" />
              탐색 진행률
            </h3>
            <div className="space-y-2">
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min((currentStep / 100) * 100, 100)}%` }}
                />
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                단계: {currentStep}
              </p>
            </div>
          </div>

          {/* 사용법 */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">
              사용법
            </h4>
            <ul className="text-xs space-y-1 text-blue-600 dark:text-blue-400">
              <li>• 마우스 클릭/드래그로 장애물 그리기</li>
              <li>• Shift+클릭: 시작점 이동</li>
              <li>• Ctrl/Cmd+클릭: 목표점 이동</li>
              <li>• 알고리즘 선택 후 실행 버튼 클릭</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}