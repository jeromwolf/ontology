'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, RotateCcw, Navigation, Info, Maximize, Trash2, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

interface Point {
  x: number
  y: number
}

interface Obstacle {
  x: number
  y: number
  radius: number
}

interface Node {
  point: Point
  parent: Node | null
  cost: number
}

type Algorithm = 'rrt' | 'rrt-star' | 'a-star'
type InteractionMode = 'start' | 'goal' | 'obstacle'

export default function PathPlanningVisualizer() {
  const [start, setStart] = useState<Point>({ x: 100, y: 500 })
  const [goal, setGoal] = useState<Point>({ x: 700, y: 100 })
  const [obstacles, setObstacles] = useState<Obstacle[]>([
    { x: 300, y: 300, radius: 50 },
    { x: 500, y: 400, radius: 60 },
    { x: 600, y: 200, radius: 45 }
  ])

  const [algorithm, setAlgorithm] = useState<Algorithm>('rrt')
  const [mode, setMode] = useState<InteractionMode>('obstacle')
  const [isRunning, setIsRunning] = useState(false)
  const [iterations, setIterations] = useState(0)
  const [pathFound, setPathFound] = useState(false)

  const [tree, setTree] = useState<Node[]>([])
  const [path, setPath] = useState<Point[]>([])

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Check if point is in collision with obstacles
  const isCollision = useCallback(
    (point: Point): boolean => {
      return obstacles.some((obs) => {
        const dx = point.x - obs.x
        const dy = point.y - obs.y
        return Math.sqrt(dx * dx + dy * dy) < obs.radius + 5
      })
    },
    [obstacles]
  )

  // Check if line segment collides with obstacles
  const lineCollision = useCallback(
    (p1: Point, p2: Point): boolean => {
      const steps = 20
      for (let i = 0; i <= steps; i++) {
        const t = i / steps
        const point = {
          x: p1.x + (p2.x - p1.x) * t,
          y: p1.y + (p2.y - p1.y) * t
        }
        if (isCollision(point)) return true
      }
      return false
    },
    [isCollision]
  )

  // Distance between two points
  const distance = (p1: Point, p2: Point): number => {
    const dx = p1.x - p2.x
    const dy = p1.y - p2.y
    return Math.sqrt(dx * dx + dy * dy)
  }

  // Find nearest node in tree
  const findNearest = (point: Point, nodes: Node[]): Node => {
    let nearest = nodes[0]
    let minDist = distance(point, nearest.point)

    nodes.forEach((node) => {
      const dist = distance(point, node.point)
      if (dist < minDist) {
        minDist = dist
        nearest = node
      }
    })

    return nearest
  }

  // Steer from one point toward another
  const steer = (from: Point, to: Point, maxDist: number = 30): Point => {
    const dist = distance(from, to)
    if (dist <= maxDist) return to

    const ratio = maxDist / dist
    return {
      x: from.x + (to.x - from.x) * ratio,
      y: from.y + (to.y - from.y) * ratio
    }
  }

  // RRT algorithm step
  const rrtStep = useCallback(() => {
    setTree((currentTree) => {
      // Random sample (90% random, 10% goal-biased)
      const randomPoint: Point =
        Math.random() < 0.9
          ? { x: Math.random() * 800, y: Math.random() * 600 }
          : goal

      // Find nearest node
      const nearest = findNearest(randomPoint, currentTree)

      // Steer toward random point
      const newPoint = steer(nearest.point, randomPoint)

      // Check collision
      if (isCollision(newPoint) || lineCollision(nearest.point, newPoint)) {
        return currentTree
      }

      // Add new node
      const newNode: Node = {
        point: newPoint,
        parent: nearest,
        cost: nearest.cost + distance(nearest.point, newPoint)
      }

      // Check if goal is reached
      if (distance(newPoint, goal) < 30 && !lineCollision(newPoint, goal)) {
        const goalNode: Node = {
          point: goal,
          parent: newNode,
          cost: newNode.cost + distance(newPoint, goal)
        }

        // Extract path
        const finalPath: Point[] = []
        let current: Node | null = goalNode
        while (current) {
          finalPath.unshift(current.point)
          current = current.parent
        }

        setPath(finalPath)
        setPathFound(true)
        setIsRunning(false)

        return [...currentTree, newNode, goalNode]
      }

      return [...currentTree, newNode]
    })

    setIterations((prev) => prev + 1)
  }, [goal, isCollision, lineCollision])

  // RRT* algorithm step (with rewiring)
  const rrtStarStep = useCallback(() => {
    setTree((currentTree) => {
      const randomPoint: Point =
        Math.random() < 0.9
          ? { x: Math.random() * 800, y: Math.random() * 600 }
          : goal

      const nearest = findNearest(randomPoint, currentTree)
      const newPoint = steer(nearest.point, randomPoint)

      if (isCollision(newPoint) || lineCollision(nearest.point, newPoint)) {
        return currentTree
      }

      // Find nearby nodes within radius
      const radius = 50
      const nearbyNodes = currentTree.filter(
        (node) => distance(node.point, newPoint) < radius
      )

      // Find best parent (lowest cost)
      let bestParent = nearest
      let minCost = nearest.cost + distance(nearest.point, newPoint)

      nearbyNodes.forEach((node) => {
        const newCost = node.cost + distance(node.point, newPoint)
        if (
          newCost < minCost &&
          !lineCollision(node.point, newPoint)
        ) {
          bestParent = node
          minCost = newCost
        }
      })

      const newNode: Node = {
        point: newPoint,
        parent: bestParent,
        cost: minCost
      }

      // Rewire nearby nodes
      const updatedTree = [...currentTree, newNode]
      nearbyNodes.forEach((node) => {
        const newCost = newNode.cost + distance(newNode.point, node.point)
        if (
          newCost < node.cost &&
          !lineCollision(newNode.point, node.point)
        ) {
          node.parent = newNode
          node.cost = newCost
        }
      })

      // Check if goal is reached
      if (distance(newPoint, goal) < 30 && !lineCollision(newPoint, goal)) {
        const goalNode: Node = {
          point: goal,
          parent: newNode,
          cost: newNode.cost + distance(newPoint, goal)
        }

        const finalPath: Point[] = []
        let current: Node | null = goalNode
        while (current) {
          finalPath.unshift(current.point)
          current = current.parent
        }

        setPath(finalPath)
        setPathFound(true)
        setIsRunning(false)

        return [...updatedTree, goalNode]
      }

      return updatedTree
    })

    setIterations((prev) => prev + 1)
  }, [goal, isCollision, lineCollision])

  // A* algorithm (grid-based)
  const runAStar = useCallback(() => {
    const gridSize = 20
    const cols = Math.ceil(800 / gridSize)
    const rows = Math.ceil(600 / gridSize)

    // Convert to grid coordinates
    const startGrid = {
      x: Math.floor(start.x / gridSize),
      y: Math.floor(start.y / gridSize)
    }
    const goalGrid = {
      x: Math.floor(goal.x / gridSize),
      y: Math.floor(goal.y / gridSize)
    }

    // Heuristic (Manhattan distance)
    const heuristic = (p: Point) => {
      return Math.abs(p.x - goalGrid.x) + Math.abs(p.y - goalGrid.y)
    }

    const openSet: Node[] = [
      { point: startGrid, parent: null, cost: 0 }
    ]
    const closedSet = new Set<string>()

    const getKey = (p: Point) => `${p.x},${p.y}`

    while (openSet.length > 0) {
      // Find node with lowest f = g + h
      let current = openSet[0]
      let currentIdx = 0

      openSet.forEach((node, idx) => {
        const currentF = current.cost + heuristic(current.point)
        const nodeF = node.cost + heuristic(node.point)
        if (nodeF < currentF) {
          current = node
          currentIdx = idx
        }
      })

      // Check if goal reached
      if (current.point.x === goalGrid.x && current.point.y === goalGrid.y) {
        const finalPath: Point[] = []
        let temp: Node | null = current
        while (temp) {
          finalPath.unshift({
            x: temp.point.x * gridSize + gridSize / 2,
            y: temp.point.y * gridSize + gridSize / 2
          })
          temp = temp.parent
        }

        setPath(finalPath)
        setPathFound(true)
        setIsRunning(false)
        setTree([])
        return
      }

      openSet.splice(currentIdx, 1)
      closedSet.add(getKey(current.point))

      // Explore neighbors
      const neighbors = [
        { x: current.point.x + 1, y: current.point.y },
        { x: current.point.x - 1, y: current.point.y },
        { x: current.point.x, y: current.point.y + 1 },
        { x: current.point.x, y: current.point.y - 1 }
      ]

      neighbors.forEach((neighbor) => {
        if (
          neighbor.x < 0 ||
          neighbor.x >= cols ||
          neighbor.y < 0 ||
          neighbor.y >= rows
        ) {
          return
        }

        const neighborPoint = {
          x: neighbor.x * gridSize + gridSize / 2,
          y: neighbor.y * gridSize + gridSize / 2
        }

        if (isCollision(neighborPoint) || closedSet.has(getKey(neighbor))) {
          return
        }

        const newCost = current.cost + 1

        const existing = openSet.find(
          (n) => n.point.x === neighbor.x && n.point.y === neighbor.y
        )

        if (!existing) {
          openSet.push({
            point: neighbor,
            parent: current,
            cost: newCost
          })
        } else if (newCost < existing.cost) {
          existing.cost = newCost
          existing.parent = current
        }
      })
    }

    // No path found
    setIsRunning(false)
  }, [start, goal, isCollision])

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.2)'
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

    // Draw obstacles
    obstacles.forEach((obs) => {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.3)'
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(obs.x, obs.y, obs.radius, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    })

    // Draw tree edges
    if (algorithm !== 'a-star') {
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)'
      ctx.lineWidth = 1
      tree.forEach((node) => {
        if (node.parent) {
          ctx.beginPath()
          ctx.moveTo(node.parent.point.x, node.parent.point.y)
          ctx.lineTo(node.point.x, node.point.y)
          ctx.stroke()
        }
      })
    }

    // Draw path
    if (path.length > 1) {
      ctx.strokeStyle = '#10b981'
      ctx.lineWidth = 4
      ctx.beginPath()
      ctx.moveTo(path[0].x, path[0].y)
      path.forEach((point) => {
        ctx.lineTo(point.x, point.y)
      })
      ctx.stroke()

      // Draw waypoints
      ctx.fillStyle = '#10b981'
      path.forEach((point) => {
        ctx.beginPath()
        ctx.arc(point.x, point.y, 4, 0, Math.PI * 2)
        ctx.fill()
      })
    }

    // Draw start
    const startGradient = ctx.createRadialGradient(start.x, start.y, 5, start.x, start.y, 20)
    startGradient.addColorStop(0, '#22c55e')
    startGradient.addColorStop(1, '#16a34a')
    ctx.fillStyle = startGradient
    ctx.beginPath()
    ctx.arc(start.x, start.y, 20, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#86efac'
    ctx.lineWidth = 3
    ctx.stroke()

    // Start label
    ctx.font = 'bold 14px Inter, sans-serif'
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'center'
    ctx.fillText('START', start.x, start.y + 5)

    // Draw goal
    const goalGradient = ctx.createRadialGradient(goal.x, goal.y, 5, goal.x, goal.y, 20)
    goalGradient.addColorStop(0, '#f97316')
    goalGradient.addColorStop(1, '#ea580c')
    ctx.fillStyle = goalGradient
    ctx.beginPath()
    ctx.arc(goal.x, goal.y, 20, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#fdba74'
    ctx.lineWidth = 3
    ctx.stroke()

    // Goal label
    ctx.fillStyle = '#fff'
    ctx.fillText('GOAL', goal.x, goal.y + 5)

    // Draw stats
    ctx.font = 'bold 14px Inter, sans-serif'
    ctx.fillStyle = '#e5e7eb'
    ctx.textAlign = 'left'
    ctx.fillText(`Algorithm: ${algorithm.toUpperCase()}`, 20, 30)
    ctx.fillText(`Iterations: ${iterations}`, 20, 50)
    if (pathFound) {
      ctx.fillStyle = '#10b981'
      ctx.fillText(`Path Found! Length: ${path.length} waypoints`, 20, 70)
    }
  }, [start, goal, obstacles, tree, path, algorithm, iterations, pathFound])

  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      draw()
      return
    }

    const animate = () => {
      if (algorithm === 'rrt') {
        rrtStep()
      } else if (algorithm === 'rrt-star') {
        rrtStarStep()
      }

      draw()

      if (isRunning) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, algorithm, draw, rrtStep, rrtStarStep])

  // Initial draw
  useEffect(() => {
    draw()
  }, [draw])

  // Canvas click handler
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * 800
    const y = ((e.clientY - rect.top) / rect.height) * 600

    if (mode === 'start') {
      setStart({ x, y })
    } else if (mode === 'goal') {
      setGoal({ x, y })
    } else if (mode === 'obstacle') {
      setObstacles((prev) => [...prev, { x, y, radius: 40 }])
    }
  }

  const handleStart = () => {
    setTree([{ point: start, parent: null, cost: 0 }])
    setPath([])
    setIterations(0)
    setPathFound(false)

    if (algorithm === 'a-star') {
      runAStar()
    } else {
      setIsRunning(true)
    }
  }

  const handleReset = () => {
    setIsRunning(false)
    setTree([])
    setPath([])
    setIterations(0)
    setPathFound(false)
  }

  const handleClearObstacles = () => {
    setObstacles([])
    handleReset()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-orange-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-between mb-4">
            <Link
              href="/modules/robotics-manipulation"
              className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700"
            >
              <ArrowLeft className="w-4 h-4 text-slate-400" />
              <span className="text-slate-300 text-sm">모듈로 돌아가기</span>
            </Link>

            <div className="flex items-center gap-3">
              <Navigation className="w-10 h-10 text-orange-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-orange-400 to-green-400 bg-clip-text text-transparent">
                Path Planning Visualizer
              </h1>
            </div>

            <Link
              href="/modules/robotics-manipulation/simulators/path-planning-visualizer"
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
              title="전체화면으로 보기"
            >
              <Maximize className="w-5 h-5 text-slate-400 hover:text-orange-400" />
            </Link>
          </div>
          <p className="text-slate-300 text-lg">
            RRT, RRT*, A* 알고리즘으로 장애물을 피하는 경로를 계획합니다
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-3 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              onClick={handleCanvasClick}
              className="w-full bg-slate-950 rounded-lg cursor-crosshair"
            />
          </div>

          {/* Controls */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6 space-y-6">
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Navigation className="w-5 h-5 text-orange-400" />
                Algorithm
              </h2>

              <div className="space-y-2">
                {[
                  { id: 'rrt', name: 'RRT', desc: 'Rapidly-exploring Random Tree' },
                  { id: 'rrt-star', name: 'RRT*', desc: 'Optimal RRT with rewiring' },
                  { id: 'a-star', name: 'A*', desc: 'Grid-based optimal search' }
                ].map((alg) => (
                  <label
                    key={alg.id}
                    className="flex items-start gap-3 cursor-pointer p-2 hover:bg-slate-700/50 rounded-lg"
                  >
                    <input
                      type="radio"
                      name="algorithm"
                      value={alg.id}
                      checked={algorithm === alg.id}
                      onChange={(e) => {
                        setAlgorithm(e.target.value as Algorithm)
                        handleReset()
                      }}
                      className="mt-1 accent-orange-500"
                    />
                    <div>
                      <div className="text-sm font-medium text-white">{alg.name}</div>
                      <div className="text-xs text-slate-400">{alg.desc}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Interaction Mode</h3>
              <div className="space-y-2">
                {[
                  { id: 'start', name: 'Set Start', color: 'green' },
                  { id: 'goal', name: 'Set Goal', color: 'orange' },
                  { id: 'obstacle', name: 'Add Obstacle', color: 'red' }
                ].map((m) => (
                  <label
                    key={m.id}
                    className="flex items-center gap-3 cursor-pointer p-2 hover:bg-slate-700/50 rounded-lg"
                  >
                    <input
                      type="radio"
                      name="mode"
                      value={m.id}
                      checked={mode === m.id}
                      onChange={(e) => setMode(e.target.value as InteractionMode)}
                      className="accent-orange-500"
                    />
                    <span className="text-sm text-slate-300">{m.name}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Controls */}
            <div className="space-y-3">
              <button
                onClick={isRunning ? () => setIsRunning(false) : handleStart}
                disabled={pathFound}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-semibold transition-colors ${
                  isRunning
                    ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                    : 'bg-orange-600 hover:bg-orange-700 text-white disabled:opacity-50'
                }`}
              >
                {isRunning ? (
                  <>
                    <Pause className="w-5 h-5" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start Planning
                  </>
                )}
              </button>

              <button
                onClick={handleReset}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition-colors"
              >
                <RotateCcw className="w-5 h-5" />
                Reset
              </button>

              <button
                onClick={handleClearObstacles}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-colors"
              >
                <Trash2 className="w-5 h-5" />
                Clear Obstacles
              </button>
            </div>

            {/* Info */}
            <div className="bg-slate-900/50 rounded-lg p-4 text-sm text-slate-300">
              <div className="flex items-start gap-2 mb-2">
                <Info className="w-4 h-4 text-orange-400 mt-0.5" />
                <p className="font-semibold text-white">Path Planning</p>
              </div>
              <p className="text-xs">
                장애물이 있는 환경에서 시작점부터 목표점까지 충돌 없는 경로를 찾습니다.
                RRT는 빠르지만 최적이 아니고, RRT*는 점진적 최적화를, A*는 그리드 기반 최적 경로를 보장합니다.
              </p>
            </div>

            {/* Stats */}
            <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-4 text-xs font-mono">
              <p className="text-orange-300 mb-2">Statistics:</p>
              <p className="text-slate-300">Obstacles: {obstacles.length}</p>
              <p className="text-slate-300">Tree Nodes: {tree.length}</p>
              {pathFound && <p className="text-green-400">Path Length: {path.length}</p>}
            </div>
          </div>
        </div>

        {/* Theory Panel */}
        <div className="mt-6 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-3 text-orange-400">경로 계획 알고리즘 비교</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">RRT</h4>
              <p>무작위 샘플링으로 빠르게 탐색하지만 최적 경로를 보장하지 않음</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">RRT*</h4>
              <p>RRT + 재연결(rewiring)로 점진적으로 최적 경로에 수렴</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">A*</h4>
              <p>그리드 기반 휴리스틱 검색으로 최적 경로 보장 (완전성 + 최적성)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
