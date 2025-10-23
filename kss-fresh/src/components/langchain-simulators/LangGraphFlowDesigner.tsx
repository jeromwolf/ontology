'use client'

import React, { useState, useRef } from 'react'
import { Play, Plus, Save, Download, GitBranch, Circle } from 'lucide-react'

interface Node {
  id: string
  label: string
  type: 'start' | 'process' | 'decision' | 'end'
  position: { x: number; y: number }
  config?: Record<string, any>
}

interface Edge {
  from: string
  to: string
  condition?: string
  label?: string
}

interface GraphState {
  [key: string]: any
}

const NODE_TYPES = {
  start: { color: '#10b981', icon: '▶️', label: 'Start' },
  process: { color: '#3b82f6', icon: '⚙️', label: 'Process' },
  decision: { color: '#f59e0b', icon: '🔀', label: 'Decision' },
  end: { color: '#ef4444', icon: '🏁', label: 'End' }
}

export default function LangGraphFlowDesigner() {
  const [nodes, setNodes] = useState<Node[]>([
    { id: 'start', label: 'Start', type: 'start', position: { x: 100, y: 50 } }
  ])
  const [edges, setEdges] = useState<Edge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [dragging, setDragging] = useState<string | null>(null)
  const [connecting, setConnecting] = useState<string | null>(null)
  const [graphState, setGraphState] = useState<GraphState>({})
  const [executionPath, setExecutionPath] = useState<string[]>([])
  const [executing, setExecuting] = useState(false)
  const canvasRef = useRef<HTMLDivElement>(null)

  const addNode = (type: keyof typeof NODE_TYPES) => {
    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      label: `${NODE_TYPES[type].label} ${nodes.length}`,
      type,
      position: { x: 100 + nodes.length * 50, y: 150 + nodes.length * 30 },
      config: {}
    }
    setNodes([...nodes, newNode])
  }

  const deleteNode = (id: string) => {
    if (id === 'start') return // Can't delete start node
    setNodes(nodes.filter(n => n.id !== id))
    setEdges(edges.filter(e => e.from !== id && e.to !== id))
    if (selectedNode === id) setSelectedNode(null)
  }

  const handleMouseDown = (id: string, e: React.MouseEvent) => {
    if (e.shiftKey) {
      // Connect nodes
      if (connecting) {
        if (connecting !== id) {
          const fromNode = nodes.find(n => n.id === connecting)
          const toNode = nodes.find(n => n.id === id)

          const newEdge: Edge = {
            from: connecting,
            to: id,
            label: fromNode?.type === 'decision' ? 'condition' : undefined
          }
          setEdges([...edges, newEdge])
        }
        setConnecting(null)
      } else {
        setConnecting(id)
      }
    } else {
      // Drag node
      setDragging(id)
      setSelectedNode(id)
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragging && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      setNodes(nodes.map(n =>
        n.id === dragging
          ? { ...n, position: { x: x - 40, y: y - 25 } }
          : n
      ))
    }
  }

  const handleMouseUp = () => {
    setDragging(null)
  }

  const executeGraph = async () => {
    setExecuting(true)
    setExecutionPath([])
    const path: string[] = []
    let currentState: GraphState = { step: 0 }

    // Find start node
    let currentNode = nodes.find(n => n.type === 'start')
    if (!currentNode) return

    path.push(currentNode.id)

    while (currentNode && currentNode.type !== 'end') {
      await new Promise(resolve => setTimeout(resolve, 800))

      // Find next node
      const outgoingEdges = edges.filter(e => e.from === currentNode!.id)

      if (outgoingEdges.length === 0) break

      if (currentNode.type === 'decision') {
        // Simulate decision
        const randomEdge = outgoingEdges[Math.floor(Math.random() * outgoingEdges.length)]
        currentNode = nodes.find(n => n.id === randomEdge.to)
        currentState.lastDecision = randomEdge.label || 'branch'
      } else {
        currentNode = nodes.find(n => n.id === outgoingEdges[0].to)
      }

      if (currentNode) {
        path.push(currentNode.id)
        currentState.step++
      }

      setExecutionPath([...path])
      setGraphState({ ...currentState })
    }

    setExecuting(false)
  }

  const exportCode = () => {
    let code = `from langgraph.graph import StateGraph, END

# Define graph state
class GraphState(TypedDict):
    step: int
    data: str

# Create graph
workflow = StateGraph(GraphState)

# Add nodes
`

    nodes.forEach(node => {
      if (node.type === 'start') return
      if (node.type === 'end') return
      code += `workflow.add_node("${node.id}", ${node.id}_node)\n`
    })

    code += `\n# Add edges\n`
    code += `workflow.set_entry_point("${nodes[0].id}")\n`

    edges.forEach(edge => {
      if (nodes.find(n => n.id === edge.from)?.type === 'decision') {
        code += `workflow.add_conditional_edges("${edge.from}", router_function)\n`
      } else {
        code += `workflow.add_edge("${edge.from}", "${edge.to}")\n`
      }
    })

    code += `\n# Compile\napp = workflow.compile()\n`

    navigator.clipboard.writeText(code)
    alert('LangGraph code copied to clipboard!')
  }

  const selectedNodeData = nodes.find(n => n.id === selectedNode)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            🔄 LangGraph Flow Designer
          </h1>
          <p className="text-gray-300 text-lg">
            Design state-based workflows with visual node editor. Shift+Click to connect nodes.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Node Palette */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Plus className="w-5 h-5" />
                Nodes
              </h3>

              <div className="space-y-2">
                {Object.entries(NODE_TYPES).map(([type, config]) => (
                  <button
                    key={type}
                    onClick={() => addNode(type as keyof typeof NODE_TYPES)}
                    disabled={type === 'start'}
                    className="w-full px-4 py-3 bg-gray-700/50 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed border border-gray-600 rounded-lg transition-all flex items-center gap-3"
                    style={{ borderLeftColor: config.color, borderLeftWidth: 4 }}
                  >
                    <span className="text-2xl">{config.icon}</span>
                    <span className="text-sm font-medium">{config.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Node Config */}
            {selectedNodeData && (
              <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4">Node Config</h3>

                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium mb-2">Label</label>
                    <input
                      type="text"
                      value={selectedNodeData.label}
                      onChange={(e) => {
                        setNodes(nodes.map(n =>
                          n.id === selectedNode
                            ? { ...n, label: e.target.value }
                            : n
                        ))
                      }}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Type</label>
                    <div className="px-3 py-2 bg-gray-900 rounded border border-gray-600">
                      {selectedNodeData.type}
                    </div>
                  </div>

                  {selectedNodeData.type !== 'start' && (
                    <button
                      onClick={() => deleteNode(selectedNodeData.id)}
                      className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
                    >
                      Delete Node
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Controls */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Actions</h3>

              <div className="space-y-2">
                <button
                  onClick={executeGraph}
                  disabled={executing || nodes.length < 2}
                  className="w-full px-4 py-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 rounded flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  {executing ? 'Running...' : 'Execute'}
                </button>

                <button
                  onClick={exportCode}
                  disabled={nodes.length < 2}
                  className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 rounded flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Export Code
                </button>
              </div>
            </div>
          </div>

          {/* Canvas */}
          <div className="lg:col-span-3 space-y-4">
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <GitBranch className="w-5 h-5" />
                Graph Canvas
              </h3>

              <div
                ref={canvasRef}
                className="relative bg-gray-900 rounded-lg border-2 border-dashed border-gray-600 h-[600px] overflow-hidden"
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              >
                {/* Edges */}
                <svg className="absolute inset-0 pointer-events-none w-full h-full">
                  {edges.map((edge, idx) => {
                    const from = nodes.find(n => n.id === edge.from)
                    const to = nodes.find(n => n.id === edge.to)
                    if (!from || !to) return null

                    const x1 = from.position.x + 40
                    const y1 = from.position.y + 25
                    const x2 = to.position.x + 40
                    const y2 = to.position.y + 25

                    const isActive = executionPath.includes(from.id) && executionPath.includes(to.id)

                    return (
                      <g key={idx}>
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke={isActive ? '#10b981' : '#f59e0b'}
                          strokeWidth={isActive ? '3' : '2'}
                          markerEnd="url(#arrowhead)"
                          className={isActive ? 'animate-pulse' : ''}
                        />
                        {edge.label && (
                          <text
                            x={(x1 + x2) / 2}
                            y={(y1 + y2) / 2 - 5}
                            fill="#9ca3af"
                            fontSize="12"
                            textAnchor="middle"
                          >
                            {edge.label}
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
                      <polygon points="0 0, 10 3, 0 6" fill="#f59e0b" />
                    </marker>
                  </defs>
                </svg>

                {/* Nodes */}
                {nodes.map(node => {
                  const config = NODE_TYPES[node.type]
                  const isActive = executionPath.includes(node.id)
                  const isCurrent = executionPath[executionPath.length - 1] === node.id

                  return (
                    <div
                      key={node.id}
                      className={`absolute cursor-move select-none transition-all ${
                        selectedNode === node.id ? 'ring-2 ring-white' : ''
                      } ${connecting === node.id ? 'ring-2 ring-green-500' : ''} ${
                        isCurrent ? 'scale-110' : ''
                      }`}
                      style={{
                        left: node.position.x,
                        top: node.position.y,
                        width: 80,
                      }}
                      onMouseDown={(e) => handleMouseDown(node.id, e)}
                    >
                      <div
                        className={`rounded-lg p-3 border-2 transition-all ${
                          isActive
                            ? 'bg-green-900/50 border-green-500'
                            : 'bg-gray-800 border-gray-600'
                        }`}
                        style={{
                          borderColor: isActive ? '#10b981' : config.color
                        }}
                      >
                        <div className="text-center">
                          <div className="text-2xl mb-1">{config.icon}</div>
                          <div className="text-xs font-medium truncate">{node.label}</div>
                        </div>
                      </div>
                    </div>
                  )
                })}

                {nodes.length === 1 && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-500 pointer-events-none">
                    <div className="text-center">
                      <Circle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Add nodes and connect them with Shift+Click</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* State Viewer */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Graph State</h3>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-900 rounded-lg border border-gray-600 p-4">
                  <div className="text-sm text-gray-400 mb-2">Current State</div>
                  <pre className="text-xs font-mono text-green-400">
                    {JSON.stringify(graphState, null, 2) || '{}'}
                  </pre>
                </div>

                <div className="bg-gray-900 rounded-lg border border-gray-600 p-4">
                  <div className="text-sm text-gray-400 mb-2">Execution Path</div>
                  <div className="space-y-1">
                    {executionPath.map((nodeId, idx) => {
                      const node = nodes.find(n => n.id === nodeId)
                      return (
                        <div key={idx} className="text-xs font-mono text-amber-400">
                          {idx + 1}. {node?.label || nodeId}
                        </div>
                      )
                    })}
                    {executionPath.length === 0 && (
                      <div className="text-xs text-gray-500">No execution yet</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
