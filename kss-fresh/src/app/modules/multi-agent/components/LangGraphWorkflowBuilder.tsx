'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Plus, Trash2, Download, GitBranch, Circle, Square } from 'lucide-react'

interface Node {
  id: string
  label: string
  type: 'agent' | 'start' | 'end'
  x: number
  y: number
  config?: {
    systemPrompt?: string
    tools?: string[]
  }
}

interface Edge {
  id: string
  from: string
  to: string
  condition?: string
  label?: string
}

interface WorkflowTemplate {
  name: string
  description: string
  nodes: Node[]
  edges: Edge[]
}

const templates: WorkflowTemplate[] = [
  {
    name: 'Content Pipeline',
    description: '콘텐츠 생성 파이프라인 워크플로우',
    nodes: [
      { id: 'start', label: 'START', type: 'start', x: 100, y: 200 },
      { id: 'researcher', label: 'Researcher', type: 'agent', x: 250, y: 200, config: { systemPrompt: 'Research topics', tools: ['search', 'scrape'] } },
      { id: 'writer', label: 'Writer', type: 'agent', x: 400, y: 200, config: { systemPrompt: 'Write content', tools: ['generate'] } },
      { id: 'reviewer', label: 'Reviewer', type: 'agent', x: 550, y: 200, config: { systemPrompt: 'Review quality', tools: ['check'] } },
      { id: 'end', label: 'END', type: 'end', x: 700, y: 200 }
    ],
    edges: [
      { id: 'e1', from: 'start', to: 'researcher' },
      { id: 'e2', from: 'researcher', to: 'writer' },
      { id: 'e3', from: 'writer', to: 'reviewer' },
      { id: 'e4', from: 'reviewer', to: 'end', condition: 'approved', label: 'Approved' },
      { id: 'e5', from: 'reviewer', to: 'writer', condition: 'needs_revision', label: 'Revise' }
    ]
  },
  {
    name: 'Customer Support',
    description: '고객 지원 자동화 워크플로우',
    nodes: [
      { id: 'start', label: 'START', type: 'start', x: 100, y: 200 },
      { id: 'classifier', label: 'Classifier', type: 'agent', x: 250, y: 200, config: { systemPrompt: 'Classify request', tools: ['analyze'] } },
      { id: 'technical', label: 'Technical', type: 'agent', x: 400, y: 100, config: { systemPrompt: 'Handle technical', tools: ['diagnose'] } },
      { id: 'billing', label: 'Billing', type: 'agent', x: 400, y: 300, config: { systemPrompt: 'Handle billing', tools: ['payment'] } },
      { id: 'responder', label: 'Responder', type: 'agent', x: 550, y: 200, config: { systemPrompt: 'Generate response', tools: ['reply'] } },
      { id: 'end', label: 'END', type: 'end', x: 700, y: 200 }
    ],
    edges: [
      { id: 'e1', from: 'start', to: 'classifier' },
      { id: 'e2', from: 'classifier', to: 'technical', condition: 'technical', label: 'Technical' },
      { id: 'e3', from: 'classifier', to: 'billing', condition: 'billing', label: 'Billing' },
      { id: 'e4', from: 'technical', to: 'responder' },
      { id: 'e5', from: 'billing', to: 'responder' },
      { id: 'e6', from: 'responder', to: 'end' }
    ]
  }
]

export default function LangGraphWorkflowBuilder() {
  const [nodes, setNodes] = useState<Node[]>([
    { id: 'start', label: 'START', type: 'start', x: 100, y: 200 },
    { id: 'end', label: 'END', type: 'end', x: 700, y: 200 }
  ])
  const [edges, setEdges] = useState<Edge[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [connecting, setConnecting] = useState<string | null>(null)
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionPath, setExecutionPath] = useState<string[]>([])
  const [executionStep, setExecutionStep] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDragging, setIsDragging] = useState<string | null>(null)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })

  useEffect(() => {
    drawGraph()
  }, [nodes, edges, selectedNode, executionPath, executionStep])

  const drawGraph = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw edges
    edges.forEach(edge => {
      const fromNode = nodes.find(n => n.id === edge.from)
      const toNode = nodes.find(n => n.id === edge.to)
      if (!fromNode || !toNode) return

      const isActive = executionPath.slice(0, executionStep + 1).includes(edge.from) &&
                       executionPath.slice(0, executionStep + 1).includes(edge.to)

      ctx.strokeStyle = isActive ? '#10b981' : '#4b5563'
      ctx.lineWidth = isActive ? 3 : 2
      ctx.beginPath()
      ctx.moveTo(fromNode.x + 40, fromNode.y + 20)

      // Draw curved line for conditional edges
      if (edge.condition) {
        const cpX = (fromNode.x + toNode.x) / 2
        const cpY = (fromNode.y + toNode.y) / 2 + (fromNode.y > toNode.y ? 50 : -50)
        ctx.quadraticCurveTo(cpX, cpY, toNode.x, toNode.y + 20)
      } else {
        ctx.lineTo(toNode.x, toNode.y + 20)
      }
      ctx.stroke()

      // Draw arrow
      const angle = Math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x)
      ctx.fillStyle = isActive ? '#10b981' : '#4b5563'
      ctx.beginPath()
      ctx.moveTo(toNode.x - 10, toNode.y + 20)
      ctx.lineTo(toNode.x - 20, toNode.y + 15)
      ctx.lineTo(toNode.x - 20, toNode.y + 25)
      ctx.closePath()
      ctx.fill()

      // Draw edge label
      if (edge.label) {
        const midX = (fromNode.x + toNode.x) / 2
        const midY = (fromNode.y + toNode.y) / 2
        ctx.fillStyle = '#9ca3af'
        ctx.font = '12px Inter'
        ctx.fillText(edge.label, midX, midY - 5)
      }
    })

    // Draw nodes
    nodes.forEach(node => {
      const isSelected = selectedNode === node.id
      const isExecuting = executionPath[executionStep] === node.id
      const isExecuted = executionPath.slice(0, executionStep).includes(node.id)

      // Node background
      if (node.type === 'start') {
        ctx.fillStyle = isExecuting ? '#10b981' : isExecuted ? '#6ee7b7' : '#059669'
      } else if (node.type === 'end') {
        ctx.fillStyle = isExecuting ? '#ef4444' : isExecuted ? '#fca5a5' : '#dc2626'
      } else {
        ctx.fillStyle = isExecuting ? '#3b82f6' : isExecuted ? '#93c5fd' : '#2563eb'
      }

      if (node.type === 'start' || node.type === 'end') {
        ctx.beginPath()
        ctx.arc(node.x + 40, node.y + 20, 30, 0, Math.PI * 2)
        ctx.fill()
      } else {
        ctx.fillRect(node.x, node.y, 80, 40)
      }

      // Selection border
      if (isSelected) {
        ctx.strokeStyle = '#fbbf24'
        ctx.lineWidth = 3
        if (node.type === 'start' || node.type === 'end') {
          ctx.beginPath()
          ctx.arc(node.x + 40, node.y + 20, 32, 0, Math.PI * 2)
          ctx.stroke()
        } else {
          ctx.strokeRect(node.x - 2, node.y - 2, 84, 44)
        }
      }

      // Node label
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 12px Inter'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(node.label, node.x + 40, node.y + 20)
    })
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const clickedNode = nodes.find(node => {
      if (node.type === 'start' || node.type === 'end') {
        const dx = x - (node.x + 40)
        const dy = y - (node.y + 20)
        return Math.sqrt(dx * dx + dy * dy) <= 30
      }
      return x >= node.x && x <= node.x + 80 && y >= node.y && y <= node.y + 40
    })

    if (clickedNode) {
      if (connecting) {
        if (connecting !== clickedNode.id) {
          addEdge(connecting, clickedNode.id)
        }
        setConnecting(null)
      } else {
        setSelectedNode(clickedNode.id)
      }
    } else {
      setSelectedNode(null)
    }
  }

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const clickedNode = nodes.find(node => {
      if (node.type === 'start' || node.type === 'end') {
        const dx = x - (node.x + 40)
        const dy = y - (node.y + 20)
        return Math.sqrt(dx * dx + dy * dy) <= 30
      }
      return x >= node.x && x <= node.x + 80 && y >= node.y && y <= node.y + 40
    })

    if (clickedNode) {
      setIsDragging(clickedNode.id)
      setDragOffset({ x: x - clickedNode.x, y: y - clickedNode.y })
    }
  }

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    setNodes(nodes.map(node =>
      node.id === isDragging
        ? { ...node, x: x - dragOffset.x, y: y - dragOffset.y }
        : node
    ))
  }

  const handleCanvasMouseUp = () => {
    setIsDragging(null)
  }

  const addNode = () => {
    const newId = `agent_${Date.now()}`
    const newNode: Node = {
      id: newId,
      label: `Agent ${nodes.filter(n => n.type === 'agent').length + 1}`,
      type: 'agent',
      x: 300 + Math.random() * 200,
      y: 150 + Math.random() * 100,
      config: {
        systemPrompt: 'Process data',
        tools: ['tool1']
      }
    }
    setNodes([...nodes, newNode])
    setSelectedNode(newId)
  }

  const deleteNode = () => {
    if (!selectedNode || selectedNode === 'start' || selectedNode === 'end') return
    setNodes(nodes.filter(n => n.id !== selectedNode))
    setEdges(edges.filter(e => e.from !== selectedNode && e.to !== selectedNode))
    setSelectedNode(null)
  }

  const addEdge = (from: string, to: string) => {
    const newEdge: Edge = {
      id: `e_${Date.now()}`,
      from,
      to
    }
    setEdges([...edges, newEdge])
  }

  const startConnection = () => {
    if (!selectedNode) return
    setConnecting(selectedNode)
  }

  const loadTemplate = (template: WorkflowTemplate) => {
    setNodes(template.nodes)
    setEdges(template.edges)
    setSelectedNode(null)
    setExecutionPath([])
    setExecutionStep(0)
  }

  const simulateExecution = async () => {
    setIsExecuting(true)
    setExecutionStep(0)

    // Build execution path from start to end
    const path: string[] = ['start']
    let current = 'start'

    while (current !== 'end' && path.length < 20) {
      const outgoingEdges = edges.filter(e => e.from === current)
      if (outgoingEdges.length === 0) break

      // Randomly select path for conditional edges
      const nextEdge = outgoingEdges[Math.floor(Math.random() * outgoingEdges.length)]
      current = nextEdge.to
      path.push(current)
    }

    setExecutionPath(path)

    // Animate execution
    for (let i = 0; i < path.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      setExecutionStep(i)
    }

    setIsExecuting(false)
  }

  const exportWorkflow = () => {
    const code = `from langgraph.graph import StateGraph, END

# Define state
class WorkflowState(TypedDict):
    messages: List[str]
    current_step: str

# Initialize graph
workflow = StateGraph(WorkflowState)

# Add nodes
${nodes.filter(n => n.type === 'agent').map(n =>
  `workflow.add_node("${n.id}", ${n.id}_agent)`
).join('\n')}

# Add edges
${edges.map(e =>
  e.condition
    ? `workflow.add_conditional_edges("${e.from}", route_${e.condition}, {"${e.to}": "${e.to}"})`
    : `workflow.add_edge("${e.from}", "${e.to}")`
).join('\n')}

# Set entry and finish points
workflow.set_entry_point("${nodes.find(n => n.type === 'start')?.id || 'start'}")
workflow.set_finish_point("${nodes.find(n => n.type === 'end')?.id || 'end'}")

# Compile
app = workflow.compile()`

    const blob = new Blob([code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'langgraph_workflow.py'
    a.click()
  }

  return (
    <div className="w-full bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 rounded-xl p-6 text-white">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">LangGraph Workflow Builder</h3>
        <p className="text-slate-300">LangGraph 워크플로우를 시각적으로 설계하고 테스트하세요</p>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-slate-300">Graph Controls</h4>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={addNode}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2 text-sm transition-colors"
            >
              <Plus className="w-4 h-4" />
              Add Agent
            </button>
            <button
              onClick={startConnection}
              disabled={!selectedNode}
              className="px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 text-sm transition-colors"
            >
              <GitBranch className="w-4 h-4" />
              Connect
            </button>
            <button
              onClick={deleteNode}
              disabled={!selectedNode || selectedNode === 'start' || selectedNode === 'end'}
              className="px-3 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 text-sm transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Delete
            </button>
            <button
              onClick={simulateExecution}
              disabled={isExecuting}
              className="px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-700 rounded-lg flex items-center gap-2 text-sm transition-colors"
            >
              <Play className="w-4 h-4" />
              Execute
            </button>
            <button
              onClick={exportWorkflow}
              className="px-3 py-2 bg-amber-600 hover:bg-amber-700 rounded-lg flex items-center gap-2 text-sm transition-colors"
            >
              <Download className="w-4 h-4" />
              Export Code
            </button>
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-slate-300">Templates</h4>
          <div className="flex flex-wrap gap-2">
            {templates.map(template => (
              <button
                key={template.name}
                onClick={() => loadTemplate(template)}
                className="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm transition-colors"
                title={template.description}
              >
                {template.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="bg-slate-800 rounded-lg p-4 mb-4 overflow-hidden">
        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          onClick={handleCanvasClick}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={handleCanvasMouseUp}
          className="w-full border border-slate-700 rounded cursor-crosshair"
        />
        {connecting && (
          <p className="text-sm text-amber-400 mt-2">
            연결 모드: {nodes.find(n => n.id === connecting)?.label}에서 다른 노드를 클릭하세요
          </p>
        )}
      </div>

      {/* Node Details */}
      {selectedNode && (
        <div className="bg-slate-800 rounded-lg p-4">
          <h4 className="text-lg font-semibold mb-3">
            Node Details: {nodes.find(n => n.id === selectedNode)?.label}
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-slate-400 mb-1">Type</p>
              <p className="font-mono">{nodes.find(n => n.id === selectedNode)?.type}</p>
            </div>
            <div>
              <p className="text-slate-400 mb-1">Incoming Edges</p>
              <p className="font-mono">{edges.filter(e => e.to === selectedNode).length}</p>
            </div>
            <div>
              <p className="text-slate-400 mb-1">Outgoing Edges</p>
              <p className="font-mono">{edges.filter(e => e.from === selectedNode).length}</p>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-6 text-sm text-slate-300">
        <div className="flex items-center gap-2">
          <Circle className="w-4 h-4 text-green-500" fill="currentColor" />
          <span>Start Node</span>
        </div>
        <div className="flex items-center gap-2">
          <Square className="w-4 h-4 text-blue-500" fill="currentColor" />
          <span>Agent Node</span>
        </div>
        <div className="flex items-center gap-2">
          <Circle className="w-4 h-4 text-red-500" fill="currentColor" />
          <span>End Node</span>
        </div>
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-purple-500" />
          <span>Conditional Edge</span>
        </div>
      </div>
    </div>
  )
}
