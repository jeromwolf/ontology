'use client'

import React, { useState } from 'react'
import { Play, Plus, Trash2, Wrench, Brain, CheckCircle, AlertCircle } from 'lucide-react'

interface CustomTool {
  id: string
  name: string
  description: string
  parameters: string[]
}

interface ExecutionStep {
  step: number
  type: 'thought' | 'action' | 'observation' | 'final'
  content: string
  tool?: string
  timestamp: number
}

type AgentType = 'react' | 'plan-execute' | 'conversational'

const BUILTIN_TOOLS = [
  {
    id: 'search',
    name: 'Search',
    description: 'Search the web for information',
    parameters: ['query']
  },
  {
    id: 'calculator',
    name: 'Calculator',
    description: 'Perform mathematical calculations',
    parameters: ['expression']
  },
  {
    id: 'weather',
    name: 'Weather',
    description: 'Get weather information',
    parameters: ['location']
  },
  {
    id: 'translator',
    name: 'Translator',
    description: 'Translate text between languages',
    parameters: ['text', 'target_language']
  }
]

export default function AgentToolsWorkshop() {
  const [agentType, setAgentType] = useState<AgentType>('react')
  const [customTools, setCustomTools] = useState<CustomTool[]>([])
  const [selectedTools, setSelectedTools] = useState<string[]>(['search', 'calculator'])
  const [task, setTask] = useState('')
  const [executionLog, setExecutionLog] = useState<ExecutionStep[]>([])
  const [executing, setExecuting] = useState(false)
  const [showToolCreator, setShowToolCreator] = useState(false)

  // Tool Creator State
  const [newToolName, setNewToolName] = useState('')
  const [newToolDesc, setNewToolDesc] = useState('')
  const [newToolParams, setNewToolParams] = useState('')

  const allTools = [...BUILTIN_TOOLS, ...customTools]
  const activeTools = allTools.filter(t => selectedTools.includes(t.id))

  const toggleTool = (toolId: string) => {
    setSelectedTools(prev =>
      prev.includes(toolId)
        ? prev.filter(id => id !== toolId)
        : [...prev, toolId]
    )
  }

  const createTool = () => {
    if (!newToolName || !newToolDesc) return

    const newTool: CustomTool = {
      id: `custom-${Date.now()}`,
      name: newToolName,
      description: newToolDesc,
      parameters: newToolParams.split(',').map(p => p.trim()).filter(Boolean)
    }

    setCustomTools([...customTools, newTool])
    setSelectedTools([...selectedTools, newTool.id])
    setNewToolName('')
    setNewToolDesc('')
    setNewToolParams('')
    setShowToolCreator(false)
  }

  const deleteTool = (toolId: string) => {
    setCustomTools(customTools.filter(t => t.id !== toolId))
    setSelectedTools(selectedTools.filter(id => id !== toolId))
  }

  const executeAgent = async () => {
    if (!task.trim() || activeTools.length === 0) return

    setExecuting(true)
    setExecutionLog([])

    const log: ExecutionStep[] = []
    let stepNum = 1

    if (agentType === 'react') {
      // ReAct Pattern: Reasoning + Acting
      log.push({
        step: stepNum++,
        type: 'thought',
        content: `I need to solve: "${task}". Let me think about which tools to use.`,
        timestamp: Date.now()
      })

      await delay(800)

      // Select appropriate tool
      const tool = activeTools[Math.floor(Math.random() * activeTools.length)]
      log.push({
        step: stepNum++,
        type: 'action',
        content: `Using ${tool.name} tool`,
        tool: tool.id,
        timestamp: Date.now()
      })

      await delay(1000)

      log.push({
        step: stepNum++,
        type: 'observation',
        content: `${tool.name} returned: [Simulated result based on "${task}"]`,
        timestamp: Date.now()
      })

      await delay(800)

      log.push({
        step: stepNum++,
        type: 'thought',
        content: 'Based on the observation, I can now formulate an answer.',
        timestamp: Date.now()
      })

      await delay(600)

      log.push({
        step: stepNum++,
        type: 'final',
        content: `Final Answer: Here's the solution to "${task}" using ${tool.name}.`,
        timestamp: Date.now()
      })
    } else if (agentType === 'plan-execute') {
      // Plan and Execute Pattern
      log.push({
        step: stepNum++,
        type: 'thought',
        content: 'Creating execution plan...',
        timestamp: Date.now()
      })

      await delay(800)

      log.push({
        step: stepNum++,
        type: 'thought',
        content: `Plan:
1. Gather information using ${activeTools[0]?.name || 'available tools'}
2. Process the data
3. Formulate final answer`,
        timestamp: Date.now()
      })

      await delay(1000)

      for (let i = 0; i < Math.min(2, activeTools.length); i++) {
        const tool = activeTools[i]
        log.push({
          step: stepNum++,
          type: 'action',
          content: `Executing step ${i + 1}: ${tool.name}`,
          tool: tool.id,
          timestamp: Date.now()
        })

        await delay(1000)

        log.push({
          step: stepNum++,
          type: 'observation',
          content: `Step ${i + 1} completed successfully`,
          timestamp: Date.now()
        })

        await delay(600)
      }

      log.push({
        step: stepNum++,
        type: 'final',
        content: 'Plan executed successfully. Task completed!',
        timestamp: Date.now()
      })
    } else {
      // Conversational Agent
      log.push({
        step: stepNum++,
        type: 'thought',
        content: 'Analyzing the conversation context...',
        timestamp: Date.now()
      })

      await delay(800)

      log.push({
        step: stepNum++,
        type: 'action',
        content: 'Retrieving relevant information',
        timestamp: Date.now()
      })

      await delay(1000)

      log.push({
        step: stepNum++,
        type: 'final',
        content: 'Based on our conversation, here\'s my response...',
        timestamp: Date.now()
      })
    }

    setExecutionLog(log)
    setExecuting(false)
  }

  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

  const getStepIcon = (type: string) => {
    switch (type) {
      case 'thought':
        return <Brain className="w-5 h-5 text-blue-500" />
      case 'action':
        return <Wrench className="w-5 h-5 text-amber-500" />
      case 'observation':
        return <AlertCircle className="w-5 h-5 text-purple-500" />
      case 'final':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            🛠️ Agent Tools Workshop
          </h1>
          <p className="text-gray-300 text-lg">
            Build AI agents with custom tools and watch them reason through tasks.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Agent Configuration */}
          <div className="lg:col-span-1 space-y-4">
            {/* Agent Type */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Agent Type</h3>

              <div className="space-y-2">
                {[
                  { id: 'react', name: 'ReAct', desc: 'Reasoning + Acting' },
                  { id: 'plan-execute', name: 'Plan & Execute', desc: 'Strategic planning' },
                  { id: 'conversational', name: 'Conversational', desc: 'Chat-based agent' }
                ].map(type => (
                  <button
                    key={type.id}
                    onClick={() => setAgentType(type.id as AgentType)}
                    className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-all ${
                      agentType === type.id
                        ? 'bg-amber-600 border-amber-500'
                        : 'bg-gray-700/50 border-gray-600 hover:bg-gray-600'
                    }`}
                  >
                    <div className="font-medium">{type.name}</div>
                    <div className="text-xs text-gray-300 mt-1">{type.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Available Tools */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold">Tools</h3>
                <button
                  onClick={() => setShowToolCreator(!showToolCreator)}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded flex items-center gap-2"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-2">
                {/* Built-in Tools */}
                <div className="text-xs font-semibold text-gray-400 mb-2">BUILT-IN</div>
                {BUILTIN_TOOLS.map(tool => (
                  <label
                    key={tool.id}
                    className="flex items-start gap-3 p-3 bg-gray-700/50 rounded-lg cursor-pointer hover:bg-gray-600"
                  >
                    <input
                      type="checkbox"
                      checked={selectedTools.includes(tool.id)}
                      onChange={() => toggleTool(tool.id)}
                      className="mt-1"
                    />
                    <div className="flex-1">
                      <div className="font-medium text-sm">{tool.name}</div>
                      <div className="text-xs text-gray-400">{tool.description}</div>
                    </div>
                  </label>
                ))}

                {/* Custom Tools */}
                {customTools.length > 0 && (
                  <>
                    <div className="text-xs font-semibold text-gray-400 mb-2 mt-4">CUSTOM</div>
                    {customTools.map(tool => (
                      <label
                        key={tool.id}
                        className="flex items-start gap-3 p-3 bg-purple-900/30 border border-purple-700 rounded-lg cursor-pointer hover:bg-purple-900/50"
                      >
                        <input
                          type="checkbox"
                          checked={selectedTools.includes(tool.id)}
                          onChange={() => toggleTool(tool.id)}
                          className="mt-1"
                        />
                        <div className="flex-1">
                          <div className="font-medium text-sm">{tool.name}</div>
                          <div className="text-xs text-gray-400">{tool.description}</div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.preventDefault()
                            deleteTool(tool.id)
                          }}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </label>
                    ))}
                  </>
                )}
              </div>
            </div>

            {/* Tool Creator */}
            {showToolCreator && (
              <div className="bg-blue-900/30 backdrop-blur border border-blue-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4">Create Tool</h3>

                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium mb-1">Name</label>
                    <input
                      type="text"
                      value={newToolName}
                      onChange={(e) => setNewToolName(e.target.value)}
                      placeholder="My Tool"
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1">Description</label>
                    <textarea
                      value={newToolDesc}
                      onChange={(e) => setNewToolDesc(e.target.value)}
                      placeholder="What does this tool do?"
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded"
                      rows={3}
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Parameters (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={newToolParams}
                      onChange={(e) => setNewToolParams(e.target.value)}
                      placeholder="param1, param2"
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded"
                    />
                  </div>

                  <button
                    onClick={createTool}
                    disabled={!newToolName || !newToolDesc}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded"
                  >
                    Create Tool
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Execution Area */}
          <div className="lg:col-span-2 space-y-4">
            {/* Task Input */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Task</h3>

              <div className="space-y-4">
                <textarea
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="What should the agent do?

Example: Calculate the weather in New York and convert the temperature to Celsius"
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-600 rounded-lg"
                  rows={4}
                />

                <div className="flex items-center gap-4">
                  <button
                    onClick={executeAgent}
                    disabled={!task.trim() || activeTools.length === 0 || executing}
                    className="flex-1 px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 rounded-lg font-medium flex items-center justify-center gap-2"
                  >
                    <Play className="w-5 h-5" />
                    {executing ? 'Executing...' : 'Execute Agent'}
                  </button>

                  <div className="text-sm text-gray-400">
                    {activeTools.length} tool{activeTools.length !== 1 ? 's' : ''} selected
                  </div>
                </div>
              </div>
            </div>

            {/* Execution Log */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Execution Log</h3>

              <div className="bg-gray-900 rounded-lg border border-gray-600 p-4 min-h-[500px]">
                {executionLog.length === 0 ? (
                  <div className="flex items-center justify-center h-[400px] text-gray-500">
                    Execute a task to see agent reasoning...
                  </div>
                ) : (
                  <div className="space-y-4">
                    {executionLog.map((step, idx) => (
                      <div
                        key={idx}
                        className={`p-4 rounded-lg border-l-4 ${
                          step.type === 'thought'
                            ? 'bg-blue-900/20 border-blue-500'
                            : step.type === 'action'
                            ? 'bg-amber-900/20 border-amber-500'
                            : step.type === 'observation'
                            ? 'bg-purple-900/20 border-purple-500'
                            : 'bg-green-900/20 border-green-500'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          {getStepIcon(step.type)}
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-xs font-semibold uppercase tracking-wide">
                                Step {step.step}: {step.type}
                              </span>
                              {step.tool && (
                                <span className="px-2 py-1 bg-gray-700 rounded text-xs">
                                  {allTools.find(t => t.id === step.tool)?.name}
                                </span>
                              )}
                            </div>
                            <div className="text-sm whitespace-pre-wrap">{step.content}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
