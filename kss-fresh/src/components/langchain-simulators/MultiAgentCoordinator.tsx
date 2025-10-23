'use client'

import React, { useState } from 'react'
import { Users, Play, MessageSquare, CheckCircle, Loader } from 'lucide-react'

interface Agent {
  id: string
  name: string
  role: string
  expertise: string
  color: string
  icon: string
}

interface Task {
  id: string
  description: string
  assignedTo: string | null
  status: 'pending' | 'in-progress' | 'completed'
  result?: string
}

interface Message {
  from: string
  to: string
  content: string
  timestamp: number
  type: 'task' | 'query' | 'response' | 'coordination'
}

const AGENT_TEMPLATES: Agent[] = [
  {
    id: 'researcher',
    name: 'Researcher',
    role: 'Research Specialist',
    expertise: 'Information gathering and analysis',
    color: '#3b82f6',
    icon: '🔍'
  },
  {
    id: 'writer',
    name: 'Writer',
    role: 'Content Creator',
    expertise: 'Writing and content generation',
    color: '#10b981',
    icon: '✍️'
  },
  {
    id: 'analyst',
    name: 'Analyst',
    role: 'Data Analyst',
    expertise: 'Data analysis and insights',
    color: '#8b5cf6',
    icon: '📊'
  },
  {
    id: 'reviewer',
    name: 'Reviewer',
    role: 'Quality Assurance',
    expertise: 'Review and quality control',
    color: '#f59e0b',
    icon: '✅'
  },
  {
    id: 'coordinator',
    name: 'Coordinator',
    role: 'Team Lead',
    expertise: 'Task coordination and delegation',
    color: '#ef4444',
    icon: '👔'
  }
]

export default function MultiAgentCoordinator() {
  const [agents, setAgents] = useState<Agent[]>([AGENT_TEMPLATES[4]]) // Start with coordinator
  const [tasks, setTasks] = useState<Task[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [projectGoal, setProjectGoal] = useState('')
  const [executing, setExecuting] = useState(false)
  const [progress, setProgress] = useState(0)

  const addAgent = (template: Agent) => {
    if (agents.find(a => a.id === template.id)) return
    setAgents([...agents, template])
  }

  const removeAgent = (id: string) => {
    if (id === 'coordinator') return // Can't remove coordinator
    setAgents(agents.filter(a => a.id !== id))
  }

  const executeProject = async () => {
    if (!projectGoal.trim() || agents.length < 2) return

    setExecuting(true)
    setProgress(0)
    setTasks([])
    setMessages([])

    const newTasks: Task[] = []
    const newMessages: Message[] = []

    // Step 1: Coordinator breaks down the project
    await delay(500)
    addMessage(newMessages, {
      from: 'coordinator',
      to: 'all',
      content: `Analyzing project goal: "${projectGoal}"`,
      timestamp: Date.now(),
      type: 'coordination'
    })

    await delay(800)

    // Create tasks based on available agents
    const taskDescriptions = generateTasks(projectGoal, agents)
    taskDescriptions.forEach((desc, idx) => {
      const task: Task = {
        id: `task-${idx}`,
        description: desc.description,
        assignedTo: desc.agent,
        status: 'pending'
      }
      newTasks.push(task)
    })

    setTasks([...newTasks])
    setProgress(20)

    addMessage(newMessages, {
      from: 'coordinator',
      to: 'all',
      content: `Created ${newTasks.length} tasks. Delegating to team members...`,
      timestamp: Date.now(),
      type: 'coordination'
    })

    await delay(600)

    // Step 2: Execute tasks
    for (let i = 0; i < newTasks.length; i++) {
      const task = newTasks[i]
      const agent = agents.find(a => a.id === task.assignedTo)
      if (!agent) continue

      // Update task status
      task.status = 'in-progress'
      setTasks([...newTasks])

      addMessage(newMessages, {
        from: 'coordinator',
        to: agent.id,
        content: `Task assigned: ${task.description}`,
        timestamp: Date.now(),
        type: 'task'
      })

      await delay(800)

      // Agent acknowledges
      addMessage(newMessages, {
        from: agent.id,
        to: 'coordinator',
        content: `Task received. Working on it...`,
        timestamp: Date.now(),
        type: 'response'
      })

      await delay(1200)

      // Complete task
      task.status = 'completed'
      task.result = generateTaskResult(agent, task.description)
      setTasks([...newTasks])

      addMessage(newMessages, {
        from: agent.id,
        to: 'coordinator',
        content: `Task completed: ${task.result}`,
        timestamp: Date.now(),
        type: 'response'
      })

      setProgress(20 + ((i + 1) / newTasks.length) * 60)
      await delay(600)
    }

    // Step 3: Review and consolidation
    const reviewer = agents.find(a => a.id === 'reviewer')
    if (reviewer) {
      addMessage(newMessages, {
        from: 'coordinator',
        to: 'reviewer',
        content: 'Please review all completed tasks',
        timestamp: Date.now(),
        type: 'task'
      })

      await delay(1000)

      addMessage(newMessages, {
        from: 'reviewer',
        to: 'coordinator',
        content: 'Review complete. All tasks meet quality standards.',
        timestamp: Date.now(),
        type: 'response'
      })

      setProgress(90)
      await delay(600)
    }

    // Final aggregation
    addMessage(newMessages, {
      from: 'coordinator',
      to: 'all',
      content: `Project completed successfully! All ${newTasks.length} tasks finished.`,
      timestamp: Date.now(),
      type: 'coordination'
    })

    setProgress(100)
    setExecuting(false)
  }

  const generateTasks = (goal: string, availableAgents: Agent[]): Array<{description: string, agent: string}> => {
    const tasks = []

    if (availableAgents.find(a => a.id === 'researcher')) {
      tasks.push({
        description: `Research and gather information about: ${goal}`,
        agent: 'researcher'
      })
    }

    if (availableAgents.find(a => a.id === 'analyst')) {
      tasks.push({
        description: `Analyze data and provide insights for: ${goal}`,
        agent: 'analyst'
      })
    }

    if (availableAgents.find(a => a.id === 'writer')) {
      tasks.push({
        description: `Write comprehensive content about: ${goal}`,
        agent: 'writer'
      })
    }

    if (availableAgents.find(a => a.id === 'reviewer')) {
      tasks.push({
        description: `Review and ensure quality of deliverables`,
        agent: 'reviewer'
      })
    }

    return tasks
  }

  const generateTaskResult = (agent: Agent, taskDesc: string): string => {
    const results: Record<string, string> = {
      researcher: 'Found 15 relevant sources and compiled key findings',
      analyst: 'Identified 3 major trends and 5 actionable insights',
      writer: 'Created 1,500-word comprehensive article with examples',
      reviewer: 'Verified accuracy, checked formatting, approved for delivery'
    }

    return results[agent.id] || 'Task completed successfully'
  }

  const addMessage = (messageList: Message[], msg: Message) => {
    messageList.push(msg)
    setMessages([...messageList])
  }

  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

  const getAgentById = (id: string) => agents.find(a => a.id === id)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            👥 Multi-Agent Coordinator
          </h1>
          <p className="text-gray-300 text-lg">
            Orchestrate multiple AI agents working together to complete complex projects.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Agent Management */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Users className="w-5 h-5" />
                Agents ({agents.length})
              </h3>

              {/* Active Agents */}
              <div className="space-y-2 mb-4">
                {agents.map(agent => (
                  <div
                    key={agent.id}
                    className="p-3 rounded-lg border-2"
                    style={{ borderColor: agent.color }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">{agent.icon}</span>
                        <div>
                          <div className="font-medium text-sm">{agent.name}</div>
                          <div className="text-xs text-gray-400">{agent.role}</div>
                        </div>
                      </div>
                      {agent.id !== 'coordinator' && (
                        <button
                          onClick={() => removeAgent(agent.id)}
                          className="text-red-400 hover:text-red-300 text-xs"
                        >
                          Remove
                        </button>
                      )}
                    </div>
                    <div className="text-xs text-gray-400">{agent.expertise}</div>
                  </div>
                ))}
              </div>

              {/* Available Agents */}
              <div>
                <div className="text-sm font-semibold mb-2 text-gray-400">ADD AGENTS</div>
                <div className="space-y-2">
                  {AGENT_TEMPLATES.filter(t => !agents.find(a => a.id === t.id)).map(template => (
                    <button
                      key={template.id}
                      onClick={() => addAgent(template)}
                      className="w-full px-3 py-2 bg-gray-700/50 hover:bg-gray-600 border border-gray-600 rounded-lg text-left flex items-center gap-2"
                    >
                      <span className="text-xl">{template.icon}</span>
                      <div className="flex-1">
                        <div className="text-sm font-medium">{template.name}</div>
                        <div className="text-xs text-gray-400">{template.role}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Project Goal */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Project Goal</h3>

              <textarea
                value={projectGoal}
                onChange={(e) => setProjectGoal(e.target.value)}
                placeholder="Describe your project...

Example: Create a comprehensive market analysis report for the AI industry"
                className="w-full px-4 py-3 bg-gray-900 border border-gray-600 rounded-lg text-sm"
                rows={5}
              />

              <button
                onClick={executeProject}
                disabled={!projectGoal.trim() || agents.length < 2 || executing}
                className="w-full mt-4 px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 rounded-lg font-medium flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                {executing ? `Executing... ${progress}%` : 'Execute Project'}
              </button>
            </div>
          </div>

          {/* Execution Area */}
          <div className="lg:col-span-2 space-y-4">
            {/* Tasks */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Tasks</h3>

              <div className="space-y-2">
                {tasks.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No tasks yet. Define a project goal and execute.
                  </div>
                ) : (
                  tasks.map(task => {
                    const agent = getAgentById(task.assignedTo || '')
                    return (
                      <div
                        key={task.id}
                        className="p-4 bg-gray-900 border border-gray-600 rounded-lg"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1">
                            <div className="text-sm font-medium mb-1">{task.description}</div>
                            {task.result && (
                              <div className="text-xs text-gray-400 mt-2">
                                Result: {task.result}
                              </div>
                            )}
                          </div>
                          <div className="ml-4">
                            {task.status === 'pending' && (
                              <span className="px-3 py-1 bg-gray-700 text-gray-300 rounded-full text-xs">
                                Pending
                              </span>
                            )}
                            {task.status === 'in-progress' && (
                              <span className="px-3 py-1 bg-blue-600 text-white rounded-full text-xs flex items-center gap-1">
                                <Loader className="w-3 h-3 animate-spin" />
                                In Progress
                              </span>
                            )}
                            {task.status === 'completed' && (
                              <span className="px-3 py-1 bg-green-600 text-white rounded-full text-xs flex items-center gap-1">
                                <CheckCircle className="w-3 h-3" />
                                Completed
                              </span>
                            )}
                          </div>
                        </div>
                        {agent && (
                          <div className="flex items-center gap-2 mt-2">
                            <span className="text-lg">{agent.icon}</span>
                            <span className="text-xs" style={{ color: agent.color }}>
                              {agent.name}
                            </span>
                          </div>
                        )}
                      </div>
                    )
                  })
                )}
              </div>
            </div>

            {/* Communication Log */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <MessageSquare className="w-5 h-5" />
                Communication Log
              </h3>

              <div className="bg-gray-900 rounded-lg border border-gray-600 p-4 max-h-[500px] overflow-y-auto">
                {messages.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    Agent communications will appear here...
                  </div>
                ) : (
                  <div className="space-y-3">
                    {messages.map((msg, idx) => {
                      const fromAgent = getAgentById(msg.from)
                      const toAgent = msg.to === 'all' ? null : getAgentById(msg.to)

                      return (
                        <div
                          key={idx}
                          className={`p-3 rounded-lg border ${
                            msg.type === 'coordination'
                              ? 'bg-amber-900/20 border-amber-700'
                              : msg.type === 'task'
                              ? 'bg-blue-900/20 border-blue-700'
                              : 'bg-gray-800 border-gray-700'
                          }`}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            {fromAgent && (
                              <>
                                <span className="text-lg">{fromAgent.icon}</span>
                                <span className="font-medium text-sm" style={{ color: fromAgent.color }}>
                                  {fromAgent.name}
                                </span>
                              </>
                            )}
                            <span className="text-gray-500 text-xs">→</span>
                            {toAgent ? (
                              <>
                                <span className="text-lg">{toAgent.icon}</span>
                                <span className="font-medium text-sm" style={{ color: toAgent.color }}>
                                  {toAgent.name}
                                </span>
                              </>
                            ) : (
                              <span className="text-sm text-gray-400">All Agents</span>
                            )}
                          </div>
                          <div className="text-sm text-gray-300">{msg.content}</div>
                          <div className="text-xs text-gray-500 mt-1">
                            {new Date(msg.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Example Projects */}
            <div className="bg-blue-900/20 backdrop-blur border border-blue-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-3 text-blue-400">💡 Example Projects</h3>

              <div className="space-y-2 text-sm">
                <button
                  onClick={() => setProjectGoal('Create a comprehensive market analysis report for the AI industry')}
                  className="w-full text-left px-3 py-2 bg-gray-700/50 hover:bg-gray-600 rounded text-gray-300"
                >
                  Market Analysis Report
                </button>
                <button
                  onClick={() => setProjectGoal('Develop a content strategy for launching a new SaaS product')}
                  className="w-full text-left px-3 py-2 bg-gray-700/50 hover:bg-gray-600 rounded text-gray-300"
                >
                  Content Strategy
                </button>
                <button
                  onClick={() => setProjectGoal('Research and write a technical blog post about LangChain best practices')}
                  className="w-full text-left px-3 py-2 bg-gray-700/50 hover:bg-gray-600 rounded text-gray-300"
                >
                  Technical Blog Post
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
