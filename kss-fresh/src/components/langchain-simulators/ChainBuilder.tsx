'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Play, Download, Trash2, Plus, Code, Zap, HelpCircle, X, Maximize, Minimize, Save, Upload, Undo, Redo } from 'lucide-react'

interface ChainComponent {
  id: string
  type: 'llm' | 'prompt' | 'parser' | 'retriever' | 'transform' |
        'vectordb' | 'memory' | 'agent' | 'tool' | 'embedding' |
        'chat' | 'search' | 'splitter' | 'conditional' | 'output'
  label: string
  config: Record<string, any>
  position: { x: number; y: number }
}

interface Connection {
  id: string
  from: string
  to: string
}

interface HistoryState {
  components: ChainComponent[]
  connections: Connection[]
}

const COMPONENT_TEMPLATES = {
  llm: {
    type: 'llm',
    label: 'LLM',
    config: { model: 'gpt-3.5-turbo', temperature: 0.7 },
    color: '#f59e0b',
    icon: '🤖'
  },
  prompt: {
    type: 'prompt',
    label: 'Prompt Template',
    config: { template: 'Answer the question: {question}' },
    color: '#3b82f6',
    icon: '📝'
  },
  parser: {
    type: 'parser',
    label: 'Output Parser',
    config: { format: 'json' },
    color: '#10b981',
    icon: '🔧'
  },
  retriever: {
    type: 'retriever',
    label: 'Retriever',
    config: { k: 3, source: 'vector_store' },
    color: '#8b5cf6',
    icon: '🔍'
  },
  transform: {
    type: 'transform',
    label: 'Transform',
    config: { operation: 'lowercase' },
    color: '#ec4899',
    icon: '⚡'
  },
  vectordb: {
    type: 'vectordb',
    label: 'Vector Database',
    config: { database: 'pinecone', index: 'default', namespace: '' },
    color: '#a855f7',
    icon: '🗄️'
  },
  memory: {
    type: 'memory',
    label: 'Memory',
    config: { type: 'buffer', maxTokens: 2000 },
    color: '#06b6d4',
    icon: '🧠'
  },
  agent: {
    type: 'agent',
    label: 'Agent',
    config: { type: 'react', maxIterations: 5, verbose: true },
    color: '#f97316',
    icon: '🤖'
  },
  tool: {
    type: 'tool',
    label: 'Tool',
    config: { name: 'calculator', description: 'Performs math calculations' },
    color: '#84cc16',
    icon: '🛠️'
  },
  embedding: {
    type: 'embedding',
    label: 'Embedding',
    config: { model: 'text-embedding-ada-002', dimensions: 1536 },
    color: '#14b8a6',
    icon: '📊'
  },
  chat: {
    type: 'chat',
    label: 'Chat Model',
    config: { model: 'gpt-3.5-turbo', streaming: false },
    color: '#f59e0b',
    icon: '💬'
  },
  search: {
    type: 'search',
    label: 'Search',
    config: { engine: 'google', maxResults: 5 },
    color: '#3b82f6',
    icon: '🔎'
  },
  splitter: {
    type: 'splitter',
    label: 'Text Splitter',
    config: { chunkSize: 1000, chunkOverlap: 200 },
    color: '#8b5cf6',
    icon: '✂️'
  },
  conditional: {
    type: 'conditional',
    label: 'Conditional',
    config: { condition: 'length > 100', truePath: '', falsePath: '' },
    color: '#ef4444',
    icon: '🔀'
  },
  output: {
    type: 'output',
    label: 'Output',
    config: { format: 'text', destination: 'console' },
    color: '#10b981',
    icon: '📤'
  }
}

export default function ChainBuilder() {
  const [components, setComponents] = useState<ChainComponent[]>([])
  const [connections, setConnections] = useState<Connection[]>([])
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null)
  const [selectedConnection, setSelectedConnection] = useState<string | null>(null)
  const [dragging, setDragging] = useState<string | null>(null)
  const [connectingFrom, setConnectingFrom] = useState<string | null>(null)
  const [connectionLine, setConnectionLine] = useState<{ x: number, y: number } | null>(null)
  const [testInput, setTestInput] = useState('')
  const [testOutput, setTestOutput] = useState('')
  const [executing, setExecuting] = useState(false)
  const [showHelp, setShowHelp] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Undo/Redo history
  const [history, setHistory] = useState<HistoryState[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)

  const canvasRef = useRef<HTMLDivElement>(null)

  const addComponent = (type: keyof typeof COMPONENT_TEMPLATES) => {
    const template = COMPONENT_TEMPLATES[type]
    const newComponent: ChainComponent = {
      id: `${type}-${Date.now()}`,
      type: template.type as any,
      label: template.label,
      config: { ...template.config },
      position: { x: 150 + components.length * 40, y: 100 + (components.length % 3) * 120 }
    }
    setComponents([...components, newComponent])
  }

  const deleteComponent = (id: string) => {
    setComponents(components.filter(c => c.id !== id))
    setConnections(connections.filter(conn => conn.from !== id && conn.to !== id))
    if (selectedComponent === id) setSelectedComponent(null)
  }

  const deleteConnection = (connId: string) => {
    setConnections(connections.filter(c => c.id !== connId))
    setSelectedConnection(null)
  }

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen()
      setIsFullscreen(true)
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
        setIsFullscreen(false)
      }
    }
  }

  // Listen for fullscreen changes (ESC key)
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  // Save current state to history
  const saveToHistory = () => {
    const newState: HistoryState = {
      components: JSON.parse(JSON.stringify(components)),
      connections: JSON.parse(JSON.stringify(connections))
    }

    // Remove any future history if we're not at the end
    const newHistory = history.slice(0, historyIndex + 1)
    newHistory.push(newState)

    // Limit history to 50 states
    if (newHistory.length > 50) {
      newHistory.shift()
    } else {
      setHistoryIndex(historyIndex + 1)
    }

    setHistory(newHistory)
  }

  // Undo function
  const undo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1
      setHistoryIndex(newIndex)
      const state = history[newIndex]
      setComponents(JSON.parse(JSON.stringify(state.components)))
      setConnections(JSON.parse(JSON.stringify(state.connections)))
    }
  }

  // Redo function
  const redo = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1
      setHistoryIndex(newIndex)
      const state = history[newIndex]
      setComponents(JSON.parse(JSON.stringify(state.components)))
      setConnections(JSON.parse(JSON.stringify(state.connections)))
    }
  }

  // Keyboard shortcuts for Undo/Redo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        undo()
      } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault()
        redo()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [historyIndex, history])

  // Save to history whenever components or connections change
  useEffect(() => {
    // Skip initial render and when restoring from history
    if (components.length === 0 && connections.length === 0 && history.length === 0) {
      // Save initial empty state
      saveToHistory()
      return
    }

    // Don't save if we're currently at this exact state (prevents duplicate saves)
    if (historyIndex >= 0 && historyIndex < history.length) {
      const currentState = history[historyIndex]
      if (
        JSON.stringify(currentState.components) === JSON.stringify(components) &&
        JSON.stringify(currentState.connections) === JSON.stringify(connections)
      ) {
        return
      }
    }

    saveToHistory()
  }, [components, connections])

  const handleOutputPortClick = (compId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (connectingFrom === compId) {
      // Cancel connection
      setConnectingFrom(null)
      setConnectionLine(null)
    } else {
      // Start connection
      setConnectingFrom(compId)
    }
  }

  const handleInputPortClick = (compId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (connectingFrom && connectingFrom !== compId) {
      // Complete connection
      const newConnection: Connection = {
        id: `conn-${Date.now()}`,
        from: connectingFrom,
        to: compId
      }
      setConnections([...connections, newConnection])
      setConnectingFrom(null)
      setConnectionLine(null)
    }
  }

  const handleMouseDown = (id: string, e: React.MouseEvent) => {
    if (e.button !== 0) return // Only left click
    e.stopPropagation()
    setDragging(id)
    setSelectedComponent(id)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragging && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      setComponents(components.map(c =>
        c.id === dragging
          ? { ...c, position: { x: Math.max(0, x - 75), y: Math.max(0, y - 35) } }
          : c
      ))
    }

    // Update connection line when connecting
    if (connectingFrom && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect()
      setConnectionLine({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      })
    }
  }

  const handleMouseUp = () => {
    setDragging(null)
  }

  const handleCanvasClick = () => {
    setSelectedComponent(null)
    setSelectedConnection(null)
  }

  const handleConnectionClick = (connId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSelectedConnection(connId)
    setSelectedComponent(null)
  }

  const executeChain = async () => {
    setExecuting(true)
    let currentData = testInput

    const ordered = getExecutionOrder()
    let output = `🔗 Chain Execution Log\n${'='.repeat(60)}\n\n`

    for (const compId of ordered) {
      const comp = components.find(c => c.id === compId)
      if (!comp) continue

      await new Promise(resolve => setTimeout(resolve, 500))

      output += `📦 ${comp.label} (${comp.type})\n`
      output += `   Input: ${currentData.substring(0, 50)}${currentData.length > 50 ? '...' : ''}\n`

      switch (comp.type) {
        case 'prompt':
          currentData = comp.config.template.replace('{question}', currentData)
          output += `   Template: ${comp.config.template}\n`
          break
        case 'llm':
          currentData = `[LLM Response] Based on "${currentData}", here's the answer: This is a simulated response from ${comp.config.model}.`
          output += `   Model: ${comp.config.model}\n`
          output += `   Temperature: ${comp.config.temperature}\n`
          break
        case 'parser':
          currentData = JSON.stringify({ parsed: currentData, format: comp.config.format })
          output += `   Format: ${comp.config.format}\n`
          break
        case 'retriever':
          currentData = `[Retrieved Docs] Top ${comp.config.k} documents related to "${currentData}"`
          output += `   Top K: ${comp.config.k}\n`
          break
        case 'transform':
          currentData = comp.config.operation === 'lowercase' ? currentData.toLowerCase() : currentData.toUpperCase()
          output += `   Operation: ${comp.config.operation}\n`
          break
      }

      output += `   Output: ${currentData.substring(0, 50)}${currentData.length > 50 ? '...' : ''}\n`
      output += `${'─'.repeat(60)}\n\n`
    }

    output += `✅ Chain Completed!\n`
    output += `\nFinal Output:\n${currentData}`

    setTestOutput(output)
    setExecuting(false)
  }

  const getExecutionOrder = (): string[] => {
    const order: string[] = []
    const visited = new Set<string>()

    const visit = (id: string) => {
      if (visited.has(id)) return
      visited.add(id)

      const incoming = connections.filter(c => c.to === id)
      incoming.forEach(conn => visit(conn.from))

      order.push(id)
    }

    components.forEach(c => visit(c.id))
    return order
  }

  const exportCode = () => {
    // Import statements based on components
    const imports = new Set<string>()
    imports.add('from langchain.chat_models import ChatOpenAI')
    imports.add('from langchain.prompts import PromptTemplate')
    imports.add('from langchain.chains import LLMChain')

    // Add component-specific imports
    components.forEach(comp => {
      switch (comp.type) {
        case 'vectordb':
          if (comp.config.database === 'pinecone') imports.add('from langchain.vectorstores import Pinecone')
          if (comp.config.database === 'weaviate') imports.add('from langchain.vectorstores import Weaviate')
          if (comp.config.database === 'chroma') imports.add('from langchain.vectorstores import Chroma')
          if (comp.config.database === 'qdrant') imports.add('from langchain.vectorstores import Qdrant')
          if (comp.config.database === 'milvus') imports.add('from langchain.vectorstores import Milvus')
          imports.add('from langchain.embeddings import OpenAIEmbeddings')
          break
        case 'memory':
          if (comp.config.type === 'buffer') imports.add('from langchain.memory import ConversationBufferMemory')
          if (comp.config.type === 'summary') imports.add('from langchain.memory import ConversationSummaryMemory')
          if (comp.config.type === 'vector_store') imports.add('from langchain.memory import VectorStoreRetrieverMemory')
          if (comp.config.type === 'entity') imports.add('from langchain.memory import ConversationEntityMemory')
          break
        case 'agent':
          imports.add('from langchain.agents import initialize_agent, AgentType')
          imports.add('from langchain.agents import Tool')
          break
        case 'tool':
          if (comp.config.name === 'calculator') imports.add('from langchain.tools import Tool')
          if (comp.config.name === 'search') imports.add('from langchain.tools import DuckDuckGoSearchRun')
          if (comp.config.name === 'wikipedia') imports.add('from langchain.tools import WikipediaQueryRun')
          break
        case 'embedding':
          imports.add('from langchain.embeddings import OpenAIEmbeddings')
          break
        case 'search':
          if (comp.config.engine === 'google') imports.add('from langchain.utilities import GoogleSearchAPIWrapper')
          if (comp.config.engine === 'duckduckgo') imports.add('from langchain.tools import DuckDuckGoSearchRun')
          break
        case 'splitter':
          imports.add('from langchain.text_splitter import RecursiveCharacterTextSplitter')
          break
        case 'retriever':
          imports.add('from langchain.vectorstores import FAISS')
          imports.add('from langchain.embeddings import OpenAIEmbeddings')
          break
      }
    })

    let code = Array.from(imports).join('\n') + '\n\nimport os\n\n'
    code += `# Set API keys\nos.environ["OPENAI_API_KEY"] = "your-api-key-here"\n\n`

    // Generate component initialization code
    components.forEach(comp => {
      switch (comp.type) {
        case 'llm':
          code += `# LLM Component\n`
          code += `llm = ChatOpenAI(\n`
          code += `    model="${comp.config.model}",\n`
          code += `    temperature=${comp.config.temperature}\n`
          code += `)\n\n`
          break

        case 'prompt':
          code += `# Prompt Template\n`
          code += `prompt = PromptTemplate.from_template(\n`
          code += `    "${comp.config.template}"\n`
          code += `)\n\n`
          break

        case 'parser':
          code += `# Output Parser (${comp.config.format})\n`
          code += `# Parse output to ${comp.config.format} format\n\n`
          break

        case 'retriever':
          code += `# Retriever Component\n`
          code += `embeddings = OpenAIEmbeddings()\n`
          code += `vectorstore = FAISS.from_texts(\n`
          code += `    ["Sample document 1", "Sample document 2"],\n`
          code += `    embeddings\n`
          code += `)\n`
          code += `retriever = vectorstore.as_retriever(search_kwargs={"k": ${comp.config.topK}})\n\n`
          break

        case 'transform':
          code += `# Transform Component\n`
          code += `# Custom transformation: ${comp.config.operation}\n\n`
          break

        case 'vectordb':
          code += `# Vector Database (${comp.config.database})\n`
          code += `embeddings = OpenAIEmbeddings()\n`
          if (comp.config.database === 'pinecone') {
            code += `import pinecone\n`
            code += `pinecone.init(api_key="your-pinecone-api-key", environment="your-env")\n`
            code += `vectorstore = Pinecone.from_texts(\n`
            code += `    ["Sample doc 1", "Sample doc 2"],\n`
            code += `    embeddings,\n`
            code += `    index_name="${comp.config.index}"\n`
            code += `)\n\n`
          } else if (comp.config.database === 'chroma') {
            code += `vectorstore = Chroma.from_texts(\n`
            code += `    ["Sample doc 1", "Sample doc 2"],\n`
            code += `    embeddings,\n`
            code += `    collection_name="${comp.config.index}"\n`
            code += `)\n\n`
          } else {
            code += `# Initialize ${comp.config.database} vectorstore\n`
            code += `# See LangChain docs for specific setup\n\n`
          }
          break

        case 'memory':
          code += `# Memory (${comp.config.type})\n`
          if (comp.config.type === 'buffer') {
            code += `memory = ConversationBufferMemory(\n`
            code += `    max_token_limit=${comp.config.maxTokens}\n`
            code += `)\n\n`
          } else if (comp.config.type === 'summary') {
            code += `memory = ConversationSummaryMemory(\n`
            code += `    llm=llm,\n`
            code += `    max_token_limit=${comp.config.maxTokens}\n`
            code += `)\n\n`
          } else if (comp.config.type === 'vector_store') {
            code += `memory = VectorStoreRetrieverMemory(\n`
            code += `    retriever=retriever\n`
            code += `)\n\n`
          } else if (comp.config.type === 'entity') {
            code += `memory = ConversationEntityMemory(\n`
            code += `    llm=llm\n`
            code += `)\n\n`
          }
          break

        case 'agent':
          code += `# Agent (${comp.config.type})\n`
          code += `tools = []  # Add tools here\n`
          if (comp.config.type === 'react') {
            code += `agent = initialize_agent(\n`
            code += `    tools,\n`
            code += `    llm,\n`
            code += `    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,\n`
            code += `    max_iterations=${comp.config.maxIterations},\n`
            code += `    verbose=True\n`
            code += `)\n\n`
          } else if (comp.config.type === 'openai_functions') {
            code += `agent = initialize_agent(\n`
            code += `    tools,\n`
            code += `    llm,\n`
            code += `    agent=AgentType.OPENAI_FUNCTIONS,\n`
            code += `    max_iterations=${comp.config.maxIterations},\n`
            code += `    verbose=True\n`
            code += `)\n\n`
          } else {
            code += `# Initialize ${comp.config.type} agent\n\n`
          }
          break

        case 'tool':
          code += `# Tool (${comp.config.name})\n`
          if (comp.config.name === 'calculator') {
            code += `def calculator(query: str) -> str:\n`
            code += `    return str(eval(query))\n\n`
            code += `calculator_tool = Tool(\n`
            code += `    name="Calculator",\n`
            code += `    func=calculator,\n`
            code += `    description="${comp.config.description}"\n`
            code += `)\n\n`
          } else if (comp.config.name === 'search') {
            code += `search = DuckDuckGoSearchRun()\n`
            code += `search_tool = Tool(\n`
            code += `    name="Search",\n`
            code += `    func=search.run,\n`
            code += `    description="${comp.config.description}"\n`
            code += `)\n\n`
          } else if (comp.config.name === 'wikipedia') {
            code += `wiki = WikipediaQueryRun()\n`
            code += `wiki_tool = Tool(\n`
            code += `    name="Wikipedia",\n`
            code += `    func=wiki.run,\n`
            code += `    description="${comp.config.description}"\n`
            code += `)\n\n`
          } else {
            code += `# Custom tool: ${comp.config.name}\n`
            code += `# Description: ${comp.config.description}\n\n`
          }
          break

        case 'embedding':
          code += `# Embedding Model (${comp.config.model})\n`
          code += `embeddings = OpenAIEmbeddings(\n`
          code += `    model="${comp.config.model}"\n`
          code += `)\n\n`
          break

        case 'chat':
          code += `# Chat Model (${comp.config.model})\n`
          code += `chat_model = ChatOpenAI(\n`
          code += `    model="${comp.config.model}",\n`
          code += `    temperature=0.7\n`
          code += `)\n\n`
          break

        case 'search':
          code += `# Search Engine (${comp.config.engine})\n`
          if (comp.config.engine === 'google') {
            code += `search = GoogleSearchAPIWrapper()\n`
            code += `results = search.run("query", num_results=${comp.config.maxResults})\n\n`
          } else if (comp.config.engine === 'duckduckgo') {
            code += `search = DuckDuckGoSearchRun()\n`
            code += `results = search.run("query")[:${comp.config.maxResults}]\n\n`
          }
          break

        case 'splitter':
          code += `# Text Splitter\n`
          code += `text_splitter = RecursiveCharacterTextSplitter(\n`
          code += `    chunk_size=${comp.config.chunkSize},\n`
          code += `    chunk_overlap=${comp.config.chunkOverlap}\n`
          code += `)\n`
          code += `chunks = text_splitter.split_text("Your long text here")\n\n`
          break

        case 'conditional':
          code += `# Conditional Logic\n`
          code += `# Condition: ${comp.config.condition}\n`
          code += `if ${comp.config.condition.replace('if ', '')}:\n`
          code += `    # Branch A\n`
          code += `    pass\n`
          code += `else:\n`
          code += `    # Branch B\n`
          code += `    pass\n\n`
          break

        case 'output':
          code += `# Output Format (${comp.config.format})\n`
          if (comp.config.format === 'json') {
            code += `import json\n`
            code += `output = json.dumps(result, indent=2)\n\n`
          } else if (comp.config.format === 'markdown') {
            code += `# Format output as Markdown\n`
            code += `output = f"## Result\\n\\n{result}"\n\n`
          } else if (comp.config.format === 'html') {
            code += `# Format output as HTML\n`
            code += `output = f"<div>{result}</div>"\n\n`
          } else {
            code += `output = str(result)\n\n`
          }
          break
      }
    })

    // Generate execution code
    code += `# Execute Chain\n`
    if (components.some(c => c.type === 'agent')) {
      code += `result = agent.run("Your question here")\n`
    } else if (components.some(c => c.type === 'llm') && components.some(c => c.type === 'prompt')) {
      code += `chain = LLMChain(llm=llm, prompt=prompt)\n`
      code += `result = chain.run(question="Your question here")\n`
    } else {
      code += `# Configure your chain based on components above\n`
      code += `# result = your_chain.run(...)\n`
    }

    code += `\nprint(result)\n`

    navigator.clipboard.writeText(code)
    alert('✅ Code copied to clipboard!')
  }

  const clearAll = () => {
    if (confirm('Clear all components and connections?')) {
      setComponents([])
      setConnections([])
      setSelectedComponent(null)
      setSelectedConnection(null)
      setTestOutput('')
    }
  }

  // Save workflow to localStorage
  const saveWorkflow = () => {
    const workflow = {
      components,
      connections,
      timestamp: new Date().toISOString(),
      version: '1.0'
    }
    localStorage.setItem('langchain-workflow', JSON.stringify(workflow))
    alert('✅ Workflow saved!')
  }

  // Load workflow from localStorage
  const loadWorkflow = () => {
    const saved = localStorage.getItem('langchain-workflow')
    if (!saved) {
      alert('❌ No saved workflow found')
      return
    }

    try {
      const workflow = JSON.parse(saved)
      setComponents(workflow.components)
      setConnections(workflow.connections)
      alert(`✅ Workflow loaded! (saved ${new Date(workflow.timestamp).toLocaleString()})`)
    } catch (error) {
      alert('❌ Failed to load workflow')
      console.error(error)
    }
  }

  // Auto-save every 30 seconds
  useEffect(() => {
    if (components.length === 0 && connections.length === 0) return

    const autoSaveInterval = setInterval(() => {
      const workflow = {
        components,
        connections,
        timestamp: new Date().toISOString(),
        version: '1.0'
      }
      localStorage.setItem('langchain-workflow-autosave', JSON.stringify(workflow))
      console.log('🔄 Auto-saved workflow')
    }, 30000) // 30 seconds

    return () => clearInterval(autoSaveInterval)
  }, [components, connections])

  const selectedComp = components.find(c => c.id === selectedComponent)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className={`container mx-auto ${isFullscreen ? 'px-2 py-2' : 'px-4 py-8'}`}>
        {/* Header - Hidden in fullscreen */}
        {!isFullscreen && (
          <div className="mb-8 flex items-start justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
                ⛓️ Chain Builder Pro
              </h1>
              <p className="text-gray-300 text-lg">
                Professional visual builder for LangChain pipelines
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowHelp(!showHelp)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2"
              >
                <HelpCircle className="w-5 h-5" />
                Help
              </button>
              <button
                onClick={toggleFullscreen}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2"
                title={isFullscreen ? "Exit Fullscreen (ESC)" : "Enter Fullscreen"}
              >
                {isFullscreen ? (
                  <>
                    <Minimize className="w-5 h-5" />
                    Exit
                  </>
                ) : (
                  <>
                    <Maximize className="w-5 h-5" />
                    Fullscreen
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Help Panel */}
        {showHelp && !isFullscreen && (
          <div className="mb-6 bg-blue-900/30 border border-blue-600 rounded-xl p-6">
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-xl font-bold text-blue-400">How to Use</h3>
              <button onClick={() => setShowHelp(false)} className="text-gray-400 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="font-semibold text-amber-400 mb-2">🎨 Add Components</h4>
                <p className="text-gray-300">Click buttons in the left palette to add components to the canvas.</p>
              </div>
              <div>
                <h4 className="font-semibold text-amber-400 mb-2">🔌 Connect Components</h4>
                <p className="text-gray-300">Click the <span className="text-green-400">green output port</span> (→), then click the <span className="text-blue-400">blue input port</span> (←) of another component.</p>
              </div>
              <div>
                <h4 className="font-semibold text-amber-400 mb-2">🎯 Move Components</h4>
                <p className="text-gray-300">Click and drag any component to reposition it on the canvas.</p>
              </div>
              <div>
                <h4 className="font-semibold text-amber-400 mb-2">🗑️ Delete</h4>
                <p className="text-gray-300">Select a component or connection, then use the delete button or press Delete key.</p>
              </div>
            </div>
          </div>
        )}

        {/* Fullscreen Floating Toolbar */}
        {isFullscreen && (
          <div className="fixed top-4 right-4 z-50 flex items-center gap-2 bg-gray-800/95 backdrop-blur border border-gray-700 rounded-lg p-2 shadow-2xl">
            <button
              onClick={() => setShowHelp(!showHelp)}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded flex items-center gap-2 text-sm"
            >
              <HelpCircle className="w-4 h-4" />
            </button>
            <button
              onClick={toggleFullscreen}
              className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2 text-sm"
              title="Exit Fullscreen (ESC)"
            >
              <Minimize className="w-4 h-4" />
              Exit
            </button>
          </div>
        )}

        <div className={`grid grid-cols-1 ${isFullscreen ? 'lg:grid-cols-12' : 'lg:grid-cols-4'} gap-${isFullscreen ? '2' : '6'}`}>
          {/* Component Palette */}
          <div className={`${isFullscreen ? 'lg:col-span-2' : 'lg:col-span-1'} space-y-4`}>
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Plus className="w-5 h-5" />
                Components
              </h3>

              <div className="space-y-2">
                {Object.entries(COMPONENT_TEMPLATES).map(([key, template]) => (
                  <button
                    key={key}
                    onClick={() => addComponent(key as keyof typeof COMPONENT_TEMPLATES)}
                    className="w-full px-4 py-3 bg-gray-700/50 hover:bg-gray-600 border border-gray-600 rounded-lg transition-all flex items-center gap-3 group"
                  >
                    <span className="text-2xl group-hover:scale-110 transition-transform">{template.icon}</span>
                    <span className="text-sm font-medium">{template.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Config Panel */}
            {selectedComp && (
              <div className="bg-gray-800/50 backdrop-blur border border-amber-500 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4 text-amber-400">⚙️ Configuration</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2 text-gray-400">Component Type</label>
                    <div className="px-3 py-2 bg-gray-700 rounded border border-gray-600 uppercase text-amber-400 font-semibold">
                      {selectedComp.type}
                    </div>
                  </div>

                  {selectedComp.type === 'llm' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium mb-2">Model</label>
                        <select
                          value={selectedComp.config.model}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, model: e.target.value } }
                                : c
                            ))
                          }}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                        >
                          <option>gpt-3.5-turbo</option>
                          <option>gpt-4</option>
                          <option>gpt-4-turbo</option>
                          <option>claude-3-opus</option>
                          <option>claude-3-sonnet</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Temperature: {selectedComp.config.temperature}</label>
                        <input
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={selectedComp.config.temperature}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, temperature: parseFloat(e.target.value) } }
                                : c
                            ))
                          }}
                          className="w-full accent-amber-500"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>Focused</span>
                          <span>Balanced</span>
                          <span>Creative</span>
                        </div>
                      </div>
                    </>
                  )}

                  {selectedComp.type === 'prompt' && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Template</label>
                      <textarea
                        value={selectedComp.config.template}
                        onChange={(e) => {
                          setComponents(components.map(c =>
                            c.id === selectedComp.id
                              ? { ...c, config: { ...c.config, template: e.target.value } }
                              : c
                          ))
                        }}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded font-mono text-sm"
                        rows={5}
                        placeholder="Use {variable} for placeholders"
                      />
                    </div>
                  )}

                  {selectedComp.type === 'retriever' && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Top K Documents: {selectedComp.config.k}</label>
                      <input
                        type="range"
                        min="1"
                        max="10"
                        value={selectedComp.config.k}
                        onChange={(e) => {
                          setComponents(components.map(c =>
                            c.id === selectedComp.id
                              ? { ...c, config: { ...c.config, k: parseInt(e.target.value) } }
                              : c
                          ))
                        }}
                        className="w-full accent-purple-500"
                      />
                    </div>
                  )}

                  {selectedComp.type === 'vectordb' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium mb-2">Database</label>
                        <select
                          value={selectedComp.config.database}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, database: e.target.value } }
                                : c
                            ))
                          }}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                        >
                          <option>pinecone</option>
                          <option>weaviate</option>
                          <option>chroma</option>
                          <option>qdrant</option>
                          <option>milvus</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Index Name</label>
                        <input
                          type="text"
                          value={selectedComp.config.index}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, index: e.target.value } }
                                : c
                            ))
                          }}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                        />
                      </div>
                    </>
                  )}

                  {selectedComp.type === 'memory' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium mb-2">Memory Type</label>
                        <select
                          value={selectedComp.config.type}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, type: e.target.value } }
                                : c
                            ))
                          }}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                        >
                          <option>buffer</option>
                          <option>summary</option>
                          <option>vector_store</option>
                          <option>entity</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Max Tokens: {selectedComp.config.maxTokens}</label>
                        <input
                          type="range"
                          min="500"
                          max="4000"
                          step="100"
                          value={selectedComp.config.maxTokens}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, maxTokens: parseInt(e.target.value) } }
                                : c
                            ))
                          }}
                          className="w-full accent-cyan-500"
                        />
                      </div>
                    </>
                  )}

                  {selectedComp.type === 'agent' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium mb-2">Agent Type</label>
                        <select
                          value={selectedComp.config.type}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, type: e.target.value } }
                                : c
                            ))
                          }}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                        >
                          <option>react</option>
                          <option>zero-shot</option>
                          <option>conversational</option>
                          <option>openai-functions</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Max Iterations: {selectedComp.config.maxIterations}</label>
                        <input
                          type="range"
                          min="1"
                          max="10"
                          value={selectedComp.config.maxIterations}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, maxIterations: parseInt(e.target.value) } }
                                : c
                            ))
                          }}
                          className="w-full accent-orange-500"
                        />
                      </div>
                    </>
                  )}

                  {selectedComp.type === 'tool' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium mb-2">Tool Name</label>
                        <select
                          value={selectedComp.config.name}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, name: e.target.value } }
                                : c
                            ))
                          }}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                        >
                          <option>calculator</option>
                          <option>search</option>
                          <option>wikipedia</option>
                          <option>weather</option>
                          <option>custom</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Description</label>
                        <input
                          type="text"
                          value={selectedComp.config.description}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, description: e.target.value } }
                                : c
                            ))
                          }}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                        />
                      </div>
                    </>
                  )}

                  {selectedComp.type === 'embedding' && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Model</label>
                      <select
                        value={selectedComp.config.model}
                        onChange={(e) => {
                          setComponents(components.map(c =>
                            c.id === selectedComp.id
                              ? { ...c, config: { ...c.config, model: e.target.value } }
                              : c
                          ))
                        }}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                      >
                        <option>text-embedding-ada-002</option>
                        <option>text-embedding-3-small</option>
                        <option>text-embedding-3-large</option>
                      </select>
                    </div>
                  )}

                  {selectedComp.type === 'chat' && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Chat Model</label>
                      <select
                        value={selectedComp.config.model}
                        onChange={(e) => {
                          setComponents(components.map(c =>
                            c.id === selectedComp.id
                              ? { ...c, config: { ...c.config, model: e.target.value } }
                              : c
                          ))
                        }}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                      >
                        <option>gpt-3.5-turbo</option>
                        <option>gpt-4</option>
                        <option>claude-3-sonnet</option>
                        <option>gemini-pro</option>
                      </select>
                    </div>
                  )}

                  {selectedComp.type === 'search' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium mb-2">Search Engine</label>
                        <select
                          value={selectedComp.config.engine}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, engine: e.target.value } }
                                : c
                            ))
                          }}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                        >
                          <option>google</option>
                          <option>bing</option>
                          <option>duckduckgo</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Max Results: {selectedComp.config.maxResults}</label>
                        <input
                          type="range"
                          min="1"
                          max="10"
                          value={selectedComp.config.maxResults}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, maxResults: parseInt(e.target.value) } }
                                : c
                            ))
                          }}
                          className="w-full accent-blue-500"
                        />
                      </div>
                    </>
                  )}

                  {selectedComp.type === 'splitter' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium mb-2">Chunk Size: {selectedComp.config.chunkSize}</label>
                        <input
                          type="range"
                          min="100"
                          max="2000"
                          step="100"
                          value={selectedComp.config.chunkSize}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, chunkSize: parseInt(e.target.value) } }
                                : c
                            ))
                          }}
                          className="w-full accent-purple-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Chunk Overlap: {selectedComp.config.chunkOverlap}</label>
                        <input
                          type="range"
                          min="0"
                          max="500"
                          step="50"
                          value={selectedComp.config.chunkOverlap}
                          onChange={(e) => {
                            setComponents(components.map(c =>
                              c.id === selectedComp.id
                                ? { ...c, config: { ...c.config, chunkOverlap: parseInt(e.target.value) } }
                                : c
                            ))
                          }}
                          className="w-full accent-purple-500"
                        />
                      </div>
                    </>
                  )}

                  {selectedComp.type === 'conditional' && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Condition</label>
                      <input
                        type="text"
                        value={selectedComp.config.condition}
                        onChange={(e) => {
                          setComponents(components.map(c =>
                            c.id === selectedComp.id
                              ? { ...c, config: { ...c.config, condition: e.target.value } }
                              : c
                          ))
                        }}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded font-mono text-sm"
                        placeholder="e.g., length > 100"
                      />
                    </div>
                  )}

                  {selectedComp.type === 'output' && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Output Format</label>
                      <select
                        value={selectedComp.config.format}
                        onChange={(e) => {
                          setComponents(components.map(c =>
                            c.id === selectedComp.id
                              ? { ...c, config: { ...c.config, format: e.target.value } }
                              : c
                          ))
                        }}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                      >
                        <option>text</option>
                        <option>json</option>
                        <option>markdown</option>
                        <option>html</option>
                      </select>
                    </div>
                  )}

                  <button
                    onClick={() => deleteComponent(selectedComp.id)}
                    className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 rounded flex items-center justify-center gap-2 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete Component
                  </button>
                </div>
              </div>
            )}

            {/* Connection Info */}
            {selectedConnection && (
              <div className="bg-gray-800/50 backdrop-blur border border-amber-500 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4 text-amber-400">🔗 Connection</h3>
                <div className="space-y-4">
                  <p className="text-sm text-gray-300">Connection selected</p>
                  <button
                    onClick={() => deleteConnection(selectedConnection)}
                    className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 rounded flex items-center justify-center gap-2 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete Connection
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Canvas */}
          <div className={`${isFullscreen ? 'lg:col-span-10' : 'lg:col-span-3'} space-y-4`}>
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold">🎨 Canvas</h3>
                <div className="flex gap-2">
                  <button
                    onClick={saveWorkflow}
                    disabled={components.length === 0}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded flex items-center gap-2 transition-colors"
                    title="Save workflow to browser"
                  >
                    <Save className="w-4 h-4" />
                    Save
                  </button>
                  <button
                    onClick={loadWorkflow}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded flex items-center gap-2 transition-colors"
                    title="Load saved workflow"
                  >
                    <Upload className="w-4 h-4" />
                    Load
                  </button>
                  <button
                    onClick={undo}
                    disabled={historyIndex <= 0}
                    className="px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed rounded flex items-center gap-2 transition-colors"
                    title="Undo (Ctrl+Z)"
                  >
                    <Undo className="w-4 h-4" />
                  </button>
                  <button
                    onClick={redo}
                    disabled={historyIndex >= history.length - 1}
                    className="px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed rounded flex items-center gap-2 transition-colors"
                    title="Redo (Ctrl+Y)"
                  >
                    <Redo className="w-4 h-4" />
                  </button>
                  <button
                    onClick={exportCode}
                    disabled={components.length === 0}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded flex items-center gap-2 transition-colors"
                  >
                    <Code className="w-4 h-4" />
                    Export Code
                  </button>
                  <button
                    onClick={clearAll}
                    disabled={components.length === 0}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded flex items-center gap-2 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    Clear All
                  </button>
                </div>
              </div>

              {/* Connection Mode Indicator */}
              {connectingFrom && (
                <div className="mb-4 px-4 py-2 bg-green-900/30 border border-green-600 rounded-lg text-sm text-green-400">
                  🔌 Connection mode active - Click on a blue input port (←) to complete the connection
                </div>
              )}

              <div
                ref={canvasRef}
                className={`relative bg-gray-900 rounded-lg border-2 border-dashed border-gray-600 ${isFullscreen ? 'h-[85vh]' : 'h-[500px]'} overflow-hidden cursor-default`}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onClick={handleCanvasClick}
              >
                {/* Connections SVG */}
                <svg className="absolute inset-0 pointer-events-none w-full h-full">
                  {connections.map((conn) => {
                    const from = components.find(c => c.id === conn.from)
                    const to = components.find(c => c.id === conn.to)
                    if (!from || !to) return null

                    const x1 = from.position.x + 150 // Right edge
                    const y1 = from.position.y + 35
                    const x2 = to.position.x // Left edge
                    const y2 = to.position.y + 35

                    const isSelected = selectedConnection === conn.id

                    return (
                      <g key={conn.id}>
                        {/* Invisible thick line for easier clicking */}
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="transparent"
                          strokeWidth="20"
                          className="pointer-events-auto cursor-pointer"
                          onClick={(e) => handleConnectionClick(conn.id, e as any)}
                        />
                        {/* Visible line */}
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke={isSelected ? '#fbbf24' : '#f59e0b'}
                          strokeWidth={isSelected ? '3' : '2'}
                          markerEnd="url(#arrowhead)"
                          className="pointer-events-none"
                        />
                      </g>
                    )
                  })}

                  {/* Temporary connection line */}
                  {connectingFrom && connectionLine && components.find(c => c.id === connectingFrom) && (
                    <line
                      x1={components.find(c => c.id === connectingFrom)!.position.x + 150}
                      y1={components.find(c => c.id === connectingFrom)!.position.y + 35}
                      x2={connectionLine.x}
                      y2={connectionLine.y}
                      stroke="#10b981"
                      strokeWidth="2"
                      strokeDasharray="5,5"
                    />
                  )}

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

                {/* Components */}
                {components.map(comp => {
                  const template = COMPONENT_TEMPLATES[comp.type]
                  const isSelected = selectedComponent === comp.id
                  const isConnectingFrom = connectingFrom === comp.id

                  return (
                    <div
                      key={comp.id}
                      className={`absolute select-none transition-all ${
                        isSelected ? 'ring-2 ring-amber-500 shadow-lg shadow-amber-500/50' : ''
                      } ${isConnectingFrom ? 'ring-2 ring-green-500 shadow-lg shadow-green-500/50' : ''}`}
                      style={{
                        left: comp.position.x,
                        top: comp.position.y,
                        width: 150,
                      }}
                    >
                      {/* Input Port */}
                      <div
                        className="absolute -left-3 top-1/2 -translate-y-1/2 w-6 h-6 bg-blue-500 rounded-full border-2 border-white cursor-pointer hover:scale-125 transition-transform flex items-center justify-center text-xs font-bold z-10"
                        onClick={(e) => handleInputPortClick(comp.id, e)}
                        title="Input Port - Click to connect"
                      >
                        ←
                      </div>

                      {/* Component Card */}
                      <div
                        className="bg-gray-800 border-2 border-gray-600 rounded-lg p-4 hover:border-amber-500 transition-all cursor-move"
                        onMouseDown={(e) => handleMouseDown(comp.id, e)}
                      >
                        <div className="text-center">
                          <div className="text-3xl mb-2">{template.icon}</div>
                          <div className="text-sm font-bold mb-1">{comp.label}</div>
                          <div className="text-xs text-gray-400 uppercase">{comp.type}</div>
                        </div>
                      </div>

                      {/* Output Port */}
                      <div
                        className={`absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full border-2 border-white cursor-pointer hover:scale-125 transition-transform flex items-center justify-center text-xs font-bold z-10 ${
                          isConnectingFrom ? 'bg-green-500 animate-pulse' : 'bg-green-600'
                        }`}
                        onClick={(e) => handleOutputPortClick(comp.id, e)}
                        title="Output Port - Click to start connection"
                      >
                        →
                      </div>
                    </div>
                  )
                })}

                {/* Empty State */}
                {components.length === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <Plus className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-lg font-medium">Start building your chain</p>
                      <p className="text-sm">Add components from the left palette</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Stats */}
              {components.length > 0 && (
                <div className="mt-4 flex items-center justify-between text-sm text-gray-400">
                  <div>
                    {components.length} component{components.length !== 1 ? 's' : ''} • {connections.length} connection{connections.length !== 1 ? 's' : ''}
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <span>Input</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-600 rounded-full"></div>
                      <span>Output</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Test Panel */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-amber-500" />
                Test Chain
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Input</label>
                  <input
                    type="text"
                    value={testInput}
                    onChange={(e) => setTestInput(e.target.value)}
                    placeholder="What is LangChain?"
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded focus:border-amber-500 focus:ring-1 focus:ring-amber-500 outline-none transition-colors"
                  />
                </div>

                <button
                  onClick={executeChain}
                  disabled={components.length === 0 || executing || !testInput}
                  className="w-full px-4 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed rounded-lg font-medium flex items-center justify-center gap-2 transition-all"
                >
                  <Play className="w-5 h-5" />
                  {executing ? 'Executing...' : 'Execute Chain'}
                </button>

                {testOutput && (
                  <div>
                    <label className="block text-sm font-medium mb-2">Execution Log</label>
                    <pre className="px-4 py-3 bg-gray-900 border border-gray-600 rounded text-xs overflow-auto max-h-[300px] whitespace-pre-wrap font-mono">
                      {testOutput}
                    </pre>
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
