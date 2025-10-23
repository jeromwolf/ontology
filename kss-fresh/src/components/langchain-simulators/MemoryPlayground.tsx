'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, Trash2, Database, Eye, BarChart3 } from 'lucide-react'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

type MemoryType = 'buffer' | 'window' | 'summary' | 'vector'

interface MemoryState {
  messages: Message[]
  summary?: string
  vectorEmbeddings?: { text: string; embedding: number[] }[]
}

export default function MemoryPlayground() {
  const [memoryType, setMemoryType] = useState<MemoryType>('buffer')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [memoryState, setMemoryState] = useState<MemoryState>({ messages: [] })
  const [windowSize, setWindowSize] = useState(5)
  const [summaryThreshold, setSummaryThreshold] = useState(10)
  const [tokenCount, setTokenCount] = useState(0)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    updateMemoryState()
  }, [messages, memoryType, windowSize, summaryThreshold])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const updateMemoryState = () => {
    let newState: MemoryState = { messages: [] }

    switch (memoryType) {
      case 'buffer':
        newState.messages = messages
        break

      case 'window':
        newState.messages = messages.slice(-windowSize)
        break

      case 'summary':
        if (messages.length > summaryThreshold) {
          const oldMessages = messages.slice(0, -summaryThreshold)
          const summary = generateSummary(oldMessages)
          newState.summary = summary
          newState.messages = messages.slice(-summaryThreshold)
        } else {
          newState.messages = messages
        }
        break

      case 'vector':
        newState.vectorEmbeddings = messages.map(m => ({
          text: m.content,
          embedding: generateMockEmbedding(m.content)
        }))
        break
    }

    setMemoryState(newState)
    calculateTokenCount(newState)
  }

  const generateSummary = (msgs: Message[]): string => {
    const topics = msgs
      .filter(m => m.role === 'user')
      .map(m => m.content.split(' ').slice(0, 3).join(' '))
      .slice(0, 3)
    return `Previous conversation covered: ${topics.join(', ')}...`
  }

  const generateMockEmbedding = (text: string): number[] => {
    // Simple mock embedding (in reality, this would call an embedding model)
    const seed = text.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
    return Array.from({ length: 8 }, (_, i) => Math.sin(seed * (i + 1)) * 0.5)
  }

  const calculateTokenCount = (state: MemoryState) => {
    let count = 0
    if (state.summary) {
      count += state.summary.split(' ').length * 1.3
    }
    state.messages.forEach(m => {
      count += m.content.split(' ').length * 1.3
    })
    setTokenCount(Math.round(count))
  }

  const sendMessage = () => {
    if (!input.trim()) return

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: Date.now()
    }

    // Simulate AI response
    const responses = [
      'That\'s an interesting point. Let me elaborate on that.',
      'I understand. Based on our conversation, I can help with that.',
      'Great question! Here\'s what I think...',
      'Yes, considering what we discussed earlier, I would suggest...',
      'I see the connection to what we talked about before.'
    ]

    const assistantMessage: Message = {
      role: 'assistant',
      content: responses[Math.floor(Math.random() * responses.length)],
      timestamp: Date.now() + 100
    }

    setMessages([...messages, userMessage, assistantMessage])
    setInput('')
  }

  const clearMemory = () => {
    if (confirm('Clear all messages?')) {
      setMessages([])
      setMemoryState({ messages: [] })
    }
  }

  const getMemoryDescription = (type: MemoryType): string => {
    switch (type) {
      case 'buffer':
        return 'Stores all conversation history in memory'
      case 'window':
        return `Keeps only the last ${windowSize} messages`
      case 'summary':
        return `Summarizes old messages after ${summaryThreshold} messages`
      case 'vector':
        return 'Stores embeddings for semantic search'
    }
  }

  const getMemorySize = (): string => {
    const baseSize = messages.length * 50 // Average message size in bytes
    switch (memoryType) {
      case 'buffer':
        return `${(baseSize / 1024).toFixed(2)} KB`
      case 'window':
        return `${((baseSize * windowSize / messages.length) / 1024).toFixed(2)} KB`
      case 'summary':
        return `${(baseSize * 0.3 / 1024).toFixed(2)} KB (compressed)`
      case 'vector':
        return `${(messages.length * 8 * 4 / 1024).toFixed(2)} KB (embeddings)`
      default:
        return '0 KB'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            💾 Memory Playground
          </h1>
          <p className="text-gray-300 text-lg">
            Test different memory types and see how they manage conversation history.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Memory Type Selector */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Database className="w-5 h-5" />
                Memory Type
              </h3>

              <div className="space-y-2">
                {(['buffer', 'window', 'summary', 'vector'] as MemoryType[]).map(type => (
                  <button
                    key={type}
                    onClick={() => setMemoryType(type)}
                    className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-all ${
                      memoryType === type
                        ? 'bg-amber-600 border-amber-500'
                        : 'bg-gray-700/50 border-gray-600 hover:bg-gray-600'
                    }`}
                  >
                    <div className="font-medium capitalize">{type} Memory</div>
                    <div className="text-xs text-gray-300 mt-1">{getMemoryDescription(type)}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Memory Configuration */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Configuration</h3>

              {memoryType === 'window' && (
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Window Size: {windowSize} messages
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    value={windowSize}
                    onChange={(e) => setWindowSize(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              )}

              {memoryType === 'summary' && (
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Summary Threshold: {summaryThreshold} messages
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    value={summaryThreshold}
                    onChange={(e) => setSummaryThreshold(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              )}

              {memoryType === 'vector' && (
                <div className="text-sm text-gray-300">
                  <p>Vector embeddings are automatically generated for semantic search.</p>
                  <p className="mt-2">Embedding dimension: 8 (mock)</p>
                </div>
              )}
            </div>

            {/* Memory Stats */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Statistics
              </h3>

              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Total Messages</span>
                  <span className="font-bold text-amber-500">{messages.length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">In Memory</span>
                  <span className="font-bold text-blue-500">{memoryState.messages.length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Memory Size</span>
                  <span className="font-bold text-green-500">{getMemorySize()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Token Count</span>
                  <span className="font-bold text-purple-500">{tokenCount}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Chat Interface */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold">Chat Interface</h3>
                <button
                  onClick={clearMemory}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded flex items-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear
                </button>
              </div>

              <div className="bg-gray-900 rounded-lg border border-gray-600 h-[400px] overflow-y-auto p-4 mb-4">
                {messages.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-gray-500">
                    Start a conversation to test memory...
                  </div>
                ) : (
                  <>
                    {messages.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`mb-3 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}
                      >
                        <div
                          className={`inline-block px-4 py-2 rounded-lg max-w-[80%] ${
                            msg.role === 'user'
                              ? 'bg-amber-600 text-white'
                              : 'bg-gray-700 text-gray-100'
                          }`}
                        >
                          <div className="text-xs font-semibold mb-1">
                            {msg.role === 'user' ? 'You' : 'Assistant'}
                          </div>
                          <div>{msg.content}</div>
                        </div>
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </>
                )}
              </div>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Type your message..."
                  className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg"
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim()}
                  className="px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 rounded-lg flex items-center gap-2"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Memory State Viewer */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5" />
                Memory State
              </h3>

              <div className="bg-gray-900 rounded-lg border border-gray-600 p-4 max-h-[300px] overflow-y-auto">
                {memoryState.summary && (
                  <div className="mb-4 p-3 bg-blue-900/30 border border-blue-700 rounded">
                    <div className="text-sm font-semibold text-blue-400 mb-1">Summary</div>
                    <div className="text-sm">{memoryState.summary}</div>
                  </div>
                )}

                {memoryType === 'vector' ? (
                  <div className="space-y-2">
                    {memoryState.vectorEmbeddings?.map((item, idx) => (
                      <div key={idx} className="p-3 bg-gray-800 rounded border border-gray-700">
                        <div className="text-xs text-gray-400 mb-1">Message {idx + 1}</div>
                        <div className="text-sm mb-2">{item.text}</div>
                        <div className="text-xs font-mono text-purple-400">
                          [{item.embedding.map(v => v.toFixed(2)).join(', ')}]
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-2">
                    {memoryState.messages.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`p-3 rounded border ${
                          msg.role === 'user'
                            ? 'bg-amber-900/30 border-amber-700'
                            : 'bg-gray-800 border-gray-700'
                        }`}
                      >
                        <div className="text-xs font-semibold mb-1">{msg.role}</div>
                        <div className="text-sm">{msg.content}</div>
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
