'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, Download, Upload, Search, Trash2, BarChart, MessageSquare } from 'lucide-react'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  tokens?: number
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: number
  totalTokens: number
}

export default function ChatHistoryManager() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConvId, setActiveConvId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Load from localStorage
    const saved = localStorage.getItem('langchain-conversations')
    if (saved) {
      try {
        setConversations(JSON.parse(saved))
      } catch (e) {
        console.error('Failed to load conversations')
      }
    } else {
      // Create default conversation
      createNewConversation()
    }
  }, [])

  useEffect(() => {
    // Save to localStorage
    if (conversations.length > 0) {
      localStorage.setItem('langchain-conversations', JSON.stringify(conversations))
    }
  }, [conversations])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversations, activeConvId])

  const createNewConversation = () => {
    const newConv: Conversation = {
      id: `conv-${Date.now()}`,
      title: `Conversation ${conversations.length + 1}`,
      messages: [],
      createdAt: Date.now(),
      totalTokens: 0
    }
    setConversations([...conversations, newConv])
    setActiveConvId(newConv.id)
  }

  const deleteConversation = (id: string) => {
    if (confirm('Delete this conversation?')) {
      setConversations(conversations.filter(c => c.id !== id))
      if (activeConvId === id) {
        setActiveConvId(conversations[0]?.id || null)
      }
    }
  }

  const sendMessage = () => {
    if (!input.trim() || !activeConvId) return

    const activeConv = conversations.find(c => c.id === activeConvId)
    if (!activeConv) return

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: Date.now(),
      tokens: Math.ceil(input.split(' ').length * 1.3)
    }

    // Simulate AI response
    const responses = [
      'I understand your question. Let me help you with that.',
      'Based on what we discussed earlier, here\'s my analysis...',
      'That\'s a great point. Let me elaborate on that.',
      'I can help you with that. Here\'s what I suggest...',
      'Looking at our conversation history, I think the best approach is...'
    ]

    const assistantMessage: Message = {
      id: `msg-${Date.now() + 1}`,
      role: 'assistant',
      content: responses[Math.floor(Math.random() * responses.length)],
      timestamp: Date.now() + 100,
      tokens: Math.ceil(responses[0].split(' ').length * 1.3)
    }

    const updatedMessages = [...activeConv.messages, userMessage, assistantMessage]
    const totalTokens = updatedMessages.reduce((sum, m) => sum + (m.tokens || 0), 0)

    setConversations(conversations.map(c =>
      c.id === activeConvId
        ? { ...c, messages: updatedMessages, totalTokens }
        : c
    ))

    setInput('')
  }

  const exportConversation = (convId: string) => {
    const conv = conversations.find(c => c.id === convId)
    if (!conv) return

    const data = {
      title: conv.title,
      createdAt: new Date(conv.createdAt).toISOString(),
      messages: conv.messages.map(m => ({
        role: m.role,
        content: m.content,
        timestamp: new Date(m.timestamp).toISOString()
      }))
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `conversation-${conv.title}-${Date.now()}.json`
    a.click()
  }

  const importConversation = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'application/json'
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string)
          const newConv: Conversation = {
            id: `conv-${Date.now()}`,
            title: data.title || 'Imported Conversation',
            messages: data.messages.map((m: any) => ({
              id: `msg-${Date.now()}-${Math.random()}`,
              role: m.role,
              content: m.content,
              timestamp: new Date(m.timestamp).getTime(),
              tokens: Math.ceil(m.content.split(' ').length * 1.3)
            })),
            createdAt: Date.now(),
            totalTokens: 0
          }
          newConv.totalTokens = newConv.messages.reduce((sum, m) => sum + (m.tokens || 0), 0)
          setConversations([...conversations, newConv])
          setActiveConvId(newConv.id)
        } catch (e) {
          alert('Failed to import conversation')
        }
      }
      reader.readAsText(file)
    }
    input.click()
  }

  const searchMessages = () => {
    if (!searchQuery.trim()) return conversations

    return conversations.map(conv => ({
      ...conv,
      messages: conv.messages.filter(m =>
        m.content.toLowerCase().includes(searchQuery.toLowerCase())
      )
    })).filter(conv => conv.messages.length > 0)
  }

  const activeConv = conversations.find(c => c.id === activeConvId)
  const displayConversations = searchQuery ? searchMessages() : conversations

  const totalMessages = conversations.reduce((sum, c) => sum + c.messages.length, 0)
  const totalTokens = conversations.reduce((sum, c) => sum + c.totalTokens, 0)
  const estimatedCost = (totalTokens / 1000) * 0.002 // $0.002 per 1K tokens (GPT-3.5)

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            💬 Chat History Manager
          </h1>
          <p className="text-gray-300 text-lg">
            Manage conversation history with export/import, search, and analytics.
          </p>
        </div>

        {/* Stats Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-4">
            <div className="text-sm text-gray-400 mb-1">Conversations</div>
            <div className="text-2xl font-bold text-amber-500">{conversations.length}</div>
          </div>
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-4">
            <div className="text-sm text-gray-400 mb-1">Total Messages</div>
            <div className="text-2xl font-bold text-blue-500">{totalMessages}</div>
          </div>
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-4">
            <div className="text-sm text-gray-400 mb-1">Total Tokens</div>
            <div className="text-2xl font-bold text-green-500">{totalTokens.toLocaleString()}</div>
          </div>
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-4">
            <div className="text-sm text-gray-400 mb-1">Estimated Cost</div>
            <div className="text-2xl font-bold text-purple-500">${estimatedCost.toFixed(4)}</div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            {/* Actions */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Actions</h3>

              <div className="space-y-2">
                <button
                  onClick={createNewConversation}
                  className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded flex items-center justify-center gap-2"
                >
                  <MessageSquare className="w-4 h-4" />
                  New Chat
                </button>

                <button
                  onClick={importConversation}
                  className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded flex items-center justify-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  Import
                </button>
              </div>
            </div>

            {/* Search */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Search</h3>

              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search messages..."
                  className="w-full px-4 py-2 pl-10 bg-gray-700 border border-gray-600 rounded"
                />
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
              </div>
            </div>

            {/* Conversation List */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Conversations</h3>

              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {displayConversations.map(conv => (
                  <div
                    key={conv.id}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      activeConvId === conv.id
                        ? 'bg-amber-600 border-amber-500'
                        : 'bg-gray-700/50 border-gray-600 hover:bg-gray-600'
                    }`}
                    onClick={() => setActiveConvId(conv.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="font-medium text-sm truncate">{conv.title}</div>
                        <div className="text-xs text-gray-400 mt-1">
                          {conv.messages.length} messages • {conv.totalTokens} tokens
                        </div>
                      </div>

                      <div className="flex gap-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            exportConversation(conv.id)
                          }}
                          className="p-1 hover:bg-gray-600 rounded"
                        >
                          <Download className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            deleteConversation(conv.id)
                          }}
                          className="p-1 hover:bg-red-600 rounded"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Chat Area */}
          <div className="lg:col-span-3 space-y-4">
            {activeConv ? (
              <>
                {/* Chat Header */}
                <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <input
                        type="text"
                        value={activeConv.title}
                        onChange={(e) => {
                          setConversations(conversations.map(c =>
                            c.id === activeConvId
                              ? { ...c, title: e.target.value }
                              : c
                          ))
                        }}
                        className="bg-transparent border-none text-xl font-bold focus:outline-none focus:ring-2 focus:ring-amber-500 rounded px-2"
                      />
                      <div className="text-sm text-gray-400 mt-1">
                        {new Date(activeConv.createdAt).toLocaleString()}
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div className="text-sm">
                        <span className="text-gray-400">Tokens:</span>
                        <span className="font-bold text-amber-500 ml-2">
                          {activeConv.totalTokens}
                        </span>
                      </div>
                      <button
                        onClick={() => exportConversation(activeConv.id)}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded flex items-center gap-2"
                      >
                        <Download className="w-4 h-4" />
                        Export
                      </button>
                    </div>
                  </div>
                </div>

                {/* Messages */}
                <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                  <div className="h-[500px] overflow-y-auto mb-4">
                    {activeConv.messages.length === 0 ? (
                      <div className="flex items-center justify-center h-full text-gray-500">
                        Start a conversation...
                      </div>
                    ) : (
                      <>
                        {activeConv.messages.map(msg => (
                          <div
                            key={msg.id}
                            className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}
                          >
                            <div
                              className={`inline-block px-4 py-3 rounded-lg max-w-[80%] ${
                                msg.role === 'user'
                                  ? 'bg-amber-600 text-white'
                                  : 'bg-gray-700 text-gray-100'
                              }`}
                            >
                              <div className="text-xs font-semibold mb-1 flex items-center gap-2">
                                {msg.role === 'user' ? 'You' : 'Assistant'}
                                <span className="text-gray-400">
                                  • {msg.tokens} tokens
                                </span>
                              </div>
                              <div>{msg.content}</div>
                              <div className="text-xs text-gray-400 mt-1">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                              </div>
                            </div>
                          </div>
                        ))}
                        <div ref={messagesEndRef} />
                      </>
                    )}
                  </div>

                  {/* Input */}
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

                {/* Analytics */}
                <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <BarChart className="w-5 h-5" />
                    Conversation Analytics
                  </h3>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-gray-900 rounded-lg">
                      <div className="text-2xl font-bold text-blue-500">
                        {activeConv.messages.filter(m => m.role === 'user').length}
                      </div>
                      <div className="text-sm text-gray-400 mt-1">User Messages</div>
                    </div>

                    <div className="text-center p-4 bg-gray-900 rounded-lg">
                      <div className="text-2xl font-bold text-green-500">
                        {activeConv.messages.filter(m => m.role === 'assistant').length}
                      </div>
                      <div className="text-sm text-gray-400 mt-1">AI Responses</div>
                    </div>

                    <div className="text-center p-4 bg-gray-900 rounded-lg">
                      <div className="text-2xl font-bold text-purple-500">
                        {activeConv.messages.length > 0
                          ? Math.round(activeConv.totalTokens / activeConv.messages.length)
                          : 0}
                      </div>
                      <div className="text-sm text-gray-400 mt-1">Avg Tokens/Msg</div>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6 h-[600px] flex items-center justify-center">
                <div className="text-center text-gray-500">
                  <MessageSquare className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Select a conversation or create a new one</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
