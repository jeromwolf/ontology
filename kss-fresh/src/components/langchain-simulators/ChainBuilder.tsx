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

// Workflow Templates
const WORKFLOW_TEMPLATES = {
  rag: {
    name: 'RAG Pipeline',
    description: 'Complete RAG system with vector database, embeddings, and retrieval',
    icon: '📚',
    components: [
      {
        id: 'splitter-1',
        type: 'splitter' as const,
        label: 'Text Splitter',
        config: { chunkSize: 1000, chunkOverlap: 200 },
        position: { x: 100, y: 100 }
      },
      {
        id: 'embedding-1',
        type: 'embedding' as const,
        label: 'Embedding',
        config: { model: 'text-embedding-ada-002', dimensions: 1536 },
        position: { x: 100, y: 250 }
      },
      {
        id: 'vectordb-1',
        type: 'vectordb' as const,
        label: 'Vector Database',
        config: { database: 'chroma', index: 'documents', namespace: '' },
        position: { x: 300, y: 175 }
      },
      {
        id: 'retriever-1',
        type: 'retriever' as const,
        label: 'Retriever',
        config: { topK: 3, threshold: 0.7 },
        position: { x: 500, y: 100 }
      },
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Prompt',
        config: { template: 'Answer based on context: {context}\n\nQuestion: {question}' },
        position: { x: 500, y: 250 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-4', temperature: 0.7, maxTokens: 500 },
        position: { x: 700, y: 175 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'text' },
        position: { x: 900, y: 175 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'splitter-1', to: 'embedding-1' },
      { id: 'conn-2', from: 'embedding-1', to: 'vectordb-1' },
      { id: 'conn-3', from: 'vectordb-1', to: 'retriever-1' },
      { id: 'conn-4', from: 'retriever-1', to: 'prompt-1' },
      { id: 'conn-5', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-6', from: 'llm-1', to: 'output-1' }
    ]
  },
  agent: {
    name: 'Agent + Tools',
    description: 'Intelligent agent with calculator and search tools',
    icon: '🤖',
    components: [
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-4', temperature: 0.7, maxTokens: 1000 },
        position: { x: 100, y: 200 }
      },
      {
        id: 'tool-1',
        type: 'tool' as const,
        label: 'Tool',
        config: { name: 'calculator', description: 'Performs mathematical calculations' },
        position: { x: 300, y: 100 }
      },
      {
        id: 'tool-2',
        type: 'tool' as const,
        label: 'Tool',
        config: { name: 'search', description: 'Searches the web for information' },
        position: { x: 300, y: 200 }
      },
      {
        id: 'tool-3',
        type: 'tool' as const,
        label: 'Tool',
        config: { name: 'wikipedia', description: 'Queries Wikipedia for facts' },
        position: { x: 300, y: 300 }
      },
      {
        id: 'agent-1',
        type: 'agent' as const,
        label: 'Agent',
        config: { type: 'react', maxIterations: 5 },
        position: { x: 500, y: 200 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'text' },
        position: { x: 700, y: 200 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'llm-1', to: 'agent-1' },
      { id: 'conn-2', from: 'tool-1', to: 'agent-1' },
      { id: 'conn-3', from: 'tool-2', to: 'agent-1' },
      { id: 'conn-4', from: 'tool-3', to: 'agent-1' },
      { id: 'conn-5', from: 'agent-1', to: 'output-1' }
    ]
  },
  conversational: {
    name: 'Conversational AI',
    description: 'Chat system with memory and context awareness',
    icon: '💬',
    components: [
      {
        id: 'memory-1',
        type: 'memory' as const,
        label: 'Memory',
        config: { type: 'buffer', maxTokens: 2000 },
        position: { x: 100, y: 150 }
      },
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Prompt',
        config: { template: 'You are a helpful assistant. Chat history: {history}\n\nHuman: {input}\nAI:' },
        position: { x: 100, y: 300 }
      },
      {
        id: 'chat-1',
        type: 'chat' as const,
        label: 'Chat Model',
        config: { model: 'gpt-3.5-turbo', temperature: 0.9, maxTokens: 500 },
        position: { x: 350, y: 225 }
      },
      {
        id: 'parser-1',
        type: 'parser' as const,
        label: 'Parser',
        config: { format: 'text' },
        position: { x: 600, y: 150 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'text' },
        position: { x: 600, y: 300 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'memory-1', to: 'chat-1' },
      { id: 'conn-2', from: 'prompt-1', to: 'chat-1' },
      { id: 'conn-3', from: 'chat-1', to: 'parser-1' },
      { id: 'conn-4', from: 'chat-1', to: 'output-1' }
    ]
  },
  summarization: {
    name: 'Document Summarization',
    description: 'Summarize long documents with map-reduce pattern',
    icon: '📝',
    components: [
      {
        id: 'splitter-1',
        type: 'splitter' as const,
        label: 'Text Splitter',
        config: { chunkSize: 2000, chunkOverlap: 100 },
        position: { x: 100, y: 200 }
      },
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Map Prompt',
        config: { template: 'Summarize this chunk concisely:\n\n{text}' },
        position: { x: 300, y: 100 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM (Map)',
        config: { model: 'gpt-3.5-turbo', temperature: 0.3, maxTokens: 300 },
        position: { x: 500, y: 100 }
      },
      {
        id: 'prompt-2',
        type: 'prompt' as const,
        label: 'Reduce Prompt',
        config: { template: 'Combine these summaries into final summary:\n\n{summaries}' },
        position: { x: 300, y: 300 }
      },
      {
        id: 'llm-2',
        type: 'llm' as const,
        label: 'LLM (Reduce)',
        config: { model: 'gpt-4', temperature: 0.5, maxTokens: 500 },
        position: { x: 500, y: 300 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'text' },
        position: { x: 700, y: 200 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'splitter-1', to: 'prompt-1' },
      { id: 'conn-2', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-3', from: 'llm-1', to: 'prompt-2' },
      { id: 'conn-4', from: 'prompt-2', to: 'llm-2' },
      { id: 'conn-5', from: 'llm-2', to: 'output-1' }
    ]
  },
  qa_with_sources: {
    name: 'Q&A with Sources',
    description: 'Answer questions with source citations',
    icon: '🔍',
    components: [
      {
        id: 'retriever-1',
        type: 'retriever' as const,
        label: 'Retriever',
        config: { topK: 5, threshold: 0.6 },
        position: { x: 100, y: 200 }
      },
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Prompt',
        config: { template: 'Answer the question and cite sources.\n\nSources: {sources}\n\nQuestion: {question}' },
        position: { x: 300, y: 200 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-4', temperature: 0.2, maxTokens: 800 },
        position: { x: 500, y: 200 }
      },
      {
        id: 'parser-1',
        type: 'parser' as const,
        label: 'Citation Parser',
        config: { format: 'structured' },
        position: { x: 700, y: 200 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'json' },
        position: { x: 900, y: 200 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'retriever-1', to: 'prompt-1' },
      { id: 'conn-2', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-3', from: 'llm-1', to: 'parser-1' },
      { id: 'conn-4', from: 'parser-1', to: 'output-1' }
    ]
  },
  sql_agent: {
    name: 'SQL Database Agent',
    description: 'Natural language to SQL queries',
    icon: '🗄️',
    components: [
      {
        id: 'tool-1',
        type: 'tool' as const,
        label: 'SQL Tool',
        config: { name: 'sql_executor', description: 'Executes SQL queries' },
        position: { x: 100, y: 100 }
      },
      {
        id: 'tool-2',
        type: 'tool' as const,
        label: 'Schema Tool',
        config: { name: 'schema_reader', description: 'Reads database schema' },
        position: { x: 100, y: 250 }
      },
      {
        id: 'agent-1',
        type: 'agent' as const,
        label: 'SQL Agent',
        config: { type: 'react', maxIterations: 10 },
        position: { x: 350, y: 175 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-4', temperature: 0, maxTokens: 1000 },
        position: { x: 600, y: 175 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'table' },
        position: { x: 800, y: 175 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'tool-1', to: 'agent-1' },
      { id: 'conn-2', from: 'tool-2', to: 'agent-1' },
      { id: 'conn-3', from: 'agent-1', to: 'llm-1' },
      { id: 'conn-4', from: 'llm-1', to: 'output-1' }
    ]
  },
  code_generator: {
    name: 'Code Generator',
    description: 'Generate code from natural language',
    icon: '💻',
    components: [
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'System Prompt',
        config: { template: 'You are an expert programmer. Generate clean, efficient code.\n\nTask: {task}\nLanguage: {language}' },
        position: { x: 100, y: 200 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'Code LLM',
        config: { model: 'gpt-4', temperature: 0.2, maxTokens: 2000 },
        position: { x: 350, y: 200 }
      },
      {
        id: 'parser-1',
        type: 'parser' as const,
        label: 'Code Parser',
        config: { format: 'code' },
        position: { x: 600, y: 200 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'code' },
        position: { x: 800, y: 200 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-2', from: 'llm-1', to: 'parser-1' },
      { id: 'conn-3', from: 'parser-1', to: 'output-1' }
    ]
  },
  translation: {
    name: 'Multi-language Translation',
    description: 'Translate text between multiple languages',
    icon: '🌐',
    components: [
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Translation Prompt',
        config: { template: 'Translate from {source_lang} to {target_lang}:\n\n{text}' },
        position: { x: 100, y: 200 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-3.5-turbo', temperature: 0.3, maxTokens: 1000 },
        position: { x: 350, y: 200 }
      },
      {
        id: 'cache-1',
        type: 'cache' as const,
        label: 'Translation Cache',
        config: { enabled: true, ttl: 3600 },
        position: { x: 600, y: 200 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'text' },
        position: { x: 800, y: 200 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-2', from: 'llm-1', to: 'cache-1' },
      { id: 'conn-3', from: 'cache-1', to: 'output-1' }
    ]
  },
  content_moderation: {
    name: 'Content Moderation',
    description: 'Filter inappropriate content with multi-stage checks',
    icon: '🛡️',
    components: [
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Moderation Prompt',
        config: { template: 'Analyze this content for policy violations:\n\n{content}' },
        position: { x: 100, y: 200 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-4', temperature: 0.1, maxTokens: 500 },
        position: { x: 350, y: 200 }
      },
      {
        id: 'conditional-1',
        type: 'conditional' as const,
        label: 'Safety Check',
        config: { condition: 'is_safe', trueLabel: 'Approved', falseLabel: 'Blocked' },
        position: { x: 600, y: 200 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'json' },
        position: { x: 800, y: 200 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-2', from: 'llm-1', to: 'conditional-1' },
      { id: 'conn-3', from: 'conditional-1', to: 'output-1' }
    ]
  },
  sentiment_analysis: {
    name: 'Sentiment Analysis',
    description: 'Analyze sentiment with detailed emotions',
    icon: '😊',
    components: [
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Analysis Prompt',
        config: { template: 'Analyze sentiment and emotions in this text:\n\n{text}' },
        position: { x: 100, y: 200 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-3.5-turbo', temperature: 0.3, maxTokens: 300 },
        position: { x: 350, y: 200 }
      },
      {
        id: 'parser-1',
        type: 'parser' as const,
        label: 'Sentiment Parser',
        config: { format: 'structured' },
        position: { x: 600, y: 200 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'json' },
        position: { x: 800, y: 200 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-2', from: 'llm-1', to: 'parser-1' },
      { id: 'conn-3', from: 'parser-1', to: 'output-1' }
    ]
  },
  recommendation: {
    name: 'Smart Recommendation',
    description: 'Personalized recommendations with user context',
    icon: '⭐',
    components: [
      {
        id: 'vectordb-1',
        type: 'vectordb' as const,
        label: 'User Profiles',
        config: { database: 'pinecone', index: 'users', namespace: 'profiles' },
        position: { x: 100, y: 100 }
      },
      {
        id: 'retriever-1',
        type: 'retriever' as const,
        label: 'Similar Users',
        config: { topK: 10, threshold: 0.75 },
        position: { x: 100, y: 250 }
      },
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Recommendation Prompt',
        config: { template: 'Based on user preferences and similar users, recommend items:\n\nUser: {user}\nSimilar: {similar}' },
        position: { x: 350, y: 175 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-4', temperature: 0.7, maxTokens: 800 },
        position: { x: 600, y: 175 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'json' },
        position: { x: 850, y: 175 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'vectordb-1', to: 'retriever-1' },
      { id: 'conn-2', from: 'retriever-1', to: 'prompt-1' },
      { id: 'conn-3', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-4', from: 'llm-1', to: 'output-1' }
    ]
  },
  email_automation: {
    name: 'Email Automation',
    description: 'Auto-draft and categorize emails with smart routing',
    icon: '📧',
    components: [
      {
        id: 'prompt-1',
        type: 'prompt' as const,
        label: 'Categorize Prompt',
        config: { template: 'Categorize this email (urgent/normal/spam):\n\n{email}' },
        position: { x: 100, y: 150 }
      },
      {
        id: 'llm-1',
        type: 'llm' as const,
        label: 'LLM',
        config: { model: 'gpt-3.5-turbo', temperature: 0.2, maxTokens: 100 },
        position: { x: 350, y: 150 }
      },
      {
        id: 'router-1',
        type: 'router' as const,
        label: 'Router',
        config: { routes: ['urgent', 'normal', 'spam'] },
        position: { x: 600, y: 150 }
      },
      {
        id: 'prompt-2',
        type: 'prompt' as const,
        label: 'Draft Reply',
        config: { template: 'Draft a professional reply to:\n\n{email}' },
        position: { x: 100, y: 300 }
      },
      {
        id: 'llm-2',
        type: 'llm' as const,
        label: 'Reply LLM',
        config: { model: 'gpt-4', temperature: 0.7, maxTokens: 500 },
        position: { x: 350, y: 300 }
      },
      {
        id: 'output-1',
        type: 'output' as const,
        label: 'Output',
        config: { format: 'json' },
        position: { x: 850, y: 225 }
      }
    ],
    connections: [
      { id: 'conn-1', from: 'prompt-1', to: 'llm-1' },
      { id: 'conn-2', from: 'llm-1', to: 'router-1' },
      { id: 'conn-3', from: 'router-1', to: 'output-1' },
      { id: 'conn-4', from: 'prompt-2', to: 'llm-2' },
      { id: 'conn-5', from: 'llm-2', to: 'output-1' }
    ]
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
  const [showTemplates, setShowTemplates] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Execution simulation state
  const [isSimulating, setIsSimulating] = useState(false)
  const [simulationLogs, setSimulationLogs] = useState<{step: number, component: string, message: string, timestamp: number}[]>([])
  const [currentStep, setCurrentStep] = useState(0)

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

  // Save workflow to localStorage with custom name
  const saveWorkflow = () => {
    const name = prompt('워크플로우 이름을 입력하세요:', 'My Workflow')
    if (!name) return

    const workflow = {
      name,
      components,
      connections,
      timestamp: new Date().toISOString(),
      version: '1.0'
    }

    // Get existing workflows
    const saved = localStorage.getItem('langchain-workflows')
    const workflows = saved ? JSON.parse(saved) : []

    // Add or update workflow
    const existingIndex = workflows.findIndex((w: any) => w.name === name)
    if (existingIndex >= 0) {
      if (confirm(`"${name}" 워크플로우가 이미 존재합니다. 덮어쓰시겠습니까?`)) {
        workflows[existingIndex] = workflow
      } else {
        return
      }
    } else {
      workflows.push(workflow)
    }

    localStorage.setItem('langchain-workflows', JSON.stringify(workflows))
    alert(`✅ "${name}" 저장 완료!`)
  }

  // Load workflow from localStorage
  const loadWorkflow = (workflowName: string) => {
    const saved = localStorage.getItem('langchain-workflows')
    if (!saved) {
      alert('❌ No saved workflows found')
      return
    }

    try {
      const workflows = JSON.parse(saved)
      const workflow = workflows.find((w: any) => w.name === workflowName)

      if (!workflow) {
        alert('❌ Workflow not found')
        return
      }

      setComponents(workflow.components)
      setConnections(workflow.connections)
      setShowTemplates(false)
      alert(`✅ "${workflowName}" 불러오기 완료! (저장: ${new Date(workflow.timestamp).toLocaleString()})`)
    } catch (error) {
      alert('❌ Failed to load workflow')
      console.error(error)
    }
  }

  // Delete saved workflow
  const deleteWorkflow = (workflowName: string) => {
    const saved = localStorage.getItem('langchain-workflows')
    if (!saved) return

    try {
      const workflows = JSON.parse(saved)
      const filtered = workflows.filter((w: any) => w.name !== workflowName)
      localStorage.setItem('langchain-workflows', JSON.stringify(filtered))
      alert(`✅ "${workflowName}" 삭제 완료!`)
      // Trigger re-render by toggling templates
      setShowTemplates(false)
      setTimeout(() => setShowTemplates(true), 10)
    } catch (error) {
      alert('❌ Failed to delete workflow')
      console.error(error)
    }
  }

  // Get saved workflows
  const getSavedWorkflows = () => {
    const saved = localStorage.getItem('langchain-workflows')
    if (!saved) return []
    try {
      return JSON.parse(saved)
    } catch {
      return []
    }
  }

  // Load template workflow
  const loadTemplate = (templateKey: keyof typeof WORKFLOW_TEMPLATES) => {
    const template = WORKFLOW_TEMPLATES[templateKey]
    setComponents(template.components as any[])
    setConnections(template.connections)
    setShowTemplates(false)
    alert(`✅ ${template.name} template loaded!`)
  }

  // Simulate workflow execution
  const simulateExecution = async () => {
    if (components.length === 0) {
      alert('⚠️ Add components to your workflow first')
      return
    }

    setIsSimulating(true)
    setSimulationLogs([])
    setCurrentStep(0)

    const logs: {step: number, component: string, message: string, timestamp: number}[] = []

    // Build execution order based on connections
    const executionOrder: string[] = []
    const visited = new Set<string>()

    // Find starting components (no incoming connections)
    const startComponents = components.filter(comp =>
      !connections.some(conn => conn.to === comp.id)
    )

    // Perform topological sort for execution order
    const buildOrder = (compId: string) => {
      if (visited.has(compId)) return
      visited.add(compId)

      const outgoing = connections.filter(conn => conn.from === compId)
      outgoing.forEach(conn => buildOrder(conn.to))

      executionOrder.unshift(compId)
    }

    startComponents.forEach(comp => buildOrder(comp.id))

    // Add any orphaned components
    components.forEach(comp => {
      if (!executionOrder.includes(comp.id)) {
        executionOrder.push(comp.id)
      }
    })

    // Simulate execution step by step
    for (let i = 0; i < executionOrder.length; i++) {
      const compId = executionOrder[i]
      const component = components.find(c => c.id === compId)
      if (!component) continue

      setCurrentStep(i + 1)

      const log = {
        step: i + 1,
        component: component.label,
        message: getExecutionMessage(component),
        timestamp: Date.now()
      }

      logs.push(log)
      setSimulationLogs([...logs])

      // Wait for visual effect
      await new Promise(resolve => setTimeout(resolve, 800))
    }

    const finalLog = {
      step: executionOrder.length + 1,
      component: 'System',
      message: '✅ Workflow execution completed successfully!',
      timestamp: Date.now()
    }
    logs.push(finalLog)
    setSimulationLogs([...logs])

    setTimeout(() => {
      setIsSimulating(false)
      setCurrentStep(0)
    }, 2000)
  }

  // Generate execution message based on component type
  const getExecutionMessage = (component: ChainComponent): string => {
    switch (component.type) {
      case 'llm':
        return `🤖 Calling LLM (${component.config.model}) with temperature ${component.config.temperature}`
      case 'prompt':
        return `📝 Processing prompt template...`
      case 'retriever':
        return `🔍 Retrieving top ${component.config.topK} documents (threshold: ${component.config.threshold})`
      case 'vectordb':
        return `💾 Querying vector database (${component.config.database})...`
      case 'embedding':
        return `🔢 Generating embeddings using ${component.config.model}...`
      case 'splitter':
        return `✂️ Splitting text (chunk size: ${component.config.chunkSize})...`
      case 'memory':
        return `🧠 Loading conversation memory (${component.config.type})...`
      case 'agent':
        return `🤖 Agent reasoning (max ${component.config.maxIterations} iterations)...`
      case 'tool':
        return `🔧 Executing tool: ${component.config.name}`
      case 'parser':
        return `📊 Parsing output format (${component.config.format})...`
      case 'chain':
        return `⛓️ Running chain: ${component.config.type}`
      case 'router':
        return `🔀 Routing to appropriate chain...`
      case 'cache':
        return `💨 Cache ${component.config.enabled ? 'HIT' : 'MISS'} (TTL: ${component.config.ttl}s)`
      case 'output':
        return `📤 Formatting output as ${component.config.format}...`
      default:
        return `⚙️ Processing ${component.label}...`
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
                onClick={simulateExecution}
                disabled={isSimulating || components.length === 0}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg flex items-center gap-2"
                title="Simulate Workflow Execution"
              >
                <Play className="w-5 h-5" />
                {isSimulating ? 'Running...' : 'Run Simulation'}
              </button>
              <button
                onClick={() => setShowTemplates(!showTemplates)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center gap-2"
                title="Load Template Workflow"
              >
                <Zap className="w-5 h-5" />
                Templates
              </button>
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

        {/* Template Gallery Modal */}
        {showTemplates && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-8 max-w-6xl w-full max-h-[85vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-3xl font-bold flex items-center gap-3">
                  <Zap className="w-8 h-8 text-purple-400" />
                  워크플로우 갤러리
                </h2>
                <button
                  onClick={() => setShowTemplates(false)}
                  className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Built-in Templates Section */}
              <div className="mb-10">
                <div className="flex items-center gap-2 mb-4">
                  <h3 className="text-xl font-bold text-purple-400">⚡ 내장 템플릿</h3>
                  <span className="text-sm text-gray-500">({Object.keys(WORKFLOW_TEMPLATES).length}개)</span>
                </div>
                <p className="text-gray-400 mb-6 text-sm">
                  바로 사용 가능한 전문 워크플로우 템플릿입니다. 클릭하면 자동으로 불러옵니다.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(WORKFLOW_TEMPLATES).map(([key, template]) => (
                    <button
                      key={key}
                      onClick={() => loadTemplate(key as keyof typeof WORKFLOW_TEMPLATES)}
                      className="bg-gray-900/50 border border-gray-700 hover:border-purple-500 rounded-xl p-5 text-left transition-all hover:scale-105"
                    >
                      <div className="text-4xl mb-3">{template.icon}</div>
                      <h4 className="text-lg font-bold mb-2">{template.name}</h4>
                      <p className="text-gray-400 text-xs mb-3 line-clamp-2">{template.description}</p>
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        <span>{template.components.length} 컴포넌트</span>
                        <span>•</span>
                        <span>{template.connections.length} 연결</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Saved Workflows Section */}
              {(() => {
                const savedWorkflows = getSavedWorkflows()
                return savedWorkflows.length > 0 ? (
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <h3 className="text-xl font-bold text-amber-400">💾 내 워크플로우</h3>
                      <span className="text-sm text-gray-500">({savedWorkflows.length}개 저장됨)</span>
                    </div>
                    <p className="text-gray-400 mb-6 text-sm">
                      저장한 커스텀 워크플로우입니다. 불러오기, 삭제가 가능합니다.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      {savedWorkflows.map((workflow: any) => (
                        <div
                          key={workflow.name}
                          className="bg-gray-900/50 border border-gray-700 hover:border-amber-500 rounded-xl p-5 transition-all group"
                        >
                          <div className="flex items-start justify-between mb-3">
                            <div className="text-4xl">📋</div>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                if (confirm(`"${workflow.name}" 워크플로우를 삭제하시겠습니까?`)) {
                                  deleteWorkflow(workflow.name)
                                }
                              }}
                              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-600/20 rounded transition-all"
                              title="삭제"
                            >
                              <X className="w-4 h-4 text-red-400" />
                            </button>
                          </div>
                          <h4 className="text-lg font-bold mb-2 truncate">{workflow.name}</h4>
                          <p className="text-gray-400 text-xs mb-3">
                            저장: {new Date(workflow.timestamp).toLocaleDateString('ko-KR')} {new Date(workflow.timestamp).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                          </p>
                          <div className="flex items-center gap-2 text-xs text-gray-500 mb-3">
                            <span>{workflow.components.length} 컴포넌트</span>
                            <span>•</span>
                            <span>{workflow.connections.length} 연결</span>
                          </div>
                          <button
                            onClick={() => loadWorkflow(workflow.name)}
                            className="w-full px-3 py-2 bg-amber-600 hover:bg-amber-700 rounded-lg text-sm font-medium transition-colors"
                          >
                            불러오기
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 bg-gray-900/30 border border-dashed border-gray-700 rounded-xl">
                    <p className="text-gray-500 text-sm">
                      💡 아직 저장된 워크플로우가 없습니다. 상단의 "Save Workflow" 버튼으로 워크플로우를 저장해보세요.
                    </p>
                  </div>
                )
              })()}
            </div>
          </div>
        )}

        {/* Fullscreen Floating Toolbar */}
        {isFullscreen && (
          <div className="fixed top-4 right-4 z-50 flex items-center gap-2 bg-gray-800/95 backdrop-blur border border-gray-700 rounded-lg p-2 shadow-2xl">
            <button
              onClick={() => setShowTemplates(!showTemplates)}
              className="px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded flex items-center gap-2 text-sm"
            >
              <Zap className="w-4 h-4" />
            </button>
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

                  // Skip rendering if template is not found (invalid component type)
                  if (!template) {
                    console.warn(`Template not found for component type: ${comp.type}`)
                    return null
                  }

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

        {/* Simulation Logs Panel */}
        {simulationLogs.length > 0 && (
          <div className="mt-6 bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold flex items-center gap-2">
                <Play className="w-6 h-6 text-green-400" />
                Execution Simulation Logs
              </h3>
              <button
                onClick={() => {
                  setSimulationLogs([])
                  setCurrentStep(0)
                }}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
              >
                Clear Logs
              </button>
            </div>

            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {simulationLogs.map((log, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg border ${
                    index === currentStep - 1 && isSimulating
                      ? 'bg-green-900/30 border-green-500 animate-pulse'
                      : 'bg-gray-900/50 border-gray-700'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-xs font-mono text-gray-500 mt-0.5">
                      Step {log.step}
                    </span>
                    <div className="flex-1">
                      <div className="font-medium text-sm text-green-400 mb-1">
                        {log.component}
                      </div>
                      <div className="text-sm text-gray-300">
                        {log.message}
                      </div>
                    </div>
                    <span className="text-xs text-gray-500">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {isSimulating && (
              <div className="mt-4 flex items-center gap-2 text-sm text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                Executing step {currentStep}...
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
