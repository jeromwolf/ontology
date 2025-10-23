'use client'

import React, { useState } from 'react'
import { Upload, Search, Database, Zap, FileText, CheckCircle } from 'lucide-react'

interface Document {
  id: string
  content: string
  metadata: { source: string; page?: number }
  chunks: string[]
  embeddings?: number[][]
}

interface RetrievalResult {
  chunk: string
  score: number
  source: string
}

type EmbeddingModel = 'openai' | 'huggingface' | 'cohere'
type VectorStore = 'faiss' | 'chroma' | 'pinecone'

export default function RAGPipelineBuilder() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [embeddingModel, setEmbeddingModel] = useState<EmbeddingModel>('openai')
  const [vectorStore, setVectorStore] = useState<VectorStore>('faiss')
  const [chunkSize, setChunkSize] = useState(500)
  const [chunkOverlap, setChunkOverlap] = useState(50)
  const [topK, setTopK] = useState(3)
  const [query, setQuery] = useState('')
  const [retrievalResults, setRetrievalResults] = useState<RetrievalResult[]>([])
  const [generatedAnswer, setGeneratedAnswer] = useState('')
  const [processing, setProcessing] = useState(false)

  const sampleDocuments = [
    {
      title: 'LangChain Overview',
      content: `LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and can reason about how to answer based on provided context. LangChain provides tools for document loading, text splitting, embeddings, vector stores, and chains. The framework supports various LLM providers including OpenAI, Anthropic, and HuggingFace.`
    },
    {
      title: 'Vector Databases',
      content: `Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are crucial for RAG systems as they enable semantic search over large document collections. Popular vector databases include FAISS, Chroma, Pinecone, and Weaviate. These databases use algorithms like HNSW and IVF for fast approximate nearest neighbor search.`
    },
    {
      title: 'Embeddings Explained',
      content: `Embeddings are numerical representations of text that capture semantic meaning. They convert words, sentences, or documents into dense vectors where similar content has similar vector representations. Common embedding models include OpenAI's text-embedding-ada-002, sentence-transformers from HuggingFace, and Cohere embeddings. The quality of embeddings directly impacts RAG system performance.`
    }
  ]

  const addSampleDocuments = () => {
    const newDocs = sampleDocuments.map((doc, idx) => {
      const chunks = chunkDocument(doc.content)
      return {
        id: `doc-${Date.now()}-${idx}`,
        content: doc.content,
        metadata: { source: doc.title },
        chunks,
        embeddings: chunks.map(chunk => generateMockEmbedding(chunk))
      }
    })
    setDocuments([...documents, ...newDocs])
  }

  const addCustomDocument = () => {
    const content = prompt('Enter document content:')
    if (!content) return

    const chunks = chunkDocument(content)
    const newDoc: Document = {
      id: `doc-${Date.now()}`,
      content,
      metadata: { source: 'Custom Document' },
      chunks,
      embeddings: chunks.map(chunk => generateMockEmbedding(chunk))
    }
    setDocuments([...documents, newDoc])
  }

  const chunkDocument = (text: string): string[] => {
    const chunks: string[] = []
    let start = 0

    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length)
      chunks.push(text.substring(start, end))
      start += chunkSize - chunkOverlap
    }

    return chunks
  }

  const generateMockEmbedding = (text: string): number[] => {
    const seed = text.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
    return Array.from({ length: 16 }, (_, i) => Math.sin(seed * (i + 1) * 0.1))
  }

  const cosineSimilarity = (a: number[], b: number[]): number => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0)
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
    return dotProduct / (magA * magB)
  }

  const performRetrieval = async () => {
    if (!query.trim() || documents.length === 0) return

    setProcessing(true)
    setRetrievalResults([])
    setGeneratedAnswer('')

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 500))

    // Generate query embedding
    const queryEmbedding = generateMockEmbedding(query)

    // Calculate similarity scores for all chunks
    const results: RetrievalResult[] = []

    documents.forEach(doc => {
      doc.chunks.forEach((chunk, idx) => {
        const chunkEmbedding = doc.embeddings![idx]
        const score = cosineSimilarity(queryEmbedding, chunkEmbedding)

        results.push({
          chunk,
          score,
          source: doc.metadata.source
        })
      })
    })

    // Sort by score and take top K
    results.sort((a, b) => b.score - a.score)
    const topResults = results.slice(0, topK)

    setRetrievalResults(topResults)

    // Generate answer
    await new Promise(resolve => setTimeout(resolve, 800))

    const context = topResults.map(r => r.chunk).join('\n\n')
    const answer = `Based on the retrieved context, here's the answer to "${query}":\n\n${topResults[0].chunk.substring(0, 150)}...\n\nThis information comes from ${topResults[0].source} with a relevance score of ${(topResults[0].score * 100).toFixed(1)}%.`

    setGeneratedAnswer(answer)
    setProcessing(false)
  }

  const clearAll = () => {
    if (confirm('Clear all documents?')) {
      setDocuments([])
      setRetrievalResults([])
      setGeneratedAnswer('')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            🔍 RAG Pipeline Builder
          </h1>
          <p className="text-gray-300 text-lg">
            Build and test complete Retrieval-Augmented Generation pipelines with custom configurations.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="lg:col-span-1 space-y-4">
            {/* Document Management */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Documents ({documents.length})
              </h3>

              <div className="space-y-2">
                <button
                  onClick={addSampleDocuments}
                  className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded flex items-center justify-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  Load Sample Docs
                </button>

                <button
                  onClick={addCustomDocument}
                  className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded flex items-center justify-center gap-2"
                >
                  <Plus className="w-4 h-4" />
                  Add Custom Doc
                </button>

                {documents.length > 0 && (
                  <button
                    onClick={clearAll}
                    className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
                  >
                    Clear All
                  </button>
                )}
              </div>

              {documents.length > 0 && (
                <div className="mt-4 space-y-2">
                  {documents.map(doc => (
                    <div key={doc.id} className="p-3 bg-gray-700/50 rounded border border-gray-600">
                      <div className="text-sm font-medium truncate">{doc.metadata.source}</div>
                      <div className="text-xs text-gray-400 mt-1">
                        {doc.chunks.length} chunks
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Chunking Config */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Chunking</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Chunk Size: {chunkSize} chars
                  </label>
                  <input
                    type="range"
                    min="100"
                    max="1000"
                    step="50"
                    value={chunkSize}
                    onChange={(e) => setChunkSize(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Overlap: {chunkOverlap} chars
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="200"
                    step="10"
                    value={chunkOverlap}
                    onChange={(e) => setChunkOverlap(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            {/* Model Config */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Models</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Embedding Model</label>
                  <select
                    value={embeddingModel}
                    onChange={(e) => setEmbeddingModel(e.target.value as EmbeddingModel)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  >
                    <option value="openai">OpenAI Ada-002</option>
                    <option value="huggingface">HuggingFace</option>
                    <option value="cohere">Cohere</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Vector Store</label>
                  <select
                    value={vectorStore}
                    onChange={(e) => setVectorStore(e.target.value as VectorStore)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                  >
                    <option value="faiss">FAISS</option>
                    <option value="chroma">Chroma</option>
                    <option value="pinecone">Pinecone</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Top K Results: {topK}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Query & Results */}
          <div className="lg:col-span-2 space-y-4">
            {/* Query Input */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Search className="w-5 h-5" />
                Query
              </h3>

              <div className="space-y-4">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your question...

Example: What is LangChain and how does it work?"
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-600 rounded-lg"
                  rows={3}
                />

                <button
                  onClick={performRetrieval}
                  disabled={!query.trim() || documents.length === 0 || processing}
                  className="w-full px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 rounded-lg font-medium flex items-center justify-center gap-2"
                >
                  <Zap className="w-5 h-5" />
                  {processing ? 'Processing...' : 'Run RAG Pipeline'}
                </button>
              </div>
            </div>

            {/* Retrieval Results */}
            {retrievalResults.length > 0 && (
              <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <Database className="w-5 h-5" />
                  Retrieved Chunks
                </h3>

                <div className="space-y-3">
                  {retrievalResults.map((result, idx) => (
                    <div
                      key={idx}
                      className="p-4 bg-gray-900 border border-gray-600 rounded-lg"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold text-amber-400">
                          Rank #{idx + 1}
                        </span>
                        <span className="text-sm text-green-400">
                          Score: {(result.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-sm text-gray-300 mb-2">{result.chunk}</div>
                      <div className="text-xs text-gray-500">
                        Source: {result.source}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Generated Answer */}
            {generatedAnswer && (
              <div className="bg-green-900/20 backdrop-blur border border-green-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  Generated Answer
                </h3>

                <div className="bg-gray-900 rounded-lg border border-gray-600 p-4">
                  <pre className="whitespace-pre-wrap text-sm text-gray-200">
                    {generatedAnswer}
                  </pre>
                </div>
              </div>
            )}

            {/* Pipeline Visualization */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Pipeline Steps</h3>

              <div className="space-y-3">
                {[
                  { icon: '📄', label: 'Document Loading', desc: `${documents.length} documents loaded` },
                  { icon: '✂️', label: 'Text Chunking', desc: `${chunkSize} chars, ${chunkOverlap} overlap` },
                  { icon: '🔢', label: 'Embedding', desc: `Using ${embeddingModel} model` },
                  { icon: '💾', label: 'Vector Storage', desc: `Stored in ${vectorStore}` },
                  { icon: '🔍', label: 'Retrieval', desc: `Top ${topK} results` },
                  { icon: '✨', label: 'Generation', desc: 'LLM generates final answer' }
                ].map((step, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-4 p-3 bg-gray-700/50 rounded-lg"
                  >
                    <div className="text-2xl">{step.icon}</div>
                    <div className="flex-1">
                      <div className="font-medium">{step.label}</div>
                      <div className="text-sm text-gray-400">{step.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
