'use client'

import React, { useState, useEffect } from 'react'
import { Calculator, DollarSign, TrendingDown, Info } from 'lucide-react'

interface ModelPricing {
  name: string
  inputCost: number // per 1M tokens
  outputCost: number // per 1M tokens
  contextWindow: number
  color: string
}

const MODELS: Record<string, ModelPricing> = {
  'gpt-3.5-turbo': {
    name: 'GPT-3.5 Turbo',
    inputCost: 0.50,
    outputCost: 1.50,
    contextWindow: 16385,
    color: '#10b981'
  },
  'gpt-4': {
    name: 'GPT-4',
    inputCost: 30.00,
    outputCost: 60.00,
    contextWindow: 8192,
    color: '#3b82f6'
  },
  'gpt-4-turbo': {
    name: 'GPT-4 Turbo',
    inputCost: 10.00,
    outputCost: 30.00,
    contextWindow: 128000,
    color: '#8b5cf6'
  },
  'claude-3-opus': {
    name: 'Claude 3 Opus',
    inputCost: 15.00,
    outputCost: 75.00,
    contextWindow: 200000,
    color: '#f59e0b'
  },
  'claude-3-sonnet': {
    name: 'Claude 3 Sonnet',
    inputCost: 3.00,
    outputCost: 15.00,
    contextWindow: 200000,
    color: '#ef4444'
  },
  'claude-3-haiku': {
    name: 'Claude 3 Haiku',
    inputCost: 0.25,
    outputCost: 1.25,
    contextWindow: 200000,
    color: '#06b6d4'
  }
}

export default function TokenCostCalculator() {
  const [selectedModel, setSelectedModel] = useState('gpt-3.5-turbo')
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [inputTokens, setInputTokens] = useState(0)
  const [outputTokens, setOutputTokens] = useState(0)
  const [inputCost, setInputCost] = useState(0)
  const [outputCost, setOutputCost] = useState(0)
  const [totalCost, setTotalCost] = useState(0)

  // Batch calculation
  const [batchRequests, setBatchRequests] = useState(1000)
  const [batchInputTokens, setBatchInputTokens] = useState(500)
  const [batchOutputTokens, setBatchOutputTokens] = useState(200)
  const [batchCost, setBatchCost] = useState(0)

  useEffect(() => {
    calculateCosts()
  }, [inputText, outputText, selectedModel])

  useEffect(() => {
    calculateBatchCost()
  }, [batchRequests, batchInputTokens, batchOutputTokens, selectedModel])

  const estimateTokens = (text: string): number => {
    // Simple estimation: ~1.3 tokens per word
    const words = text.trim().split(/\s+/).length
    return Math.ceil(words * 1.3)
  }

  const calculateCosts = () => {
    const model = MODELS[selectedModel]
    const inTokens = estimateTokens(inputText)
    const outTokens = estimateTokens(outputText)

    const inCost = (inTokens / 1_000_000) * model.inputCost
    const outCost = (outTokens / 1_000_000) * model.outputCost

    setInputTokens(inTokens)
    setOutputTokens(outTokens)
    setInputCost(inCost)
    setOutputCost(outCost)
    setTotalCost(inCost + outCost)
  }

  const calculateBatchCost = () => {
    const model = MODELS[selectedModel]
    const totalInputTokens = batchRequests * batchInputTokens
    const totalOutputTokens = batchRequests * batchOutputTokens

    const inCost = (totalInputTokens / 1_000_000) * model.inputCost
    const outCost = (totalOutputTokens / 1_000_000) * model.outputCost

    setBatchCost(inCost + outCost)
  }

  const getCheapestModel = (): string => {
    const costs = Object.entries(MODELS).map(([id, model]) => {
      const inCost = (inputTokens / 1_000_000) * model.inputCost
      const outCost = (outputTokens / 1_000_000) * model.outputCost
      return { id, cost: inCost + outCost }
    })

    costs.sort((a, b) => a.cost - b.cost)
    return costs[0].id
  }

  const model = MODELS[selectedModel]
  const cheapestModel = getCheapestModel()

  const sampleTexts = {
    short: 'What is LangChain?',
    medium: 'Explain how LangChain works and provide examples of its key components including chains, prompts, and memory management.',
    long: 'I need a comprehensive guide on building production-ready LLM applications using LangChain. Please cover the following topics: setting up the development environment, creating custom chains, implementing memory systems, integrating with vector databases, building agents with tools, and deploying to production. Include code examples and best practices for each section.'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            💰 Token & Cost Calculator
          </h1>
          <p className="text-gray-300 text-lg">
            Estimate token usage and API costs across different LLM providers.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Model Selection */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Select Model</h3>

              <div className="space-y-2">
                {Object.entries(MODELS).map(([id, modelInfo]) => (
                  <button
                    key={id}
                    onClick={() => setSelectedModel(id)}
                    className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-all ${
                      selectedModel === id
                        ? 'border-amber-500 bg-gray-700'
                        : 'border-gray-600 bg-gray-700/50 hover:bg-gray-600'
                    }`}
                  >
                    <div className="font-medium" style={{ color: modelInfo.color }}>
                      {modelInfo.name}
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      ${modelInfo.inputCost} / ${modelInfo.outputCost} per 1M tokens
                    </div>
                    <div className="text-xs text-gray-500">
                      {modelInfo.contextWindow.toLocaleString()} token context
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Samples */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Quick Samples</h3>

              <div className="space-y-2">
                {Object.entries(sampleTexts).map(([key, text]) => (
                  <button
                    key={key}
                    onClick={() => setInputText(text)}
                    className="w-full text-left px-3 py-2 bg-gray-700/50 hover:bg-gray-600 rounded text-sm"
                  >
                    <div className="font-medium capitalize">{key} Prompt</div>
                    <div className="text-xs text-gray-400 truncate">{text}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Current Selection Info */}
            <div className="bg-blue-900/20 backdrop-blur border border-blue-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-3" style={{ color: model.color }}>
                {model.name}
              </h3>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Input Cost</span>
                  <span className="font-mono">${model.inputCost}/1M</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Output Cost</span>
                  <span className="font-mono">${model.outputCost}/1M</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Context Window</span>
                  <span className="font-mono">{model.contextWindow.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Calculator */}
          <div className="lg:col-span-2 space-y-4">
            {/* Single Request Calculator */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5" />
                Single Request Calculator
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Input (Prompt)</label>
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter your input text..."
                    className="w-full px-4 py-3 bg-gray-900 border border-gray-600 rounded-lg font-mono text-sm"
                    rows={5}
                  />
                  <div className="text-xs text-gray-400 mt-1">
                    Estimated: {inputTokens} tokens
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Output (Response)</label>
                  <textarea
                    value={outputText}
                    onChange={(e) => setOutputText(e.target.value)}
                    placeholder="Enter expected output text..."
                    className="w-full px-4 py-3 bg-gray-900 border border-gray-600 rounded-lg font-mono text-sm"
                    rows={5}
                  />
                  <div className="text-xs text-gray-400 mt-1">
                    Estimated: {outputTokens} tokens
                  </div>
                </div>
              </div>
            </div>

            {/* Cost Breakdown */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <DollarSign className="w-5 h-5" />
                Cost Breakdown
              </h3>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-gray-900 rounded-lg p-4 text-center">
                  <div className="text-sm text-gray-400 mb-1">Input Cost</div>
                  <div className="text-2xl font-bold text-blue-500">
                    ${inputCost.toFixed(6)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {inputTokens} tokens
                  </div>
                </div>

                <div className="bg-gray-900 rounded-lg p-4 text-center">
                  <div className="text-sm text-gray-400 mb-1">Output Cost</div>
                  <div className="text-2xl font-bold text-green-500">
                    ${outputCost.toFixed(6)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {outputTokens} tokens
                  </div>
                </div>

                <div className="bg-gray-900 rounded-lg p-4 text-center border-2 border-amber-500">
                  <div className="text-sm text-gray-400 mb-1">Total Cost</div>
                  <div className="text-2xl font-bold text-amber-500">
                    ${totalCost.toFixed(6)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {inputTokens + outputTokens} tokens
                  </div>
                </div>
              </div>

              {cheapestModel !== selectedModel && totalCost > 0 && (
                <div className="bg-green-900/20 border border-green-700 rounded-lg p-4">
                  <div className="flex items-start gap-2">
                    <TrendingDown className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <div className="font-semibold text-green-400 mb-1">Cost Optimization Tip</div>
                      <div className="text-sm text-gray-300">
                        Switch to <strong>{MODELS[cheapestModel].name}</strong> to save{' '}
                        <strong>
                          ${(totalCost - ((inputTokens / 1_000_000) * MODELS[cheapestModel].inputCost + (outputTokens / 1_000_000) * MODELS[cheapestModel].outputCost)).toFixed(6)}
                        </strong>{' '}
                        per request
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Batch Calculator */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-4">Batch Estimation</h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Number of Requests
                  </label>
                  <input
                    type="number"
                    value={batchRequests}
                    onChange={(e) => setBatchRequests(parseInt(e.target.value) || 0)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Avg Input Tokens
                  </label>
                  <input
                    type="number"
                    value={batchInputTokens}
                    onChange={(e) => setBatchInputTokens(parseInt(e.target.value) || 0)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Avg Output Tokens
                  </label>
                  <input
                    type="number"
                    value={batchOutputTokens}
                    onChange={(e) => setBatchOutputTokens(parseInt(e.target.value) || 0)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded"
                  />
                </div>
              </div>

              <div className="bg-gradient-to-r from-amber-900/30 to-orange-900/30 border border-amber-700 rounded-lg p-6">
                <div className="text-center">
                  <div className="text-sm text-gray-400 mb-2">Estimated Batch Cost</div>
                  <div className="text-4xl font-bold text-amber-500 mb-2">
                    ${batchCost.toFixed(2)}
                  </div>
                  <div className="text-sm text-gray-400">
                    {(batchRequests * (batchInputTokens + batchOutputTokens)).toLocaleString()} total tokens
                  </div>
                </div>
              </div>
            </div>

            {/* Tips */}
            <div className="bg-blue-900/20 backdrop-blur border border-blue-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                <Info className="w-5 h-5" />
                Cost Optimization Tips
              </h3>

              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Use cheaper models (GPT-3.5, Claude Haiku) for simple tasks</li>
                <li>• Implement caching to avoid redundant API calls</li>
                <li>• Use shorter prompts when possible - every token counts!</li>
                <li>• Consider batch processing for bulk operations</li>
                <li>• Set max_tokens to limit output length</li>
                <li>• Monitor usage with LangSmith or similar tools</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
