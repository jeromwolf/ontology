'use client'

import { useState, useEffect } from 'react'
import { 
  Activity, AlertCircle, CheckCircle, Clock, Database, 
  FileText, GitBranch, RefreshCw, Search, Settings,
  TrendingUp, AlertTriangle, Zap, Shield, Brain, Eye
} from 'lucide-react'
import Link from 'next/link'

interface ModuleStatus {
  id: string
  name: string
  lastUpdate: Date
  version: string
  accuracyScore: number
  outdatedChapters: number
  brokenLinks: number
  deprecatedCode: number
  simulatorHealth: 'healthy' | 'warning' | 'critical'
  updateFrequency: 'daily' | 'weekly' | 'monthly'
  nextUpdate: Date
}

interface ValidationIssue {
  id: string
  moduleId: string
  type: 'outdated' | 'broken_link' | 'deprecated_code' | 'missing_simulator'
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  location: string
  suggestedFix?: string
  source?: string
}

interface ContentUpdate {
  id: string
  moduleId: string
  type: 'content' | 'simulator' | 'example' | 'reference'
  title: string
  description: string
  source: string
  confidence: number
  status: 'pending' | 'reviewing' | 'approved' | 'applied'
  createdAt: Date
}

export default function ContentManagerDashboard() {
  const [modules, setModules] = useState<ModuleStatus[]>([])
  const [validationIssues, setValidationIssues] = useState<ValidationIssue[]>([])
  const [contentUpdates, setContentUpdates] = useState<ContentUpdate[]>([])
  const [selectedModule, setSelectedModule] = useState<string | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [isUpdating, setIsUpdating] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'validation' | 'updates' | 'settings'>('overview')

  useEffect(() => {
    loadModuleStatuses()
    loadValidationIssues()
    loadContentUpdates()
  }, [])

  const loadModuleStatuses = async () => {
    try {
      const response = await fetch('/api/content-manager/modules')
      const data = await response.json()
      setModules(data)
    } catch (error) {
      console.error('Failed to load module statuses:', error)
      // Load mock data for demonstration
      setModules(getMockModuleStatuses())
    }
  }

  const loadValidationIssues = async () => {
    try {
      const response = await fetch('/api/content-manager/validation/issues')
      const data = await response.json()
      setValidationIssues(data)
    } catch (error) {
      console.error('Failed to load validation issues:', error)
      setValidationIssues(getMockValidationIssues())
    }
  }

  const loadContentUpdates = async () => {
    try {
      const response = await fetch('/api/content-manager/updates')
      const data = await response.json()
      setContentUpdates(data)
    } catch (error) {
      console.error('Failed to load content updates:', error)
      setContentUpdates(getMockContentUpdates())
    }
  }

  const runValidation = async (moduleId?: string) => {
    setIsValidating(true)
    try {
      const endpoint = moduleId 
        ? `/api/content-manager/validation/run?module=${moduleId}`
        : '/api/content-manager/validation/run'
      
      const response = await fetch(endpoint, { method: 'POST' })
      const newIssues = await response.json()
      setValidationIssues(newIssues)
    } catch (error) {
      console.error('Validation failed:', error)
    } finally {
      setIsValidating(false)
    }
  }

  const checkForUpdates = async (moduleId?: string) => {
    setIsUpdating(true)
    try {
      const endpoint = moduleId
        ? `/api/content-manager/updates/check?module=${moduleId}`
        : '/api/content-manager/updates/check'
      
      const response = await fetch(endpoint, { method: 'POST' })
      const updates = await response.json()
      setContentUpdates(updates)
    } catch (error) {
      console.error('Update check failed:', error)
    } finally {
      setIsUpdating(false)
    }
  }

  const applyUpdate = async (updateId: string) => {
    try {
      await fetch(`/api/content-manager/updates/${updateId}/apply`, { 
        method: 'POST' 
      })
      
      // Refresh updates
      await loadContentUpdates()
      await loadModuleStatuses()
    } catch (error) {
      console.error('Failed to apply update:', error)
    }
  }

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'text-green-500'
      case 'warning': return 'text-yellow-500'
      case 'critical': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'bg-blue-100 text-blue-800'
      case 'medium': return 'bg-yellow-100 text-yellow-800'
      case 'high': return 'bg-orange-100 text-orange-800'
      case 'critical': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-gray-100 text-gray-800'
      case 'reviewing': return 'bg-blue-100 text-blue-800'
      case 'approved': return 'bg-green-100 text-green-800'
      case 'applied': return 'bg-purple-100 text-purple-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Content Management System
              </h1>
              <p className="mt-2 text-gray-600 dark:text-gray-400">
                Monitor and update all KSS module content in real-time
              </p>
            </div>
            <div className="flex gap-3">
              <Link
                href="/modules/content-manager/review"
                className="px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 flex items-center gap-2"
              >
                <Eye className="w-4 h-4" />
                업데이트 검토
              </Link>
              <button
                onClick={() => runValidation()}
                disabled={isValidating}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 flex items-center gap-2"
              >
                {isValidating ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Shield className="w-4 h-4" />
                )}
                Validate All
              </button>
              <button
                onClick={() => checkForUpdates()}
                disabled={isUpdating}
                className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 flex items-center gap-2"
              >
                {isUpdating ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
                Check Updates
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="mt-6 flex gap-4 border-b dark:border-gray-700">
            {(['overview', 'validation', 'updates', 'settings'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`pb-3 px-1 font-medium capitalize transition-colors ${
                  activeTab === tab
                    ? 'text-blue-500 border-b-2 border-blue-500'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Total Modules</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {modules.length}
                    </p>
                  </div>
                  <Database className="w-8 h-8 text-blue-500" />
                </div>
              </div>
              
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Validation Issues</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {validationIssues.length}
                    </p>
                  </div>
                  <AlertCircle className="w-8 h-8 text-yellow-500" />
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Pending Updates</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {contentUpdates.filter(u => u.status === 'pending').length}
                    </p>
                  </div>
                  <RefreshCw className="w-8 h-8 text-green-500" />
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Avg Accuracy</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {modules.length > 0 
                        ? Math.round(modules.reduce((acc, m) => acc + m.accuracyScore, 0) / modules.length) 
                        : 0}%
                    </p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-purple-500" />
                </div>
              </div>
            </div>

            {/* Module Status Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {modules.map((module) => (
                <div
                  key={module.id}
                  className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow hover:shadow-lg transition-shadow cursor-pointer"
                  onClick={() => setSelectedModule(module.id)}
                >
                  <div className="flex justify-between items-start mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      {module.name}
                    </h3>
                    <Activity className={`w-5 h-5 ${getHealthColor(module.simulatorHealth)}`} />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Version</span>
                      <span className="text-gray-900 dark:text-white">{module.version}</span>
                    </div>
                    
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Accuracy</span>
                      <span className={`font-medium ${
                        module.accuracyScore >= 90 ? 'text-green-500' :
                        module.accuracyScore >= 70 ? 'text-yellow-500' :
                        'text-red-500'
                      }`}>
                        {module.accuracyScore}%
                      </span>
                    </div>

                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600 dark:text-gray-400">Last Update</span>
                      <span className="text-gray-900 dark:text-white">
                        {new Date(module.lastUpdate).toLocaleDateString()}
                      </span>
                    </div>

                    {/* Issues Summary */}
                    <div className="pt-3 mt-3 border-t dark:border-gray-700">
                      <div className="flex gap-3 text-xs">
                        {module.outdatedChapters > 0 && (
                          <span className="flex items-center gap-1 text-yellow-600">
                            <FileText className="w-3 h-3" />
                            {module.outdatedChapters} outdated
                          </span>
                        )}
                        {module.brokenLinks > 0 && (
                          <span className="flex items-center gap-1 text-red-600">
                            <AlertTriangle className="w-3 h-3" />
                            {module.brokenLinks} broken
                          </span>
                        )}
                        {module.deprecatedCode > 0 && (
                          <span className="flex items-center gap-1 text-orange-600">
                            <GitBranch className="w-3 h-3" />
                            {module.deprecatedCode} deprecated
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-2 pt-3">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          runValidation(module.id)
                        }}
                        className="flex-1 px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                      >
                        Validate
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          checkForUpdates(module.id)
                        }}
                        className="flex-1 px-3 py-1 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200"
                      >
                        Check Updates
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'validation' && (
          <div className="space-y-6">
            {/* Validation Issues List */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
              <div className="p-6 border-b dark:border-gray-700">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Validation Issues ({validationIssues.length})
                </h2>
              </div>
              
              <div className="divide-y dark:divide-gray-700">
                {validationIssues.map((issue) => (
                  <div key={issue.id} className="p-6 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className={`px-2 py-1 text-xs font-medium rounded ${getSeverityColor(issue.severity)}`}>
                            {issue.severity}
                          </span>
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {issue.moduleId}
                          </span>
                          <span className="text-sm text-gray-500">
                            {issue.location}
                          </span>
                        </div>
                        
                        <p className="text-gray-900 dark:text-white mb-2">
                          {issue.description}
                        </p>
                        
                        {issue.suggestedFix && (
                          <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <p className="text-sm text-blue-800 dark:text-blue-300">
                              <strong>Suggested Fix:</strong> {issue.suggestedFix}
                            </p>
                            {issue.source && (
                              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                                Source: {issue.source}
                              </p>
                            )}
                          </div>
                        )}
                      </div>
                      
                      <button className="ml-4 px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600">
                        Fix
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'updates' && (
          <div className="space-y-6">
            {/* Content Updates List */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
              <div className="p-6 border-b dark:border-gray-700">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Content Updates ({contentUpdates.length})
                </h2>
              </div>
              
              <div className="divide-y dark:divide-gray-700">
                {contentUpdates.map((update) => (
                  <div key={update.id} className="p-6 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className={`px-2 py-1 text-xs font-medium rounded ${getStatusColor(update.status)}`}>
                            {update.status}
                          </span>
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {update.moduleId}
                          </span>
                          <span className="text-sm text-gray-500">
                            {update.type}
                          </span>
                          <div className="flex items-center gap-1">
                            <Zap className="w-3 h-3 text-yellow-500" />
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              {update.confidence}% confidence
                            </span>
                          </div>
                        </div>
                        
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                          {update.title}
                        </h3>
                        
                        <p className="text-gray-600 dark:text-gray-400 mb-2">
                          {update.description}
                        </p>
                        
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <span>Source: {update.source}</span>
                          <span>{new Date(update.createdAt).toLocaleDateString()}</span>
                        </div>
                      </div>
                      
                      {update.status === 'pending' && (
                        <div className="flex gap-2 ml-4">
                          <button className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600">
                            Review
                          </button>
                          <button 
                            onClick={() => applyUpdate(update.id)}
                            className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                          >
                            Apply
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="space-y-6">
            <UpdateStrategySettings />
          </div>
        )}
      </div>
    </div>
  )
}

// Settings Component
function UpdateStrategySettings() {
  const strategies = [
    {
      module: 'Stock Analysis',
      frequency: 'daily',
      sources: ['Bloomberg', 'Reuters', 'Yahoo Finance'],
      criticalMetrics: ['Market Data', 'Company Earnings', 'Economic Indicators']
    },
    {
      module: 'LLM',
      frequency: 'weekly',
      sources: ['ArXiv', 'Hugging Face', 'OpenAI Blog'],
      criticalMetrics: ['Model Releases', 'Benchmarks', 'Research Papers']
    },
    {
      module: 'System Design',
      frequency: 'monthly',
      sources: ['High Scalability', 'AWS Blog', 'Engineering Blogs'],
      criticalMetrics: ['Architecture Patterns', 'Best Practices', 'Case Studies']
    },
    {
      module: 'Medical AI',
      frequency: 'weekly',
      sources: ['PubMed', 'Nature Medicine', 'FDA Updates'],
      criticalMetrics: ['Clinical Trials', 'Regulatory Changes', 'Research Breakthroughs']
    }
  ]

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div className="p-6 border-b dark:border-gray-700">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Module Update Strategies
        </h2>
      </div>
      
      <div className="p-6">
        <div className="space-y-6">
          {strategies.map((strategy, index) => (
            <div key={index} className="border dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  {strategy.module}
                </h3>
                <select 
                  defaultValue={strategy.frequency}
                  className="px-3 py-1 border dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="realtime">Real-time</option>
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Monitor Sources
                  </p>
                  <div className="space-y-1">
                    {strategy.sources.map((source, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <CheckCircle className="w-3 h-3 text-green-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {source}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Critical Metrics
                  </p>
                  <div className="space-y-1">
                    {strategy.criticalMetrics.map((metric, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <Brain className="w-3 h-3 text-blue-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {metric}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Mock data functions
function getMockModuleStatuses(): ModuleStatus[] {
  return [
    {
      id: 'stock-analysis',
      name: 'Stock Analysis',
      lastUpdate: new Date('2025-08-01'),
      version: '2.3.1',
      accuracyScore: 94,
      outdatedChapters: 2,
      brokenLinks: 1,
      deprecatedCode: 0,
      simulatorHealth: 'healthy',
      updateFrequency: 'daily',
      nextUpdate: new Date('2025-08-05')
    },
    {
      id: 'llm',
      name: 'Large Language Models',
      lastUpdate: new Date('2025-07-28'),
      version: '1.8.0',
      accuracyScore: 87,
      outdatedChapters: 3,
      brokenLinks: 2,
      deprecatedCode: 4,
      simulatorHealth: 'warning',
      updateFrequency: 'weekly',
      nextUpdate: new Date('2025-08-11')
    },
    {
      id: 'system-design',
      name: 'System Design',
      lastUpdate: new Date('2025-08-03'),
      version: '1.0.0',
      accuracyScore: 98,
      outdatedChapters: 0,
      brokenLinks: 0,
      deprecatedCode: 0,
      simulatorHealth: 'healthy',
      updateFrequency: 'monthly',
      nextUpdate: new Date('2025-09-03')
    },
    {
      id: 'medical-ai',
      name: 'Medical AI',
      lastUpdate: new Date('2025-07-15'),
      version: '1.5.2',
      accuracyScore: 72,
      outdatedChapters: 5,
      brokenLinks: 3,
      deprecatedCode: 6,
      simulatorHealth: 'critical',
      updateFrequency: 'weekly',
      nextUpdate: new Date('2025-08-08')
    },
    {
      id: 'ontology',
      name: 'Ontology & Semantic Web',
      lastUpdate: new Date('2025-07-20'),
      version: '2.1.0',
      accuracyScore: 91,
      outdatedChapters: 1,
      brokenLinks: 0,
      deprecatedCode: 2,
      simulatorHealth: 'healthy',
      updateFrequency: 'monthly',
      nextUpdate: new Date('2025-08-20')
    },
    {
      id: 'rag',
      name: 'RAG Systems',
      lastUpdate: new Date('2025-07-30'),
      version: '1.2.0',
      accuracyScore: 85,
      outdatedChapters: 2,
      brokenLinks: 1,
      deprecatedCode: 3,
      simulatorHealth: 'warning',
      updateFrequency: 'weekly',
      nextUpdate: new Date('2025-08-06')
    }
  ]
}

function getMockValidationIssues(): ValidationIssue[] {
  return [
    {
      id: '1',
      moduleId: 'llm',
      type: 'deprecated_code',
      severity: 'high',
      description: 'Using deprecated OpenAI API v3 endpoints in Chapter 4',
      location: 'Chapter 4 - API Integration',
      suggestedFix: 'Update to OpenAI API v4 with new authentication method',
      source: 'OpenAI Documentation'
    },
    {
      id: '2',
      moduleId: 'medical-ai',
      type: 'outdated',
      severity: 'critical',
      description: 'FDA regulations updated in July 2025 not reflected in Chapter 8',
      location: 'Chapter 8 - Regulatory Compliance',
      suggestedFix: 'Update FDA 510(k) submission requirements and AI/ML device guidelines',
      source: 'FDA.gov Official Updates'
    },
    {
      id: '3',
      moduleId: 'stock-analysis',
      type: 'broken_link',
      severity: 'medium',
      description: 'Yahoo Finance API endpoint changed',
      location: 'Chapter 3 - Real-time Data Fetching',
      suggestedFix: 'Update to new Yahoo Finance v8 API endpoint',
      source: 'Yahoo Developer Portal'
    },
    {
      id: '4',
      moduleId: 'rag',
      type: 'missing_simulator',
      severity: 'low',
      description: 'No simulator for hybrid search strategies',
      location: 'Chapter 6 - Hybrid Search',
      suggestedFix: 'Create interactive simulator combining vector and keyword search',
      source: 'User Feedback'
    }
  ]
}

function getMockContentUpdates(): ContentUpdate[] {
  return [
    {
      id: '1',
      moduleId: 'llm',
      type: 'content',
      title: 'Claude 3.5 Sonnet Performance Benchmarks',
      description: 'New benchmarks show Claude 3.5 Sonnet outperforming GPT-4 in coding tasks',
      source: 'Anthropic Blog',
      confidence: 95,
      status: 'pending',
      createdAt: new Date('2025-08-03')
    },
    {
      id: '2',
      moduleId: 'stock-analysis',
      type: 'reference',
      title: 'Fed Interest Rate Decision Impact',
      description: 'Federal Reserve announces rate cut, affecting market analysis strategies',
      source: 'Federal Reserve Press Release',
      confidence: 100,
      status: 'reviewing',
      createdAt: new Date('2025-08-02')
    },
    {
      id: '3',
      moduleId: 'medical-ai',
      type: 'simulator',
      title: 'New X-ray Analysis Model Available',
      description: 'Google releases MedPaLM-3 for improved medical imaging analysis',
      source: 'Google Research',
      confidence: 88,
      status: 'approved',
      createdAt: new Date('2025-08-01')
    },
    {
      id: '4',
      moduleId: 'system-design',
      type: 'example',
      title: 'Netflix Architecture Update 2025',
      description: 'Netflix shares new microservices patterns for 1 billion users scale',
      source: 'Netflix Tech Blog',
      confidence: 92,
      status: 'pending',
      createdAt: new Date('2025-08-04')
    }
  ]
}