'use client'

import { useState } from 'react'
import { Database, Plus, RefreshCw, CheckCircle } from 'lucide-react'

interface Feature {
  id: string
  name: string
  type: 'numerical' | 'categorical' | 'embedding'
  source: string
  freshness: number
  version: string
}

export default function FeatureStore() {
  const [features, setFeatures] = useState<Feature[]>([
    { id: '1', name: 'user_age', type: 'numerical', source: 'users_db', freshness: 95, version: 'v1.2' },
    { id: '2', name: 'user_category', type: 'categorical', source: 'users_db', freshness: 98, version: 'v1.2' },
    { id: '3', name: 'item_embedding', type: 'embedding', source: 'ml_pipeline', freshness: 87, version: 'v2.0' },
  ])

  const [newFeature, setNewFeature] = useState({
    name: '',
    type: 'numerical' as Feature['type'],
    source: 'users_db',
  })

  const addFeature = () => {
    if (!newFeature.name) return

    const feature: Feature = {
      id: Date.now().toString(),
      name: newFeature.name,
      type: newFeature.type,
      source: newFeature.source,
      freshness: Math.floor(Math.random() * 20) + 80,
      version: 'v1.0',
    }

    setFeatures([...features, feature])
    setNewFeature({ name: '', type: 'numerical', source: 'users_db' })
  }

  const refreshFeature = (id: string) => {
    setFeatures(features.map(f =>
      f.id === id ? { ...f, freshness: Math.min(100, f.freshness + Math.floor(Math.random() * 10)) } : f
    ))
  }

  const getTypeColor = (type: Feature['type']) => {
    switch (type) {
      case 'numerical': return 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400'
      case 'categorical': return 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400'
      case 'embedding': return 'bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-400'
    }
  }

  const getFreshnessColor = (freshness: number) => {
    if (freshness >= 95) return 'text-green-600'
    if (freshness >= 85) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 rounded-lg p-6 border-l-4 border-slate-600">
        <h3 className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          Feature Store 시뮬레이터
        </h3>
        <p className="text-gray-700 dark:text-gray-300">
          ML 피처를 중앙화하고 버전 관리하며 실시간으로 제공합니다
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
            <Plus className="w-5 h-5 text-slate-600" />
            Add New Feature
          </h4>

          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                Feature Name
              </label>
              <input
                type="text"
                value={newFeature.name}
                onChange={(e) => setNewFeature({ ...newFeature, name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder="e.g., 'user_clicks_7d'"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                Type
              </label>
              <select
                value={newFeature.type}
                onChange={(e) => setNewFeature({ ...newFeature, type: e.target.value as Feature['type'] })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="numerical">Numerical</option>
                <option value="categorical">Categorical</option>
                <option value="embedding">Embedding</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                Source
              </label>
              <select
                value={newFeature.source}
                onChange={(e) => setNewFeature({ ...newFeature, source: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="users_db">Users DB</option>
                <option value="events_db">Events DB</option>
                <option value="ml_pipeline">ML Pipeline</option>
                <option value="external_api">External API</option>
              </select>
            </div>

            <button
              onClick={addFeature}
              disabled={!newFeature.name}
              className="w-full px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              Add Feature
            </button>
          </div>

          <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Total Features: <span className="font-bold text-gray-900 dark:text-white">{features.length}</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Average Freshness: <span className="font-bold text-gray-900 dark:text-white">
                {(features.reduce((sum, f) => sum + f.freshness, 0) / features.length).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h4 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
              <Database className="w-5 h-5 text-slate-600" />
              Feature Registry
            </h4>

            {features.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                No features registered
              </div>
            ) : (
              <div className="space-y-3">
                {features.map((feature) => (
                  <div
                    key={feature.id}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-slate-300 dark:hover:border-slate-600 transition"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-mono font-semibold text-gray-900 dark:text-white">
                            {feature.name}
                          </span>
                          <span className={`text-xs px-2 py-0.5 rounded ${getTypeColor(feature.type)}`}>
                            {feature.type}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          Source: {feature.source} | Version: {feature.version}
                        </div>
                      </div>
                      <button
                        onClick={() => refreshFeature(feature.id)}
                        className="p-2 hover:bg-slate-50 dark:hover:bg-slate-900/20 rounded-lg transition"
                        title="Refresh feature"
                      >
                        <RefreshCw className="w-4 h-4 text-slate-600" />
                      </button>
                    </div>

                    <div className="flex items-center gap-2">
                      <div className="flex-1">
                        <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                          <span>Data Freshness</span>
                          <span className={`font-semibold ${getFreshnessColor(feature.freshness)}`}>
                            {feature.freshness}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full transition-all ${
                              feature.freshness >= 95 ? 'bg-green-500' :
                              feature.freshness >= 85 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${feature.freshness}%` }}
                          />
                        </div>
                      </div>
                      {feature.freshness >= 95 && (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-4 bg-slate-50 dark:bg-slate-900/20 rounded-lg p-4 border-l-4 border-slate-600">
            <h5 className="font-semibold mb-2 text-gray-900 dark:text-white">
              Feature Store Benefits
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>일관성</strong>: 모든 모델이 동일한 피처 정의 사용</li>
              <li>• <strong>재사용성</strong>: 한 번 정의한 피처를 여러 모델에서 활용</li>
              <li>• <strong>실시간성</strong>: 온라인/오프라인 피처 통합 제공</li>
              <li>• <strong>버전 관리</strong>: 피처 변경 이력 추적</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
