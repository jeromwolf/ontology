'use client'

import { useState, useEffect } from 'react'
import { 
  ArrowLeft, Check, X, Clock, Eye, GitCompare, 
  AlertCircle, FileText, Zap, Code, Link2,
  ChevronDown, ChevronUp, MessageSquare, Shield
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

interface UpdateReview {
  id: string
  moduleId: string
  moduleName: string
  chapter?: string
  type: 'content' | 'simulator' | 'example' | 'reference' | 'correction'
  title: string
  description: string
  oldContent?: string
  newContent?: string
  changes?: {
    additions: number
    deletions: number
    files: string[]
  }
  source: string
  sourceUrl?: string
  confidence: number
  priority: 'low' | 'medium' | 'high' | 'critical'
  status: 'pending' | 'reviewed' | 'approved' | 'rejected' | 'applied'
  createdAt: string
  reviewNotes?: string
}

export default function ContentReviewDashboard() {
  const router = useRouter()
  const [updates, setUpdates] = useState<UpdateReview[]>([])
  const [selectedUpdate, setSelectedUpdate] = useState<UpdateReview | null>(null)
  const [filter, setFilter] = useState<'all' | 'pending' | 'reviewed' | 'approved'>('pending')
  const [loading, setLoading] = useState(false)
  const [reviewNotes, setReviewNotes] = useState('')
  const [showDiff, setShowDiff] = useState(false)

  useEffect(() => {
    fetchUpdates()
  }, [filter])

  const fetchUpdates = async () => {
    setLoading(true)
    try {
      const status = filter === 'all' ? '' : `?status=${filter}`
      const response = await fetch(`/api/content-manager/updates/pending${status}`)
      const data = await response.json()
      setUpdates(data.updates || [])
    } catch (error) {
      console.error('Failed to fetch updates:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleStatusChange = async (updateId: string, newStatus: string, notes?: string) => {
    try {
      const response = await fetch('/api/content-manager/updates/pending', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          updateId,
          status: newStatus,
          reviewNotes: notes || reviewNotes
        })
      })

      if (response.ok) {
        const data = await response.json()
        alert(data.message)
        fetchUpdates()
        setSelectedUpdate(null)
        setReviewNotes('')
      }
    } catch (error) {
      console.error('Failed to update status:', error)
      alert('상태 변경에 실패했습니다.')
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
      case 'high': return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'
      case 'medium': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
      default: return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'content': return <FileText className="w-4 h-4" />
      case 'simulator': return <Zap className="w-4 h-4" />
      case 'example': return <Code className="w-4 h-4" />
      case 'reference': return <Link2 className="w-4 h-4" />
      default: return <AlertCircle className="w-4 h-4" />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <Link href="/modules/content-manager" className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white mb-4">
            <ArrowLeft className="w-4 h-4" />
            콘텐츠 관리자로 돌아가기
          </Link>
          
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            콘텐츠 업데이트 검토
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            AI가 제안한 업데이트를 검토하고 승인하세요
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="flex gap-2 mb-6">
          {(['all', 'pending', 'reviewed', 'approved'] as const).map((status) => (
            <button
              key={status}
              onClick={() => setFilter(status)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                filter === status
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {status === 'all' ? '전체' :
               status === 'pending' ? '대기 중' :
               status === 'reviewed' ? '검토됨' : '승인됨'}
            </button>
          ))}
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Updates List */}
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              업데이트 목록 ({updates.length})
            </h2>
            
            {loading ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
              </div>
            ) : updates.length === 0 ? (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-8 text-center">
                <p className="text-gray-500 dark:text-gray-400">
                  검토할 업데이트가 없습니다
                </p>
              </div>
            ) : (
              updates.map((update) => (
                <div
                  key={update.id}
                  onClick={() => setSelectedUpdate(update)}
                  className={`bg-white dark:bg-gray-800 rounded-lg p-4 cursor-pointer transition-all hover:shadow-lg ${
                    selectedUpdate?.id === update.id ? 'ring-2 ring-indigo-500' : ''
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getTypeIcon(update.type)}
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {update.title}
                      </h3>
                    </div>
                    <span className={`px-2 py-1 text-xs font-medium rounded ${getPriorityColor(update.priority)}`}>
                      {update.priority}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    {update.moduleName} {update.chapter && `- ${update.chapter}`}
                  </p>
                  
                  <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2 mb-3">
                    {update.description}
                  </p>
                  
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-3">
                      <span className="text-gray-500">
                        신뢰도: {Math.round(update.confidence * 100)}%
                      </span>
                      {update.changes && (
                        <span className="text-gray-500">
                          +{update.changes.additions} -{update.changes.deletions}
                        </span>
                      )}
                    </div>
                    <span className="text-gray-500">
                      {new Date(update.createdAt).toLocaleDateString()}
                    </span>
                  </div>

                  {update.status !== 'pending' && (
                    <div className={`mt-2 pt-2 border-t dark:border-gray-700 text-xs ${
                      update.status === 'approved' ? 'text-green-600' :
                      update.status === 'rejected' ? 'text-red-600' :
                      'text-yellow-600'
                    }`}>
                      상태: {update.status === 'approved' ? '승인됨' :
                             update.status === 'rejected' ? '거절됨' :
                             update.status === 'reviewed' ? '검토됨' : '적용됨'}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          {/* Update Detail */}
          <div className="lg:sticky lg:top-6">
            {selectedUpdate ? (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  업데이트 상세 정보
                </h2>

                {/* Update Info */}
                <div className="space-y-4 mb-6">
                  <div>
                    <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                      제목
                    </h3>
                    <p className="text-gray-900 dark:text-white">{selectedUpdate.title}</p>
                  </div>

                  <div>
                    <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                      설명
                    </h3>
                    <p className="text-gray-700 dark:text-gray-300">{selectedUpdate.description}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                        모듈
                      </h3>
                      <p className="text-gray-900 dark:text-white">{selectedUpdate.moduleName}</p>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                        유형
                      </h3>
                      <p className="text-gray-900 dark:text-white">{selectedUpdate.type}</p>
                    </div>
                  </div>

                  {selectedUpdate.source && (
                    <div>
                      <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                        출처
                      </h3>
                      {selectedUpdate.sourceUrl ? (
                        <a 
                          href={selectedUpdate.sourceUrl} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-indigo-600 hover:underline"
                        >
                          {selectedUpdate.source}
                        </a>
                      ) : (
                        <p className="text-gray-900 dark:text-white">{selectedUpdate.source}</p>
                      )}
                    </div>
                  )}
                </div>

                {/* Content Diff */}
                {(selectedUpdate.oldContent || selectedUpdate.newContent) && (
                  <div className="mb-6">
                    <button
                      onClick={() => setShowDiff(!showDiff)}
                      className="flex items-center gap-2 text-sm font-medium text-indigo-600 hover:text-indigo-700 mb-3"
                    >
                      <GitCompare className="w-4 h-4" />
                      변경 사항 보기
                      {showDiff ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>

                    {showDiff && (
                      <div className="space-y-3">
                        {selectedUpdate.oldContent && (
                          <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                            <h4 className="text-xs font-medium text-red-700 dark:text-red-400 mb-2">이전 콘텐츠</h4>
                            <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                              {selectedUpdate.oldContent}
                            </pre>
                          </div>
                        )}
                        {selectedUpdate.newContent && (
                          <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                            <h4 className="text-xs font-medium text-green-700 dark:text-green-400 mb-2">새 콘텐츠</h4>
                            <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                              {selectedUpdate.newContent}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Review Notes */}
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    검토 노트
                  </label>
                  <textarea
                    value={reviewNotes}
                    onChange={(e) => setReviewNotes(e.target.value)}
                    placeholder="검토 의견을 입력하세요..."
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                             bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                             focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    rows={3}
                  />
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3">
                  {selectedUpdate.status === 'pending' && (
                    <>
                      <button
                        onClick={() => handleStatusChange(selectedUpdate.id, 'approved')}
                        className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 
                                 flex items-center justify-center gap-2"
                      >
                        <Check className="w-4 h-4" />
                        승인
                      </button>
                      <button
                        onClick={() => handleStatusChange(selectedUpdate.id, 'reviewed')}
                        className="flex-1 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 
                                 flex items-center justify-center gap-2"
                      >
                        <Eye className="w-4 h-4" />
                        검토 완료
                      </button>
                      <button
                        onClick={() => handleStatusChange(selectedUpdate.id, 'rejected')}
                        className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 
                                 flex items-center justify-center gap-2"
                      >
                        <X className="w-4 h-4" />
                        거절
                      </button>
                    </>
                  )}
                  
                  {selectedUpdate.status === 'reviewed' && (
                    <>
                      <button
                        onClick={() => handleStatusChange(selectedUpdate.id, 'approved')}
                        className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 
                                 flex items-center justify-center gap-2"
                      >
                        <Check className="w-4 h-4" />
                        승인
                      </button>
                      <button
                        onClick={() => handleStatusChange(selectedUpdate.id, 'rejected')}
                        className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 
                                 flex items-center justify-center gap-2"
                      >
                        <X className="w-4 h-4" />
                        거절
                      </button>
                    </>
                  )}
                  
                  {selectedUpdate.status === 'approved' && (
                    <button
                      onClick={() => handleStatusChange(selectedUpdate.id, 'applied')}
                      className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 
                               flex items-center justify-center gap-2"
                    >
                      <Shield className="w-4 h-4" />
                      적용하기
                    </button>
                  )}
                </div>

                {selectedUpdate.reviewNotes && (
                  <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <h4 className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                      이전 검토 노트
                    </h4>
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {selectedUpdate.reviewNotes}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-8 text-center">
                <MessageSquare className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-500 dark:text-gray-400">
                  왼쪽 목록에서 업데이트를 선택하세요
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}