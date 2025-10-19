/**
 * Papers List Page - 사용자용 논문 목록 페이지
 * 최신 ArXiv 논문 요약을 최신순으로 표시
 */

'use client'

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface Paper {
  id: string
  arxivId: string
  title: string
  authors: string[]
  publishedDate: string
  categories: string[]
  keywords: string[]
  relatedModules: string[]
  pdfUrl: string
  status: string
  createdAt: string
}

export default function PapersPage() {
  const searchParams = useSearchParams()
  const moduleParam = searchParams.get('module')

  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>(moduleParam || 'all')

  useEffect(() => {
    fetchPapers()
  }, [])

  // URL 파라미터가 변경되면 필터 업데이트
  useEffect(() => {
    if (moduleParam && moduleParam !== filter) {
      setFilter(moduleParam)
    }
  }, [moduleParam])

  const fetchPapers = async () => {
    try {
      const res = await fetch('/api/arxiv-monitor/papers?limit=50')
      const data = await res.json()
      if (data.success && data.data?.papers) {
        // 최신순 정렬 (publishedDate 기준)
        const sorted = data.data.papers.sort(
          (a: Paper, b: Paper) =>
            new Date(b.publishedDate).getTime() - new Date(a.publishedDate).getTime()
        )
        setPapers(sorted)
      }
    } catch (error) {
      console.error('Failed to fetch papers:', error)
    } finally {
      setLoading(false)
    }
  }

  const filteredPapers = filter === 'all'
    ? papers
    : papers.filter(p => p.relatedModules.includes(filter))

  const allModules = Array.from(
    new Set(papers.flatMap(p => p.relatedModules))
  ).sort()

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-slate-600 dark:text-slate-400">논문을 불러오는 중...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
                📚 최신 AI 연구 논문
              </h1>
              <p className="text-slate-600 dark:text-slate-400">
                ArXiv에서 엄선한 최신 AI/ML 논문 요약을 확인하세요
              </p>
            </div>
            <Link
              href="/admin/arxiv-monitor"
              className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-800 transition-colors text-sm"
            >
              관리자 대시보드
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow-sm">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {papers.length}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">전체 논문</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow-sm">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {papers.filter(p => p.status === 'MDX_GENERATED').length}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">요약 완료</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow-sm">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {allModules.length}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">관련 모듈</div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow-sm">
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {papers[0] ? new Date(papers[0].publishedDate).toLocaleDateString('ko-KR') : '-'}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">최신 논문</div>
            </div>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                filter === 'all'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              전체 ({papers.length})
            </button>
            {allModules.map(module => (
              <button
                key={module}
                onClick={() => setFilter(module)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === module
                    ? 'bg-blue-600 text-white'
                    : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                }`}
              >
                {module} ({papers.filter(p => p.relatedModules.includes(module)).length})
              </button>
            ))}
          </div>
        </div>

        {/* Papers Grid */}
        {filteredPapers.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-slate-600 dark:text-slate-400 text-lg">
              아직 논문이 없습니다.
            </p>
          </div>
        ) : (
          <div className="grid gap-6">
            {filteredPapers.map(paper => (
              <Card key={paper.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <CardTitle className="text-xl mb-2 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                        <Link href={`/papers/${paper.arxivId}`}>
                          {paper.title}
                        </Link>
                      </CardTitle>
                      <CardDescription className="text-sm">
                        {paper.authors.slice(0, 3).join(', ')}
                        {paper.authors.length > 3 && ` 외 ${paper.authors.length - 3}명`}
                      </CardDescription>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <Badge variant={paper.status === 'MDX_GENERATED' ? 'default' : 'secondary'}>
                        {paper.status === 'MDX_GENERATED' ? '✅ 요약 완료' : '⏳ 처리 중'}
                      </Badge>
                      <span className="text-sm text-slate-500">
                        {new Date(paper.publishedDate).toLocaleDateString('ko-KR')}
                      </span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Categories */}
                  <div className="flex flex-wrap gap-2 mb-3">
                    {paper.categories.map(cat => (
                      <span
                        key={cat}
                        className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs rounded"
                      >
                        {cat}
                      </span>
                    ))}
                  </div>

                  {/* Keywords */}
                  {paper.keywords.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-3">
                      {paper.keywords.map(kw => (
                        <span
                          key={kw}
                          className="px-2 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs rounded"
                        >
                          🔑 {kw}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Related Modules */}
                  {paper.relatedModules.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-4">
                      {paper.relatedModules.map(module => (
                        <Link
                          key={module}
                          href={`/modules/${module}`}
                          className="px-2 py-1 bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-xs rounded hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
                        >
                          📚 {module}
                        </Link>
                      ))}
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex gap-3 pt-3 border-t dark:border-slate-700">
                    {paper.status === 'MDX_GENERATED' ? (
                      <Link
                        href={`/papers/${paper.arxivId}`}
                        className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-center text-sm font-medium"
                      >
                        📄 한글 요약 보기
                      </Link>
                    ) : (
                      <button
                        disabled
                        className="flex-1 px-4 py-2 bg-slate-400 text-white rounded-lg cursor-not-allowed text-center text-sm font-medium opacity-60"
                      >
                        ⏳ 요약 준비 중...
                      </button>
                    )}
                    <a
                      href={paper.pdfUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex-1 px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors text-center text-sm font-medium"
                    >
                      📑 원문 (PDF)
                    </a>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
