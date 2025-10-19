/**
 * ModuleRelatedPapers - 모듈별 관련 논문 컴포넌트
 * 각 모듈 페이지에서 해당 모듈과 관련된 논문만 필터링하여 표시
 */

'use client'

import { useEffect, useState } from 'react'
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

interface ModuleRelatedPapersProps {
  moduleId: string
  maxPapers?: number
  showStats?: boolean
}

export default function ModuleRelatedPapers({
  moduleId,
  maxPapers = 10,
  showStats = true
}: ModuleRelatedPapersProps) {
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModulePapers()
  }, [moduleId])

  const fetchModulePapers = async () => {
    try {
      setLoading(true)
      setError(null)

      const res = await fetch('/api/arxiv-monitor/papers?limit=100')
      const data = await res.json()

      if (data.success && data.data?.papers) {
        // 해당 모듈과 관련된 논문만 필터링
        const modulePapers = data.data.papers
          .filter((p: Paper) => p.relatedModules.includes(moduleId))
          .sort((a: Paper, b: Paper) =>
            new Date(b.publishedDate).getTime() - new Date(a.publishedDate).getTime()
          )
          .slice(0, maxPapers)

        setPapers(modulePapers)
      }
    } catch (err) {
      console.error('Failed to fetch module papers:', err)
      setError('논문을 불러오는데 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto mb-3"></div>
          <p className="text-slate-600 dark:text-slate-400 text-sm">관련 논문을 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 text-center">
        <p className="text-red-600 dark:text-red-400">{error}</p>
        <button
          onClick={fetchModulePapers}
          className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm"
        >
          다시 시도
        </button>
      </div>
    )
  }

  if (papers.length === 0) {
    return (
      <div className="bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-lg p-8 text-center">
        <div className="text-5xl mb-3">📚</div>
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
          아직 관련 논문이 없습니다
        </h3>
        <p className="text-slate-600 dark:text-slate-400 text-sm mb-4">
          이 모듈과 관련된 최신 논문이 추가되면 여기에 표시됩니다.
        </p>
        <Link
          href="/papers"
          className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
        >
          전체 논문 보기
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Stats Header */}
      {showStats && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-1">
                📄 관련 최신 연구 논문
              </h3>
              <p className="text-slate-600 dark:text-slate-400">
                이 모듈과 관련된 ArXiv 최신 논문 {papers.length}편
              </p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                {papers.length}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">편의 논문</div>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-blue-200 dark:border-blue-800">
            <div>
              <div className="text-lg font-bold text-green-600 dark:text-green-400">
                {papers.filter(p => p.status === 'MDX_GENERATED').length}
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400">요약 완료</div>
            </div>
            <div>
              <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
                {new Set(papers.flatMap(p => p.categories)).size}
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400">카테고리</div>
            </div>
            <div>
              <div className="text-lg font-bold text-orange-600 dark:text-orange-400">
                {papers[0] ? new Date(papers[0].publishedDate).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' }) : '-'}
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400">최신 논문</div>
            </div>
          </div>
        </div>
      )}

      {/* Papers List */}
      <div className="grid gap-4">
        {papers.map((paper, index) => (
          <Card key={paper.id} className="hover:shadow-lg transition-all hover:border-blue-300 dark:hover:border-blue-700">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
                      #{index + 1}
                    </span>
                    <Badge variant={paper.status === 'MDX_GENERATED' ? 'default' : 'secondary'} className="text-xs">
                      {paper.status === 'MDX_GENERATED' ? '✅ 요약 완료' : '⏳ 처리 중'}
                    </Badge>
                  </div>
                  <CardTitle className="text-lg leading-tight mb-2 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                    <Link href={`/papers/${paper.arxivId}`}>
                      {paper.title}
                    </Link>
                  </CardTitle>
                  <CardDescription className="text-sm">
                    {paper.authors.slice(0, 3).join(', ')}
                    {paper.authors.length > 3 && ` 외 ${paper.authors.length - 3}명`}
                  </CardDescription>
                </div>
                <div className="text-right">
                  <span className="text-xs text-slate-500 dark:text-slate-400 block mb-1">
                    {new Date(paper.publishedDate).toLocaleDateString('ko-KR')}
                  </span>
                  <span className="text-xs font-mono text-blue-600 dark:text-blue-400">
                    {paper.arxivId}
                  </span>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              {/* Categories */}
              {paper.categories.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {paper.categories.slice(0, 3).map(cat => (
                    <span
                      key={cat}
                      className="px-2 py-0.5 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs rounded"
                    >
                      {cat}
                    </span>
                  ))}
                  {paper.categories.length > 3 && (
                    <span className="px-2 py-0.5 text-slate-500 text-xs">
                      +{paper.categories.length - 3}
                    </span>
                  )}
                </div>
              )}

              {/* Keywords */}
              {paper.keywords.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {paper.keywords.slice(0, 5).map(kw => (
                    <span
                      key={kw}
                      className="px-2 py-0.5 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs rounded"
                    >
                      🔑 {kw}
                    </span>
                  ))}
                  {paper.keywords.length > 5 && (
                    <span className="px-2 py-0.5 text-blue-600 dark:text-blue-400 text-xs">
                      +{paper.keywords.length - 5} more
                    </span>
                  )}
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2 pt-3 border-t dark:border-slate-700">
                {paper.status === 'MDX_GENERATED' ? (
                  <Link
                    href={`/papers/${paper.arxivId}`}
                    className="flex-1 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-center text-sm font-medium"
                  >
                    📄 한글 요약 보기
                  </Link>
                ) : (
                  <button
                    disabled
                    className="flex-1 px-3 py-2 bg-slate-400 text-white rounded-lg cursor-not-allowed text-center text-sm font-medium opacity-60"
                  >
                    ⏳ 요약 준비 중...
                  </button>
                )}
                <a
                  href={paper.pdfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-1 px-3 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors text-center text-sm font-medium"
                >
                  📑 원문 (PDF)
                </a>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* View All Link */}
      {papers.length >= maxPapers && (
        <div className="text-center pt-4">
          <Link
            href={`/papers?module=${moduleId}`}
            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105 font-medium"
          >
            <span>전체 논문 보기</span>
            <span>→</span>
          </Link>
        </div>
      )}
    </div>
  )
}
