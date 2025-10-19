/**
 * RelatedPapers Component
 * 모듈과 관련된 최신 ArXiv 논문을 표시하는 컴포넌트
 */

'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
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

interface RelatedPapersProps {
  moduleId: string
  limit?: number
  title?: string
}

export default function RelatedPapers({
  moduleId,
  limit = 3,
  title = '📚 관련 최신 연구'
}: RelatedPapersProps) {
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchPapers()
  }, [moduleId])

  const fetchPapers = async () => {
    try {
      const res = await fetch(`/api/arxiv-monitor/papers?limit=50`)
      const data = await res.json()
      if (data.success && data.data?.papers) {
        // 해당 모듈과 관련된 논문 필터링
        const filtered = data.data.papers
          .filter((p: Paper) => p.relatedModules.includes(moduleId))
          .slice(0, limit)
        setPapers(filtered)
      }
    } catch (error) {
      console.error('Failed to fetch papers:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="py-8">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
          {title}
        </h2>
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    )
  }

  if (papers.length === 0) {
    return null
  }

  return (
    <div className="py-8 border-t border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
          {title}
        </h2>
        <Link
          href={`/papers?filter=${moduleId}`}
          className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 text-sm font-medium hover:underline"
        >
          전체 보기 ({papers.length}개) →
        </Link>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {papers.map(paper => (
          <Card key={paper.id} className="hover:shadow-lg transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between gap-2 mb-2">
                <Badge variant={paper.status === 'MDX_GENERATED' ? 'default' : 'secondary'} className="text-xs">
                  {paper.status === 'MDX_GENERATED' ? '✅ 요약 완료' : '⏳ 처리 중'}
                </Badge>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {new Date(paper.publishedDate).toLocaleDateString('ko-KR', {
                    month: 'short',
                    day: 'numeric'
                  })}
                </span>
              </div>
              <CardTitle className="text-base leading-tight">
                <Link
                  href={`/papers/${paper.arxivId}`}
                  className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors line-clamp-2"
                >
                  {paper.title}
                </Link>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-xs text-slate-600 dark:text-slate-400 mb-3 line-clamp-1">
                {paper.authors.slice(0, 2).join(', ')}
                {paper.authors.length > 2 && ` 외 ${paper.authors.length - 2}명`}
              </p>

              {/* Keywords */}
              {paper.keywords.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {paper.keywords.slice(0, 2).map(kw => (
                    <span
                      key={kw}
                      className="px-2 py-0.5 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs rounded"
                    >
                      {kw}
                    </span>
                  ))}
                  {paper.keywords.length > 2 && (
                    <span className="text-xs text-slate-400">
                      +{paper.keywords.length - 2}
                    </span>
                  )}
                </div>
              )}

              <div className="flex gap-2">
                {paper.status === 'MDX_GENERATED' ? (
                  <Link
                    href={`/papers/${paper.arxivId}`}
                    className="flex-1 px-3 py-1.5 bg-blue-600 text-white rounded text-center text-xs font-medium hover:bg-blue-700 transition-colors"
                  >
                    한글 요약
                  </Link>
                ) : (
                  <button
                    disabled
                    className="flex-1 px-3 py-1.5 bg-slate-300 dark:bg-slate-600 text-slate-500 dark:text-slate-400 rounded text-center text-xs font-medium cursor-not-allowed"
                  >
                    요약 준비 중
                  </button>
                )}
                <a
                  href={paper.pdfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-1 px-3 py-1.5 bg-slate-600 text-white rounded text-center text-xs font-medium hover:bg-slate-700 transition-colors"
                >
                  원문 PDF
                </a>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {papers.length >= limit && (
        <div className="mt-6 text-center">
          <Link
            href={`/papers?filter=${moduleId}`}
            className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            더 많은 {moduleId.toUpperCase()} 연구 보기 →
          </Link>
        </div>
      )}
    </div>
  )
}
