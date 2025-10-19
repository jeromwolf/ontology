/**
 * Individual Paper Page - 개별 논문 상세 페이지
 * MDX 파일을 렌더링하여 표시
 */

import { readFile } from 'fs/promises'
import { join } from 'path'
import { notFound } from 'next/navigation'
import matter from 'gray-matter'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

interface PaperMetadata {
  title: string
  arxivId: string
  authors: string[]
  publishedDate: string
  categories: string[]
  keywords: string[]
  relatedModules: string[]
  pdfUrl: string
}

async function getPaperContent(arxivId: string) {
  try {
    // arxivId 형식: 2510.08569v1 또는 2510.08569
    const cleanId = arxivId.replace('v1', '')

    // MDX 파일 경로 구성
    const [year, month] = cleanId.split('.')[0].match(/(\d{2})(\d{2})/)?.slice(1, 3) || []
    const fullYear = `20${year}`

    const mdxPath = join(
      process.cwd(),
      'src',
      'app',
      'papers',
      'papers',
      fullYear,
      month,
      `${cleanId}.mdx`
    )

    const fileContent = await readFile(mdxPath, 'utf-8')
    const { data, content } = matter(fileContent)

    return {
      metadata: data as PaperMetadata,
      content,
      path: mdxPath,
    }
  } catch (error) {
    console.error('Error reading paper:', error)
    return null
  }
}

export default async function PaperPage({
  params,
}: {
  params: { arxivId: string }
}) {
  // DB에서 논문 정보 가져오기
  const dbPaper = await prisma.arXiv_Paper.findFirst({
    where: {
      arxivId: {
        startsWith: params.arxivId.replace('v1', ''),
      },
    },
  })

  if (!dbPaper) {
    await prisma.$disconnect()
    notFound()
  }

  // MDX 파일 읽기 시도
  const paper = await getPaperContent(params.arxivId)

  // MDX 파일이 없으면 (요약 준비 중)
  if (!paper) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
        <div className="container mx-auto px-4 py-8 max-w-5xl">
          <Link
            href="/papers"
            className="inline-flex items-center text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 mb-8 transition-colors"
          >
            ← 논문 목록으로 돌아가기
          </Link>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8 text-center">
            <div className="mb-6">
              <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
                ⏳ 요약 준비 중입니다
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                이 논문의 한글 요약이 아직 생성되지 않았습니다.
              </p>
            </div>

            <div className="bg-slate-100 dark:bg-slate-700 rounded-lg p-6 mb-6 text-left">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-3">
                {dbPaper.title}
              </h2>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                👥 저자: {dbPaper.authors.join(', ')}
              </p>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                📅 발행일: {new Date(dbPaper.publishedDate).toLocaleDateString('ko-KR')}
              </p>
            </div>

            <div className="flex gap-3 justify-center">
              <a
                href={dbPaper.pdfUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                📑 원문 (PDF) 보기
              </a>
              <Link
                href="/papers"
                className="px-6 py-3 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors font-medium"
              >
                ← 논문 목록
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const { metadata, content } = paper

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/papers"
            className="inline-flex items-center text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 mb-4 transition-colors"
          >
            ← 논문 목록으로 돌아가기
          </Link>

          {/* Paper Info Card */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8 mb-8">
            <h1 className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-4">
              {metadata.title}
            </h1>

            {/* Authors */}
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-2">
                👥 저자
              </h3>
              <p className="text-slate-700 dark:text-slate-300">
                {metadata.authors.join(', ')}
              </p>
            </div>

            {/* Meta Info */}
            <div className="grid md:grid-cols-2 gap-4 mb-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-2">
                  📅 발행일
                </h3>
                <p className="text-slate-700 dark:text-slate-300">
                  {new Date(metadata.publishedDate).toLocaleDateString('ko-KR', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric',
                  })}
                </p>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-2">
                  🔗 ArXiv ID
                </h3>
                <a
                  href={metadata.pdfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 hover:underline"
                >
                  {metadata.arxivId}
                </a>
              </div>
            </div>

            {/* Categories */}
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-2">
                📂 카테고리
              </h3>
              <div className="flex flex-wrap gap-2">
                {metadata.categories.map(cat => (
                  <span
                    key={cat}
                    className="px-3 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm rounded-full"
                  >
                    {cat}
                  </span>
                ))}
              </div>
            </div>

            {/* Keywords */}
            {metadata.keywords && metadata.keywords.length > 0 && (
              <div className="mb-4">
                <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-2">
                  🔑 키워드
                </h3>
                <div className="flex flex-wrap gap-2">
                  {metadata.keywords.map(kw => (
                    <span
                      key={kw}
                      className="px-3 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-sm rounded-full"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Related Modules */}
            {metadata.relatedModules && metadata.relatedModules.length > 0 && (
              <div className="mb-4">
                <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-400 mb-2">
                  📚 관련 모듈
                </h3>
                <div className="flex flex-wrap gap-2">
                  {metadata.relatedModules.map(module => (
                    <Link
                      key={module}
                      href={`/modules/${module}`}
                      className="px-3 py-1 bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-sm rounded-full hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
                    >
                      {module}
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-4 border-t dark:border-slate-700">
              <a
                href={metadata.pdfUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-center font-medium"
              >
                📑 원문 (PDF) 보기
              </a>
              <a
                href={`https://arxiv.org/abs/${metadata.arxivId}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 px-6 py-3 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors text-center font-medium"
              >
                🔗 ArXiv 페이지
              </a>
            </div>
          </div>
        </div>

        {/* Paper Content (Markdown) */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8">
          <article className="prose prose-slate dark:prose-invert max-w-none prose-headings:font-bold prose-h1:text-3xl prose-h2:text-2xl prose-h2:mt-8 prose-h2:mb-4 prose-h3:text-xl prose-h3:mt-6 prose-h3:mb-3 prose-p:text-slate-700 dark:prose-p:text-slate-300 prose-p:leading-relaxed prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline prose-strong:text-slate-900 dark:prose-strong:text-slate-100 prose-ul:my-4 prose-li:my-2 prose-code:text-blue-600 dark:prose-code:text-blue-400 prose-code:bg-slate-100 dark:prose-code:bg-slate-700 prose-code:px-1 prose-code:py-0.5 prose-code:rounded">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {content}
            </ReactMarkdown>
          </article>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <Link
            href="/papers"
            className="inline-flex items-center px-6 py-3 bg-slate-700 text-white rounded-lg hover:bg-slate-800 transition-colors"
          >
            ← 논문 목록으로 돌아가기
          </Link>
        </div>
      </div>
    </div>
  )
}
