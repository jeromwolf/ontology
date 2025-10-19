/**
 * ArXiv Monitor - MDX File Reader API
 * GET /api/arxiv-monitor/papers/[arxivId]/mdx
 */

import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: { arxivId: string } }
) {
  try {
    const { arxivId } = params

    // arxivId 형식: 2510.08569v1 또는 2510.08569
    const cleanId = arxivId.replace('v1', '')

    // MDX 파일 경로 구성
    // 예: 2510.08569 → 2025/10
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

    console.log(`📖 Reading MDX file: ${mdxPath}`)

    // MDX 파일 읽기
    const mdxContent = await readFile(mdxPath, 'utf-8')

    return NextResponse.json({
      success: true,
      arxivId: cleanId,
      path: mdxPath,
      content: mdxContent,
      url: `/papers/papers/${fullYear}/${month}/${cleanId}`,
    })
  } catch (error) {
    console.error('Error reading MDX file:', error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to read MDX file',
      },
      { status: 404 }
    )
  }
}
