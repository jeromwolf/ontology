/**
 * ArXiv Monitor - Logs API
 * GET /api/arxiv-monitor/logs?lines=...
 */

import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { existsSync } from 'fs'
import path from 'path'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const lines = parseInt(searchParams.get('lines') || '100')

    const logPath = path.join(
      process.cwd(),
      'scripts',
      'arxiv-monitor',
      'logs',
      'arxiv-monitor.log'
    )

    // 로그 파일이 없으면 빈 배열 반환
    if (!existsSync(logPath)) {
      return NextResponse.json({
        success: true,
        data: {
          logs: [],
          message: 'Log file not found',
        },
      })
    }

    // 로그 파일 읽기
    const logContent = await readFile(logPath, 'utf-8')
    const logLines = logContent.split('\n').filter((line) => line.trim())

    // 최근 N개 라인만 반환
    const recentLogs = logLines.slice(-lines)

    // 각 라인을 파싱 (winston JSON 형식)
    const parsedLogs = recentLogs
      .map((line) => {
        try {
          return JSON.parse(line)
        } catch {
          // JSON이 아닌 경우 그대로 반환
          return { message: line, level: 'info', timestamp: new Date().toISOString() }
        }
      })
      .reverse() // 최신이 위로

    return NextResponse.json({
      success: true,
      data: {
        logs: parsedLogs,
        total: logLines.length,
        showing: parsedLogs.length,
      },
    })
  } catch (error) {
    console.error('Error in logs API:', error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
