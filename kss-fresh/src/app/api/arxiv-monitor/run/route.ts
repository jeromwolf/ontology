/**
 * ArXiv Monitor - Pipeline Run API
 * POST /api/arxiv-monitor/run
 */

import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import { PrismaClient } from '@prisma/client'

const execAsync = promisify(exec)
const prisma = new PrismaClient()

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action } = body // 'full', 'crawler', 'summarizer', 'generator', 'notifier', 'reset'

    console.log(`🚀 Starting ArXiv Monitor: ${action}`)

    // reset 액션은 직접 Prisma로 처리
    if (action === 'reset') {
      try {
        console.log('🗑️  Deleting ArXiv data...')

        // 처리 로그 먼저 삭제 (Foreign Key 때문에)
        await prisma.arXiv_ProcessingLog.deleteMany({})
        console.log('✅ Deleted all processing logs')

        // 논문 삭제
        await prisma.arXiv_Paper.deleteMany({})
        console.log('✅ Deleted all papers')

        console.log('🎉 Database reset complete!')
        await prisma.$disconnect()

        return NextResponse.json({
          success: true,
          message: 'Database reset successfully',
          action: 'reset',
        })
      } catch (error) {
        await prisma.$disconnect()
        console.error('Error resetting database:', error)
        return NextResponse.json(
          {
            success: false,
            error: error instanceof Error ? error.message : 'Reset failed',
          },
          { status: 500 }
        )
      }
    }

    // 다른 액션들은 npm 스크립트 실행
    const commands: Record<string, string> = {
      full: 'npm run dev',
      crawler: 'npm run crawler',
      summarizer: 'npm run summarizer',
      generator: 'npm run generator',
      notifier: 'npm run notifier',
    }

    const command = commands[action]
    if (!command) {
      return NextResponse.json(
        { success: false, error: 'Invalid action' },
        { status: 400 }
      )
    }

    // 명령어 실행 (background로 실행)
    const workingDir = '/Users/blockmeta/Library/CloudStorage/GoogleDrive-jeromwolf@gmail.com/내 드라이브/KellyGoogleSpace/ontology/kss-fresh/scripts/arxiv-monitor'

    // 비동기로 실행하고 즉시 응답
    exec(
      `cd "${workingDir}" && ${command}`,
      (error, stdout, stderr) => {
        if (error) {
          console.error(`Error executing ${action}:`, error)
        } else {
          console.log(`${action} completed:`, stdout)
        }
      }
    )

    return NextResponse.json({
      success: true,
      message: `${action} pipeline started`,
      action,
    })
  } catch (error) {
    console.error('Error in run API:', error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
