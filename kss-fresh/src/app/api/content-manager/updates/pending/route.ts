import { NextResponse } from 'next/server'

// 업데이트 상태 타입
type UpdateStatus = 'pending' | 'reviewed' | 'approved' | 'rejected' | 'applied'

interface PendingUpdate {
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
  status: UpdateStatus
  createdAt: Date
  reviewedAt?: Date
  reviewNotes?: string
  appliedAt?: Date
}

// Mock 데이터베이스 (실제로는 DB 사용)
let pendingUpdates: PendingUpdate[] = []

// GET: 대기 중인 업데이트 목록 조회
export async function GET(request: Request) {
  const url = new URL(request.url)
  const status = url.searchParams.get('status') as UpdateStatus | null
  const moduleId = url.searchParams.get('moduleId')
  
  let filtered = pendingUpdates
  
  if (status) {
    filtered = filtered.filter(u => u.status === status)
  }
  
  if (moduleId) {
    filtered = filtered.filter(u => u.moduleId === moduleId)
  }
  
  // 우선순위와 날짜로 정렬
  filtered.sort((a, b) => {
    const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 }
    const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority]
    if (priorityDiff !== 0) return priorityDiff
    return b.createdAt.getTime() - a.createdAt.getTime()
  })
  
  return NextResponse.json({
    updates: filtered,
    stats: {
      total: filtered.length,
      pending: filtered.filter(u => u.status === 'pending').length,
      reviewed: filtered.filter(u => u.status === 'reviewed').length,
      approved: filtered.filter(u => u.status === 'approved').length,
      rejected: filtered.filter(u => u.status === 'rejected').length,
      applied: filtered.filter(u => u.status === 'applied').length
    }
  })
}

// POST: 새로운 업데이트 제안 생성
export async function POST(request: Request) {
  const body = await request.json()
  
  const newUpdate: PendingUpdate = {
    id: `update-${Date.now()}`,
    moduleId: body.moduleId,
    moduleName: body.moduleName,
    chapter: body.chapter,
    type: body.type || 'content',
    title: body.title,
    description: body.description,
    oldContent: body.oldContent,
    newContent: body.newContent,
    changes: body.changes,
    source: body.source,
    sourceUrl: body.sourceUrl,
    confidence: body.confidence || 0.8,
    priority: body.priority || 'medium',
    status: 'pending',
    createdAt: new Date(),
  }
  
  pendingUpdates.push(newUpdate)
  
  // Slack 또는 이메일 알림 (옵션)
  // await sendNotification(newUpdate)
  
  return NextResponse.json({
    success: true,
    update: newUpdate,
    message: '업데이트 제안이 생성되었습니다. 검토 후 승인해주세요.'
  })
}

// PATCH: 업데이트 상태 변경 (검토, 승인, 거절)
export async function PATCH(request: Request) {
  const body = await request.json()
  const { updateId, status, reviewNotes } = body
  
  const updateIndex = pendingUpdates.findIndex(u => u.id === updateId)
  if (updateIndex === -1) {
    return NextResponse.json(
      { error: '업데이트를 찾을 수 없습니다.' },
      { status: 404 }
    )
  }
  
  const update = pendingUpdates[updateIndex]
  const previousStatus = update.status
  
  // 상태 변경 로직
  if (status === 'reviewed') {
    update.status = 'reviewed'
    update.reviewedAt = new Date()
    update.reviewNotes = reviewNotes
  } else if (status === 'approved') {
    if (update.status !== 'reviewed' && update.status !== 'pending') {
      return NextResponse.json(
        { error: '검토되지 않은 업데이트는 승인할 수 없습니다.' },
        { status: 400 }
      )
    }
    update.status = 'approved'
  } else if (status === 'rejected') {
    update.status = 'rejected'
    update.reviewNotes = reviewNotes
  } else if (status === 'applied') {
    if (update.status !== 'approved') {
      return NextResponse.json(
        { error: '승인되지 않은 업데이트는 적용할 수 없습니다.' },
        { status: 400 }
      )
    }
    update.status = 'applied'
    update.appliedAt = new Date()
  }
  
  pendingUpdates[updateIndex] = update
  
  return NextResponse.json({
    success: true,
    update,
    message: `업데이트 상태가 ${previousStatus}에서 ${status}로 변경되었습니다.`
  })
}

// DELETE: 업데이트 제안 삭제
export async function DELETE(request: Request) {
  const url = new URL(request.url)
  const updateId = url.searchParams.get('id')
  
  if (!updateId) {
    return NextResponse.json(
      { error: '업데이트 ID가 필요합니다.' },
      { status: 400 }
    )
  }
  
  const index = pendingUpdates.findIndex(u => u.id === updateId)
  if (index === -1) {
    return NextResponse.json(
      { error: '업데이트를 찾을 수 없습니다.' },
      { status: 404 }
    )
  }
  
  const deleted = pendingUpdates.splice(index, 1)[0]
  
  return NextResponse.json({
    success: true,
    deleted,
    message: '업데이트가 삭제되었습니다.'
  })
}