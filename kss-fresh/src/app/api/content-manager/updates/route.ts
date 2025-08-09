import { NextResponse } from 'next/server'
import { ContentUpdateManager } from '../lib/update-manager'
import { ContentUpdate } from '../types'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const moduleId = searchParams.get('module')
    const status = searchParams.get('status')
    
    const updateManager = new ContentUpdateManager()
    let updates = await updateManager.getAllUpdates()
    
    // Filter by module if specified
    if (moduleId) {
      updates = updates.filter(update => update.moduleId === moduleId)
    }
    
    // Filter by status if specified
    if (status) {
      updates = updates.filter(update => update.status === status)
    }
    
    return NextResponse.json(updates)
  } catch (error) {
    console.error('Error fetching updates:', error)
    return NextResponse.json(
      { error: 'Failed to fetch updates' },
      { status: 500 }
    )
  }
}

export async function POST(request: Request) {
  try {
    const update: ContentUpdate = await request.json()
    
    const updateManager = new ContentUpdateManager()
    const createdUpdate = await updateManager.createUpdate(update.moduleId || 'unknown', update)
    
    return NextResponse.json(createdUpdate, { status: 201 })
  } catch (error) {
    console.error('Error creating update:', error)
    return NextResponse.json(
      { error: 'Failed to create update' },
      { status: 500 }
    )
  }
}