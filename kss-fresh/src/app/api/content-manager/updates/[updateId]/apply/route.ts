import { NextResponse } from 'next/server'
import { ContentUpdateManager } from '../../../lib/update-manager'
import { ContentApplier } from '../../../lib/content-applier'

export async function POST(
  request: Request,
  { params }: { params: { updateId: string } }
) {
  try {
    const { updateId } = params
    
    const updateManager = new ContentUpdateManager()
    const applier = new ContentApplier()
    
    // Get the update
    const update = await updateManager.getUpdate(updateId)
    
    if (!update) {
      return NextResponse.json(
        { error: 'Update not found' },
        { status: 404 }
      )
    }
    
    // Apply the update
    const result = await applier.applyUpdate(update)
    
    if (result.success) {
      // Update status to applied
      await updateManager.updateStatus(updateId, 'applied')
      
      return NextResponse.json({
        success: true,
        message: 'Update applied successfully'
      })
    } else {
      return NextResponse.json(
        { 
          success: false, 
          error: result.error
        },
        { status: 400 }
      )
    }
  } catch (error) {
    console.error('Error applying update:', error)
    return NextResponse.json(
      { error: 'Failed to apply update' },
      { status: 500 }
    )
  }
}