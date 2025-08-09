import { NextResponse } from 'next/server'
import { ContentValidator } from '../validation/validator'
import { ModuleRegistry } from '../lib/module-registry'

export async function GET() {
  try {
    const registry = new ModuleRegistry()
    const modules = await registry.getAllModuleStatuses()
    
    return NextResponse.json(modules)
  } catch (error) {
    console.error('Error fetching module statuses:', error)
    return NextResponse.json(
      { error: 'Failed to fetch module statuses' },
      { status: 500 }
    )
  }
}

export async function POST(request: Request) {
  try {
    const { moduleId, action } = await request.json()
    
    const registry = new ModuleRegistry()
    
    switch (action) {
      case 'refresh':
        const status = await registry.refreshModuleStatus(moduleId)
        return NextResponse.json(status)
        
      case 'validate':
        const validator = new ContentValidator()
        const issues = await validator.validateModule(moduleId)
        return NextResponse.json(issues)
        
      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        )
    }
  } catch (error) {
    console.error('Error processing module action:', error)
    return NextResponse.json(
      { error: 'Failed to process module action' },
      { status: 500 }
    )
  }
}