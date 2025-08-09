import { NextResponse } from 'next/server'
import { AIContentAnalyzer } from '../../lib/ai-analyzer'
import { ContentUpdateManager } from '../../lib/update-manager'
import { ModuleRegistry } from '../../lib/module-registry'

export async function POST(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const moduleId = searchParams.get('module')
    
    const analyzer = new AIContentAnalyzer()
    const updateManager = new ContentUpdateManager()
    const registry = new ModuleRegistry()
    
    let updates = []
    
    if (moduleId) {
      // Check updates for specific module
      const module = await registry.getModuleById(moduleId)
      if (module) {
        const moduleUpdates = await analyzer.checkForUpdates(module)
        updates.push(...moduleUpdates)
      }
    } else {
      // Check updates for all modules
      const modules = await registry.getAllModuleStatuses()
      
      for (const module of modules) {
        const moduleUpdates = await analyzer.checkForUpdates(module)
        updates.push(...moduleUpdates)
      }
    }
    
    // Save identified updates
    for (const update of updates) {
      await updateManager.createUpdate((update as any).moduleId || moduleId || 'unknown', update)
    }
    
    return NextResponse.json(updates)
  } catch (error) {
    console.error('Error checking for updates:', error)
    return NextResponse.json(
      { error: 'Failed to check for updates' },
      { status: 500 }
    )
  }
}