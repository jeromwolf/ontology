import { NextResponse } from 'next/server'
import { ContentValidator } from '../../lib/content-validator'
import { ModuleRegistry } from '../../lib/module-registry'

export async function POST(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const moduleId = searchParams.get('module')
    
    const validator = new ContentValidator()
    const registry = new ModuleRegistry()
    
    let issues = []
    
    if (moduleId) {
      // Validate specific module
      const result = await validator.validateModule(moduleId)
      issues = result.issues
    } else {
      // Validate all modules
      const modules = await registry.getAllModuleStatuses()
      
      for (const module of modules) {
        const result = await validator.validateModule(module.id)
        issues.push(...result.issues)
      }
    }
    
    // Update module statuses based on validation results
    await registry.updateValidationResults(issues)
    
    return NextResponse.json(issues)
  } catch (error) {
    console.error('Error running validation:', error)
    return NextResponse.json(
      { error: 'Failed to run validation' },
      { status: 500 }
    )
  }
}