import { NextResponse } from 'next/server'
import { headers } from 'next/headers'
import { ModuleRegistry } from '../../lib/module-registry'
import { ContentValidator } from '../../lib/content-validator'
import { AIContentAnalyzer } from '../../lib/ai-analyzer'
import { ContentUpdateManager } from '../../lib/update-manager'

// Helper function to determine if module should be checked
async function shouldCheckModule(frequency: string, lastUpdate: Date): Promise<boolean> {
  const now = new Date()
  const hoursSinceUpdate = (now.getTime() - lastUpdate.getTime()) / (1000 * 60 * 60)
  
  switch (frequency) {
    case 'daily':
      return hoursSinceUpdate >= 24
    case 'weekly':
      return hoursSinceUpdate >= 24 * 7
    case 'monthly':
      return hoursSinceUpdate >= 24 * 30
    default:
      return false
  }
}

// This endpoint should be called by a cron job service (e.g., Vercel Cron, GitHub Actions)
export async function POST(request: Request) {
  try {
    // Verify the request is from an authorized source
    const headersList = headers()
    const authHeader = headersList.get('authorization')
    
    // In production, verify the auth token
    if (process.env.NODE_ENV === 'production' && authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }
    
    const registry = new ModuleRegistry()
    const validator = new ContentValidator()
    const analyzer = new AIContentAnalyzer()
    const updateManager = new ContentUpdateManager()
    
    const results = {
      modulesChecked: 0,
      issuesFound: 0,
      updatesIdentified: 0,
      criticalIssues: [] as any[],
      timestamp: new Date().toISOString()
    }
    
    // Get all modules that need checking
    const modules = await registry.getAllModuleStatuses()
    const today = new Date()
    
    for (const module of modules) {
      const status = await registry.getModuleById(module.id)
      if (!status) continue
      
      // Check if module needs update based on frequency
      const shouldCheck = await shouldCheckModule(status.updateFrequency, status.lastUpdate)
      
      if (shouldCheck) {
        results.modulesChecked++
        
        // Run validation
        const validationResult = await validator.validateModule(module.id)
        results.issuesFound += validationResult.issues.length
        
        // Track critical issues
        const criticalIssues = validationResult.issues.filter((i: any) => i.severity === 'critical')
        if (criticalIssues.length > 0) {
          results.criticalIssues.push({
            moduleId: module.id,
            count: criticalIssues.length,
            issues: criticalIssues.map((i: any) => i.description)
          })
        }
        
        // Check for content updates
        const updates = await analyzer.analyzeForUpdates(module.id, module)
        results.updatesIdentified += updates.length
        
        // Save updates
        for (const update of updates) {
          await updateManager.createUpdate(module.id, update)
        }
      }
    }
    
    // Send notification if critical issues found
    if (results.criticalIssues.length > 0) {
      await sendNotification(results)
    }
    
    // Log results
    console.log('Daily content check completed:', results)
    
    return NextResponse.json(results)
    
  } catch (error) {
    console.error('Error in daily content check:', error)
    return NextResponse.json(
      { error: 'Failed to complete daily check' },
      { status: 500 }
    )
  }
}


async function sendNotification(results: any): Promise<void> {
  // In production, would send email/Slack notification
  console.warn('CRITICAL ISSUES FOUND:', results.criticalIssues)
  
  // Could integrate with:
  // - Email service (SendGrid, AWS SES)
  // - Slack webhook
  // - Discord webhook
  // - SMS (Twilio)
  
  if (process.env.SLACK_WEBHOOK_URL) {
    try {
      await fetch(process.env.SLACK_WEBHOOK_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: `🚨 KSS Content Alert: ${results.criticalIssues.length} modules have critical issues`,
          attachments: results.criticalIssues.map((issue: any) => ({
            color: 'danger',
            title: `Module: ${issue.moduleId}`,
            text: `${issue.count} critical issues found`,
            fields: issue.issues.map((desc: string, i: number) => ({
              title: `Issue ${i + 1}`,
              value: desc,
              short: false
            }))
          }))
        })
      })
    } catch (error) {
      console.error('Failed to send Slack notification:', error)
    }
  }
}

// GET endpoint for manual trigger from UI
export async function GET() {
  return NextResponse.json({
    message: 'Daily check endpoint',
    info: 'Use POST method to trigger check',
    lastRun: new Date().toISOString()
  })
}