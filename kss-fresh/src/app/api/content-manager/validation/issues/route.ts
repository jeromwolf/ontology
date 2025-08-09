import { NextResponse } from 'next/server'
import { ContentValidator } from '../../lib/content-validator'
import { ValidationIssue } from '../../types'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const moduleId = searchParams.get('module')
    const severity = searchParams.get('severity')
    
    const validator = new ContentValidator()
    let issues = await validator.getAllIssues()
    
    // Filter by module if specified
    if (moduleId) {
      issues = issues.filter(issue => issue.moduleId === moduleId)
    }
    
    // Filter by severity if specified
    if (severity) {
      issues = issues.filter(issue => issue.severity === severity)
    }
    
    return NextResponse.json(issues)
  } catch (error) {
    console.error('Error fetching validation issues:', error)
    return NextResponse.json(
      { error: 'Failed to fetch validation issues' },
      { status: 500 }
    )
  }
}

export async function POST(request: Request) {
  try {
    const issue: ValidationIssue = await request.json()
    
    const validator = new ContentValidator()
    const createdIssue = await validator.createIssue(issue)
    
    return NextResponse.json(createdIssue, { status: 201 })
  } catch (error) {
    console.error('Error creating validation issue:', error)
    return NextResponse.json(
      { error: 'Failed to create validation issue' },
      { status: 500 }
    )
  }
}

export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const issueId = searchParams.get('id')
    
    if (!issueId) {
      return NextResponse.json(
        { error: 'Issue ID required' },
        { status: 400 }
      )
    }
    
    const validator = new ContentValidator()
    await validator.resolveIssue(issueId)
    
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting validation issue:', error)
    return NextResponse.json(
      { error: 'Failed to delete validation issue' },
      { status: 500 }
    )
  }
}