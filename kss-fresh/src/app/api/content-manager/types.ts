export interface ModuleStatus {
  id: string
  name: string
  lastUpdate: Date
  version: string
  accuracyScore: number
  outdatedChapters: number
  brokenLinks: number
  deprecatedCode: number
  simulatorHealth: 'healthy' | 'warning' | 'critical'
  updateFrequency: 'realtime' | 'daily' | 'weekly' | 'monthly'
  nextUpdate: Date
  metadata?: {
    totalChapters: number
    totalSimulators: number
    totalExamples: number
    lastValidation?: Date
    lastContentCheck?: Date
  }
}

export interface ValidationIssue {
  id: string
  moduleId: string
  type: 'outdated' | 'broken_link' | 'deprecated_code' | 'missing_simulator' | 'incorrect_info'
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  location: string
  suggestedFix?: string
  source?: string
  createdAt: Date
  resolvedAt?: Date
  resolvedBy?: string
}

export interface ContentUpdate {
  id: string
  moduleId: string
  type: 'content' | 'simulator' | 'example' | 'reference' | 'correction'
  title: string
  description: string
  source: string
  confidence: number
  status: 'pending' | 'reviewing' | 'approved' | 'applied' | 'rejected'
  createdAt: Date
  appliedAt?: Date
  appliedBy?: string
  changes?: {
    file: string
    before: string
    after: string
  }[]
  metadata?: {
    newsUrl?: string
    researchPaper?: string
    githubPR?: string
    authorEmail?: string
  }
}

export interface Module {
  id: string
  name: string
  path: string
  description: string
  chapters: Chapter[]
  simulators: Simulator[]
  config: ModuleConfig
}

export interface Chapter {
  id: string
  title: string
  path: string
  lastModified: Date
  content?: string
  metadata?: {
    wordCount: number
    readingTime: number
    difficulty: string
    prerequisites: string[]
  }
}

export interface Simulator {
  id: string
  name: string
  path: string
  status: 'working' | 'broken' | 'deprecated'
  lastTested: Date
  dependencies: string[]
}

export interface ModuleConfig {
  updateFrequency: 'realtime' | 'daily' | 'weekly' | 'monthly'
  sources: DataSource[]
  validators: ValidatorConfig[]
  criticalMetrics: string[]
  aiAgent?: AIAgentConfig
}

export interface DataSource {
  name: string
  url: string
  type: 'rss' | 'api' | 'scraper' | 'manual'
  frequency: string
  lastChecked?: Date
  active: boolean
}

export interface ValidatorConfig {
  type: string
  enabled: boolean
  rules: ValidationRule[]
}

export interface ValidationRule {
  id: string
  name: string
  pattern?: string
  condition: string
  action: string
  severity: 'low' | 'medium' | 'high' | 'critical'
}

export interface AIAgentConfig {
  enabled: boolean
  model: string
  checkFrequency: string
  autoApprove: boolean
  confidenceThreshold: number
  prompts: {
    contentCheck: string
    updateSuggestion: string
    validation: string
  }
}

export interface UpdateResult {
  success: boolean
  details?: any
  error?: string
  affectedFiles?: string[]
  rollbackable?: boolean
}