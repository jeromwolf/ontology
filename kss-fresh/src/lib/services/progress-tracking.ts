export interface ChapterProgress {
  moduleId: string
  chapterId: string
  userId?: string
  status: 'not-started' | 'in-progress' | 'completed'
  progress: number // 0-100 percentage
  timeSpent: number // in minutes
  lastAccessed: Date
  completedAt?: Date
  viewCount: number
  bookmarked: boolean
  lastVersion?: string // Track content version when last viewed
}

export interface ModuleProgress {
  moduleId: string
  userId?: string
  totalChapters: number
  completedChapters: number
  inProgressChapters: number
  totalProgress: number // 0-100 percentage
  totalTimeSpent: number // in minutes
  lastAccessed: Date
  startedAt: Date
  completedAt?: Date
}

export interface LearningSession {
  id: string
  moduleId: string
  chapterId: string
  userId?: string
  startTime: Date
  endTime?: Date
  duration: number // in seconds
  interactionCount: number
  scrollDepth: number // 0-100 percentage
  simulatorUsed: boolean
}

class ProgressTrackingService {
  private chapterProgress: Map<string, ChapterProgress> = new Map()
  private moduleProgress: Map<string, ModuleProgress> = new Map()
  private sessions: LearningSession[] = []
  private currentSession: LearningSession | null = null
  private listeners: ((data: any) => void)[] = []

  // Initialize service
  init() {
    this.loadFromStorage()
    this.setupAutoSave()
    this.setupVisibilityTracking()
  }

  // Chapter Progress Management
  async updateChapterProgress(
    moduleId: string, 
    chapterId: string, 
    progress: Partial<ChapterProgress>
  ): Promise<ChapterProgress> {
    const key = `${moduleId}-${chapterId}`
    const existing = this.chapterProgress.get(key)
    
    const updated: ChapterProgress = {
      moduleId,
      chapterId,
      status: 'not-started',
      progress: 0,
      timeSpent: 0,
      lastAccessed: new Date(),
      viewCount: 0,
      bookmarked: false,
      ...existing,
      ...progress,
      lastAccessed: new Date()
    }

    // Auto-determine status based on progress
    if (updated.progress === 0) {
      updated.status = 'not-started'
    } else if (updated.progress === 100) {
      updated.status = 'completed'
      if (!updated.completedAt) {
        updated.completedAt = new Date()
      }
    } else {
      updated.status = 'in-progress'
    }

    this.chapterProgress.set(key, updated)
    await this.updateModuleProgress(moduleId)
    this.saveToStorage()
    this.notifyListeners()

    return updated
  }

  // Start learning session
  async startSession(moduleId: string, chapterId: string): Promise<string> {
    // End current session if exists
    if (this.currentSession) {
      await this.endSession()
    }

    const session: LearningSession = {
      id: `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      moduleId,
      chapterId,
      startTime: new Date(),
      duration: 0,
      interactionCount: 0,
      scrollDepth: 0,
      simulatorUsed: false
    }

    this.currentSession = session
    
    // Update chapter access
    await this.updateChapterProgress(moduleId, chapterId, {
      lastAccessed: new Date(),
      viewCount: (this.getChapterProgress(moduleId, chapterId)?.viewCount || 0) + 1
    })

    return session.id
  }

  // End learning session
  async endSession(): Promise<LearningSession | null> {
    if (!this.currentSession) return null

    const endTime = new Date()
    this.currentSession.endTime = endTime
    this.currentSession.duration = Math.round(
      (endTime.getTime() - this.currentSession.startTime.getTime()) / 1000
    )

    // Update chapter time spent
    const timeSpentMinutes = Math.round(this.currentSession.duration / 60)
    await this.updateChapterProgress(
      this.currentSession.moduleId, 
      this.currentSession.chapterId, 
      {
        timeSpent: (this.getChapterProgress(
          this.currentSession.moduleId, 
          this.currentSession.chapterId
        )?.timeSpent || 0) + timeSpentMinutes
      }
    )

    this.sessions.push(this.currentSession)
    const completedSession = this.currentSession
    this.currentSession = null
    
    this.saveToStorage()
    return completedSession
  }

  // Track user interactions
  async trackInteraction(type: 'scroll' | 'click' | 'simulator' | 'bookmark') {
    if (!this.currentSession) return

    this.currentSession.interactionCount++
    
    if (type === 'simulator') {
      this.currentSession.simulatorUsed = true
    }
  }

  // Track scroll progress
  async trackScrollProgress(scrollDepth: number) {
    if (!this.currentSession) return

    this.currentSession.scrollDepth = Math.max(
      this.currentSession.scrollDepth, 
      scrollDepth
    )

    // Auto-update chapter progress based on scroll
    if (scrollDepth > 80 && this.currentSession.scrollDepth > 80) {
      await this.updateChapterProgress(
        this.currentSession.moduleId,
        this.currentSession.chapterId,
        { progress: Math.min(scrollDepth, 100) }
      )
    }
  }

  // Get chapter progress
  getChapterProgress(moduleId: string, chapterId: string): ChapterProgress | null {
    return this.chapterProgress.get(`${moduleId}-${chapterId}`) || null
  }

  // Get module progress
  getModuleProgress(moduleId: string): ModuleProgress | null {
    return this.moduleProgress.get(moduleId) || null
  }

  // Check if chapter is viewed (for notification purposes)
  isChapterViewed(moduleId: string, chapterId: string): boolean {
    const progress = this.getChapterProgress(moduleId, chapterId)
    return progress ? progress.viewCount > 0 : false
  }

  // Check if chapter content is outdated
  isChapterContentOutdated(moduleId: string, chapterId: string, currentVersion: string): boolean {
    const progress = this.getChapterProgress(moduleId, chapterId)
    return progress ? progress.lastVersion !== currentVersion : false
  }

  // Get chapters that need update notifications
  getChaptersNeedingUpdateNotification(moduleId: string, updates: {chapterId: string, version: string}[]): {
    viewed: {chapterId: string, version: string}[],
    unviewed: {chapterId: string, version: string}[]
  } {
    const viewed: {chapterId: string, version: string}[] = []
    const unviewed: {chapterId: string, version: string}[] = []

    updates.forEach(update => {
      if (this.isChapterViewed(moduleId, update.chapterId)) {
        viewed.push(update)
      } else {
        unviewed.push(update)
      }
    })

    return { viewed, unviewed }
  }

  // Get learning analytics
  getLearningAnalytics(moduleId?: string) {
    const relevantSessions = moduleId 
      ? this.sessions.filter(s => s.moduleId === moduleId)
      : this.sessions

    const totalTime = relevantSessions.reduce((sum, s) => sum + s.duration, 0)
    const avgSessionTime = relevantSessions.length > 0 
      ? totalTime / relevantSessions.length 
      : 0
    
    const simulatorUsage = relevantSessions.filter(s => s.simulatorUsed).length
    const avgScrollDepth = relevantSessions.length > 0
      ? relevantSessions.reduce((sum, s) => sum + s.scrollDepth, 0) / relevantSessions.length
      : 0

    return {
      totalSessions: relevantSessions.length,
      totalTimeMinutes: Math.round(totalTime / 60),
      avgSessionTimeMinutes: Math.round(avgSessionTime / 60),
      simulatorUsageRate: relevantSessions.length > 0 
        ? Math.round((simulatorUsage / relevantSessions.length) * 100) 
        : 0,
      avgScrollDepth: Math.round(avgScrollDepth),
      lastActiveDate: relevantSessions.length > 0 
        ? new Date(Math.max(...relevantSessions.map(s => s.startTime.getTime())))
        : null
    }
  }

  // Private methods
  private async updateModuleProgress(moduleId: string) {
    const chapterProgresses = Array.from(this.chapterProgress.values())
      .filter(p => p.moduleId === moduleId)

    if (chapterProgresses.length === 0) return

    const totalChapters = chapterProgresses.length
    const completedChapters = chapterProgresses.filter(p => p.status === 'completed').length
    const inProgressChapters = chapterProgresses.filter(p => p.status === 'in-progress').length
    const totalProgress = Math.round(
      chapterProgresses.reduce((sum, p) => sum + p.progress, 0) / totalChapters
    )
    const totalTimeSpent = chapterProgresses.reduce((sum, p) => sum + p.timeSpent, 0)
    const lastAccessed = new Date(Math.max(...chapterProgresses.map(p => p.lastAccessed.getTime())))
    
    const existing = this.moduleProgress.get(moduleId)
    const startedAt = existing?.startedAt || new Date()
    const completedAt = completedChapters === totalChapters ? new Date() : undefined

    const moduleProgress: ModuleProgress = {
      moduleId,
      totalChapters,
      completedChapters,
      inProgressChapters,
      totalProgress,
      totalTimeSpent,
      lastAccessed,
      startedAt,
      completedAt
    }

    this.moduleProgress.set(moduleId, moduleProgress)
  }

  private setupAutoSave() {
    // Auto-save every 30 seconds
    setInterval(() => {
      this.saveToStorage()
    }, 30000)
  }

  private setupVisibilityTracking() {
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', () => {
        if (document.hidden && this.currentSession) {
          this.endSession()
        }
      })

      // End session before page unload
      window.addEventListener('beforeunload', () => {
        if (this.currentSession) {
          this.endSession()
        }
      })
    }
  }

  private saveToStorage() {
    try {
      const data = {
        chapterProgress: Array.from(this.chapterProgress.entries()),
        moduleProgress: Array.from(this.moduleProgress.entries()),
        sessions: this.sessions.slice(-100) // Keep last 100 sessions
      }
      localStorage.setItem('kss-progress', JSON.stringify(data))
    } catch (error) {
      console.error('Failed to save progress to localStorage:', error)
    }
  }

  private loadFromStorage() {
    try {
      const stored = localStorage.getItem('kss-progress')
      if (stored) {
        const data = JSON.parse(stored)
        
        // Restore chapter progress
        if (data.chapterProgress) {
          this.chapterProgress = new Map(
            data.chapterProgress.map(([key, progress]: [string, any]) => [
              key,
              {
                ...progress,
                lastAccessed: new Date(progress.lastAccessed),
                completedAt: progress.completedAt ? new Date(progress.completedAt) : undefined
              }
            ])
          )
        }

        // Restore module progress
        if (data.moduleProgress) {
          this.moduleProgress = new Map(
            data.moduleProgress.map(([key, progress]: [string, any]) => [
              key,
              {
                ...progress,
                lastAccessed: new Date(progress.lastAccessed),
                startedAt: new Date(progress.startedAt),
                completedAt: progress.completedAt ? new Date(progress.completedAt) : undefined
              }
            ])
          )
        }

        // Restore sessions
        if (data.sessions) {
          this.sessions = data.sessions.map((session: any) => ({
            ...session,
            startTime: new Date(session.startTime),
            endTime: session.endTime ? new Date(session.endTime) : undefined
          }))
        }
      }
    } catch (error) {
      console.error('Failed to load progress from localStorage:', error)
    }
  }

  private notifyListeners() {
    const data = {
      chapterProgress: Array.from(this.chapterProgress.values()),
      moduleProgress: Array.from(this.moduleProgress.values())
    }
    this.listeners.forEach(callback => callback(data))
  }

  // Subscribe to progress updates
  subscribe(callback: (data: any) => void) {
    this.listeners.push(callback)
    return () => {
      this.listeners = this.listeners.filter(listener => listener !== callback)
    }
  }
}

// Export singleton instance
export const progressTracker = new ProgressTrackingService()

// Initialize on import
if (typeof window !== 'undefined') {
  progressTracker.init()
}