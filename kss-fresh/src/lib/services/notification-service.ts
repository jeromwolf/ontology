export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error' | 'content-update'
  title: string
  message: string
  moduleId?: string
  actionUrl?: string
  read: boolean
  createdAt: Date
  expiresAt?: Date
}

export interface NotificationPreferences {
  contentUpdates: boolean
  systemAlerts: boolean
  moduleValidation: boolean
  emailNotifications: boolean
  pushNotifications: boolean
}

class NotificationService {
  private listeners: ((notifications: Notification[]) => void)[] = []
  private notifications: Notification[] = []

  // Subscribe to notification updates
  subscribe(callback: (notifications: Notification[]) => void) {
    this.listeners.push(callback)
    // Send initial notifications
    callback(this.notifications)
    
    // Return unsubscribe function
    return () => {
      this.listeners = this.listeners.filter(listener => listener !== callback)
    }
  }

  // Add new notification
  async addNotification(notification: Omit<Notification, 'id' | 'read' | 'createdAt'>) {
    const newNotification: Notification = {
      id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      read: false,
      createdAt: new Date(),
      ...notification
    }

    this.notifications.unshift(newNotification)
    
    // Keep only last 50 notifications
    this.notifications = this.notifications.slice(0, 50)
    
    // Notify all listeners
    this.notifyListeners()
    
    // Save to localStorage for persistence
    this.saveToStorage()
    
    // Show browser notification if supported and enabled
    this.showBrowserNotification(newNotification)

    return newNotification
  }

  // Mark notification as read
  async markAsRead(notificationId: string) {
    const notification = this.notifications.find(n => n.id === notificationId)
    if (notification) {
      notification.read = true
      this.notifyListeners()
      this.saveToStorage()
    }
  }

  // Mark all notifications as read
  async markAllAsRead() {
    this.notifications.forEach(n => n.read = true)
    this.notifyListeners()
    this.saveToStorage()
  }

  // Remove notification
  async removeNotification(notificationId: string) {
    this.notifications = this.notifications.filter(n => n.id !== notificationId)
    this.notifyListeners()
    this.saveToStorage()
  }

  // Clear all notifications
  async clearAll() {
    this.notifications = []
    this.notifyListeners()
    this.saveToStorage()
  }

  // Get unread count
  getUnreadCount(): number {
    return this.notifications.filter(n => !n.read).length
  }

  // Get notifications by type
  getByType(type: Notification['type']): Notification[] {
    return this.notifications.filter(n => n.type === type)
  }

  // Get notifications by module
  getByModule(moduleId: string): Notification[] {
    return this.notifications.filter(n => n.moduleId === moduleId)
  }

  private notifyListeners() {
    this.listeners.forEach(callback => callback([...this.notifications]))
  }

  private saveToStorage() {
    try {
      localStorage.setItem('kss-notifications', JSON.stringify(this.notifications))
    } catch (error) {
      console.error('Failed to save notifications to localStorage:', error)
    }
  }

  private loadFromStorage() {
    try {
      const stored = localStorage.getItem('kss-notifications')
      if (stored) {
        const parsed = JSON.parse(stored)
        this.notifications = parsed.map((n: any) => ({
          ...n,
          createdAt: new Date(n.createdAt),
          expiresAt: n.expiresAt ? new Date(n.expiresAt) : undefined
        }))
        
        // Remove expired notifications
        this.notifications = this.notifications.filter(n => 
          !n.expiresAt || n.expiresAt > new Date()
        )
      }
    } catch (error) {
      console.error('Failed to load notifications from localStorage:', error)
    }
  }

  private async showBrowserNotification(notification: Notification) {
    if (!('Notification' in window)) {
      return
    }

    if (Notification.permission === 'granted') {
      new Notification(notification.title, {
        body: notification.message,
        icon: '/favicon.ico',
        badge: '/favicon.ico',
        tag: notification.id,
        requireInteraction: notification.type === 'error'
      })
    }
  }

  // Request notification permission
  async requestPermission(): Promise<NotificationPermission> {
    if (!('Notification' in window)) {
      return 'denied'
    }

    if (Notification.permission === 'default') {
      return await Notification.requestPermission()
    }

    return Notification.permission
  }

  // Initialize service
  init() {
    this.loadFromStorage()
    
    // Clean up expired notifications every minute
    setInterval(() => {
      const now = new Date()
      const initialLength = this.notifications.length
      
      this.notifications = this.notifications.filter(n => 
        !n.expiresAt || n.expiresAt > now
      )
      
      if (this.notifications.length !== initialLength) {
        this.notifyListeners()
        this.saveToStorage()
      }
    }, 60000) // 1 minute
  }
}

// Export singleton instance
export const notificationService = new NotificationService()

// Initialize on import
if (typeof window !== 'undefined') {
  notificationService.init()
}

// Content Manager specific notifications
export class ContentManagerNotifications {
  static async notifyContentUpdate(moduleId: string, title: string, description: string, confidence: number) {
    return notificationService.addNotification({
      type: 'content-update',
      title: `📚 ${title}`,
      message: `${description} (신뢰도: ${confidence}%)`,
      moduleId,
      actionUrl: `/modules/content-manager/review`,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 days
    })
  }

  static async notifyValidationIssue(moduleId: string, severity: 'low' | 'medium' | 'high' | 'critical', description: string) {
    const severityEmoji = {
      low: '🔵',
      medium: '🟡', 
      high: '🟠',
      critical: '🔴'
    }

    return notificationService.addNotification({
      type: severity === 'critical' ? 'error' : 'warning',
      title: `${severityEmoji[severity]} 검증 이슈 발견`,
      message: `${moduleId}: ${description}`,
      moduleId,
      actionUrl: `/modules/content-manager`,
      expiresAt: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000) // 3 days
    })
  }

  static async notifyUpdateApplied(moduleId: string, updateTitle: string) {
    return notificationService.addNotification({
      type: 'success',
      title: '✅ 업데이트 적용 완료',
      message: `${moduleId}: ${updateTitle}`,
      moduleId,
      actionUrl: `/modules/${moduleId}`,
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000) // 1 day
    })
  }

  static async notifyDailyCheck(totalUpdates: number, criticalIssues: number) {
    if (totalUpdates === 0 && criticalIssues === 0) {
      return notificationService.addNotification({
        type: 'success',
        title: '🎉 모든 모듈 최신 상태',
        message: '모든 콘텐츠가 최신 상태를 유지하고 있습니다.',
        actionUrl: '/modules/content-manager',
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000) // 1 day
      })
    } else {
      return notificationService.addNotification({
        type: criticalIssues > 0 ? 'error' : 'info',
        title: '📊 일일 콘텐츠 체크 완료',
        message: `발견된 업데이트: ${totalUpdates}개, 중요 이슈: ${criticalIssues}개`,
        actionUrl: '/modules/content-manager',
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000) // 1 day
      })
    }
  }

  static async notifyModuleOutdated(moduleId: string, daysSinceUpdate: number) {
    return notificationService.addNotification({
      type: 'warning',
      title: '⚠️ 모듈 업데이트 필요',
      message: `${moduleId} 모듈이 ${daysSinceUpdate}일 동안 업데이트되지 않았습니다.`,
      moduleId,
      actionUrl: `/modules/content-manager`,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 days
    })
  }
}