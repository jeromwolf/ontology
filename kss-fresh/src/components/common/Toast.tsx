'use client'

import { useEffect, useState } from 'react'
import { X, CheckCircle, AlertCircle, AlertTriangle, Info, BookOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ToastData {
  id: string
  type: 'success' | 'error' | 'warning' | 'info' | 'content-update'
  title: string
  message: string
  duration?: number
  actionUrl?: string
  actionLabel?: string
  onAction?: () => void
  onClose?: () => void
}

interface ToastProps {
  toast: ToastData
  onClose: (id: string) => void
}

const iconMap = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
  'content-update': BookOpen,
}

const colorMap = {
  success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-900 dark:text-green-100',
  error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-900 dark:text-red-100',
  warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-900 dark:text-yellow-100',
  info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-900 dark:text-blue-100',
  'content-update': 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800 text-purple-900 dark:text-purple-100',
}

const iconColorMap = {
  success: 'text-green-500',
  error: 'text-red-500',
  warning: 'text-yellow-500',
  info: 'text-blue-500',
  'content-update': 'text-purple-500',
}

export function Toast({ toast, onClose }: ToastProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isLeaving, setIsLeaving] = useState(false)

  const Icon = iconMap[toast.type]
  const duration = toast.duration ?? 5000 // 5 seconds default

  useEffect(() => {
    // Animate in
    const showTimer = setTimeout(() => setIsVisible(true), 100)

    // Auto-hide timer
    let hideTimer: NodeJS.Timeout | null = null
    if (duration > 0) {
      hideTimer = setTimeout(() => {
        handleClose()
      }, duration)
    }

    return () => {
      clearTimeout(showTimer)
      if (hideTimer) clearTimeout(hideTimer)
    }
  }, [duration])

  const handleClose = () => {
    setIsLeaving(true)
    setTimeout(() => {
      onClose(toast.id)
      toast.onClose?.()
    }, 300) // Match animation duration
  }

  const handleAction = () => {
    if (toast.actionUrl && typeof window !== 'undefined') {
      window.open(toast.actionUrl, '_blank')
    }
    toast.onAction?.()
    handleClose()
  }

  return (
    <div
      className={cn(
        'flex items-start p-4 rounded-lg border shadow-lg max-w-sm transition-all duration-300 ease-in-out transform',
        colorMap[toast.type],
        isVisible && !isLeaving ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0',
        'hover:shadow-xl'
      )}
    >
      {/* Icon */}
      <Icon className={cn('w-5 h-5 mr-3 mt-0.5 flex-shrink-0', iconColorMap[toast.type])} />
      
      {/* Content */}
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-semibold mb-1 truncate">
          {toast.title}
        </h4>
        <p className="text-sm opacity-90 leading-relaxed">
          {toast.message}
        </p>
        
        {/* Action Button */}
        {(toast.actionLabel || toast.actionUrl) && (
          <button
            onClick={handleAction}
            className="mt-2 text-xs font-medium underline hover:no-underline focus:no-underline transition-all"
          >
            {toast.actionLabel || '자세히 보기'}
          </button>
        )}
      </div>

      {/* Close Button */}
      <button
        onClick={handleClose}
        className="ml-2 p-1 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors flex-shrink-0"
        aria-label="알림 닫기"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

interface ToastContainerProps {
  toasts: ToastData[]
  onClose: (id: string) => void
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left'
}

const positionMap = {
  'top-right': 'top-4 right-4',
  'top-left': 'top-4 left-4', 
  'bottom-right': 'bottom-4 right-4',
  'bottom-left': 'bottom-4 left-4',
}

export function ToastContainer({ 
  toasts, 
  onClose, 
  position = 'top-right' 
}: ToastContainerProps) {
  return (
    <div
      className={cn(
        'fixed z-[9999] flex flex-col gap-2 pointer-events-none',
        positionMap[position]
      )}
      style={{ maxHeight: 'calc(100vh - 2rem)' }}
    >
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <Toast toast={toast} onClose={onClose} />
        </div>
      ))}
    </div>
  )
}

// Custom hook for managing toasts
export function useToast() {
  const [toasts, setToasts] = useState<ToastData[]>([])

  const addToast = (toast: Omit<ToastData, 'id'>) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newToast: ToastData = { ...toast, id }
    
    setToasts(prev => [...prev, newToast])
    return id
  }

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }

  const clearAllToasts = () => {
    setToasts([])
  }

  // Helper methods for different toast types
  const toast = {
    success: (title: string, message: string, options?: Partial<ToastData>) => 
      addToast({ type: 'success', title, message, ...options }),
      
    error: (title: string, message: string, options?: Partial<ToastData>) => 
      addToast({ type: 'error', title, message, duration: 7000, ...options }),
      
    warning: (title: string, message: string, options?: Partial<ToastData>) => 
      addToast({ type: 'warning', title, message, ...options }),
      
    info: (title: string, message: string, options?: Partial<ToastData>) => 
      addToast({ type: 'info', title, message, ...options }),
      
    contentUpdate: (title: string, message: string, moduleId?: string, options?: Partial<ToastData>) => 
      addToast({ 
        type: 'content-update', 
        title, 
        message, 
        actionUrl: moduleId ? `/modules/${moduleId}` : undefined,
        actionLabel: '업데이트 확인',
        ...options 
      }),
  }

  return {
    toasts,
    toast,
    removeToast,
    clearAllToasts,
  }
}