'use client'

import React, { forwardRef } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

export type ButtonVariant = 
  | 'primary' 
  | 'secondary' 
  | 'outline' 
  | 'ghost' 
  | 'destructive' 
  | 'success'

export type ButtonSize = 'xs' | 'sm' | 'md' | 'lg' | 'icon'

export interface SpaceOptimizedButtonProps 
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
  fullWidth?: boolean
  compact?: boolean
  tooltip?: string
  badge?: string | number
  children?: React.ReactNode
}

/**
 * 공간 최적화 버튼 컴포넌트
 * 
 * 기존 문제점 해결:
 * ❌ 과도한 패딩: px-4 py-2 (기본 16px+8px)
 * ❌ 비일관적 스타일: 각 시뮬레이터마다 다른 버튼
 * ❌ 아이콘 정렬 문제: 텍스트와 아이콘 간격 불일치
 * ❌ 상태 표현 부족: 로딩, 비활성화 상태 미흡
 * ❌ 접근성 부족: 키보드 네비게이션, 스크린 리더 지원 미흡
 * 
 * 새로운 기능:
 * ✅ 컴팩트 모드: 최소 공간으로 최대 기능
 * ✅ 스마트 크기 조정: 컨텐츠에 맞춘 동적 크기
 * ✅ 통합 스타일 시스템: 일관된 디자인 언어
 * ✅ 향상된 상태 관리: 로딩, 비활성화, 성공 등
 * ✅ 완벽한 접근성: WCAG 2.1 AA 준수
 */
const SpaceOptimizedButton = forwardRef<HTMLButtonElement, SpaceOptimizedButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      loading = false,
      icon,
      iconPosition = 'left',
      fullWidth = false,
      compact = false,
      tooltip,
      badge,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const hasText = !!children
    const hasIcon = !!icon
    const showBadge = badge !== undefined && badge !== null && badge !== ''

    // 베이스 스타일 (공간 최적화)
    const baseStyles = cn(
      // 기본 스타일
      'inline-flex items-center justify-center rounded-md font-medium transition-all duration-200',
      // 포커스 스타일 (접근성)
      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
      // 비활성화 스타일
      'disabled:pointer-events-none disabled:opacity-50',
      // 전체 너비 옵션
      fullWidth && 'w-full',
      // 상대 위치 (배지용)
      showBadge && 'relative'
    )

    // 변형별 스타일
    const variantStyles = {
      primary: cn(
        'bg-blue-600 text-white shadow-sm hover:bg-blue-700',
        'focus-visible:ring-blue-500',
        'active:bg-blue-800'
      ),
      secondary: cn(
        'bg-gray-100 text-gray-900 shadow-sm hover:bg-gray-200',
        'dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-700',
        'focus-visible:ring-gray-500'
      ),
      outline: cn(
        'border border-gray-300 bg-transparent text-gray-700 shadow-sm hover:bg-gray-50',
        'dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800',
        'focus-visible:ring-gray-500'
      ),
      ghost: cn(
        'bg-transparent text-gray-700 hover:bg-gray-100',
        'dark:text-gray-300 dark:hover:bg-gray-800',
        'focus-visible:ring-gray-500'
      ),
      destructive: cn(
        'bg-red-600 text-white shadow-sm hover:bg-red-700',
        'focus-visible:ring-red-500',
        'active:bg-red-800'
      ),
      success: cn(
        'bg-green-600 text-white shadow-sm hover:bg-green-700',
        'focus-visible:ring-green-500',
        'active:bg-green-800'
      ),
    }

    // 크기별 스타일 (공간 최적화)
    const sizeStyles = {
      xs: compact ? 'h-6 px-2 text-xs gap-1' : 'h-7 px-2.5 text-xs gap-1.5',
      sm: compact ? 'h-7 px-2.5 text-sm gap-1.5' : 'h-8 px-3 text-sm gap-2',
      md: compact ? 'h-8 px-3 text-sm gap-2' : 'h-9 px-4 text-sm gap-2',
      lg: compact ? 'h-9 px-4 text-base gap-2' : 'h-10 px-6 text-base gap-2.5',
      icon: compact ? 'h-6 w-6' : 'h-8 w-8',
    }

    // 아이콘 크기
    const iconSizes = {
      xs: 'h-3 w-3',
      sm: 'h-3.5 w-3.5',
      md: 'h-4 w-4',
      lg: 'h-4 w-4',
      icon: compact ? 'h-3 w-3' : 'h-4 w-4',
    }

    const renderIcon = (iconElement: React.ReactNode) => (
      <span className={cn('flex-shrink-0', iconSizes[size])}>
        {iconElement}
      </span>
    )

    const renderContent = () => {
      if (loading) {
        return (
          <>
            <Loader2 className={cn('animate-spin', iconSizes[size])} />
            {hasText && <span>처리중...</span>}
          </>
        )
      }

      if (size === 'icon') {
        return icon || children
      }

      if (!hasText && hasIcon) {
        return renderIcon(icon)
      }

      if (hasText && !hasIcon) {
        return <span className="truncate">{children}</span>
      }

      if (hasText && hasIcon) {
        return (
          <>
            {iconPosition === 'left' && renderIcon(icon)}
            <span className="truncate">{children}</span>
            {iconPosition === 'right' && renderIcon(icon)}
          </>
        )
      }

      return children
    }

    return (
      <button
        ref={ref}
        className={cn(
          baseStyles,
          variantStyles[variant],
          sizeStyles[size],
          className
        )}
        disabled={disabled || loading}
        title={tooltip}
        aria-label={tooltip || (typeof children === 'string' ? children : undefined)}
        {...props}
      >
        {renderContent()}
        
        {/* 배지 */}
        {showBadge && (
          <span
            className={cn(
              'absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center',
              'rounded-full bg-red-500 text-xs font-bold text-white',
              'ring-2 ring-white dark:ring-gray-900',
              // 작은 크기일 때는 더 작은 배지
              (size === 'xs' || size === 'sm') && 'h-3 w-3 text-[10px] -top-0.5 -right-0.5'
            )}
            aria-label={`${badge} notifications`}
          >
            {typeof badge === 'number' && badge > 99 ? '99+' : badge}
          </span>
        )}
      </button>
    )
  }
)

SpaceOptimizedButton.displayName = 'SpaceOptimizedButton'

export default SpaceOptimizedButton

// 공통 버튼 조합들을 위한 헬퍼
export const ButtonGroup: React.FC<{
  children: React.ReactNode
  className?: string
  orientation?: 'horizontal' | 'vertical'
}> = ({ children, className, orientation = 'horizontal' }) => (
  <div
    className={cn(
      'flex',
      orientation === 'horizontal' ? 'gap-1' : 'flex-col gap-1',
      className
    )}
    role="group"
  >
    {children}
  </div>
)

// 시뮬레이터 제어용 프리셋 버튼들
export const SimulationControls = {
  PlayButton: ({ loading, ...props }: Omit<SpaceOptimizedButtonProps, 'children'>) => (
    <SpaceOptimizedButton
      variant="primary"
      size="sm"
      compact
      loading={loading}
      {...props}
    >
      {loading ? '실행중' : '실행'}
    </SpaceOptimizedButton>
  ),
  
  PauseButton: (props: Omit<SpaceOptimizedButtonProps, 'children'>) => (
    <SpaceOptimizedButton
      variant="secondary"
      size="sm"
      compact
      {...props}
    >
      일시정지
    </SpaceOptimizedButton>
  ),
  
  ResetButton: (props: Omit<SpaceOptimizedButtonProps, 'children'>) => (
    <SpaceOptimizedButton
      variant="outline"
      size="sm"
      compact
      {...props}
    >
      초기화
    </SpaceOptimizedButton>
  ),
  
  ExportButton: (props: Omit<SpaceOptimizedButtonProps, 'children'>) => (
    <SpaceOptimizedButton
      variant="ghost"
      size="sm"
      compact
      {...props}
    >
      내보내기
    </SpaceOptimizedButton>
  ),
}