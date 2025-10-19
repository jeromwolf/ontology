import * as React from 'react'

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'destructive'
}

const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className = '', variant = 'default', ...props }, ref) => {
    const variants = {
      default: 'bg-blue-600 text-white',
      secondary: 'bg-gray-200 text-gray-900 dark:bg-gray-700 dark:text-gray-100',
      destructive: 'bg-red-600 text-white',
    }

    return (
      <div
        ref={ref}
        className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${variants[variant]} ${className}`}
        {...props}
      />
    )
  }
)
Badge.displayName = 'Badge'

export { Badge }
