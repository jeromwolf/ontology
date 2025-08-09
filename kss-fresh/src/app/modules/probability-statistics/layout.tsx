import type { Metadata } from 'next'
import { probabilityStatisticsModule } from './metadata'

export const metadata: Metadata = {
  title: `${probabilityStatisticsModule.name} | KSS`,
  description: probabilityStatisticsModule.description,
}

export default function ProbabilityStatisticsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-pink-50 dark:from-gray-900 dark:via-purple-900/10 dark:to-pink-900/10">
      <div className="absolute inset-0 bg-grid-gray-100/50 dark:bg-grid-gray-800/50 [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]" />
      <div className="relative z-10">
        {children}
      </div>
    </div>
  )
}