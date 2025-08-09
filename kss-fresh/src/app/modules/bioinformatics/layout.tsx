import { bioinformaticsMetadata } from './metadata'

export default function BioinformaticsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50 dark:from-gray-900 dark:via-emerald-950 dark:to-teal-950">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-emerald-200 dark:border-emerald-800">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent">
              {bioinformaticsMetadata.name}
            </h1>
            <p className="text-gray-600 dark:text-gray-300 mt-2">
              {bioinformaticsMetadata.description}
            </p>
            <div className="flex gap-4 mt-4 text-sm">
              <span className="px-3 py-1 bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 rounded-full">
                {bioinformaticsMetadata.category}
              </span>
              <span className="px-3 py-1 bg-teal-100 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 rounded-full">
                {bioinformaticsMetadata.duration}
              </span>
              <span className="px-3 py-1 bg-cyan-100 dark:bg-cyan-900/50 text-cyan-700 dark:text-cyan-300 rounded-full">
                난이도: {bioinformaticsMetadata.difficulty === 'advanced' ? '고급' : '중급'}
              </span>
            </div>
          </div>
        </header>
        <main>{children}</main>
      </div>
    </div>
  )
}