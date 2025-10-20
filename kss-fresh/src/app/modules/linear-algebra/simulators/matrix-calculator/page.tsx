import dynamic from 'next/dynamic'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

const MatrixCalculator = dynamic(
  () => import('@/components/linear-algebra-simulators/MatrixCalculator'),
  { ssr: false }
)

export default function MatrixCalculatorPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <Link
          href="/modules/linear-algebra"
          className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-700 mb-6 text-white"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="text-sm">모듈로 돌아가기</span>
        </Link>

        <MatrixCalculator />
      </div>
    </div>
  )
}
