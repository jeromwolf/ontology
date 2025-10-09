import Link from 'next/link'
import { ArrowRight } from 'lucide-react'

export default function Section7() {
  return (
    <section className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl p-8 text-white">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold mb-2">🎮 프롬프트 엔지니어링 시뮬레이터</h3>
          <p className="text-emerald-100">다양한 프롬프트 기법을 실시간으로 테스트하세요</p>
        </div>
        <Link
          href="/modules/rag/simulators/prompt-engineering-lab"
          className="inline-flex items-center gap-2 bg-white text-emerald-600 px-6 py-3 rounded-lg font-semibold hover:bg-emerald-50 transition-colors shadow-lg"
        >
          시뮬레이터 열기
          <ArrowRight size={20} />
        </Link>
      </div>
    </section>
  )
}
