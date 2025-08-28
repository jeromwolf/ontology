'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Brain } from 'lucide-react'

export default function Chapter2Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/advanced"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          고급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Brain size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 2: Multi-hop Reasoning</h1>
              <p className="text-purple-100 text-lg">복잡한 질문을 단계별로 분해하고 추론하는 고급 RAG 기법</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Multi-hop Reasoning 개요</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Multi-hop Reasoning은 복잡한 질문을 여러 단계로 분해하여 순차적으로 추론하는 고급 RAG 기법입니다.
          </p>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-3">핵심 특징</h3>
            <ul className="text-purple-700 dark:text-purple-300 space-y-2">
              <li>• 복잡한 질문을 하위 질문들로 분해</li>
              <li>• 각 단계별 증거 수집 및 추론</li>
              <li>• 중간 결과를 연결하여 최종 답변 생성</li>
              <li>• 추론 경로의 명시적 추적</li>
            </ul>
          </div>
        </section>

        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">구현 예시</h2>
          <div className="bg-gray-900 rounded-xl p-6">
            <pre className="text-sm text-gray-300">
{`class MultiHopRAG:
    def __init__(self, vector_db, llm_client):
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.reasoning_chain = []
        
    async def multi_hop_query(self, question: str, max_hops: int = 4):
        # 1. 질문 분해
        sub_questions = await self.decompose_question(question)
        
        # 2. 각 단계별 추론
        for hop, sub_q in enumerate(sub_questions):
            docs = await self.vector_db.search(sub_q)
            answer = await self.llm_client.generate(sub_q, docs)
            self.reasoning_chain.append({
                'hop': hop + 1,
                'question': sub_q,
                'answer': answer
            })
            
        # 3. 최종 답변 통합
        return self.synthesize_answer(question, self.reasoning_chain)`}
            </pre>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter1"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: GraphRAG 아키텍처
          </Link>
          
          <Link
            href="/modules/rag/advanced/chapter3"
            className="inline-flex items-center gap-2 bg-purple-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-purple-600 transition-colors"
          >
            다음: 분산 시스템 구축
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}