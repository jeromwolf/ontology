'use client'

import { ReactNode } from 'react'

// Chapter 1: What is RAG?
export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">LLM의 한계</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          대규모 언어 모델(LLM)은 놀라운 능력을 보여주지만, 몇 가지 근본적인 한계가 있습니다:
        </p>
        
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">할루시네이션</h3>
            <p className="text-gray-700 dark:text-gray-300">
              학습하지 않은 정보에 대해 그럴듯하지만 틀린 답변을 생성하는 현상
            </p>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">최신 정보 부족</h3>
            <p className="text-gray-700 dark:text-gray-300">
              학습 데이터 기준일 이후의 정보는 알 수 없음
            </p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">도메인 특화 지식</h3>
            <p className="text-gray-700 dark:text-gray-300">
              기업 내부 문서나 특정 도메인 지식은 학습되지 않음
            </p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">소스 추적 불가</h3>
            <p className="text-gray-700 dark:text-gray-300">
              생성된 답변의 출처를 확인할 수 없어 신뢰성 검증 어려움
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RAG의 등장</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          RAG(Retrieval-Augmented Generation)는 이러한 LLM의 한계를 극복하기 위해 등장했습니다.
          외부 지식 베이스에서 관련 정보를 검색하여 LLM에 제공함으로써 더 정확하고 신뢰할 수 있는 답변을 생성합니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6">
          <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-4">RAG의 핵심 아이디어</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">1.</span>
              <div>
                <strong>검색(Retrieval)</strong>: 사용자 질문과 관련된 문서를 지식 베이스에서 찾기
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">2.</span>
              <div>
                <strong>증강(Augmentation)</strong>: 검색된 문서를 LLM의 컨텍스트로 제공
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">3.</span>
              <div>
                <strong>생성(Generation)</strong>: 컨텍스트를 바탕으로 정확한 답변 생성
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RAG vs Fine-tuning</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">비교 항목</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">RAG</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">Fine-tuning</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">지식 업데이트</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-green-600 dark:text-green-400">실시간 가능</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">재학습 필요</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">비용</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-green-600 dark:text-green-400">상대적으로 저렴</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">GPU 비용 높음</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">소스 추적</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-green-600 dark:text-green-400">가능</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">불가능</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">정확도 제어</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-green-600 dark:text-green-400">문서 기반 100%</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-yellow-600 dark:text-yellow-400">확률적</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실제 RAG 시스템 사례</h2>
        <div className="grid gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Microsoft Copilot</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Office 문서, 이메일, 캘린더 등 기업 데이터를 활용한 업무 보조 AI
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Perplexity AI</h3>
            <p className="text-gray-600 dark:text-gray-400">
              실시간 웹 검색을 통해 최신 정보를 제공하는 AI 검색 엔진
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">ChatGPT with Browsing</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Bing 검색을 통해 실시간 정보를 보강한 답변 생성
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}