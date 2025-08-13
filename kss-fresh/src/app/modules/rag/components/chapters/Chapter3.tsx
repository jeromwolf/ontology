'use client'

import EmbeddingVisualizer from '../EmbeddingVisualizer'

// Chapter 3: Embeddings
export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">임베딩이란?</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          임베딩은 텍스트를 고차원 벡터 공간의 점으로 변환하는 과정입니다.
          의미가 유사한 텍스트는 벡터 공간에서 가까이 위치하게 됩니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">임베딩 시각화</h2>
        <EmbeddingVisualizer />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">임베딩 모델 비교</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">모델</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">차원</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">특징</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">비용</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">OpenAI text-embedding-3</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">1536-3072</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">높은 정확도</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">$0.00002/1K tokens</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">Cohere embed-v3</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">1024</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">다국어 지원</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">$0.00010/1K tokens</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">BGE-M3</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">1024</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">오픈소스</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">무료 (자체 호스팅)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}