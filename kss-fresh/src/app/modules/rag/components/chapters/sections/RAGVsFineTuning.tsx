'use client';

// RAG vs Fine-tuning 비교 테이블 섹션 컴포넌트
export default function RAGVsFineTuning() {
  return (
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
  );
}