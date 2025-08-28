'use client';

// 실제 RAG 시스템 사례를 보여주는 섹션 컴포넌트
export default function RAGExamples() {
  return (
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
  );
}