'use client';

// RAG 등장과 핵심 개념을 설명하는 섹션 컴포넌트
export default function RAGIntroduction() {
  return (
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
  );
}