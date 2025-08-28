'use client';

// LLM 한계점을 설명하는 섹션 컴포넌트
export default function LLMLimitations() {
  return (
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
  );
}