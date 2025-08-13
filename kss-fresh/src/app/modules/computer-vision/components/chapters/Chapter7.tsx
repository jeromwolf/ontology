'use client';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">얼굴 인식 파이프라인</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          얼굴 인식은 여러 단계로 구성된 복잡한 프로세스입니다. 
          검출, 정렬, 특징 추출, 매칭의 단계를 거칩니다.
        </p>

        <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-teal-900 dark:text-teal-100">처리 단계</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">1</div>
              <p className="font-medium">얼굴 검출</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">이미지에서 얼굴 영역 찾기</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">2</div>
              <p className="font-medium">얼굴 정렬</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">랜드마크 기반 정규화</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">3</div>
              <p className="font-medium">특징 추출</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">얼굴 임베딩 벡터 생성</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">4</div>
              <p className="font-medium">신원 확인</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">벡터 유사도 비교</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">감정 인식</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          얼굴 표정으로부터 감정을 인식하는 기술은 HCI, 마케팅, 의료 등 다양한 분야에서 활용됩니다.
        </p>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
          <h3 className="font-semibold mb-3">7가지 기본 감정</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😊 행복</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😢 슬픔</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😠 분노</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😱 두려움</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😲 놀람</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">🤢 혐오</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😐 중립</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">🤔 기타</div>
          </div>
        </div>
      </section>
    </div>
  );
}