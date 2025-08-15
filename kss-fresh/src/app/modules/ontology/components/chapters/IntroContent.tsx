'use client';

export default function IntroContent() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">온톨로지 시뮬레이터에 오신 것을 환영합니다!</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg leading-relaxed">
            이 시뮬레이터는 복잡한 온톨로지 개념을 <strong>직접 체험하며 학습</strong>할 수 있도록 설계되었습니다.
            이론적 설명과 함께 <strong>인터랙티브한 실습 도구</strong>를 제공하여, 
            온톨로지의 핵심 개념부터 실전 활용까지 단계별로 마스터할 수 있습니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">학습 여정</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">이론 파트</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 온톨로지의 철학적 배경</li>
              <li>• 시맨틱 웹과 링크드 데이터</li>
              <li>• RDF, RDFS, OWL 표준</li>
              <li>• SPARQL 쿼리 언어</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">실습 파트</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• RDF Triple Editor 사용</li>
              <li>• 3D 지식 그래프 시각화</li>
              <li>• 실제 온톨로지 구축</li>
              <li>• 추론 엔진 활용</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">시뮬레이터 특징</h2>
        <div className="space-y-4">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-2xl">🎯</span>
            </div>
            <div>
              <h4 className="font-semibold mb-1">체험 중심 학습</h4>
              <p className="text-gray-600 dark:text-gray-400">
                단순히 읽는 것이 아닌, 직접 만들고 실험하며 개념을 체득합니다.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-2xl">🔄</span>
            </div>
            <div>
              <h4 className="font-semibold mb-1">즉각적인 피드백</h4>
              <p className="text-gray-600 dark:text-gray-400">
                작성한 온톨로지의 유효성을 실시간으로 검증하고 추론 결과를 확인합니다.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-2xl">📊</span>
            </div>
            <div>
              <h4 className="font-semibold mb-1">시각화 도구</h4>
              <p className="text-gray-600 dark:text-gray-400">
                복잡한 관계를 2D/3D 그래프로 시각화하여 직관적으로 이해할 수 있습니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-4">시작하기 전에</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          이 과정은 프로그래밍 경험이 없어도 따라올 수 있도록 설계되었습니다.
          하지만 다음 개념에 대한 기초적인 이해가 있다면 더욱 도움이 됩니다:
        </p>
        <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
          <li>데이터베이스의 기본 개념 (테이블, 관계)</li>
          <li>웹의 작동 원리 (URL, HTTP)</li>
          <li>논리적 사고와 추론</li>
        </ul>
      </section>

      <div className="mt-12 p-6 bg-indigo-600 text-white rounded-xl text-center">
        <p className="text-lg font-medium">
          준비되셨나요? Chapter 1부터 온톨로지의 세계로 함께 떠나봅시다! 🚀
        </p>
      </div>
    </div>
  )
}