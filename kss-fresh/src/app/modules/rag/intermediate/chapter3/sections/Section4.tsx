import { AlertCircle } from 'lucide-react'

export default function Section4() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
          <AlertCircle className="text-red-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.4 에러 처리 프롬프트</h2>
          <p className="text-gray-600 dark:text-gray-400">우아한 실패와 사용자 가이드</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border border-red-200 dark:border-red-700">
          <h3 className="font-bold text-red-800 dark:text-red-200 mb-4">상황별 에러 처리 템플릿</h3>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">📭 검색 결과 없음</h4>
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm">
                <pre className="whitespace-pre-wrap">
{`죄송합니다. "{query}"에 대한 관련 정보를 찾을 수 없습니다.

다음과 같이 시도해보세요:
• 다른 키워드나 동의어를 사용해보세요
• 더 구체적이거나 일반적인 용어로 검색해보세요
• 철자와 띄어쓰기를 확인해보세요

예시: "{suggested_query}"`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">🔀 모순된 정보</h4>
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm">
                <pre className="whitespace-pre-wrap">
{`검색된 문서들에서 상충하는 정보가 발견되었습니다:

관점 1: [출처: 문서A]
"{contradicting_info_1}"

관점 2: [출처: 문서B]
"{contradicting_info_2}"

💡 이러한 차이는 다음과 같은 이유일 수 있습니다:
• 정보의 업데이트 시점 차이
• 서로 다른 맥락이나 조건
• 출처의 관점 차이

최신 정보나 공식 출처를 확인하시기 바랍니다.`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">⚠️ 부분적 정보</h4>
              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm">
                <pre className="whitespace-pre-wrap">
{`요청하신 정보의 일부만 찾을 수 있었습니다:

✅ 찾은 정보:
{found_information}

❌ 찾지 못한 정보:
{missing_information}

추가 정보가 필요하시면:
1. 더 구체적인 질문을 해주세요
2. 다른 측면에서 접근해보세요
3. 관련 전문가나 공식 문서를 참조하세요`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
