'use client';

export default function Chapter16() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">간단한 기업 가치 평가 💰</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          복잡한 계산 없이도 기업이 비싸게 사는 건지 싸게 사는 건지 알 수 있어요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏷️ PER - 가격 대비 수익 비율</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <div className="text-center mb-4">
            <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
              PER = 주가 ÷ 주당순이익
            </div>
            <p className="text-gray-700 dark:text-gray-300">
              현재 주가가 1년간 벌어들이는 이익의 몇 배인지를 나타냅니다
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-4 text-center">
              <h3 className="font-semibold text-red-800 dark:text-red-200">PER 20배 이상</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                비싸다는 신호<br/>
                신중한 검토 필요
              </p>
            </div>
            <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded-lg p-4 text-center">
              <h3 className="font-semibold text-yellow-800 dark:text-yellow-200">PER 10-20배</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                적정 수준<br/>
                업종별 차이 고려
              </p>
            </div>
            <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-4 text-center">
              <h3 className="font-semibold text-green-800 dark:text-green-200">PER 10배 이하</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                저평가 가능성<br/>
                이유 확인 필요
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📚 PBR - 가격 대비 자산 비율</h2>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
          <div className="text-center mb-4">
            <div className="text-3xl font-bold text-emerald-600 dark:text-emerald-400 mb-2">
              PBR = 주가 ÷ 주당순자산
            </div>
            <p className="text-gray-700 dark:text-gray-300">
              현재 주가가 기업이 보유한 순자산의 몇 배인지를 나타냅니다
            </p>
          </div>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-green-100 dark:bg-green-900/30 rounded">
              <span>PBR 1배 미만</span>
              <span className="font-bold text-green-600 dark:text-green-400">자산가치보다 저평가</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded">
              <span>PBR 1-2배</span>
              <span className="font-bold text-yellow-600 dark:text-yellow-400">적정 평가</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-red-100 dark:bg-red-900/30 rounded">
              <span>PBR 2배 이상</span>
              <span className="font-bold text-red-600 dark:text-red-400">고평가 가능성</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏭 업종별 특성 고려하기</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">성장주 (IT, 바이오)</h3>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• PER이 높아도 성장률 고려</li>
              <li>• 미래 수익성이 더 중요</li>
              <li>• PEG 비율 활용 권장</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">가치주 (은행, 제조)</h3>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• PBR이 더 중요한 지표</li>
              <li>• 배당 수익률 고려</li>
              <li>• 안정성이 핵심</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 실전 활용법</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-3">3단계 체크리스트</h3>
          <ol className="space-y-2 text-gray-700 dark:text-gray-300">
            <li><strong>1단계:</strong> 동종업계 평균 PER과 비교해보기</li>
            <li><strong>2단계:</strong> 과거 3년간 PER 변화 추이 확인</li>
            <li><strong>3단계:</strong> PBR도 함께 고려해서 종합 판단</li>
          </ol>
          
          <div className="mt-4 p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded">
            <strong>주의:</strong> 지표만으로 판단하지 말고 기업의 기본적인 경쟁력도 함께 고려하세요!
          </div>
        </div>
      </section>
    </div>
  )
}