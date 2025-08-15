'use client';

export default function Chapter12() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">기초 차트 읽기 📈</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          주식 차트의 기본 구성요소를 이해하고 간단한 패턴을 읽어보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🕯️ 캔들차트 기초</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            캔들차트는 하루 동안의 주가 움직임을 한눈에 보여주는 차트입니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-red-600 dark:text-red-400 mb-2">🔴 양봉 (상승)</h3>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 시가보다 종가가 높음</li>
                <li>• 빨간색 또는 흰색으로 표시</li>
                <li>• 매수세가 강했음을 의미</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">🔵 음봉 (하락)</h3>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 시가보다 종가가 낮음</li>
                <li>• 파란색 또는 검은색으로 표시</li>
                <li>• 매도세가 강했음을 의미</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 거래량의 중요성</h2>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">거래량은 주가의 신뢰도</h3>
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li>📈 <strong>상승 + 거래량 증가</strong> = 강한 상승 신호</li>
            <li>📉 <strong>하락 + 거래량 증가</strong> = 강한 하락 신호</li>
            <li>➡️ <strong>변동 + 거래량 감소</strong> = 추세 약화 신호</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📏 지지선과 저항선</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">지지선</h3>
            <p className="text-gray-600 dark:text-gray-400">
              주가가 하락할 때 멈추는 가격대. 
              이 가격에서 매수 의욕이 생겨 더 이상 떨어지지 않습니다.
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-2">저항선</h3>
            <p className="text-gray-600 dark:text-gray-400">
              주가가 상승할 때 멈추는 가격대. 
              이 가격에서 매도 의욕이 생겨 더 이상 오르지 않습니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 초보자를 위한 팁</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 일봉보다는 주봉, 월봉으로 큰 그림을 먼저 보세요</li>
            <li>• 차트만 보지 말고 기업의 실적도 함께 확인하세요</li>
            <li>• 복잡한 기술적 분석보다는 단순한 패턴부터 익히세요</li>
            <li>• 과거 차트로 연습해보며 감을 익히세요</li>
          </ul>
        </div>
      </section>
    </div>
  )
}