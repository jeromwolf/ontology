'use client';

export default function Chapter13() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">간단한 지표 활용하기 📊</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          복잡한 지표 대신 몇 가지 기본 지표만 알아도 충분합니다!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 이동평균선</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            가장 기본적이면서도 중요한 지표입니다. 일정 기간의 평균 주가를 선으로 연결한 것입니다.
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-red-600 dark:text-red-400 mb-2">20일선</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                단기 추세를 나타냅니다. 
                주가가 20일선 위에 있으면 단기 상승 추세
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">60일선</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                중기 추세를 나타냅니다. 
                지지선 역할을 하는 경우가 많습니다
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">120일선</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                장기 추세를 나타냅니다. 
                강력한 지지/저항선 역할
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 거래량 지표</h2>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">거래량은 주가의 동반자</h3>
          <div className="space-y-3 text-gray-700 dark:text-gray-300">
            <p>💪 <strong>강한 상승</strong>: 주가 상승 + 거래량 증가</p>
            <p>💪 <strong>강한 하락</strong>: 주가 하락 + 거래량 증가</p>
            <p>🤔 <strong>의심스러운 움직임</strong>: 주가 변동 + 거래량 감소</p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 RSI (상대강도지수)</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            0~100 사이의 값으로 과매수/과매도 상태를 판단하는 지표입니다.
          </p>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-red-100 dark:bg-red-900/30 rounded">
              <span>RSI 70 이상</span>
              <span className="font-bold text-red-600 dark:text-red-400">과매수 → 조정 가능성</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-green-100 dark:bg-green-900/30 rounded">
              <span>RSI 30~70</span>
              <span className="font-bold text-green-600 dark:text-green-400">정상 구간</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-blue-100 dark:bg-blue-900/30 rounded">
              <span>RSI 30 이하</span>
              <span className="font-bold text-blue-600 dark:text-blue-400">과매도 → 반등 가능성</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 초보자 사용법</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>1. 이동평균선을 먼저 익히세요 (20일, 60일, 120일)</li>
            <li>2. 거래량 증감을 항상 함께 확인하세요</li>
            <li>3. RSI는 보조 지표로만 활용하세요</li>
            <li>4. 한 가지 지표만 믿지 말고 여러 지표를 종합 판단하세요</li>
          </ul>
        </div>
      </section>
    </div>
  )
}