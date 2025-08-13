'use client'

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">기술적 분석의 3대 가정</h2>
        <div className="grid gap-4 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">1. 시장은 모든 것을 반영한다</h3>
            <p className="text-gray-700 dark:text-gray-300">
              주가에는 이미 모든 정보(기업 실적, 경제 상황, 투자심리 등)가 반영되어 있다.
            </p>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">2. 가격은 추세를 따라 움직인다</h3>
            <p className="text-gray-700 dark:text-gray-300">
              주가는 무작위로 움직이지 않고 일정한 패턴과 추세를 보인다.
            </p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">3. 역사는 반복된다</h3>
            <p className="text-gray-700 dark:text-gray-300">
              과거의 패턴은 미래에도 반복될 가능성이 높다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">캔들스틱 패턴</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          캔들스틱 차트는 일정 기간의 시가, 고가, 저가, 종가 정보를 하나의 캔들로 표현합니다.
          캔들의 모양과 조합으로 시장 심리와 향후 방향을 예측할 수 있습니다.
        </p>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-3">🟢 강세 반전 패턴</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
              <li><strong>해머:</strong> 하락 후 나타나는 긴 아래꼬리</li>
              <li><strong>불룩한 바닥:</strong> 연속된 두 개의 상승 캔들</li>
              <li><strong>조조별:</strong> 갭 하락 후 상승 마감</li>
              <li><strong>역망치:</strong> 긴 위꼬리를 가진 작은 몸통</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">🔴 약세 반전 패턴</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
              <li><strong>교수형:</strong> 상승 후 나타나는 긴 아래꼬리</li>
              <li><strong>먹구름:</strong> 연속된 두 개의 하락 캔들</li>
              <li><strong>저녁별:</strong> 갭 상승 후 하락 마감</li>
              <li><strong>유성:</strong> 긴 위꼬리를 가진 작은 몸통</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주요 기술적 지표</h2>
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">📈 이동평균선 (Moving Average)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              일정 기간의 주가를 평균내어 추세를 파악하는 가장 기본적인 지표
            </p>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <strong>단기 이평선 (5일, 20일)</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">단기 추세와 지지/저항</span>
              </div>
              <div>
                <strong>중기 이평선 (60일, 120일)</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">중기 추세 판단</span>
              </div>
              <div>
                <strong>장기 이평선 (200일, 300일)</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">장기 추세와 강력한 지지/저항</span>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">⚡ RSI (Relative Strength Index)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              0~100 사이 값으로 과매수/과매도 상태를 판단하는 모멘텀 오실레이터
            </p>
            <div className="flex items-center justify-between text-sm">
              <span className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-3 py-1 rounded">
                과매도 (30 이하)
              </span>
              <span className="text-gray-600 dark:text-gray-400">
                적정 구간 (30-70)
              </span>
              <span className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-3 py-1 rounded">
                과매수 (70 이상)
              </span>
            </div>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">🌊 MACD (Moving Average Convergence Divergence)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              단기 이평선과 장기 이평선의 차이로 추세 변화를 포착하는 지표
            </p>
            <div className="space-y-2 text-sm">
              <div><strong>MACD 선:</strong> 12일 지수이평 - 26일 지수이평</div>
              <div><strong>시그널 선:</strong> MACD의 9일 지수이평</div>
              <div><strong>매매 신호:</strong> MACD가 시그널 선을 상향/하향 돌파</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">지지선과 저항선</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/10 dark:to-orange-900/10 rounded-xl p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">📈 지지선 (Support Line)</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                주가 하락을 막아주는 가격대. 매수세가 강해지는 구간
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 과거 저점들을 연결한 선</li>
                <li>• 심리적 가격대 (1만원, 5만원 등)</li>
                <li>• 주요 이동평균선</li>
                <li>• 돌파 시 추가 하락 가능성</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">📉 저항선 (Resistance Line)</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                주가 상승을 막는 가격대. 매도세가 강해지는 구간
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 과거 고점들을 연결한 선</li>
                <li>• 심리적 저항 가격대</li>
                <li>• 기술적 지표의 과매수 구간</li>
                <li>• 돌파 시 추가 상승 가능성</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-white dark:bg-gray-800 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">💡 실전 매매 전략</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <strong className="text-emerald-600 dark:text-emerald-400">매수 타이밍</strong>
                <ul className="mt-1 space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 지지선 근처에서 반등 확인</li>
                  <li>• 저항선 돌파 후 재테스트</li>
                  <li>• 거래량 증가와 함께 신호 확인</li>
                </ul>
              </div>
              <div>
                <strong className="text-red-600 dark:text-red-400">매도 타이밍</strong>
                <ul className="mt-1 space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 저항선 근처에서 상승 둔화</li>
                  <li>• 지지선 하향 돌파</li>
                  <li>• 거래량 감소와 함께 약세 확인</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}