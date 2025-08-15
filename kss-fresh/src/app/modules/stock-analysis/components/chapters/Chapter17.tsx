'use client';

export default function Chapter17() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">매수와 매도 타이밍 ⏰</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          언제 사고 언제 파는지가 투자 성과를 좌우합니다!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 매수 타이밍</h2>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">좋은 매수 신호들</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">🎯 기술적 신호</h4>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 지지선에서 반등할 때</li>
                <li>• 이동평균선을 상향 돌파할 때</li>
                <li>• 거래량이 증가하며 상승할 때</li>
                <li>• RSI가 30 이하에서 반등할 때</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">📊 기본적 신호</h4>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 좋은 실적 발표 후</li>
                <li>• 신제품 출시나 신규 사업 진출</li>
                <li>• 동종업계 대비 저평가될 때</li>
                <li>• 배당 기준일 전</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📉 매도 타이밍</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">매도를 고려해야 할 때</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">⚠️ 위험 신호</h4>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 저항선에서 하락 반전할 때</li>
                <li>• 이동평균선을 하향 이탈할 때</li>
                <li>• 거래량이 증가하며 하락할 때</li>
                <li>• RSI가 70 이상에서 조정될 때</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">🏆 목표 달성</h4>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 설정한 목표 수익률 달성</li>
                <li>• 기업 가치 대비 과도하게 상승</li>
                <li>• 더 좋은 투자 기회 발견</li>
                <li>• 투자 목적 달성 (결혼, 내집마련 등)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 분할 매수/매도 전략</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">분할 매수</h4>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                한 번에 다 사지 말고 3-4번에 나누어 매수
              </p>
              <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                <div>1차: 30% 매수</div>
                <div>2차: 하락 시 30% 추가</div>
                <div>3차: 더 하락 시 40% 추가</div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">분할 매도</h4>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                목표가 도달 시 단계적으로 매도
              </p>
              <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                <div>1차: +20% 달성 시 30% 매도</div>
                <div>2차: +40% 달성 시 40% 매도</div>
                <div>3차: +60% 달성 시 나머지 매도</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 초보자를 위한 실전 규칙</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <div className="space-y-3">
            <div className="p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded">
              <strong>손절매 규칙:</strong> 매수가 대비 -10% 하락 시 무조건 매도
            </div>
            <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded">
              <strong>익절 규칙:</strong> +20% 상승 시 일부 매도로 원금 회수
            </div>
            <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded">
              <strong>홀딩 규칙:</strong> 기업 기본 펀더멘털이 좋으면 장기 보유
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-900/30 rounded">
            <strong>황금 규칙:</strong> 감정에 휘둘리지 말고 미리 정한 규칙을 지키세요!
          </div>
        </div>
      </section>
    </div>
  )
}