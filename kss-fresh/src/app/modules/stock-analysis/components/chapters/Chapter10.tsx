'use client'

export default function Chapter10() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">주문 유형 마스터하기 📊</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          시장가 주문과 지정가 주문의 차이점을 이해하고 상황에 맞게 활용해보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏃 시장가 주문 (Market Order)</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">즉시 체결을 원할 때</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            현재 시장에서 거래되는 가격으로 즉시 매수/매도하는 주문 방식입니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-2">✅ 장점</h4>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 100% 체결 보장</li>
                <li>• 빠른 거래 실행</li>
                <li>• 급한 매수/매도 시 유용</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">❌ 단점</h4>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 정확한 가격 예측 불가</li>
                <li>• 슬리피지 발생 가능</li>
                <li>• 변동성 큰 시장에서 위험</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 지정가 주문 (Limit Order)</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">원하는 가격에 거래하고 싶을 때</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            원하는 특정 가격을 지정하여 그 가격 이하로 매수하거나 이상으로 매도하는 방식입니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-2">✅ 장점</h4>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 원하는 가격에 거래</li>
                <li>• 가격 통제 가능</li>
                <li>• 계획적인 투자 가능</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">❌ 단점</h4>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 체결되지 않을 수 있음</li>
                <li>• 기회를 놓칠 가능성</li>
                <li>• 인내심 필요</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 상황별 주문 전략</h2>
        <div className="grid gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">시장가 주문이 좋은 경우</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>📈 급등/급락장에서 빠른 대응이 필요할 때</li>
              <li>💰 유동성이 풍부한 대형주 거래 시</li>
              <li>⏰ 장 마감 직전 반드시 매도해야 할 때</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">지정가 주문이 좋은 경우</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>🎯 목표 매수/매도 가격이 명확할 때</li>
              <li>📊 중소형주 등 유동성이 적은 종목</li>
              <li>⏳ 시간적 여유가 있을 때</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}