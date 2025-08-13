'use client'

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">주식시장의 구조</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          주식시장은 기업이 자금을 조달하고 투자자가 기업의 일부를 소유할 수 있게 해주는 
          중요한 금융 인프라입니다. 효율적인 시장에서 주가는 기업의 가치를 반영합니다.
        </p>
        
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">💼 기업 (발행자)</h3>
            <p className="text-gray-700 dark:text-gray-300">
              자금 조달을 위해 주식을 발행하고 기업 가치 증대를 통해 주주 이익 극대화
            </p>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">👥 투자자 (구매자)</h3>
            <p className="text-gray-700 dark:text-gray-300">
              자본 증식을 목적으로 기업의 성장 가능성을 분석하여 투자 결정
            </p>
          </div>
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">🏦 증권거래소</h3>
            <p className="text-gray-700 dark:text-gray-300">
              공정하고 투명한 거래 환경을 제공하는 시장 운영 기관
            </p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">🔍 중개기관</h3>
            <p className="text-gray-700 dark:text-gray-300">
              투자자와 시장을 연결하는 증권회사, 자산운용사 등의 금융기관
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">거래 시스템과 주문 유형</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/10 dark:to-orange-900/10 rounded-xl p-6 mb-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">호가창 (Order Book) 이해</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-red-600 dark:text-red-400 font-bold">📈</span>
              <div>
                <strong>매도호가</strong>: 판매자가 원하는 가격대별 물량 (Ask)
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-blue-600 dark:text-blue-400 font-bold">📉</span>
              <div>
                <strong>매수호가</strong>: 구매자가 원하는 가격대별 물량 (Bid)
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">⚡</span>
              <div>
                <strong>체결</strong>: 매수호가와 매도호가가 만나는 지점에서 거래 성사
              </div>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">시장가 주문 (Market Order)</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 현재 시장가격으로 즉시 체결</li>
              <li>• 체결 확실성 높음</li>
              <li>• 급한 거래 시 유리</li>
              <li>• 슬리피지 발생 가능</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">지정가 주문 (Limit Order)</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 원하는 가격 지정 후 대기</li>
              <li>• 가격 통제 가능</li>
              <li>• 체결되지 않을 위험</li>
              <li>• 장기 전략에 적합</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">투자 vs 투기</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">구분</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">투자 (Investment)</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">투기 (Speculation)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">시간 관점</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-emerald-600 dark:text-emerald-400">장기적 (1년+)</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">단기적 (일~월)</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">의사결정 기준</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-emerald-600 dark:text-emerald-400">기업 가치 분석</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">가격 변동성</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">위험 수준</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-emerald-600 dark:text-emerald-400">중간 위험</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">고위험</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">수익 원천</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-emerald-600 dark:text-emerald-400">기업 성장 + 배당</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">시세 차익</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주요 시장 지수</h2>
        <div className="grid gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">🇰🇷 KOSPI (Korea Composite Stock Price Index)</h3>
            <p className="text-gray-600 dark:text-gray-400">
              한국거래소 유가증권시장의 대표 지수. 시가총액 가중평균 방식으로 계산.
              삼성전자, SK하이닉스 등 대형주의 영향이 큼.
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">🇺🇸 S&P 500</h3>
            <p className="text-gray-600 dark:text-gray-400">
              미국 주식시장의 대표 지수. 500개 대형주로 구성.
              Apple, Microsoft, Amazon 등 빅테크 기업이 상당 비중 차지.
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">🇺🇸 NASDAQ Composite</h3>
            <p className="text-gray-600 dark:text-gray-400">
              나스닥 거래소의 모든 종목을 포함한 지수. 기술주 비중이 높아 성장성 지향.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}