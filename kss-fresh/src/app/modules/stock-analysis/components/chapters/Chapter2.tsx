'use client'

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">재무제표 3요소</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          기업의 재무 상태를 정확히 파악하려면 3가지 핵심 재무제표를 종합적으로 분석해야 합니다.
          각 재무제표는 기업의 다른 측면을 보여줍니다.
        </p>
        
        <div className="grid gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">📊 손익계산서 (Income Statement)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              일정 기간(분기/연간) 동안의 수익과 비용을 나타내는 실적표
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between"><span>매출액 (Revenue)</span><span>기업이 벌어들인 총 수익</span></div>
              <div className="flex justify-between"><span>영업이익 (Operating Income)</span><span>본업으로 얻은 순수익</span></div>
              <div className="flex justify-between"><span>당기순이익 (Net Income)</span><span>모든 비용 차감 후 최종 이익</span></div>
            </div>
          </div>
          
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">🏦 대차대조표 (Balance Sheet)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              특정 시점의 자산, 부채, 자본 현황을 나타내는 재무상태표
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between"><span>자산 (Assets)</span><span>기업이 소유한 모든 경제적 자원</span></div>
              <div className="flex justify-between"><span>부채 (Liabilities)</span><span>기업이 갚아야 할 채무</span></div>
              <div className="flex justify-between"><span>자본 (Equity)</span><span>자산에서 부채를 뺀 순자산</span></div>
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">💰 현금흐름표 (Cash Flow Statement)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              실제 현금의 유입과 유출을 보여주는 가장 조작하기 어려운 재무제표
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between"><span>영업 현금흐름</span><span>본업 활동으로 벌어들인 현금</span></div>
              <div className="flex justify-between"><span>투자 현금흐름</span><span>투자 활동으로 인한 현금 변화</span></div>
              <div className="flex justify-between"><span>재무 현금흐름</span><span>자금조달 활동으로 인한 현금 변화</span></div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">핵심 투자 지표</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">PER (Price-to-Earnings Ratio)</h3>
            <div className="text-2xl font-mono font-bold mb-2">주가 ÷ 주당순이익</div>
            <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
              현재 주가가 1주당 순이익의 몇 배인지를 나타내는 가장 기본적인 밸류에이션 지표
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 text-sm">
              <strong>해석</strong>: PER 15배 → 현재 수익률로 15년간 벌어야 투자원금 회수
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">PBR (Price-to-Book Ratio)</h3>
            <div className="text-2xl font-mono font-bold mb-2">주가 ÷ 주당순자산</div>
            <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
              현재 주가가 1주당 순자산(장부가치)의 몇 배인지를 나타내는 청산가치 기준 지표
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 text-sm">
              <strong>해석</strong>: PBR 1배 미만 → 장부상 자산가치보다 저평가
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">ROE (Return on Equity)</h3>
            <div className="text-2xl font-mono font-bold mb-2">당기순이익 ÷ 자본총계 × 100</div>
            <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
              기업이 자본을 얼마나 효율적으로 활용하여 이익을 창출하는지를 나타내는 수익성 지표
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 text-sm">
              <strong>우수기준</strong>: 15% 이상 시 양호, 20% 이상 시 우수
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">ROIC (Return on Invested Capital)</h3>
            <div className="text-2xl font-mono text-xs font-bold mb-2">영업이익×(1-세율) ÷ 투하자본</div>
            <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
              기업이 투자한 모든 자본(차입금 포함)을 얼마나 효율적으로 활용하는지를 측정
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 text-sm">
              <strong>투자판단</strong>: ROIC {'>'} WACC 인 기업이 가치창조
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}