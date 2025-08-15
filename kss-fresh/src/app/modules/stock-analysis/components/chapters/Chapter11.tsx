'use client';

export default function Chapter11() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">첫 주식 고르기 🎯</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          수많은 주식 중에서 어떤 기준으로 첫 주식을 선택해야 할까요?
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏆 초보자를 위한 안전한 선택 기준</h2>
        <div className="grid gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">1. 내가 아는 회사부터</h3>
            <p className="text-gray-700 dark:text-gray-300">
              평소에 사용하는 제품이나 서비스를 만드는 회사를 선택하세요.
              삼성전자, 카카오, 네이버처럼 일상에서 접하는 기업이 좋습니다.
            </p>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">2. 대형주 위주로</h3>
            <p className="text-gray-700 dark:text-gray-300">
              시가총액 상위 기업들은 상대적으로 안정적입니다.
              급격한 가격 변동이 적어 초보자에게 적합합니다.
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">3. 배당주 고려</h3>
            <p className="text-gray-700 dark:text-gray-300">
              정기적으로 배당금을 주는 회사는 안정적인 수익원이 됩니다.
              은행주, 통신주 등이 대표적인 배당주입니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 피해야 할 종목</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span>❌</span>
              <span><strong>테마주</strong>: 일시적 이슈로 급등한 주식은 위험합니다</span>
            </li>
            <li className="flex items-start gap-2">
              <span>❌</span>
              <span><strong>동전주</strong>: 주가가 너무 낮은 주식은 변동성이 매우 큽니다</span>
            </li>
            <li className="flex items-start gap-2">
              <span>❌</span>
              <span><strong>루머주</strong>: 확인되지 않은 소문에 기반한 투자는 금물</span>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 간단한 체크리스트</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3">
            <li className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>내가 이해할 수 있는 사업을 하는 회사인가?</span>
            </li>
            <li className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>최근 3년간 안정적인 매출과 이익이 있는가?</span>
            </li>
            <li className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>업계에서 인정받는 브랜드를 가지고 있는가?</span>
            </li>
            <li className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>적정한 부채 수준을 유지하고 있는가?</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}