'use client';

export default function Chapter15() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">기업 분석 기초 🏢</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          숫자로만 보는 것이 아닌, 기업의 실체를 파악하는 방법을 배워보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 기본 재무 지표</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">매출액</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              기업이 1년 동안 벌어들인 총 수익
            </p>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              ✅ 꾸준히 증가하는지 확인<br/>
              ✅ 동종업계 대비 성장률 비교
            </div>
          </div>
          
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">영업이익</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              본업으로 벌어들인 순수한 이익
            </p>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              ✅ 영업이익률 = 영업이익 ÷ 매출액<br/>
              ✅ 10% 이상이면 우수한 편
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 실제 기업 재무지표 예시</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">삼성전자 최근 3년 실적 (단위: 조원)</h3>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2">구분</th>
                  <th className="text-right py-2">2021년</th>
                  <th className="text-right py-2">2022년</th>
                  <th className="text-right py-2">2023년</th>
                  <th className="text-right py-2">성장률</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-2 font-medium">매출액</td>
                  <td className="text-right py-2">279.6</td>
                  <td className="text-right py-2">302.2</td>
                  <td className="text-right py-2">258.9</td>
                  <td className="text-right py-2 text-red-600">-14.3%</td>
                </tr>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-2 font-medium">영업이익</td>
                  <td className="text-right py-2">51.6</td>
                  <td className="text-right py-2">43.4</td>
                  <td className="text-right py-2">6.5</td>
                  <td className="text-right py-2 text-red-600">-85.0%</td>
                </tr>
                <tr>
                  <td className="py-2 font-medium">영업이익률</td>
                  <td className="text-right py-2">18.5%</td>
                  <td className="text-right py-2">14.4%</td>
                  <td className="text-right py-2">2.5%</td>
                  <td className="text-right py-2">-</td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <h3 className="font-semibold mb-4">네이버 최근 3년 실적 (단위: 조원)</h3>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2">구분</th>
                  <th className="text-right py-2">2021년</th>
                  <th className="text-right py-2">2022년</th>
                  <th className="text-right py-2">2023년</th>
                  <th className="text-right py-2">성장률</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-2 font-medium">매출액</td>
                  <td className="text-right py-2">6.8</td>
                  <td className="text-right py-2">8.2</td>
                  <td className="text-right py-2">9.7</td>
                  <td className="text-right py-2 text-green-600">+18.3%</td>
                </tr>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-2 font-medium">영업이익</td>
                  <td className="text-right py-2">1.5</td>
                  <td className="text-right py-2">1.3</td>
                  <td className="text-right py-2">1.5</td>
                  <td className="text-right py-2 text-green-600">+15.4%</td>
                </tr>
                <tr>
                  <td className="py-2 font-medium">영업이익률</td>
                  <td className="text-right py-2">22.1%</td>
                  <td className="text-right py-2">15.9%</td>
                  <td className="text-right py-2">15.5%</td>
                  <td className="text-right py-2">-</td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <div className="mt-4 p-4 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">💡 분석 포인트</h4>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>삼성전자:</strong> 반도체 시장 침체로 실적 급감, 회복 시점 주목</li>
              <li>• <strong>네이버:</strong> 꾸준한 성장세 유지, 영업이익률 개선 필요</li>
              <li>• 동일 업종 내 비교보다 각 기업의 성장 추세가 중요</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔍 기업 경쟁력 체크</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">확인해야 할 포인트</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">시장 지위</h4>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 시장 점유율은?</li>
                <li>• 업계 몇 위인가?</li>
                <li>• 경쟁사 대비 우위는?</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">브랜드 파워</h4>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 소비자 인지도는?</li>
                <li>• 브랜드 충성도는?</li>
                <li>• 가격 결정력이 있는가?</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 성장성 분석</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">과거 3년 성장률</h4>
              <p className="text-gray-700 dark:text-gray-300">
                매출액과 순이익이 꾸준히 증가했는지 확인하세요.
                일시적 급증보다는 안정적 성장이 중요합니다.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">미래 성장 동력</h4>
              <p className="text-gray-700 dark:text-gray-300">
                새로운 사업 분야, 신제품 출시, 해외 진출 등 
                미래 성장을 견인할 요소가 있는지 살펴보세요.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 초보자를 위한 간단 분석법</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ol className="space-y-3">
            <li><strong>1단계:</strong> 이 회사가 무엇을 파는지 명확히 알 수 있나?</li>
            <li><strong>2단계:</strong> 최근 3년간 매출이 증가했나?</li>
            <li><strong>3단계:</strong> 영업이익률이 5% 이상인가?</li>
            <li><strong>4단계:</strong> 부채비율이 200% 이하인가?</li>
            <li><strong>5단계:</strong> 업계에서 상위권 기업인가?</li>
          </ol>
          <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900/30 rounded">
            <strong>5개 중 4개 이상</strong> 만족하면 투자 검토 가능
          </div>
        </div>
      </section>
    </div>
  )
}