'use client';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">주식의 기본 개념</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          주식의 본질과 투자의 의미를 이해해봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주식의 기본 원리</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            회사를 설립할 때 필요한 자본을 여러 투자자가 나누어 부담하는 방식을 예로 들어보겠습니다. 
            이때 각 투자자의 지분이 바로 주식이 됩니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-2">피자 가게 = 회사</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                여러분이 차린 피자 가게가 바로 "회사"예요
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-2">피자 조각 = 주식</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                가게를 5조각으로 나눈 것이 바로 "주식"이에요
              </p>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">누가 얼마나 가지고 있나요?</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>투자자 A: 2주 (40%) - 최대 주주</li>
            <li>투자자 B: 1주 (20%) - 일반 주주</li>
            <li>투자자 C: 1주 (20%) - 일반 주주</li>
            <li>투자자 D: 1주 (20%) - 일반 주주</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주식 보유의 이점</h2>
        <div className="grid gap-4">
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">1. 주인이 됩니다</h3>
            <p className="text-gray-700 dark:text-gray-300">
              주식을 가진 만큼 그 회사의 주인이 됩니다. 
              투자자 A는 40%의 지분을, 투자자 B는 20%의 지분을 보유하게 됩니다.
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">2. 이익을 나눠 가질 수 있어요</h3>
            <p className="text-gray-700 dark:text-gray-300">
              회사가 이익을 창출하면 주주들은 보유 지분에 비례하여 배당금을 받을 수 있습니다. 
              이는 주식 투자의 중요한 수익원 중 하나입니다.
            </p>
          </div>
          
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">3. 비싸게 팔 수 있어요</h3>
            <p className="text-gray-700 dark:text-gray-300">
              회사가 성장하고 가치가 상승하면 주식의 시장 가격도 함께 상승합니다. 
              이를 통해 자본 차익을 실현할 수 있습니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실제 주식 시장의 특징</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">상장 기업의 주식</h3>
              <p className="text-gray-700 dark:text-gray-300">
                삼성전자, 카카오, 네이버 같은 회사들도 주식으로 나뉘어져 있어요. 
                개인 투자자도 이러한 대기업의 주주가 될 수 있습니다.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">주식 거래소</h3>
              <p className="text-gray-700 dark:text-gray-300">
                주식을 사고 파는 큰 시장이 있어요. 
                거래소를 통해 주식을 자유롭게 매매할 수 있습니다.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">가격 변동성</h3>
              <p className="text-gray-700 dark:text-gray-300">
                수요와 공급의 원리에 따라 주가는 실시간으로 변동합니다. 
                시장 참가자들의 매수/매도 의사가 가격을 결정합니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">핵심 요약</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3 text-lg">
            <li className="flex items-start gap-2">
              <span className="text-green-500">•</span>
              <span>주식 = 회사를 작게 나눈 조각</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">•</span>
              <span>주식을 사면 = 그 회사의 작은 주인이 됨</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">•</span>
              <span>회사가 잘 되면 = 내 주식 가치도 올라감</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">•</span>
              <span>주식시장 = 주식을 사고 파는 곳</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}