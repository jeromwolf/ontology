'use client';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">주식이 도대체 뭔가요? 🤔</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          복잡한 용어 없이 정말 쉽게 설명해드릴게요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🍕 피자로 이해하는 주식</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            친구들과 피자 가게를 차리려고 한다고 상상해보세요. 
            혼자서는 돈이 부족해서 친구 4명이 각자 돈을 모았습니다.
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
            <li>👨 철수: 2조각 (40%) - 가장 많이 투자했어요</li>
            <li>👩 영희: 1조각 (20%) - 적당히 투자했어요</li>
            <li>👨 민수: 1조각 (20%) - 영희만큼 투자했어요</li>
            <li>👩 수진: 1조각 (20%) - 민수만큼 투자했어요</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💰 주식을 가지면 뭐가 좋아요?</h2>
        <div className="grid gap-4">
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">1. 주인이 됩니다</h3>
            <p className="text-gray-700 dark:text-gray-300">
              주식을 가진 만큼 그 회사의 주인이 됩니다. 
              철수는 40%의 주인, 영희는 20%의 주인이에요!
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">2. 이익을 나눠 가질 수 있어요</h3>
            <p className="text-gray-700 dark:text-gray-300">
              피자 가게가 돈을 많이 벌면, 가진 주식만큼 이익을 나눠 받을 수 있어요. 
              이걸 "배당금"이라고 해요.
            </p>
          </div>
          
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">3. 비싸게 팔 수 있어요</h3>
            <p className="text-gray-700 dark:text-gray-300">
              피자 가게가 유명해지면, 다른 사람들이 "나도 주인이 되고 싶어!"라고 해요. 
              그러면 내 주식을 더 비싸게 팔 수 있어요!
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 실제 주식은 어떻게 다른가요?</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">🏢 큰 회사들의 주식</h3>
              <p className="text-gray-700 dark:text-gray-300">
                삼성전자, 카카오, 네이버 같은 회사들도 주식으로 나뉘어져 있어요. 
                우리도 이런 회사의 작은 주인이 될 수 있습니다!
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">🏦 주식시장</h3>
              <p className="text-gray-700 dark:text-gray-300">
                주식을 사고 파는 큰 시장이 있어요. 
                마치 온라인 쇼핑몰처럼, 원하는 회사의 주식을 살 수 있어요.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">💵 가격은 계속 변해요</h3>
              <p className="text-gray-700 dark:text-gray-300">
                많은 사람이 사고 싶으면 가격이 올라가고, 
                팔고 싶은 사람이 많으면 가격이 내려가요.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 정리하면요!</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3 text-lg">
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span>주식 = 회사를 작게 나눈 조각</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span>주식을 사면 = 그 회사의 작은 주인이 됨</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span>회사가 잘 되면 = 내 주식 가치도 올라감</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span>주식시장 = 주식을 사고 파는 곳</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}