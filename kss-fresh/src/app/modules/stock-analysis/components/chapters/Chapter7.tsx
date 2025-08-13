'use client'

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">왜 사람들이 주식을 살까? 🤑</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          은행 예금만 하면 안 되나요? 주식 투자의 이유를 알아봐요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💸 돈을 불리는 여러 가지 방법</h2>
        <div className="grid gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-2">🏦 은행 예금</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              가장 안전하지만 이자가 적어요 (연 2~3%)
            </p>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              예: 1,000만원 → 1년 후 1,020만원 (20만원 이익)
            </div>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">🏠 부동산</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              큰 돈이 필요하고 쉽게 팔기 어려워요
            </p>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              예: 아파트 사려면 수억원 필요
            </div>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">📈 주식</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              적은 돈으로 시작할 수 있고, 수익이 클 수 있어요
            </p>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              예: 1만원부터 시작 가능, 회사가 성장하면 큰 수익
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 실제 사례로 보는 주식의 매력</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">삼성전자 주식의 10년 여행</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg">
              <span className="font-medium">2014년</span>
              <span className="text-gray-600 dark:text-gray-400">1주당 약 25,000원</span>
            </div>
            <div className="flex items-center justify-center">
              <span className="text-2xl">⬇️</span>
            </div>
            <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg">
              <span className="font-medium">2024년</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">1주당 약 80,000원</span>
            </div>
            <div className="mt-4 p-4 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg text-center">
              <p className="font-bold text-emerald-800 dark:text-emerald-200">
                10년 동안 약 3배 이상 상승! 💰
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                100만원 투자 → 300만원 이상으로 증가
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 주식 투자의 장점</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">적은 돈으로 시작</h3>
            <p className="text-gray-600 dark:text-gray-400">
              1만원부터도 투자 가능! 부담 없이 시작할 수 있어요.
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">언제든 사고 팔기</h3>
            <p className="text-gray-600 dark:text-gray-400">
              필요할 때 바로 팔 수 있어요. 부동산처럼 오래 걸리지 않아요.
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-3">회사와 함께 성장</h3>
            <p className="text-gray-600 dark:text-gray-400">
              좋은 회사를 골라서 함께 성장할 수 있어요.
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">배당금 받기</h3>
            <p className="text-gray-600 dark:text-gray-400">
              회사가 번 돈의 일부를 나눠주는 배당금도 받을 수 있어요.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 하지만 조심해야 해요!</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">주식의 위험성</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-red-600 dark:text-red-400">❌</span>
              <div>
                <strong>가격이 떨어질 수 있어요</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  회사가 안 좋아지면 주식 가격도 떨어져요
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-red-600 dark:text-red-400">❌</span>
              <div>
                <strong>감정 조절이 어려워요</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  가격이 오르내릴 때마다 마음이 불안해질 수 있어요
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-red-600 dark:text-red-400">❌</span>
              <div>
                <strong>공부가 필요해요</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  좋은 회사를 고르려면 공부해야 해요
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 현명한 투자를 위한 기본 원칙</h2>
        <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-xl p-6">
          <div className="grid gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-2">1. 여유 자금으로만 투자하세요</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                당장 쓸 돈이 아닌, 없어도 생활에 지장 없는 돈으로만!
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-2">2. 분산 투자하세요</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                한 회사에만 투자하지 말고 여러 회사에 나눠서!
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-2">3. 장기적으로 생각하세요</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                하루하루 가격에 일희일비하지 말고 길게 보세요!
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}