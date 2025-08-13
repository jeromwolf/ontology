'use client'

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">주식시장은 어떻게 돌아갈까? 🏛️</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          주식을 사고 파는 시장의 기본 원리를 쉽게 알아봐요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏪 주식시장 = 큰 온라인 마켓</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            주식시장은 쿠팡이나 G마켓 같은 온라인 쇼핑몰과 비슷해요!
            다만 물건 대신 회사의 주식을 사고 파는 거죠.
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
              <div className="text-3xl mb-2">🏢</div>
              <h3 className="font-semibold mb-1">판매 상품</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">회사 주식</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
              <div className="text-3xl mb-2">👥</div>
              <h3 className="font-semibold mb-1">구매자</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">투자자들</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
              <div className="text-3xl mb-2">🏦</div>
              <h3 className="font-semibold mb-1">시장</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">증권거래소</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⏰ 주식시장 운영 시간</h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">한국 주식시장 (KOSPI, KOSDAQ)</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">정규 거래 시간</h4>
              <p className="text-2xl font-bold mb-1">오전 9시 ~ 오후 3시 30분</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                평일에만 열려요 (주말, 공휴일 휴장)
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">시간외 거래</h4>
              <p className="text-lg font-bold mb-1">오전 8:30~9:00</p>
              <p className="text-lg font-bold mb-1">오후 3:40~4:00</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                정규시간 외 추가 거래 가능
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💰 주식 가격은 어떻게 정해질까?</h2>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-4">
          <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">수요와 공급의 법칙</h3>
          <p className="text-gray-700 dark:text-gray-300">
            주식 가격도 일반 물건처럼 사고 싶은 사람과 팔고 싶은 사람의 균형으로 결정돼요!
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">📈 가격이 오를 때</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 사고 싶은 사람 {'>'} 팔고 싶은 사람</li>
              <li>• 회사 실적이 좋을 때</li>
              <li>• 좋은 뉴스가 나올 때</li>
              <li>• 경제가 좋아질 때</li>
            </ul>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">📉 가격이 내릴 때</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 팔고 싶은 사람 {'>'} 사고 싶은 사람</li>
              <li>• 회사 실적이 나쁠 때</li>
              <li>• 나쁜 뉴스가 나올 때</li>
              <li>• 경제가 안 좋을 때</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏛️ 한국의 주요 주식시장</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-2 border-blue-300 dark:border-blue-600">
            <h3 className="font-bold text-xl text-blue-600 dark:text-blue-400 mb-3">KOSPI (코스피)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              한국의 대표 주식시장. 큰 기업들이 주로 상장되어 있어요.
            </p>
            <div className="space-y-2 text-sm">
              <p><strong>대표 기업:</strong></p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 rounded-full">삼성전자</span>
                <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 rounded-full">SK하이닉스</span>
                <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 rounded-full">현대차</span>
                <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 rounded-full">LG화학</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-2 border-purple-300 dark:border-purple-600">
            <h3 className="font-bold text-xl text-purple-600 dark:text-purple-400 mb-3">KOSDAQ (코스닥)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              성장 가능성 높은 중소기업, IT기업들이 주로 상장되어 있어요.
            </p>
            <div className="space-y-2 text-sm">
              <p><strong>대표 기업:</strong></p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 rounded-full">카카오게임즈</span>
                <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 rounded-full">펄어비스</span>
                <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 rounded-full">셀트리온</span>
                <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 rounded-full">에코프로</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 시장 분위기를 보는 방법</h2>
        <div className="bg-gradient-to-r from-red-50 to-blue-50 dark:from-red-900/20 dark:to-blue-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">코스피 지수로 전체 분위기 파악하기</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold">코스피 2,500</span>
                <span className="text-sm text-gray-600 dark:text-gray-400">평균적인 수준</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div className="bg-yellow-500 h-2 rounded-full" style={{width: '50%'}}></div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-emerald-600 dark:text-emerald-400">코스피 2,800 ↑</span>
                <span className="text-sm text-emerald-600 dark:text-emerald-400">시장이 좋아요!</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div className="bg-emerald-500 h-2 rounded-full" style={{width: '80%'}}></div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-red-600 dark:text-red-400">코스피 2,200 ↓</span>
                <span className="text-sm text-red-600 dark:text-red-400">시장이 안 좋아요</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div className="bg-red-500 h-2 rounded-full" style={{width: '30%'}}></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 오늘 배운 것 정리</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3 text-lg">
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span>주식시장 = 회사 주식을 사고 파는 큰 시장</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span>평일 오전 9시 ~ 오후 3시 30분 운영</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span>수요와 공급으로 가격이 결정됨</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-2xl">✅</span>
              <span>KOSPI = 대기업, KOSDAQ = 중소/IT기업</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}