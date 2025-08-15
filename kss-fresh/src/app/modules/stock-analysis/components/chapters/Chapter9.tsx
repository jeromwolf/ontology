'use client';

export default function Chapter9() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">증권 계좌 만들기 A to Z 🏦</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          주식을 사려면 먼저 증권 계좌가 필요해요. 하나하나 차근차근 알려드릴게요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📱 어떤 증권사를 선택할까?</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            증권사는 은행처럼 여러 곳이 있어요. 각각 장단점이 있으니 자신에게 맞는 곳을 선택하세요!
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">🏢 대형 증권사</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              안정적이고 다양한 서비스 제공
            </p>
            <ul className="space-y-2 text-sm">
              <li>• 삼성증권</li>
              <li>• NH투자증권</li>
              <li>• 미래에셋증권</li>
              <li>• KB증권</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">📱 모바일 증권사</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              수수료가 저렴하고 앱이 편리함
            </p>
            <ul className="space-y-2 text-sm">
              <li>• 토스증권</li>
              <li>• 카카오페이증권</li>
              <li>• 네이버증권</li>
              <li>• 뱅크샐러드증권</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💳 계좌 개설 준비물</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-4">필요한 것들</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">🪪</span>
              <div>
                <strong>신분증</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  주민등록증 또는 운전면허증
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">🏦</span>
              <div>
                <strong>은행 계좌</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  입출금용 연결 계좌
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">📱</span>
              <div>
                <strong>휴대폰</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  본인 명의 휴대폰
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">🎂</span>
              <div>
                <strong>나이</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  만 19세 이상 (미성년자는 보호자 동의)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📲 모바일로 계좌 만들기 (토스증권 예시)</h2>
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
            <div className="space-y-6">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                <div className="flex-1">
                  <h3 className="font-semibold mb-2">앱 다운로드</h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    앱스토어/플레이스토어에서 '토스' 앱 다운로드
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                <div className="flex-1">
                  <h3 className="font-semibold mb-2">토스증권 찾기</h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    앱 하단 '전체' → '투자' → '토스증권 계좌 만들기'
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                <div className="flex-1">
                  <h3 className="font-semibold mb-2">본인 인증</h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    신분증 촬영 + 얼굴 인증 (1분 소요)
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                <div className="flex-1">
                  <h3 className="font-semibold mb-2">정보 입력</h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    투자 목적, 투자 경험 등 간단한 정보 입력
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-emerald-500 text-white rounded-full flex items-center justify-center font-bold">✓</div>
                <div className="flex-1">
                  <h3 className="font-semibold mb-2">개설 완료!</h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    바로 주식 거래 가능! (보통 5-10분 소요)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💰 계좌에 돈 넣기</h2>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-4">입금 방법</h3>
          <div className="space-y-3">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">1. 연결 계좌에서 이체</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                증권 앱에서 '입금' 버튼 → 금액 입력 → 완료!
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">2. 자동 이체 설정</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                매달 정해진 날짜에 자동으로 입금되도록 설정 가능
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 계좌 종류 이해하기</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-2 border-blue-300 dark:border-blue-600">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">일반 계좌</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>✅ 거래 제한 없음</li>
              <li>✅ 바로 개설 가능</li>
              <li>❌ 세금 혜택 없음</li>
              <li>📌 초보자 추천!</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-2 border-purple-300 dark:border-purple-600">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">ISA 계좌</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>✅ 세금 혜택 있음</li>
              <li>✅ 손익 통산 가능</li>
              <li>❌ 연 2천만원 한도</li>
              <li>❌ 3년 의무 보유</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 주의사항</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">꼭 기억하세요!</h3>
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span>🔐</span>
              <span>비밀번호는 복잡하게 설정하고 절대 남에게 알려주지 마세요</span>
            </li>
            <li className="flex items-start gap-2">
              <span>📱</span>
              <span>공인인증서나 생체인증을 꼭 설정하세요</span>
            </li>
            <li className="flex items-start gap-2">
              <span>💸</span>
              <span>처음엔 소액으로 시작하세요 (10만원 이하 추천)</span>
            </li>
            <li className="flex items-start gap-2">
              <span>📚</span>
              <span>모의투자로 먼저 연습해보는 것도 좋아요</span>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎉 축하합니다!</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 text-center">
          <div className="text-6xl mb-4">🎊</div>
          <h3 className="text-2xl font-bold mb-4">이제 주식 투자를 시작할 준비가 되었어요!</h3>
          <p className="text-gray-700 dark:text-gray-300">
            다음 챕터에서는 실제로 주문하는 방법을 배워볼게요.
          </p>
        </div>
      </section>
    </div>
  )
}