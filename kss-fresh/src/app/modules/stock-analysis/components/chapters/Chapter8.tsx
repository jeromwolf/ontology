'use client';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">주식시장의 구조와 작동 원리</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          주식시장의 기본 구조와 가격 결정 메커니즘을 이해합니다.
          시장 참여자들의 역할과 거래 시스템의 작동 원리를 체계적으로 학습해봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 주식시장의 기본 구조</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            주식시장은 기업과 투자자를 연결하는 자본시장의 핵심 인프라입니다.
            효율적인 가격 발견과 유동성 공급을 통해 경제 발전에 기여합니다.
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-2">발행시장</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                IPO, 유상증자 등 기업이 자금을 조달하는 1차 시장
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-2">유통시장</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                상장된 주식이 투자자 간 거래되는 2차 시장
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-2">장외시장</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                비상장 주식이나 대량 매매가 이루어지는 시장
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⏰ 거래 시간과 세션</h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">한국거래소(KRX) 운영 시간</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">정규시장</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="font-medium">개장 전 경쟁대량매매</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">08:00 ~ 09:00</p>
                  <p className="text-xs mt-1">전일 종가 기준 단일가 매매</p>
                </div>
                <div>
                  <p className="font-medium">정규거래</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">09:00 ~ 15:30</p>
                  <p className="text-xs mt-1">연속 경쟁매매 방식</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">시간외거래</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="font-medium">장개시전 시간외</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">07:30 ~ 08:30</p>
                </div>
                <div>
                  <p className="font-medium">장종료후 시간외</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">15:40 ~ 16:00</p>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-2">※ 단일가 매매, 당일 종가의 ±10% 이내</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💰 가격 결정 메커니즘</h2>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-4">
          <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">
            경쟁매매 원칙
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            주식 가격은 수요와 공급의 균형점에서 결정되며, 
            한국거래소는 가격우선-시간우선 원칙에 따라 체결을 중개합니다.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-l-4 border-green-500">
            <h3 className="font-semibold text-green-700 dark:text-green-300 mb-3">매수 우선순위</h3>
            <ol className="space-y-2 text-sm">
              <li><strong>1.</strong> 높은 가격 주문 우선</li>
              <li><strong>2.</strong> 동일 가격시 먼저 낸 주문 우선</li>
              <li><strong>3.</strong> 시장가 주문이 지정가보다 우선</li>
            </ol>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-l-4 border-red-500">
            <h3 className="font-semibold text-red-700 dark:text-red-300 mb-3">매도 우선순위</h3>
            <ol className="space-y-2 text-sm">
              <li><strong>1.</strong> 낮은 가격 주문 우선</li>
              <li><strong>2.</strong> 동일 가격시 먼저 낸 주문 우선</li>
              <li><strong>3.</strong> 시장가 주문이 지정가보다 우선</li>
            </ol>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏛️ 한국 주식시장 구분</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-2 border-blue-300 dark:border-blue-600">
            <h3 className="font-bold text-xl text-blue-600 dark:text-blue-400 mb-3">유가증권시장 (KOSPI)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              대형 우량기업 중심의 주요 시장
            </p>
            <div className="space-y-2 text-sm">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
                <p><strong>상장 요건:</strong> 자기자본 300억원 이상</p>
                <p><strong>시가총액:</strong> 약 2,100조원 (2024년 기준)</p>
                <p><strong>상장 기업수:</strong> 약 800개</p>
              </div>
              <div>
                <p className="font-medium mb-1">대표 기업:</p>
                <p className="text-xs">삼성전자, SK하이닉스, LG에너지솔루션, 현대차, POSCO</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-2 border-purple-300 dark:border-purple-600">
            <h3 className="font-bold text-xl text-purple-600 dark:text-purple-400 mb-3">코스닥시장 (KOSDAQ)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              혁신형 중소·벤처기업 중심 시장
            </p>
            <div className="space-y-2 text-sm">
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3">
                <p><strong>상장 요건:</strong> 자기자본 30억원 이상</p>
                <p><strong>시가총액:</strong> 약 400조원 (2024년 기준)</p>
                <p><strong>상장 기업수:</strong> 약 1,600개</p>
              </div>
              <div>
                <p className="font-medium mb-1">대표 기업:</p>
                <p className="text-xs">에코프로비엠, 셀트리온헬스케어, 카카오게임즈, 알테오젠</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 시장 지수의 이해</h2>
        <div className="bg-gradient-to-r from-red-50 to-blue-50 dark:from-red-900/20 dark:to-blue-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">주요 지수 현황과 의미</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="font-semibold">KOSPI (코스피지수)</span>
                <span className="text-sm text-gray-600 dark:text-gray-400">1980.1.4 = 100</span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm">역사적 저점</span>
                  <span className="text-sm font-medium">2009년 3월: 1,000대</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">코로나 저점</span>
                  <span className="text-sm font-medium">2020년 3월: 1,400대</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">역사적 고점</span>
                  <span className="text-sm font-medium">2021년 7월: 3,300대</span>
                </div>
                <div className="flex justify-between items-center bg-blue-50 dark:bg-blue-900/20 p-2 rounded">
                  <span className="text-sm font-medium">현재 수준 (2024년)</span>
                  <span className="text-sm font-bold">2,500 ~ 2,700대</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">지수 수준별 시장 해석</h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span><strong>2,000 이하:</strong> 역사적 저점, 극도의 침체</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span><strong>2,000 ~ 2,500:</strong> 조정 국면, 투자 기회 모색</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span><strong>2,500 ~ 3,000:</strong> 정상 범위, 기업 실적 중심</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span><strong>3,000 이상:</strong> 과열 우려, 신중한 접근 필요</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏢 시장 참여자의 역할</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">기관투자자</h4>
              <ul className="text-sm space-y-1">
                <li>• 연기금, 보험사, 자산운용사</li>
                <li>• 장기 투자, 대규모 자금 운용</li>
                <li>• 시장 안정화 역할</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">외국인투자자</h4>
              <ul className="text-sm space-y-1">
                <li>• 글로벌 펀드, 헤지펀드</li>
                <li>• 시장 유동성 공급</li>
                <li>• 환율 및 글로벌 변수 민감</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">개인투자자</h4>
              <ul className="text-sm space-y-1">
                <li>• 직접 투자 참여</li>
                <li>• 단기 거래 성향</li>
                <li>• 시장 변동성 확대 가능</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">기타 참여자</h4>
              <ul className="text-sm space-y-1">
                <li>• 증권사 자기매매</li>
                <li>• 사모펀드</li>
                <li>• 알고리즘 트레이딩</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 핵심 정리</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3">
            <li className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span>주식시장은 발행시장과 유통시장으로 구성되며, 효율적인 자본 배분 기능 수행</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span>정규거래는 09:00~15:30, 가격우선-시간우선 원칙에 따라 체결</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span>KOSPI는 대형 우량주, KOSDAQ은 혁신 성장주 중심</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span>지수 수준에 따른 시장 상황 판단과 투자 전략 수립 필요</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}