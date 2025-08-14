'use client'

export default function Chapter1() {
  return (
    <div className="max-w-4xl mx-auto">
      {/* Chapter Title */}
      <div className="mb-12">
        <h1 className="text-4xl font-bold mb-4">
          기업가치와 시장가격의 괴리: 투자 기회의 원천
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          왜 삼성전자의 실제 가치와 주가는 다를까? 이 차이에서 수익 기회가 발생합니다.
        </p>
      </div>

      {/* Introduction */}
      <section className="mb-16">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-4">핵심 개념</h2>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-lg mb-2">내재가치 (Intrinsic Value)</h3>
              <p className="text-gray-700 dark:text-gray-300">
                기업이 미래에 창출할 현금흐름의 현재가치 총합. DCF 모델로 계산하며, 
                워런 버핏은 "가격은 당신이 지불하는 것이고, 가치는 당신이 얻는 것"이라고 정의했습니다.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-lg mb-2">시장가격 (Market Price)</h3>
              <p className="text-gray-700 dark:text-gray-300">
                수요와 공급에 의해 실시간으로 결정되는 거래 가격. 
                단기적으로는 투자자 심리, 뉴스, 수급 등에 의해 변동하지만, 
                장기적으로는 내재가치에 수렴하는 경향이 있습니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Real Case Study */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-8">실제 사례 분석: 2020년 3월 코로나 폭락</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">삼성전자 사례</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">2020년 3월 19일 (저점)</h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 주가: 42,300원</li>
                <li>• PER: 13.5배</li>
                <li>• PBR: 1.1배</li>
                <li>• 시가총액: 252조원</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">2021년 1월 11일 (고점)</h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 주가: 88,800원 (+110%)</li>
                <li>• PER: 25.3배</li>
                <li>• PBR: 2.4배</li>
                <li>• 시가총액: 530조원</li>
              </ul>
            </div>
          </div>
          <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <p className="text-sm">
              <strong>핵심 인사이트:</strong> 10개월 만에 기업의 본질적 가치가 2배가 되었을까요? 
              아닙니다. 시장의 과도한 공포가 과도한 탐욕으로 바뀌었을 뿐입니다.
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">시장 공포 요인</h4>
            <ul className="text-sm space-y-1">
              <li>• 글로벌 팬데믹 불확실성</li>
              <li>• 반도체 수요 급감 우려</li>
              <li>• 무차별적 매도 압력</li>
            </ul>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">실제 펀더멘털</h4>
            <ul className="text-sm space-y-1">
              <li>• 재택근무로 IT 수요 증가</li>
              <li>• 메모리 반도체 슈퍼사이클</li>
              <li>• 견고한 현금흐름 유지</li>
            </ul>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">투자 교훈</h4>
            <ul className="text-sm space-y-1">
              <li>• 공포 = 기회</li>
              <li>• 기업 분석 > 시장 분위기</li>
              <li>• 장기 관점의 중요성</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Market Mechanism */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-8">주식시장의 가격 결정 메커니즘</h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4">1. 단기 가격 결정 요인 (노이즈)</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">심리적 요인</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• FOMO (Fear of Missing Out)</li>
                  <li>• 패닉 셀링 (Panic Selling)</li>
                  <li>• 확증 편향 (Confirmation Bias)</li>
                  <li>• 군중 심리 (Herd Mentality)</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">기술적 요인</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 알고리즘 트레이딩</li>
                  <li>• 기술적 지지/저항선</li>
                  <li>• 옵션 만기일 효과</li>
                  <li>• 인덱스 리밸런싱</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4">2. 장기 가격 결정 요인 (시그널)</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">기업 펀더멘털</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 매출 성장률과 시장 점유율</li>
                  <li>• 영업이익률과 ROE</li>
                  <li>• 부채비율과 현금흐름</li>
                  <li>• 경영진 역량과 지배구조</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">산업 및 거시경제</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 산업 성장성과 경쟁 구조</li>
                  <li>• 금리와 통화정책</li>
                  <li>• 규제 환경 변화</li>
                  <li>• 글로벌 경제 사이클</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Trading System */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-8">한국 주식시장 거래 시스템</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="p-6">
            <h3 className="text-xl font-semibold mb-4">거래 시간대별 특징</h3>
            
            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="w-24 text-sm font-medium text-gray-500">08:00-09:00</div>
                <div className="flex-1">
                  <h4 className="font-medium mb-1">장전 시간외거래</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    전일 종가 기준 ±10% 범위 내 단일가 거래. 
                    해외시장 영향을 반영한 기관/외국인 거래 집중
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-4">
                <div className="w-24 text-sm font-medium text-gray-500">09:00-15:30</div>
                <div className="flex-1">
                  <h4 className="font-medium mb-1">정규장</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    연속거래 시간. 실시간 호가 체결. 
                    09:00 시가 결정(동시호가), 15:20-15:30 종가 결정(동시호가)
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-4">
                <div className="w-24 text-sm font-medium text-gray-500">15:40-16:00</div>
                <div className="flex-1">
                  <h4 className="font-medium mb-1">장후 시간외거래</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    종가 기준 ±10% 범위 내 단일가 거래. 
                    정규장 이후 뉴스나 공시 반영
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold mb-4">주문 유형별 활용 전략</h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-medium text-red-600 dark:text-red-400">시장가 주문</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  즉시 체결 우선. 급등/급락 시 슬리피지 주의. 
                  유동성 높은 대형주에 적합
                </p>
              </div>
              <div>
                <h4 className="font-medium text-blue-600 dark:text-blue-400">지정가 주문</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  가격 통제 가능. 체결 불확실성 존재. 
                  변동성 높은 중소형주에 유리
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold mb-4">프로의 주문 기법</h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-medium text-purple-600 dark:text-purple-400">IOC/FOK 주문</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  즉시 체결 또는 취소. 대량 거래 시 시장 충격 최소화
                </p>
              </div>
              <div>
                <h4 className="font-medium text-green-600 dark:text-green-400">조건부 주문</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  특정 조건 충족 시 자동 주문. 리스크 관리와 자동화
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-16">
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6">핵심 요약</h2>
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">1</span>
              <div>
                <h3 className="font-semibold mb-1">가치와 가격의 괴리가 수익의 원천</h3>
                <p className="text-indigo-100">
                  시장은 단기적으로 투표기계지만, 장기적으로는 저울이다 - 벤저민 그레이엄
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">2</span>
              <div>
                <h3 className="font-semibold mb-1">시장 메커니즘의 이해가 성공의 기초</h3>
                <p className="text-indigo-100">
                  거래 시스템, 주문 유형, 시장 참여자를 알아야 유리한 포지션 구축 가능
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">3</span>
              <div>
                <h3 className="font-semibold mb-1">감정 배제, 데이터 중심 의사결정</h3>
                <p className="text-indigo-100">
                  공포와 탐욕의 사이클을 이해하고, 역발상 투자로 초과 수익 창출
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Next Chapter Preview */}
      <section>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-2">다음 챕터 미리보기</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            <strong>Chapter 2: 재무제표 해부학</strong> - 
            삼성전자 2024년 3분기 실적발표를 실시간으로 분석하며 
            재무제표 읽는 법을 마스터합니다.
          </p>
        </div>
      </section>
    </div>
  )
}