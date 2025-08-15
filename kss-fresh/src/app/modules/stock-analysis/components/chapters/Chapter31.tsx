'use client';

import { useState } from 'react';

export default function Chapter31() {
  const [selectedMarket, setSelectedMarket] = useState('us');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">글로벌 시장 투자</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          미국, 중국, 유럽, 신흥시장 등 글로벌 주식시장의 특성과 투자 전략을 학습합니다.
          환율, 시차, 규제 등 해외투자의 실무적 이슈와 글로벌 포트폴리오 구성 방법을 마스터해봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🌍 주요 글로벌 시장</h2>
        <div className="mb-4">
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => setSelectedMarket('us')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedMarket === 'us'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              미국
            </button>
            <button
              onClick={() => setSelectedMarket('china')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedMarket === 'china'
                  ? 'bg-red-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              중국
            </button>
            <button
              onClick={() => setSelectedMarket('europe')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedMarket === 'europe'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              유럽
            </button>
            <button
              onClick={() => setSelectedMarket('emerging')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedMarket === 'emerging'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              신흥시장
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
          {selectedMarket === 'us' && (
            <div>
              <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
                미국 시장 (NYSE, NASDAQ)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">시장 특징</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">시장 규모</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 세계 최대 주식시장</li>
                        <li>• 시가총액 $50조 이상</li>
                        <li>• 6,000개 이상 상장기업</li>
                        <li>• 일평균 거래대금 $500B</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">거래 시간</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 정규장: 9:30-16:00 (EST)</li>
                        <li>• 한국시간: 23:30-06:00</li>
                        <li>• 프리마켓: 04:00-09:30</li>
                        <li>• 애프터마켓: 16:00-20:00</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">주요 지수</h4>
                  <div className="grid md:grid-cols-3 gap-3">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm">S&P 500</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        대형주 500개 기업<br/>
                        시가총액 가중평균<br/>
                        미국 경제 대표 지수
                      </p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm">Nasdaq Composite</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        기술주 중심<br/>
                        3,000개 이상 종목<br/>
                        성장주 집중
                      </p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm">Dow Jones</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        우량주 30개<br/>
                        가격가중평균<br/>
                        전통적 지표
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">투자 방법</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`# 미국 주식 투자 체크리스트
class USMarketInvestment:
    def __init__(self):
        self.exchange_rate = 1320  # USD/KRW
        self.tax_rate = 0.22  # 양도세율
        
    def calculate_investment_cost(self, shares, price_usd):
        """투자 비용 계산 (원화 기준)"""
        # 주식 매수 금액
        stock_value_usd = shares * price_usd
        stock_value_krw = stock_value_usd * self.exchange_rate
        
        # 예상 수수료 (왕복 0.5%)
        commission = stock_value_krw * 0.005
        
        # 환전 수수료 (약 1%)
        fx_fee = stock_value_krw * 0.01
        
        total_cost = stock_value_krw + commission + fx_fee
        
        return {
            'stock_value_krw': stock_value_krw,
            'commission': commission,
            'fx_fee': fx_fee,
            'total_cost': total_cost
        }
    
    def calculate_after_tax_return(self, buy_price, sell_price, shares):
        """세후 수익률 계산"""
        profit_usd = (sell_price - buy_price) * shares
        
        if profit_usd > 0:
            # 양도세 적용 (250만원 공제)
            profit_krw = profit_usd * self.exchange_rate
            deduction = 2500000
            taxable = max(0, profit_krw - deduction)
            tax = taxable * self.tax_rate
            
            after_tax_profit = profit_krw - tax
            return_rate = after_tax_profit / (buy_price * shares * self.exchange_rate)
        else:
            return_rate = profit_usd / (buy_price * shares)
            
        return return_rate * 100`}</pre>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedMarket === 'china' && (
            <div>
              <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
                중국 시장 (상해, 심천, 홍콩)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">시장 구조</h4>
                  <div className="grid md:grid-cols-3 gap-3">
                    <div className="bg-red-50 dark:bg-red-900/20 rounded p-3">
                      <h5 className="font-medium">상해 증시 (SSE)</h5>
                      <ul className="text-sm mt-2 space-y-1">
                        <li>• A주: 위안화 거래</li>
                        <li>• B주: 외화 거래</li>
                        <li>• 과창판: 혁신기업</li>
                        <li>• 대형 국유기업 중심</li>
                      </ul>
                    </div>
                    <div className="bg-red-50 dark:bg-red-900/20 rounded p-3">
                      <h5 className="font-medium">심천 증시 (SZSE)</h5>
                      <ul className="text-sm mt-2 space-y-1">
                        <li>• 창업판: 성장기업</li>
                        <li>• 중소판: 중소기업</li>
                        <li>• 민영기업 중심</li>
                        <li>• 기술주 집중</li>
                      </ul>
                    </div>
                    <div className="bg-red-50 dark:bg-red-900/20 rounded p-3">
                      <h5 className="font-medium">홍콩 증시 (HKEX)</h5>
                      <ul className="text-sm mt-2 space-y-1">
                        <li>• H주: 중국 본토기업</li>
                        <li>• 레드칩: 중국계 기업</li>
                        <li>• 국제적 접근성</li>
                        <li>• 홍콩달러 거래</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">투자 전략</h4>
                  <div className="space-y-3">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">후강퉁/선강퉁</h5>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        홍콩을 통한 중국 본토 A주 투자 통로. 개인투자자도 접근 가능하며
                        일일 쿼터 제한이 있음. 환율 리스크와 거래세 고려 필요.
                      </p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">중국 ADR</h5>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        미국 시장에 상장된 중국 기업 주식. 알리바바, 텐센트 등 대형주 중심.
                        규제 리스크는 있지만 접근성이 좋음.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                  <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
                    ⚠️ 중국 투자 리스크
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• 정부 규제 리스크 (갑작스런 정책 변경)</li>
                    <li>• 회계 투명성 문제</li>
                    <li>• 자본 통제 및 환전 제한</li>
                    <li>• 미중 갈등에 따른 불확실성</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {selectedMarket === 'europe' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
                유럽 시장 (런던, 프랑크푸르트, 파리)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">주요 거래소</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">런던증권거래소 (LSE)</h5>
                      <ul className="text-sm space-y-1">
                        <li>• FTSE 100: 영국 대형주</li>
                        <li>• 국제 기업 다수 상장</li>
                        <li>• 파운드/유로 거래</li>
                        <li>• 브렉시트 영향</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">유로넥스트 (Euronext)</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 범유럽 거래소</li>
                        <li>• CAC 40 (프랑스)</li>
                        <li>• AEX (네덜란드)</li>
                        <li>• 유로화 통합 거래</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">섹터별 강점</h4>
                  <div className="grid md:grid-cols-3 gap-3">
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm">명품/소비재</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        LVMH, 에르메스<br/>
                        로레알, 네슬레
                      </p>
                    </div>
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm">제약/헬스케어</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        로슈, 노바티스<br/>
                        사노피, GSK
                      </p>
                    </div>
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm">산업/에너지</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        지멘스, ASML<br/>
                        토탈, BP, 쉘
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedMarket === 'emerging' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
                신흥시장 (인도, 브라질, 동남아)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">주요 신흥시장</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">인도 (BSE, NSE)</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 세계 최대 인구 시장</li>
                        <li>• IT/제약 산업 강세</li>
                        <li>• 높은 성장률 (GDP 7%+)</li>
                        <li>• 루피화 변동성</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">브라질 (B3)</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 중남미 최대 시장</li>
                        <li>• 원자재/농산물 중심</li>
                        <li>• 정치적 불안정성</li>
                        <li>• 헤알화 리스크</li>
                      </ul>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <h5 className="font-medium mb-2">동남아시아</h5>
                    <div className="grid md:grid-cols-3 gap-2 text-sm">
                      <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-2">
                        <strong>싱가포르</strong>
                        <p className="text-xs mt-1">금융 허브, REITs</p>
                      </div>
                      <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-2">
                        <strong>인도네시아</strong>
                        <p className="text-xs mt-1">내수시장, 자원</p>
                      </div>
                      <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-2">
                        <strong>베트남</strong>
                        <p className="text-xs mt-1">제조업, 성장성</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">신흥시장 투자 전략</h4>
                  <div className="space-y-2">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                      <strong>ETF 활용</strong>
                      <p className="mt-1">개별 종목보다 국가/섹터 ETF로 분산투자</p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                      <strong>현지 파트너</strong>
                      <p className="mt-1">현지 정보와 규제 이해가 필수</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💱 환율과 환헤지</h2>
        <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4">
            환율 리스크 관리
          </h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-semibold mb-3">환율이 수익률에 미치는 영향</h4>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`# 환율 영향 시뮬레이션
def calculate_fx_impact(initial_investment_krw, stock_return_pct, 
                       initial_fx_rate, final_fx_rate):
    """
    해외투자 시 환율 변동이 수익률에 미치는 영향 계산
    """
    # 초기 투자금액 (달러)
    initial_usd = initial_investment_krw / initial_fx_rate
    
    # 주식 수익 반영
    final_usd = initial_usd * (1 + stock_return_pct / 100)
    
    # 환율 변동 반영한 최종 원화 금액
    final_krw = final_usd * final_fx_rate
    
    # 총 수익률
    total_return_pct = ((final_krw - initial_investment_krw) / 
                       initial_investment_krw) * 100
    
    # 환율 기여도
    fx_return_pct = ((final_fx_rate - initial_fx_rate) / 
                    initial_fx_rate) * 100
    
    return {
        'stock_return': stock_return_pct,
        'fx_return': fx_return_pct,
        'total_return': total_return_pct,
        'fx_impact': total_return_pct - stock_return_pct
    }

# 예시: 미국주식 10% 상승, 환율 5% 하락
result = calculate_fx_impact(
    initial_investment_krw=10000000,  # 1천만원
    stock_return_pct=10,              # 주식 10% 상승
    initial_fx_rate=1300,             # 달러당 1,300원
    final_fx_rate=1235                # 달러당 1,235원 (5% 하락)
)
# 결과: 주식 +10%, 환율 -5% = 총 수익률 +4.5%`}</pre>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">환헤지 방법</h4>
              <ul className="text-sm space-y-1">
                <li>🛡️ <strong>환헤지 ETF:</strong> 자동 환헤지</li>
                <li>💵 <strong>달러 예금:</strong> 자연 헤지</li>
                <li>📊 <strong>선물환:</strong> 적극적 헤지</li>
                <li>🔄 <strong>통화 스왑:</strong> 장기 헤지</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">환헤지 고려사항</h4>
              <ul className="text-sm space-y-1">
                <li>• 헤지 비용 (연 1-3%)</li>
                <li>• 장기투자 vs 단기투자</li>
                <li>• 달러 강세/약세 전망</li>
                <li>• 포트폴리오 비중</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🌐 글로벌 포트폴리오 구성</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">지역별 자산배분 전략</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-semibold mb-3">추천 포트폴리오 예시</h4>
            <div className="space-y-3">
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                <h5 className="font-medium text-sm mb-2">보수적 포트폴리오</h5>
                <div className="grid grid-cols-4 gap-2 text-sm">
                  <div>한국: 50%</div>
                  <div>미국: 30%</div>
                  <div>선진국: 15%</div>
                  <div>신흥국: 5%</div>
                </div>
              </div>
              
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                <h5 className="font-medium text-sm mb-2">균형 포트폴리오</h5>
                <div className="grid grid-cols-4 gap-2 text-sm">
                  <div>한국: 30%</div>
                  <div>미국: 40%</div>
                  <div>선진국: 20%</div>
                  <div>신흥국: 10%</div>
                </div>
              </div>
              
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                <h5 className="font-medium text-sm mb-2">공격적 포트폴리오</h5>
                <div className="grid grid-cols-4 gap-2 text-sm">
                  <div>한국: 20%</div>
                  <div>미국: 35%</div>
                  <div>선진국: 25%</div>
                  <div>신흥국: 20%</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-3">글로벌 ETF 활용</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-medium text-sm mb-2">전세계 시장</h5>
                <ul className="text-sm space-y-1">
                  <li>• VT: 전세계 주식</li>
                  <li>• ACWI: 선진+신흥</li>
                  <li>• FTSE All-World</li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium text-sm mb-2">지역별 ETF</h5>
                <ul className="text-sm space-y-1">
                  <li>• SPY: 미국 S&P 500</li>
                  <li>• EFA: 선진국 (미국 제외)</li>
                  <li>• VWO: 신흥시장</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📋 해외투자 실무 가이드</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">해외주식 투자 프로세스</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">1. 계좌 개설</h4>
              <ul className="text-sm space-y-1">
                <li>✅ 해외주식 거래 가능 증권사 선택</li>
                <li>✅ 외화 계좌 개설 (달러, 유로 등)</li>
                <li>✅ 세금 관련 서류 작성 (W-8BEN)</li>
                <li>✅ 투자자 적합성 평가</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">2. 정보 수집</h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <strong>무료 정보원</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• Yahoo Finance</li>
                    <li>• Seeking Alpha</li>
                    <li>• Morningstar</li>
                    <li>• 각사 IR 페이지</li>
                  </ul>
                </div>
                <div>
                  <strong>유료 서비스</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• Bloomberg Terminal</li>
                    <li>• Reuters Eikon</li>
                    <li>• FactSet</li>
                    <li>• S&P Capital IQ</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">3. 세금 처리</h4>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded p-3">
                <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
                  해외주식 양도소득세
                </p>
                <ul className="text-sm space-y-1">
                  <li>• 기본공제: 연간 250만원</li>
                  <li>• 세율: 22% (지방세 포함)</li>
                  <li>• 신고: 다음해 5월 종합소득세 신고</li>
                  <li>• 손익통산: 국내외 합산 가능</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 글로벌 투자 리스크</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            주요 리스크 요인
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">시장 리스크</h4>
              <ul className="text-sm space-y-1">
                <li>• 국가별 경제 상황</li>
                <li>• 정치적 불안정성</li>
                <li>• 규제 변화</li>
                <li>• 시차로 인한 대응 지연</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">운영 리스크</h4>
              <ul className="text-sm space-y-1">
                <li>• 환율 변동</li>
                <li>• 정보 접근성 제한</li>
                <li>• 언어 장벽</li>
                <li>• 시스템 장애</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📚 글로벌 투자 체크리스트</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">해외투자 시작 전 확인사항</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>투자 국가의 경제/정치 상황 파악</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>환율 동향 및 환헤지 전략 수립</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>현지 거래시간 및 공휴일 확인</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>세금 처리 방법 숙지</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>정보 수집 채널 확보</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>포트폴리오 내 적정 비중 설정</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>비상시 청산 계획 수립</span>
            </label>
          </div>
        </div>
      </section>
    </div>
  );
}