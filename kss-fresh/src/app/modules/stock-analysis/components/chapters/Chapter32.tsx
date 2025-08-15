'use client';

import { useState } from 'react';

export default function Chapter32() {
  const [selectedAsset, setSelectedAsset] = useState('reits');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">대체투자 전략</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          부동산, 원자재, 사모펀드 등 전통적인 주식/채권 외의 대체투자 자산을 학습합니다.
          포트폴리오 다변화와 절대수익 추구를 위한 대체투자 전략을 마스터해봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏢 대체투자 자산군</h2>
        <div className="mb-4">
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => setSelectedAsset('reits')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedAsset === 'reits'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              리츠/부동산
            </button>
            <button
              onClick={() => setSelectedAsset('commodities')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedAsset === 'commodities'
                  ? 'bg-yellow-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              원자재
            </button>
            <button
              onClick={() => setSelectedAsset('pe')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedAsset === 'pe'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              사모펀드
            </button>
            <button
              onClick={() => setSelectedAsset('hedge')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedAsset === 'hedge'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              헤지펀드
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
          {selectedAsset === 'reits' && (
            <div>
              <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
                리츠(REITs)와 부동산 투자
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">리츠(REITs) 개요</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Real Estate Investment Trusts - 부동산 투자신탁으로 상장 주식처럼 거래 가능한 부동산 투자 상품
                  </p>
                  
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">리츠의 장점</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 소액으로 부동산 투자 가능</li>
                        <li>• 높은 유동성 (거래소 매매)</li>
                        <li>• 정기적인 배당 수익</li>
                        <li>• 전문적 자산 운용</li>
                        <li>• 인플레이션 헤지</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">리츠 유형</h5>
                      <ul className="text-sm space-y-1">
                        <li><strong>Equity REITs:</strong> 부동산 직접 소유</li>
                        <li><strong>Mortgage REITs:</strong> 부동산 대출</li>
                        <li><strong>Hybrid REITs:</strong> 혼합형</li>
                        <li><strong>섹터별:</strong> 주거, 오피스, 리테일, 물류</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">리츠 투자 분석</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`class REITAnalysis:
    def __init__(self):
        self.risk_free_rate = 0.03
        
    def calculate_ffo(self, net_income, depreciation, gains_on_sale):
        """FFO (Funds From Operations) 계산"""
        # 리츠 수익성의 핵심 지표
        ffo = net_income + depreciation - gains_on_sale
        return ffo
        
    def calculate_affo(self, ffo, capex, straight_line_rent):
        """AFFO (Adjusted FFO) 계산"""
        # 실제 분배 가능한 현금흐름
        affo = ffo - capex - straight_line_rent
        return affo
        
    def evaluate_reit(self, reit_data):
        """리츠 종합 평가"""
        # P/FFO 배수 (일반 주식의 PER과 유사)
        p_ffo = reit_data['price'] / reit_data['ffo_per_share']
        
        # 배당수익률
        dividend_yield = reit_data['dividend'] / reit_data['price']
        
        # NAV 프리미엄/할인
        nav_premium = (reit_data['price'] - reit_data['nav']) / reit_data['nav']
        
        # 부채비율 (LTV)
        ltv_ratio = reit_data['debt'] / reit_data['asset_value']
        
        # 점유율
        occupancy_rate = reit_data['occupied_sqft'] / reit_data['total_sqft']
        
        score = 0
        if p_ffo < 15: score += 2
        if dividend_yield > 0.05: score += 2
        if nav_premium < 0: score += 1  # NAV 대비 할인
        if ltv_ratio < 0.5: score += 2
        if occupancy_rate > 0.95: score += 2
        
        return {
            'p_ffo': p_ffo,
            'dividend_yield': dividend_yield * 100,
            'nav_premium': nav_premium * 100,
            'ltv_ratio': ltv_ratio * 100,
            'occupancy_rate': occupancy_rate * 100,
            'investment_score': score,
            'recommendation': 'BUY' if score >= 7 else 'HOLD' if score >= 4 else 'SELL'
        }`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">글로벌 리츠 시장</h4>
                  <div className="grid md:grid-cols-3 gap-3">
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
                      <h5 className="font-medium text-sm">미국 리츠</h5>
                      <ul className="text-xs mt-2 space-y-1">
                        <li>• 세계 최대 시장</li>
                        <li>• 다양한 섹터</li>
                        <li>• VNQ, XLRE ETF</li>
                      </ul>
                    </div>
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
                      <h5 className="font-medium text-sm">한국 리츠</h5>
                      <ul className="text-xs mt-2 space-y-1">
                        <li>• 성장 초기 단계</li>
                        <li>• 물류/데이터센터 인기</li>
                        <li>• 배당소득세 분리과세</li>
                      </ul>
                    </div>
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
                      <h5 className="font-medium text-sm">싱가포르 리츠</h5>
                      <ul className="text-xs mt-2 space-y-1">
                        <li>• 아시아 리츠 허브</li>
                        <li>• 높은 배당률</li>
                        <li>• S-REITs</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedAsset === 'commodities' && (
            <div>
              <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-4">
                원자재 투자
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">주요 원자재 카테고리</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">에너지</h5>
                      <ul className="text-sm space-y-1">
                        <li>🛢️ <strong>원유:</strong> WTI, Brent</li>
                        <li>⚡ <strong>천연가스:</strong> Henry Hub</li>
                        <li>🔋 <strong>리튬:</strong> 배터리 핵심 소재</li>
                        <li>⚛️ <strong>우라늄:</strong> 원자력 발전</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">금속</h5>
                      <ul className="text-sm space-y-1">
                        <li>🥇 <strong>귀금속:</strong> 금, 은, 플래티넘</li>
                        <li>🏗️ <strong>산업금속:</strong> 구리, 알루미늄</li>
                        <li>🔧 <strong>희토류:</strong> 네오디뮴, 디스프로슘</li>
                        <li>🏭 <strong>철광석:</strong> 제철 산업</li>
                      </ul>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <h5 className="font-medium mb-2">농산물</h5>
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded p-2">
                        <strong>곡물</strong>
                        <p className="text-xs mt-1">밀, 옥수수, 대두</p>
                      </div>
                      <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded p-2">
                        <strong>소프트</strong>
                        <p className="text-xs mt-1">커피, 설탕, 코코아</p>
                      </div>
                      <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded p-2">
                        <strong>축산물</strong>
                        <p className="text-xs mt-1">생우, 돈육</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">원자재 투자 방법</h4>
                  <div className="space-y-3">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">1. 원자재 ETF</h5>
                      <ul className="text-sm space-y-1">
                        <li>• DBC: 다양한 원자재 바스켓</li>
                        <li>• GLD, IAU: 금 ETF</li>
                        <li>• USO: 원유 ETF</li>
                        <li>• DBA: 농산물 ETF</li>
                      </ul>
                    </div>
                    
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">2. 원자재 관련주</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 광산 기업 (BHP, Rio Tinto)</li>
                        <li>• 에너지 기업 (Exxon, Chevron)</li>
                        <li>• 농업 기업 (ADM, Bunge)</li>
                        <li>• 가공/유통 기업</li>
                      </ul>
                    </div>
                    
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">3. 선물 계약</h5>
                      <p className="text-sm">직접적이지만 높은 레버리지와 만기 관리 필요</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">슈퍼사이클과 투자 타이밍</h4>
                  <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded p-3">
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      원자재는 수요/공급 불균형에 따른 장기 사이클을 보입니다.
                      신흥국 성장, 인프라 투자, 에너지 전환 등이 주요 동력입니다.
                    </p>
                    <ul className="text-sm mt-3 space-y-1">
                      <li>• 2000-2008: 중국 주도 슈퍼사이클</li>
                      <li>• 2020-현재: 그린 에너지 전환 사이클</li>
                      <li>• 인플레이션 헤지 수단으로 주목</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedAsset === 'pe' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
                사모펀드(Private Equity)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">사모펀드 구조와 전략</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">펀드 구조</h5>
                      <ul className="text-sm space-y-1">
                        <li><strong>GP(General Partner):</strong> 운용사</li>
                        <li><strong>LP(Limited Partner):</strong> 투자자</li>
                        <li><strong>투자기간:</strong> 보통 7-10년</li>
                        <li><strong>최소투자:</strong> 수억원 이상</li>
                        <li><strong>수수료:</strong> 2% 관리보수 + 20% 성과보수</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">투자 전략</h5>
                      <ul className="text-sm space-y-1">
                        <li><strong>Buyout:</strong> 기업 인수 후 가치 제고</li>
                        <li><strong>Growth:</strong> 성장 자본 제공</li>
                        <li><strong>Distressed:</strong> 부실기업 인수</li>
                        <li><strong>Venture:</strong> 스타트업 투자</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">가치 창출 메커니즘</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`class PrivateEquityReturns:
    def calculate_irr(self, cashflows, dates):
        """내부수익률(IRR) 계산"""
        # cashflows: [-100, 0, 0, 0, 150] (투자 및 회수)
        # dates: 각 현금흐름의 날짜
        
        from scipy.optimize import newton
        from datetime import datetime
        
        def npv(rate):
            total = 0
            for i, cf in enumerate(cashflows):
                years = (dates[i] - dates[0]).days / 365.25
                total += cf / (1 + rate) ** years
            return total
        
        try:
            irr = newton(npv, 0.1)  # 초기값 10%
            return irr * 100
        except:
            return None
    
    def value_creation_analysis(self, entry_metrics, exit_metrics):
        """가치 창출 요인 분석"""
        # 매출 성장
        revenue_growth = (exit_metrics['revenue'] / 
                         entry_metrics['revenue']) ** (1/5) - 1
        
        # 마진 개선
        ebitda_margin_delta = (exit_metrics['ebitda_margin'] - 
                              entry_metrics['ebitda_margin'])
        
        # 멀티플 확대
        multiple_expansion = (exit_metrics['ev_ebitda'] - 
                            entry_metrics['ev_ebitda'])
        
        # 레버리지 효과
        leverage_contribution = (entry_metrics['debt'] * 0.07 * 5) / \
                              entry_metrics['equity']
        
        return {
            'revenue_cagr': revenue_growth * 100,
            'margin_improvement': ebitda_margin_delta,
            'multiple_expansion': multiple_expansion,
            'leverage_effect': leverage_contribution * 100,
            'total_value_creation': (
                revenue_growth + ebitda_margin_delta/100 + 
                multiple_expansion/10 + leverage_contribution
            ) * 100
        }`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">개인투자자 접근 방법</h4>
                  <div className="space-y-2">
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3 text-sm">
                      <strong>상장 PE 펀드</strong>
                      <p className="mt-1">Blackstone, KKR, Apollo 등 상장된 운용사 주식</p>
                    </div>
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3 text-sm">
                      <strong>PE 연계 상품</strong>
                      <p className="mt-1">증권사 랩어카운트, 자산운용사 재간접펀드</p>
                    </div>
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3 text-sm">
                      <strong>크라우드펀딩</strong>
                      <p className="mt-1">소액으로 참여 가능한 온라인 플랫폼</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedAsset === 'hedge' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
                헤지펀드 전략
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">주요 헤지펀드 전략</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">Long/Short Equity</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                        저평가 주식 매수 + 고평가 주식 공매도
                      </p>
                      <ul className="text-xs space-y-1">
                        <li>• 시장 중립적 포지션</li>
                        <li>• 알파 추구</li>
                        <li>• 섹터/팩터 익스포저 관리</li>
                      </ul>
                    </div>
                    
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">Market Neutral</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                        시장 리스크 완전 헤지
                      </p>
                      <ul className="text-xs space-y-1">
                        <li>• 베타 = 0 유지</li>
                        <li>• 페어 트레이딩</li>
                        <li>• 통계적 차익거래</li>
                      </ul>
                    </div>
                    
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">Global Macro</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                        거시경제 트렌드 기반
                      </p>
                      <ul className="text-xs space-y-1">
                        <li>• 통화, 금리, 원자재</li>
                        <li>• 국가별 자산배분</li>
                        <li>• 이벤트 드리븐</li>
                      </ul>
                    </div>
                    
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">Quantitative</h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                        알고리즘 기반 체계적 거래
                      </p>
                      <ul className="text-xs space-y-1">
                        <li>• 고빈도 거래(HFT)</li>
                        <li>• 머신러닝 활용</li>
                        <li>• 리스크 패리티</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">헤지펀드 성과 분석</h4>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded p-4">
                    <h5 className="font-medium text-sm mb-3">주요 성과 지표</h5>
                    <div className="grid md:grid-cols-2 gap-3 text-sm">
                      <div>
                        <strong>절대수익 지표</strong>
                        <ul className="mt-1 space-y-1">
                          <li>• 연평균 수익률</li>
                          <li>• 월간 승률</li>
                          <li>• 최대 낙폭(MDD)</li>
                          <li>• 회복 기간</li>
                        </ul>
                      </div>
                      <div>
                        <strong>위험조정 지표</strong>
                        <ul className="mt-1 space-y-1">
                          <li>• 샤프 비율 &gt; 1.0</li>
                          <li>• 소르티노 비율</li>
                          <li>• 칼마 비율</li>
                          <li>• 정보 비율</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">헤지펀드 리스크</h4>
                  <div className="bg-red-50 dark:bg-red-900/20 rounded p-3">
                    <ul className="text-sm space-y-1">
                      <li>⚠️ <strong>유동성 리스크:</strong> 환매 제한, 게이트 조항</li>
                      <li>⚠️ <strong>레버리지 리스크:</strong> 과도한 차입 사용</li>
                      <li>⚠️ <strong>운용사 리스크:</strong> 키맨 리스크, 운영 리스크</li>
                      <li>⚠️ <strong>높은 수수료:</strong> 2/20 구조 (2% 운용보수 + 20% 성과보수)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎨 대체투자 포트폴리오 구성</h2>
        <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4">
            전통자산과 대체자산의 조합
          </h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-semibold mb-3">예일대 기금 모델 (David Swensen)</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-medium text-sm mb-2">자산배분 예시</h5>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>국내 주식:</span>
                    <span className="font-medium">5%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>해외 주식:</span>
                    <span className="font-medium">15%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>채권:</span>
                    <span className="font-medium">10%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>부동산:</span>
                    <span className="font-medium">20%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>사모펀드:</span>
                    <span className="font-medium">30%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>헤지펀드:</span>
                    <span className="font-medium">20%</span>
                  </div>
                </div>
              </div>
              <div>
                <h5 className="font-medium text-sm mb-2">기대효과</h5>
                <ul className="text-sm space-y-1">
                  <li>• 변동성 감소</li>
                  <li>• 수익률 안정화</li>
                  <li>• 하방 리스크 제한</li>
                  <li>• 인플레이션 보호</li>
                  <li>• 비상관 수익원</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-3">개인투자자를 위한 대체투자 전략</h4>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`# 대체투자 포트폴리오 최적화
import numpy as np
from scipy.optimize import minimize

class AlternativePortfolio:
    def __init__(self, returns, correlations):
        self.returns = returns  # 기대수익률
        self.corr_matrix = correlations  # 상관계수 행렬
        self.n_assets = len(returns)
        
    def portfolio_stats(self, weights):
        """포트폴리오 수익률과 위험 계산"""
        portfolio_return = np.sum(self.returns * weights)
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(self.corr_matrix, weights))
        )
        sharpe = portfolio_return / portfolio_std
        return portfolio_return, portfolio_std, sharpe
    
    def optimize_portfolio(self, target_return=None):
        """최적 포트폴리오 도출"""
        # 제약조건
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 비중 합 = 1
        ]
        
        if target_return:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(x * self.returns) - target_return
            })
        
        # 경계조건 (각 자산 0-40%)
        bounds = tuple((0, 0.4) for _ in range(self.n_assets))
        
        # 초기값
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # 최적화 (변동성 최소화)
        result = minimize(
            lambda x: np.sqrt(np.dot(x.T, np.dot(self.corr_matrix, x))),
            x0, method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        return result.x

# 사용 예시
assets = ['주식', '채권', '리츠', '원자재', '헤지펀드']
expected_returns = np.array([0.08, 0.04, 0.07, 0.06, 0.09])
correlations = np.array([
    [1.00, 0.15, 0.60, 0.30, 0.40],  # 주식
    [0.15, 1.00, 0.20, 0.10, 0.05],  # 채권
    [0.60, 0.20, 1.00, 0.40, 0.35],  # 리츠
    [0.30, 0.10, 0.40, 1.00, 0.25],  # 원자재
    [0.40, 0.05, 0.35, 0.25, 1.00]   # 헤지펀드
])

portfolio = AlternativePortfolio(expected_returns, correlations)
optimal_weights = portfolio.optimize_portfolio(target_return=0.07)`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 대체투자 실전 가이드</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">단계별 접근 방법</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">1단계: 기초 지식 습득</h4>
              <ul className="text-sm space-y-1">
                <li>✅ 각 대체자산의 특성 이해</li>
                <li>✅ 위험-수익 프로파일 분석</li>
                <li>✅ 세금 및 규제 환경 파악</li>
                <li>✅ 최소 투자금액 확인</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">2단계: 소액 시작</h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <strong>ETF로 시작</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 리츠 ETF</li>
                    <li>• 원자재 ETF</li>
                    <li>• 멀티에셋 ETF</li>
                  </ul>
                </div>
                <div>
                  <strong>간접 투자</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 재간접 펀드</li>
                    <li>• 랩어카운트</li>
                    <li>• 로보어드바이저</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">3단계: 포트폴리오 확대</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                전체 포트폴리오의 20-30%까지 점진적 확대
              </p>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded p-3">
                <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
                  주의사항
                </p>
                <ul className="text-sm mt-1 space-y-1">
                  <li>• 유동성 확보 (비상금 별도)</li>
                  <li>• 정기적 리밸런싱</li>
                  <li>• 성과 모니터링</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 대체투자 리스크 관리</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            주요 위험 요소와 대응
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">구조적 리스크</h4>
              <ul className="text-sm space-y-1">
                <li>• 낮은 유동성</li>
                <li>• 높은 최소 투자금액</li>
                <li>• 복잡한 구조</li>
                <li>• 정보 비대칭</li>
                <li>• 긴 투자 기간</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">시장 리스크</h4>
              <ul className="text-sm space-y-1">
                <li>• 가격 변동성</li>
                <li>• 상관관계 증가</li>
                <li>• 규제 변화</li>
                <li>• 거시경제 충격</li>
                <li>• 환율 리스크</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2">리스크 관리 원칙</h4>
            <ol className="text-sm space-y-1">
              <li>1. 전체 자산의 30% 이하로 제한</li>
              <li>2. 다양한 대체자산에 분산</li>
              <li>3. 투자 전 실사(Due Diligence) 철저히</li>
              <li>4. 출구 전략 사전 수립</li>
              <li>5. 정기적인 성과 평가</li>
            </ol>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📚 대체투자 체크리스트</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">투자 전 확인사항</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>투자 목적과 기간이 명확한가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>유동성 필요 시기를 고려했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>수수료와 세금을 파악했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>운용사의 실적과 신뢰도를 확인했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>최악의 시나리오를 감당할 수 있는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>전체 포트폴리오에서 적정 비중인가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>정기적인 모니터링 계획이 있는가?</span>
            </label>
          </div>
        </div>
      </section>
    </div>
  );
}