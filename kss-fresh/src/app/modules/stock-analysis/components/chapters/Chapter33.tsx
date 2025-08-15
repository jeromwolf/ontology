'use client';

import { useState } from 'react';

export default function Chapter33() {
  const [selectedStrategy, setSelectedStrategy] = useState('macro');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">매크로 트레이딩과 종합 전략</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          거시경제 분석을 기반으로 한 글로벌 매크로 트레이딩과 지금까지 학습한 모든 전략을 
          통합하는 종합적인 투자 접근법을 마스터합니다. 시장의 큰 그림을 보고 투자하는 전문가가 되어봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🌐 매크로 트레이딩 전략</h2>
        <div className="mb-4">
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => setSelectedStrategy('macro')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedStrategy === 'macro'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              거시경제 분석
            </button>
            <button
              onClick={() => setSelectedStrategy('cycles')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedStrategy === 'cycles'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              경기 사이클
            </button>
            <button
              onClick={() => setSelectedStrategy('intermarket')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedStrategy === 'intermarket'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              인터마켓
            </button>
            <button
              onClick={() => setSelectedStrategy('integration')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedStrategy === 'integration'
                  ? 'bg-orange-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              통합 전략
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
          {selectedStrategy === 'macro' && (
            <div>
              <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
                거시경제 분석 프레임워크
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">핵심 거시경제 지표</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">성장 지표</h5>
                      <ul className="text-sm space-y-1">
                        <li>📊 <strong>GDP:</strong> 경제성장률 추세</li>
                        <li>🏭 <strong>PMI:</strong> 제조업/서비스업 경기</li>
                        <li>🛒 <strong>소매판매:</strong> 소비 동향</li>
                        <li>🏗️ <strong>주택착공:</strong> 건설 경기</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">물가/유동성 지표</h5>
                      <ul className="text-sm space-y-1">
                        <li>💰 <strong>CPI/PPI:</strong> 인플레이션</li>
                        <li>📈 <strong>금리:</strong> 중앙은행 정책</li>
                        <li>💵 <strong>통화량:</strong> M1, M2</li>
                        <li>💱 <strong>환율:</strong> 달러 인덱스</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">중앙은행 정책 분석</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`class CentralBankAnalysis:
    def __init__(self):
        self.fed_funds_rate = 5.5
        self.neutral_rate = 2.5
        
    def taylor_rule(self, inflation, target_inflation, output_gap):
        """테일러 준칙을 통한 적정 금리 계산"""
        # r* = neutral rate + inflation + 0.5*(inflation-target) + 0.5*output_gap
        optimal_rate = (self.neutral_rate + inflation + 
                       0.5 * (inflation - target_inflation) + 
                       0.5 * output_gap)
        
        return {
            'current_rate': self.fed_funds_rate,
            'optimal_rate': optimal_rate,
            'rate_gap': self.fed_funds_rate - optimal_rate,
            'policy_stance': 'restrictive' if self.fed_funds_rate > optimal_rate else 'accommodative'
        }
    
    def liquidity_analysis(self, money_supply_growth, gdp_growth, inflation):
        """유동성 과잉/부족 분석"""
        # 머니 갭 = M2 성장률 - (GDP 성장률 + 인플레이션)
        money_gap = money_supply_growth - (gdp_growth + inflation)
        
        if money_gap > 3:
            liquidity_status = "과잉 유동성 - 자산 가격 상승 압력"
        elif money_gap < -3:
            liquidity_status = "유동성 부족 - 경기 둔화 위험"
        else:
            liquidity_status = "적정 유동성"
            
        return {
            'money_gap': money_gap,
            'status': liquidity_status,
            'asset_implication': self.get_asset_implication(money_gap)
        }
    
    def get_asset_implication(self, money_gap):
        """유동성에 따른 자산 배분 시사점"""
        if money_gap > 3:
            return {
                'stocks': 'Overweight',
                'bonds': 'Underweight',
                'commodities': 'Overweight',
                'cash': 'Underweight'
            }
        elif money_gap < -3:
            return {
                'stocks': 'Underweight',
                'bonds': 'Overweight',
                'commodities': 'Underweight',
                'cash': 'Overweight'
            }
        else:
            return {
                'stocks': 'Neutral',
                'bonds': 'Neutral',
                'commodities': 'Neutral',
                'cash': 'Neutral'
            }`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">글로벌 매크로 테마</h4>
                  <div className="grid md:grid-cols-2 gap-3">
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">구조적 테마</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 인구 고령화</li>
                        <li>• 탈세계화</li>
                        <li>• 에너지 전환</li>
                        <li>• 디지털 전환</li>
                      </ul>
                    </div>
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">순환적 테마</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 금리 사이클</li>
                        <li>• 상품 슈퍼사이클</li>
                        <li>• 달러 사이클</li>
                        <li>• 신용 사이클</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'cycles' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
                경기 사이클과 자산 배분
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">경기 사이클 4단계</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">1. 회복기 (Recovery)</h5>
                      <ul className="text-xs space-y-1">
                        <li><strong>특징:</strong> 경제 지표 개선 시작</li>
                        <li><strong>금리:</strong> 여전히 낮음</li>
                        <li><strong>선호자산:</strong> 주식 (소형주, 가치주)</li>
                        <li><strong>회피자산:</strong> 현금, 단기채</li>
                      </ul>
                    </div>
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">2. 확장기 (Expansion)</h5>
                      <ul className="text-xs space-y-1">
                        <li><strong>특징:</strong> 경제 활황, 고용 증가</li>
                        <li><strong>금리:</strong> 점진적 인상</li>
                        <li><strong>선호자산:</strong> 성장주, 원자재</li>
                        <li><strong>회피자산:</strong> 장기채</li>
                      </ul>
                    </div>
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">3. 둔화기 (Slowdown)</h5>
                      <ul className="text-xs space-y-1">
                        <li><strong>특징:</strong> 성장률 둔화, 인플레 상승</li>
                        <li><strong>금리:</strong> 고점 도달</li>
                        <li><strong>선호자산:</strong> 방어주, 채권</li>
                        <li><strong>회피자산:</strong> 경기민감주</li>
                      </ul>
                    </div>
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">4. 침체기 (Recession)</h5>
                      <ul className="text-xs space-y-1">
                        <li><strong>특징:</strong> 마이너스 성장</li>
                        <li><strong>금리:</strong> 급격히 하락</li>
                        <li><strong>선호자산:</strong> 국채, 금</li>
                        <li><strong>회피자산:</strong> 주식, 하이일드</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">섹터 로테이션 전략</h4>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded p-4">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b dark:border-gray-600">
                          <th className="text-left py-2">경기 단계</th>
                          <th className="text-left py-2">선호 섹터</th>
                          <th className="text-left py-2">회피 섹터</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b dark:border-gray-600">
                          <td className="py-2">회복기</td>
                          <td>금융, 부동산, 산업재</td>
                          <td>유틸리티, 필수소비재</td>
                        </tr>
                        <tr className="border-b dark:border-gray-600">
                          <td className="py-2">확장기</td>
                          <td>IT, 경기소비재, 에너지</td>
                          <td>채권형, 방어주</td>
                        </tr>
                        <tr className="border-b dark:border-gray-600">
                          <td className="py-2">둔화기</td>
                          <td>헬스케어, 필수소비재</td>
                          <td>소재, 산업재</td>
                        </tr>
                        <tr>
                          <td className="py-2">침체기</td>
                          <td>유틸리티, 통신</td>
                          <td>금융, 경기소비재</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">선행 지표 모니터링</h4>
                  <ul className="text-sm space-y-2">
                    <li>📈 <strong>수익률 곡선:</strong> 장단기 금리차 (2년-10년)</li>
                    <li>📊 <strong>LEI:</strong> 경기선행지수 변화율</li>
                    <li>🏢 <strong>신규 실업수당 청구:</strong> 고용시장 선행지표</li>
                    <li>🏗️ <strong>건축허가:</strong> 미래 건설활동 예측</li>
                    <li>📦 <strong>ISM 신규주문:</strong> 제조업 수요 전망</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'intermarket' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
                인터마켓 분석
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">4대 자산군 상관관계</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`# 인터마켓 상관관계 분석
class IntermarketAnalysis:
    def __init__(self):
        # 기본 상관관계 (정상 시장)
        self.normal_correlations = {
            'stocks_bonds': -0.3,      # 역상관
            'stocks_commodities': 0.4,  # 정상관
            'bonds_commodities': -0.5,  # 역상관
            'dollar_commodities': -0.7, # 강한 역상관
            'stocks_dollar': -0.2,      # 약한 역상관
            'bonds_dollar': 0.3         # 정상관
        }
        
    def regime_detection(self, current_correlations):
        """시장 레짐 감지"""
        # Risk-On vs Risk-Off 판단
        stock_bond_corr = current_correlations['stocks_bonds']
        
        if stock_bond_corr > 0.5:
            regime = "Risk-Off"
            characteristics = [
                "주식-채권 동반 하락",
                "안전자산 선호 (금, 달러, 엔)",
                "변동성 급등",
                "유동성 경색"
            ]
        elif stock_bond_corr < -0.5:
            regime = "Risk-On"
            characteristics = [
                "주식 상승, 채권 하락",
                "위험자산 선호",
                "변동성 하락",
                "캐리 트레이드 활발"
            ]
        else:
            regime = "Normal"
            characteristics = [
                "정상적 상관관계",
                "섹터 로테이션",
                "펀더멘털 중심",
                "선별적 투자"
            ]
            
        return regime, characteristics
    
    def lead_lag_relationships(self):
        """선행-후행 관계"""
        return {
            '채권 → 주식': "채권 수익률 상승은 3-6개월 후 주식 하락",
            '달러 → 원자재': "달러 강세는 즉시 원자재 가격 하락",
            '구리 → 주식': "구리 가격은 글로벌 경기 선행지표",
            '금/구리 비율': "Risk-On/Off 센티먼트 지표",
            'VIX → 신용스프레드': "변동성이 신용시장 리스크 선행"
        }`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">크로스 에셋 전략</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">리플레이션 트레이드</h5>
                      <ul className="text-xs space-y-1">
                        <li>• Long: 주식, 원자재, 신흥국</li>
                        <li>• Short: 채권, 달러, 금</li>
                        <li>• 배경: 경기 회복, 통화 완화</li>
                      </ul>
                    </div>
                    <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">디플레이션 헤지</h5>
                      <ul className="text-xs space-y-1">
                        <li>• Long: 국채, 금, 엔화</li>
                        <li>• Short: 주식, 원유, 신흥국</li>
                        <li>• 배경: 경기 침체, 디플레 우려</li>
                      </ul>
                    </div>
                  </div>
                  
                  <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded p-3">
                    <h5 className="font-medium text-sm mb-2">💡 실전 팁</h5>
                    <p className="text-sm">
                      달러 인덱스(DXY)를 중심으로 글로벌 자금 흐름을 파악하고,
                      금/은 비율, 구리/금 비율 등 상대가격으로 시장 심리를 측정하세요.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'integration' && (
            <div>
              <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4">
                통합 투자 전략
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">다층적 분석 프레임워크</h4>
                  <div className="space-y-3">
                    <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">1단계: 매크로 환경 분석</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 글로벌 경제 성장률과 방향성</li>
                        <li>• 중앙은행 정책 스탠스</li>
                        <li>• 지정학적 리스크 평가</li>
                        <li>• 구조적 트렌드 파악</li>
                      </ul>
                    </div>
                    
                    <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">2단계: 자산 배분 결정</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 주식/채권/대체자산 비중</li>
                        <li>• 지역별 배분 (선진국/신흥국)</li>
                        <li>• 섹터/스타일 선택</li>
                        <li>• 헤지 전략 수립</li>
                      </ul>
                    </div>
                    
                    <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">3단계: 종목 선정</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 펀더멘털 스크리닝</li>
                        <li>• 기술적 진입 타이밍</li>
                        <li>• 포지션 사이징</li>
                        <li>• 손절/익절 설정</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">포트폴리오 구축 예시</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`# 전천후 포트폴리오 구축
class AllWeatherPortfolio:
    def __init__(self, total_capital=100000000):  # 1억원
        self.capital = total_capital
        self.positions = {}
        
    def build_portfolio(self, market_regime):
        """시장 상황에 따른 포트폴리오 구축"""
        
        if market_regime == "growth_inflation_up":
            # 성장↑ 인플레↑: 원자재, 신흥국 중심
            allocation = {
                'commodities': 0.25,
                'emerging_stocks': 0.20,
                'real_estate': 0.15,
                'tips': 0.15,  # 물가연동채
                'developed_stocks': 0.15,
                'short_bonds': 0.10
            }
            
        elif market_regime == "growth_up_inflation_down":
            # 성장↑ 인플레↓: 주식 중심 (골디락스)
            allocation = {
                'developed_stocks': 0.35,
                'emerging_stocks': 0.20,
                'corporate_bonds': 0.15,
                'real_estate': 0.15,
                'commodities': 0.10,
                'cash': 0.05
            }
            
        elif market_regime == "growth_down_inflation_up":
            # 성장↓ 인플레↑: 스태그플레이션
            allocation = {
                'gold': 0.25,
                'tips': 0.20,
                'commodities': 0.15,
                'defensive_stocks': 0.15,
                'short_bonds': 0.15,
                'cash': 0.10
            }
            
        else:  # growth_down_inflation_down
            # 성장↓ 인플레↓: 디플레이션
            allocation = {
                'long_bonds': 0.30,
                'gold': 0.20,
                'defensive_stocks': 0.20,
                'quality_stocks': 0.15,
                'cash': 0.15
            }
            
        return self.execute_allocation(allocation)
    
    def execute_allocation(self, allocation):
        """실제 자산 배분 실행"""
        for asset, weight in allocation.items():
            position_size = self.capital * weight
            self.positions[asset] = {
                'weight': weight,
                'amount': position_size,
                'securities': self.select_securities(asset, position_size)
            }
        return self.positions`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">리스크 관리 통합 시스템</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium text-sm mb-2">포트폴리오 리스크</h5>
                      <ul className="text-sm space-y-1">
                        <li>• VaR (Value at Risk) 관리</li>
                        <li>• 최대 낙폭 한도 설정</li>
                        <li>• 상관관계 모니터링</li>
                        <li>• 스트레스 테스트</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium text-sm mb-2">동적 조정</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 월간 리밸런싱</li>
                        <li>• 전술적 자산배분</li>
                        <li>• 헤지 비율 조정</li>
                        <li>• 시나리오 대응</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 실전 매크로 트레이딩</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">현재 시장 분석 (2024년 기준)</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">글로벌 매크로 환경</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h5 className="font-medium text-sm mb-2">주요 이슈</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 인플레이션 완화 속도</li>
                    <li>• 중앙은행 피벗 타이밍</li>
                    <li>• 중국 경제 회복력</li>
                    <li>• 지정학적 긴장 지속</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-sm mb-2">투자 함의</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 채권 매력도 상승</li>
                    <li>• 성장주 → 가치주 전환</li>
                    <li>• 달러 약세 가능성</li>
                    <li>• 원자재 선별적 접근</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">시나리오별 전략</h4>
              <div className="space-y-3">
                <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                  <h5 className="font-medium text-sm mb-2">시나리오 1: 연착륙 (60%)</h5>
                  <p className="text-sm">
                    경기 둔화 속 인플레 안정, 금리 인하 시작<br/>
                    → 주식/채권 균형, 퀄리티 중심
                  </p>
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                  <h5 className="font-medium text-sm mb-2">시나리오 2: 경기침체 (30%)</h5>
                  <p className="text-sm">
                    실업률 급등, 수요 급감, 디플레 우려<br/>
                    → 국채 비중 확대, 방어주 중심
                  </p>
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                  <h5 className="font-medium text-sm mb-2">시나리오 3: 재인플레 (10%)</h5>
                  <p className="text-sm">
                    인플레 재상승, 금리 추가 인상<br/>
                    → 원자재, 단기채, 가치주
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 성과 측정과 개선</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">투자 성과 분석 프레임워크</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">성과 귀인 분석</h4>
              <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm">
{`# 성과 귀인 분석 (Performance Attribution)
def performance_attribution(portfolio_returns, benchmark_returns):
    """포트폴리오 초과수익 요인 분해"""
    
    # 1. 자산배분 효과
    asset_allocation_effect = sum(
        (portfolio_weight - benchmark_weight) * benchmark_return
        for asset in assets
    )
    
    # 2. 종목선택 효과  
    security_selection_effect = sum(
        benchmark_weight * (portfolio_return - benchmark_return)
        for asset in assets
    )
    
    # 3. 상호작용 효과
    interaction_effect = sum(
        (portfolio_weight - benchmark_weight) * 
        (portfolio_return - benchmark_return)
        for asset in assets
    )
    
    total_excess_return = (asset_allocation_effect + 
                          security_selection_effect + 
                          interaction_effect)
    
    return {
        'total_excess': total_excess_return,
        'asset_allocation': asset_allocation_effect,
        'security_selection': security_selection_effect,
        'interaction': interaction_effect,
        'breakdown': {
            'allocation_pct': asset_allocation_effect / total_excess_return * 100,
            'selection_pct': security_selection_effect / total_excess_return * 100
        }
    }`}</pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-3">투자 일지 작성법</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h5 className="font-medium text-sm mb-2">거래 기록</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 매매 근거와 목표</li>
                    <li>• 진입/청산 가격</li>
                    <li>• 포지션 크기와 이유</li>
                    <li>• 시장 환경 스냅샷</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-sm mb-2">사후 분석</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 의사결정 평가</li>
                    <li>• 실수와 교훈</li>
                    <li>• 개선 방안</li>
                    <li>• 심리 상태 기록</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🚀 전문 투자자로의 여정</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">지속적 성장을 위한 로드맵</h3>
          
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">1</span>
              <div className="flex-1">
                <h4 className="font-semibold">기초 다지기 (1-2년)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  재무제표 읽기, 기술적 분석, 리스크 관리 기초
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">2</span>
              <div className="flex-1">
                <h4 className="font-semibold">전문성 구축 (3-5년)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  특화 분야 선정, 시스템 구축, 실전 경험 축적
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">3</span>
              <div className="flex-1">
                <h4 className="font-semibold">통합과 혁신 (5년+)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  매크로 관점 통합, 독자적 전략 개발, 지속가능한 알파 창출
                </p>
              </div>
            </div>
          </div>

          <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-3">평생 학습 체크리스트</h4>
            <div className="space-y-2">
              <label className="flex items-start gap-3">
                <input type="checkbox" className="mt-1" />
                <span className="text-sm">매일 시장 리뷰와 학습 (최소 2시간)</span>
              </label>
              <label className="flex items-start gap-3">
                <input type="checkbox" className="mt-1" />
                <span className="text-sm">분기별 포트폴리오 전면 점검</span>
              </label>
              <label className="flex items-start gap-3">
                <input type="checkbox" className="mt-1" />
                <span className="text-sm">연간 새로운 전략/기법 1개 이상 마스터</span>
              </label>
              <label className="flex items-start gap-3">
                <input type="checkbox" className="mt-1" />
                <span className="text-sm">투자 커뮤니티 활동 및 네트워킹</span>
              </label>
              <label className="flex items-start gap-3">
                <input type="checkbox" className="mt-1" />
                <span className="text-sm">실패 분석과 개선 프로세스 정립</span>
              </label>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎓 마무리: 투자의 본질</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">성공적인 투자를 위한 핵심 원칙</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">투자 철학 정립</h4>
              <ul className="text-sm space-y-2">
                <li>💡 <strong>자기 이해:</strong> 본인의 강점, 약점, 성향 파악</li>
                <li>🎯 <strong>일관성:</strong> 원칙에 따른 일관된 의사결정</li>
                <li>📚 <strong>겸손함:</strong> 시장에 대한 경외심과 지속적 학습</li>
                <li>⚖️ <strong>균형:</strong> 수익과 리스크의 적절한 균형</li>
                <li>⏳ <strong>인내심:</strong> 장기적 관점과 복리의 힘 신뢰</li>
              </ul>
            </div>

            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-4">
              <p className="text-center font-semibold text-lg mb-2">
                "The best investment you can make is in yourself"
              </p>
              <p className="text-center text-sm text-gray-600 dark:text-gray-400">
                - Warren Buffett -
              </p>
              <p className="text-center mt-4 text-sm">
                지식과 경험에 대한 투자가 가장 높은 수익률을 가져다 줍니다.<br/>
                꾸준한 학습과 실천으로 여러분만의 투자 철학을 만들어가세요.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}