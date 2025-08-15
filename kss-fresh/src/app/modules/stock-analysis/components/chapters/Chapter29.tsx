'use client';

import { useState } from 'react';

export default function Chapter29() {
  const [selectedStrategy, setSelectedStrategy] = useState('volatility');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">고급 옵션 전략</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          전문 트레이더들이 사용하는 고급 옵션 전략과 구조화 상품을 깊이 있게 학습합니다.
          변동성 거래, 디스퍼전 트레이딩, 이색옵션까지 실무에서 활용되는 전략을 마스터해봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 고급 전략 개요</h2>
        <div className="mb-4">
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => setSelectedStrategy('volatility')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedStrategy === 'volatility'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              변동성 거래
            </button>
            <button
              onClick={() => setSelectedStrategy('dispersion')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedStrategy === 'dispersion'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              디스퍼전
            </button>
            <button
              onClick={() => setSelectedStrategy('exotic')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedStrategy === 'exotic'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              이색옵션
            </button>
            <button
              onClick={() => setSelectedStrategy('structured')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedStrategy === 'structured'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              구조화상품
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
          {selectedStrategy === 'volatility' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
                변동성 거래 (Volatility Trading)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">변동성 차익거래</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

class VolatilityArbitrage:
    def __init__(self):
        self.risk_free_rate = 0.03
        
    def black_scholes_iv(self, option_price, S, K, T, r, option_type='call'):
        """Black-Scholes 내재변동성 계산"""
        def bs_price(sigma):
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        def objective(sigma):
            return abs(bs_price(sigma) - option_price)
        
        result = minimize_scalar(objective, bounds=(0.001, 5), method='bounded')
        return result.x
    
    def volatility_surface(self, options_data):
        """변동성 곡면 구축"""
        surface = {}
        
        for strike in options_data['strikes']:
            for maturity in options_data['maturities']:
                option = options_data[(strike, maturity)]
                iv = self.black_scholes_iv(
                    option['price'], 
                    option['spot'], 
                    strike, 
                    maturity,
                    self.risk_free_rate
                )
                surface[(strike, maturity)] = iv
        
        return surface
    
    def find_arbitrage_opportunities(self, surface):
        """변동성 곡면 차익거래 기회 탐색"""
        opportunities = []
        
        # 캘린더 스프레드 기회
        for strike in surface['strikes']:
            ivs = [surface[(strike, T)] for T in sorted(surface['maturities'])]
            
            # 역전된 기간 구조 찾기
            for i in range(len(ivs)-1):
                if ivs[i] > ivs[i+1] * 1.05:  # 5% 이상 차이
                    opportunities.append({
                        'type': 'calendar_arb',
                        'strike': strike,
                        'near_iv': ivs[i],
                        'far_iv': ivs[i+1],
                        'profit_potential': (ivs[i] - ivs[i+1]) * 100
                    })
        
        return opportunities`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">변동성 스왑 (Variance Swap)</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    실현변동성과 내재변동성의 차이를 거래하는 순수 변동성 상품
                  </p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">구조</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 변동성 직접 거래</li>
                        <li>• 델타 중립 유지</li>
                        <li>• 명목원금 × (RV² - Strike²)</li>
                        <li>• 일일 리밸런싱</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">복제 전략</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 옵션 포트폴리오</li>
                        <li>• 로그 계약 근사</li>
                        <li>• 동적 헤징</li>
                        <li>• 감마 스캘핑</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">변동성 기간구조 거래</h4>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                    <strong>전략 예시: VIX 기간구조 거래</strong>
                    <ul className="mt-2 space-y-1">
                      <li>• 정상시장: 근월물 &lt; 원월물 (콘탱고)</li>
                      <li>• 스트레스: 근월물 &gt; 원월물 (백워데이션)</li>
                      <li>• 롤 수익률 활용</li>
                      <li>• VIX ETN 페어 트레이딩</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'dispersion' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
                디스퍼전 트레이딩 (Dispersion Trading)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">전략 개요</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    지수 내재변동성과 구성종목 내재변동성 간의 차이를 활용
                  </p>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`class DispersionTrading:
    def __init__(self, index_components):
        self.index = index_components
        self.weights = self.index['weights']
        
    def calculate_implied_correlation(self, index_iv, component_ivs):
        """내재 상관계수 계산"""
        # 지수 변동성 = sqrt(Σ wi*wj*σi*σj*ρij)
        weighted_var_sum = sum(
            self.weights[i]**2 * component_ivs[i]**2 
            for i in range(len(component_ivs))
        )
        
        # 평균 내재 상관계수 역산
        implied_corr = (index_iv**2 - weighted_var_sum) / (
            sum(self.weights[i] * self.weights[j] * 
                component_ivs[i] * component_ivs[j]
                for i in range(len(component_ivs))
                for j in range(len(component_ivs)) if i != j)
        )
        
        return implied_corr
    
    def setup_dispersion_trade(self, market_data):
        """디스퍼전 거래 설정"""
        index_iv = market_data['index_iv']
        component_ivs = market_data['component_ivs']
        
        implied_corr = self.calculate_implied_correlation(
            index_iv, component_ivs
        )
        
        # 내재 상관계수가 실현 상관계수보다 높을 때
        if implied_corr > market_data['realized_corr'] + 0.1:
            trade = {
                'action': 'sell_dispersion',
                'trades': [
                    {'type': 'sell', 'instrument': 'index_straddle'},
                    {'type': 'buy', 'instrument': 'component_straddles',
                     'weights': self.weights}
                ],
                'expected_profit': (implied_corr - market_data['realized_corr']) * 100
            }
        else:
            trade = None
            
        return trade`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">실전 구현</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">Long Dispersion</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 지수 스트래들 매도</li>
                        <li>• 개별종목 스트래들 매수</li>
                        <li>• 상관계수 하락 베팅</li>
                        <li>• 시장 스트레스시 수익</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">Short Dispersion</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 지수 스트래들 매수</li>
                        <li>• 개별종목 스트래들 매도</li>
                        <li>• 상관계수 상승 베팅</li>
                        <li>• 안정적 시장에서 수익</li>
                      </ul>
                    </div>
                  </div>
                  
                  <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded p-3">
                    <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
                      주의사항
                    </p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 mt-1 space-y-1">
                      <li>• 대량의 자본 필요</li>
                      <li>• 정교한 리스크 관리</li>
                      <li>• 거래비용 고려 필수</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'exotic' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
                이색옵션 (Exotic Options)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">배리어 옵션 (Barrier Options)</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">Knock-In 옵션</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 배리어 터치시 활성화</li>
                        <li>• Up-and-In / Down-and-In</li>
                        <li>• 일반 옵션보다 저렴</li>
                        <li>• 조건부 헤지 전략</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">Knock-Out 옵션</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 배리어 터치시 소멸</li>
                        <li>• Up-and-Out / Down-and-Out</li>
                        <li>• 제한적 보호 제공</li>
                        <li>• 비용 효율적 헤지</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">아시안 옵션 (Asian Options)</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    기간 평균 가격을 기준으로 하는 옵션
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                    <pre className="text-sm">
{`# 아시안 옵션 가격 계산 (Monte Carlo)
def asian_option_price(S0, K, T, r, sigma, n_simulations=10000):
    dt = T / 252  # 일일 평균
    n_steps = int(T / dt)
    
    payoffs = []
    for _ in range(n_simulations):
        path = [S0]
        for _ in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            S_new = path[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW)
            path.append(S_new)
        
        # 산술 평균
        avg_price = np.mean(path)
        payoff = max(avg_price - K, 0)
        payoffs.append(payoff)
    
    option_price = np.exp(-r*T) * np.mean(payoffs)
    return option_price`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">디지털 옵션 (Digital/Binary Options)</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">Cash-or-Nothing</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 고정 금액 지급</li>
                        <li>• All or Nothing</li>
                        <li>• 간단한 구조</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">Asset-or-Nothing</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 기초자산 지급</li>
                        <li>• 조건 충족시 현물</li>
                        <li>• 델타 헤지 복잡</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">기타 이색옵션</h4>
                  <div className="space-y-2 text-sm">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <strong>Lookback Options</strong>
                      <p>기간 중 최고/최저가 기준 행사</p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <strong>Chooser Options</strong>
                      <p>특정 시점에 콜/풋 선택 가능</p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <strong>Compound Options</strong>
                      <p>옵션의 옵션 (옵션을 살 권리)</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'structured' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
                구조화 상품 (Structured Products)
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">ELS (주가연계증권)</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`class ELS_StepDown:
    """스텝다운 ELS 구조"""
    def __init__(self, underlying, barriers, coupons, maturity):
        self.underlying = underlying
        self.barriers = barriers  # [95%, 90%, 85%, 80%, 75%, 70%]
        self.coupons = coupons    # [4.5%, 9%, 13.5%, 18%, 22.5%, 27%]
        self.maturity = maturity  # 3년
        
    def evaluate_payoff(self, price_paths):
        """조기상환 및 만기 평가"""
        initial_price = price_paths[0]
        
        # 6개월마다 조기상환 평가
        for i, eval_date in enumerate(range(6, 37, 6)):
            current_price = price_paths[eval_date * 21]  # 월 21영업일
            
            if current_price >= initial_price * self.barriers[i]:
                # 조기상환
                return {
                    'type': 'early_redemption',
                    'date': eval_date,
                    'payoff': 1 + self.coupons[i],
                    'annualized_return': self.coupons[i] * 12 / eval_date
                }
        
        # 만기 평가
        final_price = price_paths[-1]
        if final_price >= initial_price * 0.65:  # 낙인 배리어
            return {
                'type': 'maturity_redemption',
                'payoff': 1 + self.coupons[-1]
            }
        else:
            # 원금 손실
            return {
                'type': 'knock_in',
                'payoff': final_price / initial_price,
                'loss': 1 - (final_price / initial_price)
            }`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">Autocallable 구조</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium mb-2">구조 특징</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 주기적 조기상환 평가</li>
                        <li>• 스텝다운 배리어</li>
                        <li>• 누적 쿠폰 지급</li>
                        <li>• 낙인 배리어 존재</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium mb-2">리스크 요소</h5>
                      <ul className="text-sm space-y-1">
                        <li>• 원금 손실 가능</li>
                        <li>• 유동성 제약</li>
                        <li>• 발행사 신용위험</li>
                        <li>• 복잡한 구조</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">DLS (파생결합증권)</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    금리, 환율, 상품 등 다양한 기초자산 연계
                  </p>
                  <div className="space-y-2">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                      <strong>Range Accrual</strong>
                      <p>특정 범위 내 체류 일수에 비례한 이자 지급</p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                      <strong>Target Redemption</strong>
                      <p>누적 쿠폰이 목표치 도달시 조기상환</p>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                      <strong>Snowball</strong>
                      <p>이전 쿠폰 미지급시 다음 회차 누적</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 옵션 그릭스 고급 활용</h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
            2차 그릭스와 포트폴리오 관리
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Vanna (δ²V/δSδσ)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                변동성 변화에 따른 델타 변화
              </p>
              <ul className="text-sm space-y-1">
                <li>• 변동성-방향성 상관관계</li>
                <li>• 스큐 트레이딩 핵심</li>
                <li>• OTM 옵션에서 중요</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Volga (δ²V/δσ²)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                베가의 변동성 민감도
              </p>
              <ul className="text-sm space-y-1">
                <li>• 변동성의 변동성</li>
                <li>• 볼가 중립 포트폴리오</li>
                <li>• 변동성 헤지 정교화</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <h4 className="text-green-400 font-mono text-sm mb-2"># 포트폴리오 그릭스 관리</h4>
            <pre className="text-sm">
{`class GreeksManager:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        
    def calculate_portfolio_greeks(self):
        """포트폴리오 전체 그릭스 계산"""
        total_greeks = {
            'delta': 0, 'gamma': 0, 'theta': 0, 
            'vega': 0, 'vanna': 0, 'volga': 0
        }
        
        for position in self.portfolio:
            for greek, value in position.greeks.items():
                total_greeks[greek] += value * position.quantity
        
        return total_greeks
    
    def hedge_recommendations(self, target_greeks):
        """목표 그릭스 달성을 위한 헤지 추천"""
        current = self.calculate_portfolio_greeks()
        
        hedges = []
        
        # 델타 헤지
        if abs(current['delta'] - target_greeks['delta']) > 0.1:
            hedges.append({
                'type': 'delta_hedge',
                'instrument': 'futures',
                'quantity': -(current['delta'] - target_greeks['delta'])
            })
        
        # 감마 헤지
        if abs(current['gamma']) > target_greeks['gamma_limit']:
            # ATM 옵션으로 감마 조정
            hedges.append({
                'type': 'gamma_hedge',
                'instrument': 'atm_options',
                'quantity': -current['gamma'] / 0.05  # ATM 감마 추정
            })
        
        return hedges`}</pre>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔬 실전 시나리오 분석</h2>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">고급 전략 실전 적용</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">시나리오 1: 변동성 급등 예상</h4>
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3 mb-3">
                <p className="text-sm">
                  <strong>상황:</strong> FOMC 회의 전, VIX 15 → 예상 25+
                </p>
              </div>
              
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <strong>전략 구성:</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• VIX 콜 매수</li>
                    <li>• 비율 콜 스프레드</li>
                    <li>• 캘린더 스프레드 청산</li>
                  </ul>
                </div>
                <div>
                  <strong>리스크 관리:</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 포지션 크기 제한</li>
                    <li>• 시간가치 손실 대비</li>
                    <li>• 이벤트 후 청산 계획</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">시나리오 2: 섹터 로테이션</h4>
              <div className="bg-green-50 dark:bg-green-900/20 rounded p-3 mb-3">
                <p className="text-sm">
                  <strong>상황:</strong> 기술주 → 가치주 자금 이동 예상
                </p>
              </div>
              
              <div className="text-sm">
                <strong>디스퍼전 전략:</strong>
                <ol className="mt-2 space-y-1">
                  <li>1. KOSPI200 스트래들 매도 (IV: 18%)</li>
                  <li>2. 기술주 스트래들 매수 (평균 IV: 25%)</li>
                  <li>3. 금융주 스트래들 매도 (평균 IV: 15%)</li>
                  <li>4. 상관계수 변화 모니터링</li>
                </ol>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 고급 전략 리스크</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            주의해야 할 위험 요소
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">모델 리스크</h4>
              <ul className="text-sm space-y-1">
                <li>• 가정 위반 (정규분포 등)</li>
                <li>• 파라미터 추정 오류</li>
                <li>• 극단적 사건 과소평가</li>
                <li>• 과거 데이터 의존성</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">실행 리스크</h4>
              <ul className="text-sm space-y-1">
                <li>• 슬리피지 &amp; 거래비용</li>
                <li>• 유동성 부족</li>
                <li>• 시스템 장애</li>
                <li>• 규제 변경</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
              💡 리스크 관리 핵심
            </h4>
            <ol className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>1. 복잡성과 수익의 균형 고려</li>
              <li>2. 스트레스 테스트 정기 실시</li>
              <li>3. 포지션 한도 엄격 준수</li>
              <li>4. 비상 청산 계획 수립</li>
              <li>5. 지속적인 모니터링 체계</li>
            </ol>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📚 고급 전략 마스터 체크리스트</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">전문 트레이더로의 성장 경로</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>Black-Scholes 및 확률미적분 이해</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>그릭스 완벽 이해 및 활용 능력</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>변동성 곡면 분석 능력</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>프로그래밍을 통한 자동화 구현</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>리스크 관리 시스템 구축</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>실전 경험 3년 이상</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>지속적인 연구와 개선</span>
            </label>
          </div>
        </div>
      </section>
    </div>
  );
}