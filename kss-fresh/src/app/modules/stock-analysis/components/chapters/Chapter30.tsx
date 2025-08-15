'use client';

import { useState } from 'react';

export default function Chapter30() {
  const [selectedHedge, setSelectedHedge] = useState('portfolio');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">헤지 전략과 리스크 관리</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          포트폴리오 보호와 리스크 관리를 위한 체계적인 헤지 전략을 학습합니다.
          시장 리스크, 개별 종목 리스크, 통화 리스크 등 다양한 위험을 관리하는 실전 기법을 마스터해봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🛡️ 헤지의 기본 개념</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
            헤지(Hedge)란 무엇인가?
          </h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              헤지는 보유 자산의 가격 변동 위험을 줄이기 위해 반대 포지션을 취하는 리스크 관리 전략입니다.
              완벽한 헤지는 수익도 손실도 없지만, 실제로는 비용과 효과의 균형을 찾는 것이 중요합니다.
            </p>
            
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                <h4 className="font-semibold text-sm mb-2">헤지의 목적</h4>
                <ul className="text-xs space-y-1">
                  <li>• 손실 제한</li>
                  <li>• 변동성 감소</li>
                  <li>• 수익 안정화</li>
                  <li>• 규제 준수</li>
                </ul>
              </div>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                <h4 className="font-semibold text-sm mb-2">헤지 수단</h4>
                <ul className="text-xs space-y-1">
                  <li>• 선물/옵션</li>
                  <li>• 스왑</li>
                  <li>• 반대 포지션</li>
                  <li>• 분산 투자</li>
                </ul>
              </div>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                <h4 className="font-semibold text-sm mb-2">헤지 비용</h4>
                <ul className="text-xs space-y-1">
                  <li>• 프리미엄</li>
                  <li>• 기회비용</li>
                  <li>• 거래비용</li>
                  <li>• 베이시스 리스크</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
              💡 핵심 원칙
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              헤지는 보험과 같습니다. 완벽한 보호를 원한다면 비용이 높아지고,
              비용을 줄이려면 일부 리스크를 감수해야 합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 헤지 전략 유형</h2>
        <div className="mb-4">
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => setSelectedHedge('portfolio')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedHedge === 'portfolio'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              포트폴리오 헤지
            </button>
            <button
              onClick={() => setSelectedHedge('dynamic')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedHedge === 'dynamic'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              동적 헤지
            </button>
            <button
              onClick={() => setSelectedHedge('tail')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedHedge === 'tail'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              테일 리스크
            </button>
            <button
              onClick={() => setSelectedHedge('currency')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedHedge === 'currency'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              통화 헤지
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          {selectedHedge === 'portfolio' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
                포트폴리오 헤지 전략
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">베타 헤지</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`class BetaHedge:
    def __init__(self, portfolio, benchmark='KOSPI200'):
        self.portfolio = portfolio
        self.benchmark = benchmark
        
    def calculate_portfolio_beta(self, returns_data):
        """포트폴리오 베타 계산"""
        portfolio_returns = returns_data['portfolio']
        market_returns = returns_data[self.benchmark]
        
        # 개별 종목 베타 계산
        betas = {}
        for stock in self.portfolio.holdings:
            covariance = np.cov(
                returns_data[stock], 
                market_returns
            )[0, 1]
            variance = np.var(market_returns)
            betas[stock] = covariance / variance
        
        # 가중평균 베타
        portfolio_beta = sum(
            betas[stock] * weight 
            for stock, weight in self.portfolio.weights.items()
        )
        
        return portfolio_beta, betas
    
    def hedge_with_futures(self, portfolio_value, portfolio_beta):
        """선물을 이용한 베타 헤지"""
        # KOSPI200 선물 계약 규격
        futures_multiplier = 250000
        current_index = 300  # 현재 지수
        
        # 헤지 비율 계산
        hedge_ratio = portfolio_beta
        
        # 필요 계약수
        contracts_needed = -(portfolio_value * hedge_ratio) / \
                          (current_index * futures_multiplier)
        
        return {
            'hedge_ratio': hedge_ratio,
            'contracts': round(contracts_needed),
            'notional_value': abs(contracts_needed * current_index * futures_multiplier),
            'hedged_beta': portfolio_beta + contracts_needed * 1 / \
                          (portfolio_value / (current_index * futures_multiplier))
        }`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">섹터 중립 헤지</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    특정 섹터 익스포저를 중립화하여 시장 전체 리스크만 노출
                  </p>
                  <div className="grid md:grid-cols-2 gap-3">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">Long/Short 페어</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 동일 섹터 내 상대가치</li>
                        <li>• 시장 중립적 포지션</li>
                        <li>• 낮은 상관관계</li>
                      </ul>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">섹터 ETF 활용</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 섹터별 ETF 매도</li>
                        <li>• 비용 효율적</li>
                        <li>• 유동성 우수</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">최소분산 헤지</h4>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                    <p className="mb-2">헤지 비율 = ρ × (σ_포트폴리오 / σ_헤지수단)</p>
                    <ul className="space-y-1">
                      <li>• ρ: 상관계수</li>
                      <li>• σ: 표준편차</li>
                      <li>• 분산 최소화 목적</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedHedge === 'dynamic' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
                동적 헤지 전략
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">델타 헤지</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`class DeltaHedging:
    def __init__(self, option_position):
        self.option = option_position
        
    def calculate_delta_hedge(self, spot_price, volatility, time_to_expiry):
        """델타 중립 포트폴리오 구성"""
        # Black-Scholes 델타
        d1 = (np.log(spot_price / self.option.strike) + 
              (self.option.r + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        
        if self.option.type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # 헤지 수량
        hedge_shares = -delta * self.option.contracts * 100
        
        return {
            'delta': delta,
            'hedge_shares': round(hedge_shares),
            'hedge_value': hedge_shares * spot_price
        }
    
    def rebalance_hedge(self, price_path, rebalance_frequency='daily'):
        """동적 리밸런싱"""
        hedge_history = []
        total_cost = 0
        
        for i, (date, price) in enumerate(price_path.items()):
            if i % rebalance_frequency == 0:
                # 새로운 델타 계산
                time_remaining = (self.option.expiry - date).days / 365
                new_hedge = self.calculate_delta_hedge(
                    price, 
                    self.option.implied_vol, 
                    time_remaining
                )
                
                # 리밸런싱 비용
                if hedge_history:
                    shares_traded = new_hedge['hedge_shares'] - \
                                   hedge_history[-1]['hedge_shares']
                    cost = abs(shares_traded) * price * 0.001  # 거래비용
                    total_cost += cost
                
                hedge_history.append({
                    'date': date,
                    'price': price,
                    'delta': new_hedge['delta'],
                    'shares': new_hedge['hedge_shares'],
                    'cost': cost if hedge_history else 0
                })
        
        return hedge_history, total_cost`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">감마 스캘핑</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    감마를 이용한 변동성 수익 창출 전략
                  </p>
                  <div className="grid md:grid-cols-2 gap-3">
                    <div>
                      <h5 className="font-medium text-sm mb-2">Long 감마 전략</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 옵션 매수 포지션</li>
                        <li>• 가격 변동시 수익</li>
                        <li>• 세타 비용 발생</li>
                        <li>• 높은 변동성 유리</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium text-sm mb-2">실행 방법</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 델타 중립 유지</li>
                        <li>• 일정 범위 도달시 리밸런싱</li>
                        <li>• 수익 실현 후 재설정</li>
                        <li>• 거래비용 고려</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">CPPI 전략</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    Constant Proportion Portfolio Insurance
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                    <p className="mb-2">위험자산 투자금액 = m × (포트폴리오 가치 - 플로어)</p>
                    <ul className="space-y-1">
                      <li>• m: 승수 (보통 3-5)</li>
                      <li>• 플로어: 보장 원금</li>
                      <li>• 상승시 비중 확대</li>
                      <li>• 하락시 비중 축소</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedHedge === 'tail' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
                테일 리스크 헤지
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">블랙 스완 대비</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    극단적 시장 하락에 대비한 보호 전략
                  </p>
                  
                  <div className="space-y-3">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">OTM 풋옵션 매수</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 행사가: 현재가 대비 10-20% 하락</li>
                        <li>• 만기: 3-6개월 롤링</li>
                        <li>• 비용: 포트폴리오의 1-2%/년</li>
                        <li>• 급락시 큰 수익</li>
                      </ul>
                    </div>
                    
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">VIX 콜옵션</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 변동성 급등시 수익</li>
                        <li>• 음의 상관관계 활용</li>
                        <li>• 시간가치 소멸 주의</li>
                        <li>• 적정 비중 유지</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">리스크 패리티 접근</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`def tail_risk_allocation(portfolio_returns, confidence_level=0.95):
    """테일 리스크 기반 자산 배분"""
    # CVaR (Conditional Value at Risk) 계산
    def calculate_cvar(returns, alpha=0.05):
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean()
        return cvar
    
    # 각 자산의 테일 리스크 기여도
    asset_cvars = {}
    for asset in portfolio_returns.columns:
        asset_cvars[asset] = calculate_cvar(portfolio_returns[asset])
    
    # 리스크 패리티 가중치
    total_risk = sum(1/abs(cvar) for cvar in asset_cvars.values())
    weights = {
        asset: (1/abs(cvar)) / total_risk 
        for asset, cvar in asset_cvars.items()
    }
    
    # 테일 헤지 오버레이
    hedge_budget = 0.02  # 2% of portfolio
    hedge_allocation = {
        'otm_puts': hedge_budget * 0.6,
        'vix_calls': hedge_budget * 0.3,
        'gold': hedge_budget * 0.1
    }
    
    return weights, hedge_allocation`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">시나리오 기반 헤지</h4>
                  <div className="grid md:grid-cols-2 gap-3 text-sm">
                    <div className="bg-red-50 dark:bg-red-900/20 rounded p-3">
                      <h5 className="font-medium mb-2">금융위기 시나리오</h5>
                      <ul className="space-y-1">
                        <li>• 주식 -40%</li>
                        <li>• 신용 스프레드 확대</li>
                        <li>• 달러 강세</li>
                        <li>→ 국채, 달러, 금 보유</li>
                      </ul>
                    </div>
                    <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-3">
                      <h5 className="font-medium mb-2">인플레이션 시나리오</h5>
                      <ul className="space-y-1">
                        <li>• 금리 급등</li>
                        <li>• 실물자산 상승</li>
                        <li>• 통화 약세</li>
                        <li>→ 원자재, TIPS, 부동산</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedHedge === 'currency' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
                통화 헤지 전략
              </h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">해외투자 환헤지</h4>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
{`class CurrencyHedge:
    def __init__(self, base_currency='KRW'):
        self.base = base_currency
        
    def forward_hedge(self, exposure, forward_rate, spot_rate):
        """선물환을 이용한 헤지"""
        # 헤지 비용/수익 계산
        forward_points = forward_rate - spot_rate
        hedge_cost_pct = (forward_points / spot_rate) * 100
        
        # 연환산 헤지 비용
        annual_cost = hedge_cost_pct * 12  # 월물 기준
        
        return {
            'hedge_amount': exposure,
            'forward_rate': forward_rate,
            'hedge_cost_%': hedge_cost_pct,
            'annual_cost_%': annual_cost,
            'breakeven_move': forward_points
        }
    
    def option_hedge(self, exposure, spot, strike, premium, option_type='put'):
        """옵션을 이용한 헤지"""
        if option_type == 'put':
            # 하방 보호
            protected_level = strike
            max_loss = (spot - strike) + premium
            participation = "무제한 상승 참여"
        else:  # call for short exposure
            protected_level = strike
            max_loss = (strike - spot) + premium
            participation = "무제한 하락 참여"
        
        return {
            'protection_level': protected_level,
            'premium_cost': premium,
            'max_loss': max_loss,
            'participation': participation,
            'breakeven': spot + premium if option_type == 'put' else spot - premium
        }`}</pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">자연적 헤지</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    사업 구조를 통한 환위험 상쇄
                  </p>
                  <div className="grid md:grid-cols-2 gap-3">
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">수출입 매칭</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 동일 통화 수입/지출</li>
                        <li>• 현금흐름 매칭</li>
                        <li>• 헤지 비용 절감</li>
                      </ul>
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded p-3">
                      <h5 className="font-medium text-sm mb-2">차입 통화 매칭</h5>
                      <ul className="text-xs space-y-1">
                        <li>• 자산 통화로 차입</li>
                        <li>• 부채로 헤지 효과</li>
                        <li>• 금리 차이 고려</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-3">헤지 비율 결정</h4>
                  <div className="space-y-3">
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3 text-sm">
                      <h5 className="font-medium mb-2">최적 헤지 비율 고려사항</h5>
                      <ul className="space-y-1">
                        <li>• 환율 전망과 불확실성</li>
                        <li>• 헤지 비용 대비 효과</li>
                        <li>• 기업 재무 상황</li>
                        <li>• 경쟁사 헤지 정책</li>
                      </ul>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 text-center text-sm">
                      <div className="bg-green-100 dark:bg-green-900/20 rounded p-2">
                        <div className="font-semibold">0-30%</div>
                        <div className="text-xs">투기적</div>
                      </div>
                      <div className="bg-yellow-100 dark:bg-yellow-900/20 rounded p-2">
                        <div className="font-semibold">40-60%</div>
                        <div className="text-xs">균형적</div>
                      </div>
                      <div className="bg-red-100 dark:bg-red-900/20 rounded p-2">
                        <div className="font-semibold">70-100%</div>
                        <div className="text-xs">보수적</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 실전 헤지 구현</h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
            통합 리스크 관리 시스템
          </h3>
          
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-4">
            <pre className="text-sm">
{`class IntegratedRiskManagement:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.risk_limits = {
            'var_limit': 0.02,  # 2% daily VaR
            'max_drawdown': 0.15,  # 15% max drawdown
            'concentration': 0.1,  # 10% single position
            'leverage': 2.0  # 2x max leverage
        }
        
    def calculate_portfolio_risks(self):
        """포트폴리오 전체 리스크 측정"""
        risks = {}
        
        # Value at Risk (95% confidence)
        returns = self.portfolio.get_returns()
        risks['var_95'] = np.percentile(returns, 5)
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        risks['max_drawdown'] = drawdown.min()
        
        # Concentration Risk
        weights = self.portfolio.get_weights()
        risks['max_concentration'] = weights.max()
        
        # Greeks (if options)
        if self.portfolio.has_options():
            greeks = self.portfolio.calculate_greeks()
            risks['delta'] = greeks['delta']
            risks['gamma'] = greeks['gamma']
            risks['vega'] = greeks['vega']
        
        return risks
    
    def generate_hedge_recommendations(self, current_risks):
        """리스크 기반 헤지 추천"""
        recommendations = []
        
        # VaR 초과시
        if abs(current_risks['var_95']) > self.risk_limits['var_limit']:
            recommendations.append({
                'type': 'reduce_exposure',
                'action': 'VaR 한도 초과 - 포지션 축소 또는 보호적 풋 매수',
                'urgency': 'high'
            })
        
        # 집중도 초과시
        if current_risks['max_concentration'] > self.risk_limits['concentration']:
            recommendations.append({
                'type': 'diversify',
                'action': '단일 종목 집중도 초과 - 분산 투자 필요',
                'urgency': 'medium'
            })
        
        # 델타 과다 노출시
        if 'delta' in current_risks and abs(current_risks['delta']) > 0.7:
            recommendations.append({
                'type': 'delta_hedge',
                'action': f"델타 {current_risks['delta']:.2f} - 선물 헤지 고려",
                'contracts': -current_risks['delta'] * self.portfolio.value / 75000000
            })
        
        return recommendations`}</pre>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">헤지 실행 프로세스</h4>
              <ol className="text-sm space-y-2">
                <li className="flex items-start">
                  <span className="font-semibold mr-2">1.</span>
                  <div>
                    <strong>리스크 식별</strong>
                    <p className="text-xs text-gray-600 dark:text-gray-400">시장, 개별, 통화 리스크 측정</p>
                  </div>
                </li>
                <li className="flex items-start">
                  <span className="font-semibold mr-2">2.</span>
                  <div>
                    <strong>헤지 설계</strong>
                    <p className="text-xs text-gray-600 dark:text-gray-400">적절한 상품과 비율 결정</p>
                  </div>
                </li>
                <li className="flex items-start">
                  <span className="font-semibold mr-2">3.</span>
                  <div>
                    <strong>실행 및 모니터링</strong>
                    <p className="text-xs text-gray-600 dark:text-gray-400">거래 실행, 효과성 추적</p>
                  </div>
                </li>
                <li className="flex items-start">
                  <span className="font-semibold mr-2">4.</span>
                  <div>
                    <strong>조정 및 재평가</strong>
                    <p className="text-xs text-gray-600 dark:text-gray-400">시장 변화에 따른 조정</p>
                  </div>
                </li>
              </ol>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">헤지 효과성 평가</h4>
              <div className="space-y-3">
                <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                  <h5 className="font-medium mb-1">회귀분석</h5>
                  <p className="text-xs">헤지 대상과 헤지 수단의 상관관계</p>
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                  <h5 className="font-medium mb-1">Dollar Offset</h5>
                  <p className="text-xs">헤지 손익 / 헤지 대상 손익 비율</p>
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                  <h5 className="font-medium mb-1">변동성 감소율</h5>
                  <p className="text-xs">헤지 전후 포트폴리오 변동성 비교</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💰 헤지 비용 관리</h2>
        <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4">
            비용 효율적인 헤지 전략
          </h3>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">비용 구조</h4>
              <ul className="text-sm space-y-1">
                <li>💵 옵션 프리미엄</li>
                <li>📊 베이시스 리스크</li>
                <li>🔄 롤오버 비용</li>
                <li>💱 거래 수수료</li>
                <li>📉 기회비용</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">비용 절감 방법</h4>
              <ul className="text-sm space-y-1">
                <li>• 부분 헤지 활용</li>
                <li>• 제로코스트 칼라</li>
                <li>• 자연적 헤지 극대화</li>
                <li>• 동적 헤지 비율</li>
                <li>• 크로스 헤지 활용</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">ROI 분석</h4>
              <div className="text-sm space-y-2">
                <div className="flex justify-between">
                  <span>헤지 비용:</span>
                  <span className="font-semibold">-2%/년</span>
                </div>
                <div className="flex justify-between">
                  <span>위험 감소:</span>
                  <span className="font-semibold">-50%</span>
                </div>
                <div className="flex justify-between">
                  <span>샤프비율 개선:</span>
                  <span className="font-semibold text-green-600">+0.3</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 헤지의 함정</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            주의해야 할 위험 요소
          </h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">과도한 헤지</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                완벽한 헤지를 추구하면 수익 기회도 사라집니다. 
                헤지는 보험이지 수익 전략이 아닙니다.
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">베이시스 리스크</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                헤지 수단과 헤지 대상의 가격이 다르게 움직일 수 있습니다.
                완벽한 상관관계는 존재하지 않습니다.
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">모델 리스크</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                헤지 비율 계산 모델이 틀릴 수 있습니다.
                과거 데이터가 미래를 보장하지 않습니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📋 헤지 전략 체크리스트</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">효과적인 헤지를 위한 점검사항</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>헤지 목적과 목표가 명확한가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>리스크 측정이 정확한가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>헤지 비용이 감당 가능한가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>헤지 효과를 정기적으로 평가하는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>극단적 시나리오를 고려했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>문서화와 보고 체계가 있는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>조직 내 헤지 정책이 확립되어 있는가?</span>
            </label>
          </div>
        </div>
      </section>
    </div>
  );
}