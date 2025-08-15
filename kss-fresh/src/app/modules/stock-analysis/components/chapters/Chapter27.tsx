'use client';

export default function Chapter27() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">팩터 모델 구축</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          시장 수익률을 설명하는 체계적인 요인들을 발굴하고, 이를 활용한 투자 전략을 구축하는 방법을 배워봅시다.
          학계에서 검증된 팩터부터 새로운 팩터 발굴까지 실전 팩터 투자의 모든 것을 다룹니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 팩터 투자의 이해</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
            팩터(Factor)란 무엇인가?
          </h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              팩터는 주식 수익률의 횡단면적 차이를 설명하는 공통 요인입니다.
              체계적이고 지속적인 초과 수익의 원천이며, 리스크 프리미엄으로 해석됩니다.
            </p>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">전통적 관점</h4>
                <ul className="text-sm space-y-1">
                  <li>• CAPM: 시장 베타만 고려</li>
                  <li>• 단일 팩터 모델</li>
                  <li>• 리스크 = 변동성</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">현대적 관점</h4>
                <ul className="text-sm space-y-1">
                  <li>• 다중 팩터 모델</li>
                  <li>• 행동재무학적 해석</li>
                  <li>• 리스크 + 비효율성</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
              💡 핵심 원리
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              팩터 투자는 "왜 특정 주식이 다른 주식보다 높은 수익률을 보이는가?"라는 
              질문에 대한 체계적인 답을 찾는 과정입니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 주요 팩터 모델</h2>
        <div className="space-y-4">
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">
              1. Fama-French 3팩터 모델
            </h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-4">
              <pre className="text-sm">
{`# Fama-French 3팩터 모델 구현
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

class FamaFrench3Factor:
    def __init__(self):
        self.factors = ['MKT-RF', 'SMB', 'HML']
        
    def construct_factors(self, stock_data, market_data):
        """팩터 구성"""
        # 시장 팩터 (Market - Risk Free)
        market_factor = market_data['return'] - market_data['rf_rate']
        
        # SMB (Small Minus Big) - 규모 팩터
        # 시가총액 기준 상하위 30% 분류
        size_sorted = stock_data.sort_values('market_cap')
        small_cap = size_sorted.iloc[:int(len(size_sorted)*0.3)]
        large_cap = size_sorted.iloc[-int(len(size_sorted)*0.3):]
        
        smb = small_cap['return'].mean() - large_cap['return'].mean()
        
        # HML (High Minus Low) - 가치 팩터
        # B/M ratio 기준 상하위 30% 분류
        value_sorted = stock_data.sort_values('book_to_market')
        value_stocks = value_sorted.iloc[-int(len(value_sorted)*0.3):]
        growth_stocks = value_sorted.iloc[:int(len(value_sorted)*0.3)]
        
        hml = value_stocks['return'].mean() - growth_stocks['return'].mean()
        
        return {
            'MKT-RF': market_factor,
            'SMB': smb,
            'HML': hml
        }
    
    def estimate_model(self, portfolio_returns, factor_returns):
        """팩터 모델 추정"""
        # 초과 수익률 계산
        Y = portfolio_returns - factor_returns['RF']
        X = factor_returns[self.factors]
        
        # OLS 회귀분석
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        
        # 결과 해석
        results = {
            'alpha': model.params[0],
            'beta_market': model.params[1],
            'beta_smb': model.params[2],
            'beta_hml': model.params[3],
            'r_squared': model.rsquared,
            'p_values': model.pvalues
        }
        
        return results, model`}</pre>
            </div>
            
            <div className="grid md:grid-cols-3 gap-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-1">시장 팩터 (MKT)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  시장 전체의 초과 수익률
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-1">규모 팩터 (SMB)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  소형주 - 대형주 수익률 차이
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-sm mb-1">가치 팩터 (HML)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  가치주 - 성장주 수익률 차이
                </p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">
              2. Carhart 4팩터 모델
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Fama-French 3팩터에 모멘텀 팩터를 추가한 모델
            </p>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`# 모멘텀 팩터 구성
def construct_momentum_factor(stock_data, formation_period=12, skip_month=1):
    """
    UMD (Up Minus Down) - 모멘텀 팩터
    과거 12개월 수익률 기준 (최근 1개월 제외)
    """
    # 과거 수익률 계산
    returns = stock_data.pivot(
        index='date', 
        columns='ticker', 
        values='return'
    )
    
    # Formation period 수익률 (12개월, 최근 1개월 제외)
    formation_returns = returns.shift(skip_month).rolling(
        window=formation_period
    ).apply(lambda x: (1 + x).prod() - 1)
    
    # 상위 30% (Winners) vs 하위 30% (Losers)
    for date in formation_returns.index:
        daily_returns = formation_returns.loc[date].dropna()
        
        n_stocks = len(daily_returns)
        n_portfolio = int(n_stocks * 0.3)
        
        winners = daily_returns.nlargest(n_portfolio).index
        losers = daily_returns.nsmallest(n_portfolio).index
        
        # 다음 달 수익률
        next_month = date + pd.DateOffset(months=1)
        if next_month in returns.index:
            winner_ret = returns.loc[next_month, winners].mean()
            loser_ret = returns.loc[next_month, losers].mean()
            
            umd = winner_ret - loser_ret
            
    return umd`}</pre>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-3">
              3. Fama-French 5팩터 모델
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              수익성(RMW)과 투자(CMA) 팩터를 추가한 확장 모델
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">RMW (Robust Minus Weak)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  수익성이 높은 기업 - 낮은 기업
                </p>
                <ul className="text-sm space-y-1">
                  <li>• 영업이익률 (Operating Profitability)</li>
                  <li>• ROE, ROA 등 수익성 지표</li>
                  <li>• 지속 가능한 이익 창출 능력</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">CMA (Conservative Minus Aggressive)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  보수적 투자 기업 - 공격적 투자 기업
                </p>
                <ul className="text-sm space-y-1">
                  <li>• 자산 성장률</li>
                  <li>• CAPEX 비율</li>
                  <li>• 투자 보수성</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔬 새로운 팩터 발굴</h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
            팩터 발굴 프로세스
          </h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">1. 가설 수립</h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <strong>행동재무학적 접근</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 투자자 심리 편향</li>
                    <li>• 정보 처리 비효율성</li>
                    <li>• 시장 이상 현상</li>
                  </ul>
                </div>
                <div>
                  <strong>경제학적 접근</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 리스크 프리미엄</li>
                    <li>• 시장 마찰</li>
                    <li>• 구조적 요인</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">2. 팩터 구성 및 검증</h4>
              <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm">
{`class FactorResearch:
    def __init__(self, universe, start_date, end_date):
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        
    def create_quality_factor(self, fundamental_data):
        """품질 팩터 구성 예시"""
        # 품질 점수 계산
        quality_score = pd.DataFrame()
        
        # 1. 수익성 (Profitability)
        quality_score['roe'] = fundamental_data['net_income'] / \
                               fundamental_data['equity']
        quality_score['roa'] = fundamental_data['net_income'] / \
                               fundamental_data['total_assets']
        quality_score['gross_profit'] = fundamental_data['gross_profit'] / \
                                       fundamental_data['total_assets']
        
        # 2. 성장성 (Growth)
        quality_score['earnings_growth'] = fundamental_data['net_income'].pct_change(4)
        quality_score['revenue_growth'] = fundamental_data['revenue'].pct_change(4)
        
        # 3. 안전성 (Safety)
        quality_score['debt_to_equity'] = fundamental_data['total_debt'] / \
                                         fundamental_data['equity']
        quality_score['current_ratio'] = fundamental_data['current_assets'] / \
                                        fundamental_data['current_liabilities']
        
        # 4. 이익 품질 (Earnings Quality)
        quality_score['accruals'] = (fundamental_data['net_income'] - 
                                    fundamental_data['operating_cash_flow']) / \
                                   fundamental_data['total_assets']
        
        # 종합 점수 (Z-score 정규화 후 가중 평균)
        for col in quality_score.columns:
            quality_score[col] = (quality_score[col] - quality_score[col].mean()) / \
                                quality_score[col].std()
        
        weights = {
            'roe': 0.2, 'roa': 0.15, 'gross_profit': 0.15,
            'earnings_growth': 0.1, 'revenue_growth': 0.1,
            'debt_to_equity': -0.15, 'current_ratio': 0.1,
            'accruals': -0.05
        }
        
        quality_score['total'] = sum(quality_score[col] * weight 
                                    for col, weight in weights.items())
        
        return quality_score
    
    def backtest_factor(self, factor_scores, returns, n_portfolios=5):
        """팩터 백테스트"""
        results = []
        
        for date in factor_scores.index:
            # 팩터 점수로 포트폴리오 구성
            daily_scores = factor_scores.loc[date].dropna()
            daily_returns = returns.loc[date]
            
            # 5분위 포트폴리오
            labels = [f'Q{i}' for i in range(1, n_portfolios+1)]
            portfolios = pd.qcut(daily_scores, n_portfolios, labels=labels)
            
            # 각 포트폴리오 수익률
            portfolio_returns = {}
            for label in labels:
                stocks = portfolios[portfolios == label].index
                portfolio_returns[label] = daily_returns[stocks].mean()
            
            # Long-Short 포트폴리오
            portfolio_returns['Long-Short'] = (
                portfolio_returns[f'Q{n_portfolios}'] - portfolio_returns['Q1']
            )
            
            results.append(portfolio_returns)
        
        return pd.DataFrame(results)`}</pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">3. 통계적 검증</h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <strong>수익성 검증</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 평균 수익률 & 샤프 비율</li>
                    <li>• 정보 비율 (IR)</li>
                    <li>• 최대 낙폭 (MDD)</li>
                    <li>• 승률 & 손익비</li>
                  </ul>
                </div>
                <div>
                  <strong>강건성 검증</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 다양한 기간 테스트</li>
                    <li>• 다양한 시장 테스트</li>
                    <li>• 거래비용 고려</li>
                    <li>• 다중검정 보정</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💼 멀티팩터 포트폴리오</h2>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">팩터 결합 전략</h3>
          
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-4">
            <pre className="text-sm">
{`class MultiFactorPortfolio:
    def __init__(self, factors):
        self.factors = factors
        
    def factor_timing(self, factor_returns, macro_indicators):
        """팩터 타이밍 모델"""
        # 각 팩터의 예상 성과 예측
        factor_predictions = {}
        
        for factor in self.factors:
            # 팩터별 예측 모델 (예: 경제 지표 활용)
            X = macro_indicators[['term_spread', 'credit_spread', 
                                 'vix', 'inflation', 'gdp_growth']]
            y = factor_returns[factor].shift(-1)  # 다음 기 수익률
            
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X[:-1], y[:-1])
            
            factor_predictions[factor] = model.predict(X.iloc[[-1]])[0]
        
        return factor_predictions
    
    def optimize_factor_weights(self, factor_returns, constraints=None):
        """팩터 가중치 최적화"""
        # 평균-분산 최적화
        mean_returns = factor_returns.mean()
        cov_matrix = factor_returns.cov()
        
        n_factors = len(self.factors)
        
        # 목적함수: 샤프비율 최대화
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_std
        
        # 제약조건
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 가중치 합 = 1
        ]
        
        # 경계조건 (각 팩터 0-40%)
        bounds = tuple((0, 0.4) for _ in range(n_factors))
        
        # 최적화
        init_weights = np.array([1/n_factors] * n_factors)
        result = optimize.minimize(
            negative_sharpe, init_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        return result.x
    
    def construct_portfolio(self, stock_scores, factor_weights):
        """최종 포트폴리오 구성"""
        # 종합 점수 계산
        composite_score = pd.DataFrame()
        
        for factor, weight in zip(self.factors, factor_weights):
            # 각 팩터 점수를 0-1로 정규화
            normalized_score = stock_scores[factor].rank(pct=True)
            composite_score[factor] = normalized_score * weight
        
        # 최종 점수
        composite_score['total'] = composite_score.sum(axis=1)
        
        # 상위 N개 종목 선택
        n_stocks = 50
        selected_stocks = composite_score.nlargest(n_stocks, 'total').index
        
        # 종목별 가중치 (점수 비례)
        stock_weights = composite_score.loc[selected_stocks, 'total']
        stock_weights = stock_weights / stock_weights.sum()
        
        return selected_stocks, stock_weights`}</pre>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">동일 가중</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                모든 팩터에 동일한 가중치 부여. 
                단순하지만 효과적인 접근법.
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">리스크 패리티</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                각 팩터의 리스크 기여도를 동일하게 조정. 
                변동성 역가중.
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">동적 할당</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                시장 상황에 따라 팩터 가중치 조정. 
                팩터 타이밍 전략.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 실전 구현 예시</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">한국 시장 팩터 모델 구축</h3>
          
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm">
{`# 한국 시장 특화 팩터 모델
class KoreaFactorModel:
    def __init__(self):
        self.factors = {
            'value': ['PBR', 'PER', 'PSR', 'EV/EBITDA'],
            'quality': ['ROE', 'ROA', '영업이익률', '부채비율'],
            'momentum': ['1M', '3M', '6M', '12M'],
            'low_vol': ['일간변동성', '베타', '하방변동성'],
            'growth': ['매출성장률', '이익성장률', 'EPS성장률']
        }
    
    def korea_specific_factors(self, data):
        """한국 시장 특화 팩터"""
        # 1. 대주주 지분율 팩터
        data['owner_factor'] = data['대주주지분율'] / 100
        
        # 2. 외국인 선호도 팩터
        data['foreign_factor'] = data['외국인보유비율'] / \
                                data['유동주식비율']
        
        # 3. 수급 팩터
        data['supply_demand'] = (data['매수잔량'] - data['매도잔량']) / \
                               (data['매수잔량'] + data['매도잔량'])
        
        # 4. 테마 팩터 (산업 모멘텀)
        industry_momentum = data.groupby('업종')['수익률'].transform('mean')
        data['theme_factor'] = industry_momentum
        
        return data
    
    def sector_neutral_portfolio(self, scores, sector_map, n_stocks_per_sector=5):
        """섹터 중립 포트폴리오"""
        selected_stocks = []
        
        for sector in sector_map['섹터'].unique():
            # 섹터 내 종목
            sector_stocks = sector_map[sector_map['섹터'] == sector]['종목코드']
            sector_scores = scores[scores.index.isin(sector_stocks)]
            
            # 섹터 내 상위 종목 선택
            top_stocks = sector_scores.nlargest(n_stocks_per_sector)
            selected_stocks.extend(top_stocks.index.tolist())
        
        return selected_stocks`}</pre>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 팩터 투자의 함정</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            주의해야 할 위험 요소
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">팩터 크라우딩</h4>
              <ul className="text-sm space-y-1">
                <li>• 인기 팩터에 자금 집중</li>
                <li>• 밸류에이션 상승</li>
                <li>• 급격한 청산 위험</li>
                <li>• 수익률 감소</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">팩터 디케이</h4>
              <ul className="text-sm space-y-1">
                <li>• 시간 경과에 따른 효과 감소</li>
                <li>• 시장 구조 변화</li>
                <li>• 규제 변경 영향</li>
                <li>• 기술 발전의 영향</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
              💡 해결 방안
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 mt-2 space-y-1">
              <li>• 다양한 팩터 조합으로 분산</li>
              <li>• 정기적인 팩터 유효성 검증</li>
              <li>• 거래비용 현실적 반영</li>
              <li>• 팩터 타이밍 전략 병행</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 실전 체크리스트</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">팩터 모델 구축 전 확인사항</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>경제적 근거가 명확한 팩터인가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>충분한 기간(10년 이상) 백테스트를 수행했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>다양한 시장 환경에서 검증했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>거래비용과 시장 충격을 고려했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>다른 팩터와의 상관관계를 분석했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>실시간 구현 가능한 데이터인가?</span>
            </label>
          </div>
        </div>
      </section>
    </div>
  );
}