'use client';

export default function Chapter25() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">퀀트 투자의 이해</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          데이터와 수학적 모델을 활용한 계량 투자의 세계로 들어가봅시다.
          퀀트 투자의 기본 개념부터 실제 구현까지 체계적으로 학습합니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🧮 퀀트 투자란?</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
            Quantitative Investment 개요
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">전통적 투자 vs 퀀트 투자</h4>
              <div className="space-y-2 text-sm">
                <div>
                  <strong className="text-gray-700 dark:text-gray-300">전통적 투자:</strong>
                  <ul className="pl-4 mt-1">
                    <li>• 정성적 분석 중심</li>
                    <li>• 경험과 직관 활용</li>
                    <li>• 개별 종목 심층 분석</li>
                  </ul>
                </div>
                <div className="mt-3">
                  <strong className="text-blue-700 dark:text-blue-300">퀀트 투자:</strong>
                  <ul className="pl-4 mt-1">
                    <li>• 정량적 데이터 분석</li>
                    <li>• 통계/수학 모델 활용</li>
                    <li>• 대량 종목 동시 분석</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">퀀트 투자의 장점</h4>
              <ul className="text-sm space-y-1">
                <li>✅ 감정 배제 - 객관적 의사결정</li>
                <li>✅ 백테스팅 - 전략 사전 검증</li>
                <li>✅ 일관성 - 규칙 기반 투자</li>
                <li>✅ 확장성 - 다수 종목 동시 관리</li>
                <li>✅ 리스크 관리 - 체계적 위험 통제</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
              💡 핵심 포인트
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              퀀트 투자는 기존 투자를 대체하는 것이 아니라 보완하는 도구입니다.
              데이터와 직관을 적절히 조합하는 것이 중요합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 퀀트 투자의 핵심 요소</h2>
        <div className="space-y-4">
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">
              1. 데이터 (Data)
            </h3>
            <div className="grid md:grid-cols-3 gap-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-medium text-sm mb-2">가격 데이터</h4>
                <ul className="text-xs space-y-1">
                  <li>• 주가 (OHLCV)</li>
                  <li>• 수익률</li>
                  <li>• 변동성</li>
                  <li>• 거래량</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-medium text-sm mb-2">재무 데이터</h4>
                <ul className="text-xs space-y-1">
                  <li>• 매출/이익</li>
                  <li>• PER/PBR</li>
                  <li>• ROE/ROA</li>
                  <li>• 부채비율</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-medium text-sm mb-2">대안 데이터</h4>
                <ul className="text-xs space-y-1">
                  <li>• 뉴스 감성</li>
                  <li>• SNS 데이터</li>
                  <li>• 위성 이미지</li>
                  <li>• 웹 트래픽</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">
              2. 팩터 (Factor)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">전통적 팩터</h4>
                <ul className="text-sm space-y-1">
                  <li>📈 <strong>가치 (Value):</strong> 저평가 종목</li>
                  <li>🚀 <strong>모멘텀 (Momentum):</strong> 상승 추세</li>
                  <li>📊 <strong>규모 (Size):</strong> 소형주 효과</li>
                  <li>💎 <strong>품질 (Quality):</strong> 우량 기업</li>
                  <li>📉 <strong>저변동성 (Low Vol):</strong> 안정적 종목</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">스마트 베타 전략</h4>
                <ul className="text-sm space-y-1">
                  <li>• 단일 팩터 전략</li>
                  <li>• 멀티 팩터 전략</li>
                  <li>• 팩터 타이밍</li>
                  <li>• 팩터 로테이션</li>
                  <li>• 리스크 패리티</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-3">
              3. 모델 (Model)
            </h3>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-medium mb-2">통계 모델</h4>
                <div className="grid md:grid-cols-2 gap-3 text-sm">
                  <ul>
                    <li>• 선형 회귀 분석</li>
                    <li>• 시계열 분석 (ARIMA)</li>
                    <li>• 상관관계 분석</li>
                  </ul>
                  <ul>
                    <li>• 공적분 검정</li>
                    <li>• GARCH 모델</li>
                    <li>• 베이지안 추론</li>
                  </ul>
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-medium mb-2">머신러닝 모델</h4>
                <div className="grid md:grid-cols-2 gap-3 text-sm">
                  <ul>
                    <li>• Random Forest</li>
                    <li>• XGBoost</li>
                    <li>• Neural Networks</li>
                  </ul>
                  <ul>
                    <li>• SVM</li>
                    <li>• LSTM</li>
                    <li>• Ensemble</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔬 기초 퀀트 분석 실습</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">Python으로 시작하는 퀀트 분석</h3>
          
          <div className="space-y-4">
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <h4 className="text-green-400 font-mono text-sm mb-2"># 1. 데이터 수집 및 전처리</h4>
              <pre className="text-sm">
{`import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# 주가 데이터 수집
def get_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        data[ticker] = df['Close']
    
    return pd.DataFrame(data)

# 수익률 계산
def calculate_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

# 예시: FAANG 주식 데이터
tickers = ['AAPL', 'AMZN', 'NFLX', 'GOOGL', 'META']
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

prices = get_stock_data(tickers, start_date, end_date)
returns = calculate_returns(prices)`}</pre>
            </div>

            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <h4 className="text-green-400 font-mono text-sm mb-2"># 2. 기본 통계 분석</h4>
              <pre className="text-sm">
{`# 기술 통계량
def analyze_returns(returns):
    stats = pd.DataFrame({
        '평균 수익률': returns.mean() * 252,  # 연율화
        '변동성': returns.std() * np.sqrt(252),
        '샤프 비율': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        '최대 수익률': returns.max(),
        '최소 수익률': returns.min(),
        '왜도': returns.skew(),
        '첨도': returns.kurtosis()
    })
    return stats

# 상관관계 분석
correlation_matrix = returns.corr()

# 베타 계산 (시장 대비)
def calculate_beta(stock_returns, market_returns):
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta`}</pre>
            </div>

            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <h4 className="text-green-400 font-mono text-sm mb-2"># 3. 간단한 모멘텀 전략</h4>
              <pre className="text-sm">
{`def momentum_strategy(prices, lookback=20, holding=5):
    """
    단순 모멘텀 전략
    - lookback: 과거 수익률 계산 기간
    - holding: 보유 기간
    """
    # 과거 수익률 계산
    momentum = prices.pct_change(lookback)
    
    # 상위 20% 종목 선택
    positions = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    for date in momentum.index[lookback::holding]:
        # 해당 날짜의 모멘텀 순위
        rank = momentum.loc[date].rank(ascending=False)
        # 상위 20% 종목만 선택 (5개 중 1개)
        selected = rank <= len(prices.columns) * 0.2
        positions.loc[date] = selected.astype(int)
    
    # 포지션 forward fill
    positions = positions.fillna(method='ffill').fillna(0)
    
    # 전략 수익률 계산
    strategy_returns = (positions.shift(1) * returns).sum(axis=1)
    
    return positions, strategy_returns`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 포트폴리오 최적화</h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
            현대 포트폴리오 이론 (MPT)
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">효율적 프론티어</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                주어진 위험 수준에서 최대 수익률을 제공하는 포트폴리오의 집합
              </p>
              <ul className="text-sm space-y-1">
                <li>• 위험-수익 트레이드오프</li>
                <li>• 분산 투자 효과</li>
                <li>• 최적 가중치 계산</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">최적화 방법</h4>
              <ul className="text-sm space-y-1">
                <li>📊 최소 분산 포트폴리오</li>
                <li>📈 최대 샤프비율 포트폴리오</li>
                <li>⚖️ 리스크 패리티</li>
                <li>🎯 목표 수익률 포트폴리오</li>
                <li>🛡️ CVaR 최적화</li>
              </ul>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <h4 className="text-green-400 font-mono text-sm mb-2"># 포트폴리오 최적화 예제</h4>
            <pre className="text-sm">
{`from scipy.optimize import minimize

def portfolio_optimization(returns, target_return=None):
    """마코위츠 포트폴리오 최적화"""
    n_assets = len(returns.columns)
    
    # 연평균 수익률과 공분산 행렬
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # 포트폴리오 성과 계산 함수
    def portfolio_stats(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio
    
    # 목적 함수: 샤프비율 최대화 (음수로 변환)
    def neg_sharpe(weights):
        return -portfolio_stats(weights)[2]
    
    # 제약 조건
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # 초기 추정치
    init_weights = np.array([1/n_assets] * n_assets)
    
    # 최적화 실행
    result = minimize(neg_sharpe, init_weights, 
                     method='SLSQP', bounds=bounds, 
                     constraints=constraints)
    
    return result.x`}</pre>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 퀀트 투자의 함정</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            주의해야 할 위험 요소
          </h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">1. 과최적화 (Overfitting)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                과거 데이터에 지나치게 맞춘 모델은 미래에 작동하지 않습니다.
              </p>
              <ul className="text-sm space-y-1">
                <li>• 파라미터가 너무 많은 모델 지양</li>
                <li>• Out-of-sample 테스트 필수</li>
                <li>• Cross-validation 활용</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">2. 데이터 마이닝 편향</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                우연히 발견한 패턴을 의미 있는 것으로 착각하는 오류
              </p>
              <ul className="text-sm space-y-1">
                <li>• 다중 검정 문제 고려</li>
                <li>• 경제적 논리 확인</li>
                <li>• 통계적 유의성 검증</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">3. 거래 비용 과소평가</h4>
              <ul className="text-sm space-y-1">
                <li>• 수수료, 세금 고려</li>
                <li>• 슬리피지 반영</li>
                <li>• 시장 충격 비용</li>
                <li>• 기회비용 계산</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">4. 체제 변화 (Regime Change)</h4>
              <ul className="text-sm space-y-1">
                <li>• 시장 구조 변화 모니터링</li>
                <li>• 모델 성과 지속 추적</li>
                <li>• 적응형 모델 고려</li>
                <li>• 정기적 리밸런싱</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 실전 퀀트 투자 시작하기</h2>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">단계별 접근 방법</h3>
          
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">1</span>
              <div className="flex-1">
                <h4 className="font-semibold">기초 데이터 분석 능력 습득</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Python, pandas, numpy 기본 활용법 학습
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">2</span>
              <div className="flex-1">
                <h4 className="font-semibold">단순 전략부터 시작</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  이동평균, RSI 등 검증된 지표로 시작
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">3</span>
              <div className="flex-1">
                <h4 className="font-semibold">백테스팅 프레임워크 구축</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  거래비용, 슬리피지를 고려한 현실적 시뮬레이션
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">4</span>
              <div className="flex-1">
                <h4 className="font-semibold">소액 실전 투자</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  페이퍼 트레이딩 후 소액으로 실제 검증
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">5</span>
              <div className="flex-1">
                <h4 className="font-semibold">지속적 개선</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  성과 분석, 모델 업데이트, 새로운 팩터 연구
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📚 추천 학습 자료</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-semibold mb-2">필독서</h4>
            <ul className="text-sm space-y-1">
              <li>📖 Quantitative Portfolio Management</li>
              <li>📖 Active Portfolio Management (Grinold)</li>
              <li>📖 Machine Learning for Asset Managers</li>
              <li>📖 Advances in Financial ML (Lopez)</li>
            </ul>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-semibold mb-2">온라인 리소스</h4>
            <ul className="text-sm space-y-1">
              <li>🌐 QuantConnect 튜토리얼</li>
              <li>🌐 Quantopian 아카이브</li>
              <li>🌐 Two Sigma 팩터 연구</li>
              <li>🌐 AQR 리서치 페이퍼</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}