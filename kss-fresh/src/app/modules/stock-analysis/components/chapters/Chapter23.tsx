'use client';

import { useState } from 'react';

export default function Chapter23() {
  const [selectedStrategy, setSelectedStrategy] = useState('momentum');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">시스템 트레이딩 입문</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          감정을 배제하고 규칙에 따라 매매하는 시스템 트레이딩의 세계로 들어가봅시다.
          전략 설계부터 백테스팅, 최적화, 실전 운용까지 체계적으로 배워봅니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 시스템 트레이딩이란?</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">핵심 개념</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">시스템 트레이딩의 장점</h4>
              <ul className="text-sm space-y-1">
                <li>✅ 감정 배제 - 일관된 규칙 적용</li>
                <li>✅ 백테스팅 - 과거 데이터로 검증</li>
                <li>✅ 리스크 관리 - 체계적 손실 제한</li>
                <li>✅ 시간 효율 - 자동화 가능</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">필요한 요소</h4>
              <ul className="text-sm space-y-1">
                <li>📊 명확한 진입/청산 규칙</li>
                <li>💰 포지션 사이징 전략</li>
                <li>🛡️ 리스크 관리 규칙</li>
                <li>📈 성과 평가 지표</li>
              </ul>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
              ⚠️ 중요: 시스템 트레이딩 ≠ 자동매매
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              시스템 트레이딩은 규칙 기반 매매를 의미하며, 수동으로도 실행 가능합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📐 트레이딩 시스템 설계</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">시스템 구성 요소</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
                1. 시장 필터 (Market Filter)
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                거래할 시장 상황을 정의
              </p>
              <ul className="text-sm space-y-1 pl-4">
                <li>• 추세 시장 vs 횡보 시장</li>
                <li>• 변동성 레벨 (VIX &lt; 20 등)</li>
                <li>• 거래량 조건</li>
                <li>• 시간대 필터 (장 초반 제외 등)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">
                2. 진입 신호 (Entry Signal)
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                포지션 진입 조건
              </p>
              <div className="grid md:grid-cols-2 gap-2 text-sm">
                <div>
                  <strong>추세 추종:</strong>
                  <ul className="pl-4">
                    <li>• 이평선 돌파</li>
                    <li>• 채널 브레이크아웃</li>
                    <li>• 모멘텀 지표</li>
                  </ul>
                </div>
                <div>
                  <strong>평균 회귀:</strong>
                  <ul className="pl-4">
                    <li>• RSI 과매도/과매수</li>
                    <li>• 볼린저밴드 터치</li>
                    <li>• 지지/저항 반등</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">
                3. 청산 규칙 (Exit Rules)
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                포지션 종료 조건
              </p>
              <ul className="text-sm space-y-1 pl-4">
                <li>• 목표가 도달 (고정 % 또는 ATR 배수)</li>
                <li>• 손절가 터치 (고정 % 또는 변동성 기반)</li>
                <li>• 시간 기반 청산 (N일 후 자동 청산)</li>
                <li>• 트레일링 스탑 (이익 보호)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">
                4. 포지션 사이징 (Position Sizing)
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                거래 규모 결정
              </p>
              <ul className="text-sm space-y-1 pl-4">
                <li>• 고정 금액/주식수</li>
                <li>• 계좌의 고정 비율 (예: 2%)</li>
                <li>• 켈리 공식 (Kelly Criterion)</li>
                <li>• 변동성 기반 (ATR 활용)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🧪 백테스팅 (Backtesting)</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">백테스팅 프로세스</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Step 1: 데이터 준비</h4>
              <ul className="text-sm space-y-1">
                <li>• 충분한 기간의 과거 데이터 (최소 5년 권장)</li>
                <li>• 생존 편향 제거 (상장폐지 종목 포함)</li>
                <li>• 배당금, 분할 조정</li>
                <li>• 거래 비용 고려 (수수료, 슬리피지)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Step 2: 시뮬레이션 실행</h4>
              <div className="bg-gray-100 dark:bg-gray-600 rounded p-3 font-mono text-sm mb-2">
                <code>
                  for date in trading_days:<br/>
                  &nbsp;&nbsp;signals = check_entry_signals(data, date)<br/>
                  &nbsp;&nbsp;if signals:<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;execute_trades(signals)<br/>
                  &nbsp;&nbsp;update_positions()<br/>
                  &nbsp;&nbsp;check_exit_conditions()
                </code>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Step 3: 성과 분석</h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <strong>수익성 지표:</strong>
                  <ul className="pl-4">
                    <li>• 총 수익률 (CAGR)</li>
                    <li>• 샤프 비율 (Sharpe Ratio)</li>
                    <li>• 승률 &amp; 손익비</li>
                    <li>• 최대 연속 손실</li>
                  </ul>
                </div>
                <div>
                  <strong>위험 지표:</strong>
                  <ul className="pl-4">
                    <li>• 최대 낙폭 (MDD)</li>
                    <li>• 변동성 (표준편차)</li>
                    <li>• 하방 편차</li>
                    <li>• 회복 기간</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ 최적화와 과최적화</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            과최적화(Overfitting)의 위험
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">과최적화 징후</h4>
              <ul className="text-sm space-y-1">
                <li>🚨 너무 많은 매개변수 (5개 이상)</li>
                <li>🚨 특정 기간에만 작동</li>
                <li>🚨 작은 변경에 민감한 결과</li>
                <li>🚨 비현실적으로 높은 수익률</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">방지 방법</h4>
              <ul className="text-sm space-y-1">
                <li>✅ Walk-Forward Analysis</li>
                <li>✅ Out-of-Sample 테스트</li>
                <li>✅ 단순한 규칙 선호</li>
                <li>✅ 다양한 시장 상황 테스트</li>
              </ul>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 mt-4">
            <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
              Walk-Forward Analysis
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              1. 전체 데이터를 여러 구간으로 분할<br/>
              2. 첫 구간에서 최적화<br/>
              3. 다음 구간에서 검증<br/>
              4. 구간을 이동하며 반복
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 대표적인 시스템 트레이딩 전략</h2>
        <div className="mb-4">
          <select 
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
            className="px-4 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
          >
            <option value="momentum">모멘텀 전략</option>
            <option value="meanreversion">평균회귀 전략</option>
            <option value="breakout">브레이크아웃 전략</option>
            <option value="pairs">페어 트레이딩</option>
          </select>
        </div>

        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          {selectedStrategy === 'momentum' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">
                듀얼 모멘텀 전략
              </h3>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">전략 개요</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  절대 모멘텀과 상대 모멘텀을 결합한 전략
                </p>
                <div className="space-y-2 text-sm">
                  <div><strong>진입 조건:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• 12개월 수익률 &gt; 0 (절대 모멘텀)</li>
                    <li>• 섹터 내 상위 20% (상대 모멘텀)</li>
                    <li>• 20일 이평선 &gt; 50일 이평선</li>
                  </ul>
                  <div className="mt-2"><strong>청산 조건:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• 20일 이평선 &lt; 50일 이평선</li>
                    <li>• 트레일링 스탑 -10%</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'meanreversion' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">
                RSI 평균회귀 전략
              </h3>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">전략 개요</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  과매도/과매수 구간에서 반대 포지션 진입
                </p>
                <div className="space-y-2 text-sm">
                  <div><strong>매수 조건:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• RSI(14) &lt; 30</li>
                    <li>• 가격 &gt; 200일 이평선 (상승 추세)</li>
                    <li>• 거래량 &gt; 20일 평균 거래량</li>
                  </ul>
                  <div className="mt-2"><strong>매도 조건:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• RSI(14) &gt; 70 또는</li>
                    <li>• 수익률 +5% 또는</li>
                    <li>• 손실률 -3%</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'breakout' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">
                채널 브레이크아웃 전략
              </h3>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">전략 개요</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  가격이 일정 기간 최고/최저를 돌파할 때 진입
                </p>
                <div className="space-y-2 text-sm">
                  <div><strong>매수 조건:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• 20일 최고가 돌파</li>
                    <li>• 거래량 &gt; 50일 평균의 150%</li>
                    <li>• ADX &gt; 25 (추세 강도)</li>
                  </ul>
                  <div className="mt-2"><strong>청산 조건:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• 10일 최저가 하향 돌파</li>
                    <li>• 초기 스탑: 진입가 -2%</li>
                    <li>• 트레일링 스탑: 고점 대비 -5%</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {selectedStrategy === 'pairs' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">
                통계적 페어 트레이딩
              </h3>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">전략 개요</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  상관관계가 높은 두 종목의 스프레드 거래
                </p>
                <div className="space-y-2 text-sm">
                  <div><strong>진입 조건:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• 두 종목 상관계수 &gt; 0.8</li>
                    <li>• 스프레드 &gt; 평균 + 2σ</li>
                    <li>• 공적분 검정 통과</li>
                  </ul>
                  <div className="mt-2"><strong>포지션:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• 고평가 종목 매도</li>
                    <li>• 저평가 종목 매수</li>
                    <li>• 시장 중립적 포지션</li>
                  </ul>
                  <div className="mt-2"><strong>청산:</strong></div>
                  <ul className="pl-4 space-y-1">
                    <li>• 스프레드 평균 회귀</li>
                    <li>• 최대 보유 기간 30일</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💻 시스템 구현 예시</h2>
        <div className="bg-gray-900 text-gray-100 rounded-lg p-6 overflow-x-auto">
          <h3 className="font-semibold mb-3 text-green-400"># 간단한 이동평균 크로스오버 전략</h3>
          <pre className="text-sm">
{`import pandas as pd
import numpy as np

class MovingAverageCrossover:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # 이동평균 계산
        signals['short_ma'] = data['close'].rolling(
            window=self.short_window, min_periods=1
        ).mean()
        signals['long_ma'] = data['close'].rolling(
            window=self.long_window, min_periods=1
        ).mean()
        
        # 매수/매도 신호 생성
        signals['signal'] = 0.0
        signals['signal'][self.short_window:] = np.where(
            signals['short_ma'][self.short_window:] > 
            signals['long_ma'][self.short_window:], 1.0, 0.0
        )
        signals['positions'] = signals['signal'].diff()
        
        return signals`}</pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 시스템 성과 평가</h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
            주요 성과 지표
          </h3>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">수익성 지표</h4>
              <ul className="text-sm space-y-1">
                <li><strong>CAGR:</strong> 연평균 수익률</li>
                <li><strong>총 수익률:</strong> 전체 기간 수익</li>
                <li><strong>월별 수익률:</strong> 안정성 확인</li>
                <li><strong>승률:</strong> 수익 거래 비율</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">위험 지표</h4>
              <ul className="text-sm space-y-1">
                <li><strong>최대낙폭:</strong> 최대 손실 구간</li>
                <li><strong>변동성:</strong> 수익률 표준편차</li>
                <li><strong>왜도:</strong> 분포의 비대칭성</li>
                <li><strong>첨도:</strong> 극단값 발생 빈도</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">효율성 지표</h4>
              <ul className="text-sm space-y-1">
                <li><strong>샤프비율:</strong> 위험 대비 수익</li>
                <li><strong>소르티노비율:</strong> 하방 위험 고려</li>
                <li><strong>칼마비율:</strong> MDD 대비 수익</li>
                <li><strong>정보비율:</strong> 초과수익 안정성</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 실전 체크리스트</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">시스템 트레이딩 시작 전 확인사항</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>백테스팅 기간이 최소 5년 이상인가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>거래 비용(수수료, 슬리피지)을 고려했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>Out-of-sample 테스트를 수행했는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>최대낙폭이 감당 가능한 수준인가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>시스템 고장 시 대응 계획이 있는가?</span>
            </label>
          </div>
        </div>
      </section>
    </div>
  );
}