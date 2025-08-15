'use client';

import { useState } from 'react';

export default function Chapter24() {
  const [activeTab, setActiveTab] = useState('architecture');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">자동매매 전략 구축</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          시스템 트레이딩을 자동화하여 24시간 시장을 모니터링하고 거래하는 방법을 배워봅시다.
          Python과 API를 활용한 실제 자동매매 시스템 구축 과정을 단계별로 알아봅니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏗️ 자동매매 시스템 아키텍처</h2>
        <div className="mb-4">
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab('architecture')}
              className={`px-4 py-2 rounded-lg font-medium ${
                activeTab === 'architecture'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              시스템 구조
            </button>
            <button
              onClick={() => setActiveTab('components')}
              className={`px-4 py-2 rounded-lg font-medium ${
                activeTab === 'components'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              핵심 컴포넌트
            </button>
            <button
              onClick={() => setActiveTab('flow')}
              className={`px-4 py-2 rounded-lg font-medium ${
                activeTab === 'flow'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              실행 플로우
            </button>
          </div>
        </div>

        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          {activeTab === 'architecture' && (
            <div>
              <h3 className="font-semibold mb-4">자동매매 시스템 전체 구조</h3>
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <pre className="text-sm">
{`┌─────────────────────────────────────────────┐
│             자동매매 시스템                   │
├─────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐         │
│  │ Data Feed   │  │ Strategy     │         │
│  │ (실시간시세) │→ │ Engine       │         │
│  └─────────────┘  └──────────────┘         │
│         ↓                ↓                  │
│  ┌─────────────┐  ┌──────────────┐         │
│  │ Database    │  │ Order        │         │
│  │ (데이터저장) │← │ Management   │         │
│  └─────────────┘  └──────────────┘         │
│                          ↓                  │
│                   ┌──────────────┐         │
│                   │ Broker API   │         │
│                   │ (증권사연동)  │         │
│                   └──────────────┘         │
└─────────────────────────────────────────────┘`}</pre>
              </div>
            </div>
          )}

          {activeTab === 'components' && (
            <div>
              <h3 className="font-semibold mb-4">핵심 컴포넌트 상세</h3>
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                    1. Data Feed Module
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• 실시간 시세 수신 (WebSocket)</li>
                    <li>• 과거 데이터 조회 (REST API)</li>
                    <li>• 데이터 정규화 및 검증</li>
                    <li>• 시계열 데이터 관리</li>
                  </ul>
                </div>
                <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                    2. Strategy Engine
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• 신호 생성 로직 실행</li>
                    <li>• 기술적 지표 계산</li>
                    <li>• 포지션 관리</li>
                    <li>• 리스크 체크</li>
                  </ul>
                </div>
                <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
                    3. Order Management System
                  </h4>
                  <ul className="text-sm space-y-1">
                    <li>• 주문 생성 및 전송</li>
                    <li>• 주문 상태 추적</li>
                    <li>• 부분 체결 처리</li>
                    <li>• 오류 처리 및 재시도</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'flow' && (
            <div>
              <h3 className="font-semibold mb-4">실행 플로우</h3>
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <ol className="space-y-3 text-sm">
                  <li className="flex items-start gap-2">
                    <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0">1</span>
                    <div>
                      <strong>시장 데이터 수신</strong>
                      <p className="text-gray-600 dark:text-gray-400">실시간 가격, 거래량, 호가 정보</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0">2</span>
                    <div>
                      <strong>전략 신호 계산</strong>
                      <p className="text-gray-600 dark:text-gray-400">지표 업데이트, 진입/청산 조건 확인</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0">3</span>
                    <div>
                      <strong>리스크 검증</strong>
                      <p className="text-gray-600 dark:text-gray-400">포지션 한도, 손실 한도 체크</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0">4</span>
                    <div>
                      <strong>주문 실행</strong>
                      <p className="text-gray-600 dark:text-gray-400">API를 통한 매수/매도 주문</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0">5</span>
                    <div>
                      <strong>모니터링 & 로깅</strong>
                      <p className="text-gray-600 dark:text-gray-400">체결 확인, 성과 기록, 오류 알림</p>
                    </div>
                  </li>
                </ol>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔌 증권사 API 연동</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
            주요 증권사 API 비교
          </h3>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-100 dark:bg-blue-800">
                <tr>
                  <th className="px-4 py-2 text-left">증권사</th>
                  <th className="px-4 py-2 text-center">실시간</th>
                  <th className="px-4 py-2 text-center">주식</th>
                  <th className="px-4 py-2 text-center">선물/옵션</th>
                  <th className="px-4 py-2 text-center">모의투자</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-300 dark:divide-gray-600">
                <tr className="bg-white dark:bg-gray-800">
                  <td className="px-4 py-2">한국투자증권</td>
                  <td className="px-4 py-2 text-center">✅</td>
                  <td className="px-4 py-2 text-center">✅</td>
                  <td className="px-4 py-2 text-center">✅</td>
                  <td className="px-4 py-2 text-center">✅</td>
                </tr>
                <tr className="bg-white dark:bg-gray-800">
                  <td className="px-4 py-2">키움증권</td>
                  <td className="px-4 py-2 text-center">✅</td>
                  <td className="px-4 py-2 text-center">✅</td>
                  <td className="px-4 py-2 text-center">✅</td>
                  <td className="px-4 py-2 text-center">❌</td>
                </tr>
                <tr className="bg-white dark:bg-gray-800">
                  <td className="px-4 py-2">이베스트투자증권</td>
                  <td className="px-4 py-2 text-center">✅</td>
                  <td className="px-4 py-2 text-center">✅</td>
                  <td className="px-4 py-2 text-center">❌</td>
                  <td className="px-4 py-2 text-center">✅</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              💡 Tip: 모의투자를 지원하는 증권사로 시작하여 충분히 테스트 후 실전 적용
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💻 Python 구현 예제</h2>
        <div className="space-y-4">
          <div className="bg-gray-900 text-gray-100 rounded-lg p-6 overflow-x-auto">
            <h3 className="font-semibold mb-3 text-green-400"># 자동매매 시스템 기본 구조</h3>
            <pre className="text-sm">
{`import asyncio
import pandas as pd
from datetime import datetime
import logging

class AutoTradingSystem:
    def __init__(self, strategy, broker_api):
        self.strategy = strategy
        self.broker = broker_api
        self.positions = {}
        self.is_running = False
        
    async def start(self):
        """자동매매 시작"""
        self.is_running = True
        logging.info("자동매매 시스템 시작")
        
        # 동시 실행 태스크
        tasks = [
            self.market_data_handler(),
            self.strategy_executor(),
            self.risk_monitor()
        ]
        
        await asyncio.gather(*tasks)
    
    async def market_data_handler(self):
        """실시간 시세 처리"""
        async for data in self.broker.stream_market_data():
            if not self.is_running:
                break
                
            # 데이터 전처리
            processed_data = self.preprocess_data(data)
            
            # 전략에 데이터 전달
            await self.strategy.on_data(processed_data)
    
    async def strategy_executor(self):
        """전략 실행 및 주문 처리"""
        while self.is_running:
            # 전략 신호 확인
            signals = await self.strategy.get_signals()
            
            for signal in signals:
                # 리스크 체크
                if self.check_risk_limits(signal):
                    # 주문 실행
                    order = await self.execute_order(signal)
                    logging.info(f"주문 실행: {order}")
            
            await asyncio.sleep(1)  # 1초 대기`}</pre>
          </div>

          <div className="bg-gray-900 text-gray-100 rounded-lg p-6 overflow-x-auto">
            <h3 className="font-semibold mb-3 text-green-400"># 리스크 관리 모듈</h3>
            <pre className="text-sm">
{`class RiskManager:
    def __init__(self, max_position_size=0.1, max_daily_loss=0.02):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0
        
    def check_position_size(self, order, account_value):
        """포지션 크기 제한 확인"""
        position_value = order.quantity * order.price
        position_ratio = position_value / account_value
        
        if position_ratio > self.max_position_size:
            logging.warning(f"포지션 한도 초과: {position_ratio:.2%}")
            return False
        return True
    
    def check_daily_loss_limit(self, account_value):
        """일일 손실 한도 확인"""
        loss_ratio = abs(self.daily_pnl) / account_value
        
        if loss_ratio > self.max_daily_loss:
            logging.error(f"일일 손실한도 도달: {loss_ratio:.2%}")
            return False
        return True
    
    def update_pnl(self, realized_pnl):
        """실현 손익 업데이트"""
        self.daily_pnl += realized_pnl`}</pre>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ 실시간 데이터 처리</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
            효율적인 데이터 처리 방법
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">데이터 스트리밍</h4>
              <ul className="text-sm space-y-1">
                <li>• WebSocket 연결 관리</li>
                <li>• 재연결 로직 구현</li>
                <li>• 메시지 큐 활용 (Redis)</li>
                <li>• 비동기 처리 (asyncio)</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">데이터 저장</h4>
              <ul className="text-sm space-y-1">
                <li>• 시계열 DB 활용 (InfluxDB)</li>
                <li>• 압축 저장 (Parquet)</li>
                <li>• 인메모리 캐시 (Redis)</li>
                <li>• 백업 및 복구 전략</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2">실시간 지표 계산</h4>
            <pre className="text-sm bg-gray-100 dark:bg-gray-700 p-3 rounded overflow-x-auto">
{`# 증분 이동평균 계산 (메모리 효율적)
class IncrementalMA:
    def __init__(self, period):
        self.period = period
        self.values = deque(maxlen=period)
        self.sum = 0
    
    def update(self, value):
        if len(self.values) == self.period:
            self.sum -= self.values[0]
        self.values.append(value)
        self.sum += value
        return self.sum / len(self.values)`}</pre>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🛡️ 안정성과 예외 처리</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            자동매매 시스템의 위험 요소
          </h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">1. 시스템 오류 대응</h4>
              <ul className="text-sm space-y-1">
                <li>⚠️ API 연결 끊김 → 자동 재연결 + 알림</li>
                <li>⚠️ 데이터 오류 → 검증 로직 + 필터링</li>
                <li>⚠️ 주문 실패 → 재시도 + 수동 개입</li>
                <li>⚠️ 시스템 다운 → 백업 서버 + 복구</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">2. 시장 이상 상황</h4>
              <ul className="text-sm space-y-1">
                <li>🚨 서킷브레이커 → 거래 중단 로직</li>
                <li>🚨 급격한 변동성 → 포지션 축소</li>
                <li>🚨 유동성 부족 → 주문 분할</li>
                <li>🚨 시스템 트레이딩 금지 → 수동 전환</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">3. 모니터링 시스템</h4>
              <pre className="text-sm bg-gray-100 dark:bg-gray-700 p-3 rounded overflow-x-auto">
{`# 텔레그램 알림 설정
async def send_alert(message, level="INFO"):
    if level == "CRITICAL":
        # 긴급 알림 (전화/SMS)
        await send_sms(message)
    
    # 텔레그램 메시지
    await telegram_bot.send_message(
        chat_id=CHAT_ID,
        text=f"[{level}] {message}"
    )`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 성과 모니터링</h2>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
            실시간 대시보드 구성
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">실시간 지표</h4>
              <ul className="text-sm space-y-1">
                <li>📈 현재 포지션 및 손익</li>
                <li>📊 일/주/월 수익률</li>
                <li>⚡ 거래 빈도 및 승률</li>
                <li>💰 계좌 잔고 변화</li>
                <li>🎯 전략별 성과 분석</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">시스템 상태</h4>
              <ul className="text-sm space-y-1">
                <li>🟢 API 연결 상태</li>
                <li>💾 메모리/CPU 사용률</li>
                <li>📡 네트워크 지연시간</li>
                <li>⏱️ 주문 처리 속도</li>
                <li>🔔 오류 발생 현황</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2">Grafana 대시보드 예시</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-4 text-center text-sm">
              <p>📊 실시간 수익률 차트</p>
              <p>📈 포지션 히트맵</p>
              <p>🎯 전략 성과 비교</p>
              <p>⚠️ 리스크 지표 모니터</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🚀 실전 운용 가이드</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">단계별 실전 적용</h3>
          
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">1</span>
              <div className="flex-1">
                <h4 className="font-semibold">모의투자 (1-3개월)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  • 시스템 안정성 검증<br/>
                  • 버그 수정 및 최적화<br/>
                  • 다양한 시장 상황 테스트
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">2</span>
              <div className="flex-1">
                <h4 className="font-semibold">소액 실전 (3-6개월)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  • 최소 금액으로 시작<br/>
                  • 실제 체결 및 슬리피지 확인<br/>
                  • 심리적 압박 테스트
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">3</span>
              <div className="flex-1">
                <h4 className="font-semibold">점진적 확대</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  • 성과 검증 후 자금 증액<br/>
                  • 복수 전략 포트폴리오<br/>
                  • 지속적 개선 및 업데이트
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">✅ 자동매매 체크리스트</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">실전 운용 전 최종 점검</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>모든 예외 상황에 대한 처리 로직이 구현되어 있는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>시스템 다운 시 포지션 관리 계획이 있는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>실시간 모니터링과 알림 시스템이 작동하는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>일일 손실 한도와 킬 스위치가 설정되어 있는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>백업 시스템과 복구 절차가 준비되어 있는가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>세금 신고를 위한 거래 기록이 자동 저장되는가?</span>
            </label>
          </div>
        </div>
      </section>
    </div>
  )
}