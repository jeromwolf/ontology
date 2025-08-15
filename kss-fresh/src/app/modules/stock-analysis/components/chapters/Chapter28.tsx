'use client';

import { useState } from 'react';

export default function Chapter28() {
  const [selectedOption, setSelectedOption] = useState('call');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">파생상품의 이해와 활용</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          옵션과 선물을 중심으로 파생상품의 기본 개념부터 실전 거래 전략까지 체계적으로 학습합니다.
          리스크 관리와 수익 창출을 위한 다양한 파생상품 활용법을 마스터해봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 파생상품의 기초</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
            파생상품이란?
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">주요 특징</h4>
              <ul className="text-sm space-y-1">
                <li>📊 기초자산에서 파생된 금융상품</li>
                <li>💰 레버리지 효과 (증거금 거래)</li>
                <li>⏰ 만기일이 존재</li>
                <li>🔄 양방향 거래 가능 (매수/매도)</li>
                <li>🛡️ 헤지와 투기 모두 가능</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">주요 종류</h4>
              <ul className="text-sm space-y-1">
                <li><strong>선물(Futures):</strong> 표준화된 거래소 상품</li>
                <li><strong>옵션(Options):</strong> 권리를 거래</li>
                <li><strong>스왑(Swaps):</strong> 현금흐름 교환</li>
                <li><strong>선도(Forwards):</strong> 맞춤형 장외 거래</li>
              </ul>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
              ⚠️ 중요 경고
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              파생상품은 레버리지로 인해 높은 수익과 함께 큰 손실 가능성이 있습니다.
              충분한 이해 없이 거래하면 원금 이상의 손실이 발생할 수 있습니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 선물(Futures) 거래</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
            KOSPI200 선물 이해하기
          </h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-semibold mb-3">선물 거래 메커니즘</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-4 text-sm">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <strong>거래 단위</strong>
                  <ul className="mt-2 space-y-1">
                    <li>• 계약 단위: 지수 × 250,000원</li>
                    <li>• 틱 단위: 0.05포인트</li>
                    <li>• 틱 가치: 12,500원</li>
                    <li>• 증거금률: 약 15%</li>
                  </ul>
                </div>
                <div>
                  <strong>결제 방식</strong>
                  <ul className="mt-2 space-y-1">
                    <li>• 일일정산 (Mark to Market)</li>
                    <li>• 최종결제: 현금결제</li>
                    <li>• 만기일: 매월 둘째 목요일</li>
                    <li>• 롤오버 필요</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <h4 className="text-green-400 font-mono text-sm mb-2"># 선물 거래 손익 계산</h4>
            <pre className="text-sm">
{`class FuturesTrading:
    def __init__(self, contract_multiplier=250000, tick_size=0.05):
        self.multiplier = contract_multiplier
        self.tick_size = tick_size
        self.tick_value = tick_size * contract_multiplier
        
    def calculate_pnl(self, entry_price, exit_price, contracts, position='long'):
        """선물 거래 손익 계산"""
        if position == 'long':
            point_change = exit_price - entry_price
        else:  # short
            point_change = entry_price - exit_price
            
        # 손익 계산
        pnl = point_change * self.multiplier * contracts
        
        # 틱 수 계산
        ticks = point_change / self.tick_size
        
        return {
            'pnl': pnl,
            'point_change': point_change,
            'ticks': ticks,
            'tick_pnl': ticks * self.tick_value * contracts
        }
    
    def margin_requirements(self, current_price, contracts):
        """증거금 요구사항 계산"""
        contract_value = current_price * self.multiplier * contracts
        
        # 증거금률 (약 15%)
        initial_margin = contract_value * 0.15
        maintenance_margin = contract_value * 0.12
        
        return {
            'contract_value': contract_value,
            'initial_margin': initial_margin,
            'maintenance_margin': maintenance_margin,
            'leverage': contract_value / initial_margin
        }`}</pre>
          </div>

          <div className="mt-4 grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">선물 활용 전략</h4>
              <ul className="text-sm space-y-1">
                <li>🛡️ <strong>헤지:</strong> 현물 포지션 보호</li>
                <li>📊 <strong>차익거래:</strong> 현선 스프레드</li>
                <li>📈 <strong>방향성:</strong> 지수 상승/하락 베팅</li>
                <li>🔄 <strong>스프레드:</strong> 근월물-원월물</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">리스크 관리</h4>
              <ul className="text-sm space-y-1">
                <li>• 손절선 설정 필수</li>
                <li>• 포지션 크기 제한</li>
                <li>• 추가증거금 대비</li>
                <li>• 롤오버 비용 고려</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎲 옵션(Options) 거래</h2>
        <div className="mb-4">
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedOption('call')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedOption === 'call'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              콜옵션
            </button>
            <button
              onClick={() => setSelectedOption('put')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedOption === 'put'
                  ? 'bg-red-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              풋옵션
            </button>
            <button
              onClick={() => setSelectedOption('greeks')}
              className={`px-4 py-2 rounded-lg font-medium ${
                selectedOption === 'greeks'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              그릭스
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          {selectedOption === 'call' && (
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4">
                콜옵션 (Call Option)
              </h3>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  특정 가격(행사가)에 기초자산을 <strong>살 수 있는 권리</strong>
                </p>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">콜 매수 (Long Call)</h4>
                    <ul className="text-sm space-y-1">
                      <li>• 최대 손실: 프리미엄</li>
                      <li>• 최대 이익: 무제한</li>
                      <li>• 손익분기점: 행사가 + 프리미엄</li>
                      <li>• 전망: 강한 상승 예상</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">콜 매도 (Short Call)</h4>
                    <ul className="text-sm space-y-1">
                      <li>• 최대 손실: 무제한</li>
                      <li>• 최대 이익: 프리미엄</li>
                      <li>• 손익분기점: 행사가 + 프리미엄</li>
                      <li>• 전망: 하락 또는 보합</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold mb-2">손익 구조</h4>
                <pre className="text-sm">
{`행사가: 300, 프리미엄: 5

        Long Call                    Short Call
         +                             +
         |    /                       |_____
         |   /                        |     \\
         |  /                         |      \\
    ━━━━━|━━━━━━━━━━━              ━━━━━━━━━━━━━━━━━
       295 300 305                  295 300 305
         |                            |
         - (프리미엄 손실)              - (무제한 손실)`}</pre>
              </div>
            </div>
          )}

          {selectedOption === 'put' && (
            <div>
              <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
                풋옵션 (Put Option)
              </h3>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  특정 가격(행사가)에 기초자산을 <strong>팔 수 있는 권리</strong>
                </p>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">풋 매수 (Long Put)</h4>
                    <ul className="text-sm space-y-1">
                      <li>• 최대 손실: 프리미엄</li>
                      <li>• 최대 이익: 행사가 - 프리미엄</li>
                      <li>• 손익분기점: 행사가 - 프리미엄</li>
                      <li>• 전망: 강한 하락 예상</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">풋 매도 (Short Put)</h4>
                    <ul className="text-sm space-y-1">
                      <li>• 최대 손실: 행사가 - 프리미엄</li>
                      <li>• 최대 이익: 프리미엄</li>
                      <li>• 손익분기점: 행사가 - 프리미엄</li>
                      <li>• 전망: 상승 또는 보합</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">보호적 풋 (Protective Put)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  현물 보유 + 풋 매수 = 하방 리스크 제한
                </p>
                <ul className="text-sm space-y-1">
                  <li>• 보험료 개념의 프리미엄</li>
                  <li>• 최대 손실 한정</li>
                  <li>• 상승 이익은 유지</li>
                </ul>
              </div>
            </div>
          )}

          {selectedOption === 'greeks' && (
            <div>
              <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
                옵션 그릭스 (Greeks)
              </h3>
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-2">델타 (Delta) - Δ</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    기초자산 가격 1포인트 변화에 대한 옵션 가격 변화
                  </p>
                  <ul className="text-sm space-y-1">
                    <li>• 콜옵션: 0 ~ 1</li>
                    <li>• 풋옵션: -1 ~ 0</li>
                    <li>• ATM: 약 0.5 (콜), -0.5 (풋)</li>
                    <li>• 헤지 비율로 활용</li>
                  </ul>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-2">감마 (Gamma) - Γ</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    델타의 변화율 (델타의 델타)
                  </p>
                  <ul className="text-sm space-y-1">
                    <li>• ATM에서 최대</li>
                    <li>• 만기 임박시 급증</li>
                    <li>• 감마 리스크 관리 중요</li>
                  </ul>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-2">세타 (Theta) - Θ</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    시간 경과에 따른 옵션 가치 감소 (시간가치 소멸)
                  </p>
                  <ul className="text-sm space-y-1">
                    <li>• 매수자: 음수 (불리)</li>
                    <li>• 매도자: 양수 (유리)</li>
                    <li>• 만기 임박시 가속화</li>
                  </ul>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-2">베가 (Vega) - ν</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    내재변동성 1% 변화에 대한 옵션 가격 변화
                  </p>
                  <ul className="text-sm space-y-1">
                    <li>• ATM에서 최대</li>
                    <li>• 잔존기간 길수록 큼</li>
                    <li>• 변동성 거래의 핵심</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 옵션 거래 전략</h2>
        <div className="space-y-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4">
              기본 전략
            </h3>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Covered Call</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  현물 보유 + 콜 매도
                </p>
                <ul className="text-sm space-y-1">
                  <li>• 추가 수익 창출</li>
                  <li>• 상승 이익 제한</li>
                  <li>• 하락 리스크 일부 완화</li>
                  <li>• 보수적 전략</li>
                </ul>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Cash Secured Put</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  현금 보유 + 풋 매도
                </p>
                <ul className="text-sm space-y-1">
                  <li>• 목표가 매수 전략</li>
                  <li>• 프리미엄 수익</li>
                  <li>• 하락시 주식 배정</li>
                  <li>• 현금 확보 필수</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4">
              스프레드 전략
            </h3>
            
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <h4 className="text-green-400 font-mono text-sm mb-2"># 불 콜 스프레드 (Bull Call Spread)</h4>
              <pre className="text-sm">
{`def bull_call_spread(spot_price, lower_strike, upper_strike, 
                     lower_premium, upper_premium):
    """
    완만한 상승 전망시 사용
    낮은 행사가 콜 매수 + 높은 행사가 콜 매도
    """
    net_debit = lower_premium - upper_premium
    
    # 손익 계산
    if spot_price <= lower_strike:
        pnl = -net_debit
    elif spot_price >= upper_strike:
        pnl = (upper_strike - lower_strike) - net_debit
    else:
        pnl = (spot_price - lower_strike) - net_debit
    
    return {
        'pnl': pnl,
        'max_profit': (upper_strike - lower_strike) - net_debit,
        'max_loss': -net_debit,
        'breakeven': lower_strike + net_debit
    }

# 예시: 290 콜 매수(7) + 310 콜 매도(2)
result = bull_call_spread(
    spot_price=305, 
    lower_strike=290, 
    upper_strike=310,
    lower_premium=7, 
    upper_premium=2
)`}</pre>
            </div>

            <div className="mt-4 grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">수직 스프레드</h4>
                <ul className="text-sm space-y-1">
                  <li>• Bull Call/Put Spread</li>
                  <li>• Bear Call/Put Spread</li>
                  <li>• 리스크/수익 제한</li>
                  <li>• 자본 효율적</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">수평 스프레드</h4>
                <ul className="text-sm space-y-1">
                  <li>• Calendar Spread</li>
                  <li>• 시간가치 차이 활용</li>
                  <li>• 변동성 거래</li>
                  <li>• 세타 수익</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💎 고급 옵션 전략</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
            변동성 거래 전략
          </h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">스트래들 (Straddle)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                ATM 콜 매수 + ATM 풋 매수
              </p>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <strong>적합 상황:</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 큰 변동성 예상</li>
                    <li>• 방향성 불확실</li>
                    <li>• 이벤트 전 (실적 발표 등)</li>
                  </ul>
                </div>
                <div>
                  <strong>손익 구조:</strong>
                  <ul className="mt-1 space-y-1">
                    <li>• 최대 손실: 프리미엄 합</li>
                    <li>• 최대 이익: 무제한</li>
                    <li>• 양방향 수익 가능</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">아이언 콘도르 (Iron Condor)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                Bear Call Spread + Bull Put Spread
              </p>
              <div className="bg-gray-100 dark:bg-gray-700 rounded p-3 text-sm">
                <pre>
{`구성:
- 280 Put 매도
- 270 Put 매수  } Bull Put Spread
- 320 Call 매도
- 330 Call 매수 } Bear Call Spread

장점: 양쪽에서 프리미엄 수취
단점: 이익 제한, 양방향 리스크`}</pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">버터플라이 (Butterfly)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                1 ITM 콜 매수 + 2 ATM 콜 매도 + 1 OTM 콜 매수
              </p>
              <ul className="text-sm space-y-1">
                <li>• 특정 가격대 도달 예상</li>
                <li>• 낮은 비용, 제한된 수익</li>
                <li>• 시간가치 소멸 활용</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 실전 거래 예시</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">KOSPI200 옵션 실전 시나리오</h3>
          
          <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
            <h4 className="font-semibold mb-3">시나리오: 실적 시즌 변동성 거래</h4>
            
            <div className="space-y-3 text-sm">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
                <strong>상황:</strong>
                <p>삼성전자 실적 발표 전, KOSPI200 지수 300포인트</p>
                <p>내재변동성 상승 (IV: 15% → 20%)</p>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                <strong>전략: Long Straddle</strong>
                <ul className="mt-1 space-y-1">
                  <li>• 300 콜 매수: 프리미엄 5.0</li>
                  <li>• 300 풋 매수: 프리미엄 4.5</li>
                  <li>• 총 비용: 9.5 포인트</li>
                </ul>
              </div>
              
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded p-3">
                <strong>결과 시나리오:</strong>
                <ul className="mt-1 space-y-1">
                  <li>📈 지수 315: 이익 5.5 (58% 수익)</li>
                  <li>📉 지수 285: 이익 5.5 (58% 수익)</li>
                  <li>➡️ 지수 300: 손실 9.5 (100% 손실)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 리스크 관리</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4">
            파생상품 거래의 위험 요소
          </h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">주요 리스크</h4>
              <ul className="text-sm space-y-1">
                <li>🔥 레버리지 리스크</li>
                <li>⏰ 시간가치 소멸</li>
                <li>📊 변동성 리스크</li>
                <li>💧 유동성 리스크</li>
                <li>📌 핀 리스크 (만기시)</li>
                <li>🔄 롤오버 리스크</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">리스크 관리 원칙</h4>
              <ul className="text-sm space-y-1">
                <li>✅ 전체 자산의 10% 이하</li>
                <li>✅ 손실 한도 사전 설정</li>
                <li>✅ 복잡한 전략 지양</li>
                <li>✅ 충분한 시뮬레이션</li>
                <li>✅ 지속적 모니터링</li>
                <li>✅ 비상 계획 수립</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
              💡 초보자 권장사항
            </p>
            <ol className="text-sm text-gray-600 dark:text-gray-400 mt-2 space-y-1">
              <li>1. 모의투자로 충분한 연습</li>
              <li>2. 단순한 전략부터 시작</li>
              <li>3. 매도 포지션 신중히 접근</li>
              <li>4. 만기일 가까운 옵션 주의</li>
              <li>5. 전문가 교육 이수 권장</li>
            </ol>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📚 학습 로드맵</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">파생상품 마스터 과정</h3>
          
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">1</span>
              <div className="flex-1">
                <h4 className="font-semibold">이론 학습 (2-4주)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  파생상품 기초, 가격결정 이론, 그릭스 이해
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">2</span>
              <div className="flex-1">
                <h4 className="font-semibold">모의투자 (1-2개월)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  기본 전략 연습, 손익 시뮬레이션, 리스크 체험
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">3</span>
              <div className="flex-1">
                <h4 className="font-semibold">소액 실전 (3-6개월)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  단순 전략 실행, 실제 체결 경험, 심리 관리
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0">4</span>
              <div className="flex-1">
                <h4 className="font-semibold">전략 확대</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  복합 전략 구사, 포트폴리오 헤지, 수익 모델 구축
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}