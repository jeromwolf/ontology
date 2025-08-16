'use client';

import React from 'react';
import { DollarSign, TrendingUp, TrendingDown, Shield, Calculator, AlertTriangle, BarChart3, Target } from 'lucide-react';

export default function Chapter37() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">통화 헤지 전략</h1>
      
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <DollarSign className="w-8 h-8 text-green-500" />
          환율 변동이 투자 수익률에 미치는 영향
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">환율 영향 계산 공식</h3>
          <div className="bg-white dark:bg-gray-800 rounded p-4 font-mono text-sm">
            <p className="mb-2">총 수익률 = (1 + 현지통화 수익률) × (1 + 환율 변동률) - 1</p>
            <p className="text-gray-600 dark:text-gray-400">≈ 현지통화 수익률 + 환율 변동률</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              환율 상승 시 (원화 약세)
            </h3>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <p className="font-medium">예시: USD/KRW 1,200 → 1,300</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">환율 수익: +8.33%</p>
              </div>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>✅ 해외자산 원화 가치 상승</li>
                <li>✅ 환차익 발생</li>
                <li>✅ 수익률 부스터 효과</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingDown className="w-5 h-5" />
              환율 하락 시 (원화 강세)
            </h3>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <p className="font-medium">예시: USD/KRW 1,300 → 1,200</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">환율 손실: -7.69%</p>
              </div>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>❌ 해외자산 원화 가치 하락</li>
                <li>❌ 환차손 발생</li>
                <li>❌ 수익률 훼손</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 mt-6">
          <h3 className="text-lg font-semibold mb-3">실제 사례: 2022년 미국 주식 투자</h3>
          <div className="grid md:grid-cols-3 gap-4 text-center">
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">S&P 500 수익률</p>
              <p className="text-2xl font-bold text-red-600">-18.1%</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">USD/KRW 변동</p>
              <p className="text-2xl font-bold text-green-600">+9.5%</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">원화 환산 수익률</p>
              <p className="text-2xl font-bold text-orange-600">-10.3%</p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Shield className="w-8 h-8 text-blue-500" />
          주요 헤징 수단
        </h2>
        
        <div className="space-y-6">
          {/* 선물환 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-semibold mb-4">1. 선물환 (Forward)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">특징</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>• 미래 특정 시점의 환율 고정</li>
                  <li>• 은행과의 장외거래 (OTC)</li>
                  <li>• 맞춤형 계약 가능</li>
                  <li>• 최소 거래 금액 존재</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-3">적합한 경우</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>✓ 만기가 확실한 투자</li>
                  <li>✓ 대규모 자금 헤지</li>
                  <li>✓ 환율 확정성 중요</li>
                  <li>✓ 기업/기관 투자자</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
              <p className="text-sm">
                <strong>예시:</strong> 3개월 후 $100,000 매도 선물환 계약 (환율 1,250원 고정)
              </p>
            </div>
          </div>

          {/* 통화 옵션 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-semibold mb-4">2. 통화 옵션 (Currency Options)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">특징</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>• 권리만 있고 의무는 없음</li>
                  <li>• 프리미엄 지불 필요</li>
                  <li>• 유리한 환율 변동 시 수익 가능</li>
                  <li>• 다양한 전략 구성 가능</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-3">옵션 전략</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>📈 Put 옵션: 환율 하락 대비</li>
                  <li>📉 Call 옵션: 환율 상승 대비</li>
                  <li>🎯 Collar: 상하한 설정</li>
                  <li>💎 Seagull: 비용 절감형</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
              <p className="text-sm">
                <strong>예시:</strong> USD Put 옵션 매수 (행사가 1,200원, 프리미엄 2%)
              </p>
            </div>
          </div>

          {/* 통화 스왑 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-semibold mb-4">3. 통화 스왑 (Currency Swap)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">특징</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>• 원금과 이자의 교환</li>
                  <li>• 장기 헤지에 적합</li>
                  <li>• 자금 조달 비용 절감</li>
                  <li>• 복잡한 구조</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-3">활용 사례</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>• 해외 채권 투자</li>
                  <li>• 다국적 기업 자금관리</li>
                  <li>• 장기 해외 프로젝트</li>
                  <li>• Cross-border M&A</li>
                </ul>
              </div>
            </div>
          </div>

          {/* ETF 헤지 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-semibold mb-4">4. 환헤지 ETF</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">장점</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>✅ 개인투자자 접근 용이</li>
                  <li>✅ 전문가가 헤지 관리</li>
                  <li>✅ 소액 투자 가능</li>
                  <li>✅ 유동성 높음</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-3">단점</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>❌ 헤지 비용 내재</li>
                  <li>❌ 추적 오차 발생</li>
                  <li>❌ 제한적 상품군</li>
                  <li>❌ 롤오버 비용</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded">
              <p className="text-sm">
                <strong>대표 상품:</strong> TIGER 미국나스닥100(H), KODEX 미국S&P500(H)
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Calculator className="w-8 h-8 text-purple-500" />
          헤지 전략별 비교
        </h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">Natural Hedge vs Financial Hedge</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-4">
              <h4 className="font-medium mb-3">Natural Hedge (자연 헤지)</h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 수입/지출 통화 매칭</li>
                <li>• 다통화 분산 투자</li>
                <li>• 현지 차입을 통한 투자</li>
                <li>• 비용 없음, 구조적 헤지</li>
              </ul>
            </div>
            
            <div className="bg-green-50 dark:bg-green-900/20 rounded p-4">
              <h4 className="font-medium mb-3">Financial Hedge (금융 헤지)</h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 파생상품 활용</li>
                <li>• 정확한 헤지 비율 설정</li>
                <li>• 비용 발생</li>
                <li>• 유연한 조정 가능</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 mt-6">
          <h3 className="text-lg font-semibold mb-4">헤지 비율 결정</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-white dark:bg-gray-700 rounded">
              <span>0% 헤지 (No Hedge)</span>
              <span className="text-sm text-gray-600 dark:text-gray-400">환율 위험 100% 노출</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-white dark:bg-gray-700 rounded">
              <span>50% 부분 헤지</span>
              <span className="text-sm text-gray-600 dark:text-gray-400">위험과 기회의 균형</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-white dark:bg-gray-700 rounded">
              <span>100% 완전 헤지</span>
              <span className="text-sm text-gray-600 dark:text-gray-400">환율 변동 중립화</span>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-yellow-100 dark:bg-yellow-900/20 rounded">
            <p className="text-sm">
              💡 <strong>최적 헤지 비율 = 자산 상관계수 × 변동성 비율</strong>
            </p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <AlertTriangle className="w-8 h-8 text-orange-500" />
          신흥국 통화 헤지
        </h2>
        
        <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">NDF (Non-Deliverable Forward)</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3">특징</h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 실물 인도 없이 차액 정산</li>
                <li>• 자본 통제 국가 통화 헤지</li>
                <li>• USD로 정산</li>
                <li>• 높은 스프레드</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium mb-3">주요 NDF 통화</h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>🇨🇳 CNY (중국 위안)</li>
                <li>🇮🇳 INR (인도 루피)</li>
                <li>🇧🇷 BRL (브라질 헤알)</li>
                <li>🇰🇷 KRW (한국 원)</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded">
            <p className="text-sm">
              <strong>주의:</strong> 신흥국 통화는 변동성이 크고 헤지 비용이 높아 선택적 헤지 권장
            </p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <BarChart3 className="w-8 h-8 text-green-500" />
          실전 헤지 전략 사례
        </h2>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">사례 1: 미국 주식 장기 투자자</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <p className="font-medium mb-2">투자 프로필</p>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 투자금액: 1억원</li>
                  <li>• 투자기간: 5년+</li>
                  <li>• 리스크 성향: 중립</li>
                </ul>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-4">
                <p className="font-medium mb-2">추천 전략</p>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 헤지 비율: 30-50%</li>
                  <li>• 수단: 환헤지 ETF + 일부 언헤지</li>
                  <li>• 이유: 장기 환율 중립, 비용 절감</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">사례 2: 글로벌 채권 투자자</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <p className="font-medium mb-2">투자 프로필</p>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 투자금액: 5억원</li>
                  <li>• 투자기간: 3년</li>
                  <li>• 목표: 안정적 수익</li>
                </ul>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 rounded p-4">
                <p className="font-medium mb-2">추천 전략</p>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 헤지 비율: 80-100%</li>
                  <li>• 수단: 선물환 + 통화스왑</li>
                  <li>• 이유: 채권은 환율 민감, 수익 안정성</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">통화 헤지 의사결정 프레임워크</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">🎯 헤지해야 할 때</h3>
            <ul className="space-y-2">
              <li>✓ 단기 투자 (1년 미만)</li>
              <li>✓ 고정수익 자산 (채권)</li>
              <li>✓ 은퇴자금 등 안정성 중요</li>
              <li>✓ 환율 약세 전망 시</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3">🚫 헤지하지 말아야 할 때</h3>
            <ul className="space-y-2">
              <li>✓ 장기 투자 (10년+)</li>
              <li>✓ 성장주 위주 투자</li>
              <li>✓ 적립식 투자</li>
              <li>✓ 환율 강세 전망 시</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-white/20 rounded">
          <p className="text-center">
            💡 헤지는 보험이다. 비용 대비 효익을 항상 고려하라!
          </p>
        </div>
      </div>
    </div>
  );
}