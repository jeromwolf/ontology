'use client';

import React from 'react';
import { Globe, PieChart, TrendingUp, Shield, BarChart3, AlertTriangle, Target, Building } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter39() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">국제 분산투자</h1>
      
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Globe className="w-8 h-8 text-blue-500" />
          국제 분산투자의 이론적 근거
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">현대 포트폴리오 이론의 확장</h3>
          <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
            <p className="font-mono text-sm mb-2">
              포트폴리오 위험 = √(w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρ₁₂σ₁σ₂)
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              ρ₁₂ (상관계수)가 낮을수록 분산투자 효과 증대
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
              <p className="text-2xl font-bold text-blue-600 mb-1">0.65</p>
              <p className="text-sm">국내 주식 간 평균 상관계수</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
              <p className="text-2xl font-bold text-green-600 mb-1">0.42</p>
              <p className="text-sm">국가 간 주식 평균 상관계수</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
              <p className="text-2xl font-bold text-purple-600 mb-1">35%</p>
              <p className="text-sm">리스크 감소 효과</p>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-3">분산투자의 실증적 효과</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-green-500 text-xl">📈</span>
              <div>
                <p className="font-medium">수익률 개선</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  1970-2023년 글로벌 분산 포트폴리오가 단일 국가 대비 연평균 1.8% 초과 수익
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-blue-500 text-xl">📉</span>
              <div>
                <p className="font-medium">변동성 감소</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  표준편차 22% → 14%로 36% 감소 (20개국 균등 분산 시)
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-purple-500 text-xl">🛡️</span>
              <div>
                <p className="font-medium">최대 낙폭 완화</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  단일 국가 -55% vs 글로벌 분산 -38% (2008년 금융위기)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <PieChart className="w-8 h-8 text-green-500" />
          최적 국제 분산투자 비중
        </h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold mb-4">시가총액 가중 vs 최적화 모델</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3">글로벌 시가총액 비중 (2024)</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-blue-500 rounded"></span>
                    미국
                  </span>
                  <span className="font-medium">42.5%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-red-500 rounded"></span>
                    일본
                  </span>
                  <span className="font-medium">6.1%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-green-500 rounded"></span>
                    중국
                  </span>
                  <span className="font-medium">3.2%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-purple-500 rounded"></span>
                    유럽
                  </span>
                  <span className="font-medium">15.3%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-yellow-500 rounded"></span>
                    신흥국
                  </span>
                  <span className="font-medium">12.8%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-gray-500 rounded"></span>
                    기타
                  </span>
                  <span className="font-medium">20.1%</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-3">한국 투자자 권장 비중</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-orange-500 rounded"></span>
                    한국
                  </span>
                  <span className="font-medium">30-50%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-blue-500 rounded"></span>
                    미국
                  </span>
                  <span className="font-medium">25-35%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-purple-500 rounded"></span>
                    선진국 (일본/유럽)
                  </span>
                  <span className="font-medium">10-15%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 bg-yellow-500 rounded"></span>
                    신흥국
                  </span>
                  <span className="font-medium">5-10%</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
            <p className="text-sm">
              💡 <strong>홈 바이어스 완화 전략:</strong> 초기 해외 투자 비중을 20%에서 시작하여 
              경험 축적에 따라 점진적으로 50%까지 확대
            </p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <TrendingUp className="w-8 h-8 text-purple-500" />
          상관관계 변화와 위기 시 대응
        </h2>
        
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4">위기 시 상관관계 증가 현상</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium mb-2">평상시</h4>
              <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                <li>• 미국-유럽: 0.65</li>
                <li>• 미국-아시아: 0.52</li>
                <li>• 유럽-아시아: 0.48</li>
                <li>• 선진국-신흥국: 0.43</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium mb-2">금융위기 시</h4>
              <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                <li>• 미국-유럽: 0.88 (+35%)</li>
                <li>• 미국-아시아: 0.79 (+52%)</li>
                <li>• 유럽-아시아: 0.75 (+56%)</li>
                <li>• 선진국-신흥국: 0.72 (+67%)</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-yellow-100 dark:bg-yellow-900/20 rounded">
            <p className="text-sm">
              ⚠️ <strong>주의:</strong> "When markets fall, correlations go to 1" 
              - 위기 시에는 분산투자 효과가 제한적
            </p>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">위기 대응 전략</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-green-500">✓</span>
              <div>
                <p className="font-medium">안전자산 포함</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  금, 미국 국채, 일본 엔화 등 전통적 안전자산 10-20% 보유
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-green-500">✓</span>
              <div>
                <p className="font-medium">저상관 자산 발굴</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  대체투자(리츠, 인프라, 원자재) 활용으로 포트폴리오 강건성 제고
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-green-500">✓</span>
              <div>
                <p className="font-medium">동적 자산배분</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  시장 변동성 증가 시 위험자산 비중 자동 축소 메커니즘
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Building className="w-8 h-8 text-orange-500" />
          해외 시장 접근 방법
        </h2>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-semibold mb-4">1. ADR/GDR (예탁증권)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">장점</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>✅ 달러로 거래 (환전 불필요)</li>
                  <li>✅ 미국 규제 적용 (투명성)</li>
                  <li>✅ 배당금 자동 환전</li>
                  <li>✅ 친숙한 거래 환경</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-3">단점</h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>❌ 제한적 종목 수</li>
                  <li>❌ 유동성 부족 가능</li>
                  <li>❌ 현지 주가와 괴리</li>
                  <li>❌ 추가 수수료 가능</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-sm">
                <strong>대표 ADR:</strong> TSM (대만), BABA (중국), SAP (독일), SONY (일본)
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-semibold mb-4">2. 해외 ETF</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">국가별 ETF</h4>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• SPY/VOO - 미국 S&P 500</li>
                  <li>• EWJ - 일본 MSCI</li>
                  <li>• FXI - 중국 대형주</li>
                  <li>• EWG - 독일 MSCI</li>
                  <li>• EEM - 신흥국 전체</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-3">글로벌/지역 ETF</h4>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• VT - 전세계 주식</li>
                  <li>• VXUS - 미국 제외 전세계</li>
                  <li>• VGK - 유럽 주식</li>
                  <li>• VWO - 신흥국 주식</li>
                  <li>• ACWI - MSCI 전세계</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
              <p className="text-sm">
                💡 <strong>팁:</strong> 보수율 0.1% 이하, 일평균 거래량 100만주 이상 ETF 선택
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-semibold mb-4">3. 직접 투자</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium mb-2 text-center">미국 🇺🇸</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• IB, Schwab 계좌</li>
                  <li>• 실시간 호가</li>
                  <li>• 프리/애프터 거래</li>
                </ul>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium mb-2 text-center">일본 🇯🇵</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 라쿠텐, SBI 증권</li>
                  <li>• 단위주 거래</li>
                  <li>• NISA 계좌 가능</li>
                </ul>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium mb-2 text-center">중국 🇨🇳</h4>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 후강통/선강통</li>
                  <li>• A주 제한적 접근</li>
                  <li>• H주 자유 거래</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Shield className="w-8 h-8 text-red-500" />
          국가 리스크 평가와 관리
        </h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4">주요 국가 리스크 요인</h3>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-2">리스크 유형</th>
                  <th className="text-left p-2">평가 지표</th>
                  <th className="text-left p-2">대응 방안</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="p-2 font-medium">정치적 리스크</td>
                  <td className="p-2">
                    <ul className="text-xs space-y-1">
                      <li>• 정권 안정성</li>
                      <li>• 법치주의 수준</li>
                      <li>• 부패 지수</li>
                    </ul>
                  </td>
                  <td className="p-2">
                    <ul className="text-xs space-y-1">
                      <li>• 민주주의 국가 선호</li>
                      <li>• 정치 보험 가입</li>
                      <li>• 분산 투자</li>
                    </ul>
                  </td>
                </tr>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="p-2 font-medium">규제 리스크</td>
                  <td className="p-2">
                    <ul className="text-xs space-y-1">
                      <li>• 외국인 투자 제한</li>
                      <li>• 자본 통제</li>
                      <li>• 세금 정책 변화</li>
                    </ul>
                  </td>
                  <td className="p-2">
                    <ul className="text-xs space-y-1">
                      <li>• ADR/ETF 활용</li>
                      <li>• 현지 파트너십</li>
                      <li>• 규제 모니터링</li>
                    </ul>
                  </td>
                </tr>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="p-2 font-medium">환율 리스크</td>
                  <td className="p-2">
                    <ul className="text-xs space-y-1">
                      <li>• 환율 변동성</li>
                      <li>• 경상수지</li>
                      <li>• 외환보유고</li>
                    </ul>
                  </td>
                  <td className="p-2">
                    <ul className="text-xs space-y-1">
                      <li>• 환헤지 전략</li>
                      <li>• 자연 헤지</li>
                      <li>• 통화 분산</li>
                    </ul>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 mt-6">
          <h3 className="text-lg font-semibold mb-4">국가별 리스크 점수 (2024)</h3>
          <div className="space-y-3">
            {[
              { country: '🇸🇬 싱가포르', score: 92, color: 'bg-green-500' },
              { country: '🇺🇸 미국', score: 85, color: 'bg-green-500' },
              { country: '🇯🇵 일본', score: 82, color: 'bg-green-500' },
              { country: '🇰🇷 한국', score: 78, color: 'bg-blue-500' },
              { country: '🇨🇳 중국', score: 65, color: 'bg-yellow-500' },
              { country: '🇧🇷 브라질', score: 58, color: 'bg-orange-500' },
              { country: '🇹🇷 터키', score: 45, color: 'bg-red-500' }
            ].map((item, idx) => (
              <div key={idx} className="flex items-center justify-between">
                <span className="font-medium">{item.country}</span>
                <div className="flex items-center gap-3 flex-1 ml-4">
                  <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                    <div 
                      className={`h-4 rounded-full ${item.color}`}
                      style={{ width: `${item.score}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium w-10">{item.score}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Target className="w-8 h-8 text-green-500" />
          실전 국제 분산투자 포트폴리오
        </h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">보수적 포트폴리오 (안정 추구형)</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">자산 배분</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 한국 주식/채권: 50%</li>
                  <li>• 미국 주식: 20%</li>
                  <li>• 미국 채권: 15%</li>
                  <li>• 선진국 주식: 10%</li>
                  <li>• 금/원자재: 5%</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">추천 상품</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• KODEX 200</li>
                  <li>• SPY (환헤지)</li>
                  <li>• AGG (미국 채권)</li>
                  <li>• VEA (선진국)</li>
                  <li>• GLD (금)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">균형형 포트폴리오 (성장과 안정)</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">자산 배분</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 한국 주식: 35%</li>
                  <li>• 미국 주식: 30%</li>
                  <li>• 선진국 주식: 15%</li>
                  <li>• 신흥국 주식: 10%</li>
                  <li>• 대체투자: 10%</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">추천 상품</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• TIGER MSCI Korea</li>
                  <li>• QQQ + VOO</li>
                  <li>• VGK (유럽)</li>
                  <li>• EEM (신흥국)</li>
                  <li>• VNQ (리츠)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">공격적 포트폴리오 (고성장 추구)</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">자산 배분</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 한국 성장주: 25%</li>
                  <li>• 미국 기술주: 35%</li>
                  <li>• 아시아 성장주: 20%</li>
                  <li>• 신흥국 주식: 15%</li>
                  <li>• 테마 ETF: 5%</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">추천 상품</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• KODEX 코스닥150</li>
                  <li>• ARKK + QQQ</li>
                  <li>• ASHR (중국 A주)</li>
                  <li>• FM (프론티어)</li>
                  <li>• ICLN (클린에너지)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">국제 분산투자 실행 가이드</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">✅ 단계별 실행 계획</h3>
            <ol className="space-y-2 list-decimal list-inside">
              <li>해외 증권사 계좌 개설</li>
              <li>소액으로 ETF 투자 시작</li>
              <li>점진적 비중 확대 (연 5-10%)</li>
              <li>개별 종목 리서치 병행</li>
              <li>정기적 리밸런싱 (분기별)</li>
            </ol>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3">⚠️ 주의사항</h3>
            <ul className="space-y-2">
              <li>• 과도한 거래 비용 주의</li>
              <li>• 세금 영향 사전 검토</li>
              <li>• 환율 급변동 대비</li>
              <li>• 정보 비대칭 인정</li>
              <li>• 장기 관점 유지</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-white/20 rounded">
          <p className="text-center font-semibold">
            "Don't put all your eggs in one basket, especially if it's your home country basket."
          </p>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 글로벌 증권 플랫폼 & 도구',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Interactive Brokers',
                authors: 'Interactive Brokers Group',
                year: '2025',
                description: '전 세계 150개 시장 접근 가능한 글로벌 온라인 증권사. 저렴한 수수료와 다양한 투자 상품 제공',
                link: 'https://www.interactivebrokers.com/'
              },
              {
                title: 'Charles Schwab International',
                authors: 'Charles Schwab',
                year: '2025',
                description: '미국 증권 계좌 개설 가능. 외국인 투자자를 위한 글로벌 서비스 제공',
                link: 'https://international.schwab.com/'
              },
              {
                title: 'MSCI Index Explorer',
                authors: 'MSCI Inc.',
                year: '2025',
                description: '글로벌 주가지수 데이터 및 국가별 시장 분석. 선진국/신흥국 분류 기준 제공',
                link: 'https://www.msci.com/our-solutions/indexes'
              },
              {
                title: 'ETF.com - Global ETF Database',
                authors: 'ETF.com',
                year: '2025',
                description: '전 세계 ETF 비교 및 분석 도구. 국가별, 자산별 ETF 검색 및 성과 비교',
                link: 'https://www.etf.com/'
              },
              {
                title: 'Vanguard Global Investing',
                authors: 'The Vanguard Group',
                year: '2025',
                description: '저비용 글로벌 인덱스 펀드 및 ETF. VT, VXUS 등 대표적인 국제 분산투자 상품',
                link: 'https://investor.vanguard.com/investment-products/list/all'
              }
            ]
          },
          {
            title: '📖 핵심 연구 & 리포트',
            icon: 'research' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Home Bias in Portfolio Choice',
                authors: 'French, K. R., & Poterba, J. M.',
                year: '1991',
                description: '투자자들이 자국 자산을 과도하게 선호하는 "홈 바이어스" 현상을 처음 규명한 고전적 연구',
                link: 'https://www.jstor.org/stable/2006858'
              },
              {
                title: 'International Diversification: Theory and Evidence',
                authors: 'Solnik, B. H.',
                year: '1974',
                description: '국제 분산투자의 이론적 근거를 제시한 선구적 논문. 상관관계 분석 및 최적 포트폴리오 구성',
                link: 'https://doi.org/10.2307/2326623'
              },
              {
                title: 'Global Investment Returns Yearbook',
                authors: 'Credit Suisse Research Institute',
                year: '2024',
                description: '1900년부터 현재까지 주요 국가 주식시장 수익률 데이터베이스. 장기 성과 비교 분석',
                link: 'https://www.credit-suisse.com/about-us/en/reports-research/global-investment-returns-yearbook.html'
              },
              {
                title: 'The Benefits of International Diversification',
                authors: 'Vanguard Research',
                year: '2023',
                description: '현대 시장 환경에서 국제 분산투자의 효과를 재평가. 상관관계 변화 및 최적 배분 제안',
                link: 'https://corporate.vanguard.com/content/corporatesite/us/en/corp/research.html'
              }
            ]
          },
          {
            title: '🛠️ 실전 리소스 & 가이드',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Currency Hedging Guide',
                authors: 'BlackRock',
                year: '2024',
                description: '환율 리스크 관리 전략. 환헤지 ETF vs 비헤지 ETF 선택 가이드',
                link: 'https://www.blackrock.com/institutions/en-us/insights/portfolio-design/currency-hedging'
              },
              {
                title: '해외주식 투자 세금 가이드',
                authors: '국세청',
                year: '2024',
                description: '해외 주식 양도소득세, 배당소득세 신고 방법. 증권사별 세무 서비스 비교',
                link: 'https://www.nts.go.kr/'
              },
              {
                title: 'Country Risk Assessment',
                authors: 'S&P Global Ratings',
                year: '2025',
                description: '국가별 신용등급 및 리스크 평가. 정치·경제·사회적 안정성 분석',
                link: 'https://www.spglobal.com/ratings/'
              },
              {
                title: 'ETF Screener & Comparison Tool',
                authors: 'Morningstar',
                year: '2025',
                description: '글로벌 ETF 성과 비교, 보수율 분석, 추적오차 평가 도구',
                link: 'https://www.morningstar.com/etfs'
              },
              {
                title: 'Global Market Research',
                authors: 'JP Morgan Asset Management',
                year: '2024',
                description: '지역별 시장 전망, 자산배분 전략, 경제 사이클 분석 리포트',
                link: 'https://am.jpmorgan.com/us/en/asset-management/adv/insights/market-insights/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}