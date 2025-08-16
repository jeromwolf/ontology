'use client';

import React from 'react';
import { Globe, TrendingUp, Building, DollarSign, BarChart3, AlertTriangle, Map, Target } from 'lucide-react';

export default function Chapter38() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">글로벌 매크로 투자</h1>
      
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Globe className="w-8 h-8 text-blue-500" />
          글로벌 매크로 투자란?
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6 mb-6">
          <p className="text-lg mb-4">
            글로벌 매크로 투자는 전 세계 경제, 정치, 사회적 변화를 분석하여 
            국가, 통화, 자산군 수준에서 투자 기회를 포착하는 Top-down 접근법입니다.
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
              <p className="text-3xl mb-2">🌍</p>
              <p className="font-semibold">글로벌 시야</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">국경 없는 투자</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
              <p className="text-3xl mb-2">📊</p>
              <p className="font-semibold">매크로 분석</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">큰 그림 관점</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
              <p className="text-3xl mb-2">🎯</p>
              <p className="font-semibold">다자산 전략</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">주식, 채권, 통화, 원자재</p>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-3">전설적인 매크로 투자자들</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-2xl">👤</span>
              <div>
                <p className="font-semibold">조지 소로스 (George Soros)</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  1992년 파운드화 공매도로 10억 달러 수익 - "영란은행을 무너뜨린 사나이"
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">👤</span>
              <div>
                <p className="font-semibold">레이 달리오 (Ray Dalio)</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Bridgewater Associates 창업자 - 경제 머신의 작동 원리 체계화
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">👤</span>
              <div>
                <p className="font-semibold">스탠리 드러켄밀러 (Stanley Druckenmiller)</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  30년간 연평균 30% 수익률 - 소로스의 수제자
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <TrendingUp className="w-8 h-8 text-green-500" />
          글로벌 경제 사이클 분석
        </h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold mb-4">경제 사이클과 자산 배분</h3>
          
          <div className="space-y-4">
            <div className="bg-green-50 dark:bg-green-900/20 rounded p-4">
              <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">
                🌱 Recovery (회복기)
              </h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium mb-1">경제 상황</p>
                  <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• GDP 성장률 회복</li>
                    <li>• 실업률 감소 시작</li>
                    <li>• 기업 실적 개선</li>
                  </ul>
                </div>
                <div>
                  <p className="text-sm font-medium mb-1">선호 자산</p>
                  <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                    <li>✓ 주식 (특히 경기민감주)</li>
                    <li>✓ 신흥국 자산</li>
                    <li>✓ 하이일드 채권</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-4">
              <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">
                🚀 Expansion (확장기)
              </h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium mb-1">경제 상황</p>
                  <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 고성장 지속</li>
                    <li>• 완전 고용 근접</li>
                    <li>• 인플레이션 상승</li>
                  </ul>
                </div>
                <div>
                  <p className="text-sm font-medium mb-1">선호 자산</p>
                  <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                    <li>✓ 성장주</li>
                    <li>✓ 원자재/에너지</li>
                    <li>✓ 부동산</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-4">
              <h4 className="font-medium text-orange-700 dark:text-orange-300 mb-2">
                ⚡ Slowdown (둔화기)
              </h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium mb-1">경제 상황</p>
                  <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 성장률 둔화</li>
                    <li>• 금리 인상 압력</li>
                    <li>• 기업 마진 압박</li>
                  </ul>
                </div>
                <div>
                  <p className="text-sm font-medium mb-1">선호 자산</p>
                  <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                    <li>✓ 우량 채권</li>
                    <li>✓ 방어주</li>
                    <li>✓ 금</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 rounded p-4">
              <h4 className="font-medium text-red-700 dark:text-red-300 mb-2">
                📉 Recession (침체기)
              </h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium mb-1">경제 상황</p>
                  <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 마이너스 성장</li>
                    <li>• 실업률 급증</li>
                    <li>• 디플레이션 우려</li>
                  </ul>
                </div>
                <div>
                  <p className="text-sm font-medium mb-1">선호 자산</p>
                  <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                    <li>✓ 국채</li>
                    <li>✓ 현금</li>
                    <li>✓ 안전 통화 (USD, JPY)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Building className="w-8 h-8 text-purple-500" />
          주요국 중앙은행 정책 분석
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">🇺🇸 연준 (Fed)</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <div>
                  <p className="font-medium">이중 책무 (Dual Mandate)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    물가안정 + 완전고용
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <div>
                  <p className="font-medium">주요 지표</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    PCE 인플레이션, 실업률, GDP
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <div>
                  <p className="font-medium">정책 도구</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    금리, QE/QT, Forward Guidance
                  </p>
                </div>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">🇪🇺 유럽중앙은행 (ECB)</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <div>
                  <p className="font-medium">단일 책무</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    물가안정 (인플레이션 2% 목표)
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <div>
                  <p className="font-medium">특수성</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    19개국 통화정책 조율
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <div>
                  <p className="font-medium">과제</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    재정정책 분산, 남북 격차
                  </p>
                </div>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">🇯🇵 일본은행 (BOJ)</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <span className="text-red-500">•</span>
                <div>
                  <p className="font-medium">초완화 정책</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    YCC (수익률곡선 통제)
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500">•</span>
                <div>
                  <p className="font-medium">디플레이션 탈출</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    30년 장기 과제
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-500">•</span>
                <div>
                  <p className="font-medium">엔캐리 트레이드</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    글로벌 유동성 공급원
                  </p>
                </div>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">🇨🇳 인민은행 (PBOC)</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <div>
                  <p className="font-medium">다중 목표</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    성장, 고용, 물가, 국제수지
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <div>
                  <p className="font-medium">특수 도구</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    지준율, 창구지도, 위안화 관리
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-500">•</span>
                <div>
                  <p className="font-medium">정책 특징</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    점진주의, 실용주의
                  </p>
                </div>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Map className="w-8 h-8 text-red-500" />
          Top-down 투자 프로세스
        </h2>
        
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4">1단계: 글로벌 매크로 환경 분석</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <h4 className="font-medium mb-2">경제 지표</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• GDP 성장률</li>
                  <li>• 인플레이션</li>
                  <li>• 실업률</li>
                  <li>• PMI 지수</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <h4 className="font-medium mb-2">정책 환경</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 중앙은행 정책</li>
                  <li>• 재정 정책</li>
                  <li>• 규제 변화</li>
                  <li>• 무역 정책</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <h4 className="font-medium mb-2">지정학적 요인</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 국제 관계</li>
                  <li>• 지역 분쟁</li>
                  <li>• 에너지 안보</li>
                  <li>• 공급망 재편</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4">2단계: 국가/지역 선택</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">선진국 vs 신흥국</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 성장 잠재력 비교</li>
                  <li>• 밸류에이션 수준</li>
                  <li>• 정치적 안정성</li>
                  <li>• 통화 강세/약세</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">상대적 매력도</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 경제 성장 모멘텀</li>
                  <li>• 구조적 개혁 진행</li>
                  <li>• 자본 유입/유출</li>
                  <li>• 인구 구조 변화</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-yellow-50 to-green-50 dark:from-yellow-900/20 dark:to-green-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4">3단계: 자산군 배분</h3>
            <div className="grid md:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
                <p className="text-2xl mb-1">📈</p>
                <p className="font-medium">주식</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">성장기 선호</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
                <p className="text-2xl mb-1">📊</p>
                <p className="font-medium">채권</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">둔화기 선호</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
                <p className="text-2xl mb-1">💱</p>
                <p className="font-medium">통화</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">정책 차별화</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4 text-center">
                <p className="text-2xl mb-1">🛢️</p>
                <p className="font-medium">원자재</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">인플레 헤지</p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4">4단계: 포지션 구축 및 리스크 관리</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">포지션 사이징</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 확신도에 따른 가중치</li>
                  <li>• 상관관계 고려</li>
                  <li>• 유동성 확보</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">리스크 관리</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 손절 기준 설정</li>
                  <li>• 헤지 전략 수립</li>
                  <li>• 시나리오 분석</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <AlertTriangle className="w-8 h-8 text-orange-500" />
          지정학적 리스크 분석
        </h2>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">주요 지정학적 이슈</h3>
            
            <div className="space-y-4">
              <div className="border-l-4 border-red-500 pl-4">
                <h4 className="font-medium mb-1">🇺🇸🇨🇳 미중 패권 경쟁</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  기술 패권, 공급망 재편, 금융 디커플링
                </p>
              </div>
              <div className="border-l-4 border-blue-500 pl-4">
                <h4 className="font-medium mb-1">🇷🇺🇺🇦 러시아-우크라이나 전쟁</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  에너지 위기, 식량 안보, NATO 확대
                </p>
              </div>
              <div className="border-l-4 border-green-500 pl-4">
                <h4 className="font-medium mb-1">🌍 기후변화 대응</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  탄소 국경세, 에너지 전환, ESG 규제
                </p>
              </div>
              <div className="border-l-4 border-purple-500 pl-4">
                <h4 className="font-medium mb-1">🏭 공급망 재편</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  프렌드쇼어링, 니어쇼어링, 중복성 확보
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Target className="w-8 h-8 text-green-500" />
          실전 매크로 트레이드 사례
        </h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">사례 1: 2024년 일본 정상화 트레이드</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">투자 논리</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• BOJ 정책 전환 신호</li>
                  <li>• 인플레이션 2% 도달</li>
                  <li>• 임금 상승 가속화</li>
                  <li>• YCC 정책 수정</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">포지션</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>✓ 일본 은행주 매수</li>
                  <li>✓ 일본 국채 공매도</li>
                  <li>✓ USD/JPY 공매도</li>
                  <li>✓ 일본 부동산 REIT</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">사례 2: 신흥국 차별화 전략</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">투자 논리</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 중국 리오프닝 수혜</li>
                  <li>• 원자재 가격 상승</li>
                  <li>• 개혁 진행 국가 선별</li>
                  <li>• 달러 약세 전망</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">포지션</h4>
                <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                  <li>✓ 인도/인도네시아 주식</li>
                  <li>✓ 브라질 헤알 매수</li>
                  <li>✓ 멕시코 채권</li>
                  <li>✓ 터키 회피</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">글로벌 매크로 투자 체크리스트</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">📊 분석 포인트</h3>
            <ul className="space-y-2">
              <li>✓ 글로벌 유동성 사이클</li>
              <li>✓ 달러 강세/약세 사이클</li>
              <li>✓ 금리 차별화 동향</li>
              <li>✓ 상품 가격 사이클</li>
              <li>✓ 신용 스프레드 변화</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3">⚠️ 리스크 요인</h3>
            <ul className="space-y-2">
              <li>✓ 정책 실수 가능성</li>
              <li>✓ 블랙스완 이벤트</li>
              <li>✓ 시장 포지셔닝</li>
              <li>✓ 상관관계 붕괴</li>
              <li>✓ 유동성 고갈</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-white/20 rounded">
          <p className="text-center font-semibold">
            "The big money is not in the buying and selling, but in the waiting." - Charlie Munger
          </p>
        </div>
      </div>
    </div>
  );
}