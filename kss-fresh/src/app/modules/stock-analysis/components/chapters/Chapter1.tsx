'use client';

import React from 'react';

export default function Chapter1() {
  return (
    <div className="max-w-4xl mx-auto">
      {/* Chapter Title */}
      <div className="mb-12">
        <h1 className="text-4xl font-bold mb-4">
          글로벌 금융시장의 이해
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          전 세계 주식시장의 구조, 참여자, 거래 시스템을 완전히 파악하고 투자의 기초를 확립합니다
        </p>
      </div>

      {/* Learning Objectives */}
      <section className="mb-16">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-4">학습 목표</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="text-blue-600 dark:text-blue-400 mt-1">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  글로벌 주요 증권거래소(NYSE, NASDAQ, 런던, 도쿄, 홍콩)의 특징과 차이점 파악
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-blue-600 dark:text-blue-400 mt-1">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  시장 참여자별 역할과 영향력 (기관투자자, 개인투자자, HFT) 이해
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-blue-600 dark:text-blue-400 mt-1">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  주문 유형과 체결 시스템 완전 마스터 (시장가, 지정가, 조건부 주문)
                </p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="text-blue-600 dark:text-blue-400 mt-1">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  IPO, 증자, 액면분할 등 기업 행위가 주가에 미치는 영향 분석
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-blue-600 dark:text-blue-400 mt-1">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  공매도 메커니즘과 시장에 미치는 영향 이해
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-blue-600 dark:text-blue-400 mt-1">✓</span>
                <p className="text-gray-700 dark:text-gray-300">
                  Level 2 호가창 분석을 통한 매수/매도 압력 판단 능력 습득
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 1: Global Stock Exchanges */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-8">1. 글로벌 증권거래소의 이해</h2>
        
        <div className="grid gap-6">
          {/* NYSE */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-blue-900 dark:text-blue-100">
                🇺🇸 NYSE (New York Stock Exchange)
              </h3>
              <span className="text-sm text-gray-500">설립: 1792년</span>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>시가총액:</strong> $25조+ (세계 1위)
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>상장기업 수:</strong> 약 2,800개
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>거래시간:</strong> 09:30-16:00 EST (한국시간 23:30-06:00)
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>특징:</strong> 전통적인 대형 우량주 중심
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>대표기업:</strong> Berkshire Hathaway, JP Morgan, Coca-Cola, Disney
                </p>
              </div>
            </div>
          </div>

          {/* NASDAQ */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-purple-900 dark:text-purple-100">
                🇺🇸 NASDAQ
              </h3>
              <span className="text-sm text-gray-500">설립: 1971년</span>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>시가총액:</strong> $20조+ (세계 2위)
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>상장기업 수:</strong> 약 3,700개
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>거래시간:</strong> NYSE와 동일
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>특징:</strong> 기술주 중심, 전자거래 선도
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>대표기업:</strong> Apple, Microsoft, Amazon, Google, Tesla
                </p>
              </div>
            </div>
          </div>

          {/* Tokyo Stock Exchange */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-red-900 dark:text-red-100">
                🇯🇵 도쿄증권거래소 (TSE)
              </h3>
              <span className="text-sm text-gray-500">설립: 1878년</span>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>시가총액:</strong> $6조+ (아시아 1위)
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>상장기업 수:</strong> 약 3,900개
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>거래시간:</strong> 09:00-15:00 JST (한국과 동일)
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>특징:</strong> 제조업 강국의 특성 반영
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>대표기업:</strong> Toyota, Sony, SoftBank, Nintendo
                </p>
              </div>
            </div>
          </div>

          {/* Hong Kong */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-green-900 dark:text-green-100">
                🇭🇰 홍콩거래소 (HKEX)
              </h3>
              <span className="text-sm text-gray-500">설립: 1891년</span>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>시가총액:</strong> $4조+ (아시아 3위)
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>상장기업 수:</strong> 약 2,600개
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>거래시간:</strong> 09:30-16:00 HKT
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>특징:</strong> 중국 기업의 해외 상장 창구
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>대표기업:</strong> Tencent, Alibaba, HSBC, AIA
                </p>
              </div>
            </div>
          </div>

          {/* London Stock Exchange */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-indigo-900 dark:text-indigo-100">
                🇬🇧 런던증권거래소 (LSE)
              </h3>
              <span className="text-sm text-gray-500">설립: 1698년</span>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>시가총액:</strong> $3.7조+ (유럽 1위)
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>상장기업 수:</strong> 약 2,000개
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>거래시간:</strong> 08:00-16:30 GMT (한국시간 17:00-01:30)
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>특징:</strong> 국제적 금융 허브, 원자재/에너지 기업 다수
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>대표기업:</strong> Shell, HSBC, BP, Unilever
                </p>
              </div>
            </div>
          </div>

          {/* Korea Exchange */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border-2 border-blue-500">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-blue-600 dark:text-blue-400">
                🇰🇷 한국거래소 (KRX)
              </h3>
              <span className="text-sm text-gray-500">설립: 2005년 (통합)</span>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>시가총액:</strong> $2조+ (세계 11위)
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>상장기업 수:</strong> KOSPI 800개, KOSDAQ 1,500개
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>거래시간:</strong> 09:00-15:30 KST
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <strong>특징:</strong> IT/반도체 강국, 개인투자자 비중 높음
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>대표기업:</strong> Samsung Electronics, SK Hynix, Naver, Kakao
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: Market Participants */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-8">2. 시장 참여자 분석</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Institutional Investors */}
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-green-800 dark:text-green-200 mb-4">
              🏢 기관투자자 (Institutional Investors)
            </h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-green-700 dark:text-green-300">연기금 (Pension Funds)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  국민연금(NPS), 캘리포니아 공무원연금(CalPERS) 등 거대 자금 운용
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-green-700 dark:text-green-300">자산운용사</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  BlackRock ($10조 운용), Vanguard, Fidelity 등 글로벌 펀드
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-green-700 dark:text-green-300">헤지펀드</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Bridgewater, Renaissance Technologies 등 절대수익 추구
                </p>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3 mt-4">
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  <strong>특징:</strong> 장기투자, 대량매매, 의결권 행사, ESG 투자 중시
                </p>
              </div>
            </div>
          </div>

          {/* Foreign Investors */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-blue-800 dark:text-blue-200 mb-4">
              🌍 외국인투자자 (Foreign Investors)
            </h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-blue-700 dark:text-blue-300">국부펀드 (SWF)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  노르웨이 정부연금기금($1.4조), 싱가포르 GIC, 중동 오일머니
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-blue-700 dark:text-blue-300">글로벌 뮤추얼펀드</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  신흥시장 펀드, 아시아 전문 펀드 등 지역/섹터별 특화
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-blue-700 dark:text-blue-300">ETF 운용사</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  MSCI Korea ETF, iShares 등 패시브 투자 주도
                </p>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3 mt-4">
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  <strong>특징:</strong> 환율 헤지, 국가 리스크 민감, 유동성 중시
                </p>
              </div>
            </div>
          </div>

          {/* Individual Investors */}
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-purple-800 dark:text-purple-200 mb-4">
              👥 개인투자자 (Retail Investors)
            </h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-purple-700 dark:text-purple-300">한국 개미투자자</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  거래대금의 70% 차지, 단타 선호, 모바일 트레이딩 활발
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-purple-700 dark:text-purple-300">미국 Robinhood 세대</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  밈 주식 열풍, 옵션 거래 증가, 소셜미디어 영향력
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-purple-700 dark:text-purple-300">슈퍼개미</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  자산 10억원 이상, 전업투자자, 독자적 분석 능력
                </p>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3 mt-4">
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  <strong>특징:</strong> 감정적 거래, 군집행동, 정보 비대칭에 취약
                </p>
              </div>
            </div>
          </div>

          {/* HFT / Algo Trading */}
          <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-red-800 dark:text-red-200 mb-4">
              ⚡ HFT & 알고리즘 트레이딩
            </h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-red-700 dark:text-red-300">초단타 매매 (HFT)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  마이크로초 단위 거래, 일일 거래량의 50%+, 시장 유동성 공급
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-red-700 dark:text-red-300">퀀트 펀드</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  수학/통계 모델 기반, AI/ML 활용, 시장 중립 전략
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-red-700 dark:text-red-300">마켓메이커</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  호가 스프레드 수익, 유동성 공급 의무, 리스크 관리 중시
                </p>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-3 mt-4">
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  <strong>특징:</strong> 초고속 거래, 작은 마진×대량거래, 기술력 경쟁
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: Corporate Actions */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-8">3. 기업 행위와 주가 영향</h2>
        
        <div className="space-y-6">
          {/* IPO */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold mb-4">📈 IPO (Initial Public Offering)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">프로세스</h4>
                <ol className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>1. 주관사 선정 및 실사</li>
                  <li>2. 증권신고서 제출</li>
                  <li>3. 수요예측 (기관투자자)</li>
                  <li>4. 공모가 결정</li>
                  <li>5. 청약 (일반투자자)</li>
                  <li>6. 배정 및 상장</li>
                </ol>
              </div>
              <div>
                <h4 className="font-semibold mb-2">투자 포인트</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 공모가 할인율 분석</li>
                  <li>• 청약 경쟁률 확인</li>
                  <li>• 보호예수 물량 파악</li>
                  <li>• 상장 후 수급 예측</li>
                  <li>• 동종업계 밸류에이션 비교</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-xs text-blue-800 dark:text-blue-200">
                <strong>최근 사례:</strong> 크래프톤 IPO (2021) - 공모가 498,000원 → 상장 첫날 755,000원 (+51.6%)
              </p>
            </div>
          </div>

          {/* Stock Split */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold mb-4">✂️ 액면분할 (Stock Split)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">목적</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 주가 접근성 향상</li>
                  <li>• 유동성 증대</li>
                  <li>• 개인투자자 유입 촉진</li>
                  <li>• 거래 활성화</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">영향</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 주가는 분할비율만큼 하락</li>
                  <li>• 주식수는 분할비율만큼 증가</li>
                  <li>• 시가총액은 불변</li>
                  <li>• 심리적 매수세 유입 가능</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <p className="text-xs text-green-800 dark:text-green-200">
                <strong>예시:</strong> 삼성전자 50:1 액면분할 (2018) - 266만원 → 5.32만원, 거래량 20배 증가
              </p>
            </div>
          </div>

          {/* Capital Increase */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold mb-4">💰 유상증자 (Rights Offering)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">유형</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 주주배정: 기존 주주 우선</li>
                  <li>• 일반공모: 불특정 다수</li>
                  <li>• 제3자배정: 특정인 대상</li>
                  <li>• 전환사채(CB): 채권→주식 전환</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">주가 영향</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 단기: 희석 우려로 하락</li>
                  <li>• 중기: 자금 용도에 따라 상이</li>
                  <li>• 할인율이 클수록 하락폭 증가</li>
                  <li>• 신주 상장시 수급 압력</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <p className="text-xs text-red-800 dark:text-red-200">
                <strong>주의:</strong> 유상증자 발표 → 주가 10-20% 급락 가능 → 권리락 후 추가 하락 주의
              </p>
            </div>
          </div>

          {/* Short Selling */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold mb-4">📉 공매도 (Short Selling)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">메커니즘</h4>
                <ol className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>1. 주식 차입 (증권사/기관)</li>
                  <li>2. 차입 주식 매도</li>
                  <li>3. 주가 하락 대기</li>
                  <li>4. 낮은 가격에 매수</li>
                  <li>5. 차입 주식 상환</li>
                  <li>6. 차익 실현</li>
                </ol>
              </div>
              <div>
                <h4 className="font-semibold mb-2">시장 영향</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 가격발견 기능</li>
                  <li>• 거품 제거 효과</li>
                  <li>• 유동성 공급</li>
                  <li>• 과도시 주가 급락 위험</li>
                  <li>• 공매도 과열종목 지정</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <p className="text-xs text-yellow-800 dark:text-yellow-200">
                <strong>규제:</strong> 한국은 개인 공매도 금지, 기관/외국인만 가능, 공매도 잔고 일일 공시
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: Order Types */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-8">4. 주문 유형 마스터하기</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Market Order */}
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-blue-800 dark:text-blue-200 mb-3">
              시장가 주문 (Market Order)
            </h3>
            <div className="space-y-2 text-sm">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>정의:</strong> 가격 조건 없이 즉시 체결을 원하는 주문
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>장점:</strong> 100% 체결 보장, 빠른 진입/청산
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>단점:</strong> 체결가격 불확실, 슬리피지 발생
              </p>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded p-2 mt-3">
                <p className="text-xs">적합한 상황: 급등/급락시 빠른 대응, 유동성 풍부한 종목</p>
              </div>
            </div>
          </div>

          {/* Limit Order */}
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-green-800 dark:text-green-200 mb-3">
              지정가 주문 (Limit Order)
            </h3>
            <div className="space-y-2 text-sm">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>정의:</strong> 특정 가격 이하 매수/이상 매도 주문
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>장점:</strong> 체결가격 확실, 계획적 매매
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>단점:</strong> 미체결 위험, 기회 상실 가능
              </p>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded p-2 mt-3">
                <p className="text-xs">적합한 상황: 목표가 명확, 변동성 낮은 시장</p>
              </div>
            </div>
          </div>

          {/* Stop Loss */}
          <div className="bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-red-800 dark:text-red-200 mb-3">
              손절매 주문 (Stop Loss)
            </h3>
            <div className="space-y-2 text-sm">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>정의:</strong> 특정 가격 도달시 시장가 매도 전환
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>장점:</strong> 손실 제한, 리스크 관리
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>단점:</strong> 일시적 하락에도 청산
              </p>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded p-2 mt-3">
                <p className="text-xs">설정 기준: 매수가 -3~7%, 지지선 하단</p>
              </div>
            </div>
          </div>

          {/* IOC/FOK */}
          <div className="bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-800 dark:text-purple-200 mb-3">
              조건부 주문 (IOC/FOK)
            </h3>
            <div className="space-y-2 text-sm">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>IOC:</strong> 즉시 체결 가능한 수량만 체결 후 잔량 취소
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>FOK:</strong> 전량 체결 또는 전량 취소
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>활용:</strong> 대량 매매시 시장 충격 최소화
              </p>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded p-2 mt-3">
                <p className="text-xs">기관/외국인 선호, 알고리즘 매매 활용</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Simulator Connection */}
      <section className="mb-16">
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-4">🎮 실전 시뮬레이터로 연습하기</h2>
          <p className="mb-6">
            학습한 내용을 바로 적용해볼 수 있는 인터랙티브 시뮬레이터를 준비했습니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <a href="/modules/stock-analysis/simulators/order-book-simulator" 
               className="bg-white/20 hover:bg-white/30 rounded-lg p-4 transition-colors">
              <h3 className="font-bold mb-2">📊 호가창 시뮬레이터</h3>
              <p className="text-sm">실시간 호가 변화와 체결 과정을 체험</p>
            </a>
            <a href="/modules/stock-analysis/simulators/trading-simulator" 
               className="bg-white/20 hover:bg-white/30 rounded-lg p-4 transition-colors">
              <h3 className="font-bold mb-2">💹 모의투자 시스템</h3>
              <p className="text-sm">가상자금으로 실전 매매 연습</p>
            </a>
          </div>
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-16">
        <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6">📌 핵심 정리</h2>
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">1</span>
              <div>
                <h3 className="font-semibold mb-1">글로벌 시장은 24시간 연결되어 있다</h3>
                <p className="text-slate-200">
                  미국 시장 마감 → 아시아 시장 개장 → 유럽 시장으로 이어지는 글로벌 자금 흐름 이해
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">2</span>
              <div>
                <h3 className="font-semibold mb-1">각 시장 참여자의 특성과 전략이 다르다</h3>
                <p className="text-slate-200">
                  기관은 장기투자, 외국인은 환율 헤지, 개인은 단타 위주 - 각자의 행동 패턴 파악
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">3</span>
              <div>
                <h3 className="font-semibold mb-1">기업 행위가 주가에 미치는 영향 분석</h3>
                <p className="text-slate-200">
                  IPO, 유상증자, 액면분할, 공매도 등의 메커니즘과 투자 기회 포착
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Next Chapter Preview */}
      <section>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-2">다음 챕터 미리보기</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            <strong>Chapter 2: 재무제표 해부학</strong> - 
            삼성전자 2024년 3분기 실적발표를 실시간으로 분석하며 
            재무제표 읽는 법을 마스터합니다.
          </p>
        </div>
      </section>
    </div>
  );
}