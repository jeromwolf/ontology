'use client';

import React from 'react';
import { Globe, TrendingUp, Cpu, Heart, Building, Zap, ShoppingCart, Wifi, Factory, Fuel, DollarSign } from 'lucide-react';

export default function Chapter35() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">글로벌 섹터 이해</h1>
      
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Globe className="w-8 h-8 text-blue-500" />
          GICS (Global Industry Classification Standard)
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <p className="text-lg mb-4">
            GICS는 MSCI와 S&P가 공동 개발한 글로벌 산업 분류 기준으로, 
            전 세계 주식을 11개 섹터, 24개 산업군, 69개 산업, 158개 세부산업으로 분류합니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h3 className="font-semibold mb-2">섹터 분류의 중요성</h3>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>포트폴리오 분산투자 기준</li>
                <li>섹터 로테이션 전략 수립</li>
                <li>상대 성과 비교 분석</li>
                <li>리스크 관리 도구</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h3 className="font-semibold mb-2">2018년 개편 주요 변화</h3>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>통신 서비스 섹터 신설</li>
                <li>IT 섹터에서 일부 이동</li>
                <li>소비재 섹터 재편성</li>
                <li>REIT 섹터 독립</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6">11개 GICS 섹터 상세 분석</h2>
        
        <div className="space-y-6">
          {/* Information Technology */}
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Cpu className="w-6 h-6" />
              1. Information Technology (정보기술)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>소프트웨어 & 서비스</li>
                  <li>하드웨어 & 장비</li>
                  <li>반도체 & 반도체 장비</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 Apple, Microsoft, NVIDIA</li>
                  <li>🇰🇷 삼성전자, SK하이닉스</li>
                  <li>🇹🇼 TSMC</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 고성장 잠재력 • 높은 변동성 • R&D 투자 중요 • 경기 민감도 높음
              </p>
            </div>
          </div>

          {/* Healthcare */}
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Heart className="w-6 h-6" />
              2. Health Care (헬스케어)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>제약</li>
                  <li>바이오테크놀로지</li>
                  <li>의료장비 & 서비스</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 Johnson & Johnson, Pfizer</li>
                  <li>🇨🇭 Roche, Novartis</li>
                  <li>🇬🇧 AstraZeneca</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 방어적 성격 • 인구 고령화 수혜 • 규제 리스크 • 장기 R&D 사이클
              </p>
            </div>
          </div>

          {/* Financials */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Building className="w-6 h-6" />
              3. Financials (금융)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>은행</li>
                  <li>보험</li>
                  <li>자본시장</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 JPMorgan, Bank of America</li>
                  <li>🇨🇳 ICBC, China Construction Bank</li>
                  <li>🇯🇵 Mitsubishi UFJ</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 금리 민감도 높음 • 경기 순환적 • 규제 영향 큼 • 배당 수익률 양호
              </p>
            </div>
          </div>

          {/* Consumer Discretionary */}
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <ShoppingCart className="w-6 h-6" />
              4. Consumer Discretionary (경기소비재)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>자동차 & 부품</li>
                  <li>소매 (전문점)</li>
                  <li>호텔, 레스토랑 & 레저</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 Amazon, Tesla, Home Depot</li>
                  <li>🇩🇪 Mercedes-Benz, BMW</li>
                  <li>🇯🇵 Toyota, Sony</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 경기 민감도 매우 높음 • 소비자 신뢰도 중요 • 브랜드 가치 • 온라인 전환
              </p>
            </div>
          </div>

          {/* Communication Services */}
          <div className="bg-cyan-50 dark:bg-cyan-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Wifi className="w-6 h-6" />
              5. Communication Services (통신서비스)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>통신 서비스</li>
                  <li>미디어 & 엔터테인먼트</li>
                  <li>인터랙티브 미디어</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 Meta, Alphabet, Netflix</li>
                  <li>🇨🇳 Tencent, Alibaba</li>
                  <li>🇺🇸 AT&T, Verizon</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 구독 수익 모델 • 네트워크 효과 • 콘텐츠 투자 중요 • 규제 이슈
              </p>
            </div>
          </div>

          {/* Industrials */}
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Factory className="w-6 h-6" />
              6. Industrials (산업재)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>항공우주 & 방위</li>
                  <li>기계 & 장비</li>
                  <li>운송</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 Boeing, Caterpillar, UPS</li>
                  <li>🇪🇺 Airbus, Siemens</li>
                  <li>🇯🇵 Hitachi, Mitsubishi Heavy</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 경기 순환적 • 인프라 투자 수혜 • 글로벌 무역 영향 • 자동화 트렌드
              </p>
            </div>
          </div>

          {/* Consumer Staples */}
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <ShoppingCart className="w-6 h-6" />
              7. Consumer Staples (필수소비재)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>식품 & 음료</li>
                  <li>가정용품 & 개인용품</li>
                  <li>식품 유통</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 P&G, Coca-Cola, Walmart</li>
                  <li>🇨🇭 Nestlé</li>
                  <li>🇬🇧 Unilever</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 방어적 성격 강함 • 안정적 수익 • 배당 매력적 • 브랜드 파워 중요
              </p>
            </div>
          </div>

          {/* Energy */}
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Fuel className="w-6 h-6" />
              8. Energy (에너지)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>석유 & 가스 탐사/생산</li>
                  <li>에너지 장비 & 서비스</li>
                  <li>정유 & 마케팅</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 Exxon Mobil, Chevron</li>
                  <li>🇸🇦 Saudi Aramco</li>
                  <li>🇪🇺 Shell, BP, TotalEnergies</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 원자재 가격 직결 • 지정학적 리스크 • ESG 압력 • 에너지 전환 도전
              </p>
            </div>
          </div>

          {/* Utilities */}
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Zap className="w-6 h-6" />
              9. Utilities (유틸리티)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>전력</li>
                  <li>가스</li>
                  <li>복합 유틸리티</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 NextEra Energy, Dominion</li>
                  <li>🇪🇸 Iberdrola</li>
                  <li>🇮🇹 Enel</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 극도로 방어적 • 높은 배당 수익률 • 금리 민감 • 재생에너지 전환
              </p>
            </div>
          </div>

          {/* Real Estate */}
          <div className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Building className="w-6 h-6" />
              10. Real Estate (부동산)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>부동산 투자신탁 (REITs)</li>
                  <li>부동산 관리 & 개발</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 American Tower, Prologis</li>
                  <li>🇸🇬 CapitaLand</li>
                  <li>🇯🇵 Mitsubishi Estate</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 인플레이션 헤지 • 금리 민감 • 안정적 임대수익 • 지역별 차이 큼
              </p>
            </div>
          </div>

          {/* Materials */}
          <div className="bg-brown-50 dark:bg-brown-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-3">
              <Factory className="w-6 h-6" />
              11. Materials (소재)
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">주요 산업군</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>화학</li>
                  <li>금속 & 광업</li>
                  <li>종이 & 임산물</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">대표 기업</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>🇺🇸 Linde, Air Products</li>
                  <li>🇦🇺 BHP, Rio Tinto</li>
                  <li>🇩🇪 BASF</li>
                </ul>
              </div>
            </div>
            <div className="mt-4">
              <h4 className="font-medium mb-2">섹터 특성</h4>
              <p className="text-gray-700 dark:text-gray-300">
                • 경기 순환적 • 원자재 가격 연동 • 중국 수요 영향 • ESG 이슈
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <TrendingUp className="w-8 h-8 text-green-500" />
          섹터 로테이션 전략
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">경제 사이클별 섹터 성과</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium text-green-600 mb-2">🌱 경기 회복기 (Early Cycle)</h4>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>선호 섹터:</strong> 경기소비재, 금융, 산업재<br/>
                <strong>특징:</strong> 금리 인하, 신용 확대, 소비 심리 개선
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium text-blue-600 mb-2">🚀 경기 확장기 (Mid Cycle)</h4>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>선호 섹터:</strong> IT, 산업재, 에너지<br/>
                <strong>특징:</strong> 기업 이익 증가, 고용 확대, 투자 증가
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium text-orange-600 mb-2">🌅 경기 성숙기 (Late Cycle)</h4>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>선호 섹터:</strong> 에너지, 소재, 필수소비재<br/>
                <strong>특징:</strong> 인플레이션 상승, 금리 인상, 성장 둔화
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium text-red-600 mb-2">📉 경기 후퇴기 (Recession)</h4>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>선호 섹터:</strong> 필수소비재, 헬스케어, 유틸리티<br/>
                <strong>특징:</strong> 방어적 투자, 안전자산 선호, 배당주 선호
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6">신흥 섹터 및 테마</h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-green-100 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-3">🌿 클린 에너지</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>태양광: ENPH, SEDG</li>
              <li>풍력: VWDRY, NEE</li>
              <li>전기차: TSLA, NIO, RIVN</li>
              <li>배터리: CATL, LG에너지솔루션</li>
            </ul>
          </div>
          
          <div className="bg-blue-100 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-3">🚀 우주 항공</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>발사체: SpaceX (비상장)</li>
              <li>위성: IRDM, MAXR</li>
              <li>우주관광: SPCE</li>
              <li>방위산업: LMT, NOC</li>
            </ul>
          </div>
          
          <div className="bg-purple-100 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-3">🔒 사이버 보안</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>엔드포인트: CRWD, S</li>
              <li>네트워크: PANW, FTNT</li>
              <li>클라우드: ZS, OKTA</li>
              <li>ID 관리: CYBR, PING</li>
            </ul>
          </div>
        </div>
      </section>

      <div className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">섹터 분석 체크리스트</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">📊 정량 분석</h3>
            <ul className="space-y-2">
              <li>✓ 섹터 PER vs 시장 평균</li>
              <li>✓ 매출 성장률 추이</li>
              <li>✓ 영업이익률 변화</li>
              <li>✓ ROE 및 부채비율</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3">📈 정성 분석</h3>
            <ul className="space-y-2">
              <li>✓ 규제 환경 변화</li>
              <li>✓ 기술 혁신 동향</li>
              <li>✓ 소비자 트렌드</li>
              <li>✓ 경쟁 구도 변화</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}