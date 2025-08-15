'use client';

import { useState } from 'react';

export default function Chapter4() {
  const [portfolioType, setPortfolioType] = useState('conservative');
  
  const portfolioExamples = {
    conservative: {
      name: '안정형 포트폴리오',
      allocation: [
        { asset: '국내 채권', percent: 40, color: '#3b82f6' },
        { asset: '해외 채권', percent: 20, color: '#60a5fa' },
        { asset: '국내 주식', percent: 20, color: '#ef4444' },
        { asset: '해외 주식', percent: 10, color: '#f87171' },
        { asset: '현금/MMF', percent: 10, color: '#10b981' }
      ],
      expectedReturn: '연 4-6%',
      risk: '낮음',
      suitable: '은퇴 준비자, 안정 추구형'
    },
    balanced: {
      name: '균형형 포트폴리오',
      allocation: [
        { asset: '국내 주식', percent: 30, color: '#ef4444' },
        { asset: '해외 주식', percent: 25, color: '#f87171' },
        { asset: '국내 채권', percent: 20, color: '#3b82f6' },
        { asset: '해외 채권', percent: 15, color: '#60a5fa' },
        { asset: '대체투자', percent: 10, color: '#8b5cf6' }
      ],
      expectedReturn: '연 6-10%',
      risk: '중간',
      suitable: '30-40대, 장기 투자자'
    },
    aggressive: {
      name: '공격형 포트폴리오',
      allocation: [
        { asset: '국내 주식', percent: 40, color: '#ef4444' },
        { asset: '해외 주식', percent: 35, color: '#f87171' },
        { asset: '신흥국 주식', percent: 15, color: '#dc2626' },
        { asset: '대체투자', percent: 10, color: '#8b5cf6' }
      ],
      expectedReturn: '연 10-15%',
      risk: '높음',
      suitable: '20-30대, 위험 감수형'
    }
  };

  const currentPortfolio = portfolioExamples[portfolioType];

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">포트폴리오 구성의 기본 📊</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          성공적인 투자의 핵심은 분산투자입니다. 
          자신의 투자 목표와 위험 성향에 맞는 포트폴리오를 구성하는 방법을 배워봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 포트폴리오 예시</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
          {/* 포트폴리오 타입 선택 버튼 */}
          <div className="flex flex-wrap gap-3 mb-6 justify-center">
            <button
              onClick={() => setPortfolioType('conservative')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                portfolioType === 'conservative'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              안정형
            </button>
            <button
              onClick={() => setPortfolioType('balanced')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                portfolioType === 'balanced'
                  ? 'bg-green-500 text-white'
                  : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              균형형
            </button>
            <button
              onClick={() => setPortfolioType('aggressive')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                portfolioType === 'aggressive'
                  ? 'bg-red-500 text-white'
                  : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              공격형
            </button>
          </div>

          {/* 포트폴리오 차트 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h3 className="font-semibold text-xl mb-4">{currentPortfolio.name}</h3>
            
            <div className="grid md:grid-cols-2 gap-8">
              {/* 파이 차트 (간단한 시각화) */}
              <div className="flex items-center justify-center">
                <div className="relative w-48 h-48">
                  <svg viewBox="0 0 42 42" className="w-full h-full">
                    {currentPortfolio.allocation.reduce((acc, item, index) => {
                      const offset = acc.offset;
                      const dashArray = `${item.percent} ${100 - item.percent}`;
                      
                      acc.elements.push(
                        <circle
                          key={index}
                          cx="21"
                          cy="21"
                          r="15.915"
                          fill="transparent"
                          stroke={item.color}
                          strokeWidth="3"
                          strokeDasharray={dashArray}
                          strokeDashoffset={-offset}
                          transform="rotate(-90 21 21)"
                        />
                      );
                      
                      acc.offset += item.percent;
                      return acc;
                    }, { elements: [], offset: 0 }).elements}
                  </svg>
                </div>
              </div>
              
              {/* 자산 배분 상세 */}
              <div className="space-y-3">
                {currentPortfolio.allocation.map((item, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-4 h-4 rounded" 
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm font-medium">{item.asset}</span>
                    </div>
                    <span className="text-sm font-bold">{item.percent}%</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">예상 수익률</p>
                <p className="text-lg font-bold mt-1">{currentPortfolio.expectedReturn}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">위험도</p>
                <p className="text-lg font-bold mt-1">{currentPortfolio.risk}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">적합한 투자자</p>
                <p className="text-sm font-medium mt-1">{currentPortfolio.suitable}</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔑 분산투자의 핵심 원칙</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">자산군 분산</h3>
            <ul className="space-y-2 text-sm">
              <li>• 주식, 채권, 원자재, 부동산 등</li>
              <li>• 서로 다른 특성의 자산 보유</li>
              <li>• 경제 상황별 대응력 향상</li>
              <li>• 전체 포트폴리오 변동성 감소</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">지역 분산</h3>
            <ul className="space-y-2 text-sm">
              <li>• 국내, 선진국, 신흥국 균형 배치</li>
              <li>• 환율 리스크 헤지 효과</li>
              <li>• 글로벌 성장 기회 포착</li>
              <li>• 특정 국가 리스크 완화</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">섹터 분산</h3>
            <ul className="space-y-2 text-sm">
              <li>• IT, 금융, 헬스케어, 소비재 등</li>
              <li>• 산업별 사이클 차이 활용</li>
              <li>• 구조적 성장 트렌드 포착</li>
              <li>• 섹터 로테이션 대응</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">시간 분산</h3>
            <ul className="space-y-2 text-sm">
              <li>• 적립식 투자로 평균 매수가 낮춤</li>
              <li>• 시장 타이밍 리스크 회피</li>
              <li>• 장기적 복리 효과 극대화</li>
              <li>• 심리적 부담 완화</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📐 나이별 자산 배분 가이드</h2>
        <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">100 - 나이 법칙</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            전통적으로 주식 비중은 (100 - 나이)%로 설정하는 것이 안전하다고 알려져 있습니다.
            하지만 최근에는 기대수명 증가로 (120 - 나이)% 법칙도 고려됩니다.
          </p>
          
          <div className="grid md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
              <h4 className="font-semibold text-lg mb-2">20대</h4>
              <div className="text-3xl font-bold text-green-600 mb-1">80%</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">주식 비중</p>
              <p className="text-xs mt-2">공격적 성장 추구</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
              <h4 className="font-semibold text-lg mb-2">30대</h4>
              <div className="text-3xl font-bold text-blue-600 mb-1">70%</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">주식 비중</p>
              <p className="text-xs mt-2">성장과 안정 균형</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
              <h4 className="font-semibold text-lg mb-2">40대</h4>
              <div className="text-3xl font-bold text-purple-600 mb-1">60%</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">주식 비중</p>
              <p className="text-xs mt-2">안정성 비중 증가</p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
              <h4 className="font-semibold text-lg mb-2">50대+</h4>
              <div className="text-3xl font-bold text-red-600 mb-1">50%</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">주식 비중</p>
              <p className="text-xs mt-2">원금 보존 중시</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔄 리밸런싱 전략</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-4">
            정기적인 포트폴리오 재조정
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            시장 변동으로 인해 초기 설정한 자산 비중이 변하게 됩니다. 
            정기적인 리밸런싱으로 목표 비중을 유지하세요.
          </p>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-3">리밸런싱 예시</h4>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2">자산</th>
                  <th className="text-center py-2">목표 비중</th>
                  <th className="text-center py-2">현재 비중</th>
                  <th className="text-center py-2">조정 필요</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-2">국내 주식</td>
                  <td className="text-center">30%</td>
                  <td className="text-center text-red-600">35%</td>
                  <td className="text-center">-5%</td>
                </tr>
                <tr className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-2">해외 주식</td>
                  <td className="text-center">25%</td>
                  <td className="text-center text-green-600">20%</td>
                  <td className="text-center">+5%</td>
                </tr>
                <tr>
                  <td className="py-2">채권</td>
                  <td className="text-center">35%</td>
                  <td className="text-center">35%</td>
                  <td className="text-center">0%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 실전 팁</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <ul className="space-y-3">
            <li className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span>처음에는 간단하게 시작하세요. 국내 주식 ETF + 채권 ETF로도 충분합니다.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span>비용이 낮은 인덱스 펀드나 ETF를 활용하세요.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span>감정에 휘둘리지 말고 계획한 비중을 유지하세요.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span>최소 분기에 한 번은 포트폴리오를 점검하세요.</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}