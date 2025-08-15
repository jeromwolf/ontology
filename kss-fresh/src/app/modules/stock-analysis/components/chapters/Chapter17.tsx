'use client';

import { useState } from 'react';

export default function Chapter17() {
  const [selectedMonth, setSelectedMonth] = useState(0);
  
  const tradingData = [
    { month: '1월', profit: 5.2, trades: 12, winRate: 58 },
    { month: '2월', profit: -2.3, trades: 15, winRate: 40 },
    { month: '3월', profit: 8.7, trades: 18, winRate: 61 },
    { month: '4월', profit: 3.5, trades: 10, winRate: 60 },
    { month: '5월', profit: -1.2, trades: 20, winRate: 45 },
    { month: '6월', profit: 12.4, trades: 14, winRate: 71 },
  ];

  const maxProfit = Math.max(...tradingData.map(d => Math.abs(d.profit)));

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">트레이딩 일지 작성법 📝</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          성공적인 투자자가 되기 위한 첫걸음, 체계적인 매매 기록을 시작하세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 나의 트레이딩 성과</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">월별 수익률 추이</h3>
          
          {/* 수익률 그래프 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <div className="relative h-64">
              {/* Y축 라벨 */}
              <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-gray-600 dark:text-gray-400">
                <span>15%</span>
                <span>10%</span>
                <span>5%</span>
                <span>0%</span>
                <span>-5%</span>
              </div>
              
              {/* 그래프 영역 */}
              <div className="ml-12 h-full flex items-end justify-between gap-2">
                {tradingData.map((data, index) => (
                  <div key={index} className="flex-1 flex flex-col items-center">
                    <button
                      onClick={() => setSelectedMonth(index)}
                      className={`relative w-full transition-all ${
                        selectedMonth === index ? 'scale-105' : ''
                      }`}
                    >
                      <div
                        className={`w-full rounded-t transition-all ${
                          data.profit >= 0 
                            ? 'bg-green-500 hover:bg-green-600' 
                            : 'bg-red-500 hover:bg-red-600'
                        }`}
                        style={{
                          height: `${Math.abs(data.profit) / maxProfit * 120}px`,
                          marginBottom: data.profit >= 0 ? '0' : `${120}px`,
                          marginTop: data.profit >= 0 ? `${120}px` : '0',
                        }}
                      />
                      <span className="text-xs mt-2 block">{data.month}</span>
                    </button>
                  </div>
                ))}
              </div>
              
              {/* 0% 기준선 */}
              <div className="absolute left-12 right-0 top-1/2 border-t border-gray-400 dark:border-gray-600" />
            </div>
          </div>

          {/* 선택된 월 상세 정보 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-3">{tradingData[selectedMonth].month} 트레이딩 상세</h4>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">수익률</p>
                <p className={`text-2xl font-bold ${
                  tradingData[selectedMonth].profit >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {tradingData[selectedMonth].profit >= 0 ? '+' : ''}{tradingData[selectedMonth].profit}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">거래 횟수</p>
                <p className="text-2xl font-bold">{tradingData[selectedMonth].trades}회</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">승률</p>
                <p className="text-2xl font-bold">{tradingData[selectedMonth].winRate}%</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📝 트레이딩 일지 작성 요령</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-4">필수 기록 항목</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">📌 거래 정보</h4>
              <ul className="space-y-2 text-sm">
                <li>• <strong>날짜/시간:</strong> 정확한 매매 시점</li>
                <li>• <strong>종목명:</strong> 매매한 주식</li>
                <li>• <strong>매매 구분:</strong> 매수/매도</li>
                <li>• <strong>수량:</strong> 거래 주식 수</li>
                <li>• <strong>가격:</strong> 체결 가격</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-3">💭 투자 근거</h4>
              <ul className="space-y-2 text-sm">
                <li>• <strong>매매 이유:</strong> 왜 샀는지/팔았는지</li>
                <li>• <strong>목표가:</strong> 예상 수익률</li>
                <li>• <strong>손절가:</strong> 위험 관리선</li>
                <li>• <strong>시장 상황:</strong> 당시 시장 분위기</li>
                <li>• <strong>감정 상태:</strong> 심리적 요인</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 트레이딩 일지 샘플</h2>
        <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl p-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="border-b border-gray-200 dark:border-gray-700 pb-3 mb-3">
              <div className="flex justify-between items-center">
                <h4 className="font-semibold">2024년 8월 15일 - 삼성전자 매수</h4>
                <span className="text-sm text-gray-600 dark:text-gray-400">09:30</span>
              </div>
            </div>
            
            <div className="grid md:grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">거래 정보</p>
                <ul className="text-sm space-y-1">
                  <li>• 매수가: 71,500원</li>
                  <li>• 수량: 10주</li>
                  <li>• 투자금: 715,000원</li>
                </ul>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">목표/손절</p>
                <ul className="text-sm space-y-1">
                  <li>• 목표가: 78,000원 (+9.1%)</li>
                  <li>• 손절가: 68,000원 (-4.9%)</li>
                  <li>• 보유 기간: 중기(1-3개월)</li>
                </ul>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-900/50 rounded p-3">
              <p className="text-sm font-medium mb-1">매수 근거</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                1) 반도체 시장 회복 신호 포착<br/>
                2) 20일 이동평균선 지지 확인<br/>
                3) 외국인 순매수 전환<br/>
                4) PER 역사적 저점 수준
              </p>
            </div>
            
            <div className="mt-3 flex items-center gap-4">
              <span className="text-sm">감정 상태:</span>
              <div className="flex gap-2">
                <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 rounded text-xs">자신감</span>
                <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 rounded text-xs">차분함</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 일지 작성의 효과</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">단기 효과</h4>
              <ul className="space-y-2 text-sm">
                <li>✅ 감정적 매매 방지</li>
                <li>✅ 실수 패턴 발견</li>
                <li>✅ 규율 있는 투자 습관</li>
                <li>✅ 손실 최소화</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">장기 효과</h4>
              <ul className="space-y-2 text-sm">
                <li>🎯 나만의 투자 전략 구축</li>
                <li>🎯 시장 패턴 이해도 향상</li>
                <li>🎯 안정적 수익률 달성</li>
                <li>🎯 투자 실력 향상</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-white dark:bg-gray-800 rounded-lg">
            <p className="text-center text-sm text-gray-700 dark:text-gray-300">
              <strong>"성공한 투자자들의 공통점은 꼼꼼한 기록 습관입니다"</strong><br/>
              매일 5분만 투자해서 일지를 작성하면, 1년 후 놀라운 변화를 경험하게 됩니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🚀 실천 가이드</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">오늘부터 시작하세요!</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">1</span>
              <div>
                <p className="font-medium">노트나 엑셀 파일 준비</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  간단한 양식으로 시작하세요. 완벽할 필요 없습니다.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">2</span>
              <div>
                <p className="font-medium">매매 직후 바로 기록</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  시간이 지나면 기억이 왜곡됩니다. 즉시 기록하세요.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">3</span>
              <div>
                <p className="font-medium">주말마다 복기</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  한 주의 매매를 돌아보고 개선점을 찾으세요.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}