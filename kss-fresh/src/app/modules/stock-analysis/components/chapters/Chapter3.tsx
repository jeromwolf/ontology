'use client';

import { useState } from 'react';

export default function Chapter3() {
  const [selectedPattern, setSelectedPattern] = useState('head-shoulders');
  
  // 패턴별 가격 데이터
  const patternData = {
    'head-shoulders': [
      50, 52, 54, 56, 58, 55, 53, 51, 54, 57, 60, 63, 65, 62, 59, 56, 53, 50, 47, 44, 42, 40
    ],
    'double-top': [
      45, 47, 50, 53, 56, 58, 60, 58, 56, 54, 52, 54, 56, 58, 60, 59, 57, 54, 51, 48, 45, 42
    ],
    'ascending-triangle': [
      40, 38, 42, 40, 44, 42, 46, 44, 48, 46, 50, 48, 50, 49, 50, 52, 54, 56, 58, 60, 62, 64
    ],
    'cup-handle': [
      55, 52, 48, 45, 43, 42, 41, 42, 43, 45, 48, 52, 55, 54, 53, 54, 55, 57, 60, 63, 66, 68
    ]
  };

  const patterns = [
    { id: 'head-shoulders', name: '헤드앤숄더', type: 'bearish' },
    { id: 'double-top', name: '더블탑', type: 'bearish' },
    { id: 'ascending-triangle', name: '상승삼각형', type: 'bullish' },
    { id: 'cup-handle', name: '컵앤핸들', type: 'bullish' }
  ];

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">차트 패턴 인식 📊</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          반복되는 차트 패턴을 통해 시장의 심리와 향후 방향을 예측하는 방법을 배워봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 주요 차트 패턴 시뮬레이터</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
          {/* 패턴 선택 버튼 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
            {patterns.map((pattern) => (
              <button
                key={pattern.id}
                onClick={() => setSelectedPattern(pattern.id)}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  selectedPattern === pattern.id
                    ? pattern.type === 'bullish'
                      ? 'bg-green-500 text-white'
                      : 'bg-red-500 text-white'
                    : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                {pattern.name}
              </button>
            ))}
          </div>

          {/* 차트 그래프 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <div className="relative h-64">
              <svg viewBox="0 0 400 200" className="w-full h-full">
                {/* 그리드 라인 */}
                {[0, 50, 100, 150, 200].map((y) => (
                  <line
                    key={y}
                    x1="0"
                    y1={y}
                    x2="400"
                    y2={y}
                    stroke="#e5e7eb"
                    strokeDasharray="2,2"
                  />
                ))}
                
                {/* 가격 차트 */}
                <polyline
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  points={patternData[selectedPattern]
                    .map((price, index) => {
                      const x = (index / (patternData[selectedPattern].length - 1)) * 380 + 10;
                      const y = 200 - ((price - 35) * 3);
                      return `${x},${y}`;
                    })
                    .join(' ')}
                />
                
                {/* 패턴별 주요 포인트 표시 */}
                {selectedPattern === 'head-shoulders' && (
                  <>
                    <circle cx="110" cy="65" r="4" fill="#ef4444" />
                    <circle cx="210" cy="35" r="4" fill="#ef4444" />
                    <circle cx="300" cy="65" r="4" fill="#ef4444" />
                    <text x="110" y="55" className="text-xs fill-red-600" textAnchor="middle">왼쪽 어깨</text>
                    <text x="210" y="25" className="text-xs fill-red-600" textAnchor="middle">머리</text>
                    <text x="300" y="55" className="text-xs fill-red-600" textAnchor="middle">오른쪽 어깨</text>
                  </>
                )}
              </svg>
            </div>
            
            {/* 패턴 설명 */}
            <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
              <h4 className="font-semibold mb-2">
                {patterns.find(p => p.id === selectedPattern)?.name} 패턴
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {selectedPattern === 'head-shoulders' && 
                  "상승 추세 후 나타나는 대표적인 하락 반전 패턴. 세 개의 고점 중 가운데가 가장 높고, 네크라인 하향 돌파 시 하락 전환 신호."}
                {selectedPattern === 'double-top' && 
                  "두 개의 비슷한 고점을 형성하는 하락 반전 패턴. 두 번째 고점이 첫 번째보다 약간 낮으면 더 강한 신호."}
                {selectedPattern === 'ascending-triangle' && 
                  "수평 저항선과 상승하는 지지선이 만나는 상승 지속 패턴. 저항선 돌파 시 강한 상승 신호."}
                {selectedPattern === 'cup-handle' && 
                  "U자형 바닥을 형성 후 작은 조정을 거치는 상승 지속 패턴. 핸들 구간 돌파 시 매수 시점."}
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">기술적 분석의 3대 가정</h2>
        <div className="grid gap-4 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">1. 시장은 모든 것을 반영한다</h3>
            <p className="text-gray-700 dark:text-gray-300">
              주가에는 이미 모든 정보(기업 실적, 경제 상황, 투자심리 등)가 반영되어 있다.
            </p>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">2. 가격은 추세를 따라 움직인다</h3>
            <p className="text-gray-700 dark:text-gray-300">
              주가는 무작위로 움직이지 않고 일정한 패턴과 추세를 보인다.
            </p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">3. 역사는 반복된다</h3>
            <p className="text-gray-700 dark:text-gray-300">
              과거의 패턴은 미래에도 반복될 가능성이 높다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 거래량 패턴 분석</h2>
        <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">가격과 거래량의 관계</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="relative h-48">
              <svg viewBox="0 0 400 180" className="w-full h-full">
                {/* 가격 차트 (위쪽) */}
                <polyline
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  points="20,100 60,90 100,80 140,85 180,70 220,60 260,65 300,50 340,40 380,35"
                />
                
                {/* 거래량 바 차트 (아래쪽) */}
                {[20, 60, 100, 140, 180, 220, 260, 300, 340, 380].map((x, index) => {
                  const volumes = [30, 35, 45, 40, 55, 65, 60, 70, 75, 80];
                  return (
                    <rect
                      key={index}
                      x={x - 10}
                      y={180 - volumes[index]}
                      width="20"
                      height={volumes[index]}
                      fill={index < 5 ? "#ef4444" : "#10b981"}
                      opacity="0.7"
                    />
                  );
                })}
                
                {/* 라벨 */}
                <text x="200" y="20" className="text-sm font-medium" textAnchor="middle">가격</text>
                <text x="200" y="170" className="text-sm font-medium" textAnchor="middle">거래량</text>
              </svg>
            </div>
            
            <div className="mt-4 grid md:grid-cols-2 gap-4 text-sm">
              <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                <h4 className="font-semibold text-green-700 dark:text-green-300 mb-1">
                  ✅ 건전한 상승 패턴
                </h4>
                <p className="text-gray-700 dark:text-gray-300">
                  가격 상승 + 거래량 증가<br/>
                  많은 투자자의 참여로 지속 가능한 상승
                </p>
              </div>
              <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                <h4 className="font-semibold text-red-700 dark:text-red-300 mb-1">
                  ⚠️ 위험한 상승 패턴
                </h4>
                <p className="text-gray-700 dark:text-gray-300">
                  가격 상승 + 거래량 감소<br/>
                  매수세 약화로 조정 가능성 높음
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">캔들스틱 패턴</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          캔들스틱 차트는 일정 기간의 시가, 고가, 저가, 종가 정보를 하나의 캔들로 표현합니다.
          캔들의 모양과 조합으로 시장 심리와 향후 방향을 예측할 수 있습니다.
        </p>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-3">🟢 강세 반전 패턴</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
              <li><strong>해머:</strong> 하락 후 나타나는 긴 아래꼬리</li>
              <li><strong>불룩한 바닥:</strong> 연속된 두 개의 상승 캔들</li>
              <li><strong>조조별:</strong> 갭 하락 후 상승 마감</li>
              <li><strong>역망치:</strong> 긴 위꼬리를 가진 작은 몸통</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">🔴 약세 반전 패턴</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300 text-sm">
              <li><strong>교수형:</strong> 상승 후 나타나는 긴 아래꼬리</li>
              <li><strong>먹구름:</strong> 연속된 두 개의 하락 캔들</li>
              <li><strong>저녁별:</strong> 갭 상승 후 하락 마감</li>
              <li><strong>유성:</strong> 긴 위꼬리를 가진 작은 몸통</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주요 기술적 지표</h2>
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">📈 이동평균선 (Moving Average)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              일정 기간의 주가를 평균내어 추세를 파악하는 가장 기본적인 지표
            </p>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <strong>단기 이평선 (5일, 20일)</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">단기 추세와 지지/저항</span>
              </div>
              <div>
                <strong>중기 이평선 (60일, 120일)</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">중기 추세 판단</span>
              </div>
              <div>
                <strong>장기 이평선 (200일, 300일)</strong><br/>
                <span className="text-gray-600 dark:text-gray-400">장기 추세와 강력한 지지/저항</span>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">⚡ RSI (Relative Strength Index)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              0~100 사이 값으로 과매수/과매도 상태를 판단하는 모멘텀 오실레이터
            </p>
            <div className="flex items-center justify-between text-sm">
              <span className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-3 py-1 rounded">
                과매도 (30 이하)
              </span>
              <span className="text-gray-600 dark:text-gray-400">
                적정 구간 (30-70)
              </span>
              <span className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-3 py-1 rounded">
                과매수 (70 이상)
              </span>
            </div>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-3">🌊 MACD (Moving Average Convergence Divergence)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              단기 이평선과 장기 이평선의 차이로 추세 변화를 포착하는 지표
            </p>
            <div className="space-y-2 text-sm">
              <div><strong>MACD 선:</strong> 12일 지수이평 - 26일 지수이평</div>
              <div><strong>시그널 선:</strong> MACD의 9일 지수이평</div>
              <div><strong>매매 신호:</strong> MACD가 시그널 선을 상향/하향 돌파</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">지지선과 저항선</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/10 dark:to-orange-900/10 rounded-xl p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">📈 지지선 (Support Line)</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                주가 하락을 막아주는 가격대. 매수세가 강해지는 구간
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 과거 저점들을 연결한 선</li>
                <li>• 심리적 가격대 (1만원, 5만원 등)</li>
                <li>• 주요 이동평균선</li>
                <li>• 돌파 시 추가 하락 가능성</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">📉 저항선 (Resistance Line)</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                주가 상승을 막는 가격대. 매도세가 강해지는 구간
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 과거 고점들을 연결한 선</li>
                <li>• 심리적 저항 가격대</li>
                <li>• 기술적 지표의 과매수 구간</li>
                <li>• 돌파 시 추가 상승 가능성</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-white dark:bg-gray-800 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">💡 실전 매매 전략</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <strong className="text-emerald-600 dark:text-emerald-400">매수 타이밍</strong>
                <ul className="mt-1 space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 지지선 근처에서 반등 확인</li>
                  <li>• 저항선 돌파 후 재테스트</li>
                  <li>• 거래량 증가와 함께 신호 확인</li>
                </ul>
              </div>
              <div>
                <strong className="text-red-600 dark:text-red-400">매도 타이밍</strong>
                <ul className="mt-1 space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 저항선 근처에서 상승 둔화</li>
                  <li>• 지지선 하향 돌파</li>
                  <li>• 거래량 감소와 함께 약세 확인</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}