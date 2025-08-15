'use client';

import { useState } from 'react';

export default function Chapter22() {
  const [selectedWave, setSelectedWave] = useState('impulse');
  const [selectedHarmonic, setSelectedHarmonic] = useState('gartley');
  
  // 엘리어트 파동 데이터
  const wavePatterns = {
    impulse: {
      name: '충격파동 (1-2-3-4-5)',
      data: [40, 60, 50, 80, 70, 90, 85, 100],
      labels: ['시작', '1파', '2파', '3파', '4파', '5파', 'A파', '완료'],
      color: '#3b82f6'
    },
    corrective: {
      name: '조정파동 (A-B-C)',
      data: [90, 70, 80, 60, 65, 50],
      labels: ['시작', 'A파', 'B파', 'C파', '반등', '완료'],
      color: '#ef4444'
    }
  };

  // 하모닉 패턴 좌표 데이터
  const harmonicPatterns = {
    gartley: {
      name: 'Gartley 패턴',
      points: [
        { x: 20, y: 150, label: 'X' },
        { x: 60, y: 50, label: 'A' },
        { x: 100, y: 90, label: 'B' },
        { x: 140, y: 70, label: 'C' },
        { x: 180, y: 85, label: 'D' }
      ],
      ratios: ['XA', 'AB=0.618 XA', 'BC=0.382-0.886 AB', 'CD=0.786 XA']
    },
    butterfly: {
      name: 'Butterfly 패턴',
      points: [
        { x: 20, y: 120, label: 'X' },
        { x: 60, y: 40, label: 'A' },
        { x: 100, y: 65, label: 'B' },
        { x: 140, y: 50, label: 'C' },
        { x: 180, y: 20, label: 'D' }
      ],
      ratios: ['XA', 'AB=0.786 XA', 'BC=0.382-0.886 AB', 'CD=1.27-1.618 XA']
    },
    bat: {
      name: 'Bat 패턴',
      points: [
        { x: 20, y: 140, label: 'X' },
        { x: 60, y: 60, label: 'A' },
        { x: 100, y: 100, label: 'B' },
        { x: 140, y: 80, label: 'C' },
        { x: 180, y: 75, label: 'D' }
      ],
      ratios: ['XA', 'AB=0.382-0.5 XA', 'BC=0.382-0.886 AB', 'CD=0.886 XA']
    }
  };

  const currentWave = wavePatterns[selectedWave];
  const currentHarmonic = harmonicPatterns[selectedHarmonic];

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">고급 차트 패턴과 하모닉 트레이딩</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          기본적인 차트 패턴을 넘어, 프로 트레이더들이 사용하는 고급 기법들을 배워봅시다.
          엘리어트 파동, 하모닉 패턴, 피보나치의 심화 활용법을 마스터하게 됩니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 엘리어트 파동 이론 시뮬레이터</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">파동의 기본 구조</h3>
          
          {/* 파동 선택 버튼 */}
          <div className="flex gap-3 mb-6">
            <button
              onClick={() => setSelectedWave('impulse')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                selectedWave === 'impulse'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              충격파동 (5파)
            </button>
            <button
              onClick={() => setSelectedWave('corrective')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                selectedWave === 'corrective'
                  ? 'bg-red-500 text-white'
                  : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              조정파동 (ABC)
            </button>
          </div>

          {/* 엘리어트 파동 차트 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-4">
            <h4 className="font-semibold mb-3">{currentWave.name}</h4>
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
                
                {/* 파동 차트 */}
                <polyline
                  fill="none"
                  stroke={currentWave.color}
                  strokeWidth="3"
                  points={currentWave.data
                    .map((value, index) => {
                      const x = (index / (currentWave.data.length - 1)) * 380 + 10;
                      const y = 200 - (value * 1.8);
                      return `${x},${y}`;
                    })
                    .join(' ')}
                />
                
                {/* 라벨 */}
                {currentWave.data.map((value, index) => {
                  const x = (index / (currentWave.data.length - 1)) * 380 + 10;
                  const y = 200 - (value * 1.8);
                  return (
                    <g key={index}>
                      <circle cx={x} cy={y} r="4" fill={currentWave.color} />
                      <text x={x} y={y - 10} className="text-xs font-medium" textAnchor="middle">
                        {currentWave.labels[index]}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded">
              <h5 className="font-medium mb-2">핵심 규칙</h5>
              {selectedWave === 'impulse' ? (
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 3파는 절대 가장 짧은 파동이 될 수 없음</li>
                  <li>• 2파는 1파의 시작점 아래로 내려갈 수 없음</li>
                  <li>• 4파는 1파의 고점과 겹칠 수 없음</li>
                  <li>• 충격파 이후 ABC 조정파가 따라옴</li>
                </ul>
              ) : (
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 지그재그: 5-3-5 구조의 급격한 조정</li>
                  <li>• 플랫: 3-3-5 구조의 횡보 조정</li>
                  <li>• 삼각형: 수렴하는 5개 파동의 조정</li>
                  <li>• 조정 완료 후 새로운 충격파 시작</li>
                </ul>
              )}
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🦋 하모닉 패턴 시뮬레이터</h2>
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          {/* 패턴 선택 버튼 */}
          <div className="flex flex-wrap gap-3 mb-6">
            {Object.keys(harmonicPatterns).map((pattern) => (
              <button
                key={pattern}
                onClick={() => setSelectedHarmonic(pattern)}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  selectedHarmonic === pattern
                    ? 'bg-purple-500 text-white'
                    : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                {harmonicPatterns[pattern].name}
              </button>
            ))}
          </div>

          {/* 하모닉 패턴 차트 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <div className="relative h-64 mb-4">
              <svg viewBox="0 0 200 200" className="w-full h-full">
                {/* 그리드 */}
                {[0, 50, 100, 150, 200].map((pos) => (
                  <g key={pos}>
                    <line x1={pos} y1="0" x2={pos} y2="200" stroke="#e5e7eb" strokeDasharray="2,2" />
                    <line x1="0" y1={pos} x2="200" y2={pos} stroke="#e5e7eb" strokeDasharray="2,2" />
                  </g>
                ))}
                
                {/* 패턴 라인 */}
                <polyline
                  fill="none"
                  stroke="#8b5cf6"
                  strokeWidth="2"
                  points={currentHarmonic.points
                    .map(p => `${p.x},${p.y}`)
                    .join(' ')}
                />
                
                {/* 피보나치 레벨 라인 (점선) */}
                {currentHarmonic.points.slice(0, -1).map((point, index) => {
                  const nextPoint = currentHarmonic.points[index + 1];
                  return (
                    <line
                      key={index}
                      x1={point.x}
                      y1={point.y}
                      x2={nextPoint.x}
                      y2={nextPoint.y}
                      stroke="#6366f1"
                      strokeWidth="1"
                      strokeDasharray="5,5"
                      opacity="0.3"
                    />
                  );
                })}
                
                {/* 포인트와 라벨 */}
                {currentHarmonic.points.map((point, index) => (
                  <g key={index}>
                    <circle cx={point.x} cy={point.y} r="6" fill="#8b5cf6" />
                    <text 
                      x={point.x} 
                      y={point.y - 10} 
                      className="text-sm font-bold fill-purple-600 dark:fill-purple-400" 
                      textAnchor="middle"
                    >
                      {point.label}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
            
            {/* 비율 정보 */}
            <div className="space-y-2">
              <h4 className="font-semibold mb-2">피보나치 비율</h4>
              {currentHarmonic.ratios.map((ratio, index) => (
                <div key={index} className="text-sm text-gray-600 dark:text-gray-400">
                  • {ratio}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">패턴별 성공률</h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Bat 패턴</span>
                  <span className="text-sm font-medium">75%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-purple-600 h-2 rounded-full" style={{width: '75%'}}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Gartley 패턴</span>
                  <span className="text-sm font-medium">70%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-indigo-600 h-2 rounded-full" style={{width: '70%'}}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Butterfly 패턴</span>
                  <span className="text-sm font-medium">65%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full" style={{width: '65%'}}></div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">트레이딩 팁</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>PRZ(Potential Reversal Zone)에서 추가 확인 신호 대기</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>RSI 다이버전스와 함께 나타날 때 신뢰도 상승</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">✓</span>
                <span>손절은 X포인트 약간 위/아래 설정</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔢 피보나치 고급 활용법</h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">피보나치 클러스터 (Confluence) 시뮬레이터</h3>
          
          {/* 피보나치 레벨 차트 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-4">
            <div className="relative h-64">
              <svg viewBox="0 0 400 200" className="w-full h-full">
                {/* 가격 차트 */}
                <polyline
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  points="20,180 60,160 100,140 140,120 180,100 220,130 260,110 300,90 340,70 380,80"
                />
                
                {/* 피보나치 레벨들 */}
                {[
                  { level: 0.236, y: 160, color: '#10b981' },
                  { level: 0.382, y: 140, color: '#3b82f6' },
                  { level: 0.5, y: 125, color: '#8b5cf6' },
                  { level: 0.618, y: 110, color: '#ef4444' },
                  { level: 0.786, y: 95, color: '#f59e0b' }
                ].map((fib, index) => (
                  <g key={index}>
                    <line
                      x1="0"
                      y1={fib.y}
                      x2="400"
                      y2={fib.y}
                      stroke={fib.color}
                      strokeDasharray="5,5"
                      opacity="0.7"
                    />
                    <text
                      x="10"
                      y={fib.y - 5}
                      className="text-xs font-medium"
                      fill={fib.color}
                    >
                      {(fib.level * 100).toFixed(1)}%
                    </text>
                  </g>
                ))}
                
                {/* 클러스터 존 표시 */}
                <rect
                  x="0"
                  y="105"
                  width="400"
                  height="10"
                  fill="#ef4444"
                  opacity="0.2"
                />
                <text x="200" y="100" className="text-sm font-bold fill-red-600" textAnchor="middle">
                  강력한 지지/저항 구간
                </text>
              </svg>
            </div>
            
            <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                여러 피보나치 레벨이 겹치는 구간(클러스터)은 특히 강력한 지지/저항으로 작용합니다.
                위 차트에서는 61.8% 레벨 근처에 여러 피보나치가 수렴하고 있습니다.
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">시간 피보나치</h4>
              <ul className="text-sm space-y-1">
                <li>• 주요 저점/고점에서 8, 13, 21, 34일</li>
                <li>• 시간대별 변곡점 예측</li>
                <li>• 사이클 분석과 결합</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">가격 피보나치</h4>
              <ul className="text-sm space-y-1">
                <li>• 다중 스윙의 피보나치 레벨 중첩</li>
                <li>• 확장 레벨: 127.2%, 161.8%, 261.8%</li>
                <li>• 내부 되돌림: 23.6%, 78.6%, 88.6%</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2">피보나치 채널</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              추세선에 피보나치 비율을 적용하여 가격 채널 구성
            </p>
            <ul className="text-sm space-y-1">
              <li>• 0% - 기준 추세선</li>
              <li>• 61.8% - 1차 저항/지지</li>
              <li>• 100% - 2차 저항/지지</li>
              <li>• 161.8% - 극단적 레벨</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⏱️ 멀티 타임프레임 분석</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">3-Screen Trading System</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                Screen 1: 주 추세 (주봉/일봉)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• MACD 히스토그램으로 추세 방향 확인</li>
                <li>• 이동평균선 정렬 상태 점검</li>
                <li>• 주요 지지/저항 레벨 식별</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                Screen 2: 중간 신호 (일봉/4시간봉)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 스토캐스틱/RSI로 진입 타이밍</li>
                <li>• 주 추세와 반대 방향 신호 필터링</li>
                <li>• Force Index로 매수/매도 압력 측정</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
                Screen 3: 정확한 진입 (시간봉/분봉)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 브레이크아웃 또는 풀백 진입</li>
                <li>• 트레일링 스탑 설정</li>
                <li>• 목표가 및 손절가 미세 조정</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 실전 적용 팁</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">고급 패턴 트레이딩 체크리스트</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">패턴 완성도 확인</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  피보나치 비율이 ±5% 이내로 정확한가?
                </p>
              </div>
            </label>
            
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">다중 확인 (Confluence)</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  2개 이상의 기술적 요소가 같은 레벨을 지지하는가?
                </p>
              </div>
            </label>
            
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">시장 환경 적합성</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  현재 시장이 패턴 트레이딩에 적합한 변동성을 보이는가?
                </p>
              </div>
            </label>
            
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">리스크 관리 계획</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  PRZ(Potential Reversal Zone)에서의 손절 및 목표가 설정
                </p>
              </div>
            </label>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 주의사항</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">
            고급 기법의 함정
          </h3>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <div>
                <strong>과도한 복잡성:</strong> 단순한 것이 더 효과적일 때가 많음
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <div>
                <strong>확증 편향:</strong> 원하는 패턴을 억지로 찾으려 하지 말 것
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <div>
                <strong>백테스팅 부족:</strong> 실전 적용 전 충분한 검증 필수
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <div>
                <strong>시장 상황 무시:</strong> 모든 패턴이 모든 시장에서 작동하지 않음
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}