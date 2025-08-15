'use client';

import { Battery, Zap } from 'lucide-react';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          전동화와 배터리 관리
        </h2>
        
        <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            자율주행의 미래는 전동화와 함께합니다. Tesla, BYD, 현대차의 EV 혁신부터 차세대 배터리 기술,
            무선 충전까지 - 지속가능한 모빌리티를 위한 핵심 기술들을 학습합니다.
            특히 BMS(Battery Management System)는 안전하고 효율적인 EV 운영의 핵심입니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔋 EV 파워트레인 시스템
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              <Battery className="inline w-5 h-5 mr-2" />
              EV 구성 요소
            </h4>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center text-xs font-bold text-blue-600 dark:text-blue-400">1</div>
                <div>
                  <span className="font-semibold text-gray-900 dark:text-white">배터리 팩</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">리튬이온, 고체전해질, 나트륨이온</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center text-xs font-bold text-green-600 dark:text-green-400">2</div>
                <div>
                  <span className="font-semibold text-gray-900 dark:text-white">인버터</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">DC→AC 변환, SiC 반도체 사용</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center text-xs font-bold text-purple-600 dark:text-purple-400">3</div>
                <div>
                  <span className="font-semibold text-gray-900 dark:text-white">모터</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">PMSM, BLDC, 인휠 모터</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center text-xs font-bold text-orange-600 dark:text-orange-400">4</div>
                <div>
                  <span className="font-semibold text-gray-900 dark:text-white">충전 시스템</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">AC/DC 충전, 무선 충전</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              <Zap className="inline w-5 h-5 mr-2" />
              주요 EV 제조사 비교
            </h4>
            <div className="space-y-3">
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-bold text-red-600 dark:text-red-400">Tesla</span>
                  <span className="text-xs bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 px-2 py-0.5 rounded">4680 셀</span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">에너지 밀도: 296 Wh/kg</p>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-bold text-blue-600 dark:text-blue-400">BYD</span>
                  <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 px-2 py-0.5 rounded">Blade</span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">안전성 특화: LFP 기반</p>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-bold text-green-600 dark:text-green-400">현대차</span>
                  <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 px-2 py-0.5 rounded">E-GMP</span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">초고속 충전: 18분 80%</p>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🧠 BMS (Battery Management System)
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">배터리 상태 추정 알고리즘</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# SOC (State of Charge) 추정 - Kalman Filter 기반
class SOCEstimator:
    def __init__(self):
        # 칼만 필터 초기화
        self.x = np.array([1.0])  # 초기 SOC = 100%
        self.P = np.array([[0.1]])  # 초기 오차 공분산
        self.Q = np.array([[1e-5]])  # 프로세스 노이즈
        self.R = np.array([[0.01]])  # 측정 노이즈
        
    def predict(self, current, dt):
        """전류 적분으로 SOC 예측"""
        coulomb_efficiency = 0.99
        capacity = 75000  # 75kWh = 75,000Wh
        
        # SOC 변화량 계산
        dsoc = -(current * dt * coulomb_efficiency) / capacity
        
        # 상태 예측
        self.x[0] += dsoc
        self.P[0,0] += self.Q[0,0]
        
    def update(self, voltage_measurement):
        """전압 측정값으로 SOC 보정"""
        # OCV-SOC 룩업 테이블에서 예상 전압 계산
        predicted_voltage = self.ocv_lookup(self.x[0])
        
        # 칼만 필터 업데이트
        innovation = voltage_measurement - predicted_voltage
        S = self.P[0,0] + self.R[0,0]
        K = self.P[0,0] / S
        
        self.x[0] += K * innovation
        self.P[0,0] *= (1 - K)
        
        return self.x[0]  # 추정된 SOC 반환
    
    def ocv_lookup(self, soc):
        """SOC에 따른 개방전압(OCV) 계산"""
        # 실제로는 배터리별 특성 데이터 사용
        return 3.2 + 0.8 * soc  # 간단한 선형 모델`}</pre>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">배터리 안전 관리</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 배터리 안전 감시 시스템
class BatterySafetyManager:
    def __init__(self):
        self.safety_limits = {
            'voltage_max': 4.2,      # 셀당 최대 전압 (V)
            'voltage_min': 2.5,      # 셀당 최소 전압 (V)
            'temp_max': 60,          # 최대 온도 (°C)
            'temp_min': -20,         # 최소 온도 (°C)
            'current_max': 200,      # 최대 전류 (A)
        }
        
    def check_safety(self, cell_voltages, temperatures, current):
        """실시간 안전 상태 체크"""
        alarms = []
        
        # 전압 체크
        for i, voltage in enumerate(cell_voltages):
            if voltage > self.safety_limits['voltage_max']:
                alarms.append(f"Cell {i}: Overvoltage {voltage:.2f}V")
                self.emergency_action('overvoltage', i)
            elif voltage < self.safety_limits['voltage_min']:
                alarms.append(f"Cell {i}: Undervoltage {voltage:.2f}V")
                self.emergency_action('undervoltage', i)
        
        # 온도 체크
        for i, temp in enumerate(temperatures):
            if temp > self.safety_limits['temp_max']:
                alarms.append(f"Module {i}: Overtemp {temp:.1f}°C")
                self.emergency_action('overtemp', i)
        
        # 전류 체크
        if abs(current) > self.safety_limits['current_max']:
            alarms.append(f"Overcurrent {current:.1f}A")
            self.emergency_action('overcurrent')
            
        return alarms
    
    def emergency_action(self, fault_type, module_id=None):
        """비상 상황 대응"""
        if fault_type in ['overvoltage', 'overtemp']:
            # 즉시 충전 중단
            self.stop_charging()
            # 해당 모듈 격리
            if module_id is not None:
                self.isolate_module(module_id)
        
        elif fault_type == 'overcurrent':
            # 전류 제한
            self.limit_current(self.safety_limits['current_max'] * 0.8)
        
        # 경고 신호 전송
        self.send_warning_to_vehicle_system(fault_type)`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          ⚡ 충전 기술
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">유선 충전</h4>
            <div className="space-y-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <h5 className="font-bold text-blue-700 dark:text-blue-400 mb-2">DC 급속 충전</h5>
                <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <div className="flex justify-between">
                    <span>CCS1/2:</span>
                    <span>최대 350kW</span>
                  </div>
                  <div className="flex justify-between">
                    <span>CHAdeMO:</span>
                    <span>최대 400kW</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tesla SC:</span>
                    <span>최대 250kW</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                <h5 className="font-bold text-green-700 dark:text-green-400 mb-2">AC 완속 충전</h5>
                <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <div className="flex justify-between">
                    <span>Type 1:</span>
                    <span>최대 7kW (미국)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Type 2:</span>
                    <span>최대 22kW (유럽)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>가정용:</span>
                    <span>3.3-11kW</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">무선 충전 (WPT)</h4>
            <div className="space-y-4">
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                <h5 className="font-bold text-purple-700 dark:text-purple-400 mb-2">전자기 유도 방식</h5>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  • 주파수: 85kHz (SAE J2954)<br/>
                  • 효율: 90-95%<br/>
                  • 거리: 10-25cm<br/>
                  • 전력: 3.7-22kW
                </div>
              </div>
              
              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                <h5 className="font-bold text-orange-700 dark:text-orange-400 mb-2">동적 무선 충전</h5>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  • 주행 중 충전 가능<br/>
                  • 도로 매설형 송신부<br/>
                  • 100kW급 고출력 개발 중<br/>
                  • 2030년 상용화 목표
                </div>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🌡️ 배터리 열관리 시스템
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="space-y-4">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">냉각 시스템 종류</h4>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h5 className="font-bold text-blue-700 dark:text-blue-400 mb-2">공랭식</h5>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 구조 간단</li>
                    <li>• 저비용</li>
                    <li>• 냉각 효율 제한</li>
                    <li>• 소형 배터리 적합</li>
                  </ul>
                </div>
                
                <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                  <h5 className="font-bold text-green-700 dark:text-green-400 mb-2">수랭식</h5>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 높은 냉각 효율</li>
                    <li>• 정밀한 온도 제어</li>
                    <li>• 복잡한 시스템</li>
                    <li>• 대용량 배터리 필수</li>
                  </ul>
                </div>
                
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <h5 className="font-bold text-purple-700 dark:text-purple-400 mb-2">직접 냉각</h5>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 절연 냉매 사용</li>
                    <li>• 최고 냉각 성능</li>
                    <li>• 미래 기술</li>
                    <li>• 개발 단계</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🔮 차세대 배터리 기술
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6">
            <h4 className="font-bold text-yellow-700 dark:text-yellow-400 mb-3">
              🔥 고체전해질 배터리
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>에너지 밀도:</strong> 500Wh/kg 이상</li>
              <li>• <strong>안전성:</strong> 화재 위험 없음</li>
              <li>• <strong>수명:</strong> 100만km 이상</li>
              <li>• <strong>상용화:</strong> 2027-2030년</li>
              <li>• <strong>주요 기업:</strong> Toyota, Samsung SDI</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-3">
              ⚡ 나트륨이온 배터리
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>원료:</strong> 풍부한 나트륨 사용</li>
              <li>• <strong>비용:</strong> 30-40% 저렴</li>
              <li>• <strong>안전성:</strong> 높은 열 안정성</li>
              <li>• <strong>상용화:</strong> 2024년 시작</li>
              <li>• <strong>주요 기업:</strong> CATL, BYD</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}