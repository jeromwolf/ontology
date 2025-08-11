'use client'

import { Car, Plane, Train, Bike } from 'lucide-react'

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          MaaS와 미래 모빌리티
        </h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Mobility as a Service(MaaS)는 단순한 이동 수단을 넘어 통합된 모빌리티 생태계를 의미합니다.
            자율주행, 공유 모빌리티, UAM(도심항공모빌리티), 하이퍼루프까지 - 도시 교통의 패러다임을
            완전히 바꿀 혁신적인 미래 모빌리티 서비스들을 탐구합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚌 MaaS 플랫폼 아키텍처
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              통합 모빌리티 플랫폼
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# MaaS 플랫폼 아키텍처
class MaaSPlatform:
    def __init__(self):
        self.transport_providers = {
            'ride_sharing': ['Uber', 'Lyft'],
            'bike_sharing': ['Lime', 'Bird'],
            'public_transit': ['Bus', 'Metro', 'Train'],
            'autonomous_taxi': ['Waymo', 'Cruise'],
            'air_taxi': ['Joby', 'Lilium']
        }
        
        self.payment_gateway = PaymentGateway()
        self.route_optimizer = MultiModalRouteOptimizer()
        
    def plan_journey(self, origin, destination, preferences):
        # 다중 교통수단 경로 계획
        options = []
        
        # 각 교통수단 조합 계산
        for mode_combo in self.get_mode_combinations():
            route = self.route_optimizer.calculate(
                origin, destination, mode_combo, preferences
            )
            
            options.append({
                'route': route,
                'duration': route.total_time,
                'cost': route.total_cost,
                'carbon_footprint': route.co2_emissions,
                'comfort_level': route.comfort_score
            })
        
        return sorted(options, key=lambda x: self.score_route(x, preferences))`}</pre>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              실시간 최적화 시스템
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# 동적 경로 재계획
class DynamicRouteOptimizer:
    def __init__(self):
        self.traffic_predictor = TrafficPredictor()
        self.availability_tracker = AvailabilityTracker()
        
    def optimize_in_transit(self, current_journey):
        # 실시간 상황 변화 감지
        current_conditions = {
            'traffic': self.get_real_time_traffic(),
            'weather': self.get_weather_conditions(),
            'vehicle_availability': self.check_availability(),
            'incidents': self.get_incidents()
        }
        
        # 재최적화 필요성 판단
        if self.needs_reoptimization(current_conditions):
            # 대안 경로 계산
            alternatives = self.compute_alternatives(
                current_journey.current_position,
                current_journey.destination
            )
            
            # 사용자에게 대안 제시
            if alternatives[0].improvement > 0.2:  # 20% 개선
                return self.suggest_route_change(alternatives[0])
        
        return current_journey`}</pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚁 UAM (Urban Air Mobility)
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">eVTOL 항공기 기술</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h5 className="font-bold text-blue-600 dark:text-blue-400 mb-2">멀티콥터형</h5>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 구조 간단, 안정성 높음</li>
                    <li>• 수직 이착륙 용이</li>
                    <li>• 항속거리 제한 (50-100km)</li>
                    <li>• 도심 단거리 운송 적합</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h5 className="font-bold text-green-600 dark:text-green-400 mb-2">틸트로터형</h5>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 높은 순항 효율</li>
                    <li>• 장거리 비행 가능 (200-300km)</li>
                    <li>• 복잡한 전환 메커니즘</li>
                    <li>• 도시 간 연결 적합</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">Vertiport 인프라</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# Vertiport 운영 시스템
class VertiportManagement:
    def __init__(self):
        self.landing_pads = 4
        self.charging_stations = 8
        self.passenger_capacity = 100
        
    def schedule_landing(self, evtol_flight):
        """eVTOL 착륙 스케줄링"""
        # 가용 착륙장 확인
        available_pad = self.find_available_pad(
            evtol_flight.eta,
            evtol_flight.duration_on_pad
        )
        
        if available_pad:
            # 착륙 시퀀스 생성
            landing_sequence = {
                'pad_id': available_pad,
                'approach_vector': self.calculate_approach(),
                'landing_time': evtol_flight.eta,
                'charging_bay': self.assign_charging_bay()
            }
            
            # 교통 관제 시스템과 통합
            self.utm_integration.register_landing(landing_sequence)
            
            return landing_sequence
        else:
            # 대기 패턴 지시
            return self.assign_holding_pattern(evtol_flight)`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚄 하이퍼루프와 초고속 교통
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              하이퍼루프 기술
            </h4>
            <div className="space-y-3">
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                <h5 className="font-bold text-purple-700 dark:text-purple-400 mb-1">진공 튜브</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  • 공기 압력: 100 Pa (대기압의 1/1000)<br/>
                  • 공기 저항 최소화<br/>
                  • 에너지 효율 극대화
                </p>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                <h5 className="font-bold text-blue-700 dark:text-blue-400 mb-1">자기부상 추진</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  • 비접촉 부상 및 추진<br/>
                  • 마찰 제로<br/>
                  • 최고 속도: 1,200 km/h
                </p>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                <h5 className="font-bold text-green-700 dark:text-green-400 mb-1">캡슐 설계</h5>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  • 승객 28-40명 수용<br/>
                  • 압력 유지 시스템<br/>
                  • 비상 탈출 장치
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              자율주행 대중교통
            </h4>
            <div className="space-y-3">
              <div className="flex items-center gap-3 p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                <Car className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                <div>
                  <h5 className="font-bold text-orange-700 dark:text-orange-400">자율주행 버스</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    고정 노선, 24시간 운행, 수요 응답형
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <Train className="w-6 h-6 text-green-600 dark:text-green-400" />
                <div>
                  <h5 className="font-bold text-green-700 dark:text-green-400">자율주행 트램</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    궤도 없는 가상 레일, 도심 순환
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <Bike className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                <div>
                  <h5 className="font-bold text-blue-700 dark:text-blue-400">자율주행 마이크로 모빌리티</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    라스트마일 연결, 자동 재배치
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🌐 스마트시티 통합
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="space-y-4">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">디지털 트윈 기반 교통 관리</h4>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 스마트시티 교통 디지털 트윈
class SmartCityDigitalTwin:
    def __init__(self):
        self.city_model = CityModel3D()
        self.sensor_network = IoTSensorNetwork()
        self.ai_predictor = TrafficAIPredictor()
        
    def real_time_optimization(self):
        """실시간 도시 교통 최적화"""
        # 센서 데이터 수집
        current_state = {
            'traffic_flow': self.sensor_network.get_traffic_data(),
            'public_transit': self.get_transit_status(),
            'parking': self.get_parking_availability(),
            'air_quality': self.sensor_network.get_air_quality(),
            'events': self.get_city_events()
        }
        
        # AI 예측 모델 실행
        predictions = self.ai_predictor.predict_next_hour(current_state)
        
        # 최적화 전략 수립
        optimization_actions = []
        
        if predictions['congestion_level'] > 0.7:
            # 교통 분산 전략
            optimization_actions.extend([
                self.adjust_traffic_signals(predictions['congestion_areas']),
                self.reroute_public_transit(),
                self.activate_congestion_pricing(),
                self.promote_alternative_transport()
            ])
        
        # 디지털 트윈에서 시뮬레이션
        simulation_results = self.city_model.simulate(
            optimization_actions,
            time_horizon=60  # 60분
        )
        
        # 최적 전략 실행
        if simulation_results['improvement'] > 0.15:  # 15% 개선
            self.execute_optimization(optimization_actions)
        
        return optimization_actions`}</pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🔮 2030년 모빌리티 비전
        </h3>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-4">
            <Plane className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">
              도심 항공 택시
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              3차원 도시 교통망 구축, 30분 내 도시 횡단
            </p>
          </div>
          
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
            <Car className="w-8 h-8 text-green-600 dark:text-green-400 mb-2" />
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-2">
              완전 자율주행
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Level 5 자율주행 상용화, 운전면허 불필요
            </p>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-4">
            <Train className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-2" />
            <h4 className="font-bold text-purple-700 dark:text-purple-400 mb-2">
              초고속 교통
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              하이퍼루프로 도시 간 30분 연결
            </p>
          </div>
          
          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-4">
            <Bike className="w-8 h-8 text-orange-600 dark:text-orange-400 mb-2" />
            <h4 className="font-bold text-orange-700 dark:text-orange-400 mb-2">
              통합 MaaS
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              하나의 앱으로 모든 이동 수단 이용
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}