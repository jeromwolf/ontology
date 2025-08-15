'use client';

import { Car, Wifi, Shield, Radio } from 'lucide-react';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          V2X 통신과 스마트 인프라
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Vehicle-to-Everything (V2X) 통신은 자율주행의 완성체입니다. 차량이 다른 차량, 인프라,
            보행자와 실시간으로 정보를 주고받아 더 안전하고 효율적인 교통 시스템을 구현합니다.
            5G 기반 C-V2X로 진화하며 스마트시티의 핵심 기술이 되고 있습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          📡 V2X 통신 유형
        </h3>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Car className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              <h4 className="font-bold text-gray-900 dark:text-white">V2V</h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Vehicle-to-Vehicle
            </p>
            <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <li>• 위치, 속도, 방향 공유</li>
              <li>• 긴급 브레이킹 알림</li>
              <li>• 협력 주행</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Wifi className="w-8 h-8 text-green-600 dark:text-green-400" />
              <h4 className="font-bold text-gray-900 dark:text-white">V2I</h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Vehicle-to-Infrastructure
            </p>
            <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <li>• 신호등 상태 정보</li>
              <li>• 도로 상황 업데이트</li>
              <li>• 교통 최적화</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              <h4 className="font-bold text-gray-900 dark:text-white">V2P</h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Vehicle-to-Pedestrian
            </p>
            <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <li>• 보행자 위치 감지</li>
              <li>• 횡단보도 안전</li>
              <li>• 스마트폰 연동</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Radio className="w-8 h-8 text-orange-600 dark:text-orange-400" />
              <h4 className="font-bold text-gray-900 dark:text-white">V2N</h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Vehicle-to-Network
            </p>
            <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <li>• 클라우드 서비스</li>
              <li>• 교통 관제 센터</li>
              <li>• 빅데이터 분석</li>
            </ul>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🏗️ 5G C-V2X 아키텍처
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">통신 스택 구조</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# C-V2X 프로토콜 스택
class CV2XStack:
    def __init__(self):
        # 응용 계층
        self.applications = {
            'cooperative_awareness': CAMService(),
            'decentralized_notification': DENMService(),
            'basic_safety': BSMService()
        }
        
        # 전송 계층
        self.transport = {
            'geonetworking': GeoNetworking(),
            'btp': BTP()  # Basic Transport Protocol
        }
        
        # 액세스 계층
        self.access = {
            'pc5': PC5Interface(),  # Direct communication
            'uu': UuInterface()     # Network communication
        }
    
    def send_cam_message(self, vehicle_state):
        # Cooperative Awareness Message
        cam = {
            'station_id': self.vehicle_id,
            'position': vehicle_state.position,
            'speed': vehicle_state.speed,
            'heading': vehicle_state.heading,
            'timestamp': time.time()
        }
        
        # 지리적 멀티캐스트로 근거리 차량들에게 전송
        self.transport['geonetworking'].geocast(
            cam, 
            area_of_interest=Circle(vehicle_state.position, radius=300)
        )`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚦 스마트 교통 인프라
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">적응형 신호 제어</h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# 실시간 교통 상황 기반 신호 최적화
class AdaptiveTrafficControl:
    def __init__(self):
        self.traffic_detector = TrafficDetector()
        self.signal_optimizer = SignalOptimizer()
        
    def optimize_signals(self, intersection_id):
        # 각 방향별 교통량 측정
        traffic_data = self.traffic_detector.get_current_traffic()
        
        # 대기 큐 길이 계산
        queue_lengths = {}
        for direction in ['north', 'south', 'east', 'west']:
            queue_lengths[direction] = self.calculate_queue_length(
                traffic_data[direction]
            )
        
        # 최적 신호 시간 계산
        optimal_timing = self.signal_optimizer.optimize(
            queue_lengths,
            constraints={
                'min_green_time': 10,  # 최소 녹색등 시간
                'max_cycle_time': 120,  # 최대 사이클 시간
                'pedestrian_time': 8   # 보행자 신호 시간
            }
        )
        
        return optimal_timing`}</pre>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">교통 흐름 예측</h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# LSTM 기반 교통 흐름 예측
class TrafficFlowPredictor:
    def __init__(self):
        self.model = nn.LSTM(
            input_size=4,  # 속도, 밀도, 유량, 점유율
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.output_layer = nn.Linear(64, 1)
    
    def predict_traffic_flow(self, historical_data):
        # 과거 30분 데이터로 다음 15분 예측
        with torch.no_grad():
            lstm_out, _ = self.model(historical_data)
            prediction = self.output_layer(lstm_out[:, -1, :])
        
        return prediction
    
    def preemptive_signal_control(self, predicted_flow):
        # 예측된 교통량에 따라 사전 신호 조정
        if predicted_flow > self.congestion_threshold:
            return self.implement_congestion_strategy()
        else:
            return self.implement_normal_strategy()`}</pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🤝 협력 주행 (Cooperative Driving)
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="space-y-4">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">차량 군집 주행 (Platooning)</h4>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 자동 군집 주행 시스템
class VehiclePlatooning:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.platoon_members = []
        self.leader_id = None
        
    def join_platoon(self, leader_vehicle):
        # 군집 합류 프로토콜
        join_request = {
            'type': 'PLATOON_JOIN_REQUEST',
            'vehicle_id': self.vehicle_id,
            'capabilities': self.get_vehicle_capabilities(),
            'desired_position': self.calculate_optimal_position()
        }
        
        # V2V 통신으로 리더에게 요청 전송
        response = self.send_v2v_message(leader_vehicle, join_request)
        
        if response['status'] == 'ACCEPTED':
            self.leader_id = leader_vehicle
            self.follow_leader()
    
    def follow_leader(self):
        while self.in_platoon:
            # 리더로부터 주행 정보 수신
            leader_state = self.receive_leader_state()
            
            # 최적 간격 유지 (CACC: Cooperative Adaptive Cruise Control)
            target_gap = self.calculate_safe_gap(leader_state.speed)
            current_gap = self.measure_gap_to_leader()
            
            # 제어 명령 계산
            acceleration = self.cacc_controller(target_gap, current_gap)
            self.apply_control(acceleration, leader_state.steering)
            
            time.sleep(0.1)  # 10Hz 제어 주기`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔒 사이버보안과 프라이버시
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h4 className="font-bold text-red-700 dark:text-red-400 mb-3">
              🛡️ 보안 위협
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>메시지 위조:</strong> 가짜 교통 정보 주입</li>
              <li>• <strong>중간자 공격:</strong> V2V 통신 가로채기</li>
              <li>• <strong>서비스 거부:</strong> 통신 채널 마비</li>
              <li>• <strong>프라이버시 침해:</strong> 위치 추적</li>
              <li>• <strong>차량 하이재킹:</strong> 원격 제어 탈취</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-3">
              🔐 보안 대책
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>PKI 인증:</strong> 디지털 인증서 기반</li>
              <li>• <strong>메시지 서명:</strong> 무결성 보장</li>
              <li>• <strong>익명화:</strong> 위치 프라이버시 보호</li>
              <li>• <strong>침입 탐지:</strong> 실시간 위협 모니터링</li>
              <li>• <strong>보안 업데이트:</strong> OTA 펌웨어 업데이트</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🌏 글로벌 표준화 현황
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">
              🇺🇸 미국 (DSRC)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              5.9GHz 대역, IEEE 802.11p 기반
            </p>
            <div className="mt-2 text-xs text-blue-600 dark:text-blue-400">
              →C-V2X로 전환 중
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-2">
              🇪🇺 유럽 (C-ITS)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              ETSI 표준, Hybrid 접근법
            </p>
            <div className="mt-2 text-xs text-green-600 dark:text-green-400">
              DSRC + C-V2X 병행
            </div>
          </div>
          
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h4 className="font-bold text-red-700 dark:text-red-400 mb-2">
              🇨🇳 중국 (C-V2X)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              5G 기반, 국가 주도 표준화
            </p>
            <div className="mt-2 text-xs text-red-600 dark:text-red-400">
              상용화 선도
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}