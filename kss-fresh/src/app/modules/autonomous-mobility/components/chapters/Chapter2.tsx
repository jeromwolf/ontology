'use client'

import { Eye, Cpu, Radio, MapPin, Navigation } from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          센서 융합과 인지 시스템
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            자율주행차의 "눈"과 "뇌"에 해당하는 센서 시스템과 인지 알고리즘입니다.
            LiDAR의 정밀한 3D 스캔, 카메라의 풍부한 시각 정보, 레이더의 전천후 감지 능력을
            융합하여 인간보다 뛰어난 인지 성능을 구현합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          📡 핵심 센서 기술
        </h3>
        
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Eye className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              <h4 className="font-bold text-gray-900 dark:text-white">LiDAR</h4>
            </div>
            <div className="space-y-3">
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">원리</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  레이저 펄스로 거리 측정 (Time-of-Flight)
                </p>
              </div>
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">장점</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  높은 정확도 (±2cm), 3D 포인트클라우드
                </p>
              </div>
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">기업</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Velodyne, Luminar, Ouster
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Cpu className="w-8 h-8 text-green-600 dark:text-green-400" />
              <h4 className="font-bold text-gray-900 dark:text-white">Camera</h4>
            </div>
            <div className="space-y-3">
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">원리</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  RGB 이미지 + Stereo Vision
                </p>
              </div>
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">장점</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  색상 정보, 표지판/신호등 인식
                </p>
              </div>
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">AI 모델</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  YOLO, Faster R-CNN, SegNet
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Radio className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              <h4 className="font-bold text-gray-900 dark:text-white">Radar</h4>
            </div>
            <div className="space-y-3">
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">원리</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  FMCW 주파수 변조 전파
                </p>
              </div>
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">장점</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  전천후, 속도 측정, 장거리
                </p>
              </div>
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">주파수</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  77GHz, 79GHz (mmWave)
                </p>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🧠 센서 퓨전 알고리즘
        </h3>
        
        {/* 칼만 필터 상세 설명 추가 */}
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            📊 칼만 필터 (Kalman Filter) - 센서 퓨전의 핵심
          </h4>
          
          <div className="space-y-4">
            <p className="text-gray-700 dark:text-gray-300">
              칼만 필터는 <strong>노이즈가 있는 센서 데이터로부터 더 정확한 상태를 추정</strong>하는 
              최적 상태 추정 알고리즘입니다. 자율주행에서는 여러 센서의 불확실한 측정값을 융합하여 
              차량과 주변 객체의 정확한 위치와 속도를 추정하는 데 필수적입니다.
            </p>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 dark:text-white mb-2">🔄 칼만 필터의 2단계 순환 과정</h5>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-purple-50 dark:bg-purple-900/30 rounded-lg p-4">
                  <h6 className="font-bold text-purple-700 dark:text-purple-300 mb-2">1. 예측 단계 (Prediction)</h6>
                  <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                    <li>• 이전 상태를 기반으로 다음 상태 예측</li>
                    <li>• 운동 모델 사용 (예: 등속 운동)</li>
                    <li>• 불확실성(공분산) 증가</li>
                  </ul>
                </div>
                <div className="bg-indigo-50 dark:bg-indigo-900/30 rounded-lg p-4">
                  <h6 className="font-bold text-indigo-700 dark:text-indigo-300 mb-2">2. 업데이트 단계 (Update)</h6>
                  <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                    <li>• 센서 측정값으로 예측값 보정</li>
                    <li>• 칼만 이득(Kalman Gain) 계산</li>
                    <li>• 불확실성 감소</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 dark:text-white mb-2">💡 칼만 이득 (Kalman Gain)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                예측값과 측정값 중 어느 것을 더 신뢰할지 결정하는 가중치입니다:
              </p>
              <div className="bg-gray-100 dark:bg-gray-900 rounded p-3">
                <code className="text-xs font-mono">
                  K = P_predicted / (P_predicted + R_measurement)<br/>
                  • K → 1: 측정값을 더 신뢰 (센서 정확도 높음)<br/>
                  • K → 0: 예측값을 더 신뢰 (센서 노이즈 많음)
                </code>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 dark:text-white mb-2">🚗 자율주행에서의 실제 응용</h5>
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 차량 추적을 위한 칼만 필터 구현
class VehicleKalmanFilter:
    def __init__(self):
        # 상태 벡터: [x위치, y위치, x속도, y속도]
        self.state = np.array([0, 0, 0, 0])
        
        # 상태 전이 행렬 (등속 운동 모델)
        self.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        
        # 측정 행렬 (위치만 측정)
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])
        
        # 프로세스 노이즈 (가속도 불확실성)
        self.Q = np.eye(4) * 0.1
        
        # 측정 노이즈 (센서 정확도)
        self.R = np.eye(2) * 0.5
        
        # 오차 공분산 행렬
        self.P = np.eye(4) * 100
    
    def predict(self):
        """예측 단계: 운동 모델로 다음 상태 예측"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        """업데이트 단계: 센서 측정값으로 보정"""
        # 혁신(Innovation) = 측정값 - 예측값
        y = measurement - self.H @ self.state
        
        # 혁신 공분산
        S = self.H @ self.P @ self.H.T + self.R
        
        # 칼만 이득 계산
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 상태 업데이트
        self.state = self.state + K @ y
        
        # 오차 공분산 업데이트
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state`}</pre>
            </div>
            
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 dark:text-white mb-2">🎯 센서 퓨전에서의 장점</h5>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>노이즈 제거:</strong> 각 센서의 측정 오차를 효과적으로 필터링</li>
                <li>• <strong>예측 능력:</strong> 센서 데이터가 일시적으로 없어도 상태 추정 가능</li>
                <li>• <strong>다중 센서 통합:</strong> LiDAR, 카메라, 레이더 데이터를 최적으로 결합</li>
                <li>• <strong>실시간 처리:</strong> 계산이 간단하여 30Hz 이상의 실시간 처리 가능</li>
                <li>• <strong>불확실성 추정:</strong> 추정값의 신뢰도를 함께 제공</li>
              </ul>
            </div>
            
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 dark:text-white mb-2">🔧 확장 칼만 필터 (EKF)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                레이더와 같은 비선형 센서 데이터를 처리하기 위해 확장 칼만 필터를 사용합니다.
                레이더는 극좌표계(거리, 각도)로 측정하므로 직교좌표계로 변환 시 비선형성이 발생합니다:
              </p>
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs mt-2 overflow-x-auto">
{`# EKF에서 레이더 데이터 처리
def radar_measurement_function(state):
    """비선형 측정 함수 h(x)"""
    px, py, vx, vy = state
    rho = sqrt(px**2 + py**2)      # 거리
    phi = atan2(py, px)             # 각도
    rho_dot = (px*vx + py*vy)/rho  # 거리 변화율
    return [rho, phi, rho_dot]`}</pre>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">1️⃣ 데이터 레벨 융합</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 포인트클라우드 + RGB 이미지 융합
def sensor_fusion_early(lidar_points, camera_image):
    # 좌표계 변환
    projected_points = project_lidar_to_camera(lidar_points)
    
    # RGB-D 생성
    depth_map = create_depth_map(projected_points)
    rgbd_image = np.concatenate([camera_image, depth_map], axis=2)
    
    return rgbd_image`}</pre>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">2️⃣ 특징 레벨 융합</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 각 센서별 특징 추출 후 융합
def sensor_fusion_feature(lidar_features, camera_features, radar_features):
    # Attention 메커니즘으로 가중치 계산
    attention_weights = calculate_attention([lidar_features, camera_features, radar_features])
    
    # 가중 평균으로 융합
    fused_features = weighted_average(features, attention_weights)
    
    return fused_features`}</pre>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">3️⃣ 결정 레벨 융합</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 각 센서의 독립적 판단을 종합
def sensor_fusion_decision(detections_lidar, detections_camera, detections_radar):
    # Kalman Filter로 상태 추정
    for detection in all_detections:
        track = associate_with_existing_track(detection)
        if track:
            track.update(detection)
        else:
            create_new_track(detection)
    
    return validated_tracks`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🗺️ HD맵과 로컬라이제이션
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              <MapPin className="inline w-5 h-5 mr-2" />
              HD맵 구성 요소
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>Lane Network:</strong> 차선 중심선, 경계선</li>
              <li>• <strong>Traffic Elements:</strong> 신호등, 표지판</li>
              <li>• <strong>Road Features:</strong> 연석, 가드레일</li>
              <li>• <strong>Semantic Info:</strong> 속도제한, 우선순위</li>
              <li>• <strong>정확도:</strong> 센티미터급 (±10cm)</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              <Navigation className="inline w-5 h-5 mr-2" />
              SLAM 기술
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>Visual SLAM:</strong> ORB-SLAM, VINS</li>
              <li>• <strong>LiDAR SLAM:</strong> LOAM, LeGO-LOAM</li>
              <li>• <strong>Multi-modal:</strong> 센서 융합 SLAM</li>
              <li>• <strong>Loop Closure:</strong> 누적 오차 보정</li>
              <li>• <strong>실시간성:</strong> 30Hz 이상 처리</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🔬 최신 연구 동향
        </h3>
        
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-3">
                🧪 Solid-State LiDAR
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                기계식 회전 부품 제거로 내구성 향상
              </p>
              <div className="flex gap-2">
                <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-xs rounded">Luminar</span>
                <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-xs rounded">Aeye</span>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-indigo-700 dark:text-indigo-400 mb-3">
                🤖 Neuromorphic Vision
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                이벤트 기반 시각 센서로 초저전력 구현
              </p>
              <div className="flex gap-2">
                <span className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 text-xs rounded">Prophesee</span>
                <span className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 text-xs rounded">Intel</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}