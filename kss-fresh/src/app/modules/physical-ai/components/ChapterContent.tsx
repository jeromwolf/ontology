'use client'

import React from 'react'
import { Play, Pause, RotateCw, Zap, Target, Activity } from 'lucide-react'

interface ChapterContentProps {
  chapterId: number
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const renderChapterContent = () => {
    switch (chapterId) {
      case 1:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>Physical AI 개요와 미래</h2>
            
            <h3>1. Physical AI란 무엇인가?</h3>
            <p>
              Physical AI는 현실 세계와 직접 상호작용하는 인공지능 시스템을 의미합니다. 
              디지털 환경에서만 작동하는 전통적인 AI와 달리, Physical AI는 센서, 로봇, 
              액추에이터를 통해 물리적 세계를 인식하고 조작합니다.
            </p>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg my-6">
              <h4 className="text-purple-900 dark:text-purple-100 font-semibold mb-3">
                젠슨 황의 COSMOS 비전
              </h4>
              <ul className="space-y-2">
                <li>• <strong>디지털 트윈</strong>: 물리 세계의 완벽한 디지털 복제</li>
                <li>• <strong>시뮬레이션 우선</strong>: 실제 세계 배포 전 가상 환경에서 학습</li>
                <li>• <strong>물리 법칙 통합</strong>: AI가 물리학을 이해하고 활용</li>
                <li>• <strong>실시간 적응</strong>: 환경 변화에 즉각 대응</li>
              </ul>
            </div>

            <h3>2. Digital AI vs Physical AI</h3>
            <div className="grid md:grid-cols-2 gap-6 my-6">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <h5 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">Digital AI</h5>
                <ul className="text-sm space-y-1">
                  <li>• 데이터와 정보 처리</li>
                  <li>• 패턴 인식과 예측</li>
                  <li>• 텍스트, 이미지, 음성 처리</li>
                  <li>• 소프트웨어 기반 작동</li>
                </ul>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                <h5 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">Physical AI</h5>
                <ul className="text-sm space-y-1">
                  <li>• 물리적 상호작용</li>
                  <li>• 센서 융합과 제어</li>
                  <li>• 로봇, 드론, 자율주행차</li>
                  <li>• 하드웨어-소프트웨어 통합</li>
                </ul>
              </div>
            </div>
          </div>
        )

      case 2:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>로보틱스와 제어 시스템</h2>
            
            <h3>1. 뉴턴역학 기초</h3>
            <p>
              로봇을 제어하기 위해서는 먼저 물체의 운동을 지배하는 기본 법칙을 이해해야 합니다.
              뉴턴의 운동 법칙은 모든 로봇 제어의 기초가 됩니다.
            </p>

            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-lg my-6">
              <h4 className="text-indigo-900 dark:text-indigo-100 font-semibold mb-4">
                뉴턴의 운동 법칙
              </h4>
              
              <div className="space-y-4">
                <div>
                  <h5 className="font-medium mb-2">제1법칙 (관성의 법칙)</h5>
                  <p className="text-sm mb-2">
                    외력이 작용하지 않는 한, 정지한 물체는 계속 정지하고 운동하는 물체는 등속직선운동을 계속한다.
                  </p>
                  <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto">
{`// 로봇의 관성 고려
if (externalForce === 0) {
  velocity = constant;  // 속도 유지
  position += velocity * deltaTime;
}`}
                  </pre>
                </div>

                <div>
                  <h5 className="font-medium mb-2">제2법칙 (가속도의 법칙)</h5>
                  <p className="text-sm mb-2">
                    물체의 가속도는 작용하는 힘에 비례하고 질량에 반비례한다. F = ma
                  </p>
                  <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto">
{`// 로봇 팔의 가속도 계산
function calculateAcceleration(force, mass) {
  return force / mass;  // a = F/m
}

// 토크와 각가속도
function calculateAngularAcceleration(torque, inertia) {
  return torque / inertia;  // α = τ/I
}`}
                  </pre>
                </div>

                <div>
                  <h5 className="font-medium mb-2">제3법칙 (작용-반작용의 법칙)</h5>
                  <p className="text-sm mb-2">
                    모든 작용에는 크기가 같고 방향이 반대인 반작용이 있다.
                  </p>
                  <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto">
{`// 로봇이 물체를 밀 때
robotForceOnObject = pushForce;
objectForceOnRobot = -pushForce;  // 반작용

// 보행 로봇의 지면 반력
groundReactionForce = -robotWeight;`}
                  </pre>
                </div>
              </div>
            </div>

            <h3>2. 로봇 운동학 (Kinematics)</h3>
            <p>
              운동학은 힘을 고려하지 않고 로봇의 위치, 속도, 가속도 관계를 다룹니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg my-4">
              <h5 className="font-semibold mb-3">순운동학 (Forward Kinematics)</h5>
              <p className="text-sm mb-3">관절 각도 → 엔드이펙터 위치</p>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`// 2-링크 로봇 팔의 순운동학
function forwardKinematics(theta1, theta2, L1, L2) {
  // 첫 번째 관절 위치
  x1 = L1 * cos(theta1);
  y1 = L1 * sin(theta1);
  
  // 엔드이펙터 위치
  x = x1 + L2 * cos(theta1 + theta2);
  y = y1 + L2 * sin(theta1 + theta2);
  
  return { x, y };
}`}
              </pre>
            </div>

            <h3>3. 로봇 동역학 (Dynamics)</h3>
            <p>
              동역학은 힘과 토크가 로봇의 운동에 미치는 영향을 분석합니다.
            </p>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg my-4">
              <h5 className="font-semibold mb-3">뉴턴-오일러 방정식</h5>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`// 로봇 링크의 동역학 방정식
class RobotLink {
  constructor(mass, inertia, length) {
    this.mass = mass;
    this.inertia = inertia;
    this.length = length;
  }
  
  // 뉴턴의 제2법칙 적용
  calculateForces(acceleration) {
    const linearForce = this.mass * acceleration;
    return linearForce;
  }
  
  // 회전 운동에 대한 오일러 방정식
  calculateTorque(angularAcceleration, angularVelocity) {
    const torque = this.inertia * angularAcceleration + 
                   cross(angularVelocity, this.inertia * angularVelocity);
    return torque;
  }
}`}
              </pre>
            </div>

            <h3>4. 제어 시스템과 물리학의 통합</h3>
            
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">PID 제어기에서의 물리학</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`class PIDController {
  constructor(Kp, Ki, Kd, mass) {
    this.Kp = Kp;  // 비례 게인
    this.Ki = Ki;  // 적분 게인
    this.Kd = Kd;  // 미분 게인
    this.mass = mass;  // 로봇 질량
    this.integral = 0;
    this.prevError = 0;
  }
  
  calculate(setpoint, current, dt) {
    const error = setpoint - current;
    
    // PID 계산
    const P = this.Kp * error;
    this.integral += error * dt;
    const I = this.Ki * this.integral;
    const D = this.Kd * (error - this.prevError) / dt;
    
    // 제어 출력 (힘)
    const force = P + I + D;
    
    // 뉴턴 제2법칙으로 가속도 계산
    const acceleration = force / this.mass;
    
    this.prevError = error;
    return { force, acceleration };
  }
}`}
              </pre>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">실제 응용 예제: 로봇 팔 제어</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`// 중력 보상을 포함한 로봇 팔 제어
class RobotArmController {
  constructor(links) {
    this.links = links;
    this.gravity = 9.81;
  }
  
  // 중력 토크 계산 (뉴턴역학)
  calculateGravityCompensation(jointAngles) {
    const gravityTorques = [];
    
    for (let i = 0; i < this.links.length; i++) {
      const link = this.links[i];
      const angle = jointAngles[i];
      
      // 중력에 의한 토크: τ = r × F = r × mg
      const torque = link.length * link.mass * this.gravity * 
                     Math.cos(angle) / 2;  // 질량중심이 링크 중앙
      
      gravityTorques.push(torque);
    }
    
    return gravityTorques;
  }
  
  // 운동 방정식: τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
  calculateTorques(q, qDot, qDDot) {
    const M = this.calculateMassMatrix(q);      // 관성 행렬
    const C = this.calculateCoriolisMatrix(q, qDot);  // 코리올리/원심력
    const G = this.calculateGravityCompensation(q);   // 중력
    
    // τ = Mq̈ + Cq̇ + G
    const torques = [];
    for (let i = 0; i < q.length; i++) {
      let torque = G[i];
      for (let j = 0; j < q.length; j++) {
        torque += M[i][j] * qDDot[j] + C[i][j] * qDot[j];
      }
      torques.push(torque);
    }
    
    return torques;
  }
}`}
              </pre>
            </div>

            <h3>5. 시뮬레이션과 실습</h3>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <p className="mb-3">
                뉴턴역학을 기반으로 한 로봇 제어를 직접 체험해보세요:
              </p>
              <ul className="space-y-2">
                <li>• <strong>진자 시뮬레이터</strong>: 단진자와 이중진자의 운동</li>
                <li>• <strong>2D 로봇 팔</strong>: 순운동학과 역운동학 시각화</li>
                <li>• <strong>충돌 시뮬레이션</strong>: 운동량 보존 법칙</li>
                <li>• <strong>중력 보상 데모</strong>: 로봇 팔의 중력 영향 제어</li>
              </ul>
            </div>
          </div>
        )

      case 3:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>컴퓨터 비전과 인식</h2>
            
            <h3>1. 실시간 객체 탐지</h3>
            <p>
              Physical AI가 현실 세계와 상호작용하기 위해서는 주변 환경을 정확하게 인식해야 합니다.
            </p>

            <div className="bg-teal-50 dark:bg-teal-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">YOLO 실시간 탐지</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`import cv2
import torch

class YOLODetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = 0.45  # 신뢰도 임계값
        
    def detect(self, frame):
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \\
                             int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, detections`}
              </pre>
            </div>
          </div>
        )

      case 4:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>강화학습과 제어</h2>
            
            <h3>1. 강화학습 기초</h3>
            <p>
              Physical AI는 시행착오를 통해 최적의 행동을 학습합니다. 
              강화학습은 로봇이 환경과 상호작용하며 스스로 학습하는 핵심 기술입니다.
            </p>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">Q-Learning 예제</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.99):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)  # 탐색
        return np.argmax(self.q_table[state])  # 활용
        
    def update(self, state, action, reward, next_state):
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q
        
        self.epsilon *= self.epsilon_decay`}
              </pre>
            </div>
          </div>
        )

      case 5:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>디지털 트윈과 시뮬레이션</h2>
            
            <h3>1. 디지털 트윈 개념</h3>
            <p>
              디지털 트윈은 물리적 시스템의 실시간 디지털 복제본으로, 
              Physical AI의 핵심 기술입니다.
            </p>

            <div className="bg-cyan-50 dark:bg-cyan-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">디지털 트윈 구현</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`class DigitalTwin:
    def __init__(self, physical_system):
        self.physical_state = physical_system.get_state()
        self.simulation_model = self.create_model()
        self.sync_interval = 0.1  # 100ms
        
    def synchronize(self):
        # 물리 시스템에서 센서 데이터 수집
        sensor_data = self.collect_sensor_data()
        
        # 디지털 모델 업데이트
        self.update_model(sensor_data)
        
        # 예측 및 최적화
        predictions = self.predict_future_states()
        optimizations = self.optimize_performance()
        
        return predictions, optimizations`}
              </pre>
            </div>
          </div>
        )

      case 6:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>센서 융합과 상태 추정</h2>
            
            <h3>1. 칼만 필터</h3>
            <p>
              여러 센서의 노이즈가 있는 측정값을 융합하여 정확한 상태를 추정합니다.
            </p>

            <div className="bg-pink-50 dark:bg-pink-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">확장 칼만 필터 (EKF)</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`class ExtendedKalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x  # 상태 벡터 차원
        self.dim_z = dim_z  # 측정 벡터 차원
        
        self.x = np.zeros(dim_x)  # 상태 추정값
        self.P = np.eye(dim_x)     # 오차 공분산
        self.Q = np.eye(dim_x) * 0.1  # 프로세스 노이즈
        self.R = np.eye(dim_z) * 1.0  # 측정 노이즈
        
    def predict(self, f, F_jacobian):
        # 예측 단계
        self.x = f(self.x)  # 비선형 상태 전이
        F = F_jacobian(self.x)  # 야코비안
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z, h, H_jacobian):
        # 업데이트 단계
        y = z - h(self.x)  # 혁신
        H = H_jacobian(self.x)  # 측정 야코비안
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # 칼만 게인
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P`}
              </pre>
            </div>
          </div>
        )

      case 7:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>엣지 AI와 실시간 처리</h2>
            
            <h3>1. 엣지 컴퓨팅의 중요성</h3>
            <p>
              Physical AI는 밀리초 단위의 반응이 필요하므로, 
              클라우드가 아닌 엣지에서 처리해야 합니다.
            </p>

            <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">모델 경량화 기법</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`# TensorFlow Lite 변환
import tensorflow as tf

def quantize_model(model_path):
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 대표 데이터셋 생성
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            yield [data]
    
    # 변환기 설정
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    
    # INT8 양자화
    tflite_model = converter.convert()
    
    return tflite_model`}
              </pre>
            </div>
          </div>
        )

      case 8:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>Physical AI 응용 사례</h2>
            
            <h3>1. 산업용 로봇</h3>
            <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">협동 로봇 (Cobot) 시스템</h4>
              <ul className="space-y-2">
                <li>• <strong>Universal Robots</strong>: 힘 제어 기반 안전 협업</li>
                <li>• <strong>ABB YuMi</strong>: 듀얼 암 정밀 조립</li>
                <li>• <strong>KUKA LBR iiwa</strong>: 7축 민감 로봇</li>
              </ul>
            </div>

            <h3>2. 자율주행 시스템</h3>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">Tesla FSD (Full Self-Driving)</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`# Tesla의 비전 기반 접근
class TeslaFSD:
    def __init__(self):
        self.cameras = [Camera() for _ in range(8)]
        self.neural_net = HydraNet()  # 다중 작업 신경망
        
    def process_frame(self):
        # 8개 카메라에서 동시 입력
        images = [cam.capture() for cam in self.cameras]
        
        # 단일 신경망으로 다중 작업 처리
        outputs = self.neural_net.predict(images)
        
        return {
            'objects': outputs['detection'],
            'lanes': outputs['lane_detection'],
            'depth': outputs['depth_estimation'],
            'motion': outputs['optical_flow']
        }`}
              </pre>
            </div>

            <h3>3. 드론과 UAV</h3>
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">자율 비행 제어</h4>
              <ul className="space-y-2">
                <li>• <strong>DJI</strong>: 장애물 회피와 자동 귀환</li>
                <li>• <strong>Skydio</strong>: AI 기반 자율 추적</li>
                <li>• <strong>Wing (Google)</strong>: 배송 드론 네비게이션</li>
              </ul>
            </div>

            <h3>4. 휴머노이드 로봇</h3>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">Boston Dynamics Atlas</h4>
              <p className="mb-3">
                전신 제어와 동적 균형을 통한 파쿠르, 백플립 등 고난도 동작 수행
              </p>
              <ul className="space-y-2">
                <li>• <strong>모델 예측 제어</strong>: 100Hz 전신 궤적 최적화</li>
                <li>• <strong>접촉 감지</strong>: 발과 손의 힘 센서</li>
                <li>• <strong>시각 인식</strong>: 지형 매핑과 장애물 감지</li>
              </ul>
            </div>
          </div>
        )

      case 9:
        return (
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <h2>메타버스와 Physical AI 통합</h2>
            
            <h3>1. NVIDIA Omniverse와 Physical AI</h3>
            <p>
              Omniverse는 물리적으로 정확한 디지털 트윈을 생성하고 
              AI를 훈련시키는 플랫폼입니다.
            </p>

            <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">Omniverse 핵심 기능</h4>
              <ul className="space-y-3">
                <li>
                  <strong>PhysX 5.0</strong>: 실시간 물리 시뮬레이션
                  <pre className="bg-white dark:bg-gray-900 p-2 rounded text-sm mt-2">
{`// 유체 시뮬레이션
physx::PxFluidSystem* fluid = physics->createFluidSystem();
fluid->setViscosity(0.001f);  // 물의 점성
fluid->setSurfaceTension(0.0728f);  // 표면 장력`}
                  </pre>
                </li>
                <li>
                  <strong>Isaac Sim</strong>: 로봇 시뮬레이션 환경
                  <pre className="bg-white dark:bg-gray-900 p-2 rounded text-sm mt-2">
{`# Isaac Gym에서 로봇 훈련
env = gym.create_env(SimType.PhysX, num_envs=1024)
robot = env.add_actor("franka_panda.usd")
robot.train_with_rl(PPO_config)`}
                  </pre>
                </li>
                <li>
                  <strong>RTX 실시간 레이트레이싱</strong>: 사실적인 조명과 반사
                </li>
              </ul>
            </div>

            <h3>2. 디지털 트윈 도시</h3>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">스마트 시티 시뮬레이션</h4>
              <p className="mb-4">
                도시 전체를 디지털 트윈으로 구현하여 교통, 에너지, 안전을 최적화
              </p>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded">
                  <h5 className="font-medium mb-2">교통 최적화</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 실시간 신호등 제어</li>
                    <li>• 자율주행차 경로 조정</li>
                    <li>• 대중교통 스케줄링</li>
                  </ul>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded">
                  <h5 className="font-medium mb-2">에너지 관리</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 스마트 그리드 제어</li>
                    <li>• 건물 에너지 최적화</li>
                    <li>• 재생 에너지 예측</li>
                  </ul>
                </div>
              </div>
            </div>

            <h3>3. XR과 Physical AI</h3>
            <div className="bg-cyan-50 dark:bg-cyan-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">증강현실 로봇 제어</h4>
              <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`// Unity + ROS2 통합
public class ARRobotController : MonoBehaviour {
    private ROS2UnityComponent ros2;
    private ARRaycastManager raycastManager;
    
    void Start() {
        ros2 = GetComponent<ROS2UnityComponent>();
        ros2.CreateNode("ar_robot_controller");
    }
    
    void OnTouchScreen(Vector2 touchPos) {
        // AR 공간에서 터치 위치를 3D 좌표로 변환
        List<ARRaycastHit> hits = new List<ARRaycastHit>();
        raycastManager.Raycast(touchPos, hits);
        
        if (hits.Count > 0) {
            Vector3 worldPos = hits[0].pose.position;
            
            // ROS2로 로봇 이동 명령 전송
            var moveGoal = new MoveBaseGoal();
            moveGoal.target_pose.pose.position = worldPos;
            ros2.Publish("/move_base/goal", moveGoal);
        }
    }
}`}
              </pre>
            </div>

            <h3>4. COSMOS 비전 실현</h3>
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-lg my-6">
              <h4 className="font-semibold mb-3">Physical AI의 미래</h4>
              <p className="mb-4">
                젠슨 황이 제시한 COSMOS는 물리 세계 전체를 시뮬레이션하고 
                AI가 현실에서 행동하기 전에 가상으로 학습하는 플랫폼입니다.
              </p>
              
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center flex-shrink-0">1</div>
                  <div>
                    <strong>Foundation World Model</strong>
                    <p className="text-sm mt-1">물리 법칙을 이해하는 거대 AI 모델</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center flex-shrink-0">2</div>
                  <div>
                    <strong>Synthetic Data Generation</strong>
                    <p className="text-sm mt-1">현실보다 다양한 시뮬레이션 데이터</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center flex-shrink-0">3</div>
                  <div>
                    <strong>Zero-Shot Transfer</strong>
                    <p className="text-sm mt-1">시뮬레이션에서 현실로 직접 전이</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg mt-6">
              <p className="text-sm">
                <strong>💡 실습 제안:</strong> Omniverse Physics Lab 시뮬레이터에서 
                물리 법칙과 AI 제어를 통합한 메타버스 환경을 직접 체험해보세요!
              </p>
            </div>
          </div>
        )

      default:
        return <div>챕터를 찾을 수 없습니다.</div>
    }
  }

  return (
    <div>
      {renderChapterContent()}
    </div>
  )
}