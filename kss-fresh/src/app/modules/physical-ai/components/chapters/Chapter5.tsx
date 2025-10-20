'use client';

import React from 'react';
import { Cpu, Zap, Network, Cloud, Server, WifiOff } from 'lucide-react';

export default function Chapter5() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-2xl p-8 mb-8 border border-cyan-200 dark:border-cyan-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-14 h-14 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
            <Cpu className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white m-0">
            IoT & Edge Computing
          </h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0 leading-relaxed">
          Physical AI는 실시간으로 동작해야 합니다. 클라우드로 데이터를 보내고 기다릴 수 없습니다.
          <strong className="text-cyan-600 dark:text-cyan-400"> Edge AI</strong>는 로봇이 스스로 생각하고 즉각 반응하게 만듭니다.
        </p>
      </div>

      {/* Introduction */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <WifiOff className="text-cyan-600" />
          왜 Edge AI가 필요한가?
        </h2>

        <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold text-red-700 dark:text-red-400 mt-0">🚨 클라우드의 치명적 한계</h3>
          <ul className="space-y-2 mb-0">
            <li><strong>레이턴시 (Latency)</strong>: 클라우드 왕복 시간 100-300ms → 자율주행차는 10ms 이내 판단 필요</li>
            <li><strong>네트워크 의존성</strong>: WiFi 끊기면 로봇 멈춤 → 공장, 병원, 우주에서는 치명적</li>
            <li><strong>대역폭 비용</strong>: 4K 카메라 8개 = 시간당 100GB → 클라우드 비용 폭탄</li>
            <li><strong>프라이버시</strong>: 가정용 로봇이 모든 영상을 서버에 전송? → 개인정보 유출 위험</li>
          </ul>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border-2 border-red-300 dark:border-red-700">
            <div className="flex items-center gap-2 mb-3">
              <Cloud className="text-red-500" />
              <h3 className="text-xl font-bold mt-0">Cloud AI (전통 방식)</h3>
            </div>
            <ul className="space-y-1 text-sm mb-0">
              <li>✅ 강력한 컴퓨팅 파워 (GPU 클러스터)</li>
              <li>✅ 무제한 스토리지</li>
              <li>❌ 레이턴시 100-300ms (너무 느림)</li>
              <li>❌ 네트워크 필수 (끊기면 작동 불가)</li>
              <li>❌ 대역폭 비용 고가</li>
              <li>❌ 프라이버시 위험</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border-2 border-green-300 dark:border-green-700">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="text-green-500" />
              <h3 className="text-xl font-bold mt-0">Edge AI (Physical AI 필수)</h3>
            </div>
            <ul className="space-y-1 text-sm mb-0">
              <li>✅ 초저지연 1-10ms (실시간 반응)</li>
              <li>✅ 오프라인 작동 (네트워크 불필요)</li>
              <li>✅ 대역폭 비용 제로</li>
              <li>✅ 프라이버시 보장 (로컬 처리)</li>
              <li>❌ 제한된 컴퓨팅 파워</li>
              <li>❌ 모델 최적화 필수 (경량화)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Edge AI Hardware */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Cpu className="text-orange-600" />
          Edge AI 칩셋 - 로봇의 두뇌
        </h2>

        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-2xl font-bold mb-4">🏆 NVIDIA Jetson Series - 업계 표준</h3>
          <p className="text-lg mb-4">
            테슬라, 보스턴 다이내믹스, NASA가 사용하는 <strong>엣지 AI 플랫폼의 절대 강자</strong>입니다.
          </p>

          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
              <h4 className="text-lg font-bold text-green-600 mb-2">Jetson Nano</h4>
              <div className="text-3xl font-bold mb-2">$59</div>
              <div className="text-sm space-y-1">
                <div>0.5 TFLOPS (FP16)</div>
                <div>4GB RAM</div>
                <div>5W 전력</div>
                <div className="text-green-600 font-semibold mt-2">입문용 / 교육</div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center border-2 border-blue-500">
              <h4 className="text-lg font-bold text-blue-600 mb-2">Jetson Orin Nano</h4>
              <div className="text-3xl font-bold mb-2">$499</div>
              <div className="text-sm space-y-1">
                <div>40 TOPS (INT8)</div>
                <div>8GB RAM</div>
                <div>15W 전력</div>
                <div className="text-blue-600 font-semibold mt-2">드론 / 로봇 팔</div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center border-2 border-purple-500">
              <h4 className="text-lg font-bold text-purple-600 mb-2">Jetson AGX Orin</h4>
              <div className="text-3xl font-bold mb-2">$1,999</div>
              <div className="text-sm space-y-1">
                <div>275 TOPS (INT8)</div>
                <div>64GB RAM</div>
                <div>60W 전력</div>
                <div className="text-purple-600 font-semibold mt-2">자율주행 / 휴머노이드</div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">실제 사용 사례:</h4>
            <ul className="space-y-1 text-sm mb-0">
              <li>🚗 <strong>Tesla Bot (Optimus)</strong> - 커스텀 Jetson 기반 칩</li>
              <li>🤖 <strong>Boston Dynamics Atlas</strong> - Jetson AGX Orin 64GB</li>
              <li>🚁 <strong>Skydio 드론</strong> - Jetson Xavier NX (자율 장애물 회피)</li>
              <li>🏭 <strong>삼성전자 스마트 팩토리</strong> - Jetson Orin Nano (검사 로봇)</li>
            </ul>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-blue-600 mb-3">Google Coral TPU</h3>
            <div className="space-y-2 text-sm">
              <div><strong>가격</strong>: $59.99 (USB Accelerator)</div>
              <div><strong>성능</strong>: 4 TOPS @ 2W (전력 효율 최고)</div>
              <div><strong>특징</strong>: TensorFlow Lite 전용, MobileNet 최적화</div>
              <div><strong>용도</strong>: 저전력 IoT (스마트 도어벨, CCTV)</div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-purple-600 mb-3">Intel Movidius VPU</h3>
            <div className="space-y-2 text-sm">
              <div><strong>가격</strong>: $79 (Neural Compute Stick 2)</div>
              <div><strong>성능</strong>: 1 TOPS @ 1W</div>
              <div><strong>특징</strong>: OpenVINO 툴킷, USB 스틱 형태</div>
              <div><strong>용도</strong>: 프로토타이핑, 엣지 비전 AI</div>
            </div>
          </div>
        </div>
      </section>

      {/* Real-time Inference */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Zap className="text-yellow-600" />
          실시간 추론 (Real-time Inference)
        </h2>

        <p className="text-lg mb-4">
          클라우드 AI 모델을 그대로 엣지에서 실행하면 <strong className="text-red-600">너무 느립니다</strong>.
          모델을 <strong className="text-green-600">경량화(Optimization)</strong>해야 합니다.
        </p>

        <div className="bg-cyan-50 dark:bg-cyan-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🔧 모델 최적화 3단계</h3>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-blue-600 mb-2">1️⃣ 양자화 (Quantization)</h4>
              <p className="text-sm mb-2">
                32비트 부동소수점 → 8비트 정수로 변환 (정확도 1-2% 손실, 속도 4배 향상)
              </p>
              <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# PyTorch 양자화
import torch
from torch.quantization import quantize_dynamic

# 원본 모델 (FP32)
model = MyNeuralNetwork()

# 동적 양자화 (INT8)
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},  # 양자화할 레이어
    dtype=torch.qint8
)

# 크기: 100MB → 25MB (75% 감소)
# 속도: 50ms → 12ms (4배 빨라짐)`}
              </pre>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-green-600 mb-2">2️⃣ 프루닝 (Pruning) - 불필요한 뉴런 제거</h4>
              <p className="text-sm mb-2">
                중요하지 않은 가중치를 0으로 만들어 모델 경량화 (50-90% 파라미터 제거 가능)
              </p>
              <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# PyTorch Pruning
import torch.nn.utils.prune as prune

# L1 Unstructured Pruning (50% 가중치 제거)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.5)

# Permanent pruning (실제로 제거)
prune.remove(module, 'weight')

# 결과: 파라미터 50% 감소, 속도 30% 향상`}
              </pre>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-purple-600 mb-2">3️⃣ 지식 증류 (Knowledge Distillation)</h4>
              <p className="text-sm mb-2">
                큰 Teacher 모델이 작은 Student 모델을 가르침 (정확도 유지하며 크기 10배 감소)
              </p>
              <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# Teacher-Student Distillation
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets (Teacher의 확률 분포)
        soft_loss = nn.KLDivLoss()(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        )

        # Hard targets (실제 레이블)
        hard_loss = F.cross_entropy(student_logits, labels)

        return 0.7 * soft_loss + 0.3 * hard_loss

# 예시: GPT-4 (Teacher) → DistilGPT (Student)
# 1750억 파라미터 → 82억 파라미터 (20배 감소)
# 성능: 97% 유지, 추론 속도: 10배 빨라짐`}
              </pre>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-6 rounded-lg">
          <h3 className="text-xl font-bold text-green-700 dark:text-green-400 mt-0">🎯 실전 사례: Tesla FSD의 모델 최적화</h3>
          <ul className="space-y-2 mb-0">
            <li><strong>원본 모델</strong>: ResNet-101 + Transformer (8GB VRAM 필요)</li>
            <li><strong>최적화 후</strong>: 커스텀 INT8 모델 (HW4.0 칩에서 실행 가능)</li>
            <li><strong>추론 속도</strong>: 36 FPS (27ms/frame) - 8개 카메라 동시 처리</li>
            <li><strong>전력 소비</strong>: 72W (Jetson AGX Orin 수준)</li>
          </ul>
        </div>
      </section>

      {/* IoT Communication */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Network className="text-indigo-600" />
          IoT 통신 프로토콜 - 로봇 간 대화
        </h2>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-blue-600 mb-3">MQTT (Message Queue Telemetry Transport)</h3>
            <div className="space-y-2 text-sm mb-4">
              <div><strong>특징</strong>: Publish-Subscribe 모델, 초경량 (2KB 메모리)</div>
              <div><strong>레이턴시</strong>: 10-50ms (로컬 네트워크)</div>
              <div><strong>용도</strong>: IoT 센서 네트워크, 로봇 군집 제어</div>
            </div>

            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# Python MQTT (paho-mqtt)
import paho.mqtt.client as mqtt

class RobotFleet:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.connect("localhost", 1883)

    def on_message(self, client, userdata, msg):
        # 다른 로봇의 위치 정보 수신
        data = json.loads(msg.payload)
        print(f"Robot {data['id']}: {data['position']}")

    def publish_position(self, x, y):
        self.client.publish(
            "fleet/position",
            json.dumps({"id": self.robot_id, "position": [x, y]})
        )

# Amazon Warehouse에서 수천 대의 로봇이 MQTT로 통신`}
            </pre>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-purple-600 mb-3">ROS 2 DDS (Data Distribution Service)</h3>
            <div className="space-y-2 text-sm mb-4">
              <div><strong>특징</strong>: 로봇 전용 미들웨어, Peer-to-Peer</div>
              <div><strong>레이턴시</strong>: 1-5ms (초저지연)</div>
              <div><strong>용도</strong>: 자율주행, 드론, 휴머노이드</div>
            </div>

            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# ROS 2 Publisher (C++)
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class CameraNode : public rclcpp::Node {
public:
    CameraNode() : Node("camera_node") {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera/image", 10
        );

        timer_ = this->create_wall_timer(
            33ms,  // 30 FPS
            std::bind(&CameraNode::publish_frame, this)
        );
    }

private:
    void publish_frame() {
        auto msg = sensor_msgs::msg::Image();
        // 카메라 프레임 퍼블리시
        publisher_->publish(msg);
    }
};

// Boston Dynamics Spot, Tesla Bot 모두 ROS 2 사용`}
            </pre>
          </div>
        </div>
      </section>

      {/* Edge vs Cloud Architecture */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Server className="text-teal-600" />
          하이브리드 아키텍처 - 최고의 선택
        </h2>

        <p className="text-lg mb-4">
          실전에서는 <strong className="text-purple-600">Edge + Cloud 하이브리드</strong>를 사용합니다.
          실시간 작업은 엣지에서, 학습과 업데이트는 클라우드에서.
        </p>

        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-2xl font-bold mb-4">🏗️ Tesla의 하이브리드 아키텍처</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-green-400">
              <h4 className="font-bold text-green-600 mb-2">Edge (차량 내부 - HW4.0 칩)</h4>
              <ul className="text-sm space-y-1 mb-0">
                <li>✅ 실시간 추론 (1-10ms)</li>
                <li>✅ 오프라인 자율주행 (네트워크 불필요)</li>
                <li>✅ 8개 카메라 동시 처리 (36 FPS)</li>
                <li>✅ 장애물 회피, 차선 유지</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-blue-400">
              <h4 className="font-bold text-blue-600 mb-2">Cloud (Tesla Dojo 슈퍼컴퓨터)</h4>
              <ul className="text-sm space-y-1 mb-0">
                <li>✅ 전 세계 차량 데이터 수집</li>
                <li>✅ 모델 재학습 (ExaFLOP 연산)</li>
                <li>✅ 새 모델 Over-the-Air 업데이트</li>
                <li>✅ Fleet Learning (집단 지능)</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-white dark:bg-gray-800 p-4 rounded-lg">
            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto mb-0">
{`# 하이브리드 아키텍처 구현
class HybridAISystem:
    def __init__(self):
        self.edge_model = self.load_edge_model()  # 경량화된 모델
        self.cloud_client = CloudAPIClient()

    def process_frame(self, frame):
        # 1. Edge에서 실시간 추론
        predictions = self.edge_model.infer(frame)

        # 2. 불확실한 케이스만 Cloud로 전송
        if predictions['confidence'] < 0.7:
            cloud_result = self.cloud_client.query(frame)
            self.log_for_training(frame, cloud_result)  # 재학습용 데이터
            return cloud_result

        return predictions

    def update_model(self):
        # Cloud에서 새 모델 다운로드 (OTA 업데이트)
        new_model = self.cloud_client.download_latest_model()
        self.edge_model = new_model

# 결과: 99.9% 케이스는 Edge에서 처리 (빠름)
#       0.1% 어려운 케이스만 Cloud 활용 (정확함)`}
            </pre>
          </div>
        </div>
      </section>

      {/* Real-world Implementation */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Zap className="text-orange-600" />
          실전 구현: Jetson에서 YOLO 실행
        </h2>

        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-bold mb-4">🎯 엔드-투-엔드 예제: 실시간 객체 탐지 로봇</h3>

          <pre className="bg-gray-900 text-gray-100 p-4 rounded text-sm overflow-x-auto">
{`# requirements.txt
# ultralytics==8.0.0
# opencv-python==4.8.0
# paho-mqtt==1.6.1

import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
import time

class EdgeAIRobot:
    def __init__(self, model_path='yolov8n.pt'):
        # 1. 모델 로드 (Jetson에서 TensorRT로 최적화)
        self.model = YOLO(model_path)
        self.model.fuse()  # 레이어 퓨전으로 속도 향상

        # 2. MQTT 클라이언트 초기화
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect("localhost", 1883)

        # 3. 카메라 초기화 (CSI 카메라)
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 성능 측정
        self.fps_counter = 0
        self.start_time = time.time()

    def run(self):
        print("🤖 Edge AI Robot started!")

        while True:
            # 1. 카메라에서 프레임 읽기
            ret, frame = self.camera.read()
            if not ret:
                break

            # 2. YOLO 추론 (Jetson에서 실행)
            results = self.model(frame, verbose=False)

            # 3. 결과 파싱
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    detections.append({
                        'class': self.model.names[cls],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })

            # 4. MQTT로 다른 시스템에 전송
            if detections:
                self.mqtt_client.publish(
                    "robot/detections",
                    json.dumps({
                        'timestamp': time.time(),
                        'detections': detections
                    })
                )

            # 5. 로봇 제어 로직 (예: 사람 추적)
            for det in detections:
                if det['class'] == 'person' and det['confidence'] > 0.7:
                    self.follow_person(det['bbox'])

            # 6. FPS 계산
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:
                elapsed = time.time() - self.start_time
                fps = 30 / elapsed
                print(f"⚡ FPS: {fps:.1f} | Detections: {len(detections)}")
                self.start_time = time.time()

    def follow_person(self, bbox):
        # 간단한 추적 로직: 프레임 중앙으로 사람 이동
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2

        frame_center = 320  # 640 / 2
        error = center_x - frame_center

        # MQTT로 모터 제어 명령 전송
        self.mqtt_client.publish(
            "robot/motor/turn",
            json.dumps({'angle': error * 0.1})  # 비례 제어
        )

if __name__ == "__main__":
    robot = EdgeAIRobot(model_path='yolov8n.pt')
    robot.run()

# 결과 (Jetson Orin Nano 기준):
# - FPS: 25-30 (실시간 처리 가능)
# - 레이턴시: 30-40ms (프레임당)
# - 전력 소비: 10W (배터리로 3시간 작동)
# - 정확도: mAP 45% (YOLOv8n 기준)`}
          </pre>

          <div className="mt-4 bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">성능 최적화 팁:</h4>
            <ul className="text-sm space-y-1 mb-0">
              <li>1️⃣ <strong>TensorRT 변환</strong>: <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">yolo export model=yolov8n.pt format=engine</code> → 2배 빨라짐</li>
              <li>2️⃣ <strong>해상도 조절</strong>: 640×480 → 416×416으로 낮추면 FPS 50% 향상</li>
              <li>3️⃣ <strong>배치 처리</strong>: 여러 프레임을 묶어서 추론 (GPU 활용률 증가)</li>
              <li>4️⃣ <strong>선택적 추론</strong>: 매 프레임이 아닌 3프레임마다 추론 (30 FPS → 10 FPS 추론으로도 충분)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Summary */}
      <section className="my-8">
        <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 border-l-4 border-cyan-500 p-6 rounded-lg">
          <h3 className="text-2xl font-bold mb-4">📌 핵심 요약</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-bold text-cyan-600 mb-2">Edge AI가 필요한 이유</h4>
              <ul className="text-sm space-y-1">
                <li>✅ 초저지연 (1-10ms) 실시간 반응</li>
                <li>✅ 오프라인 작동 (네트워크 불필요)</li>
                <li>✅ 프라이버시 보장 (로컬 처리)</li>
                <li>✅ 대역폭 비용 절감</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-blue-600 mb-2">필수 기술 스택</h4>
              <ul className="text-sm space-y-1">
                <li>🔧 <strong>하드웨어</strong>: NVIDIA Jetson, Google Coral</li>
                <li>⚡ <strong>최적화</strong>: 양자화, 프루닝, 지식 증류</li>
                <li>📡 <strong>통신</strong>: MQTT, ROS 2 DDS</li>
                <li>🏗️ <strong>아키텍처</strong>: Edge + Cloud 하이브리드</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Next Chapter Teaser */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border-l-4 border-purple-500 p-6 rounded-lg">
        <h3 className="text-2xl font-bold mb-2">다음 챕터 미리보기</h3>
        <p className="text-lg font-semibold mb-2">Chapter 6: 자율주행 모빌리티</p>
        <p className="mb-0">
          Edge AI로 무장한 로봇이 이제 도로 위를 달립니다.
          Waymo, Tesla FSD, Cruise의 자율주행 기술 스택을 완전 분해합니다.
          <strong className="text-purple-600"> SLAM, 경로 계획, 센서 퓨전의 모든 것!</strong>
        </p>
      </div>
    </div>
  );
}