'use client';

import React from 'react';
import { Eye, Camera, Layers, Zap, Maximize, Grid3x3, Box } from 'lucide-react';

export default function Chapter3() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-2xl p-8 mb-8 border border-teal-200 dark:border-teal-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-teal-500 rounded-xl flex items-center justify-center">
            <Eye className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">컴퓨터 비전과 인식 시스템</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          로봇의 눈 - 현실 세계를 이해하는 AI의 시각 능력
        </p>
      </div>

      {/* Introduction */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Camera className="text-teal-600" />
          Physical AI에서 컴퓨터 비전의 중요성
        </h2>

        <div className="bg-gradient-to-r from-teal-50 to-blue-50 dark:from-teal-900/20 dark:to-blue-900/20 p-6 rounded-lg border-l-4 border-teal-500 mb-6">
          <h3 className="text-xl font-bold mb-4">👁️ 인간의 80%는 시각 정보</h3>
          <p className="mb-4">
            인간이 세상을 이해하는 정보의 <strong>80%가 시각</strong>에서 옵니다.
            Physical AI도 마찬가지입니다. 로봇이 물체를 잡고, 장애물을 피하고,
            사람과 협업하려면 <strong>정확하고 빠른 시각 인식</strong>이 필수입니다.
          </p>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">🎯</div>
              <h4 className="font-bold text-sm mb-2">객체 탐지</h4>
              <p className="text-xs">What: 무엇이 있는가?</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">📍</div>
              <h4 className="font-bold text-sm mb-2">위치 추정</h4>
              <p className="text-xs">Where: 어디에 있는가?</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">📐</div>
              <h4 className="font-bold text-sm mb-2">깊이 인식</h4>
              <p className="text-xs">How far: 얼마나 멀리?</p>
            </div>
          </div>
        </div>
      </section>

      {/* 1. Real-time Object Detection */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Box className="text-blue-600" />
          1. 실시간 객체 탐지 (Object Detection)
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🎯 YOLO (You Only Look Once) - 실시간 탐지의 정석</h3>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">왜 YOLO인가?</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-bold text-sm mb-2 text-blue-600">전통적 방식 (R-CNN)</h5>
                <ul className="text-sm space-y-1">
                  <li>❌ 속도: 초당 5-10 프레임</li>
                  <li>❌ 처리: 영역 제안 → 분류 (2단계)</li>
                  <li>❌ 실시간 불가능</li>
                </ul>
              </div>
              <div className="border-l-2 border-blue-300 pl-4">
                <h5 className="font-bold text-sm mb-2 text-green-600">YOLO</h5>
                <ul className="text-sm space-y-1">
                  <li>✅ 속도: 초당 30-60 프레임</li>
                  <li>✅ 처리: 단일 신경망 (1단계)</li>
                  <li>✅ 실시간 가능</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">
            <pre className="text-sm">
{`# YOLOv8 실시간 객체 탐지 (Ultralytics)
from ultralytics import YOLO
import cv2

class RobotVision:
    def __init__(self):
        # YOLOv8 모델 로드
        self.model = YOLO('yolov8n.pt')  # nano 버전 (빠름)
        self.confidence_threshold = 0.5

    def detect_objects(self, frame):
        # 단일 프레임 추론
        results = self.model(frame, conf=self.confidence_threshold)

        objects = []
        for result in results:
            boxes = result.boxes  # 바운딩 박스
            for box in boxes:
                # 좌표 및 정보 추출
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1+x2)/2, (y1+y2)/2)
                })

        return objects

    def real_time_detection(self):
        cap = cv2.VideoCapture(0)  # 웹캠

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 객체 탐지 (30 FPS 이상)
            objects = self.detect_objects(frame)

            # 시각화
            annotated_frame = self.model(frame)[0].plot()
            cv2.imshow('Robot Vision', annotated_frame)

            # 로봇 제어 로직
            self.robot_decision(objects)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def robot_decision(self, objects):
        # 탐지된 객체 기반 로봇 행동 결정
        for obj in objects:
            if obj['class'] == 'person':
                print(f"사람 감지! 안전 거리 유지")
            elif obj['class'] == 'bottle':
                print(f"병 발견 at {obj['center']}")
                # 로봇 팔 이동 명령`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <h4 className="font-bold mb-2">🚀 YOLOv8 (2023) - 최신 버전의 개선점</h4>
            <ul className="text-sm space-y-2">
              <li>✅ <strong>Anchor-Free 설계</strong>: 사전 정의된 앵커 박스 불필요, 더 유연한 탐지</li>
              <li>✅ <strong>향상된 정확도</strong>: YOLOv5 대비 mAP 10% 향상</li>
              <li>✅ <strong>더 작은 모델</strong>: Nano 모델 6MB (모바일/엣지 최적화)</li>
              <li>✅ <strong>다중 작업</strong>: 탐지 + 세그멘테이션 + 자세 추정 통합</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. Depth Estimation */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Layers className="text-purple-600" />
          2. 깊이 추정 (Depth Estimation)
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">📐 3D 공간 이해 - 단안 vs 스테레오</h3>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg">
              <h4 className="font-bold mb-3 text-blue-600">단안 깊이 추정 (Monocular)</h4>
              <p className="text-sm mb-3">
                <strong>단일 카메라</strong>로 깊이를 추정합니다.
                AI가 이미지 속 단서 (크기, 가림, 원근)를 학습해 거리를 예측합니다.
              </p>
              <ul className="text-sm space-y-2">
                <li>✅ <strong>장점</strong>: 저렴, 컴팩트, 1개 카메라만 필요</li>
                <li>❌ <strong>단점</strong>: 정확도 낮음 (±10-20% 오차)</li>
                <li>🎯 <strong>사용처</strong>: Tesla FSD, 스마트폰 카메라</li>
              </ul>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border-2 border-green-500">
              <h4 className="font-bold mb-3 text-green-600">스테레오 비전 (Stereo Vision)</h4>
              <p className="text-sm mb-3">
                <strong>2개 카메라</strong>로 양안 시차를 계산해 정확한 3D 맵 생성.
                인간 눈과 동일한 원리입니다.
              </p>
              <ul className="text-sm space-y-2">
                <li>✅ <strong>장점</strong>: 높은 정확도 (±1-2% 오차)</li>
                <li>❌ <strong>단점</strong>: 2개 카메라, 캘리브레이션 필요</li>
                <li>🎯 <strong>사용처</strong>: 자율주행, 로봇 팔, 드론</li>
              </ul>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">
            <pre className="text-sm">
{`# MiDaS - 단안 깊이 추정 (Intel)
import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self):
        # MiDaS 모델 로드
        self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.model.eval()

        # Transform
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = midas_transforms.small_transform

    def estimate_depth(self, frame):
        # 전처리
        input_batch = self.transform(frame).unsqueeze(0)

        # 추론 (GPU 사용)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

        # Depth map (0-255 스케일)
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)

        return depth_map

    def get_distance_to_object(self, depth_map, bbox):
        # 바운딩 박스 내 평균 깊이 계산
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        roi = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(roi)

        # 실제 거리로 변환 (캘리브레이션 필요)
        # 여기서는 상대적 깊이만 제공
        return avg_depth

# 사용 예시
depth_estimator = DepthEstimator()
yolo_detector = RobotVision()

frame = cv2.imread('scene.jpg')
objects = yolo_detector.detect_objects(frame)
depth_map = depth_estimator.estimate_depth(frame)

# 각 객체까지의 거리 계산
for obj in objects:
    distance = depth_estimator.get_distance_to_object(depth_map, obj['bbox'])
    print(f"{obj['class']}: 상대 거리 {distance:.2f}")`}
            </pre>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-l-4 border-purple-500">
            <h4 className="font-bold mb-2">🌟 최신 기술: LiDAR + Vision 융합</h4>
            <p className="text-sm mb-3">
              고급 Physical AI는 <strong>카메라 + LiDAR</strong>를 결합해 최고 정확도를 달성합니다.
            </p>
            <div className="grid md:grid-cols-2 gap-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <strong>카메라</strong>: 색상, 질감, 객체 분류
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <strong>LiDAR</strong>: 정밀한 3D 거리 (±1cm)
              </div>
            </div>
            <p className="text-sm mt-3">
              <strong>예시</strong>: Waymo 자율주행차는 5개 LiDAR + 29개 카메라 사용
            </p>
          </div>
        </div>
      </section>

      {/* 3. Semantic Segmentation */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Grid3x3 className="text-orange-600" />
          3. 의미론적 분할 (Semantic Segmentation)
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🎨 픽셀 단위 이해 - 이미지의 모든 영역 분류</h3>

          <div className="bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">객체 탐지 vs 세그멘테이션</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-bold text-sm mb-2">객체 탐지 (YOLO)</h5>
                <p className="text-sm mb-2">바운딩 박스로 객체 위치만 표시</p>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-xs">
                  "이 사각형 안에 차가 있다"
                </div>
              </div>
              <div className="border-l-2 border-orange-300 pl-4">
                <h5 className="font-bold text-sm mb-2">세그멘테이션 (Mask R-CNN)</h5>
                <p className="text-sm mb-2">객체의 정확한 모양을 픽셀 단위로 분리</p>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-xs">
                  "이 픽셀들이 차의 실제 윤곽이다"
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">
            <pre className="text-sm">
{`# Segment Anything Model (SAM) - Meta AI 2023
from segment_anything import sam_model_registry, SamPredictor
import cv2

class SemanticSegmenter:
    def __init__(self):
        # SAM 모델 로드
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        self.predictor = SamPredictor(sam)

    def segment_object(self, image, point_coords):
        """
        point_coords: 사용자가 클릭한 점 좌표
        예: [(100, 200)] - "이 지점의 물체를 분리해줘"
        """
        self.predictor.set_image(image)

        # 클릭한 점을 기반으로 마스크 생성
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=np.array([1] * len(point_coords)),  # 1 = foreground
            multimask_output=True
        )

        # 가장 높은 점수의 마스크 선택
        best_mask = masks[np.argmax(scores)]
        return best_mask

    def apply_mask_to_image(self, image, mask):
        # 마스크 영역만 컬러로, 나머지는 회색
        result = image.copy()
        result[~mask] = result[~mask] * 0.3  # 배경 어둡게

        # 마스크 경계선 그리기
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result

# 로봇이 물체를 잡기 위한 정확한 위치 파악
segmenter = SemanticSegmenter()
image = cv2.imread('workspace.jpg')

# 로봇이 잡을 물체를 클릭 (또는 자동 탐지)
target_point = [(320, 240)]  # 이미지 중심
mask = segmenter.segment_object(image, target_point)

# 물체의 무게중심 계산 (로봇 그리퍼 위치)
moments = cv2.moments(mask.astype(np.uint8))
center_x = int(moments['m10'] / moments['m00'])
center_y = int(moments['m01'] / moments['m00'])

print(f"그리퍼 목표 위치: ({center_x}, {center_y})")`}
            </pre>
          </div>

          <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg border-l-4 border-cyan-500">
            <h4 className="font-bold mb-2">🤖 로봇 응용 사례</h4>
            <ul className="text-sm space-y-2">
              <li>
                <strong>• 로봇 팔 제어</strong>
                <p className="mt-1">물체의 정확한 윤곽을 파악해 최적의 그리퍼 위치 계산</p>
              </li>
              <li>
                <strong>• 자율주행</strong>
                <p className="mt-1">도로, 보도, 차선, 신호등을 픽셀 단위로 분류</p>
              </li>
              <li>
                <strong>• 의료 로봇</strong>
                <p className="mt-1">수술 중 정밀한 조직 구분 (종양 vs 정상 조직)</p>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 4. Pose Estimation */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Maximize className="text-green-600" />
          4. 자세 추정 (Pose Estimation)
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🤸 인간 및 로봇의 관절 위치 추적</h3>

          <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">왜 자세 추정이 중요한가?</h4>
            <p className="text-sm mb-3">
              휴머노이드 로봇이 <strong>사람과 협업</strong>하려면 사람의 행동을 이해해야 합니다.
              자세 추정은 17개 관절 (손목, 팔꿈치, 어깨, 무릎 등)의 3D 위치를 실시간으로 추적합니다.
            </p>
            <div className="grid md:grid-cols-3 gap-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <strong>제스처 인식</strong><br/>
                <span className="text-xs">손을 들면 로봇에게 신호</span>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <strong>안전 감지</strong><br/>
                <span className="text-xs">사람이 넘어지면 즉시 도움</span>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <strong>동작 모방</strong><br/>
                <span className="text-xs">사람의 행동을 보고 학습</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">
            <pre className="text-sm">
{`# MediaPipe Pose - Google의 실시간 자세 추정
import mediapipe as mp
import cv2

class HumanPoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def track_pose(self, frame):
        # RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # 33개 관절 좌표 추출
            landmarks = results.pose_landmarks.landmark

            # 주요 관절 위치
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

            # 제스처 감지 예시: 양손을 머리 위로
            if (left_wrist.y < nose.y and right_wrist.y < nose.y):
                return "HANDS_UP"  # 로봇에게 정지 신호

            # 스켈레톤 시각화
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame, results.pose_landmarks

    def calculate_joint_angle(self, p1, p2, p3):
        """
        3개 관절로 각도 계산 (예: 팔꿈치 각도)
        p1: 어깨, p2: 팔꿈치, p3: 손목
        """
        import numpy as np

        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])

        angle = np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )
        return np.degrees(angle)

# 협업 로봇 응용
tracker = HumanPoseTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gesture, landmarks = tracker.track_pose(frame)

    if gesture == "HANDS_UP":
        print("🚨 긴급 정지 신호 감지! 로봇 동작 중지")
        # robot.emergency_stop()

    cv2.imshow('Human-Robot Collaboration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break`}
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
            <h4 className="font-bold mb-2">🏭 산업 응용 사례</h4>
            <div className="space-y-3 text-sm">
              <div>
                <strong>• BMW 공장 (Figure AI)</strong>
                <p className="mt-1">작업자의 자세를 분석해 로봇이 적절한 위치에서 부품 전달</p>
              </div>
              <div>
                <strong>• Amazon 물류센터</strong>
                <p className="mt-1">작업자 피로도 감지 (구부린 자세 지속 시 경고)</p>
              </div>
              <div>
                <strong>• 재활 로봇</strong>
                <p className="mt-1">환자의 관절 각도를 실시간 추적하며 물리치료 보조</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Sensor Fusion */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Zap className="text-indigo-600" />
          5. 센서 융합 (Sensor Fusion)
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">🔗 여러 센서를 통합해 완벽한 인지</h3>

          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">왜 하나의 센서로는 부족한가?</h4>
            <p className="text-sm mb-4">
              각 센서는 <strong>장점과 한계</strong>가 있습니다. 여러 센서를 결합하면 약점을 보완하고
              <strong>신뢰성과 정확도</strong>를 극대화할 수 있습니다.
            </p>

            <div className="grid md:grid-cols-3 gap-3 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <strong>카메라</strong><br/>
                <span className="text-xs text-green-600">✅ 색상, 질감 풍부</span><br/>
                <span className="text-xs text-red-600">❌ 어둠에 약함</span>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <strong>LiDAR</strong><br/>
                <span className="text-xs text-green-600">✅ 정밀 거리 측정</span><br/>
                <span className="text-xs text-red-600">❌ 비, 눈에 취약</span>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <strong>레이더</strong><br/>
                <span className="text-xs text-green-600">✅ 날씨 무관</span><br/>
                <span className="text-xs text-red-600">❌ 해상도 낮음</span>
              </div>
            </div>

            <div className="mt-4 p-3 bg-green-100 dark:bg-green-900/30 rounded">
              <strong className="text-green-700 dark:text-green-300">센서 융합 결과</strong>
              <p className="text-sm mt-2">
                카메라 (객체 분류) + LiDAR (거리) + 레이더 (속도) =
                <strong>완벽한 3D 인지 시스템</strong>
              </p>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Kalman Filter 기반 센서 융합
import numpy as np

class SensorFusion:
    def __init__(self):
        # 칼만 필터 상태 (위치, 속도)
        self.state = np.array([0.0, 0.0])  # [position, velocity]
        self.P = np.eye(2)  # 공분산 행렬

        # 프로세스 노이즈
        self.Q = np.array([[0.01, 0], [0, 0.01]])

    def predict(self, dt):
        # 상태 예측 (등속도 모델)
        F = np.array([[1, dt], [0, 1]])  # 상태 전이 행렬
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def update_camera(self, measured_position):
        # 카메라 측정값으로 업데이트
        H = np.array([[1, 0]])  # 측정 행렬 (위치만)
        R = np.array([[0.5]])   # 카메라 노이즈 (정확도 중간)

        y = measured_position - (H @ self.state)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    def update_lidar(self, measured_position):
        # LiDAR 측정값으로 업데이트 (더 정확함)
        H = np.array([[1, 0]])
        R = np.array([[0.1]])  # LiDAR 노이즈 (정확도 높음)

        y = measured_position - (H @ self.state)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    def get_fused_position(self):
        return self.state[0]

# 실시간 센서 융합 예시
fusion = SensorFusion()

while robot.is_running():
    dt = 0.1  # 100ms

    # 예측 단계
    fusion.predict(dt)

    # 카메라 데이터 수신 시
    if camera.has_data():
        camera_pos = camera.get_object_position()
        fusion.update_camera(camera_pos)

    # LiDAR 데이터 수신 시
    if lidar.has_data():
        lidar_pos = lidar.get_distance()
        fusion.update_lidar(lidar_pos)

    # 융합된 위치 사용
    fused_position = fusion.get_fused_position()
    robot.move_to(fused_position)`}
            </pre>
          </div>
        </div>
      </section>

      {/* Summary */}
      <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 border-l-4 border-teal-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-3">📌 핵심 요약</h3>
        <ul className="space-y-2 text-sm">
          <li>✅ <strong>객체 탐지 (YOLO)</strong>: 실시간 30-60 FPS, 바운딩 박스</li>
          <li>✅ <strong>깊이 추정</strong>: 단안 (저렴) vs 스테레오 (정확) vs LiDAR (최고)</li>
          <li>✅ <strong>세그멘테이션</strong>: 픽셀 단위 객체 분리, 정밀 제어</li>
          <li>✅ <strong>자세 추정</strong>: 33개 관절 추적, 인간-로봇 협업</li>
          <li>✅ <strong>센서 융합</strong>: 카메라 + LiDAR + 레이더 = 완벽한 인지</li>
        </ul>
      </div>

      {/* Next Chapter */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg">
        <h3 className="text-xl font-bold mb-2">다음 단계: 강화학습과 로봇 제어</h3>
        <p className="text-gray-700 dark:text-gray-300">
          다음 챕터에서는 로봇이 <strong>시행착오를 통해 스스로 학습</strong>하는
          강화학습 알고리즘과 정밀한 모터 제어 기술을 배웁니다.
        </p>
      </div>
    </div>
  )
}