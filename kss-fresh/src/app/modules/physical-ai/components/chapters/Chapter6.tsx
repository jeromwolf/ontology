'use client';

import React from 'react';
import { Car, MapPin, Radar, Route, Navigation, Zap } from 'lucide-react';

export default function Chapter6() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl p-8 mb-8 border border-purple-200 dark:border-purple-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg">
            <Car className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white m-0">
            자율주행 모빌리티
          </h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0 leading-relaxed">
          Physical AI의 최고 난이도 챌린지.
          <strong className="text-purple-600 dark:text-purple-400"> 시속 100km로 달리면서 1cm 오차로 제어</strong>하고,
          예측 불가능한 보행자를 피하며, 법규를 지켜야 합니다.
        </p>
      </div>

      {/* Introduction */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Navigation className="text-purple-600" />
          자율주행의 난이도 - 왜 이렇게 어려운가?
        </h2>

        <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold text-red-700 dark:text-red-400 mt-0">🚨 자율주행이 어려운 5가지 이유</h3>
          <ul className="space-y-2 mb-0">
            <li><strong>Long Tail 문제</strong>: 99.9% 상황은 쉽지만, 0.1% 엣지 케이스가 치명적 (갑자기 튀어나온 사슴, 역주행 차량)</li>
            <li><strong>실시간 제약</strong>: 시속 100km = 초당 27m 이동 → 10ms 이내 판단 필수</li>
            <li><strong>센서 노이즈</strong>: 비 오면 카메라 흐려짐, 눈 오면 LiDAR 오작동</li>
            <li><strong>법적 책임</strong>: 사고 발생 시 누구 책임? → 완벽한 안전성 요구</li>
            <li><strong>복잡한 상호작용</strong>: 다른 차량/보행자의 의도를 예측해야 함 (끼어들기, 급정거)</li>
          </ul>
        </div>

        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg text-center">
            <div className="text-4xl font-bold text-green-600 mb-2">99.9%</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">일반 도로 주행<br/>(이미 해결됨)</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg text-center">
            <div className="text-4xl font-bold text-yellow-600 mb-2">0.09%</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">복잡한 상황<br/>(대부분 가능)</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg text-center border-2 border-red-500">
            <div className="text-4xl font-bold text-red-600 mb-2">0.01%</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Long Tail 케이스<br/>(여전히 어려움)</div>
          </div>
        </div>

        <p className="text-lg">
          Waymo는 <strong className="text-blue-600">2,000만 마일</strong>을 주행했고,
          Tesla는 <strong className="text-green-600">70억 마일</strong> 데이터를 수집했습니다.
          그럼에도 완전 자율주행(Level 5)은 아직 요원합니다.
        </p>
      </section>

      {/* Self-Driving Levels */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Car className="text-blue-600" />
          자율주행 레벨 - SAE 기준
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 bg-gray-400 rounded-full flex items-center justify-center text-white font-bold">0</div>
              <h3 className="text-lg font-bold m-0">Level 0 - No Automation</h3>
            </div>
            <p className="text-sm mb-0">완전 수동 운전 (크루즈 컨트롤 없음)</p>
          </div>

          <div className="bg-blue-100 dark:bg-blue-900/30 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 bg-blue-400 rounded-full flex items-center justify-center text-white font-bold">1</div>
              <h3 className="text-lg font-bold m-0">Level 1 - Driver Assistance</h3>
            </div>
            <p className="text-sm mb-1">어댑티브 크루즈 컨트롤(ACC) 또는 차선 유지 중 하나만</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-0">예시: 대부분의 현대 차량</p>
          </div>

          <div className="bg-green-100 dark:bg-green-900/30 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white font-bold">2</div>
              <h3 className="text-lg font-bold m-0">Level 2 - Partial Automation</h3>
            </div>
            <p className="text-sm mb-1">ACC + 차선 유지 동시 작동 (운전자는 항상 주시)</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-0">예시: Tesla Autopilot, GM Super Cruise</p>
          </div>

          <div className="bg-yellow-100 dark:bg-yellow-900/30 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center text-white font-bold">3</div>
              <h3 className="text-lg font-bold m-0">Level 3 - Conditional Automation</h3>
            </div>
            <p className="text-sm mb-1">특정 조건(고속도로)에서 완전 자율 (운전자 개입 가능해야 함)</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-0">예시: Mercedes Drive Pilot (독일 한정)</p>
          </div>

          <div className="bg-orange-100 dark:bg-orange-900/30 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center text-white font-bold">4</div>
              <h3 className="text-lg font-bold m-0">Level 4 - High Automation</h3>
            </div>
            <p className="text-sm mb-1">제한된 지역(Geofenced)에서 완전 자율 (운전대 없어도 됨)</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-0">예시: Waymo One (피닉스, 샌프란시스코), Cruise (샌프란시스코 - 중단됨)</p>
          </div>

          <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg border-2 border-purple-500">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">5</div>
              <h3 className="text-lg font-bold m-0">Level 5 - Full Automation</h3>
            </div>
            <p className="text-sm mb-1">모든 도로, 모든 조건에서 완전 자율 (운전대/페달 불필요)</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-0">예시: 아직 존재하지 않음 (목표: 2030년대?)</p>
          </div>
        </div>
      </section>

      {/* Sensor Stack */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Radar className="text-cyan-600" />
          센서 스택 - 로봇의 눈과 귀
        </h2>

        <div className="grid md:grid-cols-3 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-blue-600 mb-3">카메라 (Vision)</h3>
            <div className="space-y-2 text-sm">
              <div><strong>장점</strong>: 색상, 텍스트, 교통 신호 인식</div>
              <div><strong>단점</strong>: 거리 측정 부정확, 날씨 영향</div>
              <div><strong>해상도</strong>: 1280×960 (Tesla) ~ 4K</div>
              <div><strong>가격</strong>: $50-200/대</div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-green-600 mb-3">LiDAR (레이저)</h3>
            <div className="space-y-2 text-sm">
              <div><strong>장점</strong>: 정밀한 3D 거리 측정 (±2cm)</div>
              <div><strong>단점</strong>: 비싸고, 비/눈에 약함</div>
              <div><strong>범위</strong>: 200m (Waymo) ~ 300m</div>
              <div><strong>가격</strong>: $1,000 (Livox) ~ $75,000 (Velodyne)</div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-purple-600 mb-3">Radar (레이더)</h3>
            <div className="space-y-2 text-sm">
              <div><strong>장점</strong>: 날씨 무관, 속도 측정 정확</div>
              <div><strong>단점</strong>: 해상도 낮음 (형태 구분 어려움)</div>
              <div><strong>범위</strong>: 250m (장거리 레이더)</div>
              <div><strong>가격</strong>: $100-500/대</div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-2xl font-bold mb-4">🔥 센서 융합 전쟁: Tesla vs Waymo</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-blue-500">
              <h4 className="font-bold text-blue-600 mb-2">Tesla - Vision Only (카메라만)</h4>
              <ul className="text-sm space-y-1 mb-0">
                <li>✅ 저렴함: 8개 카메라 = $400</li>
                <li>✅ 확장 가능: 수백만 대 차량 배포</li>
                <li>✅ 사람처럼 보기: 시각 정보만으로 인식</li>
                <li>❌ 거리 추정 부정확 (Depth Estimation 필요)</li>
                <li>❌ 악천후 취약 (안개, 눈부심)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-green-500">
              <h4 className="font-bold text-green-600 mb-2">Waymo - Sensor Fusion (All-in)</h4>
              <ul className="text-sm space-y-1 mb-0">
                <li>✅ 정확함: LiDAR로 cm 단위 거리 측정</li>
                <li>✅ 안전함: 센서 중복 (Redundancy)</li>
                <li>✅ 악천후 대응: 레이더 백업</li>
                <li>❌ 비쌈: 차량당 $200,000 (양산 불가)</li>
                <li>❌ 제한된 지역만 운영 (HD Map 필요)</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-pink-50 dark:bg-pink-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-bold mb-4">센서 융합 알고리즘 - 확장 칼만 필터 (EKF)</h3>
          <p className="text-sm mb-4">
            여러 센서의 노이즈가 있는 측정값을 융합하여 정확한 상태를 추정합니다.
            GPS (±5m 오차) + IMU (±0.1° 오차) + LiDAR (±2cm 오차) → <strong>±5cm 정확도</strong>
          </p>

          <pre className="bg-gray-900 text-gray-100 p-4 rounded text-xs overflow-x-auto">
{`import numpy as np

class ExtendedKalmanFilter:
    """자율주행차의 위치/속도/방향 추정"""
    def __init__(self, dim_x=6, dim_z=3):
        # 상태 벡터: [x, y, θ, vx, vy, ω] (위치, 각도, 속도, 각속도)
        self.dim_x = dim_x
        self.dim_z = dim_z  # 측정 벡터: [x_gps, y_gps, θ_imu]

        self.x = np.zeros(dim_x)  # 상태 추정값
        self.P = np.eye(dim_x)    # 오차 공분산
        self.Q = np.eye(dim_x) * 0.01  # 프로세스 노이즈 (모델 불확실성)
        self.R = np.diag([5.0, 5.0, 0.1])  # 측정 노이즈 (GPS ±5m, IMU ±0.1°)

    def predict(self, dt, u):
        """예측 단계 - 이전 상태로부터 현재 상태 예측"""
        # 상태 전이 함수 (비선형 운동 모델)
        x, y, theta, vx, vy, omega = self.x

        # 새로운 상태 예측
        x_new = x + vx * np.cos(theta) * dt - vy * np.sin(theta) * dt
        y_new = y + vx * np.sin(theta) * dt + vy * np.cos(theta) * dt
        theta_new = theta + omega * dt

        self.x = np.array([x_new, y_new, theta_new, vx, vy, omega])

        # 야코비안 행렬 (선형화)
        F = np.eye(6)
        F[0, 2] = -vx * np.sin(theta) * dt - vy * np.cos(theta) * dt
        F[1, 2] = vx * np.cos(theta) * dt - vy * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt

        # 공분산 업데이트
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """업데이트 단계 - 센서 측정값으로 보정"""
        # 측정 함수 (상태 → 측정값 매핑)
        h = np.array([self.x[0], self.x[1], self.x[2]])  # [x, y, θ]

        # 혁신 (Innovation): 측정값 - 예측값
        y = z - h

        # 측정 야코비안
        H = np.zeros((3, 6))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # θ

        # 칼만 게인 계산
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 상태 및 공분산 업데이트
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P

# 실시간 사용 예시
ekf = ExtendedKalmanFilter()

# 100Hz 루프 (10ms마다 실행)
for t in range(1000):
    dt = 0.01  # 10ms

    # 1. 예측 (IMU 데이터 사용)
    u = get_imu_data()  # 가속도, 각속도
    ekf.predict(dt, u)

    # 2. 업데이트 (GPS + LiDAR + 카메라)
    if t % 10 == 0:  # GPS는 10Hz
        z_gps = get_gps_position()
        ekf.update(z_gps)

    # 3. 현재 위치 추정값
    estimated_position = ekf.x[:3]  # [x, y, θ]
    print(f"Position: {estimated_position}")

# 결과: ±5cm 정확도로 위치 추정 (GPS 단독 대비 100배 향상)`}
          </pre>
        </div>
      </section>

      {/* SLAM */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <MapPin className="text-orange-600" />
          SLAM - 동시 위치 추정 및 지도 작성
        </h2>

        <p className="text-lg mb-4">
          <strong className="text-orange-600">SLAM (Simultaneous Localization and Mapping)</strong>은
          로봇이 미지의 환경에서 지도를 만들면서 동시에 자신의 위치를 찾는 기술입니다.
        </p>

        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🗺️ SLAM의 핵심 문제</h3>
          <ul className="space-y-2">
            <li><strong>닭이 먼저? 달걀이 먼저?</strong> - 위치를 알아야 지도를 만들고, 지도가 있어야 위치를 안다</li>
            <li><strong>Loop Closure</strong>: 같은 장소에 다시 왔을 때 인식하고 오차 보정</li>
            <li><strong>데이터 연관</strong>: 현재 관측과 과거 관측 매칭</li>
          </ul>

          <div className="mt-4 bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">SLAM 알고리즘 비교</h4>
            <div className="grid md:grid-cols-3 gap-3 text-sm">
              <div>
                <div className="font-bold text-blue-600">EKF-SLAM</div>
                <div>가장 오래된 방법</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">O(n²) 복잡도 - 느림</div>
              </div>
              <div>
                <div className="font-bold text-green-600">GraphSLAM</div>
                <div>최적화 기반 (g2o)</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">정확하지만 계산량 큼</div>
              </div>
              <div>
                <div className="font-bold text-purple-600">ORB-SLAM3</div>
                <div>카메라 기반 (최신)</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">실시간, 정확, 업계 표준</div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <h3 className="text-xl font-bold mb-4">ORB-SLAM3 구현 (간소화 버전)</h3>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded text-xs overflow-x-auto">
{`# ORB-SLAM3 핵심 파이프라인
import cv2
import numpy as np

class ORBSLAM:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)  # ORB 특징점 추출기
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # 특징점 매칭
        self.keyframes = []  # 키프레임 저장
        self.map_points = []  # 3D 맵 포인트

    def track_frame(self, frame):
        """1. 트래킹: 현재 프레임의 카메라 위치 추정"""
        # 특징점 추출
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)

        if len(self.keyframes) > 0:
            # 이전 키프레임과 매칭
            prev_kf = self.keyframes[-1]
            matches = self.bf.knnMatch(descriptors, prev_kf['descriptors'], k=2)

            # Lowe's ratio test (좋은 매칭만 선택)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # PnP (Perspective-n-Point)로 카메라 포즈 추정
            if len(good_matches) > 10:
                pose = self.estimate_pose(good_matches, prev_kf)
                return pose

        return None

    def create_keyframe(self, frame, pose):
        """2. 키프레임 생성: 중요한 프레임만 저장"""
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)

        keyframe = {
            'frame': frame,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose,
            'timestamp': time.time()
        }

        self.keyframes.append(keyframe)

    def triangulate_points(self, kf1, kf2):
        """3. 삼각측량: 2D 특징점 → 3D 맵 포인트 변환"""
        # 두 키프레임 간 매칭
        matches = self.bf.knnMatch(kf1['descriptors'], kf2['descriptors'], k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # 카메라 행렬
        K = np.array([[718.856, 0, 607.1928],
                      [0, 718.856, 185.2157],
                      [0, 0, 1]])

        # 삼각측량으로 3D 포인트 계산
        for match in good_matches:
            pt1 = kf1['keypoints'][match.queryIdx].pt
            pt2 = kf2['keypoints'][match.trainIdx].pt

            # DLT (Direct Linear Transform)
            point_3d = cv2.triangulatePoints(
                kf1['pose'] @ K,
                kf2['pose'] @ K,
                pt1, pt2
            )

            self.map_points.append(point_3d)

    def loop_closure(self):
        """4. 루프 클로저: 같은 장소 재방문 감지 및 오차 보정"""
        if len(self.keyframes) < 20:
            return

        current_kf = self.keyframes[-1]

        for i, old_kf in enumerate(self.keyframes[:-20]):
            # 현재 프레임과 과거 프레임 비교
            matches = self.bf.knnMatch(
                current_kf['descriptors'],
                old_kf['descriptors'],
                k=2
            )

            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            # 충분한 매칭 = 같은 장소!
            if len(good_matches) > 50:
                print(f"Loop detected at keyframe {i}!")
                self.optimize_graph()  # 전체 그래프 최적화
                break

# 실시간 SLAM 실행
slam = ORBSLAM()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # 1. 트래킹
    pose = slam.track_frame(frame)

    # 2. 키프레임 생성 (5프레임마다)
    if frame_count % 5 == 0:
        slam.create_keyframe(frame, pose)

    # 3. 삼각측량 (새 키프레임마다)
    if len(slam.keyframes) >= 2:
        slam.triangulate_points(slam.keyframes[-2], slam.keyframes[-1])

    # 4. 루프 클로저 (100프레임마다)
    if frame_count % 100 == 0:
        slam.loop_closure()

    frame_count += 1

# 결과: 실시간으로 3D 지도 생성 + 카메라 위치 추정`}
          </pre>
        </div>
      </section>

      {/* Path Planning */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Route className="text-teal-600" />
          경로 계획 (Path Planning)
        </h2>

        <p className="text-lg mb-4">
          지도를 알았고, 위치를 알았다면, 이제 <strong className="text-teal-600">어떻게 갈 것인가?</strong>를 결정해야 합니다.
        </p>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-blue-600 mb-3">Global Path Planning</h3>
            <p className="text-sm mb-3">
              출발지 → 목적지까지의 전체 경로 (A* 알고리즘)
            </p>
            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# A* (A-Star) 알고리즘
import heapq

def a_star(grid, start, goal):
    """그리드 맵에서 최단 경로 찾기"""
    def heuristic(a, b):
        # 맨해튼 거리
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            # 경로 재구성
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            neighbor = (current[0]+dx, current[1]+dy)

            if not is_valid(grid, neighbor):
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # 경로 없음`}
            </pre>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-green-600 mb-3">Local Path Planning</h3>
            <p className="text-sm mb-3">
              실시간 장애물 회피 (DWA - Dynamic Window Approach)
            </p>
            <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# DWA - 동적 창 접근법
def dwa(robot_state, goal, obstacles):
    """실시간 장애물 회피"""
    v, w = robot_state['velocity'], robot_state['angular_vel']

    # 가능한 속도 범위 (동적 창)
    v_min = max(0, v - a_max * dt)
    v_max = min(v_limit, v + a_max * dt)
    w_min = max(-w_limit, w - alpha_max * dt)
    w_max = min(w_limit, w + alpha_max * dt)

    best_score = -float('inf')
    best_v, best_w = 0, 0

    # 모든 속도 조합 평가
    for v_cand in np.arange(v_min, v_max, 0.1):
        for w_cand in np.arange(w_min, w_max, 0.1):
            # 이 속도로 3초간 주행 시뮬레이션
            trajectory = simulate(v_cand, w_cand, 3.0)

            # 충돌 체크
            if collides(trajectory, obstacles):
                continue

            # 평가 함수: 목표 근접 + 속도 + 장애물 거리
            score = (
                alpha * heading_score(trajectory, goal) +
                beta * velocity_score(v_cand) +
                gamma * clearance_score(trajectory, obstacles)
            )

            if score > best_score:
                best_score = score
                best_v, best_w = v_cand, w_cand

    return best_v, best_w

# 100Hz로 실시간 실행
while True:
    v, w = dwa(robot_state, goal, obstacles)
    robot.set_velocity(v, w)
    time.sleep(0.01)  # 10ms`}
            </pre>
          </div>
        </div>
      </section>

      {/* Real-world Systems */}
      <section className="my-8">
        <h2 className="flex items-center gap-3 text-3xl font-bold mb-6">
          <Zap className="text-yellow-600" />
          실전 사례 - Waymo vs Tesla
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-500">
            <h3 className="text-2xl font-bold text-blue-600 mb-4">Waymo Driver</h3>

            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">센서</div>
                <div className="text-sm">29개 카메라 + 5개 LiDAR + 6개 레이더</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">HD Map</div>
                <div className="text-sm">cm 단위 정밀 지도 (Geofenced)</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">성능</div>
                <div className="text-sm">2,000만 마일 주행, 사고율 0.41/백만 마일</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">장점</div>
                <div className="text-sm">안전하고 정확 (Level 4 달성)</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">단점</div>
                <div className="text-sm">비싸고 (차량당 $200k) 제한된 지역만</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-lg border-2 border-green-500">
            <h3 className="text-2xl font-bold text-green-600 mb-4">Tesla FSD (Supervised)</h3>

            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">센서</div>
                <div className="text-sm">8개 카메라 (Vision Only, No LiDAR)</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">Neural Network</div>
                <div className="text-sm">Transformer 기반 End-to-End 학습</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">Fleet Learning</div>
                <div className="text-sm">70억 마일 데이터 (전 세계 차량에서 수집)</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">장점</div>
                <div className="text-sm">저렴하고 ($400) 모든 도로에서 작동</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-bold mb-1">단점</div>
                <div className="text-sm">Level 2 (운전자 감독 필수)</div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-500 p-6 rounded-lg">
          <h3 className="text-xl font-bold text-purple-700 dark:text-purple-400 mt-0">🏆 누가 이길까?</h3>
          <p className="mb-2">
            <strong>Waymo</strong>는 안전과 정확성에서 승리 (Level 4 달성)
          </p>
          <p className="mb-2">
            <strong>Tesla</strong>는 확장성과 비용에서 승리 (500만 대 차량 배포)
          </p>
          <p className="mb-0">
            결론: <strong className="text-purple-600">두 가지 접근법 모두 유효</strong>합니다.
            Waymo는 로보택시, Tesla는 개인 차량에 최적화되어 있습니다.
          </p>
        </div>
      </section>

      {/* Summary */}
      <section className="my-8">
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border-l-4 border-purple-500 p-6 rounded-lg">
          <h3 className="text-2xl font-bold mb-4">📌 핵심 요약</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-bold text-purple-600 mb-2">자율주행 기술 스택</h4>
              <ul className="text-sm space-y-1">
                <li>🎥 <strong>센서</strong>: 카메라, LiDAR, 레이더 융합</li>
                <li>🗺️ <strong>SLAM</strong>: 실시간 지도 생성 + 위치 추정</li>
                <li>🛣️ <strong>경로 계획</strong>: A* (전역) + DWA (지역)</li>
                <li>🤖 <strong>제어</strong>: MPC, PID, Pure Pursuit</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-pink-600 mb-2">실전 인사이트</h4>
              <ul className="text-sm space-y-1">
                <li>✅ 센서 융합은 필수 (EKF로 cm 정확도)</li>
                <li>✅ Long Tail 문제가 가장 어려움</li>
                <li>✅ Level 5는 아직 10년 이상 걸림</li>
                <li>✅ Waymo vs Tesla: 두 접근법 모두 유효</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Next Chapter Teaser */}
      <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 border-l-4 border-orange-500 p-6 rounded-lg">
        <h3 className="text-2xl font-bold mb-2">다음 챕터 미리보기</h3>
        <p className="text-lg font-semibold mb-2">Chapter 7: 한국 제조업 혁신 전략</p>
        <p className="mb-0">
          Physical AI 기술을 한국 제조업에 어떻게 적용할 것인가?
          다크 팩토리, 디지털 트윈, 50조 달러 시장의 기회를 잡는 7가지 전략!
        </p>
      </div>
    </div>
  );
}