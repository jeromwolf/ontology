'use client'

import React from 'react'
import ChapterNavigation from '../ChapterNavigation'

export default function Chapter7() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Chapter 7: ROS2 프로그래밍</h1>
        <p className="text-xl text-white/90">
          실전 로봇 제어를 위한 ROS2 기초와 응용
        </p>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        {/* 1. ROS2 Overview */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1. ROS2란?
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>ROS2 (Robot Operating System 2)</strong>는 로봇 소프트웨어 개발을 위한
              오픈소스 미들웨어 프레임워크입니다. 센서 데이터 처리, 모션 제어, 시뮬레이션 등
              로봇 개발에 필요한 모든 도구와 라이브러리를 제공합니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            ROS1 vs ROS2
          </h3>

          <table className="min-w-full border border-blue-300 dark:border-blue-700 mt-3 mb-6">
            <thead className="bg-blue-100 dark:bg-blue-900/50">
              <tr>
                <th className="px-4 py-2 border-b text-left">특징</th>
                <th className="px-4 py-2 border-b text-left">ROS1</th>
                <th className="px-4 py-2 border-b text-left">ROS2</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">통신 방식</td>
                <td className="px-4 py-2">TCPROS (Master 중심)</td>
                <td className="px-4 py-2">DDS (분산형, P2P)</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">실시간 성능</td>
                <td className="px-4 py-2">제한적</td>
                <td className="px-4 py-2">Real-Time 지원</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">보안</td>
                <td className="px-4 py-2">없음</td>
                <td className="px-4 py-2">DDS Security 지원</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">플랫폼</td>
                <td className="px-4 py-2">Linux 중심</td>
                <td className="px-4 py-2">Linux, Windows, macOS</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">지원 언어</td>
                <td className="px-4 py-2">Python, C++</td>
                <td className="px-4 py-2">Python, C++, Rust 등</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-semibold">적용 분야</td>
                <td className="px-4 py-2">연구, 프로토타입</td>
                <td className="px-4 py-2">산업용 로봇, 상용 제품</td>
              </tr>
            </tbody>
          </table>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
            <h3 className="text-xl font-semibold text-green-700 dark:text-green-300 mb-3">
              주요 배포판 (Distributions)
            </h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>Humble Hawksbill (2022)</strong>: LTS (Long-Term Support), Ubuntu 22.04</li>
              <li><strong>Iron Irwini (2023)</strong>: 9개월 지원</li>
              <li><strong>Jazzy Jalisco (2024)</strong>: LTS, Ubuntu 24.04</li>
              <li><strong>Rolling Ridley</strong>: 최신 개발 버전 (지속 업데이트)</li>
            </ul>
          </div>
        </section>

        {/* 2. ROS2 Core Concepts */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            2. ROS2 핵심 개념
          </h2>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            2.1 노드 (Node)
          </h3>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              ROS2의 기본 실행 단위로, 특정 기능을 담당하는 독립적인 프로세스입니다.
            </p>
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">예시:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li><code>/camera_node</code>: 카메라 데이터 발행</li>
              <li><code>/controller_node</code>: 모터 제어 명령 수신</li>
              <li><code>/planner_node</code>: 경로 계획 수행</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            2.2 토픽 (Topic)
          </h3>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-6 border-l-4 border-purple-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              노드 간 데이터를 교환하는 <strong>비동기 단방향 통신</strong> 채널입니다.
              Publish-Subscribe 패턴을 사용합니다.
            </p>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">
              특징
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li><strong>1:N 통신</strong>: 하나의 Publisher, 여러 Subscriber</li>
              <li><strong>비동기</strong>: 응답을 기다리지 않음</li>
              <li><strong>연속적 데이터</strong>: 센서 데이터, 상태 정보 등</li>
            </ul>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">예시:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li><code>/camera/image_raw</code>: 카메라 이미지</li>
              <li><code>/joint_states</code>: 관절 위치/속도</li>
              <li><code>/cmd_vel</code>: 속도 명령</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            2.3 서비스 (Service)
          </h3>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 mb-6 border-l-4 border-green-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              <strong>동기 양방향 통신</strong>으로, 요청(Request)과 응답(Response) 구조를 가집니다.
            </p>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              특징
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li><strong>1:1 통신</strong>: Client와 Server</li>
              <li><strong>동기</strong>: 응답을 받을 때까지 대기</li>
              <li><strong>일회성 작업</strong>: 설정 변경, 계산 요청 등</li>
            </ul>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">예시:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li><code>/spawn</code>: 로봇 생성</li>
              <li><code>/reset_simulation</code>: 시뮬레이션 초기화</li>
              <li><code>/get_plan</code>: 경로 계획 요청</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            2.4 액션 (Action)
          </h3>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              <strong>장시간 실행되는 작업</strong>을 위한 비동기 양방향 통신입니다.
              Goal, Feedback, Result 세 가지 메시지를 사용합니다.
            </p>

            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-3">
              특징
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li><strong>Goal</strong>: 작업 목표 전송</li>
              <li><strong>Feedback</strong>: 진행 상황 실시간 수신</li>
              <li><strong>Result</strong>: 최종 결과 수신</li>
              <li><strong>취소 가능</strong>: 실행 중 작업 중단 가능</li>
            </ul>

            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">예시:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li><code>/move_to_pose</code>: 목표 위치로 이동</li>
              <li><code>/follow_trajectory</code>: 궤적 추종</li>
              <li><code>/grasp_object</code>: 물체 파지</li>
            </ul>
          </div>
        </section>

        {/* 3. ROS2 Basic Commands */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3. ROS2 기본 명령어
          </h2>

          <div className="bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-green-400 text-sm overflow-x-auto">
              <code>{`# 노드 목록 확인
ros2 node list

# 노드 정보 확인
ros2 node info /node_name

# 토픽 목록 확인
ros2 topic list

# 토픽 데이터 실시간 출력
ros2 topic echo /topic_name

# 토픽에 데이터 발행
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}, angular: {z: 0.2}}"

# 토픽 주파수 확인
ros2 topic hz /topic_name

# 서비스 목록 확인
ros2 service list

# 서비스 호출
ros2 service call /service_name service_type "{param: value}"

# 액션 목록 확인
ros2 action list

# 액션 전송
ros2 action send_goal /action_name action_type "{goal_data}"

# 파라미터 목록 확인
ros2 param list

# 파라미터 값 확인
ros2 param get /node_name param_name

# 파라미터 설정
ros2 param set /node_name param_name value

# TF 트리 확인
ros2 run tf2_tools view_frames

# 패키지 생성
ros2 pkg create --build-type ament_python my_package

# 빌드
colcon build

# 워크스페이스 소스
source install/setup.bash`}</code>
            </pre>
          </div>
        </section>

        {/* 4. Simple Publisher/Subscriber */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            4. 간단한 Publisher/Subscriber 예제
          </h2>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            4.1 Publisher (Python)
          </h3>

          <div className="bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-green-400 text-sm overflow-x-auto">
              <code>{`import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')

        # Publisher 생성
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # 타이머 생성 (0.5초마다 콜백 실행)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'

        # 메시지 발행
        self.publisher_.publish(msg)

        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass

    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()`}</code>
            </pre>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            4.2 Subscriber (Python)
          </h3>

          <div className="bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-green-400 text-sm overflow-x-auto">
              <code>{`import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')

        # Subscriber 생성
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # 메시지 수신 시 호출되는 콜백
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass

    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()`}</code>
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              실행 방법
            </h3>
            <pre className="text-sm bg-white dark:bg-gray-800 rounded p-3 overflow-x-auto">
              <code>{`# 터미널 1: Publisher 실행
ros2 run my_package publisher

# 터미널 2: Subscriber 실행
ros2 run my_package subscriber

# 터미널 3: 토픽 확인
ros2 topic echo /topic`}</code>
            </pre>
          </div>
        </section>

        {/* 5. MoveIt2 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            5. MoveIt2 - 모션 플래닝 프레임워크
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              MoveIt2란?
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              매니퓰레이터를 위한 종합 모션 플래닝 프레임워크입니다. 역기구학, 경로 계획,
              충돌 감지, 그리핑 등의 기능을 통합 제공합니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            주요 기능
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
              <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
                모션 플래닝
              </h4>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>RRT, RRT*, PRM</li>
                <li>OMPL 라이브러리 통합</li>
                <li>Cartesian path planning</li>
              </ul>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">
                충돌 감지
              </h4>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>자체 충돌 검사</li>
                <li>환경 장애물 회피</li>
                <li>허용 충돌 매트릭스</li>
              </ul>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border border-orange-200 dark:border-orange-800">
              <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">
                역기구학
              </h4>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>KDL 솔버</li>
                <li>TRAC-IK (빠른 수렴)</li>
                <li>Pick IK (최신)</li>
              </ul>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border border-red-200 dark:border-red-800">
              <h4 className="font-semibold text-red-700 dark:text-red-300 mb-2">
                시각화
              </h4>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>RViz2 통합</li>
                <li>대화형 마커</li>
                <li>Planning Scene 표시</li>
              </ul>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            MoveIt2 Python 예제
          </h3>

          <div className="bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-green-400 text-sm overflow-x-auto">
              <code>{`import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from geometry_msgs.msg import PoseStamped

class SimpleMoveitDemo(Node):
    def __init__(self):
        super().__init__('simple_moveit_demo')

        # MoveIt2 인터페이스 초기화
        self.moveit = MoveItPy(node=self)

        # Planning group 선택 (매니퓰레이터)
        self.arm = self.moveit.get_planning_component("arm")

        self.get_logger().info("MoveIt2 initialized")

    def move_to_pose(self, x, y, z, roll, pitch, yaw):
        """목표 위치로 이동"""

        # 목표 Pose 생성
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position.x = x
        pose_goal.pose.position.y = y
        pose_goal.pose.position.z = z

        # Quaternion 변환 (roll, pitch, yaw → quaternion)
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(roll, pitch, yaw)
        pose_goal.pose.orientation.x = q[0]
        pose_goal.pose.orientation.y = q[1]
        pose_goal.pose.orientation.z = q[2]
        pose_goal.pose.orientation.w = q[3]

        # 목표 설정
        self.arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="tool0")

        # 경로 계획
        plan_result = self.arm.plan()

        if plan_result:
            self.get_logger().info("Planning succeeded, executing...")

            # 실행
            robot_trajectory = plan_result.trajectory
            self.arm.execute(robot_trajectory)

            self.get_logger().info("Execution complete")
        else:
            self.get_logger().error("Planning failed")

    def move_to_joint_values(self, joint_values):
        """관절 각도로 이동"""

        # 목표 관절 상태 설정
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name="ready")  # 또는 joint_values 직접 설정

        # 계획 및 실행
        plan_result = self.arm.plan()

        if plan_result:
            self.arm.execute(plan_result.trajectory)

def main(args=None):
    rclpy.init(args=args)

    demo = SimpleMoveitDemo()

    # 예제: 특정 위치로 이동
    demo.move_to_pose(x=0.3, y=0.2, z=0.5, roll=0.0, pitch=1.57, yaw=0.0)

    rclpy.spin(demo)

    demo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()`}</code>
            </pre>
          </div>
        </section>

        {/* 6. Gazebo Simulation */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            6. Gazebo 시뮬레이션
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              Gazebo란?
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              3D 로봇 시뮬레이터로, 물리 엔진을 사용하여 실제 로봇의 동작을 정확히 재현합니다.
              센서, 액추에이터, 환경을 모두 시뮬레이션할 수 있습니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            주요 특징
          </h3>

          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6 mb-6">
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>물리 엔진</strong>: ODE, Bullet, Simbody, DART 지원</li>
              <li><strong>센서 시뮬레이션</strong>: 카메라, LiDAR, IMU, GPS, Force/Torque</li>
              <li><strong>환경 모델링</strong>: URDF, SDF 형식 지원</li>
              <li><strong>플러그인 시스템</strong>: 커스텀 센서/제어 추가 가능</li>
              <li><strong>ROS2 통합</strong>: gz_ros2_control 패키지</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            시뮬레이션 실행 예제
          </h3>

          <div className="bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-green-400 text-sm overflow-x-auto">
              <code>{`# Gazebo Fortress/Garden/Harmonic 설치 (ROS2 Humble 기준)
sudo apt install ros-humble-ros-gz

# 시뮬레이션 실행
ros2 launch my_robot_description gazebo.launch.py

# 또는 직접 실행
gz sim -r empty.sdf

# ROS2-Gazebo 브릿지 실행
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock

# 로봇 상태 발행
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(xacro my_robot.urdf.xacro)"`}</code>
            </pre>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            URDF 예제 (간단한 2-링크 로봇)
          </h3>

          <div className="bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-green-400 text-sm overflow-x-auto">
              <code>{`<?xml version="1.0"?>
<robot name="two_link_arm">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Link 1 -->
  <link name="link1">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.02"/>
      </geometry>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.02"/>
      </geometry>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joint 1 (Revolute) -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Link 2 -->
  <link name="link2">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.02"/>
      </geometry>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.02"/>
      </geometry>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joint 2 (Revolute) -->
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>
</robot>`}</code>
            </pre>
          </div>
        </section>

        {/* 7. ROS2 Control */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            7. ROS2 Control
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              ROS2 Control이란?
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              로봇 제어를 위한 표준화된 프레임워크로, 하드웨어 인터페이스와 컨트롤러를 분리하여
              다양한 로봇에 동일한 제어 알고리즘을 적용할 수 있습니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            핵심 구성 요소
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
              <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
                Hardware Interface
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                실제 하드웨어 또는 시뮬레이터와의 인터페이스.
                State Interface (읽기)와 Command Interface (쓰기)로 구성.
              </p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">
                Controller Manager
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                컨트롤러 로딩, 시작, 정지를 관리하는 노드.
                여러 컨트롤러를 동시에 실행 가능.
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border border-orange-200 dark:border-orange-800">
              <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">
                Controllers
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                실제 제어 알고리즘 구현. Position, Velocity, Effort,
                Joint Trajectory Controller 등.
              </p>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border border-red-200 dark:border-red-800">
              <h4 className="font-semibold text-red-700 dark:text-red-300 mb-2">
                Resource Manager
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                하드웨어 리소스 관리. 여러 컨트롤러가
                동일한 관절에 접근하지 못하도록 제어.
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            주요 컨트롤러 종류
          </h3>

          <table className="min-w-full border border-blue-300 dark:border-blue-700 mt-3">
            <thead className="bg-blue-100 dark:bg-blue-900/50">
              <tr>
                <th className="px-4 py-2 border-b text-left">컨트롤러</th>
                <th className="px-4 py-2 border-b text-left">설명</th>
                <th className="px-4 py-2 border-b text-left">응용</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">joint_trajectory_controller</td>
                <td className="px-4 py-2">관절 궤적 추종</td>
                <td className="px-4 py-2">매니퓰레이터 제어</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">diff_drive_controller</td>
                <td className="px-4 py-2">차동 구동</td>
                <td className="px-4 py-2">모바일 로봇</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">gripper_action_controller</td>
                <td className="px-4 py-2">그리퍼 제어</td>
                <td className="px-4 py-2">파지 작업</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-semibold">joint_state_broadcaster</td>
                <td className="px-4 py-2">관절 상태 발행</td>
                <td className="px-4 py-2">모든 로봇</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* 8. Summary */}
        <section className="mb-12 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-8 border border-orange-200 dark:border-orange-800">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            핵심 요약
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                1. ROS2 핵심 개념
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>노드</strong>: 독립적 실행 단위</li>
                <li><strong>토픽</strong>: 비동기 단방향 통신 (센서 데이터)</li>
                <li><strong>서비스</strong>: 동기 양방향 통신 (일회성 작업)</li>
                <li><strong>액션</strong>: 비동기 장기 작업 (Goal, Feedback, Result)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                2. MoveIt2
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>종합 모션 플래닝 프레임워크</li>
                <li>OMPL 기반 경로 계획, 충돌 감지</li>
                <li>다양한 역기구학 솔버 (KDL, TRAC-IK, Pick IK)</li>
                <li>RViz2 통합 시각화</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                3. Gazebo 시뮬레이션
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>물리 엔진 기반 3D 시뮬레이터</li>
                <li>센서 시뮬레이션 (카메라, LiDAR, IMU 등)</li>
                <li>URDF/SDF 로봇 모델링</li>
                <li>ROS2 완벽 통합</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                4. ROS2 Control
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>표준화된 로봇 제어 프레임워크</li>
                <li>하드웨어 독립적 컨트롤러 개발</li>
                <li>Joint Trajectory Controller, Diff Drive Controller 등</li>
                <li>실제 로봇과 시뮬레이터 간 코드 재사용</li>
              </ul>
            </div>
          </div>

          <div className="mt-8 bg-orange-100 dark:bg-orange-900/30 rounded-lg p-6 border-l-4 border-orange-500">
            <h3 className="text-xl font-semibold text-orange-700 dark:text-orange-300 mb-3">
              다음 단계: 협동 로봇 (Cobot)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              다음 챕터에서는 사람과 함께 작업하는 <strong>협동 로봇(Collaborative Robot)</strong>의
              안전 기준, Force/Torque 제어, 그리고 Human-Robot Interaction 기술을 배웁니다.
            </p>
          </div>
        </section>

        {/* Chapter Navigation */}
        <ChapterNavigation
          currentChapter={7}
          totalChapters={8}
          moduleSlug="robotics-manipulation"
        />
      </div>
    </div>
  )
}
