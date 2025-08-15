'use client';

import { 
  Bot, Cog, Users, Shield, Network, Eye, Activity, MapPin, Brain, Building, TestTube, Code
} from 'lucide-react';
import Link from 'next/link';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Cog className="w-6 h-6 text-slate-600" />
            산업용 로봇 5대 종류
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">1. 관절형 로봇 (Articulated Robot)</h4>
              <p className="text-sm text-blue-700 dark:text-blue-300 mb-2">가장 범용적인 6축 로봇팔로 복잡한 3D 작업 수행</p>
              <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
                <li>• 작업 반경: 500mm~3,000mm, 하중: 3kg~300kg</li>
                <li>• 용접, 도장, 조립, 팔레타이징에 최적화</li>
                <li>• 대표 모델: KUKA KR 시리즈, ABB IRB 시리즈</li>
                <li>• 정밀도: ±0.02mm, 반복도: ±0.01mm</li>
              </ul>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-400 rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">2. 직교좌표형 로봇 (Cartesian Robot)</h4>
              <p className="text-sm text-green-700 dark:text-green-300 mb-2">X-Y-Z 축의 직선 운동으로 정밀한 위치 제어</p>
              <ul className="text-xs text-green-600 dark:text-green-400 space-y-1">
                <li>• 갲트리 방식: 대형 작업공간, 높은 강성</li>
                <li>• 픽앤플레이스, 3D 프린팅, CNC 로딩에 사용</li>
                <li>• 프로그래밍 간단: 직교좌표계 기반</li>
                <li>• 속도: 최대 10m/s, 가속도: 50m/s²</li>
              </ul>
            </div>

            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-400 rounded">
              <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">3. SCARA 로봇 (Selective Compliant Robot)</h4>
              <p className="text-sm text-purple-700 dark:text-purple-300 mb-2">수평면 작업에 특화된 고속 조립용 로봇</p>
              <ul className="text-xs text-purple-600 dark:text-purple-400 space-y-1">
                <li>• 4축 구조: R-R-P-R (회전-회전-직선-회전)</li>
                <li>• 전자부품 조립, 검사, 패키징에 최적화</li>
                <li>• 초고속: 사이클 타임 1초 이내</li>
                <li>• 수직 방향 컴플라이언스로 안전한 조립</li>
              </ul>
            </div>

            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-400 rounded">
              <h4 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">4. 델타 로봇 (Delta Robot)</h4>
              <p className="text-sm text-orange-700 dark:text-orange-300 mb-2">병렬 구조의 초고속 피킹 전용 로봇</p>
              <ul className="text-xs text-orange-600 dark:text-orange-400 space-y-1">
                <li>• 3개 팔의 병렬 구동으로 초고속 실현</li>
                <li>• 식품, 의약품 패키징에 주로 사용</li>
                <li>• 속도: 300 picks/min, 가속도: 100G</li>
                <li>• 작업 공간: 원통형, 높은 정밀도</li>
              </ul>
            </div>

            <div className="p-4 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-400 rounded">
              <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">5. 원통좌표형 로봇 (Cylindrical Robot)</h4>
              <p className="text-sm text-red-700 dark:text-red-300 mb-2">원통형 작업공간을 가진 간단한 구조의 로봇</p>
              <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
                <li>• R-P-P 구조: 회전-직선-직선 운동</li>
                <li>• 단순 반복 작업에 적합 (스폿 용접 등)</li>
                <li>• 저비용, 높은 신뢰성</li>
                <li>• 제한된 유연성, 특수 용도 전용</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 시뮬레이터 체험 섹션 */}
        <div className="mt-8 p-6 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded-xl border border-orange-200 dark:border-orange-800">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-200 mb-2">
                🎮 생산 라인 모니터링 시뮬레이터
              </h3>
              <p className="text-sm text-orange-700 dark:text-orange-300">
                자동화된 생산 라인의 실시간 모니터링과 로봇 협업을 체험해보세요.
              </p>
            </div>
            <Link
              href="/modules/smart-factory/simulators/production-line-monitor?from=/modules/smart-factory/robotics-automation"
              className="inline-flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors"
            >
              <span>시뮬레이터 체험</span>
              <span className="text-lg">→</span>
            </Link>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Users className="w-6 h-6 text-slate-600" />
            협동로봇(Cobot) 혁신
          </h3>
          <div className="space-y-6">
            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
                <Shield className="w-5 h-5 text-green-500" />
                안전 기술 표준 (ISO 10218, ISO 15066)
              </h4>
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded border">
                <div className="grid grid-cols-1 gap-4 text-sm">
                  <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded border border-green-200 dark:border-green-700">
                    <h5 className="font-medium text-green-700 dark:text-green-300 mb-2">4가지 협업 모드</h5>
                    <div className="space-y-2 text-xs text-green-600 dark:text-green-400">
                      <div><strong>1. 안전 정격 모니터링 정지:</strong> 인간 접근 시 자동 정지</div>
                      <div><strong>2. 핸드 가이딩:</strong> 인간이 로봇을 직접 조작</div>
                      <div><strong>3. 속도/거리 모니터링:</strong> 거리에 따른 속도 조절</div>
                      <div><strong>4. 파워/포스 제한:</strong> 충돌 시 안전한 힘으로 제한</div>
                    </div>
                  </div>
                  <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded border border-blue-200 dark:border-blue-700">
                    <h5 className="font-medium text-blue-700 dark:text-blue-300 mb-2">안전 센서 기술</h5>
                    <div className="grid grid-cols-2 gap-2 text-xs text-blue-600 dark:text-blue-400">
                      <div>• 토크 센서 (각 관절)</div>
                      <div>• 6축 Force/Torque 센서</div>
                      <div>• 스킨 센서 (접촉 감지)</div>
                      <div>• 3D 비전 시스템</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
                <Network className="w-5 h-5 text-blue-500" />
                ROS 기반 프로그래밍
              </h4>
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded border">
                <div className="space-y-3">
                  <div className="text-xs text-slate-600 dark:text-slate-400">
                    <div className="flex items-center gap-2 mb-1">
                      <Code className="w-4 h-4" />
                      <span className="font-medium">ROS 노드 프로그래밍 예제</span>
                    </div>
                    <div className="bg-gray-900 text-green-400 p-3 rounded font-mono text-xs overflow-x-auto">
{`#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_commander import RobotCommander, PlanningSceneInterface

class CollaborativeRobotController:
    def __init__(self):
        rospy.init_node('cobot_controller')
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = self.robot.get_group('manipulator')
        
        # 안전 파라미터 설정
        self.group.set_max_velocity_scaling_factor(0.3)  # 30% 속도로 제한
        self.group.set_max_acceleration_scaling_factor(0.3)
        
        # Force/Torque 임계값
        self.force_threshold = 20.0  # 20N
        self.torque_threshold = 5.0  # 5Nm
        
    def safe_move_to_pose(self, target_pose):
        # 경로 계획
        self.group.set_pose_target(target_pose)
        plan = self.group.plan()
        
        # 안전성 검증
        if self.is_path_safe(plan[1]):
            # 실시간 모니터링과 함께 실행
            self.group.execute(plan[1], wait=False)
            self.monitor_execution()
        else:
            rospy.logwarn("Unsafe path detected. Aborting motion.")
    
    def monitor_execution(self):
        rate = rospy.Rate(100)  # 100Hz 모니터링
        while self.group.get_current_joint_values():
            # Force/Torque 모니터링
            wrench = self.get_end_effector_wrench()
            if (abs(wrench.force.x) > self.force_threshold or 
                abs(wrench.torque.z) > self.torque_threshold):
                self.emergency_stop()
                break
            rate.sleep()
    
    def emergency_stop(self):
        self.group.stop()
        rospy.logwarn("Emergency stop triggered!")

# 사용 예제
controller = CollaborativeRobotController()
target = PoseStamped()
target.pose.position.x = 0.5
target.pose.position.y = 0.2
target.pose.position.z = 0.3
controller.safe_move_to_pose(target)`}
                    </div>
                  </div>
                  <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• ROS 2 기반 실시간 제어 시스템</li>
                    <li>• MoveIt! 모션 플래닝 라이브러리</li>
                    <li>• 안전 감시: 100Hz 실시간 모니터링</li>
                    <li>• 크로스 플랫폼: Ubuntu, ROS Industrial</li>
                  </ul>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
                <Eye className="w-5 h-5 text-purple-500" />
                로봇 비전 시스템
              </h4>
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded border">
                <h5 className="font-medium text-slate-700 dark:text-slate-300 mb-2">2D/3D 통합 비전</h5>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="bg-blue-100 dark:bg-blue-900/30 p-2 rounded">
                      <strong className="text-blue-700 dark:text-blue-300">2D 비전</strong>
                      <ul className="text-blue-600 dark:text-blue-400 mt-1 space-y-0.5">
                        <li>• 패턴 매칭</li>
                        <li>• OCR/바코드 인식</li>
                        <li>• 결함 검출</li>
                        <li>• 칼라 분류</li>
                      </ul>
                    </div>
                    <div className="bg-purple-100 dark:bg-purple-900/30 p-2 rounded">
                      <strong className="text-purple-700 dark:text-purple-300">3D 비전</strong>
                      <ul className="text-purple-600 dark:text-purple-400 mt-1 space-y-0.5">
                        <li>• 깊이 맵 생성</li>
                        <li>• 6D 포즈 추정</li>
                        <li>• Bin Picking</li>
                        <li>• 충돌 회피</li>
                      </ul>
                    </div>
                  </div>
                  <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                    <strong>대표 응용:</strong> Random Bin Picking, Quality Inspection, Assembly Guidance
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* AGV/AMR 시스템 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Activity className="w-7 h-7 text-slate-600" />
          AGV/AMR 시스템: 무인 운송의 진화
        </h3>
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-lg border border-blue-200 dark:border-blue-700">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-4 flex items-center gap-2">
              <MapPin className="w-5 h-5" />
              전통적 AGV (Automated Guided Vehicle)
            </h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-blue-900/30 p-3 rounded border">
                <h5 className="font-medium text-blue-700 dark:text-blue-300 mb-2">가이드 방식</h5>
                <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
                  <li>• 자기 테이프: 바닥 매설, 정확한 경로</li>
                  <li>• 레이저 가이드: 반사판 기준 삼각측량</li>
                  <li>• 유도선: 전류 유도, 고정 경로</li>
                  <li>• 비전 가이드: QR 코드, 색선 추적</li>
                </ul>
              </div>
              <div className="text-xs text-blue-700 dark:text-blue-300">
                <strong>장점:</strong> 높은 정밀도, 안정성<br/>
                <strong>단점:</strong> 경로 변경 어려움, 인프라 투자 대
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-lg border border-green-200 dark:border-green-700">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5" />
              지능형 AMR (Autonomous Mobile Robot)
            </h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-green-900/30 p-3 rounded border">
                <h5 className="font-medium text-green-700 dark:text-green-300 mb-2">SLAM 기술</h5>
                <ul className="text-xs text-green-600 dark:text-green-400 space-y-1">
                  <li>• LiDAR SLAM: 2D/3D 환경 매핑</li>
                  <li>• Visual SLAM: 카메라 기반</li>
                  <li>• IMU 융합: 관성 측정 보정</li>
                  <li>• 실시간 리매핑: 동적 환경 대응</li>
                </ul>
              </div>
              <div className="text-xs text-green-700 dark:text-green-300">
                <strong>장점:</strong> 유연한 경로, 자율 주행<br/>
                <strong>단점:</strong> 복잡한 환경에서 불안정
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 p-6 rounded-lg border border-purple-200 dark:border-purple-700">
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-4 flex items-center gap-2">
              <Network className="w-5 h-5" />
              플릿 관리 시스템 (FMS)
            </h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-purple-900/30 p-3 rounded border">
                <h5 className="font-medium text-purple-700 dark:text-purple-300 mb-2">중앙 제어</h5>
                <ul className="text-xs text-purple-600 dark:text-purple-400 space-y-1">
                  <li>• 트래픽 제어: 교차로, 좁은 통로</li>
                  <li>• 작업 스케줄링: 최적 경로, 우선순위</li>
                  <li>• 배터리 관리: 자동 충전 스케줄</li>
                  <li>• 예측 정비: 주행 거리, 사용 시간</li>
                </ul>
              </div>
              <div className="text-xs text-purple-700 dark:text-purple-300">
                <strong>효과:</strong> 운영 효율 40% 향상, 충돌 사고 0건 달성
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 글로벌 기업 도입 사례 */}
      <div className="bg-gradient-to-br from-slate-50 to-gray-50 dark:from-slate-800 dark:to-gray-800 rounded-2xl p-8 border border-slate-200 dark:border-slate-700">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Building className="w-7 h-7 text-slate-600" />
          로봇 자동화 도입 사례
        </h3>
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-slate-700 rounded-xl p-6 border border-slate-200 dark:border-slate-600">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center">
                <span className="text-red-600 dark:text-red-400 font-bold text-sm">Tesla</span>
              </div>
              <div>
                <h4 className="font-bold text-gray-900 dark:text-white">Tesla Gigafactory</h4>
                <p className="text-xs text-gray-500 dark:text-gray-400">전기차 배터리 제조</p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded border border-red-200 dark:border-red-700">
                <h5 className="font-semibold text-red-800 dark:text-red-200 text-sm mb-1">Alien Dreadnought 라인</h5>
                <p className="text-xs text-red-700 dark:text-red-300">완전 자동화 배터리 셀 생산</p>
              </div>
              <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <li>• <strong>자동화율:</strong> 95% (인간 개입 최소화)</li>
                <li>• <strong>생산 속도:</strong> 초당 10개 배터리 셀</li>
                <li>• <strong>품질:</strong> PPM 단위 불량률</li>
                <li>• <strong>유연성:</strong> 5분 내 제품 변경</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-700 rounded-xl p-6 border border-slate-200 dark:border-slate-600">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <span className="text-blue-600 dark:text-blue-400 font-bold text-sm">Amazon</span>
              </div>
              <div>
                <h4 className="font-bold text-gray-900 dark:text-white">Amazon Fulfillment</h4>
                <p className="text-xs text-gray-500 dark:text-gray-400">물류창고 자동화</p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-700">
                <h5 className="font-semibold text-blue-800 dark:text-blue-200 text-sm mb-1">Kiva 로봇 시스템</h5>
                <p className="text-xs text-blue-700 dark:text-blue-300">45,000대 AMR 동시 운영</p>
              </div>
              <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <li>• <strong>처리량:</strong> 시간당 1,000개 주문</li>
                <li>• <strong>효율:</strong> 피킹 시간 75% 단축</li>
                <li>• <strong>정확도:</strong> 99.99% 배송 정확도</li>
                <li>• <strong>운영비:</strong> 20% 절감 효과</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-700 rounded-xl p-6 border border-slate-200 dark:border-slate-600">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                <span className="text-green-600 dark:text-green-400 font-bold text-sm">Foxconn</span>
              </div>
              <div>
                <h4 className="font-bold text-gray-900 dark:text-white">Foxconn Smart Factory</h4>
                <p className="text-xs text-gray-500 dark:text-gray-400">전자제품 조립</p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded border border-green-200 dark:border-green-700">
                <h5 className="font-semibold text-green-800 dark:text-green-200 text-sm mb-1">Foxbot 협동로봇</h5>
                <p className="text-xs text-green-700 dark:text-green-300">인간과 로봇의 완벽한 협업</p>
              </div>
              <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <li>• <strong>배치:</strong> 공장당 1,000대 로봇</li>
                <li>• <strong>협업:</strong> 안전사고 0건 달성</li>
                <li>• <strong>생산성:</strong> 조립 속도 3배 향상</li>
                <li>• <strong>품질:</strong> 검사 정확도 99.5%</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* 실습 프로젝트 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <TestTube className="w-7 h-7 text-slate-600" />
          실습: ROS 시뮬레이터로 로봇 제어 프로그래밍
        </h3>
        <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
{`# Gazebo 시뮬레이터에서 6축 로봇팔 제어
#!/usr/bin/env python3

import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_msgs.msg import DisplayTrajectory
import sys

class RobotArmController:
    def __init__(self):
        # MoveIt! 초기화
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('robot_arm_controller')
        
        # 로봇 및 씬 설정
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        
        # 디스플레이 퍼블리셔
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            DisplayTrajectory, queue_size=20
        )
        
        # 로봇 상태 출력
        planning_frame = self.group.get_planning_frame()
        eef_link = self.group.get_end_effector_link()
        group_names = self.robot.get_group_names()
        
        print(f"Planning frame: {planning_frame}")
        print(f"End effector link: {eef_link}")
        print(f"Available Planning Groups: {group_names}")
    
    def plan_and_execute_pose(self, pose_target):
        """목표 포즈로 이동"""
        self.group.set_pose_target(pose_target)
        
        # 경로 계획
        plan = self.group.plan()
        
        if len(plan[1].joint_trajectory.points) > 0:
            print("✅ 경로 계획 성공!")
            
            # 궤적 시각화
            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan[1])
            self.display_trajectory_publisher.publish(display_trajectory)
            
            # 실행
            self.group.execute(plan[1], wait=True)
            self.group.stop()
            self.group.clear_pose_targets()
            
            return True
        else:
            print("❌ 경로 계획 실패!")
            return False
    
    def plan_and_execute_joint_goal(self, joint_goal):
        """조인트 각도로 이동"""
        self.group.go(joint_goal, wait=True)
        self.group.stop()
    
    def add_collision_object(self, name, pose, size):
        """충돌 객체 추가"""
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.group.get_planning_frame()
        box_pose.pose = pose
        
        self.scene.add_box(name, box_pose, size)
        return self.wait_for_state_update(name, box_is_known=True)
    
    def wait_for_state_update(self, box_name, box_is_known=False, timeout=4):
        """씬 업데이트 대기"""
        start = rospy.get_time()
        seconds = rospy.get_time()
        
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = self.scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0
            
            is_known = box_name in self.scene.get_known_object_names()
            
            if (box_is_known == is_known):
                return True
                
            rospy.sleep(0.1)
            seconds = rospy.get_time()
            
        return False

# 실제 시뮬레이션 시나리오
def main():
    controller = RobotArmController()
    
    # 시나리오 1: 홈 포지션으로 이동
    print("🏠 홈 포지션으로 이동...")
    joint_goal = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    controller.plan_and_execute_joint_goal(joint_goal)
    
    rospy.sleep(2)
    
    # 시나리오 2: 충돌 객체 추가 (테이블)
    print("📦 테이블 추가...")
    table_pose = geometry_msgs.msg.Pose()
    table_pose.position.x = 0.5
    table_pose.position.y = 0
    table_pose.position.z = -0.1
    controller.add_collision_object("table", table_pose, (0.8, 0.8, 0.2))
    
    rospy.sleep(1)
    
    # 시나리오 3: 피킹 위치로 이동
    print("🎯 피킹 위치로 이동...")
    pose_target = geometry_msgs.msg.Pose()
    pose_target.orientation.w = 1.0
    pose_target.position.x = 0.4
    pose_target.position.y = 0.1
    pose_target.position.z = 0.4
    
    if controller.plan_and_execute_pose(pose_target):
        print("✅ 피킹 완료!")
        
        rospy.sleep(2)
        
        # 시나리오 4: 플레이싱 위치로 이동
        print("📍 플레이싱 위치로 이동...")
        pose_target.position.x = 0.1
        pose_target.position.y = 0.4
        pose_target.position.z = 0.4
        
        if controller.plan_and_execute_pose(pose_target):
            print("✅ 플레이싱 완료!")
        
    print("🎉 로봇 작업 완료!")
    
    # 정리
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()`}
        </div>
        <div className="mt-4 grid lg:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded border border-blue-200 dark:border-blue-700">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">시뮬레이터 환경</h4>
            <ul className="text-xs text-blue-700 dark:text-blue-300 space-y-1">
              <li>• <strong>Gazebo:</strong> 물리 엔진 기반 시뮬레이션</li>
              <li>• <strong>RViz:</strong> 3D 시각화 및 디버깅</li>
              <li>• <strong>MoveIt!:</strong> 모션 플래닝 라이브러리</li>
              <li>• <strong>로봇 모델:</strong> UR5, ABB IRB120 지원</li>
            </ul>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded border border-green-200 dark:border-green-700">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">학습 효과</h4>
            <ul className="text-xs text-green-700 dark:text-green-300 space-y-1">
              <li>• <strong>안전:</strong> 실제 로봇 사용 전 검증</li>
              <li>• <strong>비용 절약:</strong> 하드웨어 없이 학습</li>
              <li>• <strong>반복 학습:</strong> 무제한 시나리오 테스트</li>
              <li>• <strong>실무 연계:</strong> 실제 로봇에 바로 적용</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}