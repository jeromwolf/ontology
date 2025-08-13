'use client'

import React from 'react'

export default function Chapter9() {
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
}