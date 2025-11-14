'use client';

import React from 'react';
import References from '@/components/common/References';

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

      {/* References */}
      <div className="not-prose mt-12">
        <References
          sections={[
            {
              title: '📚 로보틱스 플랫폼',
              icon: 'web' as const,
              color: 'border-teal-500',
              items: [
                {
                  title: 'NVIDIA Isaac Sim',
                  authors: 'NVIDIA',
                  year: '2024',
                  description: 'Physical AI 시뮬레이션 플랫폼 - 로봇 훈련을 위한 Omniverse 기반 환경',
                  link: 'https://developer.nvidia.com/isaac-sim'
                },
                {
                  title: 'PyBullet',
                  authors: 'Erwin Coumans',
                  year: '2024',
                  description: '물리 시뮬레이션 라이브러리 - 로봇 제어 및 RL 연구에 널리 사용',
                  link: 'https://pybullet.org/'
                },
                {
                  title: 'MuJoCo',
                  authors: 'DeepMind',
                  year: '2024',
                  description: '고성능 물리 엔진 - 복잡한 로봇 시스템 시뮬레이션',
                  link: 'https://mujoco.org/'
                },
                {
                  title: 'Gazebo',
                  authors: 'Open Robotics',
                  year: '2024',
                  description: 'ROS 통합 로봇 시뮬레이터 - 센서, 환경, 물리 시뮬레이션',
                  link: 'https://gazebosim.org/'
                },
                {
                  title: 'NVIDIA Omniverse',
                  authors: 'NVIDIA',
                  year: '2024',
                  description: '디지털 트윈 플랫폼 - PhysX 5.0 기반 실시간 물리 시뮬레이션',
                  link: 'https://www.nvidia.com/en-us/omniverse/'
                }
              ]
            },
            {
              title: '📖 핵심 논문',
              icon: 'research' as const,
              color: 'border-teal-500',
              items: [
                {
                  title: 'Embodied AI (AI2-THOR)',
                  authors: 'Deitke et al.',
                  year: '2020',
                  description: '로봇이 물리 환경에서 학습하는 Embodied AI 프레임워크',
                  link: 'https://arxiv.org/abs/1712.05474'
                },
                {
                  title: 'RT-1: Robotics Transformer',
                  authors: 'Brohan et al. (Google)',
                  year: '2022',
                  description: 'Transformer 기반 로봇 제어 - 13만 개 실제 데모 학습',
                  link: 'https://arxiv.org/abs/2212.06817'
                },
                {
                  title: 'Sim-to-Real Transfer',
                  authors: 'Peng et al.',
                  year: '2018',
                  description: '시뮬레이션 학습을 실제 로봇으로 전이 - Domain Randomization',
                  link: 'https://arxiv.org/abs/1710.06537'
                },
                {
                  title: 'NVIDIA COSMOS Platform',
                  authors: 'NVIDIA',
                  year: '2024',
                  description: 'Physical AI 세계 모델 - Foundation World Model for Robotics',
                  link: 'https://www.nvidia.com/en-us/ai-data-science/cosmos/'
                }
              ]
            },
            {
              title: '🛠️ 실전 도구',
              icon: 'tools' as const,
              color: 'border-teal-500',
              items: [
                {
                  title: 'ROS 2 (Robot Operating System)',
                  authors: 'Open Robotics',
                  year: '2024',
                  description: '로봇 미들웨어 표준 - 센서, 제어, 통신 통합 프레임워크',
                  link: 'https://docs.ros.org/en/rolling/'
                },
                {
                  title: 'OpenAI Gym',
                  authors: 'OpenAI',
                  year: '2024',
                  description: '강화학습 환경 표준 - 로봇 제어 벤치마크',
                  link: 'https://www.gymlibrary.dev/'
                },
                {
                  title: 'Stable Baselines3',
                  authors: 'DLR-RM',
                  year: '2024',
                  description: 'RL 알고리즘 라이브러리 - PPO, SAC, TD3 구현',
                  link: 'https://stable-baselines3.readthedocs.io/'
                },
                {
                  title: 'NVIDIA Isaac SDK',
                  authors: 'NVIDIA',
                  year: '2024',
                  description: '로봇 개발 도구 - 센서 처리, 내비게이션, 조작 알고리즘',
                  link: 'https://developer.nvidia.com/isaac-sdk'
                },
                {
                  title: 'PyRobot',
                  authors: 'Facebook AI Research',
                  year: '2024',
                  description: '통합 로봇 인터페이스 - 다양한 로봇 플랫폼 추상화',
                  link: 'https://github.com/facebookresearch/pyrobot'
                }
              ]
            }
          ]}
        />
      </div>
    </div>
  )
}