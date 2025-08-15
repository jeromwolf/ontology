'use client';

import { TestTube } from 'lucide-react';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          시뮬레이션과 검증
        </h2>
        
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            자율주행 개발에서 시뮬레이션은 필수입니다. 실제 도로에서 위험한 시나리오를 무제한 테스트하고,
            수백만 마일의 주행 데이터를 단시간에 생성할 수 있습니다. CARLA, AirSim 등 업계 표준 시뮬레이터를
            활용한 체계적인 검증 방법론을 학습합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🏎️ CARLA 시뮬레이터
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              <TestTube className="inline w-5 h-5 mr-2" />
              CARLA 아키텍처
            </h4>
            <div className="space-y-3">
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">Server</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Unreal Engine 4 기반 3D 시뮬레이션 환경
                </p>
              </div>
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">Client</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Python API로 제어하는 자율주행 에이전트
                </p>
              </div>
              <div>
                <span className="font-semibold text-gray-900 dark:text-white">ScenarioRunner</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  시나리오 기반 테스트 자동화
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              시뮬레이션 환경 설정
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# CARLA 클라이언트 설정
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

# 날씨 설정
weather = carla.WeatherParameters(
    cloudiness=10.0,
    precipitation=30.0,
    sun_altitude_angle=70.0
)
world.set_weather(weather)

# 차량 스폰
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)`}</pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎮 센서 시뮬레이션
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">LiDAR 센서 시뮬레이션</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# LiDAR 센서 설정 및 데이터 수집
class LidarSimulator:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        
        # LiDAR 센서 생성
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('points_per_second', '600000')
        
        # 센서 부착
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar_sensor = world.spawn_actor(
            lidar_bp, 
            lidar_transform, 
            attach_to=vehicle
        )
        
        # 데이터 콜백 등록
        self.lidar_sensor.listen(self.process_lidar_data)
    
    def process_lidar_data(self, data):
        # 포인트 클라우드 처리
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # 지면 제거 및 클러스터링
        ground_points = self.remove_ground(points)
        clusters = self.cluster_points(ground_points)
        
        # 객체 감지
        objects = self.detect_objects(clusters)
        return objects`}</pre>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">카메라 센서 시뮬레이션</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 다중 카메라 설정
class CameraSimulator:
    def __init__(self, world, vehicle):
        self.cameras = {}
        
        # 전방, 후방, 좌우 카메라 설정
        camera_positions = {
            'front': carla.Transform(carla.Location(x=1.5, z=2.4)),
            'rear': carla.Transform(carla.Location(x=-1.5, z=2.4), 
                                  carla.Rotation(yaw=180)),
            'left': carla.Transform(carla.Location(y=-0.8, z=2.4), 
                                  carla.Rotation(yaw=-90)),
            'right': carla.Transform(carla.Location(y=0.8, z=2.4), 
                                   carla.Rotation(yaw=90))
        }
        
        for name, transform in camera_positions.items():
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '1920')
            camera_bp.set_attribute('image_size_y', '1080')
            camera_bp.set_attribute('fov', '90')
            
            camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            camera.listen(lambda image, n=name: self.process_image(image, n))
            self.cameras[name] = camera
    
    def process_image(self, image, camera_name):
        # 이미지를 numpy 배열로 변환
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        img_array = img_array[:, :, :3]  # RGBA → RGB
        
        # AI 모델로 객체 감지
        detections = self.object_detector.detect(img_array)
        return detections`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🧪 시나리오 테스트
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              위험 시나리오 생성
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# 긴급 제동 시나리오
class EmergencyBrakeScenario:
    def __init__(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        
    def execute(self):
        # 전방에 갑자기 나타나는 보행자
        pedestrian_bp = random.choice(
            world.get_blueprint_library().filter('walker.pedestrian.*')
        )
        
        # 자차 전방 20m에 보행자 생성
        ego_transform = self.ego_vehicle.get_transform()
        spawn_point = ego_transform.location + \
                     ego_transform.get_forward_vector() * 20
        
        pedestrian = world.spawn_actor(
            pedestrian_bp, 
            carla.Transform(spawn_point)
        )
        
        # 보행자가 갑자기 도로로 진입
        control = carla.WalkerControl()
        control.speed = 5.0  # 5 m/s
        control.direction = ego_transform.get_right_vector()
        pedestrian.apply_control(control)
        
        # 자율주행 시스템의 반응 측정
        self.measure_response()`}</pre>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              성능 평가 메트릭
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# 자율주행 성능 평가
class PerformanceEvaluator:
    def __init__(self):
        self.metrics = {
            'safety': [],
            'comfort': [],
            'efficiency': []
        }
    
    def evaluate_safety(self, scenario_data):
        # 최소 안전 거리 유지 여부
        min_distance = min(scenario_data['distances'])
        ttc = self.calculate_time_to_collision(scenario_data)
        
        safety_score = {
            'min_distance': min_distance,
            'ttc': ttc,
            'collisions': scenario_data['collision_count'],
            'near_misses': scenario_data['near_miss_count']
        }
        
        return safety_score
    
    def evaluate_comfort(self, vehicle_data):
        # 승차감 평가
        accelerations = vehicle_data['accelerations']
        jerk = np.diff(accelerations) / 0.1  # dt = 0.1s
        
        comfort_score = {
            'max_acceleration': np.max(np.abs(accelerations)),
            'max_jerk': np.max(np.abs(jerk)),
            'avg_jerk': np.mean(np.abs(jerk))
        }
        
        return comfort_score`}</pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔄 자동화된 테스트 파이프라인
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="space-y-4">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">CI/CD 통합 테스트</h4>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# 자동화된 시뮬레이션 테스트
class AutomatedTestPipeline:
    def __init__(self):
        self.test_scenarios = [
            'highway_merge',
            'urban_intersection',
            'pedestrian_crossing',
            'emergency_vehicle',
            'construction_zone',
            'adverse_weather'
        ]
        
    def run_regression_tests(self, autonomous_system):
        results = {}
        
        for scenario in self.test_scenarios:
            # 시나리오별 테스트 실행
            test_runner = ScenarioTestRunner(scenario)
            scenario_results = test_runner.run(autonomous_system)
            
            # 결과 분석
            passed = self.analyze_results(scenario_results)
            results[scenario] = {
                'passed': passed,
                'metrics': scenario_results,
                'logs': test_runner.get_logs()
            }
            
            # 실패 시 상세 분석
            if not passed:
                self.debug_failure(scenario, scenario_results)
        
        # 테스트 리포트 생성
        self.generate_report(results)
        return all(r['passed'] for r in results.values())`}</pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🎯 검증 단계별 접근
        </h3>
        
        <div className="grid md:grid-cols-4 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">
              Software-in-the-Loop
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              순수 소프트웨어 환경에서 알고리즘 검증
            </p>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-2">
              Hardware-in-the-Loop
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              실제 ECU와 가상 환경을 연결하여 테스트
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-bold text-purple-700 dark:text-purple-400 mb-2">
              Vehicle-in-the-Loop
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              실제 차량과 가상 환경을 연결한 최종 테스트
            </p>
          </div>
          
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
            <h4 className="font-bold text-orange-700 dark:text-orange-400 mb-2">
              Real-World Testing
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              실제 도로에서의 최종 검증 및 데이터 수집
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}