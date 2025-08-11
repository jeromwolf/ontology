'use client'

import { Route } from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          경로 계획과 제어
        </h2>
        
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            자율주행차가 목적지까지 안전하고 효율적으로 이동하는 "두뇌"에 해당합니다.
            실시간으로 변하는 도로 환경에서 최적의 경로를 계획하고, 차량의 물리적 한계를
            고려한 정밀한 제어를 수행합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🗺️ 경로 계획 알고리즘
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              <Route className="inline w-5 h-5 mr-2" />
              A* 알고리즘
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# A* 경로 계획 구현
class AStarPlanner:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        
    def plan(self, start, goal):
        open_set = PriorityQueue()
        open_set.put((0, start))
        
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while not open_set.empty():
            current = open_set.get()[1]
            
            if current == goal:
                return self.reconstruct_path(current)
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.distance(current, neighbor)
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))`}</pre>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              <Route className="inline w-5 h-5 mr-2" />
              RRT* (Rapidly-exploring Random Tree)
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# RRT* 경로 계획
class RRTStar:
    def plan(self, start, goal, obstacles):
        tree = [start]
        
        for i in range(max_iterations):
            # 랜덤 샘플링
            x_rand = self.sample_free_space()
            
            # 가장 가까운 노드 찾기
            x_nearest = self.nearest_neighbor(tree, x_rand)
            
            # 스티어링
            x_new = self.steer(x_nearest, x_rand)
            
            if self.collision_free(x_nearest, x_new, obstacles):
                # 근처 노드들 재연결 (최적화)
                x_near = self.near_neighbors(tree, x_new)
                x_parent = self.choose_parent(x_near, x_new)
                
                tree.add(x_new, x_parent)
                self.rewire(tree, x_near, x_new)
                
                if self.near_goal(x_new, goal):
                    return self.extract_path(tree, start, goal)`}</pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎯 경로 추종 제어
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">Pure Pursuit 알고리즘</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# Pure Pursuit 경로 추종
class PurePursuitController:
    def __init__(self, wheelbase):
        self.L = wheelbase  # 휠베이스
        self.lookahead_distance = 5.0  # 전방 주시 거리
        
    def compute_steering(self, current_pose, path):
        # 전방 주시점 찾기
        lookahead_point = self.find_lookahead_point(current_pose, path)
        
        # 차량 좌표계로 변환
        dx = lookahead_point.x - current_pose.x
        dy = lookahead_point.y - current_pose.y
        
        # 차량 좌표계에서의 상대 위치
        alpha = atan2(dy, dx) - current_pose.yaw
        
        # 조향각 계산
        ld = sqrt(dx**2 + dy**2)
        steering_angle = atan2(2 * self.L * sin(alpha), ld)
        
        return steering_angle`}</pre>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">Model Predictive Control (MPC)</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# MPC 제어기
class ModelPredictiveController:
    def __init__(self, vehicle_model):
        self.model = vehicle_model
        self.horizon = 10  # 예측 구간
        self.dt = 0.1  # 시간 간격
        
    def optimize_control(self, current_state, reference_traj):
        """최적 제어 입력 계산"""
        # 목적 함수: 경로 추종 오차 + 제어 입력 최소화
        def cost_function(u_sequence):
            cost = 0
            state = current_state
            
            for i in range(self.horizon):
                # 차량 모델로 상태 예측
                state = self.model.predict(state, u_sequence[i])
                
                # 경로 오차
                error = self.compute_error(state, reference_traj[i])
                cost += error.T @ Q @ error
                
                # 제어 입력 페널티
                cost += u_sequence[i].T @ R @ u_sequence[i]
                
            return cost
        
        # 제약 조건
        constraints = [
            {'type': 'ineq', 'fun': lambda u: self.max_steering - abs(u[0])},
            {'type': 'ineq', 'fun': lambda u: self.max_accel - abs(u[1])}
        ]
        
        # 최적화
        result = minimize(cost_function, u_init, constraints=constraints)
        return result.x[0]  # 첫 번째 제어 입력만 사용`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚗 차량 동역학 모델
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Kinematic Bicycle Model
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# 운동학 모델 (저속)
def kinematic_model(state, control, dt):
    x, y, theta, v = state
    delta, a = control  # 조향각, 가속도
    
    # 상태 업데이트
    x_new = x + v * cos(theta) * dt
    y_new = y + v * sin(theta) * dt
    theta_new = theta + (v/L) * tan(delta) * dt
    v_new = v + a * dt
    
    return [x_new, y_new, theta_new, v_new]`}</pre>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Dynamic Model
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# 동역학 모델 (고속)
def dynamic_model(state, control, params):
    # 타이어 슬립 각도 고려
    alpha_f = delta - atan2(vy + lf*r, vx)
    alpha_r = -atan2(vy - lr*r, vx)
    
    # 타이어 힘 (Pacejka 모델)
    Fy_f = Df * sin(Cf * atan(Bf * alpha_f))
    Fy_r = Dr * sin(Cr * atan(Br * alpha_r))
    
    # 가속도 계산
    ax = (Fx - Fy_f*sin(delta))/m + vy*r
    ay = (Fy_f*cos(delta) + Fy_r)/m - vx*r
    
    return ax, ay`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          📊 계획 계층 구조
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">
              Route Planning
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              도로 네트워크 수준의 경로 결정
            </p>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-2">
              Motion Planning
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              구체적인 궤적 생성과 시공간 경로 계획
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-bold text-purple-700 dark:text-purple-400 mb-2">
              Control Execution
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              실제 액추에이터 제어 신호 생성
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}