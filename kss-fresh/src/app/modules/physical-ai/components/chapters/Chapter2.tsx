'use client'

import React from 'react'

export default function Chapter2() {
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
}