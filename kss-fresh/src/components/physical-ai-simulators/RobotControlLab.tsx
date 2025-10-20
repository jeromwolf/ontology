'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Target, Move } from 'lucide-react';

interface RobotArm {
  joint1: number; // base rotation
  joint2: number; // shoulder
  joint3: number; // elbow
  endEffector: { x: number; y: number };
}

export default function RobotControlLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [robotArm, setRobotArm] = useState<RobotArm>({
    joint1: 0,
    joint2: Math.PI / 4,
    joint3: Math.PI / 4,
    endEffector: { x: 0, y: 0 }
  });

  const [targetPosition, setTargetPosition] = useState({ x: 200, y: 100 });
  const [isMoving, setIsMoving] = useState(false);
  const [controlMode, setControlMode] = useState<'forward' | 'inverse'>('forward');
  const [pathHistory, setPathHistory] = useState<{x: number, y: number}[]>([]);

  // 로봇 팔 파라미터
  const L1 = 100; // 첫 번째 링크 길이
  const L2 = 80;  // 두 번째 링크 길이
  const L3 = 60;  // 세 번째 링크 길이
  const baseX = 400;
  const baseY = 400;

  // 순기구학 (Forward Kinematics)
  const forwardKinematics = (j1: number, j2: number, j3: number) => {
    // Joint 1 (base)
    const x1 = baseX;
    const y1 = baseY;

    // Joint 2 (shoulder)
    const x2 = x1 + L1 * Math.cos(j1);
    const y2 = y1 - L1 * Math.sin(j1);

    // Joint 3 (elbow)
    const x3 = x2 + L2 * Math.cos(j1 + j2);
    const y3 = y2 - L2 * Math.sin(j1 + j2);

    // End effector
    const x4 = x3 + L3 * Math.cos(j1 + j2 + j3);
    const y4 = y3 - L3 * Math.sin(j1 + j2 + j3);

    return {
      joint1Pos: { x: x1, y: y1 },
      joint2Pos: { x: x2, y: y2 },
      joint3Pos: { x: x3, y: y3 },
      endEffector: { x: x4, y: y4 }
    };
  };

  // 역기구학 (Inverse Kinematics) - Analytical Solution
  const inverseKinematics = (targetX: number, targetY: number) => {
    // 목표 지점을 베이스 좌표계로 변환
    const dx = targetX - baseX;
    const dy = baseY - targetY; // Y축 반전

    const distance = Math.sqrt(dx * dx + dy * dy);

    // 도달 가능 범위 체크
    const maxReach = L1 + L2 + L3;
    const minReach = Math.abs(L1 - L2 - L3);

    if (distance > maxReach || distance < minReach) {
      return null; // 도달 불가
    }

    // 2-링크 역기구학 (L1 + L2를 하나의 체인으로 간주)
    const effectiveLength = L1 + L2;

    // Elbow-up 솔루션
    const j1 = Math.atan2(dy, dx);

    // Law of cosines for j2
    const D = (distance * distance - effectiveLength * effectiveLength - L3 * L3) / (2 * effectiveLength * L3);
    const D_clamped = Math.max(-1, Math.min(1, D)); // Clamp to [-1, 1]
    const j3 = Math.acos(D_clamped);

    // Calculate j2
    const alpha = Math.atan2(L3 * Math.sin(j3), effectiveLength + L3 * Math.cos(j3));
    const beta = Math.atan2(dy, dx);
    const j2 = beta - alpha - j1;

    return {
      joint1: j1,
      joint2: j2,
      joint3: j3 - Math.PI
    };
  };

  // Canvas 드로잉
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let i = 0; i < canvas.width; i += 50) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height);
      ctx.stroke();
    }
    for (let i = 0; i < canvas.height; i += 50) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i);
      ctx.stroke();
    }

    // 도달 가능 범위 표시
    const maxReach = L1 + L2 + L3;
    const minReach = Math.abs(L1 - L2 - L3);

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // 최대 도달 범위
    ctx.beginPath();
    ctx.arc(baseX, baseY, maxReach, 0, Math.PI * 2);
    ctx.stroke();

    // 최소 도달 범위
    ctx.beginPath();
    ctx.arc(baseX, baseY, minReach, 0, Math.PI * 2);
    ctx.stroke();

    ctx.setLineDash([]);

    // Path history (fading trail)
    if (pathHistory.length > 1) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      pathHistory.forEach((point, i) => {
        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }

    // 로봇 팔 계산
    const positions = forwardKinematics(
      robotArm.joint1,
      robotArm.joint2,
      robotArm.joint3
    );

    // 로봇 팔 그리기
    // Base
    ctx.fillStyle = '#475569';
    ctx.beginPath();
    ctx.arc(baseX, baseY, 15, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(baseX - 20, baseY);
    ctx.lineTo(baseX + 20, baseY);
    ctx.stroke();

    // Link 1
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(positions.joint1Pos.x, positions.joint1Pos.y);
    ctx.lineTo(positions.joint2Pos.x, positions.joint2Pos.y);
    ctx.stroke();

    // Link 2
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(positions.joint2Pos.x, positions.joint2Pos.y);
    ctx.lineTo(positions.joint3Pos.x, positions.joint3Pos.y);
    ctx.stroke();

    // Link 3
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(positions.joint3Pos.x, positions.joint3Pos.y);
    ctx.lineTo(positions.endEffector.x, positions.endEffector.y);
    ctx.stroke();

    // Joints
    const drawJoint = (x: number, y: number, size: number, color: string) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.stroke();
    };

    drawJoint(positions.joint1Pos.x, positions.joint1Pos.y, 10, '#94a3b8');
    drawJoint(positions.joint2Pos.x, positions.joint2Pos.y, 8, '#94a3b8');
    drawJoint(positions.joint3Pos.x, positions.joint3Pos.y, 6, '#94a3b8');

    // End Effector (gripper)
    const ee = positions.endEffector;
    const angle = robotArm.joint1 + robotArm.joint2 + robotArm.joint3;

    // Gripper arms
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.moveTo(ee.x, ee.y);
    ctx.lineTo(
      ee.x + 15 * Math.cos(angle + Math.PI/4),
      ee.y - 15 * Math.sin(angle + Math.PI/4)
    );
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(ee.x, ee.y);
    ctx.lineTo(
      ee.x + 15 * Math.cos(angle - Math.PI/4),
      ee.y - 15 * Math.sin(angle - Math.PI/4)
    );
    ctx.stroke();

    // End effector marker
    drawJoint(ee.x, ee.y, 10, '#3b82f6');

    // 목표 지점
    ctx.strokeStyle = '#22c55e';
    ctx.fillStyle = '#22c55e44';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(targetPosition.x, targetPosition.y, 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // 크로스헤어
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(targetPosition.x - 20, targetPosition.y);
    ctx.lineTo(targetPosition.x + 20, targetPosition.y);
    ctx.moveTo(targetPosition.x, targetPosition.y - 20);
    ctx.lineTo(targetPosition.x, targetPosition.y + 20);
    ctx.stroke();

    // 거리 표시
    const distance = Math.sqrt(
      Math.pow(ee.x - targetPosition.x, 2) +
      Math.pow(ee.y - targetPosition.y, 2)
    );

    ctx.fillStyle = '#ffffff';
    ctx.font = '14px monospace';
    ctx.fillText(`Distance: ${distance.toFixed(1)}px`, 10, 30);

    // Joint angles
    ctx.fillText(`J1: ${(robotArm.joint1 * 180 / Math.PI).toFixed(1)}°`, 10, 50);
    ctx.fillText(`J2: ${(robotArm.joint2 * 180 / Math.PI).toFixed(1)}°`, 10, 70);
    ctx.fillText(`J3: ${(robotArm.joint3 * 180 / Math.PI).toFixed(1)}°`, 10, 90);

    // End effector position
    ctx.fillText(`EE: (${ee.x.toFixed(0)}, ${ee.y.toFixed(0)})`, 10, 110);

  }, [robotArm, targetPosition, pathHistory]);

  // Inverse Kinematics 자동 실행
  useEffect(() => {
    if (!isMoving || controlMode !== 'inverse') return;

    const interval = setInterval(() => {
      const solution = inverseKinematics(targetPosition.x, targetPosition.y);

      if (solution) {
        setRobotArm(prev => {
          const newArm = {
            joint1: solution.joint1,
            joint2: solution.joint2,
            joint3: solution.joint3,
            endEffector: prev.endEffector
          };

          const positions = forwardKinematics(newArm.joint1, newArm.joint2, newArm.joint3);
          newArm.endEffector = positions.endEffector;

          // Update path history
          setPathHistory(prev => [...prev.slice(-200), positions.endEffector]);

          return newArm;
        });
      }
    }, 50);

    return () => clearInterval(interval);
  }, [isMoving, targetPosition, controlMode]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setTargetPosition({ x, y });

    if (controlMode === 'inverse') {
      const solution = inverseKinematics(x, y);
      if (solution) {
        setRobotArm(prev => ({
          ...prev,
          joint1: solution.joint1,
          joint2: solution.joint2,
          joint3: solution.joint3
        }));
      }
    }
  };

  const handleReset = () => {
    setRobotArm({
      joint1: 0,
      joint2: Math.PI / 4,
      joint3: Math.PI / 4,
      endEffector: { x: 0, y: 0 }
    });
    setTargetPosition({ x: 200, y: 100 });
    setPathHistory([]);
    setIsMoving(false);
  };

  const handleJointChange = (joint: 'joint1' | 'joint2' | 'joint3', value: number) => {
    setRobotArm(prev => {
      const newArm = { ...prev, [joint]: value };
      const positions = forwardKinematics(newArm.joint1, newArm.joint2, newArm.joint3);
      newArm.endEffector = positions.endEffector;
      return newArm;
    });
  };

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">로봇 제어 실험실</h2>
          <p className="text-gray-400 text-sm">순기구학(FK)과 역기구학(IK)을 시각화하는 3-DOF 로봇 팔</p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setIsMoving(!isMoving)}
            className={`px-4 py-2 ${isMoving ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white rounded-lg flex items-center gap-2 transition-colors`}
          >
            {isMoving ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isMoving ? '정지' : '자동 추적'}
          </button>
          <button
            onClick={handleReset}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg flex items-center gap-2 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            초기화
          </button>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-4 mb-4">
        <div className="col-span-2 bg-gray-800 rounded-lg p-4">
          <label className="flex items-center gap-2 mb-2">
            <input
              type="radio"
              checked={controlMode === 'forward'}
              onChange={() => setControlMode('forward')}
            />
            <Move className="w-4 h-4 text-blue-400" />
            <span className="text-white font-semibold">순기구학 (Forward Kinematics)</span>
          </label>
          <p className="text-xs text-gray-400">관절 각도로 끝점 위치 계산</p>
        </div>

        <div className="col-span-2 bg-gray-800 rounded-lg p-4">
          <label className="flex items-center gap-2 mb-2">
            <input
              type="radio"
              checked={controlMode === 'inverse'}
              onChange={() => setControlMode('inverse')}
            />
            <Target className="w-4 h-4 text-green-400" />
            <span className="text-white font-semibold">역기구학 (Inverse Kinematics)</span>
          </label>
          <p className="text-xs text-gray-400">목표 위치로 관절 각도 계산</p>
        </div>

        <div className="bg-blue-900/30 rounded-lg p-4">
          <div className="text-xs text-blue-400 mb-1">Current Mode</div>
          <div className="text-lg font-bold text-blue-400">
            {controlMode === 'forward' ? 'FK' : 'IK'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        {controlMode === 'forward' && (
          <>
            <div className="bg-gray-800 rounded-lg p-4">
              <label className="text-sm font-semibold text-white mb-2 block">
                Joint 1 (Base): {(robotArm.joint1 * 180 / Math.PI).toFixed(1)}°
              </label>
              <input
                type="range"
                min={-Math.PI}
                max={Math.PI}
                step={0.01}
                value={robotArm.joint1}
                onChange={(e) => handleJointChange('joint1', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-400 mt-1">베이스 회전 (-180° ~ 180°)</div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4">
              <label className="text-sm font-semibold text-white mb-2 block">
                Joint 2 (Shoulder): {(robotArm.joint2 * 180 / Math.PI).toFixed(1)}°
              </label>
              <input
                type="range"
                min={-Math.PI}
                max={Math.PI}
                step={0.01}
                value={robotArm.joint2}
                onChange={(e) => handleJointChange('joint2', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-400 mt-1">어깨 관절 (-180° ~ 180°)</div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4">
              <label className="text-sm font-semibold text-white mb-2 block">
                Joint 3 (Elbow): {(robotArm.joint3 * 180 / Math.PI).toFixed(1)}°
              </label>
              <input
                type="range"
                min={-Math.PI}
                max={Math.PI}
                step={0.01}
                value={robotArm.joint3}
                onChange={(e) => handleJointChange('joint3', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-400 mt-1">팔꿈치 관절 (-180° ~ 180°)</div>
            </div>
          </>
        )}

        {controlMode === 'inverse' && (
          <div className="col-span-3 bg-gray-800 rounded-lg p-4">
            <p className="text-sm text-white mb-2">
              <strong className="text-green-400">역기구학 모드:</strong> 캔버스를 클릭하여 목표 위치 설정
            </p>
            <p className="text-xs text-gray-400">
              로봇 팔이 자동으로 목표 지점에 도달하도록 관절 각도가 계산됩니다.
              초록색 원 안쪽을 클릭하세요.
            </p>
          </div>
        )}
      </div>

      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        onClick={handleCanvasClick}
        className="w-full bg-gray-950 rounded-lg cursor-crosshair mb-4"
      />

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="font-semibold text-white mb-2">💡 로봇 운동학</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <h4 className="font-semibold text-blue-400 mb-1">순기구학 (Forward Kinematics)</h4>
            <ul className="space-y-1 text-xs">
              <li>• 입력: 관절 각도 (θ1, θ2, θ3)</li>
              <li>• 출력: 끝점 위치 (x, y)</li>
              <li>• 방법: 삼각함수로 직접 계산</li>
              <li>• 용도: 모션 계획, 시뮬레이션</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-green-400 mb-1">역기구학 (Inverse Kinematics)</h4>
            <ul className="space-y-1 text-xs">
              <li>• 입력: 목표 위치 (x, y)</li>
              <li>• 출력: 관절 각도 (θ1, θ2, θ3)</li>
              <li>• 방법: Analytical/Numerical 솔루션</li>
              <li>• 용도: 로봇 제어, 경로 추적</li>
            </ul>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-gray-700">
          <p className="text-xs text-gray-400">
            <strong>빨간색 링크</strong>: Link 1 (L={L1}px) |
            <strong className="text-orange-400"> 주황색 링크</strong>: Link 2 (L={L2}px) |
            <strong className="text-green-400"> 초록색 링크</strong>: Link 3 (L={L3}px)
          </p>
          <p className="text-xs text-gray-400 mt-1">
            도달 범위: {Math.abs(L1 - L2 - L3)}px ~ {L1 + L2 + L3}px (점선 원으로 표시)
          </p>
        </div>
      </div>
    </div>
  );
}