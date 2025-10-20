'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Settings } from 'lucide-react';

interface SensorData {
  gps: { x: number; y: number; accuracy: number };
  imu: { x: number; y: number; heading: number };
  fusion: { x: number; y: number; heading: number };
  groundTruth: { x: number; y: number };
}

export default function SensorFusionSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [time, setTime] = useState(0);
  const [gpsNoise, setGpsNoise] = useState(5.0);
  const [imuNoise, setImuNoise] = useState(0.1);
  const [sensorData, setSensorData] = useState<SensorData[]>([]);

  // 칼만 필터 상태
  const kalmanState = useRef({
    x: [0, 0, 0, 0, 0, 0], // [x, y, θ, vx, vy, ω]
    P: Array(6).fill(0).map(() => Array(6).fill(0).map((_, i, arr) => i === arr.indexOf(i) ? 1 : 0)),
    Q: Array(6).fill(0).map(() => Array(6).fill(0).map((_, i, arr) => i === arr.indexOf(i) ? 0.01 : 0)),
    R: [gpsNoise, gpsNoise, imuNoise]
  });

  // 실제 경로 생성 (원형 궤적)
  const generateGroundTruth = (t: number) => {
    const radius = 150;
    const centerX = 400;
    const centerY = 300;
    const angularVel = 0.02;

    return {
      x: centerX + radius * Math.cos(angularVel * t),
      y: centerY + radius * Math.sin(angularVel * t),
      heading: angularVel * t + Math.PI / 2
    };
  };

  // GPS 센서 시뮬레이션 (노이즈 추가)
  const simulateGPS = (truth: { x: number; y: number }) => {
    return {
      x: truth.x + (Math.random() - 0.5) * gpsNoise * 2,
      y: truth.y + (Math.random() - 0.5) * gpsNoise * 2,
      accuracy: gpsNoise
    };
  };

  // IMU 센서 시뮬레이션 (드리프트 추가)
  const simulateIMU = (truth: { x: number; y: number; heading: number }, prevIMU: any) => {
    const drift = prevIMU ? 0.01 : 0;
    return {
      x: truth.x + (Math.random() - 0.5) * imuNoise * 2 + drift,
      y: truth.y + (Math.random() - 0.5) * imuNoise * 2 + drift,
      heading: truth.heading + (Math.random() - 0.5) * 0.05
    };
  };

  // 확장 칼만 필터 - 예측 단계
  const kalmanPredict = (dt: number) => {
    const state = kalmanState.current;
    const [x, y, theta, vx, vy, omega] = state.x;

    // 상태 전이 (비선형)
    const x_new = x + vx * Math.cos(theta) * dt - vy * Math.sin(theta) * dt;
    const y_new = y + vx * Math.sin(theta) * dt + vy * Math.cos(theta) * dt;
    const theta_new = theta + omega * dt;

    state.x = [x_new, y_new, theta_new, vx, vy, omega];

    // 야코비안 행렬 F
    const F = [
      [1, 0, -vx * Math.sin(theta) * dt - vy * Math.cos(theta) * dt, Math.cos(theta) * dt, -Math.sin(theta) * dt, 0],
      [0, 1, vx * Math.cos(theta) * dt - vy * Math.sin(theta) * dt, Math.sin(theta) * dt, Math.cos(theta) * dt, 0],
      [0, 0, 1, 0, 0, dt],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1]
    ];

    // 공분산 업데이트: P = F * P * F^T + Q
    const P_new = matrixMultiply(matrixMultiply(F, state.P), transpose(F));
    state.P = matrixAdd(P_new, state.Q);
  };

  // 확장 칼만 필터 - 업데이트 단계
  const kalmanUpdate = (z: number[]) => {
    const state = kalmanState.current;

    // 측정 행렬 H (x, y, θ 측정)
    const H = [
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0]
    ];

    // 혁신 (Innovation)
    const h = [state.x[0], state.x[1], state.x[2]];
    const y = z.map((zi, i) => zi - h[i]);

    // 칼만 게인 계산
    const S = matrixAdd(
      matrixMultiply(matrixMultiply(H, state.P), transpose(H)),
      [[state.R[0], 0, 0], [0, state.R[1], 0], [0, 0, state.R[2]]]
    );

    const K = matrixMultiply(
      matrixMultiply(state.P, transpose(H)),
      matrixInverse(S)
    );

    // 상태 업데이트
    const K_y = matrixMultiply(K, y.map(v => [v]));
    state.x = state.x.map((xi, i) => xi + K_y[i][0]);

    // 공분산 업데이트
    const I_KH = matrixSubtract(
      identityMatrix(6),
      matrixMultiply(K, H)
    );
    state.P = matrixMultiply(I_KH, state.P);
  };

  // 행렬 연산 헬퍼 함수들
  const matrixMultiply = (A: number[][], B: number[][]) => {
    const result: number[][] = [];
    for (let i = 0; i < A.length; i++) {
      result[i] = [];
      for (let j = 0; j < B[0].length; j++) {
        result[i][j] = 0;
        for (let k = 0; k < A[0].length; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return result;
  };

  const transpose = (A: number[][]) => {
    return A[0].map((_, i) => A.map(row => row[i]));
  };

  const matrixAdd = (A: number[][], B: number[][]) => {
    return A.map((row, i) => row.map((val, j) => val + B[i][j]));
  };

  const matrixSubtract = (A: number[][], B: number[][]) => {
    return A.map((row, i) => row.map((val, j) => val - B[i][j]));
  };

  const identityMatrix = (n: number) => {
    return Array(n).fill(0).map((_, i) => Array(n).fill(0).map((_, j) => i === j ? 1 : 0));
  };

  const matrixInverse = (A: number[][]) => {
    // 3x3 행렬 역행렬 (간소화)
    const det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (Math.abs(det) < 1e-10) return identityMatrix(3);

    const inv = [
      [
        (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det,
        (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det,
        (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det
      ],
      [
        (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det,
        (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det,
        (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det
      ],
      [
        (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det,
        (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det,
        (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det
      ]
    ];

    return inv;
  };

  // 시뮬레이션 루프
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setTime(t => {
        const newTime = t + 1;

        // Ground Truth
        const truth = generateGroundTruth(newTime);

        // 센서 시뮬레이션
        const gps = simulateGPS(truth);
        const prevIMU = sensorData[sensorData.length - 1]?.imu;
        const imu = simulateIMU(truth, prevIMU);

        // 칼만 필터 실행
        kalmanPredict(0.1);
        kalmanUpdate([gps.x, gps.y, imu.heading]);

        const fusion = {
          x: kalmanState.current.x[0],
          y: kalmanState.current.x[1],
          heading: kalmanState.current.x[2]
        };

        setSensorData(prev => [...prev.slice(-200), {
          gps,
          imu,
          fusion,
          groundTruth: { x: truth.x, y: truth.y }
        }]);

        return newTime;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isRunning, gpsNoise, imuNoise]);

  // Canvas 렌더링
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    ctx.strokeStyle = '#2a2a4e';
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

    // 경로 그리기
    if (sensorData.length > 1) {
      // Ground Truth (녹색)
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 2;
      ctx.beginPath();
      sensorData.forEach((data, i) => {
        if (i === 0) ctx.moveTo(data.groundTruth.x, data.groundTruth.y);
        else ctx.lineTo(data.groundTruth.x, data.groundTruth.y);
      });
      ctx.stroke();

      // GPS (빨강, 점선)
      ctx.strokeStyle = '#ff4444';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      sensorData.forEach((data, i) => {
        if (i === 0) ctx.moveTo(data.gps.x, data.gps.y);
        else ctx.lineTo(data.gps.x, data.gps.y);
      });
      ctx.stroke();

      // IMU (파랑, 점선)
      ctx.strokeStyle = '#4488ff';
      ctx.lineWidth = 1;
      ctx.beginPath();
      sensorData.forEach((data, i) => {
        if (i === 0) ctx.moveTo(data.imu.x, data.imu.y);
        else ctx.lineTo(data.imu.x, data.imu.y);
      });
      ctx.stroke();

      // Kalman Fusion (노랑, 굵음)
      ctx.strokeStyle = '#ffdd00';
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      ctx.beginPath();
      sensorData.forEach((data, i) => {
        if (i === 0) ctx.moveTo(data.fusion.x, data.fusion.y);
        else ctx.lineTo(data.fusion.x, data.fusion.y);
      });
      ctx.stroke();

      // 현재 위치 마커
      const current = sensorData[sensorData.length - 1];

      // Ground Truth
      ctx.fillStyle = '#00ff88';
      ctx.beginPath();
      ctx.arc(current.groundTruth.x, current.groundTruth.y, 8, 0, Math.PI * 2);
      ctx.fill();

      // GPS (불확실성 원)
      ctx.strokeStyle = '#ff444488';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(current.gps.x, current.gps.y, current.gps.accuracy, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#ff4444';
      ctx.beginPath();
      ctx.arc(current.gps.x, current.gps.y, 6, 0, Math.PI * 2);
      ctx.fill();

      // IMU
      ctx.fillStyle = '#4488ff';
      ctx.beginPath();
      ctx.arc(current.imu.x, current.imu.y, 6, 0, Math.PI * 2);
      ctx.fill();

      // Kalman Fusion
      ctx.fillStyle = '#ffdd00';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(current.fusion.x, current.fusion.y, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // 범례
    const legends = [
      { color: '#00ff88', label: 'Ground Truth (실제 경로)' },
      { color: '#ff4444', label: 'GPS (±' + gpsNoise.toFixed(1) + 'm)' },
      { color: '#4488ff', label: 'IMU (±' + imuNoise.toFixed(2) + '°)' },
      { color: '#ffdd00', label: 'Kalman Fusion (융합)' }
    ];

    legends.forEach((legend, i) => {
      ctx.fillStyle = legend.color;
      ctx.fillRect(10, 10 + i * 25, 15, 15);
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px monospace';
      ctx.fillText(legend.label, 30, 22 + i * 25);
    });

  }, [sensorData, gpsNoise, imuNoise]);

  const handleReset = () => {
    setIsRunning(false);
    setTime(0);
    setSensorData([]);
    kalmanState.current = {
      x: [400, 300, 0, 0, 0, 0.02],
      P: Array(6).fill(0).map(() => Array(6).fill(0).map((_, i, arr) => i === arr.indexOf(i) ? 1 : 0)),
      Q: Array(6).fill(0).map(() => Array(6).fill(0).map((_, i, arr) => i === arr.indexOf(i) ? 0.01 : 0)),
      R: [gpsNoise, gpsNoise, imuNoise]
    };
  };

  // 에러 계산
  const calculateError = () => {
    if (sensorData.length === 0) return { gps: 0, imu: 0, fusion: 0 };

    const errors = sensorData.slice(-50).map(data => ({
      gps: Math.sqrt(
        Math.pow(data.gps.x - data.groundTruth.x, 2) +
        Math.pow(data.gps.y - data.groundTruth.y, 2)
      ),
      imu: Math.sqrt(
        Math.pow(data.imu.x - data.groundTruth.x, 2) +
        Math.pow(data.imu.y - data.groundTruth.y, 2)
      ),
      fusion: Math.sqrt(
        Math.pow(data.fusion.x - data.groundTruth.x, 2) +
        Math.pow(data.fusion.y - data.groundTruth.y, 2)
      )
    }));

    return {
      gps: errors.reduce((sum, e) => sum + e.gps, 0) / errors.length,
      imu: errors.reduce((sum, e) => sum + e.imu, 0) / errors.length,
      fusion: errors.reduce((sum, e) => sum + e.fusion, 0) / errors.length
    };
  };

  const errors = calculateError();

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">센서 융합 시뮬레이터</h2>
          <p className="text-gray-400 text-sm">확장 칼만 필터 (EKF)를 활용한 GPS + IMU 센서 융합</p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? '일시정지' : '시작'}
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

      <div className="grid grid-cols-4 gap-4 mb-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-xs text-gray-400 mb-1">Time</div>
          <div className="text-2xl font-bold text-white">{(time * 0.1).toFixed(1)}s</div>
        </div>
        <div className="bg-red-900/30 rounded-lg p-4">
          <div className="text-xs text-red-400 mb-1">GPS Error</div>
          <div className="text-2xl font-bold text-red-400">±{errors.gps.toFixed(2)}m</div>
        </div>
        <div className="bg-blue-900/30 rounded-lg p-4">
          <div className="text-xs text-blue-400 mb-1">IMU Error</div>
          <div className="text-2xl font-bold text-blue-400">±{errors.imu.toFixed(2)}m</div>
        </div>
        <div className="bg-yellow-900/30 rounded-lg p-4">
          <div className="text-xs text-yellow-400 mb-1">Fusion Error</div>
          <div className="text-2xl font-bold text-yellow-400">±{errors.fusion.toFixed(2)}m</div>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="w-full bg-gray-950 rounded-lg mb-4"
      />

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-4 h-4 text-gray-400" />
            <label className="text-sm font-semibold text-white">GPS 노이즈: ±{gpsNoise.toFixed(1)}m</label>
          </div>
          <input
            type="range"
            min="1"
            max="20"
            step="0.5"
            value={gpsNoise}
            onChange={(e) => setGpsNoise(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-gray-400 mt-2">GPS 정확도 조절 (일반적으로 ±5m)</div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="w-4 h-4 text-gray-400" />
            <label className="text-sm font-semibold text-white">IMU 노이즈: ±{imuNoise.toFixed(2)}°</label>
          </div>
          <input
            type="range"
            min="0.01"
            max="1.0"
            step="0.01"
            value={imuNoise}
            onChange={(e) => setImuNoise(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-gray-400 mt-2">IMU 드리프트 조절 (일반적으로 ±0.1°)</div>
        </div>
      </div>

      <div className="mt-4 bg-gray-800 rounded-lg p-4">
        <h3 className="font-semibold text-white mb-2">💡 센서 융합의 원리</h3>
        <ul className="text-sm text-gray-300 space-y-1">
          <li>• <strong className="text-green-400">GPS</strong>: 장기적으로 정확하지만 단기 노이즈가 큼 (±5m)</li>
          <li>• <strong className="text-blue-400">IMU</strong>: 단기적으로 정확하지만 시간이 지나면 드리프트 발생</li>
          <li>• <strong className="text-yellow-400">칼만 필터</strong>: 두 센서의 장점을 결합하여 ±0.5m 이하 정확도 달성</li>
          <li>• 자율주행차, 드론, 스마트폰 등 모든 모바일 로봇이 사용하는 핵심 기술</li>
        </ul>
      </div>
    </div>
  );
}