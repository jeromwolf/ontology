'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Lock, Users, Cloud, Shield, Database, Activity } from 'lucide-react';

interface PrivacyMethod {
  id: string;
  name: string;
  description: string;
  privacyLevel: number;
  utilityLevel: number;
}

const privacyMethods: PrivacyMethod[] = [
  {
    id: 'differential',
    name: '차분 프라이버시',
    description: '노이즈 추가를 통한 개인정보 보호',
    privacyLevel: 0.9,
    utilityLevel: 0.7
  },
  {
    id: 'federated',
    name: '연합 학습',
    description: '데이터를 중앙에 모으지 않고 분산 학습',
    privacyLevel: 0.85,
    utilityLevel: 0.8
  },
  {
    id: 'homomorphic',
    name: '동형 암호화',
    description: '암호화된 상태에서 연산 수행',
    privacyLevel: 1.0,
    utilityLevel: 0.5
  },
  {
    id: 'secure-multiparty',
    name: '안전한 다자간 계산',
    description: '여러 당사자가 데이터를 공개하지 않고 계산',
    privacyLevel: 0.95,
    utilityLevel: 0.6
  }
];

export default function PrivacyPreservingML() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedMethod, setSelectedMethod] = useState<PrivacyMethod>(privacyMethods[0]);
  const [epsilon, setEpsilon] = useState(1.0);
  const [noiseLevel, setNoiseLevel] = useState(0.1);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [privacyBudget, setPrivacyBudget] = useState(10);
  const [usedBudget, setUsedBudget] = useState(0);

  // 시각화 그리기
  const drawVisualization = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    switch (selectedMethod.id) {
      case 'differential':
        drawDifferentialPrivacy(ctx, canvas.width, canvas.height);
        break;
      case 'federated':
        drawFederatedLearning(ctx, canvas.width, canvas.height);
        break;
      case 'homomorphic':
        drawHomomorphicEncryption(ctx, canvas.width, canvas.height);
        break;
      case 'secure-multiparty':
        drawSecureMultiparty(ctx, canvas.width, canvas.height);
        break;
    }
  }, [selectedMethod]);

  // 차분 프라이버시 시각화
  const drawDifferentialPrivacy = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // 원본 데이터
    ctx.fillStyle = '#3B82F6';
    ctx.fillRect(50, 50, 100, 150);
    ctx.fillStyle = '#fff';
    ctx.font = '14px sans-serif';
    ctx.fillText('원본 데이터', 70, 120);

    // 화살표
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(170, 125);
    ctx.lineTo(230, 125);
    ctx.stroke();

    // 노이즈 추가
    ctx.fillStyle = '#EF4444';
    const noiseIntensity = noiseLevel * 50;
    for (let i = 0; i < 20; i++) {
      const x = 250 + Math.random() * 80;
      const y = 50 + Math.random() * 150;
      const size = Math.random() * noiseIntensity;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();
    }

    // 노이즈가 추가된 데이터
    ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
    ctx.fillRect(250, 50, 100, 150);
    ctx.fillStyle = '#fff';
    ctx.fillText('노이즈 추가', 265, 120);

    // 프라이버시 레벨
    ctx.fillStyle = '#10B981';
    ctx.fillRect(50, 250, (width - 100) * (1 - epsilon / 10), 20);
    ctx.fillStyle = '#000';
    ctx.font = '12px sans-serif';
    ctx.fillText(`Privacy Level: ε = ${epsilon.toFixed(2)}`, 50, 290);
  };

  // 연합 학습 시각화
  const drawFederatedLearning = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // 중앙 서버
    ctx.fillStyle = '#6366F1';
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, 40, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '12px sans-serif';
    ctx.fillText('서버', width / 2 - 15, height / 2 + 5);

    // 클라이언트들
    const clients = 5;
    for (let i = 0; i < clients; i++) {
      const angle = (i / clients) * Math.PI * 2;
      const x = width / 2 + Math.cos(angle) * 120;
      const y = height / 2 + Math.sin(angle) * 120;
      
      ctx.fillStyle = '#10B981';
      ctx.beginPath();
      ctx.arc(x, y, 25, 0, Math.PI * 2);
      ctx.fill();
      
      // 연결선
      ctx.strokeStyle = '#666';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(width / 2, height / 2);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 학습 진행 상태
    if (isTraining && epoch > 0) {
      ctx.fillStyle = 'rgba(34, 197, 94, 0.3)';
      ctx.beginPath();
      ctx.arc(width / 2, height / 2, 40 + epoch * 5, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  // 동형 암호화 시각화
  const drawHomomorphicEncryption = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // 평문
    ctx.fillStyle = '#3B82F6';
    ctx.fillRect(30, 80, 80, 60);
    ctx.fillStyle = '#fff';
    ctx.font = '12px sans-serif';
    ctx.fillText('평문', 55, 115);

    // 암호화
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(120, 110);
    ctx.lineTo(150, 110);
    ctx.stroke();

    // 암호문
    ctx.fillStyle = '#000';
    ctx.font = '10px monospace';
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 8; j++) {
        ctx.fillText(Math.random() > 0.5 ? '1' : '0', 160 + j * 10, 90 + i * 15);
      }
    }

    // 연산
    ctx.fillStyle = '#EF4444';
    ctx.fillRect(160, 160, 80, 40);
    ctx.fillStyle = '#fff';
    ctx.fillText('암호화된', 175, 180);
    ctx.fillText('연산', 185, 195);

    // 결과
    ctx.strokeStyle = '#666';
    ctx.beginPath();
    ctx.moveTo(250, 180);
    ctx.lineTo(280, 180);
    ctx.stroke();

    // 암호화된 결과
    ctx.fillStyle = '#000';
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 6; j++) {
        ctx.fillText(Math.random() > 0.5 ? '1' : '0', 290 + j * 10, 170 + i * 15);
      }
    }
  };

  // 안전한 다자간 계산 시각화
  const drawSecureMultiparty = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const parties = [
      { x: 100, y: 80, name: 'Party A' },
      { x: 300, y: 80, name: 'Party B' },
      { x: 200, y: 220, name: 'Party C' }
    ];

    // 각 당사자
    parties.forEach((party, i) => {
      ctx.fillStyle = ['#3B82F6', '#10B981', '#EF4444'][i];
      ctx.beginPath();
      ctx.arc(party.x, party.y, 35, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = '12px sans-serif';
      ctx.fillText(party.name, party.x - 25, party.y + 5);
    });

    // 비밀 공유 연결
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    parties.forEach((party1, i) => {
      parties.forEach((party2, j) => {
        if (i < j) {
          ctx.beginPath();
          ctx.moveTo(party1.x, party1.y);
          ctx.lineTo(party2.x, party2.y);
          ctx.stroke();
        }
      });
    });
    ctx.setLineDash([]);

    // 중앙 계산 결과
    ctx.fillStyle = '#6366F1';
    ctx.beginPath();
    ctx.arc(200, 140, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText('f(x,y,z)', 180, 145);
  };

  // 학습 시뮬레이션
  const startTraining = useCallback(() => {
    setIsTraining(true);
    setEpoch(0);
    setAccuracy(0);
    setUsedBudget(0);

    const interval = setInterval(() => {
      setEpoch(prev => {
        if (prev >= 10) {
          setIsTraining(false);
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      });

      setAccuracy(prev => Math.min(0.95, prev + 0.08 + Math.random() * 0.04));
      setUsedBudget(prev => Math.min(privacyBudget, prev + epsilon));
    }, 1000);
  }, [epsilon, privacyBudget]);

  useEffect(() => {
    drawVisualization();
  }, [drawVisualization]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
      <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
        프라이버시 보호 ML 시뮬레이터
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold mb-3">시각화</h4>
          <canvas
            ref={canvasRef}
            width={400}
            height={300}
            className="border border-gray-300 dark:border-gray-600 rounded-lg w-full"
          />

          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg">
              <div className="flex items-center mb-2">
                <Shield className="w-5 h-5 mr-2 text-blue-600 dark:text-blue-400" />
                <span className="font-semibold">프라이버시</span>
              </div>
              <div className="relative h-3 bg-gray-200 dark:bg-gray-700 rounded-full">
                <div
                  className="absolute h-full bg-blue-600 rounded-full"
                  style={{ width: `${selectedMethod.privacyLevel * 100}%` }}
                />
              </div>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {(selectedMethod.privacyLevel * 100).toFixed(0)}%
              </span>
            </div>

            <div className="bg-green-50 dark:bg-green-900/30 p-3 rounded-lg">
              <div className="flex items-center mb-2">
                <Activity className="w-5 h-5 mr-2 text-green-600 dark:text-green-400" />
                <span className="font-semibold">유용성</span>
              </div>
              <div className="relative h-3 bg-gray-200 dark:bg-gray-700 rounded-full">
                <div
                  className="absolute h-full bg-green-600 rounded-full"
                  style={{ width: `${selectedMethod.utilityLevel * 100}%` }}
                />
              </div>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {(selectedMethod.utilityLevel * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-lg font-semibold mb-3">설정</h4>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">프라이버시 보호 방법</label>
              <select
                value={selectedMethod.id}
                onChange={(e) => {
                  const method = privacyMethods.find(m => m.id === e.target.value);
                  if (method) setSelectedMethod(method);
                }}
                className="w-full px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
              >
                {privacyMethods.map(method => (
                  <option key={method.id} value={method.id}>
                    {method.name}
                  </option>
                ))}
              </select>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {selectedMethod.description}
              </p>
            </div>

            {selectedMethod.id === 'differential' && (
              <>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    프라이버시 파라미터 (ε): {epsilon.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="10"
                    step="0.1"
                    value={epsilon}
                    onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <p className="text-xs text-gray-500">작을수록 강한 프라이버시</p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    노이즈 레벨: {(noiseLevel * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={noiseLevel}
                    onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </>
            )}

            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h5 className="font-semibold mb-2">학습 상태</h5>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Epoch:</span>
                  <span>{epoch} / 10</span>
                </div>
                <div className="flex justify-between">
                  <span>정확도:</span>
                  <span>{(accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>프라이버시 예산:</span>
                  <span>{usedBudget.toFixed(2)} / {privacyBudget}</span>
                </div>
              </div>
            </div>

            <button
              onClick={startTraining}
              disabled={isTraining}
              className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isTraining ? '학습 중...' : '학습 시작'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}