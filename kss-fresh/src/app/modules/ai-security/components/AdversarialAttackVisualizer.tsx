'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { AlertTriangle, Shield, Zap, RefreshCw } from 'lucide-react';

interface AttackMethod {
  name: string;
  epsilon: number;
  description: string;
}

const attackMethods: AttackMethod[] = [
  { name: 'FGSM', epsilon: 0.1, description: 'Fast Gradient Sign Method - 빠르고 간단한 공격' },
  { name: 'PGD', epsilon: 0.05, description: 'Projected Gradient Descent - 반복적 최적화 공격' },
  { name: 'C&W', epsilon: 0.03, description: 'Carlini & Wagner - 최소 왜곡 공격' },
  { name: 'DeepFool', epsilon: 0.02, description: 'DeepFool - 최소 perturbation 찾기' }
];

export default function AdversarialAttackVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedAttack, setSelectedAttack] = useState<AttackMethod>(attackMethods[0]);
  const [isAttacking, setIsAttacking] = useState(false);
  const [originalClass, setOriginalClass] = useState('고양이');
  const [adversarialClass, setAdversarialClass] = useState('');
  const [confidence, setConfidence] = useState(0.95);
  const [adversarialConfidence, setAdversarialConfidence] = useState(0);
  const [perturbationStrength, setPerturbationStrength] = useState(0.1);

  // 이미지 그리기
  const drawImage = useCallback((addNoise: boolean = false) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 캔버스 초기화
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 원본 이미지 시뮬레이션 (고양이)
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 고양이 그리기
    ctx.fillStyle = '#666';
    // 몸통
    ctx.beginPath();
    ctx.ellipse(150, 180, 60, 40, 0, 0, Math.PI * 2);
    ctx.fill();

    // 머리
    ctx.beginPath();
    ctx.arc(150, 120, 35, 0, Math.PI * 2);
    ctx.fill();

    // 귀
    ctx.beginPath();
    ctx.moveTo(120, 110);
    ctx.lineTo(110, 85);
    ctx.lineTo(130, 95);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(180, 110);
    ctx.lineTo(190, 85);
    ctx.lineTo(170, 95);
    ctx.fill();

    // 꼬리
    ctx.beginPath();
    ctx.moveTo(200, 180);
    ctx.quadraticCurveTo(240, 160, 250, 140);
    ctx.lineWidth = 15;
    ctx.strokeStyle = '#666';
    ctx.stroke();

    // 적대적 노이즈 추가
    if (addNoise && isAttacking) {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      for (let i = 0; i < data.length; i += 4) {
        // RGB 채널에 노이즈 추가
        const noise = (Math.random() - 0.5) * perturbationStrength * 255;
        data[i] = Math.max(0, Math.min(255, data[i] + noise));     // R
        data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise)); // G
        data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise)); // B
      }

      ctx.putImageData(imageData, 0, 0);

      // 시각적 효과를 위한 오버레이
      ctx.fillStyle = `rgba(255, 0, 0, ${perturbationStrength * 0.2})`;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, [isAttacking, perturbationStrength]);

  // 공격 시뮬레이션
  const simulateAttack = useCallback(() => {
    setIsAttacking(true);
    setPerturbationStrength(selectedAttack.epsilon);

    // 공격 결과 시뮬레이션
    setTimeout(() => {
      const adversarialClasses = ['개', '토스터', '자동차', '비행기'];
      const randomClass = adversarialClasses[Math.floor(Math.random() * adversarialClasses.length)];
      setAdversarialClass(randomClass);
      setAdversarialConfidence(0.8 + Math.random() * 0.15);
    }, 500);
  }, [selectedAttack]);

  // 초기화
  const reset = useCallback(() => {
    setIsAttacking(false);
    setAdversarialClass('');
    setAdversarialConfidence(0);
    setPerturbationStrength(0.1);
  }, []);

  useEffect(() => {
    drawImage(isAttacking);
  }, [drawImage, isAttacking]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
      <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
        적대적 공격 시각화
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold mb-3">입력 이미지</h4>
          <div className="relative">
            <canvas
              ref={canvasRef}
              width={300}
              height={300}
              className="border border-gray-300 dark:border-gray-600 rounded-lg"
            />
            {isAttacking && (
              <div className="absolute top-2 right-2">
                <AlertTriangle className="w-6 h-6 text-red-500 animate-pulse" />
              </div>
            )}
          </div>

          <div className="mt-4 space-y-3">
            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">원본 예측:</span>
                <span className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                  {originalClass} ({(confidence * 100).toFixed(1)}%)
                </span>
              </div>
            </div>

            {isAttacking && adversarialClass && (
              <div className="bg-red-50 dark:bg-red-900/30 p-3 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">적대적 예측:</span>
                  <span className="text-lg font-semibold text-red-600 dark:text-red-400">
                    {adversarialClass} ({(adversarialConfidence * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        <div>
          <h4 className="text-lg font-semibold mb-3">공격 설정</h4>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">공격 방법</label>
              <select
                value={selectedAttack.name}
                onChange={(e) => {
                  const attack = attackMethods.find(a => a.name === e.target.value);
                  if (attack) setSelectedAttack(attack);
                }}
                className="w-full px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
              >
                {attackMethods.map(attack => (
                  <option key={attack.name} value={attack.name}>
                    {attack.name}
                  </option>
                ))}
              </select>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {selectedAttack.description}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Perturbation 강도 (ε): {selectedAttack.epsilon}
              </label>
              <input
                type="range"
                min="0.01"
                max="0.3"
                step="0.01"
                value={selectedAttack.epsilon}
                onChange={(e) => {
                  setSelectedAttack({
                    ...selectedAttack,
                    epsilon: parseFloat(e.target.value)
                  });
                }}
                className="w-full"
              />
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/30 p-4 rounded-lg">
              <h5 className="font-semibold mb-2 flex items-center">
                <Shield className="w-4 h-4 mr-2" />
                공격 원리
              </h5>
              <p className="text-sm">
                적대적 공격은 입력에 미세한 노이즈를 추가하여 모델의 예측을 변경합니다.
                이 노이즈는 인간의 눈으로는 거의 감지할 수 없지만, 모델에는 큰 영향을 미칩니다.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={simulateAttack}
                disabled={isAttacking}
                className="flex-1 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                <Zap className="w-4 h-4 mr-2" />
                공격 실행
              </button>
              
              <button
                onClick={reset}
                className="flex-1 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 flex items-center justify-center"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                초기화
              </button>
            </div>
          </div>

          {isAttacking && (
            <div className="mt-4 bg-red-50 dark:bg-red-900/30 p-4 rounded-lg">
              <h5 className="font-semibold mb-2">공격 결과</h5>
              <ul className="text-sm space-y-1">
                <li>• Perturbation 적용됨: ε = {selectedAttack.epsilon}</li>
                <li>• 원본 클래스: {originalClass} → {adversarialClass}</li>
                <li>• 신뢰도 변화: {(confidence * 100).toFixed(1)}% → {(adversarialConfidence * 100).toFixed(1)}%</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}