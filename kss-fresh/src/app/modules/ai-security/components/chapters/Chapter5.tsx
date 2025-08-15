'use client';

import SecurityAuditTool from '../SecurityAuditTool';

export default function Chapter5() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h1>견고성과 방어</h1>
      
      <section className="mb-8">
        <h2>1. 적대적 학습</h2>
        <p>
          적대적 학습(Adversarial Training)은 학습 과정에서 적대적 예제를
          포함시켜 모델의 견고성을 향상시키는 방법입니다.
        </p>
        
        <h3>기본 원리</h3>
        <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto">
{`min_θ E[(x,y)~D] max_δ∈S L(fθ(x+δ), y)

여기서:
- θ: 모델 파라미터
- δ: 적대적 perturbation
- S: 허용된 perturbation 집합
- L: 손실 함수`}
        </pre>
        
        <h3>학습 전략</h3>
        <ul>
          <li><strong>PGD-AT</strong>: 강력한 적대적 예제로 학습</li>
          <li><strong>TRADES</strong>: 정확도와 견고성의 균형</li>
          <li><strong>MART</strong>: 잘못 분류된 예제에 집중</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>2. 방어 메커니즘</h2>
        
        <div className="bg-green-50 dark:bg-green-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">입력 전처리</h4>
          <ul className="space-y-2">
            <li>• <strong>Feature Squeezing</strong>: 색상 깊이 감소</li>
            <li>• <strong>JPEG 압축</strong>: 고주파 노이즈 제거</li>
            <li>• <strong>Spatial Smoothing</strong>: 공간적 필터링</li>
            <li>• <strong>Pixel Deflection</strong>: 랜덤 픽셀 변환</li>
          </ul>
        </div>
        
        <div className="bg-blue-50 dark:bg-blue-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">모델 기반 방어</h4>
          <ul className="space-y-2">
            <li>• <strong>Defensive Distillation</strong>: 부드러운 예측</li>
            <li>• <strong>Ensemble Methods</strong>: 다중 모델 합의</li>
            <li>• <strong>Random Smoothing</strong>: 확률적 방어</li>
            <li>• <strong>Input Gradient Regularization</strong></li>
          </ul>
        </div>
      </section>

      <section className="mb-8">
        <h2>3. 인증된 방어</h2>
        <p>
          인증된 방어(Certified Defense)는 특정 크기의 perturbation에 대해
          수학적으로 보장된 견고성을 제공합니다.
        </p>
        
        <h3>주요 기법</h3>
        <ul>
          <li>
            <strong>Randomized Smoothing</strong>
            <p className="text-gray-600 dark:text-gray-400">
              노이즈를 추가한 입력의 평균 예측으로 견고성 인증
            </p>
          </li>
          <li>
            <strong>Interval Bound Propagation</strong>
            <p className="text-gray-600 dark:text-gray-400">
              각 레이어의 출력 범위를 추적하여 견고성 검증
            </p>
          </li>
          <li>
            <strong>Convex Relaxation</strong>
            <p className="text-gray-600 dark:text-gray-400">
              비선형 활성화 함수의 convex 근사
            </p>
          </li>
        </ul>
      </section>

      <div className="my-8">
        <SecurityAuditTool />
      </div>
    </div>
  )
}