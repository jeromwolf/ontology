'use client';

import PrivacyPreservingML from '../PrivacyPreservingML';

export default function Chapter4() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h1>프라이버시 보호 ML</h1>
      
      <section className="mb-8">
        <h2>1. 차분 프라이버시</h2>
        <p>
          차분 프라이버시(Differential Privacy)는 개별 데이터 포인트의 
          프라이버시를 보호하면서 통계적 분석을 가능하게 하는 수학적 프레임워크입니다.
        </p>
        
        <h3>핵심 개념</h3>
        <ul>
          <li><strong>ε-차분 프라이버시</strong>: 프라이버시 손실의 상한</li>
          <li><strong>노이즈 추가</strong>: Laplace 또는 Gaussian 노이즈</li>
          <li><strong>프라이버시 예산</strong>: 총 프라이버시 손실 관리</li>
        </ul>
        
        <div className="bg-blue-50 dark:bg-blue-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">DP-SGD</h4>
          <p>
            차분 프라이버시를 적용한 확률적 경사 하강법:
          </p>
          <ul className="space-y-2 mt-2">
            <li>• 그래디언트 클리핑</li>
            <li>• 노이즈 추가</li>
            <li>• 프라이버시 회계</li>
          </ul>
        </div>
      </section>

      <section className="mb-8">
        <h2>2. 연합 학습</h2>
        <p>
          연합 학습(Federated Learning)은 데이터를 중앙 서버로 수집하지 않고
          분산된 환경에서 모델을 학습하는 방법입니다.
        </p>
        
        <h3>작동 원리</h3>
        <ol>
          <li>중앙 서버가 초기 모델 배포</li>
          <li>각 클라이언트가 로컬 데이터로 학습</li>
          <li>모델 업데이트만 서버로 전송</li>
          <li>서버가 업데이트 집계 및 새 모델 배포</li>
        </ol>
        
        <h3>보안 강화 기법</h3>
        <ul>
          <li><strong>Secure Aggregation</strong>: 암호화된 집계</li>
          <li><strong>동형 암호화</strong>: 암호화된 상태에서 연산</li>
          <li><strong>차분 프라이버시</strong>: 업데이트에 노이즈 추가</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>3. 프라이버시 보호 기법</h2>
        
        <h3>PATE (Private Aggregation of Teacher Ensembles)</h3>
        <p>
          여러 교사 모델의 합의를 통해 프라이버시를 보호하면서 학생 모델을 학습합니다.
        </p>
        
        <h3>Split Learning</h3>
        <p>
          모델을 여러 부분으로 나누어 각 당사자가 일부만 보유하고 학습합니다.
        </p>
        
        <h3>Homomorphic Encryption</h3>
        <p>
          암호화된 데이터에서 직접 연산을 수행하여 프라이버시를 완벽하게 보호합니다.
        </p>
      </section>

      <div className="my-8">
        <PrivacyPreservingML />
      </div>
    </div>
  )
}