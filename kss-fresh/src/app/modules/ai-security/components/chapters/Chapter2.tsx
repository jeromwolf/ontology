'use client'

import AdversarialAttackVisualizer from '../AdversarialAttackVisualizer'

export default function Chapter2() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h1>적대적 공격</h1>
      
      <section className="mb-8">
        <h2>1. 적대적 예제란?</h2>
        <p>
          적대적 예제(Adversarial Examples)는 의도적으로 설계된 입력으로,
          인간에게는 정상적으로 보이지만 AI 모델을 속여 잘못된 예측을 하도록 만듭니다.
        </p>
        
        <h3>적대적 공격의 특징</h3>
        <ul>
          <li><strong>미세한 변화</strong>: 육안으로 구분하기 어려운 작은 노이즈 추가</li>
          <li><strong>전이성(Transferability)</strong>: 한 모델에서 생성된 적대적 예제가 다른 모델에도 효과적</li>
          <li><strong>견고성</strong>: 압축, 크기 조정 등의 변환에도 공격 효과 유지</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>2. 공격 기법 분류</h2>
        
        <h3>White-box 공격</h3>
        <p>모델의 구조와 가중치를 완전히 알고 있는 상황에서의 공격</p>
        <ul>
          <li><strong>FGSM (Fast Gradient Sign Method)</strong>: 가장 간단하고 빠른 공격</li>
          <li><strong>PGD (Projected Gradient Descent)</strong>: 반복적인 최적화를 통한 강력한 공격</li>
          <li><strong>C&W (Carlini & Wagner)</strong>: 최소한의 왜곡으로 강력한 공격</li>
        </ul>
        
        <h3>Black-box 공격</h3>
        <p>모델의 내부 구조를 모르고 API 접근만 가능한 상황</p>
        <ul>
          <li><strong>전이 기반 공격</strong>: 대체 모델에서 생성한 적대적 예제 사용</li>
          <li><strong>쿼리 기반 공격</strong>: 반복적인 쿼리를 통한 공격 최적화</li>
          <li><strong>결정 경계 공격</strong>: 분류 경계 추정을 통한 공격</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>3. 실제 공격 시나리오</h2>
        
        <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">자율주행차 공격</h4>
          <p>
            정지 신호에 작은 스티커를 붙여 속도 제한 표지판으로 오인식하게 만드는 공격
          </p>
        </div>
        
        <div className="bg-gray-50 dark:bg-gray-800/50 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">얼굴 인식 회피</h4>
          <p>
            특수하게 설계된 안경이나 패치를 착용하여 얼굴 인식 시스템을 회피
          </p>
        </div>
      </section>

      <div className="my-8">
        <AdversarialAttackVisualizer />
      </div>
    </div>
  )
}