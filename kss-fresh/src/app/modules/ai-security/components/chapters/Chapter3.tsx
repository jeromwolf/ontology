'use client'

import ModelSecurityAnalyzer from '../ModelSecurityAnalyzer'

export default function Chapter3() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h1>모델 보안</h1>
      
      <section className="mb-8">
        <h2>1. 모델 추출 공격</h2>
        <p>
          모델 추출(Model Extraction)은 공격자가 대상 모델의 기능을 복제하여
          자신만의 대체 모델을 만드는 공격입니다.
        </p>
        
        <h3>공격 방법</h3>
        <ul>
          <li><strong>API 남용</strong>: 대량의 쿼리를 통한 입출력 수집</li>
          <li><strong>모델 역공학</strong>: 수집된 데이터로 모델 구조 추론</li>
          <li><strong>지식 증류</strong>: 수집된 데이터로 새 모델 학습</li>
        </ul>
        
        <h3>위험성</h3>
        <ul>
          <li>지적 재산권 침해</li>
          <li>비즈니스 모델 손상</li>
          <li>추출된 모델을 통한 2차 공격</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>2. 백도어 공격</h2>
        <p>
          백도어 공격은 모델에 숨겨진 악의적 동작을 삽입하는 공격으로,
          특정 트리거가 있을 때만 활성화됩니다.
        </p>
        
        <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">백도어 유형</h4>
          <ul className="space-y-2">
            <li>• <strong>데이터 중독</strong>: 학습 데이터에 트리거 삽입</li>
            <li>• <strong>모델 조작</strong>: 사전 학습된 모델에 백도어 삽입</li>
            <li>• <strong>공급망 공격</strong>: 서드파티 모델/데이터셋 오염</li>
          </ul>
        </div>
        
        <h3>탐지 방법</h3>
        <ul>
          <li>Neural Cleanse: 트리거 역공학</li>
          <li>활성화 패턴 분석</li>
          <li>모델 프루닝을 통한 백도어 제거</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>3. 모델 보호 기법</h2>
        
        <h3>워터마킹</h3>
        <p>
          모델에 고유한 서명을 삽입하여 소유권을 증명하고 불법 복제를 탐지합니다.
        </p>
        
        <h3>모델 암호화</h3>
        <p>
          배포된 모델의 가중치를 암호화하여 역공학을 방지합니다.
        </p>
        
        <h3>API 보안</h3>
        <ul>
          <li>Rate limiting</li>
          <li>쿼리 패턴 모니터링</li>
          <li>출력 난독화</li>
          <li>적응형 응답</li>
        </ul>
      </section>

      <div className="my-8">
        <ModelSecurityAnalyzer />
      </div>
    </div>
  )
}