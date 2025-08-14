'use client'

import ThreatDetectionDashboard from '../ThreatDetectionDashboard'

export default function Chapter6() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h1>보안 테스팅</h1>
      
      <section className="mb-8">
        <h2>1. 보안 평가 프레임워크</h2>
        <p>
          AI 시스템의 보안을 체계적으로 평가하기 위한 프레임워크와 도구들을 활용합니다.
        </p>
        
        <h3>평가 영역</h3>
        <ul>
          <li><strong>견고성 평가</strong>: 적대적 예제에 대한 저항성</li>
          <li><strong>프라이버시 평가</strong>: 멤버십 추론, 모델 역전 공격</li>
          <li><strong>공정성 평가</strong>: 편향과 차별 검사</li>
          <li><strong>설명가능성</strong>: 모델 결정의 해석 가능성</li>
        </ul>
        
        <div className="bg-yellow-50 dark:bg-yellow-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">테스팅 도구</h4>
          <ul className="space-y-2">
            <li>• <strong>CleverHans</strong>: 적대적 예제 생성 및 평가</li>
            <li>• <strong>Foolbox</strong>: 다양한 공격 기법 구현</li>
            <li>• <strong>ART (Adversarial Robustness Toolbox)</strong>: IBM의 종합 도구</li>
            <li>• <strong>TextAttack</strong>: NLP 모델 공격 프레임워크</li>
          </ul>
        </div>
      </section>

      <section className="mb-8">
        <h2>2. 테스트 시나리오</h2>
        
        <h3>White-box 테스팅</h3>
        <p>모델의 내부 구조를 완전히 알고 있는 상황에서의 테스트</p>
        <ul>
          <li>그래디언트 기반 공격 테스트</li>
          <li>모델 파라미터 분석</li>
          <li>활성화 패턴 검사</li>
        </ul>
        
        <h3>Black-box 테스팅</h3>
        <p>API 접근만 가능한 상황에서의 테스트</p>
        <ul>
          <li>쿼리 효율성 평가</li>
          <li>전이 공격 테스트</li>
          <li>API 남용 시뮬레이션</li>
        </ul>
        
        <h3>Gray-box 테스팅</h3>
        <p>부분적인 정보만 알고 있는 상황</p>
        <ul>
          <li>모델 구조 추론</li>
          <li>하이브리드 공격 테스트</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>3. 자동화된 보안 감사</h2>
        <p>
          CI/CD 파이프라인에 통합 가능한 자동화된 보안 테스트를 구현합니다.
        </p>
        
        <h3>테스트 자동화 단계</h3>
        <ol>
          <li>
            <strong>데이터 검증</strong>
            <ul>
              <li>입력 데이터 무결성 검사</li>
              <li>데이터 중독 탐지</li>
              <li>이상치 검출</li>
            </ul>
          </li>
          <li>
            <strong>모델 검증</strong>
            <ul>
              <li>백도어 스캔</li>
              <li>견고성 벤치마크</li>
              <li>성능 저하 모니터링</li>
            </ul>
          </li>
          <li>
            <strong>런타임 모니터링</strong>
            <ul>
              <li>실시간 이상 탐지</li>
              <li>공격 패턴 인식</li>
              <li>자동 대응 시스템</li>
            </ul>
          </li>
        </ol>
      </section>

      <div className="my-8">
        <ThreatDetectionDashboard />
      </div>
    </div>
  )
}