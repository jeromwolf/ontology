'use client';

import ThreatDetectionDashboard from '../ThreatDetectionDashboard';

export default function Chapter1() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h1>AI 보안 기초</h1>
      
      <section className="mb-8">
        <h2>1. AI 보안의 중요성</h2>
        <p>
          인공지능 시스템이 우리 생활의 모든 영역에 통합되면서, AI 보안은 점점 더 중요해지고 있습니다.
          의료, 금융, 자율주행, 보안 시스템 등 중요한 의사결정에 AI가 사용되면서,
          이러한 시스템의 보안 취약점은 심각한 결과를 초래할 수 있습니다.
        </p>
        
        <h3>주요 보안 위협</h3>
        <ul>
          <li><strong>적대적 공격(Adversarial Attacks)</strong>: 모델을 속이기 위한 입력 조작</li>
          <li><strong>모델 추출(Model Extraction)</strong>: API를 통한 모델 복제</li>
          <li><strong>데이터 중독(Data Poisoning)</strong>: 학습 데이터 오염</li>
          <li><strong>프라이버시 침해</strong>: 학습 데이터 정보 유출</li>
          <li><strong>백도어 공격</strong>: 숨겨진 악의적 동작 삽입</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>2. AI 시스템의 공격 표면</h2>
        <p>
          AI 시스템은 전통적인 소프트웨어와는 다른 독특한 공격 표면을 가지고 있습니다.
        </p>
        
        <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">학습 단계 공격</h4>
          <ul className="space-y-2">
            <li>• 데이터 수집 과정에서의 오염</li>
            <li>• 라벨링 과정 조작</li>
            <li>• 학습 알고리즘 취약점 악용</li>
            <li>• 하이퍼파라미터 조작</li>
          </ul>
        </div>
        
        <div className="bg-gray-50 dark:bg-gray-800/50 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">추론 단계 공격</h4>
          <ul className="space-y-2">
            <li>• 적대적 예제 입력</li>
            <li>• 모델 역공학</li>
            <li>• 사이드 채널 공격</li>
            <li>• API 남용</li>
          </ul>
        </div>
      </section>

      <section className="mb-8">
        <h2>3. 보안 원칙과 베스트 프랙티스</h2>
        <p>
          AI 시스템 보안을 위한 기본 원칙들을 이해하고 적용해야 합니다.
        </p>
        
        <h3>Defense in Depth</h3>
        <p>
          여러 계층의 보안 메커니즘을 구현하여 단일 실패 지점을 방지합니다.
        </p>
        
        <h3>Security by Design</h3>
        <p>
          개발 초기 단계부터 보안을 고려하여 설계합니다.
        </p>
        
        <h3>지속적인 모니터링</h3>
        <p>
          배포된 모델의 동작을 지속적으로 모니터링하고 이상 징후를 탐지합니다.
        </p>
      </section>

      <div className="my-8">
        <ThreatDetectionDashboard />
      </div>
    </div>
  )
}