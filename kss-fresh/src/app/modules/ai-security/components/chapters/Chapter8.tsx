'use client'

import SecurityAuditTool from '../SecurityAuditTool'

export default function Chapter8() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h1>사례 연구</h1>
      
      <section className="mb-8">
        <h2>1. 실제 AI 보안 사고</h2>
        <p>
          실제로 발생한 AI 보안 사고들을 분석하여 교훈을 얻고 방어 전략을 수립합니다.
        </p>
        
        <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
          <h3>사례 1: Microsoft Tay 챗봇 (2016)</h3>
          <p><strong>사고 개요:</strong> Twitter에서 운영된 AI 챗봇이 악의적인 사용자들의 조작으로 인해 부적절한 발언을 학습</p>
          <p><strong>원인:</strong></p>
          <ul>
            <li>• 실시간 학습 시스템의 취약점</li>
            <li>• 입력 필터링 부재</li>
            <li>• 악의적 데이터에 대한 방어 메커니즘 부족</li>
          </ul>
          <p><strong>교훈:</strong></p>
          <ul>
            <li>• 사용자 입력에 대한 엄격한 검증 필요</li>
            <li>• 온라인 학습 시스템의 위험성 인식</li>
            <li>• 콘텐츠 모더레이션 시스템 구축</li>
          </ul>
        </div>
        
        <div className="bg-yellow-50 dark:bg-yellow-950/30 p-6 rounded-lg my-4">
          <h3>사례 2: 자율주행차 적대적 공격 (2018)</h3>
          <p><strong>사고 개요:</strong> 연구자들이 도로 표지판에 스티커를 붙여 Tesla Autopilot을 속이는 데 성공</p>
          <p><strong>공격 방법:</strong></p>
          <ul>
            <li>• 정지 표지판에 작은 스티커 부착</li>
            <li>• 차선에 테이프로 가짜 표시</li>
            <li>• 속도 제한 표지판 조작</li>
          </ul>
          <p><strong>대응 방안:</strong></p>
          <ul>
            <li>• 다중 센서 융합</li>
            <li>• 컨텍스트 기반 검증</li>
            <li>• 적대적 학습 적용</li>
          </ul>
        </div>
      </section>

      <section className="mb-8">
        <h2>2. 산업별 보안 사례</h2>
        
        <h3>금융 AI 보안</h3>
        <div className="bg-blue-50 dark:bg-blue-950/30 p-6 rounded-lg my-4">
          <p><strong>과제:</strong> 사기 탐지 시스템의 보안</p>
          <ul className="mt-2">
            <li>• 공격자가 탐지 회피를 위한 패턴 학습</li>
            <li>• 모델 추출을 통한 우회 방법 개발</li>
            <li>• 고객 데이터 프라이버시 보호</li>
          </ul>
          <p className="mt-4"><strong>해결책:</strong></p>
          <ul>
            <li>• 앙상블 모델 사용</li>
            <li>• 지속적인 모델 업데이트</li>
            <li>• 연합 학습 도입</li>
          </ul>
        </div>
        
        <h3>의료 AI 보안</h3>
        <div className="bg-green-50 dark:bg-green-950/30 p-6 rounded-lg my-4">
          <p><strong>과제:</strong> 의료 영상 진단 AI의 보안</p>
          <ul className="mt-2">
            <li>• 적대적 예제로 인한 오진 위험</li>
            <li>• 환자 데이터 프라이버시</li>
            <li>• 규제 준수 (HIPAA 등)</li>
          </ul>
          <p className="mt-4"><strong>해결책:</strong></p>
          <ul>
            <li>• 차분 프라이버시 적용</li>
            <li>• 설명가능한 AI 도입</li>
            <li>• 엄격한 접근 제어</li>
          </ul>
        </div>
      </section>

      <section className="mb-8">
        <h2>3. 미래 전망과 과제</h2>
        
        <h3>신흥 위협</h3>
        <ul>
          <li><strong>생성 AI 악용</strong>: 딥페이크, 합성 콘텐츠</li>
          <li><strong>LLM 보안</strong>: 프롬프트 인젝션, 정보 유출</li>
          <li><strong>양자 컴퓨팅 위협</strong>: 기존 암호화 무력화</li>
        </ul>
        
        <h3>방어 기술 발전</h3>
        <ul>
          <li><strong>Zero Trust AI</strong>: 모든 입력을 의심</li>
          <li><strong>블록체인 기반 모델 검증</strong></li>
          <li><strong>양자 저항 암호화</strong></li>
          <li><strong>자율적 방어 시스템</strong></li>
        </ul>
        
        <div className="bg-purple-50 dark:bg-purple-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">Best Practices</h4>
          <ol className="space-y-2">
            <li>1. Security by Design 원칙 적용</li>
            <li>2. 지속적인 위협 모델링</li>
            <li>3. 다계층 방어 전략</li>
            <li>4. 정기적인 보안 감사</li>
            <li>5. 사고 대응 계획 수립</li>
            <li>6. 보안 인식 교육</li>
          </ol>
        </div>
      </section>

      <div className="my-8">
        <SecurityAuditTool />
      </div>
    </div>
  )
}