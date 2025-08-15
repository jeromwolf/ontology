'use client';

import ModelSecurityAnalyzer from '../ModelSecurityAnalyzer';

export default function Chapter7() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h1>배포 보안</h1>
      
      <section className="mb-8">
        <h2>1. 보안 아키텍처</h2>
        <p>
          프로덕션 환경에서 AI 시스템을 안전하게 배포하기 위한 아키텍처 설계가 중요합니다.
        </p>
        
        <h3>계층별 보안</h3>
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-800/50 p-6 rounded-lg">
            <h4 className="font-semibold mb-2">인프라 계층</h4>
            <ul className="space-y-2">
              <li>• 네트워크 격리 및 세분화</li>
              <li>• 안전한 컨테이너 오케스트레이션</li>
              <li>• 하드웨어 보안 모듈(HSM) 활용</li>
              <li>• 신뢰할 수 있는 실행 환경(TEE)</li>
            </ul>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-950/30 p-6 rounded-lg">
            <h4 className="font-semibold mb-2">애플리케이션 계층</h4>
            <ul className="space-y-2">
              <li>• API 게이트웨이 및 rate limiting</li>
              <li>• 입력 검증 및 sanitization</li>
              <li>• 모델 서빙 보안</li>
              <li>• 로깅 및 감사 추적</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/30 p-6 rounded-lg">
            <h4 className="font-semibold mb-2">데이터 계층</h4>
            <ul className="space-y-2">
              <li>• 암호화된 데이터 저장</li>
              <li>• 안전한 데이터 파이프라인</li>
              <li>• 접근 제어 및 권한 관리</li>
              <li>• 데이터 마스킹 및 익명화</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="mb-8">
        <h2>2. MLOps 보안</h2>
        <p>
          ML 파이프라인의 각 단계에서 보안을 통합하여 안전한 ML 운영을 구현합니다.
        </p>
        
        <h3>보안 파이프라인</h3>
        <ul>
          <li>
            <strong>코드 보안</strong>
            <ul className="mt-2 ml-4">
              <li>• 정적 코드 분석(SAST)</li>
              <li>• 의존성 취약점 스캔</li>
              <li>• 코드 서명 및 검증</li>
            </ul>
          </li>
          <li>
            <strong>모델 보안</strong>
            <ul className="mt-2 ml-4">
              <li>• 모델 버전 관리 및 추적</li>
              <li>• 모델 무결성 검증</li>
              <li>• A/B 테스트 보안</li>
            </ul>
          </li>
          <li>
            <strong>배포 보안</strong>
            <ul className="mt-2 ml-4">
              <li>• 안전한 CI/CD 파이프라인</li>
              <li>• 자동화된 보안 테스트</li>
              <li>• 롤백 및 복구 계획</li>
            </ul>
          </li>
        </ul>
      </section>

      <section className="mb-8">
        <h2>3. 모니터링과 대응</h2>
        
        <h3>실시간 모니터링</h3>
        <p>
          배포된 AI 시스템의 보안 상태를 지속적으로 모니터링합니다.
        </p>
        
        <div className="bg-red-50 dark:bg-red-950/30 p-6 rounded-lg my-4">
          <h4 className="font-semibold mb-2">주요 모니터링 지표</h4>
          <ul className="space-y-2">
            <li>• 비정상적인 입력 패턴</li>
            <li>• 예측 신뢰도 변화</li>
            <li>• API 사용 패턴</li>
            <li>• 시스템 리소스 사용량</li>
            <li>• 에러율 및 지연 시간</li>
          </ul>
        </div>
        
        <h3>사고 대응 계획</h3>
        <ol>
          <li><strong>탐지</strong>: 자동화된 알림 시스템</li>
          <li><strong>분류</strong>: 위협 수준 평가</li>
          <li><strong>격리</strong>: 영향받은 시스템 격리</li>
          <li><strong>복구</strong>: 안전한 상태로 복원</li>
          <li><strong>분석</strong>: 사후 분석 및 개선</li>
        </ol>
      </section>

      <div className="my-8">
        <ModelSecurityAnalyzer />
      </div>
    </div>
  )
}