'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Eye } from 'lucide-react'
import References from '@/components/common/References'

export default function Section4() {
  return (
    <>
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
            <Eye className="text-purple-600" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.4 Zero Trust Architecture</h2>
            <p className="text-gray-600 dark:text-gray-400">"절대 신뢰하지 않고 항상 검증"</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-3">Zero Trust 원칙</h3>
            <ul className="space-y-2 text-purple-700 dark:text-purple-300">
              <li>✓ 모든 요청을 검증</li>
              <li>✓ 최소 권한 원칙 적용</li>
              <li>✓ 지속적인 모니터링</li>
              <li>✓ 컨텍스트 기반 접근 제어</li>
            </ul>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">Zero Trust RAG 구현</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
              <code>{`import jwt
from datetime import datetime, timedelta
import redis
from cryptography.fernet import Fernet

class ZeroTrustRAG:
    def __init__(self):
        self.redis_client = redis.Redis()
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)

    def authenticate_request(self, token: str, ip_address: str) -> Dict:
        """모든 요청에 대한 인증"""
        try:
            # 1. 토큰 검증
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])

            # 2. IP 화이트리스트 확인
            if not self._check_ip_whitelist(ip_address, payload['user_id']):
                return {'success': False, 'reason': 'IP not authorized'}

            # 3. 세션 유효성 확인
            if not self._verify_session(payload['session_id']):
                return {'success': False, 'reason': 'Invalid session'}

            # 4. 요청 빈도 제한 확인
            if not self._check_rate_limit(payload['user_id']):
                return {'success': False, 'reason': 'Rate limit exceeded'}

            return {'success': True, 'user_id': payload['user_id']}

        except jwt.ExpiredSignatureError:
            return {'success': False, 'reason': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'success': False, 'reason': 'Invalid token'}

    def create_secure_context(self, user_id: str, documents: List[str]) -> str:
        """암호화된 컨텍스트 생성"""
        # 1. 사용자별 암호화 키 생성
        user_key = self._get_user_encryption_key(user_id)

        # 2. 문서 암호화
        encrypted_docs = []
        for doc in documents:
            encrypted = self.cipher.encrypt(doc.encode())
            encrypted_docs.append(encrypted)

        # 3. 임시 접근 토큰 생성
        access_token = self._create_context_token(user_id, encrypted_docs)

        return access_token

    def _check_rate_limit(self, user_id: str) -> bool:
        """API 호출 빈도 제한"""
        key = f"rate_limit:{user_id}"
        current_count = self.redis_client.incr(key)

        if current_count == 1:
            self.redis_client.expire(key, 60)  # 1분 윈도우

        return current_count <= 100  # 분당 100회 제한

# 실무 체크리스트
security_checklist = {
    "PII Protection": [
        "✓ 자동 PII 탐지 시스템 구축",
        "✓ 마스킹 정책 수립 및 적용",
        "✓ 정기적인 PII 스캔 실행"
    ],
    "Prompt Injection": [
        "✓ 입력 검증 레이어 구현",
        "✓ 패턴 기반 필터링",
        "✓ 의미적 분석 적용"
    ],
    "Access Control": [
        "✓ RBAC 시스템 구현",
        "✓ 문서별 권한 설정",
        "✓ 감사 로그 시스템"
    ],
    "Zero Trust": [
        "✓ 모든 요청 인증",
        "✓ 암호화 적용",
        "✓ 실시간 모니터링"
    ]
}`}</code>
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-green-800 dark:text-green-200 mb-3">Production 체크리스트</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">필수 구현 사항</h4>
                <ul className="space-y-1 text-sm text-green-700 dark:text-green-300">
                  <li>✅ PII 자동 탐지 및 마스킹</li>
                  <li>✅ 프롬프트 인젝션 방어</li>
                  <li>✅ 역할 기반 접근 제어</li>
                  <li>✅ 감사 로그 시스템</li>
                  <li>✅ 암호화 (전송/저장)</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">모니터링 지표</h4>
                <ul className="space-y-1 text-sm text-green-700 dark:text-green-300">
                  <li>📊 PII 탐지율</li>
                  <li>📊 차단된 악성 프롬프트</li>
                  <li>📊 권한 위반 시도</li>
                  <li>📊 API 호출 패턴</li>
                  <li>📊 데이터 접근 로그</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 보안 & 프라이버시 프레임워크',
            icon: 'web' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'Microsoft Presidio: PII Detection & Anonymization',
                authors: 'Microsoft',
                year: '2024',
                description: 'PII 자동 탐지 라이브러리 - 50+ 엔티티 타입, 다국어 지원, 마스킹/암호화/해싱',
                link: 'https://microsoft.github.io/presidio/'
              },
              {
                title: 'OWASP Top 10 for LLM Applications',
                authors: 'OWASP Foundation',
                year: '2024',
                description: 'LLM 애플리케이션 10대 보안 취약점 - Prompt Injection, Data Leakage, Model DoS',
                link: 'https://owasp.org/www-project-top-10-for-large-language-model-applications/'
              },
              {
                title: 'NeMo Guardrails: Content Moderation',
                authors: 'NVIDIA',
                year: '2024',
                description: 'LLM 가드레일 프레임워크 - 입력/출력 검증, 토픽 제어, 안전성 보장',
                link: 'https://github.com/NVIDIA/NeMo-Guardrails'
              },
              {
                title: 'LangKit: LLM Security Toolkit',
                authors: 'WhyLabs',
                year: '2024',
                description: 'LLM 보안 도구 - 프롬프트 인젝션 탐지, 토픽 분류, 독성 감지',
                link: 'https://github.com/whylabs/langkit'
              },
              {
                title: 'GDPR Compliance for AI Systems',
                authors: 'European Commission',
                year: '2024',
                description: 'AI 시스템 GDPR 준수 - 데이터 최소화, 삭제권, 설명 가능성 요구사항',
                link: 'https://ec.europa.eu/info/law/law-topic/data-protection_en'
              }
            ]
          },
          {
            title: '📖 보안 연구 논문',
            icon: 'research' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'Prompt Injection Attacks and Defenses',
                authors: 'Greshake et al., ETH Zurich',
                year: '2024',
                description: '프롬프트 인젝션 공격 분류 - Indirect Injection, Context Poisoning, 방어 전략',
                link: 'https://arxiv.org/abs/2302.12173'
              },
              {
                title: 'Jailbreaking ChatGPT via Prompt Engineering',
                authors: 'Liu et al., Singapore Management University',
                year: '2024',
                description: '탈옥 공격 패턴 분석 - 역할 변경, 시나리오 조작, 다층 방어 필요성',
                link: 'https://arxiv.org/abs/2305.13860'
              },
              {
                title: 'Membership Inference Attacks Against LLMs',
                authors: 'Carlini et al., Google DeepMind',
                year: '2024',
                description: '학습 데이터 유출 공격 - 모델이 특정 텍스트를 기억하는지 판별',
                link: 'https://arxiv.org/abs/2012.07805'
              },
              {
                title: 'Differential Privacy for RAG Systems',
                authors: 'Tramer et al., Stanford',
                year: '2024',
                description: 'RAG에서 차분 프라이버시 - Noise Injection, Privacy Budget, Utility Trade-off',
                link: 'https://arxiv.org/abs/2301.07320'
              }
            ]
          },
          {
            title: '🛠️ 보안 도구 & 서비스',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'LLM Guard: Security Toolkit',
                authors: 'Protect AI',
                year: '2024',
                description: '포괄적 LLM 보안 - 입력 검증, 출력 필터링, PII 마스킹, 독성 감지',
                link: 'https://llm-guard.com/'
              },
              {
                title: 'Rebuff: Prompt Injection Detector',
                authors: 'Rebuff AI',
                year: '2024',
                description: '프롬프트 인젝션 전문 탐지 - 실시간 API, 머신러닝 기반 분석',
                link: 'https://github.com/protectai/rebuff'
              },
              {
                title: 'Vault by HashiCorp: Secrets Management',
                authors: 'HashiCorp',
                year: '2024',
                description: '암호화 키 관리 - API 키, DB 자격증명, 동적 시크릿 생성',
                link: 'https://www.vaultproject.io/'
              },
              {
                title: 'Snyk for AI: Vulnerability Scanning',
                authors: 'Snyk',
                year: '2024',
                description: 'AI/ML 보안 스캐닝 - 의존성 취약점, 모델 보안, 공급망 리스크',
                link: 'https://snyk.io/product/snyk-code/'
              },
              {
                title: 'AWS Macie: Data Privacy Discovery',
                authors: 'Amazon Web Services',
                year: '2024',
                description: 'S3 데이터 PII 자동 탐지 - 머신러닝 기반, 자동 분류, 컴플라이언스',
                link: 'https://aws.amazon.com/macie/'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
        <Link
          href="/modules/rag/supplementary/chapter1"
          className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
        >
          <ArrowLeft size={20} />
          이전: RAGAS 평가 프레임워크
        </Link>

        <Link
          href="/modules/rag/supplementary/chapter3"
          className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
        >
          다음: Cost Optimization
          <ArrowRight size={20} />
        </Link>
      </div>
    </>
  )
}
