'use client';

import React from 'react';
import References from '@/components/common/References';

// Chapter 8: 클라우드 보안
export default function Chapter8() {
  return (
    <div className="space-y-8">
      {/* Introduction */}
      <section>
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">
          클라우드 보안
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed text-lg">
          클라우드 환경에서 데이터, 애플리케이션, 인프라를 보호하기 위한 핵심 보안 개념과 
          AWS, Azure, GCP의 보안 서비스를 학습합니다. 공동 책임 모델부터 Zero Trust 아키텍처까지 다룹니다.
        </p>
      </section>

      {/* 1. 공동 책임 모델 */}
      <section className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border-l-4 border-red-500">
        <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4 text-2xl">
          1. 공동 책임 모델 (Shared Responsibility Model)
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 개념</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              클라우드 보안은 <strong>클라우드 제공자와 고객이 책임을 나눠</strong> 가집니다.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-red-300 dark:border-red-700">
              <strong className="text-red-700 dark:text-red-300 block mb-3 text-lg">AWS/Azure/GCP 책임 (OF the Cloud)</strong>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>✓ <strong>물리적 보안</strong>: 데이터센터 출입 통제</li>
                <li>✓ <strong>하드웨어</strong>: 서버, 네트워크 장비</li>
                <li>✓ <strong>인프라 소프트웨어</strong>: 하이퍼바이저, 네트워크 가상화</li>
                <li>✓ <strong>관리형 서비스</strong>: RDS, DynamoDB 패치</li>
                <li>✓ <strong>리전/AZ</strong>: 고가용성 인프라</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-orange-300 dark:border-orange-700">
              <strong className="text-orange-700 dark:text-orange-300 block mb-3 text-lg">고객 책임 (IN the Cloud)</strong>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>✓ <strong>데이터 암호화</strong>: 저장/전송 중 데이터</li>
                <li>✓ <strong>IAM 관리</strong>: 사용자 권한, 정책</li>
                <li>✓ <strong>애플리케이션 보안</strong>: 코드 취약점</li>
                <li>✓ <strong>OS 패치</strong>: EC2, VM 운영체제</li>
                <li>✓ <strong>네트워크 구성</strong>: VPC, 방화벽 규칙</li>
                <li>✓ <strong>백업</strong>: 데이터 복구 계획</li>
              </ul>
            </div>
          </div>

          <div className="bg-red-100 dark:bg-red-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-red-900 dark:text-red-100 mb-2">서비스별 책임 범위</h4>
            <ul className="space-y-1 text-sm text-red-800 dark:text-red-200">
              <li>• <strong>IaaS (EC2, VM)</strong>: 고객 책임 많음 (OS, 미들웨어, 데이터)</li>
              <li>• <strong>PaaS (RDS, App Service)</strong>: OS 패치는 클라우드 책임</li>
              <li>• <strong>SaaS (Office 365, Salesforce)</strong>: 대부분 클라우드 책임 (고객은 데이터/접근 관리만)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. IAM (Identity and Access Management) */}
      <section className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
        <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4 text-2xl">
          2. IAM (Identity and Access Management)
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 원칙</h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>최소 권한 원칙 (Least Privilege)</strong>: 필요한 최소 권한만 부여</li>
              <li>• <strong>권한 분리 (Separation of Duties)</strong>: 한 사람이 모든 권한 보유 금지</li>
              <li>• <strong>정기 검토</strong>: 불필요한 권한 주기적 제거</li>
              <li>• <strong>MFA 필수</strong>: 다중 인증 활성화 (특히 루트 계정)</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">플랫폼별 IAM</h4>
            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-3">AWS IAM</strong>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong className="text-gray-900 dark:text-white block mb-2">핵심 개념</strong>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      <li>• <strong>Users</strong>: 개인 사용자</li>
                      <li>• <strong>Groups</strong>: 사용자 그룹 (예: Developers)</li>
                      <li>• <strong>Roles</strong>: EC2, Lambda 등 AWS 리소스가 임시 권한</li>
                      <li>• <strong>Policies</strong>: JSON 형식 권한 정의</li>
                    </ul>
                  </div>
                  <div>
                    <strong className="text-gray-900 dark:text-white block mb-2">모범 사례</strong>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      <li>✓ 루트 계정 사용 금지 (MFA 설정만)</li>
                      <li>✓ 개인별 IAM User 생성</li>
                      <li>✓ AWS Organizations로 멀티 계정 관리</li>
                      <li>✓ IAM Access Analyzer로 외부 공유 감지</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-3">Azure AD (Active Directory)</strong>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong className="text-gray-900 dark:text-white block mb-2">핵심 개념</strong>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      <li>• <strong>Users & Groups</strong>: 조직 계층 구조</li>
                      <li>• <strong>Service Principals</strong>: 애플리케이션 ID</li>
                      <li>• <strong>Managed Identities</strong>: VM, Functions 자동 인증</li>
                      <li>• <strong>RBAC</strong>: 역할 기반 접근 제어</li>
                    </ul>
                  </div>
                  <div>
                    <strong className="text-gray-900 dark:text-white block mb-2">특징</strong>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      <li>✓ 조건부 액세스 (IP, 디바이스 기반)</li>
                      <li>✓ Privileged Identity Management (JIT)</li>
                      <li>✓ Azure AD B2C (고객 ID 관리)</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-3">Google Cloud IAM</strong>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>구조</strong>: Organization → Folder → Project → Resource (계층적 권한 상속)</li>
                  <li>• <strong>Predefined Roles</strong>: 100+ 사전 정의 역할 (Owner, Editor, Viewer 등)</li>
                  <li>• <strong>Custom Roles</strong>: 세밀한 권한 조합</li>
                  <li>• <strong>Service Accounts</strong>: 애플리케이션 인증 (키 자동 로테이션)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. 네트워크 보안 */}
      <section className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
        <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4 text-2xl">
          3. 네트워크 보안
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">VPC (Virtual Private Cloud)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              클라우드 내 격리된 프라이빗 네트워크 - AWS VPC, Azure VNet, GCP VPC
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">핵심 구성 요소</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>Subnets</strong>: Public (인터넷 접근) vs Private</li>
                  <li>• <strong>Route Tables</strong>: 트래픽 라우팅 규칙</li>
                  <li>• <strong>Internet Gateway</strong>: VPC ↔ 인터넷</li>
                  <li>• <strong>NAT Gateway</strong>: Private 서브넷 → 인터넷 (아웃바운드만)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">보안 모범 사례</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>✓ DB는 항상 Private 서브넷 배치</li>
                  <li>✓ 다층 방어 (Multi-tier Architecture)</li>
                  <li>✓ VPC Flow Logs로 트래픽 감사</li>
                  <li>✓ VPC Peering으로 안전한 VPC 간 통신</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">방화벽 & 보안 그룹</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">AWS Security Groups (상태 저장)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 인스턴스 레벨 방화벽 (EC2, RDS, ALB 등)</li>
                  <li>• 허용 규칙만 정의 (기본 전체 거부)</li>
                  <li>• 상태 저장: 아웃바운드 응답 자동 허용</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">Network ACLs (상태 비저장)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 서브넷 레벨 방화벽</li>
                  <li>• 허용/거부 규칙 모두 정의</li>
                  <li>• 상태 비저장: 인바운드/아웃바운드 규칙 각각 필요</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">Azure NSG (Network Security Group)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• AWS Security Group + NACL 역할 통합</li>
                  <li>• 서브넷 또는 NIC에 연결</li>
                  <li>• 우선순위 기반 규칙 평가</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">고급 네트워크 보안</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-1 text-sm">AWS Network Firewall</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Managed 방화벽 (IDS/IPS, DPI)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-1 text-sm">Azure Firewall</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  중앙화된 네트워크 보안 정책
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-1 text-sm">AWS WAF (Web Application Firewall)</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  SQL Injection, XSS 공격 차단
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-1 text-sm">DDoS Protection</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  AWS Shield, Azure DDoS Protection
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 데이터 암호화 */}
      <section className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
        <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4 text-2xl">
          4. 데이터 암호화
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">암호화 유형</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-3">저장 중 암호화 (At Rest)</strong>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>S3/Blob/Cloud Storage</strong>: AES-256 기본 암호화</li>
                  <li>• <strong>RDS/SQL Database</strong>: TDE (Transparent Data Encryption)</li>
                  <li>• <strong>EBS/Disk</strong>: 볼륨 레벨 암호화</li>
                  <li>• <strong>DynamoDB/Cosmos DB</strong>: 자동 암호화</li>
                </ul>
                <div className="mt-3 p-2 bg-green-50 dark:bg-green-900/30 rounded text-xs">
                  모범 사례: 모든 데이터 기본 암호화 활성화
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-3">전송 중 암호화 (In Transit)</strong>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>HTTPS/TLS</strong>: 모든 API 통신</li>
                  <li>• <strong>VPN</strong>: 온프레미스 ↔ 클라우드</li>
                  <li>• <strong>AWS Direct Connect/ExpressRoute</strong>: 전용 회선 (암호화 선택)</li>
                  <li>• <strong>데이터베이스</strong>: SSL/TLS 연결 강제</li>
                </ul>
                <div className="mt-3 p-2 bg-green-50 dark:bg-green-900/30 rounded text-xs">
                  모범 사례: HTTP 비활성화, TLS 1.2+ 사용
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">키 관리 서비스</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">AWS KMS (Key Management Service)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• AWS 관리형 키 vs 고객 관리형 키 (CMK)</li>
                  <li>• 키 로테이션 자동화 (1년 주기)</li>
                  <li>• CloudTrail로 모든 키 사용 감사</li>
                  <li>• 가격: CMK $1/월 + API 호출 $0.03/10,000건</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">Azure Key Vault</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 키, 비밀(암호), 인증서 중앙 관리</li>
                  <li>• HSM (Hardware Security Module) 지원</li>
                  <li>• RBAC 통합 권한 관리</li>
                  <li>• Managed Identity로 안전한 접근</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">Google Cloud KMS</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 대칭/비대칭 키 지원</li>
                  <li>• Cloud HSM: FIPS 140-2 Level 3 준수</li>
                  <li>• External Key Manager (EKM): 외부 키 사용</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-green-100 dark:bg-green-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-green-900 dark:text-green-100 mb-2">Secrets 관리</h4>
            <ul className="space-y-1 text-sm text-green-800 dark:text-green-200">
              <li>• <strong>AWS Secrets Manager</strong>: DB 암호, API 키 자동 로테이션</li>
              <li>• <strong>Azure Key Vault Secrets</strong>: 애플리케이션 비밀 저장</li>
              <li>• <strong>Google Secret Manager</strong>: 버전 관리, 접근 감사</li>
              <li>✗ <strong>절대 금지</strong>: 코드/환경 변수에 하드코딩</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. 컴플라이언스 & 거버넌스 */}
      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
        <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-4 text-2xl">
          5. 컴플라이언스 & 거버넌스
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">주요 컴플라이언스 인증</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-yellow-700 dark:text-yellow-300 block mb-2">글로벌 표준</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>ISO 27001</strong>: 정보 보안 관리</li>
                  <li>• <strong>SOC 1/2/3</strong>: 서비스 조직 통제</li>
                  <li>• <strong>PCI DSS</strong>: 카드 결제 보안</li>
                  <li>• <strong>GDPR</strong>: EU 개인정보 보호</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-yellow-700 dark:text-yellow-300 block mb-2">산업별 인증</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>HIPAA</strong>: 의료 데이터 (미국)</li>
                  <li>• <strong>FedRAMP</strong>: 미국 연방 정부</li>
                  <li>• <strong>ISMS-P</strong>: 한국 개인정보보호</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">거버넌스 도구</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-yellow-700 dark:text-yellow-300 block mb-2">AWS</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>AWS Config</strong>: 리소스 변경 추적 및 규정 준수</li>
                  <li>• <strong>AWS Organizations</strong>: 멀티 계정 중앙 관리</li>
                  <li>• <strong>Service Control Policies (SCP)</strong>: 계정별 권한 제한</li>
                  <li>• <strong>AWS Audit Manager</strong>: 자동 컴플라이언스 보고서</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-yellow-700 dark:text-yellow-300 block mb-2">Azure</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>Azure Policy</strong>: 리소스 생성 규칙 강제 (태그 필수 등)</li>
                  <li>• <strong>Azure Blueprints</strong>: 표준 환경 템플릿</li>
                  <li>• <strong>Microsoft Defender for Cloud</strong>: 보안 점수 및 권장 사항</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-yellow-700 dark:text-yellow-300 block mb-2">GCP</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>Organization Policy Service</strong>: 프로젝트 정책 강제</li>
                  <li>• <strong>Security Command Center</strong>: 통합 보안 대시보드</li>
                  <li>• <strong>Cloud Asset Inventory</strong>: 전체 리소스 가시성</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. 보안 모니터링 & 로깅 */}
      <section className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6 border-l-4 border-teal-500">
        <h3 className="font-semibold text-teal-800 dark:text-teal-200 mb-4 text-2xl">
          6. 보안 모니터링 & 로깅
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 로깅 서비스</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-teal-700 dark:text-teal-300 block mb-2">AWS CloudTrail</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 모든 AWS API 호출 기록 (누가, 언제, 무엇을)</li>
                  <li>• S3에 로그 저장 (암호화 필수)</li>
                  <li>• CloudWatch Logs로 실시간 알림</li>
                  <li>• 사용 사례: "누가 이 EC2를 종료했나?"</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-teal-700 dark:text-teal-300 block mb-2">Azure Monitor & Log Analytics</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 통합 모니터링 플랫폼</li>
                  <li>• KQL (Kusto Query Language)로 로그 분석</li>
                  <li>• Activity Logs: 관리 작업 추적</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-teal-700 dark:text-teal-300 block mb-2">Google Cloud Logging (Stackdriver)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 모든 GCP 서비스 로그 자동 수집</li>
                  <li>• Admin Activity Logs (무료 400일 보관)</li>
                  <li>• 로그 기반 메트릭 및 알림</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">위협 탐지 & SIEM</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-teal-700 dark:text-teal-300 block mb-2">AWS GuardDuty</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  ML 기반 위협 탐지 (비정상 API 호출, 비트코인 마이닝, 암호화폐 채굴 탐지)
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  가격: VPC Flow Logs 분석 GB당 $1.00
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-teal-700 dark:text-teal-300 block mb-2">Microsoft Sentinel</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  클라우드 네이티브 SIEM (Security Information and Event Management)
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  AI 기반 위협 분석, 자동 대응
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-teal-700 dark:text-teal-300 block mb-2">Google Security Command Center Premium</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  Event Threat Detection, Container Threat Detection
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-teal-700 dark:text-teal-300 block mb-2">AWS Security Hub</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  통합 보안 대시보드 (GuardDuty, Inspector, Macie 결과 집계)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Summary */}
      <section className="bg-gradient-to-r from-red-100 to-orange-100 dark:from-red-900/30 dark:to-orange-900/30 rounded-lg p-6">
        <h3 className="font-semibold text-red-900 dark:text-red-100 mb-4 text-xl">
          📚 학습 요약
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <h4 className="font-semibold text-red-800 dark:text-red-200">핵심 개념</h4>
            <ul className="space-y-1 text-red-700 dark:text-red-300">
              <li>✓ 공동 책임 모델: 클라우드/고객 역할 분담</li>
              <li>✓ IAM: 최소 권한 원칙, MFA 필수</li>
              <li>✓ VPC: Private 서브넷, Security Groups</li>
              <li>✓ 암호화: At Rest + In Transit (KMS)</li>
            </ul>
          </div>
          <div className="space-y-2">
            <h4 className="font-semibold text-red-800 dark:text-red-200">보안 도구</h4>
            <ul className="space-y-1 text-red-700 dark:text-red-300">
              <li>✓ CloudTrail/Monitor: 모든 작업 감사</li>
              <li>✓ GuardDuty/Sentinel: 위협 탐지</li>
              <li>✓ Config/Policy: 컴플라이언스 강제</li>
              <li>✓ WAF: 웹 공격 차단</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 보안 문서',
            icon: 'web' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'AWS Security Best Practices',
                url: 'https://docs.aws.amazon.com/security/',
                description: 'AWS 보안 모범 사례 백서'
              },
              {
                title: 'Azure Security Documentation',
                url: 'https://learn.microsoft.com/en-us/azure/security/',
                description: 'Azure 보안 센터 및 가이드'
              },
              {
                title: 'Google Cloud Security',
                url: 'https://cloud.google.com/security',
                description: 'GCP 보안 프레임워크 및 도구'
              },
              {
                title: 'CIS Benchmarks',
                url: 'https://www.cisecurity.org/cis-benchmarks',
                description: 'AWS/Azure/GCP 보안 설정 기준'
              }
            ]
          },
          {
            title: '📖 보안 아키텍처 & 컴플라이언스',
            icon: 'research' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'NIST Cybersecurity Framework',
                url: 'https://www.nist.gov/cyberframework',
                description: '클라우드 보안 프레임워크'
              },
              {
                title: 'AWS Well-Architected - Security Pillar',
                url: 'https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html',
                description: 'AWS 보안 아키텍처 설계 원칙'
              },
              {
                title: 'GDPR Compliance Guide',
                url: 'https://gdpr.eu/',
                description: 'EU 개인정보 보호 규정'
              }
            ]
          },
          {
            title: '🛠️ 보안 도구 & 서비스',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'AWS Security Hub',
                url: 'https://aws.amazon.com/security-hub/',
                description: '통합 보안 모니터링 및 컴플라이언스'
              },
              {
                title: 'Microsoft Defender for Cloud',
                url: 'https://azure.microsoft.com/en-us/products/defender-for-cloud/',
                description: 'Azure 멀티 클라우드 보안'
              },
              {
                title: 'Terraform Sentinel',
                url: 'https://www.terraform.io/docs/cloud/sentinel/',
                description: 'IaC 보안 정책 강제'
              },
              {
                title: 'Aqua Security',
                url: 'https://www.aquasec.com/',
                description: '컨테이너 & 클라우드 보안 플랫폼'
              }
            ]
          },
          {
            title: '🎓 자격증 & 학습',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'AWS Certified Security - Specialty',
                url: 'https://aws.amazon.com/certification/certified-security-specialty/',
                description: 'AWS 보안 전문가 자격증'
              },
              {
                title: 'Azure Security Engineer Associate (AZ-500)',
                url: 'https://learn.microsoft.com/en-us/certifications/azure-security-engineer/',
                description: 'Azure 보안 엔지니어 인증'
              },
              {
                title: 'Google Professional Cloud Security Engineer',
                url: 'https://cloud.google.com/certification/cloud-security-engineer',
                description: 'GCP 보안 전문가 인증'
              },
              {
                title: 'SANS Cloud Security',
                url: 'https://www.sans.org/cloud-security/',
                description: '클라우드 보안 교육 및 인증'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
