'use client';

import React from 'react';
import References from '@/components/common/References';

// Chapter 10: 멀티 클라우드 전략
export default function Chapter10() {
  return (
    <div className="space-y-8">
      {/* Introduction */}
      <section>
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
          멀티 클라우드 & 하이브리드 클라우드 전략
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed text-lg">
          단일 클라우드의 한계를 극복하고, 여러 클라우드 플랫폼을 전략적으로 활용하여 
          벤더 종속성 회피, 최적의 서비스 조합, 지리적 분산을 달성하는 방법을 학습합니다.
        </p>
      </section>

      {/* 1. 멀티 클라우드 vs 하이브리드 클라우드 */}
      <section className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6 border-l-4 border-indigo-500">
        <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4 text-2xl">
          1. 멀티 클라우드 vs 하이브리드 클라우드
        </h3>
        
        <div className="space-y-6">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-indigo-300 dark:border-indigo-700">
              <strong className="text-indigo-700 dark:text-indigo-300 block mb-3 text-lg">멀티 클라우드 (Multi-Cloud)</strong>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                <strong>여러 퍼블릭 클라우드를 동시에 사용</strong> (AWS + Azure + GCP)
              </p>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>목적</strong>: 벤더 종속 회피, Best-of-Breed 선택</li>
                <li>• <strong>예시</strong>: AWS (컴퓨팅) + GCP (빅데이터) + Azure (AI)</li>
                <li>• <strong>복잡도</strong>: 높음 (각 플랫폼 전문 지식 필요)</li>
                <li>• <strong>비용</strong>: 데이터 전송 비용 증가 가능</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border-2 border-purple-300 dark:border-purple-700">
              <strong className="text-purple-700 dark:text-purple-300 block mb-3 text-lg">하이브리드 클라우드 (Hybrid Cloud)</strong>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                <strong>온프레미스 + 퍼블릭 클라우드 통합</strong>
              </p>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>목적</strong>: 규정 준수, 레거시 시스템 유지</li>
                <li>• <strong>예시</strong>: 민감 데이터(온프레미스) + 웹 앱(클라우드)</li>
                <li>• <strong>연결</strong>: VPN, Direct Connect, ExpressRoute</li>
                <li>• <strong>도구</strong>: Azure Arc, AWS Outposts, Google Anthos</li>
              </ul>
            </div>
          </div>

          <div className="bg-indigo-100 dark:bg-indigo-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-indigo-900 dark:text-indigo-100 mb-2">실제 기업 사례</h4>
            <ul className="space-y-1 text-sm text-indigo-800 dark:text-indigo-200">
              <li>🏢 <strong>Netflix</strong>: AWS 중심 (컴퓨팅, 스토리지) + GCP (데이터 분석)</li>
              <li>🏦 <strong>Capital One</strong>: AWS (퍼블릭 클라우드) + 온프레미스 (규제 데이터)</li>
              <li>🛒 <strong>Spotify</strong>: AWS (EU 리전) + GCP (미국 리전) - 지리적 분산</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. 멀티 클라우드 도입 이유 */}
      <section className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
        <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4 text-2xl">
          2. 멀티 클라우드 도입 이유
        </h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-purple-700 dark:text-purple-300 block mb-2">1. 벤더 종속 회피 (Vendor Lock-in Avoidance)</strong>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 단일 클라우드 장애 시 비즈니스 연속성 보장</li>
              <li>• 가격 협상력 강화 (대안 존재)</li>
              <li>• 특정 벤더의 정책 변경 리스크 완화</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-purple-700 dark:text-purple-300 block mb-2">2. Best-of-Breed (각 클라우드의 강점 활용)</strong>
            <div className="grid md:grid-cols-3 gap-3 mt-3 text-sm">
              <div className="bg-purple-50 dark:bg-purple-900/30 p-3 rounded">
                <strong className="text-purple-800 dark:text-purple-200 block mb-1">AWS 강점</strong>
                <ul className="text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• EC2 (가장 많은 인스턴스 타입)</li>
                  <li>• Lambda (서버리스 선구자)</li>
                  <li>• S3 (사실상 표준 스토리지)</li>
                </ul>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/30 p-3 rounded">
                <strong className="text-purple-800 dark:text-purple-200 block mb-1">Azure 강점</strong>
                <ul className="text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• Azure AD (엔터프라이즈 인증)</li>
                  <li>• Hybrid Cloud (Azure Arc)</li>
                  <li>• Azure OpenAI (GPT-4 공식 파트너)</li>
                </ul>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/30 p-3 rounded">
                <strong className="text-purple-800 dark:text-purple-200 block mb-1">GCP 강점</strong>
                <ul className="text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• BigQuery (초고속 데이터 분석)</li>
                  <li>• Kubernetes (GKE 가장 앞섬)</li>
                  <li>• Vertex AI (ML 통합 플랫폼)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-purple-700 dark:text-purple-300 block mb-2">3. 지리적 분산 (Geographic Distribution)</strong>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              특정 리전에 클라우드가 없거나, 규제 요구사항 충족
            </p>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• AWS: 중국 리전 (Ningxia, Beijing) - 특별 파트너십</li>
              <li>• Azure: 정부 클라우드 (Azure Government) - 미국 정부 전용</li>
              <li>• GCP: 유럽 GDPR 준수 리전 우수</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-purple-700 dark:text-purple-300 block mb-2">4. 재해 복구 (Disaster Recovery)</strong>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              한 클라우드 전체 장애 시에도 다른 클라우드에서 서비스 지속<br/>
              예시: Primary (AWS us-east-1) → Failover (GCP us-central1)
            </p>
          </div>
        </div>
      </section>

      {/* 3. 멀티 클라우드 아키텍처 패턴 */}
      <section className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
        <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4 text-2xl">
          3. 멀티 클라우드 아키텍처 패턴
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">1. 분산 워크로드 (Distributed Workload)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                각 클라우드에서 <strong>독립적인 애플리케이션</strong> 실행
              </p>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-blue-700 dark:text-blue-300 block mb-2">구성 예시</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• <strong>AWS</strong>: 웹 애플리케이션 (EC2, RDS)</li>
                    <li>• <strong>GCP</strong>: 데이터 분석 파이프라인 (BigQuery, Dataflow)</li>
                    <li>• <strong>Azure</strong>: AI 모델 훈련 (Azure ML)</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-blue-700 dark:text-blue-300 block mb-2">장단점</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>✓ 간단한 관리 (워크로드 분리)</li>
                    <li>✓ 데이터 전송 최소화</li>
                    <li>✗ 통합 모니터링 어려움</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">2. 복제 & 동기화 (Replication & Sync)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                동일한 애플리케이션을 <strong>여러 클라우드에 복제</strong> (Active-Active 또는 Active-Passive)
              </p>
              <div className="space-y-3">
                <div>
                  <strong className="text-blue-700 dark:text-blue-300 block mb-2 text-sm">Active-Active (양쪽 모두 트래픽 처리)</strong>
                  <ul className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                    <li>• Global Load Balancer로 트래픽 분산 (예: Cloudflare, Akamai)</li>
                    <li>• 데이터베이스 양방향 복제 (CockroachDB, MongoDB Atlas)</li>
                    <li>✓ 최고의 가용성, ✗ 복잡한 데이터 동기화</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-blue-700 dark:text-blue-300 block mb-2 text-sm">Active-Passive (평소엔 한쪽만, 장애 시 전환)</strong>
                  <ul className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                    <li>• Primary: AWS, Secondary: GCP (Standby)</li>
                    <li>• 정기 백업 → 다른 클라우드로 복제</li>
                    <li>✓ 비용 효율적, ✗ 장애 시 RTO/RPO 존재</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">3. 클라우드 버스팅 (Cloud Bursting)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                <strong>온프레미스/프라이빗 클라우드가 메인</strong>, 피크 타임에만 퍼블릭 클라우드로 확장
              </p>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li>• 평소: 온프레미스 (100대 서버)</li>
                <li>• 블랙프라이데이: 온프레미스 + AWS (추가 500대)</li>
                <li>• 도구: VMware Cloud, Azure Stack HCI</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 멀티 클라우드 관리 도구 */}
      <section className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
        <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4 text-2xl">
          4. 멀티 클라우드 관리 도구
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Infrastructure as Code (IaC)</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">Terraform (HashiCorp)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>멀티 클라우드 표준</strong> IaC 도구</li>
                  <li>• AWS, Azure, GCP 모두 단일 문법</li>
                  <li>• 선언적 구문 (HCL)</li>
                  <li>• Terraform Cloud로 상태 관리</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">Pulumi</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 프로그래밍 언어 사용 (TypeScript, Python, Go)</li>
                  <li>• 멀티 클라우드 지원</li>
                  <li>• 조건문, 반복문 등 프로그래밍 로직 활용</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Kubernetes 기반 멀티 클라우드</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">Google Anthos</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• GKE 기반 <strong>멀티 클라우드 K8s 관리 플랫폼</strong></li>
                  <li>• GCP + AWS + Azure + 온프레미스 통합</li>
                  <li>• 중앙화된 정책 관리 (Config Management)</li>
                  <li>• Service Mesh (Istio) 내장</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">Azure Arc</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• Azure Portal에서 <strong>모든 클라우드 K8s 관리</strong></li>
                  <li>• AWS EKS, GCP GKE, 온프레미스 K8s 연결</li>
                  <li>• Azure Policy로 거버넌스 강제</li>
                  <li>• GitOps (Flux) 지원</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">Rancher</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 오픈소스 K8s 관리 플랫폼</li>
                  <li>• 멀티 클러스터 대시보드</li>
                  <li>• 모든 클라우드 K8s 배포 및 관리</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">통합 모니터링 & 로깅</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">Datadog</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 멀티 클라우드 통합 모니터링</li>
                  <li>• AWS, Azure, GCP 메트릭 한눈에</li>
                  <li>• APM, 로그, 트레이싱</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-green-700 dark:text-green-300 block mb-2">New Relic / Dynatrace</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 클라우드 중립적 APM</li>
                  <li>• 분산 추적</li>
                  <li>• AI 기반 이상 감지</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 데이터 통합 & 네트워킹 */}
      <section className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
        <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4 text-2xl">
          5. 멀티 클라우드 데이터 통합 & 네트워킹
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">클라우드 간 네트워크 연결</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">VPN (Virtual Private Network)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 가장 간단한 방법 (인터넷 경유)</li>
                  <li>• AWS VPC ↔ Azure VNet: Site-to-Site VPN</li>
                  <li>• <strong>단점</strong>: 대역폭 제한, 암호화 오버헤드</li>
                  <li>• <strong>비용</strong>: 저렴 (~$0.05/GB)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">Direct Connection (전용 회선)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• AWS Direct Connect + Azure ExpressRoute</li>
                  <li>• 고대역폭 (10Gbps ~ 100Gbps)</li>
                  <li>• 낮은 지연시간, 안정적</li>
                  <li>• <strong>비용</strong>: 높음 (포트 비용 + 데이터 전송)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">Cloud Interconnect (파트너 연결)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• Equinix, Megaport 등 중개업체</li>
                  <li>• 한 곳에서 모든 클라우드 연결</li>
                  <li>• 유연한 대역폭 조정</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">데이터 동기화 전략</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">ETL/ELT 파이프라인</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• Apache Airflow (오픈소스)</li>
                  <li>• Fivetran, Stitch (SaaS)</li>
                  <li>• AWS Glue ↔ BigQuery 데이터 전송</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">Change Data Capture (CDC)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• Debezium (Kafka 기반)</li>
                  <li>• AWS DMS (Database Migration Service)</li>
                  <li>• 실시간 데이터 복제</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. 멀티 클라우드 도전 과제 */}
      <section className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border-l-4 border-red-500">
        <h3 className="font-semibold text-red-800 dark:text-red-200 mb-4 text-2xl">
          6. 멀티 클라우드 도전 과제 & 해결책
        </h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-red-700 dark:text-red-300 block mb-2">1. 복잡성 증가</strong>
            <div className="grid md:grid-cols-2 gap-4 text-sm mt-2">
              <div>
                <strong className="text-gray-900 dark:text-white block mb-1">문제</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 각 클라우드 API, 서비스 학습 필요</li>
                  <li>• 운영 팀 부담 증가</li>
                  <li>• 문제 디버깅 어려움</li>
                </ul>
              </div>
              <div>
                <strong className="text-gray-900 dark:text-white block mb-1">해결책</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>✓ IaC (Terraform) 사용</li>
                  <li>✓ Kubernetes로 추상화</li>
                  <li>✓ 통합 모니터링 도구 (Datadog)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-red-700 dark:text-red-300 block mb-2">2. 데이터 전송 비용 (Egress Costs)</strong>
            <div className="grid md:grid-cols-2 gap-4 text-sm mt-2">
              <div>
                <strong className="text-gray-900 dark:text-white block mb-1">문제</strong>
                <p className="text-gray-700 dark:text-gray-300">
                  클라우드 간 데이터 전송은 매우 비쌈<br/>
                  예: AWS → GCP 1TB 전송 = ~$90
                </p>
              </div>
              <div>
                <strong className="text-gray-900 dark:text-white block mb-1">해결책</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>✓ 데이터 전송 최소화 (로컬 처리)</li>
                  <li>✓ 압축 사용</li>
                  <li>✓ Direct Connect 활용 (대량 전송 시)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-red-700 dark:text-red-300 block mb-2">3. 보안 & 컴플라이언스</strong>
            <div className="grid md:grid-cols-2 gap-4 text-sm mt-2">
              <div>
                <strong className="text-gray-900 dark:text-white block mb-1">문제</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• 각 클라우드마다 다른 보안 정책</li>
                  <li>• 감사 추적 통합 어려움</li>
                  <li>• 규정 준수 증명 복잡</li>
                </ul>
              </div>
              <div>
                <strong className="text-gray-900 dark:text-white block mb-1">해결책</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>✓ 중앙화된 IAM (Okta, Azure AD)</li>
                  <li>✓ SIEM 도구 (Splunk, Sentinel)</li>
                  <li>✓ Policy as Code (Open Policy Agent)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-red-700 dark:text-red-300 block mb-2">4. 기술 인력 부족</strong>
            <div className="grid md:grid-cols-2 gap-4 text-sm mt-2">
              <div>
                <strong className="text-gray-900 dark:text-white block mb-1">문제</strong>
                <p className="text-gray-700 dark:text-gray-300">
                  AWS, Azure, GCP 모두 능숙한 엔지니어 부족
                </p>
              </div>
              <div>
                <strong className="text-gray-900 dark:text-white block mb-1">해결책</strong>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li>✓ T자형 인재 (깊이 + 넓이)</li>
                  <li>✓ 팀별 클라우드 전문화</li>
                  <li>✓ 매니지드 서비스 활용</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Summary */}
      <section className="bg-gradient-to-r from-indigo-100 to-purple-100 dark:from-indigo-900/30 dark:to-purple-900/30 rounded-lg p-6">
        <h3 className="font-semibold text-indigo-900 dark:text-indigo-100 mb-4 text-xl">
          📚 학습 요약
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <h4 className="font-semibold text-indigo-800 dark:text-indigo-200">핵심 개념</h4>
            <ul className="space-y-1 text-indigo-700 dark:text-indigo-300">
              <li>✓ 멀티 클라우드: 여러 퍼블릭 클라우드 동시 사용</li>
              <li>✓ 하이브리드 클라우드: 온프레미스 + 클라우드</li>
              <li>✓ Best-of-Breed: 각 클라우드 강점 활용</li>
              <li>✓ 벤더 종속 회피: 협상력 강화</li>
            </ul>
          </div>
          <div className="space-y-2">
            <h4 className="font-semibold text-indigo-800 dark:text-indigo-200">관리 도구</h4>
            <ul className="space-y-1 text-indigo-700 dark:text-indigo-300">
              <li>✓ Terraform: 멀티 클라우드 IaC 표준</li>
              <li>✓ Kubernetes: 컨테이너 추상화 레이어</li>
              <li>✓ Anthos/Arc: 통합 K8s 관리</li>
              <li>✓ Datadog: 통합 모니터링</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 멀티 클라우드 전략 가이드',
            icon: 'web',
            color: 'border-indigo-500',
            items: [
              {
                title: 'Google Anthos Documentation',
                link: 'https://cloud.google.com/anthos/docs',
                description: 'Google 멀티 클라우드 플랫폼 공식 문서'
              },
              {
                title: 'Azure Arc Overview',
                link: 'https://learn.microsoft.com/en-us/azure/azure-arc/',
                description: 'Azure 하이브리드/멀티 클라우드 관리'
              },
              {
                title: 'AWS Outposts',
                link: 'https://aws.amazon.com/outposts/',
                description: 'AWS 하이브리드 클라우드 솔루션'
              }
            ]
          },
          {
            title: '📖 백서 & 연구',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Multi-Cloud Strategy Guide (Gartner)',
                link: 'https://www.gartner.com/en/documents/3956799',
                description: '가트너 멀티 클라우드 전략 가이드'
              },
              {
                title: 'CNCF Multi-Cloud Working Group',
                link: 'https://github.com/cncf/wg-multitenancy',
                description: '클라우드 네이티브 멀티 클라우드 표준화'
              }
            ]
          },
          {
            title: '🛠️ 멀티 클라우드 도구',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Terraform',
                link: 'https://www.terraform.io/',
                description: '멀티 클라우드 IaC 표준 도구'
              },
              {
                title: 'Pulumi',
                link: 'https://www.pulumi.com/',
                description: '프로그래밍 언어 기반 IaC'
              },
              {
                title: 'Rancher',
                link: 'https://www.rancher.com/',
                description: '멀티 클라우드 Kubernetes 관리'
              },
              {
                title: 'Datadog',
                link: 'https://www.datadoghq.com/',
                description: '멀티 클라우드 통합 모니터링'
              }
            ]
          },
          {
            title: '🎓 학습 리소스',
            icon: 'book',
            color: 'border-blue-500',
            items: [
              {
                title: 'CNCF Landscape',
                link: 'https://landscape.cncf.io/',
                description: '클라우드 네이티브 도구 전체 지도'
              },
              {
                title: 'Multi-Cloud Architecture Best Practices',
                link: 'https://cloud.google.com/architecture/multi-cloud-patterns',
                description: 'Google Cloud 멀티 클라우드 패턴'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
