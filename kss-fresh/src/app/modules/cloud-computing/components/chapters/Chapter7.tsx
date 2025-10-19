'use client';

import React from 'react';
import References from '@/components/common/References';

// Chapter 7: 컨테이너와 오케스트레이션
export default function Chapter7() {
  return (
    <div className="space-y-8">
      {/* Introduction */}
      <section>
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
          컨테이너와 오케스트레이션
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed text-lg">
          Docker 컨테이너 기술과 Kubernetes 오케스트레이션을 클라우드 환경에서 활용하여 
          확장 가능하고 이식성 높은 애플리케이션을 구축하는 방법을 학습합니다.
        </p>
      </section>

      {/* 1. Docker 컨테이너 기초 */}
      <section className="bg-cyan-50 dark:bg-cyan-900/20 rounded-lg p-6 border-l-4 border-cyan-500">
        <h3 className="font-semibold text-cyan-800 dark:text-cyan-200 mb-4 text-2xl">
          1. Docker 컨테이너 기초
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">컨테이너란?</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              애플리케이션과 모든 의존성을 하나의 패키지로 묶어 <strong>어디서든 동일하게 실행</strong>할 수 있도록 하는 경량 가상화 기술입니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-cyan-700 dark:text-cyan-300 block mb-2">VM vs Container</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>VM</strong>: 전체 OS 포함, GB 단위, 분 단위 부팅</li>
                  <li>• <strong>Container</strong>: OS 커널 공유, MB 단위, 초 단위 시작</li>
                  <li>• <strong>이식성</strong>: 컨테이너가 훨씬 우수</li>
                  <li>• <strong>오버헤드</strong>: 컨테이너가 훨씬 적음</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-cyan-700 dark:text-cyan-300 block mb-2">핵심 장점</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>✓ <strong>일관성</strong>: "내 컴퓨터에서는 되는데" 문제 해결</li>
                  <li>✓ <strong>격리</strong>: 애플리케이션 간 충돌 방지</li>
                  <li>✓ <strong>효율성</strong>: 리소스 공유로 밀도 향상</li>
                  <li>✓ <strong>속도</strong>: 빠른 시작/중지</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Docker 핵심 개념</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-cyan-700 dark:text-cyan-300 block mb-2">Dockerfile</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  컨테이너 이미지를 빌드하는 명령어 스크립트
                </p>
                <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-xs overflow-x-auto">
{`FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]`}
                </pre>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-cyan-700 dark:text-cyan-300 block mb-2">Image (이미지)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  실행 가능한 패키지 (읽기 전용 템플릿). Docker Hub, ECR, ACR, GCR에서 공유
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-cyan-700 dark:text-cyan-300 block mb-2">Container (컨테이너)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  이미지의 실행 인스턴스 (런타임 객체)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. Kubernetes 기초 */}
      <section className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
        <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4 text-2xl">
          2. Kubernetes (K8s) 오케스트레이션
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Kubernetes란?</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              컨테이너화된 애플리케이션의 <strong>배포, 스케일링, 관리를 자동화</strong>하는 오픈소스 플랫폼입니다. 
              (Google이 개발, CNCF가 관리)
            </p>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>자동 스케일링</strong>: CPU/메모리 사용량에 따라 Pod 수 조정 (HPA)</li>
              <li>• <strong>자가 치유</strong>: 실패한 컨테이너 자동 재시작</li>
              <li>• <strong>로드 밸런싱</strong>: 트래픽을 여러 Pod에 분산</li>
              <li>• <strong>롤링 업데이트</strong>: 무중단 배포 (Blue-Green, Canary)</li>
              <li>• <strong>서비스 디스커버리</strong>: DNS 기반 서비스 검색</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 아키텍처</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-3">Control Plane (마스터)</strong>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>API Server</strong>: 모든 작업의 진입점</li>
                  <li>• <strong>etcd</strong>: 클러스터 상태 저장 (Key-Value)</li>
                  <li>• <strong>Scheduler</strong>: Pod를 어느 노드에 배치할지 결정</li>
                  <li>• <strong>Controller Manager</strong>: 원하는 상태 유지</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-3">Worker Node</strong>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>Kubelet</strong>: Pod 실행 관리</li>
                  <li>• <strong>Kube-proxy</strong>: 네트워크 프록시</li>
                  <li>• <strong>Container Runtime</strong>: Docker, containerd</li>
                  <li>• <strong>Pods</strong>: 실제 컨테이너 실행</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">주요 리소스</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">Pod</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  하나 이상의 컨테이너 그룹 (배포 최소 단위)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">Deployment</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Pod의 선언적 업데이트 (원하는 replica 수 유지)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">Service</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Pod에 대한 안정적인 네트워크 엔드포인트 (ClusterIP, NodePort, LoadBalancer)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">ConfigMap & Secret</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  설정 데이터 및 민감 정보 관리
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-blue-700 dark:text-blue-300 block mb-1 text-sm">Ingress</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  HTTP/HTTPS 라우팅 규칙 (도메인 기반 라우팅)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. 클라우드 Kubernetes 서비스 */}
      <section className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6 border-l-4 border-indigo-500">
        <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-4 text-2xl">
          3. 관리형 Kubernetes 서비스
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">AWS EKS (Elastic Kubernetes Service)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>Control Plane 완전 관리</strong>: 고가용성 (3개 AZ에 분산)</li>
                <li>• <strong>가격</strong>: 클러스터당 $0.10/시간 + 워커 노드 EC2 비용</li>
                <li>• <strong>통합</strong>: IAM 인증, CloudWatch 로깅, ALB/NLB 자동 연결</li>
                <li>• <strong>Fargate 지원</strong>: 서버리스 Pod 실행 (노드 관리 불필요)</li>
                <li>• <strong>EKS Anywhere</strong>: 온프레미스에서도 EKS 실행</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Azure AKS (Azure Kubernetes Service)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>Control Plane 무료</strong>: 컨트롤 플레인 비용 없음 (노드만 과금)</li>
                <li>• <strong>Azure AD 통합</strong>: RBAC 및 싱글 사인온</li>
                <li>• <strong>Virtual Nodes</strong>: Azure Container Instances로 서버리스 확장</li>
                <li>• <strong>Azure Monitor</strong>: 컨테이너 인사이트 기본 제공</li>
                <li>• <strong>Azure Dev Spaces</strong>: 팀 협업 개발 환경</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">GCP GKE (Google Kubernetes Engine)</h4>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• <strong>Autopilot 모드</strong>: 노드 관리 완전 자동화 (K8s 네이티브 서버리스)</li>
                <li>• <strong>Standard 모드</strong>: 세밀한 제어 가능</li>
                <li>• <strong>가격</strong>: 클러스터당 $0.10/시간 (Autopilot은 Pod 리소스만 과금)</li>
                <li>• <strong>GKE Enterprise</strong>: 멀티 클러스터 관리 (Anthos)</li>
                <li>• <strong>가장 빠른 업그레이드</strong>: Kubernetes 최신 버전 즉시 지원</li>
              </ul>
            </div>
          </div>

          <div className="bg-indigo-100 dark:bg-indigo-900/30 p-4 rounded-lg">
            <h4 className="font-semibold text-indigo-900 dark:text-indigo-100 mb-2">선택 가이드</h4>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <strong className="text-indigo-800 dark:text-indigo-200 block mb-1">EKS 선택</strong>
                <ul className="text-indigo-700 dark:text-indigo-300 space-y-1">
                  <li>• AWS 중심 인프라</li>
                  <li>• Fargate 서버리스 원할 때</li>
                </ul>
              </div>
              <div>
                <strong className="text-indigo-800 dark:text-indigo-200 block mb-1">AKS 선택</strong>
                <ul className="text-indigo-700 dark:text-indigo-300 space-y-1">
                  <li>• Azure 생태계</li>
                  <li>• 비용 최소화 (Control Plane 무료)</li>
                </ul>
              </div>
              <div>
                <strong className="text-indigo-800 dark:text-indigo-200 block mb-1">GKE 선택</strong>
                <ul className="text-indigo-700 dark:text-indigo-300 space-y-1">
                  <li>• 최신 K8s 기능</li>
                  <li>• Autopilot 완전 관리형</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 컨테이너 레지스트리 */}
      <section className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
        <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4 text-2xl">
          4. 컨테이너 레지스트리
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">클라우드 프라이빗 레지스트리</h4>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">AWS ECR</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 완전 관리형</li>
                  <li>• IAM 통합 인증</li>
                  <li>• 이미지 스캔 (보안 취약점)</li>
                  <li>• 가격: $0.10/GB/월</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">Azure ACR</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• Geo-replication (전 세계 배포)</li>
                  <li>• Azure AD 인증</li>
                  <li>• 웹훅 지원 (CI/CD)</li>
                  <li>• 가격: $0.167/GB/월 (Standard)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-2">GCP GCR/Artifact Registry</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• Artifact Registry (차세대)</li>
                  <li>• 멀티 포맷 (Docker, Maven, npm)</li>
                  <li>• 취약점 스캔 내장</li>
                  <li>• 가격: $0.10/GB/월</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">Public 레지스트리</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-1">Docker Hub</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  가장 큰 공개 레지스트리. 무료 (Pull 제한: 100/6시간), Pro: $5/월 (무제한)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-purple-700 dark:text-purple-300 block mb-1">GitHub Container Registry (GHCR)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  GitHub Actions와 완벽 통합. Public 저장소 무료
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 배포 전략 */}
      <section className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
        <h3 className="font-semibold text-green-800 dark:text-green-200 mb-4 text-2xl">
          5. Kubernetes 배포 전략
        </h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-green-700 dark:text-green-300 block mb-2">Rolling Update (롤링 업데이트)</strong>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              Pod를 하나씩 점진적으로 교체 (기본 전략)
            </p>
            <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <li>✓ 무중단 배포</li>
              <li>✓ 리소스 효율적</li>
              <li>✗ 두 버전이 동시 실행 (호환성 필요)</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-green-700 dark:text-green-300 block mb-2">Blue-Green Deployment</strong>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              새 버전(Green)을 완전히 배포 후 트래픽 전환
            </p>
            <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <li>✓ 즉시 롤백 가능 (Service만 변경)</li>
              <li>✓ 한 버전만 실행</li>
              <li>✗ 2배 리소스 필요</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-green-700 dark:text-green-300 block mb-2">Canary Deployment (카나리 배포)</strong>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              소수 사용자(5~10%)에게 먼저 배포 후 점진 확대
            </p>
            <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <li>✓ 위험 최소화 (일부만 영향)</li>
              <li>✓ A/B 테스트 가능</li>
              <li>✗ 복잡한 트래픽 분할 필요 (Istio, Linkerd)</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-green-700 dark:text-green-300 block mb-2">Recreate (재생성)</strong>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              기존 Pod 모두 종료 후 새 Pod 시작
            </p>
            <ul className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
              <li>✓ 간단함</li>
              <li>✗ 다운타임 발생</li>
              <li>• 사용 사례: 두 버전 동시 실행 불가능한 경우</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 6. 모니터링 & 로깅 */}
      <section className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
        <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-4 text-2xl">
          6. 모니터링 & 로깅
        </h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">핵심 도구</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">Prometheus + Grafana</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>Prometheus</strong>: 메트릭 수집 (Pull 모델)</li>
                  <li>• <strong>Grafana</strong>: 시각화 대시보드</li>
                  <li>• K8s 표준 모니터링 스택</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-2">ELK Stack (Elasticsearch, Logstash, Kibana)</strong>
                <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>• 중앙화된 로그 수집</li>
                  <li>• 로그 검색 및 분석</li>
                  <li>• 대안: Loki (Grafana Labs)</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">클라우드 네이티브 모니터링</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-1">AWS CloudWatch Container Insights</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  EKS/ECS 메트릭 자동 수집 (CPU, 메모리, 네트워크)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-1">Azure Monitor for Containers</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  AKS 통합 모니터링 (로그 분석, 메트릭)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
                <strong className="text-orange-700 dark:text-orange-300 block mb-1">Google Cloud Operations (Stackdriver)</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  GKE 기본 통합 (로깅, 모니터링, 추적)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Summary */}
      <section className="bg-gradient-to-r from-cyan-100 to-blue-100 dark:from-cyan-900/30 dark:to-blue-900/30 rounded-lg p-6">
        <h3 className="font-semibold text-cyan-900 dark:text-cyan-100 mb-4 text-xl">
          📚 학습 요약
        </h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <h4 className="font-semibold text-cyan-800 dark:text-cyan-200">핵심 개념</h4>
            <ul className="space-y-1 text-cyan-700 dark:text-cyan-300">
              <li>✓ Docker: 컨테이너 표준 (이식성, 격리)</li>
              <li>✓ Kubernetes: 컨테이너 오케스트레이션 (자동 스케일링, 자가 치유)</li>
              <li>✓ EKS/AKS/GKE: 관리형 K8s (Control Plane 관리 불필요)</li>
              <li>✓ ECR/ACR/GCR: 프라이빗 컨테이너 레지스트리</li>
            </ul>
          </div>
          <div className="space-y-2">
            <h4 className="font-semibold text-cyan-800 dark:text-cyan-200">운영 전략</h4>
            <ul className="space-y-1 text-cyan-700 dark:text-cyan-300">
              <li>✓ Rolling Update: 무중단 점진 배포</li>
              <li>✓ Blue-Green: 즉시 롤백 가능</li>
              <li>✓ Canary: 위험 최소화</li>
              <li>✓ Prometheus + Grafana: 모니터링</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 문서',
            icon: 'web' as const,
            color: 'border-cyan-500',
            items: [
              {
                title: 'Kubernetes 공식 문서',
                url: 'https://kubernetes.io/docs/home/',
                description: 'K8s 완전 가이드 및 API 레퍼런스'
              },
              {
                title: 'Docker 공식 문서',
                url: 'https://docs.docker.com/',
                description: 'Docker 엔진, Compose, Swarm 문서'
              },
              {
                title: 'AWS EKS Documentation',
                url: 'https://docs.aws.amazon.com/eks/',
                description: 'EKS 사용자 가이드 및 모범 사례'
              },
              {
                title: 'Azure AKS Documentation',
                url: 'https://learn.microsoft.com/en-us/azure/aks/',
                description: 'AKS 배포 및 관리 가이드'
              },
              {
                title: 'Google GKE Documentation',
                url: 'https://cloud.google.com/kubernetes-engine/docs',
                description: 'GKE 및 Autopilot 모드 문서'
              }
            ]
          },
          {
            title: '📖 핵심 서적',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Kubernetes in Action (Manning)',
                url: 'https://www.manning.com/books/kubernetes-in-action-second-edition',
                description: 'K8s 실전 가이드 (2nd Edition)'
              },
              {
                title: 'The Kubernetes Book (Nigel Poulton)',
                url: 'https://www.amazon.com/Kubernetes-Book-Nigel-Poulton/dp/1521823634',
                description: '초보자를 위한 K8s 입문서'
              },
              {
                title: 'Production Kubernetes (O\'Reilly)',
                url: 'https://www.oreilly.com/library/view/production-kubernetes/9781492092292/',
                description: '프로덕션 K8s 운영 가이드'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Helm',
                url: 'https://helm.sh/',
                description: 'Kubernetes 패키지 관리자 (차트)'
              },
              {
                title: 'Istio',
                url: 'https://istio.io/',
                description: '서비스 메시 (트래픽 관리, 보안, 관측성)'
              },
              {
                title: 'ArgoCD',
                url: 'https://argo-cd.readthedocs.io/',
                description: 'GitOps 기반 CD 도구'
              },
              {
                title: 'Lens',
                url: 'https://k8slens.dev/',
                description: 'K8s IDE (클러스터 관리 GUI)'
              }
            ]
          },
          {
            title: '🎓 학습 리소스',
            icon: 'web' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'CNCF Training',
                url: 'https://www.cncf.io/training/',
                description: 'CKA, CKAD, CKS 자격증 과정'
              },
              {
                title: 'Katacoda (Interactive Learning)',
                url: 'https://www.katacoda.com/courses/kubernetes',
                description: '브라우저 기반 K8s 실습'
              },
              {
                title: 'Kubernetes The Hard Way',
                url: 'https://github.com/kelseyhightower/kubernetes-the-hard-way',
                description: 'K8s 클러스터 수동 구축 (깊은 이해)'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
