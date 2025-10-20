import React from 'react';
import { Cloud, Lock, Shield, AlertTriangle, Code, Database } from 'lucide-react';
import References from '../References';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          클라우드 보안
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          AWS, Azure, GCP 실전 보안 설정과 IaC (Infrastructure as Code)를 학습합니다
        </p>
      </div>

      {/* 2024-2025 클라우드 보안 위협 트렌드 */}
      <section className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <AlertTriangle className="w-7 h-7" />
          2024-2025 클라우드 보안 위협 트렌드
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">🔑 IAM 오구성 공격 급증</h3>
            <p className="text-sm mb-2">
              과도한 권한, 오래된 액세스 키, MFA 미적용으로 인한 데이터 유출 증가 (+125%)
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              출처: Ermetic Cloud Security Report 2024
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">🐳 컨테이너 공급망 공격</h3>
            <p className="text-sm mb-2">
              악성 Docker 이미지, Kubernetes 설정 오류로 인한 암호화폐 채굴 증가
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              출처: Sysdig 2024 Cloud-Native Security Report
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">☁️ 멀티클라우드 복잡성 증가</h3>
            <p className="text-sm mb-2">
              평균 기업이 3.4개 클라우드 사용, 통합 보안 관리의 어려움
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              출처: Flexera 2024 State of the Cloud Report
            </p>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <h3 className="font-bold text-lg mb-2">💰 크립토재킹 진화</h3>
            <p className="text-sm mb-2">
              EC2, Lambda 등 서버리스 환경을 노린 자동화된 크립토재킹 증가
            </p>
            <p className="text-xs bg-black/30 rounded px-2 py-1 inline-block">
              평균 피해액: $53,000/월 (Palo Alto Unit 42)
            </p>
          </div>
        </div>
      </section>

      {/* AWS IAM 최소 권한 원칙 실전 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Lock className="w-7 h-7 text-blue-600" />
          AWS IAM 최소 권한 원칙 (Least Privilege)
        </h2>

        <div className="mb-6 bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-l-4 border-red-500">
          <h3 className="font-bold text-red-900 dark:text-red-300 mb-2">🎯 실제 사례</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
            Capital One 데이터 유출 (2019): 과도한 IAM 권한으로 1억 명 이상의 개인정보 유출
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            피해액: $190M 벌금 / 원인: EC2 인스턴스에 S3 전체 접근 권한 부여
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-300">
            <h3 className="font-bold text-lg mb-3 text-red-900 dark:text-red-300">
              ❌ 과도한 권한 (Bad Practice)
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": "*"
    }
  ]
}

// 🚨 문제점:
// 1. 모든 S3 버킷에 접근 가능
// 2. 삭제, 수정, 읽기 모두 가능
// 3. 다른 계정 버킷도 접근 가능`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              ✅ 최소 권한 (Best Practice)
            </h3>
            <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::my-app-bucket/*",
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": "192.0.2.0/24"
        },
        "StringEquals": {
          "aws:PrincipalOrgID": "o-xxxxx"
        }
      }
    }
  ]
}

// ✅ 개선점:
// 1. 특정 버킷만 접근
// 2. 읽기/쓰기만 허용 (삭제 불가)
// 3. IP 제한 및 조직 제한`}
            </pre>
          </div>
        </div>

        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h4 className="font-bold text-blue-900 dark:text-blue-300 mb-2">🛡️ IAM 보안 체크리스트</h4>
          <div className="grid md:grid-cols-2 gap-3 text-sm">
            <div>
              <p className="font-semibold text-blue-800 dark:text-blue-200 mb-1">정책 관리</p>
              <ul className="text-gray-700 dark:text-gray-300 ml-4 space-y-1">
                <li>• 루트 계정 사용 금지</li>
                <li>• MFA 필수 활성화</li>
                <li>• 액세스 키 정기 로테이션 (90일)</li>
                <li>• IAM Access Analyzer 활성화</li>
              </ul>
            </div>
            <div>
              <p className="font-semibold text-blue-800 dark:text-blue-200 mb-1">모니터링</p>
              <ul className="text-gray-700 dark:text-gray-300 ml-4 space-y-1">
                <li>• CloudTrail 모든 리전 활성화</li>
                <li>• GuardDuty 위협 탐지</li>
                <li>• IAM 크레덴셜 보고서 월간 검토</li>
                <li>• 미사용 권한 자동 제거</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* S3 버킷 보안 설정 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Database className="w-7 h-7 text-orange-600" />
          S3 버킷 보안 설정 (Terraform)
        </h2>

        <div className="mb-6 bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-l-4 border-orange-500">
          <h3 className="font-bold text-orange-900 dark:text-orange-300 mb-2">📊 통계</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            2024년 클라우드 데이터 유출의 <strong>65%</strong>가 잘못 구성된 S3 버킷에서 발생
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">출처: Verizon DBIR 2024</p>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
          <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
            ✅ 안전한 S3 버킷 (Terraform)
          </h3>
          <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`# 1. S3 버킷 생성 (보안 강화)
resource "aws_s3_bucket" "secure_bucket" {
  bucket = "my-secure-app-data-prod"

  tags = {
    Environment = "Production"
    Security    = "High"
  }
}

# 2. 퍼블릭 액세스 완전 차단
resource "aws_s3_bucket_public_access_block" "block" {
  bucket = aws_s3_bucket.secure_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 3. 서버 측 암호화 (AES-256 또는 KMS)
resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.secure_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3_key.arn
    }
    bucket_key_enabled = true
  }
}

# 4. 버전 관리 활성화 (랜섬웨어 대응)
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.secure_bucket.id

  versioning_configuration {
    status     = "Enabled"
    mfa_delete = "Enabled"  # MFA 없이 삭제 불가
  }
}

# 5. 로깅 활성화 (CloudTrail + S3 Access Logs)
resource "aws_s3_bucket_logging" "logging" {
  bucket = aws_s3_bucket.secure_bucket.id

  target_bucket = aws_s3_bucket.log_bucket.id
  target_prefix = "s3-access-logs/"
}

# 6. 라이프사이클 정책 (비용 최적화)
resource "aws_s3_bucket_lifecycle_configuration" "lifecycle" {
  bucket = aws_s3_bucket.secure_bucket.id

  rule {
    id     = "move-to-glacier"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    noncurrent_version_expiration {
      days = 30
    }
  }
}`}
          </pre>
        </div>

        <div className="mt-4 bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
          <h4 className="font-bold text-orange-900 dark:text-orange-300 mb-2">🔍 S3 보안 검증 도구</h4>
          <div className="grid md:grid-cols-3 gap-2 text-sm">
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong className="text-orange-700 dark:text-orange-300">AWS Trusted Advisor</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">무료 / 자동 스캔 / 보안 권고</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong className="text-orange-700 dark:text-orange-300">ScoutSuite</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">오픈소스 / 멀티클라우드</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong className="text-orange-700 dark:text-orange-300">Prowler</strong>
              <p className="text-xs text-gray-600 dark:text-gray-400">CIS 벤치마크 / 300+ 검사</p>
            </div>
          </div>
        </div>
      </section>

      {/* Kubernetes 보안 설정 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          Kubernetes 보안 설정 (Pod Security Standards)
        </h2>

        <div className="mb-6 bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg border-l-4 border-indigo-500">
          <h3 className="font-bold text-indigo-900 dark:text-indigo-300 mb-2">🎯 2024년 Kubernetes 보안 위협</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
            Tesla 크립토재킹 공격: 잘못 구성된 Kubernetes Dashboard로 EC2 인스턴스 탈취
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            원인: 인증 없는 대시보드 노출 + privileged 컨테이너
          </p>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
          <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
            ✅ 안전한 Pod 설정 (Restricted Profile)
          </h3>
          <pre className="bg-gray-900 text-green-400 p-4 rounded text-sm overflow-x-auto font-mono">
{`apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  labels:
    app: secure-app
spec:
  # 1. Security Context (Restricted)
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault

  containers:
  - name: app
    image: my-app:1.0.0

    # 2. Container Security Context
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
          - ALL
      runAsNonRoot: true
      runAsUser: 1000

    # 3. Resource Limits (DDoS/Cryptojacking 방지)
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"

    # 4. Health Checks
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 15
      periodSeconds: 20

    # 5. Read-only Volumes
    volumeMounts:
    - name: config
      mountPath: /etc/config
      readOnly: true

  # 6. Network Policy 적용
  volumes:
  - name: config
    configMap:
      name: app-config`}
          </pre>
        </div>

        <div className="mt-4 grid md:grid-cols-2 gap-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
            <h4 className="font-bold text-indigo-900 dark:text-indigo-300 mb-2">🛡️ Network Policy 예제</h4>
            <pre className="bg-gray-900 text-green-400 p-3 rounded text-sm overflow-x-auto font-mono">
{`apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
spec:
  podSelector:
    matchLabels:
      app: secure-app
  policyTypes:
    - Ingress
  ingress:
    - from:
      - podSelector:
          matchLabels:
            role: frontend
      ports:
        - protocol: TCP
          port: 8080`}
            </pre>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h4 className="font-bold text-purple-900 dark:text-purple-300 mb-2">🔧 Kubernetes 보안 도구</h4>
            <div className="space-y-2 text-sm">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-purple-700 dark:text-purple-300">Falco</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">런타임 위협 탐지 (CNCF)</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-purple-700 dark:text-purple-300">Trivy</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">이미지 취약점 스캔</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-purple-700 dark:text-purple-300">kube-bench</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">CIS 벤치마크 검사</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-purple-700 dark:text-purple-300">OPA Gatekeeper</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">Policy as Code</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CSPM/CWPP 도구 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Shield className="w-7 h-7 text-green-600" />
          CSPM & CWPP 도구
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-2 border-green-300">
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              🔍 CSPM (Cloud Security Posture Management)
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              클라우드 인프라 오구성 탐지 및 자동 수정
            </p>

            <div className="space-y-2 text-sm">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-green-700 dark:text-green-300">Wiz</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">멀티클라우드 / 그래프 기반 / $12B 밸류에이션</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-green-700 dark:text-green-300">Orca Security</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">에이전트리스 / SideScanning 기술</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-green-700 dark:text-green-300">Prisma Cloud (Palo Alto)</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">CSPM + CWPP 통합 / 기업용</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-green-700 dark:text-green-300">Prowler (오픈소스)</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">AWS/Azure/GCP 지원 / 무료</p>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-2 border-blue-300">
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">
              🛡️ CWPP (Cloud Workload Protection Platform)
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              VM, 컨테이너, 서버리스 런타임 보호
            </p>

            <div className="space-y-2 text-sm">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-blue-700 dark:text-blue-300">Aqua Security</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">컨테이너 특화 / Kubernetes 네이티브</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-blue-700 dark:text-blue-300">Sysdig Secure</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">Falco 기반 / 런타임 탐지</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-blue-700 dark:text-blue-300">Lacework</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">행동 분석 / 머신러닝 기반</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <strong className="text-blue-700 dark:text-blue-300">Datadog Cloud Security</strong>
                <p className="text-xs text-gray-600 dark:text-gray-400">통합 모니터링 + 보안</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg border-l-4 border-yellow-500">
          <h4 className="font-bold text-yellow-900 dark:text-yellow-300 mb-2">💡 CSPM vs CWPP 선택 가이드</h4>
          <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-700 dark:text-gray-300">
            <div>
              <p className="font-semibold mb-1">CSPM 우선 (예방)</p>
              <ul className="ml-4 space-y-1">
                <li>• 인프라 오구성 주요 위협</li>
                <li>• 컴플라이언스 요구사항 중요</li>
                <li>• 멀티클라우드 환경</li>
              </ul>
            </div>
            <div>
              <p className="font-semibold mb-1">CWPP 우선 (탐지)</p>
              <ul className="ml-4 space-y-1">
                <li>• 런타임 위협 우려</li>
                <li>• 컨테이너/K8s 환경</li>
                <li>• 제로데이 공격 대응</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 클라우드 보안 프레임워크',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'AWS Well-Architected Framework - Security Pillar',
                url: 'https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/',
                description: 'AWS 보안 아키텍처 설계 원칙 및 모범 사례'
              },
              {
                title: 'Azure Security Benchmarks',
                url: 'https://learn.microsoft.com/en-us/security/benchmark/azure/',
                description: 'Microsoft Azure 보안 설정 기준 (CIS 기반)'
              },
              {
                title: 'GCP Security Best Practices',
                url: 'https://cloud.google.com/security/best-practices',
                description: 'Google Cloud 보안 권장 사항 (IAM, VPC, 암호화 등)'
              },
              {
                title: 'CIS Cloud Foundations Benchmark',
                url: 'https://www.cisecurity.org/benchmark/cloud',
                description: 'AWS/Azure/GCP 보안 설정 벤치마크 (무료)'
              }
            ]
          },
          {
            title: '🔧 IaC 보안 도구',
            icon: 'tools' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'Terraform Sentinel',
                url: 'https://www.terraform.io/cloud-docs/sentinel',
                description: 'Terraform용 Policy as Code 프레임워크 (Terraform Cloud 전용)'
              },
              {
                title: 'Checkov',
                url: 'https://www.checkov.io/',
                description: '오픈소스 IaC 스캐너 - Terraform, CloudFormation, Kubernetes'
              },
              {
                title: 'tfsec',
                url: 'https://aquasecurity.github.io/tfsec/',
                description: 'Terraform 전문 보안 스캐너 (Aqua Security 제작)'
              },
              {
                title: 'Terrascan',
                url: 'https://runterrascan.io/',
                description: '500+ 정책 라이브러리 / 멀티클라우드 IaC 스캔'
              },
              {
                title: 'AWS CloudFormation Guard',
                url: 'https://github.com/aws-cloudformation/cloudformation-guard',
                description: 'AWS 공식 Policy as Code 도구 (오픈소스)'
              }
            ]
          },
          {
            title: '🐳 Kubernetes & 컨테이너 보안',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Kubernetes Pod Security Standards',
                url: 'https://kubernetes.io/docs/concepts/security/pod-security-standards/',
                description: '공식 Pod 보안 프로파일 (Privileged/Baseline/Restricted)'
              },
              {
                title: 'Falco Rules Repository',
                url: 'https://github.com/falcosecurity/rules',
                description: 'CNCF Falco 런타임 탐지 룰 (100+ 프리셋)'
              },
              {
                title: 'Aqua Security Trivy',
                url: 'https://trivy.dev/',
                description: '오픈소스 컨테이너 이미지 취약점 스캐너'
              },
              {
                title: 'kube-bench CIS Benchmark',
                url: 'https://github.com/aquasecurity/kube-bench',
                description: 'Kubernetes CIS 벤치마크 자동화 도구'
              },
              {
                title: 'OPA Gatekeeper Policies',
                url: 'https://open-policy-agent.github.io/gatekeeper-library/',
                description: 'Kubernetes Policy as Code 템플릿 라이브러리'
              }
            ]
          },
          {
            title: '📊 CSPM/CWPP 리소스',
            icon: 'research' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Prowler - AWS Security Assessments',
                url: 'https://github.com/prowler-cloud/prowler',
                description: '오픈소스 CSPM 도구 - AWS 300+ 검사 항목 (무료)'
              },
              {
                title: 'ScoutSuite',
                url: 'https://github.com/nccgroup/ScoutSuite',
                description: '멀티클라우드 보안 감사 도구 (AWS/Azure/GCP/AliCloud)'
              },
              {
                title: 'CloudSploit Scans',
                url: 'https://github.com/aquasecurity/cloudsploit',
                description: 'Aqua Security의 클라우드 설정 검증 도구'
              },
              {
                title: 'Wiz Threat Research',
                url: 'https://www.wiz.io/blog',
                description: '클라우드 보안 위협 분석 블로그 (최신 CVE, 공격 기법)'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
