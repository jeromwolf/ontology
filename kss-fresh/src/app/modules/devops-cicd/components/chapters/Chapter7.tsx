'use client';

import React from 'react';
import { GitBranch, GitMerge, Lock, Shield, Zap, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react';

export default function Chapter7() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-indigo-50 to-cyan-50 dark:from-indigo-900/20 dark:to-cyan-900/20 rounded-2xl p-8 mb-8 border border-indigo-200 dark:border-indigo-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-indigo-500 rounded-xl flex items-center justify-center">
            <GitBranch className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">GitOps와 배포 전략</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          Git을 Single Source of Truth로 활용하는 GitOps 원칙과 ArgoCD, Flux를 통한 선언적 배포 자동화
        </p>
      </div>

      {/* GitOps Fundamentals */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <GitBranch className="text-indigo-600" />
          GitOps란 무엇인가?
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">GitOps 4대 원칙</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
              <h4 className="font-bold mb-2">1️⃣ Declarative (선언적)</h4>
              <p className="text-sm">전체 시스템이 선언적으로 정의되어야 합니다. Kubernetes YAML, Terraform HCL 등 원하는 상태를 기술합니다.</p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
              <h4 className="font-bold mb-2">2️⃣ Versioned (버전 관리)</h4>
              <p className="text-sm">원하는 상태가 Git에 저장되고 버전 관리됩니다. 모든 변경 사항이 커밋 히스토리로 추적 가능합니다.</p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-l-4 border-purple-500">
              <h4 className="font-bold mb-2">3️⃣ Pulled Automatically (자동 동기화)</h4>
              <p className="text-sm">GitOps 에이전트가 Git 저장소를 지속적으로 모니터링하고 클러스터에 자동 반영합니다.</p>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-l-4 border-orange-500">
              <h4 className="font-bold mb-2">4️⃣ Continuously Reconciled (지속적 조정)</h4>
              <p className="text-sm">Git 상태와 실제 환경의 차이(drift)를 감지하고 자동으로 동기화합니다.</p>
            </div>
          </div>
        </div>

        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-lg mb-6">
          <h3 className="font-bold mb-3">GitOps vs Traditional CI/CD</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-red-600 mb-2">❌ Traditional Push-Based CD</h4>
              <ul className="text-sm space-y-1">
                <li>• CI 도구가 직접 클러스터에 배포 (kubectl, helm)</li>
                <li>• 클러스터 자격증명을 CI/CD에 저장</li>
                <li>• 수동 개입 필요 (승인, 롤백)</li>
                <li>• Git과 클러스터 상태 불일치 가능</li>
                <li>• 보안 위험 (외부에서 클러스터 접근)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-green-600 mb-2">✅ GitOps Pull-Based CD</h4>
              <ul className="text-sm space-y-1">
                <li>• 클러스터 내 에이전트가 Git을 주기적으로 pull</li>
                <li>• 자격증명이 클러스터 내부에만 존재</li>
                <li>• 완전 자동화 (self-healing)</li>
                <li>• Git = Single Source of Truth (항상 동기화)</li>
                <li>• 향상된 보안 (외부 접근 불필요)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ArgoCD */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Zap className="text-cyan-600" />
          ArgoCD - GitOps Continuous Delivery
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">ArgoCD 설치 및 설정</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# ArgoCD 설치 (Kubernetes)
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# ArgoCD Server 외부 노출 (LoadBalancer)
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'

# 초기 admin 비밀번호 확인
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# ArgoCD CLI 로그인
argocd login <ARGOCD_SERVER>

# 비밀번호 변경
argocd account update-password`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Application 정의 및 배포</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">
            <pre className="text-sm">
{`# application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-production
  namespace: argocd
spec:
  project: default

  # Git 저장소 설정
  source:
    repoURL: https://github.com/myorg/k8s-manifests
    targetRevision: main
    path: apps/myapp/overlays/production

    # Kustomize 사용 (또는 helm, directory)
    kustomize:
      images:
      - myapp=myregistry/myapp:v1.2.3

  # 배포 대상 클러스터
  destination:
    server: https://kubernetes.default.svc
    namespace: production

  # 동기화 정책
  syncPolicy:
    automated:
      prune: true       # Git에서 삭제된 리소스 자동 제거
      selfHeal: true    # 클러스터에서 수동 변경 시 자동 복구
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

# Application 생성
kubectl apply -f application.yaml

# 또는 CLI로 생성
argocd app create myapp-production \\
  --repo https://github.com/myorg/k8s-manifests \\
  --path apps/myapp/overlays/production \\
  --dest-server https://kubernetes.default.svc \\
  --dest-namespace production \\
  --sync-policy automated \\
  --auto-prune \\
  --self-heal`}
            </pre>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-cyan-50 dark:bg-cyan-900/20 p-3 rounded">
              <h4 className="font-bold text-sm mb-1">Sync Policy</h4>
              <p className="text-xs">Git 변경 감지 시 자동 동기화 여부 설정</p>
            </div>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
              <h4 className="font-bold text-sm mb-1">Prune</h4>
              <p className="text-xs">Git에서 제거된 리소스를 클러스터에서도 삭제</p>
            </div>
            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-3 rounded">
              <h4 className="font-bold text-sm mb-1">Self Heal</h4>
              <p className="text-xs">수동 변경 감지 시 Git 상태로 자동 복구</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">필수 CLI 명령어</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Application 목록 확인
argocd app list

# Application 상태 확인 (동기화 상태, 헬스 상태)
argocd app get myapp-production

# 수동 동기화
argocd app sync myapp-production

# 특정 리소스만 동기화
argocd app sync myapp-production --resource apps:Deployment:myapp

# 동기화 히스토리 확인
argocd app history myapp-production

# 이전 버전으로 롤백
argocd app rollback myapp-production 5

# Application 삭제 (클러스터 리소스도 함께 삭제)
argocd app delete myapp-production --cascade

# Diff 확인 (Git vs 클러스터)
argocd app diff myapp-production

# 실시간 로그 확인
argocd app logs myapp-production -f`}
            </pre>
          </div>
        </div>
      </section>

      {/* Flux CD */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <RefreshCw className="text-purple-600" />
          Flux CD - GitOps Toolkit
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Flux 설치 및 부트스트랩</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Flux CLI 설치 (macOS)
brew install fluxcd/tap/flux

# GitHub Personal Access Token 설정
export GITHUB_TOKEN=<your-token>

# Flux 부트스트랩 (자동으로 Git 저장소 생성 및 Flux 설치)
flux bootstrap github \\
  --owner=myorg \\
  --repository=fleet-infra \\
  --branch=main \\
  --path=clusters/production \\
  --personal

# 설치 확인
flux check

# Git 저장소 상태 확인
flux get sources git`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">GitRepository & Kustomization 정의</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# 1. GitRepository - Git 저장소 연결
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: myapp
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/myorg/myapp-k8s
  ref:
    branch: main
  secretRef:
    name: github-credentials

---
# 2. Kustomization - 배포 정의
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: myapp-production
  namespace: flux-system
spec:
  interval: 5m
  path: ./apps/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: myapp
  healthChecks:
  - apiVersion: apps/v1
    kind: Deployment
    name: myapp
    namespace: production
  timeout: 2m

# 적용
kubectl apply -f flux-kustomization.yaml

# 상태 확인
flux get kustomizations
flux logs --kind=Kustomization --name=myapp-production`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">HelmRelease - Helm Chart 배포</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# HelmRepository 정의
apiVersion: source.toolkit.fluxcd.io/v1beta2
kind: HelmRepository
metadata:
  name: bitnami
  namespace: flux-system
spec:
  interval: 1h
  url: https://charts.bitnami.com/bitnami

---
# HelmRelease 정의
apiVersion: helm.toolkit.fluxcd.io/v2beta1
kind: HelmRelease
metadata:
  name: redis
  namespace: production
spec:
  interval: 5m
  chart:
    spec:
      chart: redis
      version: '17.x'
      sourceRef:
        kind: HelmRepository
        name: bitnami
        namespace: flux-system
  values:
    auth:
      enabled: true
      password: \${REDIS_PASSWORD}
    master:
      persistence:
        size: 10Gi
  valuesFrom:
  - kind: Secret
    name: redis-values
    valuesKey: values.yaml

# 적용 및 확인
kubectl apply -f helmrelease.yaml
flux get helmreleases -n production`}
            </pre>
          </div>
        </div>
      </section>

      {/* Repository Structure */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <GitMerge className="text-green-600" />
          GitOps 저장소 구조
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">권장 디렉토리 구조 (Kustomize 기반)</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`k8s-manifests/
├── base/                          # 공통 베이스 매니페스트
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── kustomization.yaml
│
├── overlays/
│   ├── development/               # 개발 환경
│   │   ├── kustomization.yaml
│   │   ├── patch-replicas.yaml    # replicas: 1
│   │   └── patch-resources.yaml   # 낮은 리소스 요청
│   │
│   ├── staging/                   # 스테이징 환경
│   │   ├── kustomization.yaml
│   │   ├── patch-replicas.yaml    # replicas: 2
│   │   └── ingress.yaml
│   │
│   └── production/                # 프로덕션 환경
│       ├── kustomization.yaml
│       ├── patch-replicas.yaml    # replicas: 5
│       ├── patch-resources.yaml   # 높은 리소스 요청/제한
│       ├── hpa.yaml               # Horizontal Pod Autoscaler
│       └── pdb.yaml               # Pod Disruption Budget
│
└── apps/
    ├── app1/
    │   └── (위와 동일한 구조)
    └── app2/
        └── (위와 동일한 구조)

# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- deployment.yaml
- service.yaml
- configmap.yaml

# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: production
bases:
- ../../base
patchesStrategicMerge:
- patch-replicas.yaml
- patch-resources.yaml
resources:
- hpa.yaml
- pdb.yaml
images:
- name: myapp
  newTag: v1.2.3`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">App of Apps 패턴 (ArgoCD)</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Root Application (부모 App)
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: root-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/argocd-apps
    targetRevision: main
    path: apps
  destination:
    server: https://kubernetes.default.svc
    namespace: argocd
  syncPolicy:
    automated:
      prune: true
      selfHeal: true

# apps/ 디렉토리 구조
apps/
├── app1.yaml          # 자식 Application 정의
├── app2.yaml
├── monitoring.yaml
└── ingress-nginx.yaml

# apps/app1.yaml (자식 App)
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: app1-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/k8s-manifests
    path: apps/app1/overlays/production
  destination:
    namespace: production
  syncPolicy:
    automated: {}

# 장점: 하나의 root-app만 생성하면
# 모든 자식 애플리케이션이 자동으로 관리됨`}
            </pre>
          </div>
        </div>
      </section>

      {/* Secret Management */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Lock className="text-red-600" />
          GitOps 환경에서의 Secret 관리
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Sealed Secrets (Bitnami)</h3>
          <p className="mb-4">평문 Secret을 암호화하여 Git에 안전하게 저장 가능</p>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Sealed Secrets Controller 설치
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# kubeseal CLI 설치 (macOS)
brew install kubeseal

# 일반 Secret 생성
kubectl create secret generic mysecret \\
  --from-literal=password=myP@ssw0rd \\
  --dry-run=client -o yaml > mysecret.yaml

# Sealed Secret으로 암호화 (공개키 사용)
kubeseal -f mysecret.yaml -w mysealedsecret.yaml

# mysealedsecret.yaml (암호화됨, Git에 커밋 가능!)
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: mysecret
  namespace: production
spec:
  encryptedData:
    password: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEq...

# Git에 커밋
git add mysealedsecret.yaml
git commit -m "Add sealed secret"
git push

# 클러스터에 배포되면 자동으로 복호화되어 Secret 생성됨
kubectl get secret mysecret -n production -o yaml`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">External Secrets Operator (ESO)</h3>
          <p className="mb-4">AWS Secrets Manager, HashiCorp Vault 등 외부 Secret Store 연동</p>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# External Secrets Operator 설치
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace

# SecretStore 정의 (AWS Secrets Manager)
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets
  namespace: production
spec:
  provider:
    aws:
      service: SecretsManager
      region: ap-northeast-2
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa

---
# ExternalSecret 정의
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets
    kind: SecretStore
  target:
    name: db-secret
    creationPolicy: Owner
  data:
  - secretKey: username
    remoteRef:
      key: prod/database/credentials
      property: username
  - secretKey: password
    remoteRef:
      key: prod/database/credentials
      property: password

# 자동으로 Kubernetes Secret이 생성됨
kubectl get secret db-secret -n production`}
            </pre>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h4 className="font-bold mb-2">Sealed Secrets</h4>
            <ul className="text-sm space-y-1">
              <li className="flex items-start gap-1">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span>간단한 설정</span>
              </li>
              <li className="flex items-start gap-1">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span>Git 친화적</span>
              </li>
              <li className="flex items-start gap-1">
                <AlertCircle className="text-orange-500 flex-shrink-0 mt-0.5" size={16} />
                <span>Secret 회전 수동</span>
              </li>
            </ul>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h4 className="font-bold mb-2">External Secrets</h4>
            <ul className="text-sm space-y-1">
              <li className="flex items-start gap-1">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span>중앙화된 관리</span>
              </li>
              <li className="flex items-start gap-1">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span>자동 Secret 회전</span>
              </li>
              <li className="flex items-start gap-1">
                <AlertCircle className="text-orange-500 flex-shrink-0 mt-0.5" size={16} />
                <span>외부 의존성</span>
              </li>
            </ul>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h4 className="font-bold mb-2">SOPS (Mozilla)</h4>
            <ul className="text-sm space-y-1">
              <li className="flex items-start gap-1">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span>부분 암호화 가능</span>
              </li>
              <li className="flex items-start gap-1">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span>다양한 KMS 지원</span>
              </li>
              <li className="flex items-start gap-1">
                <AlertCircle className="text-orange-500 flex-shrink-0 mt-0.5" size={16} />
                <span>추가 도구 필요</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Progressive Delivery */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Shield className="text-emerald-600" />
          Progressive Delivery with Flagger
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Canary 배포 자동화</h3>
          <p className="mb-4">Flagger는 Istio, Linkerd, NGINX와 통합되어 자동화된 Canary 배포를 제공합니다.</p>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Flagger 설치 (Istio 사용 시)
kubectl apply -k github.com/fluxcd/flagger//kustomize/istio

# Canary 리소스 정의
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: myapp
  namespace: production
spec:
  # 대상 Deployment
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp

  # Service 설정
  service:
    port: 80
    targetPort: 8080
    gateways:
    - public-gateway
    hosts:
    - myapp.example.com

  # Canary 분석 (메트릭 기반 자동 판단)
  analysis:
    interval: 1m
    threshold: 5          # 5회 연속 성공 시 승급
    maxWeight: 50         # 최대 50%까지 트래픽 전환
    stepWeight: 10        # 10%씩 점진적 증가

    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99           # 99% 이상 성공률 유지
      interval: 1m

    - name: request-duration
      thresholdRange:
        max: 500          # 500ms 이하 응답 시간
      interval: 1m

    # 웹훅 알림
    webhooks:
    - name: load-test
      url: http://flagger-loadtester/
      timeout: 5s
      metadata:
        cmd: "hey -z 1m -q 10 -c 2 http://myapp-canary:80/"

    - name: slack-notification
      url: https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# 배포 프로세스:
# 1. 새 버전 이미지 배포 → Canary Pod 생성
# 2. 10% 트래픽 전환 → 1분간 메트릭 수집
# 3. 메트릭 통과 → 20%, 30% ... 50%까지 점진적 증가
# 4. 모든 단계 성공 → 100% 전환 및 이전 버전 종료
# 5. 실패 시 자동 롤백`}
            </pre>
          </div>
        </div>

        <div className="bg-emerald-50 dark:bg-emerald-900/20 p-6 rounded-lg">
          <h3 className="font-bold mb-3">Progressive Delivery 장점</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <ul className="text-sm space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>자동 롤백</strong>: 메트릭 기반으로 문제 감지 시 즉시 롤백</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>리스크 감소</strong>: 소수 사용자에게 먼저 배포하여 영향 최소화</span>
              </li>
            </ul>
            <ul className="text-sm space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>관찰 가능성</strong>: 실시간 메트릭과 알림으로 배포 상태 추적</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>무중단 배포</strong>: Blue-Green보다 리소스 효율적</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4">💡 GitOps 모범 사례</h2>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">저장소 분리</h3>
                <p className="text-sm">애플리케이션 코드와 인프라 매니페스트를 별도 저장소로 관리 (보안, 접근 권한 분리)</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">환경별 브랜치 전략</h3>
                <p className="text-sm">develop → staging → main 브랜치 또는 overlays로 환경 분리</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">Pull Request 워크플로우</h3>
                <p className="text-sm">매니페스트 변경 시 PR을 통한 코드 리뷰 및 자동 검증 (kubeval, conftest)</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">이미지 태그 고정</h3>
                <p className="text-sm">latest 대신 명시적 버전 태그 사용 (v1.2.3 또는 Git SHA)</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">알림 설정</h3>
                <p className="text-sm">동기화 실패, 헬스체크 실패 시 Slack/Teams 알림 연동</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">GitOps 에이전트 고가용성</h3>
                <p className="text-sm">ArgoCD/Flux 컨트롤러를 HA 모드로 운영 (replicas ≥ 2)</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border-l-4 border-orange-500">
          <h3 className="font-bold mb-3 flex items-center gap-2">
            <AlertCircle className="text-orange-500" />
            피해야 할 안티 패턴
          </h3>
          <ul className="space-y-2 text-sm">
            <li>❌ <strong>kubectl apply 직접 실행</strong>: GitOps 원칙 위배, drift 발생</li>
            <li>❌ <strong>Secret을 평문으로 Git에 커밋</strong>: 보안 사고의 주요 원인</li>
            <li>❌ <strong>Self-heal 없이 운영</strong>: 수동 변경 시 Git과 불일치 상태 유지</li>
            <li>❌ <strong>매니페스트 검증 없이 배포</strong>: 잘못된 YAML로 인한 배포 실패</li>
          </ul>
        </div>
      </section>

      {/* Next Steps */}
      <div className="bg-indigo-50 dark:bg-indigo-900/20 border-l-4 border-indigo-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-2">다음 단계: 모니터링 & 보안</h3>
        <p className="text-gray-700 dark:text-gray-300">
          Prometheus/Grafana를 통한 메트릭 수집 및 시각화, 로그 집계(ELK/Loki), 분산 추적(Jaeger),
          그리고 Kubernetes 보안 강화(NetworkPolicy, PSA, Falco 등)를 학습합니다.
        </p>
      </div>
    </div>
  );
}