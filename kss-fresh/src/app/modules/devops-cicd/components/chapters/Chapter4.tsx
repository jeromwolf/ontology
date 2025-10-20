'use client';

import React from 'react';
import { Cpu, Box, Network, HardDrive, CheckCircle, AlertCircle } from 'lucide-react';

export default function Chapter4() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl p-8 mb-8 border border-blue-200 dark:border-blue-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Kubernetes 기초</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          컨테이너 오케스트레이션의 사실상 표준, Kubernetes 아키텍처와 핵심 개념
        </p>
      </div>

      {/* Kubernetes Architecture */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Network className="text-blue-600" />
          Kubernetes 아키텍처
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Control Plane (Master Node)</h3>
          <div className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">kube-apiserver</h4>
              <p className="text-sm">모든 API 요청을 처리하는 중앙 허브. kubectl, CI/CD 도구가 통신하는 엔드포인트</p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">etcd</h4>
              <p className="text-sm">클러스터의 모든 상태 정보를 저장하는 분산 키-값 저장소 (백업 필수!)</p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">kube-scheduler</h4>
              <p className="text-sm">Pod을 어느 Worker Node에 배치할지 결정 (리소스, 친화성, 안티-친화성 고려)</p>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">kube-controller-manager</h4>
              <p className="text-sm">Deployment, ReplicaSet 등의 컨트롤러를 실행하여 원하는 상태 유지</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">Worker Node</h3>
          <div className="space-y-4">
            <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">kubelet</h4>
              <p className="text-sm">각 노드에서 실행되는 에이전트. Pod 생명주기 관리 및 컨테이너 런타임과 통신</p>
            </div>
            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">kube-proxy</h4>
              <p className="text-sm">네트워크 프록시. Service의 로드밸런싱 및 네트워크 규칙 관리</p>
            </div>
            <div className="bg-teal-50 dark:bg-teal-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">Container Runtime (containerd/CRI-O)</h4>
              <p className="text-sm">실제로 컨테이너를 실행하는 런타임 (Docker는 deprecated)</p>
            </div>
          </div>
        </div>
      </section>

      {/* Core Objects */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Box className="text-green-600" />
          핵심 오브젝트
        </h2>

        <div className="space-y-6">
          {/* Pod */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3">Pod - Kubernetes의 최소 배포 단위</h3>
            <p className="mb-4">하나 이상의 컨테이너를 묶은 그룹. 같은 Pod 내 컨테이너는 IP, 볼륨, 네임스페이스 공유</p>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
              <pre className="text-sm">
{`# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.25
    ports:
    - containerPort: 80
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"

# Pod 생성
kubectl apply -f pod.yaml

# Pod 확인
kubectl get pods
kubectl describe pod nginx-pod
kubectl logs nginx-pod`}
              </pre>
            </div>
          </div>

          {/* Deployment */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3">Deployment - 선언적 배포 관리</h3>
            <p className="mb-4">ReplicaSet을 관리하며 롤링 업데이트, 롤백 기능 제공</p>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
              <pre className="text-sm">
{`# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3  # 3개의 Pod 유지
  selector:
    matchLabels:
      app: nginx
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # 동시에 추가 가능한 최대 Pod 수
      maxUnavailable: 1  # 업데이트 중 비가용 최대 Pod 수
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80

# 배포
kubectl apply -f deployment.yaml

# 롤링 업데이트
kubectl set image deployment/nginx-deployment nginx=nginx:1.26

# 롤백
kubectl rollout undo deployment/nginx-deployment

# 배포 상태 확인
kubectl rollout status deployment/nginx-deployment`}
              </pre>
            </div>
          </div>

          {/* Service */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3">Service - Pod 네트워크 노출</h3>
            <p className="mb-4">Pod의 동적 IP 문제를 해결하는 고정 엔드포인트 + 로드밸런싱</p>
            <div className="grid md:grid-cols-3 gap-4 mb-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
                <h4 className="font-bold text-sm mb-1">ClusterIP (기본)</h4>
                <p className="text-xs">클러스터 내부에서만 접근</p>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
                <h4 className="font-bold text-sm mb-1">NodePort</h4>
                <p className="text-xs">모든 노드의 특정 포트로 노출</p>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
                <h4 className="font-bold text-sm mb-1">LoadBalancer</h4>
                <p className="text-xs">클라우드 LB 자동 프로비저닝</p>
              </div>
            </div>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
              <pre className="text-sm">
{`# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  type: LoadBalancer  # ClusterIP, NodePort, LoadBalancer
  selector:
    app: nginx  # 이 라벨을 가진 Pod들에게 트래픽 전달
  ports:
  - protocol: TCP
    port: 80        # Service 포트
    targetPort: 80  # Pod 컨테이너 포트

# Service 생성
kubectl apply -f service.yaml

# Service 확인
kubectl get svc
kubectl describe svc nginx-service

# Service 엔드포인트 확인 (연결된 Pod IP들)
kubectl get endpoints nginx-service`}
              </pre>
            </div>
          </div>

          {/* ConfigMap & Secret */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3">ConfigMap & Secret - 설정 관리</h3>
            <div className="grid md:grid-cols-2 gap-4 mb-4">
              <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded">
                <h4 className="font-bold mb-2">ConfigMap</h4>
                <p className="text-sm">환경 변수, 설정 파일 등 일반 데이터 저장</p>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded">
                <h4 className="font-bold mb-2">Secret</h4>
                <p className="text-sm">비밀번호, API 키 등 민감 데이터 저장 (Base64 인코딩)</p>
              </div>
            </div>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
              <pre className="text-sm">
{`# ConfigMap 생성
kubectl create configmap app-config \\
  --from-literal=DATABASE_URL=postgres://db:5432 \\
  --from-file=config.json

# Secret 생성
kubectl create secret generic db-secret \\
  --from-literal=password=myP@ssw0rd

# Pod에서 사용
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:1.0
    env:
    - name: DATABASE_URL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: DATABASE_URL
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* kubectl Commands */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4">⚡ 필수 kubectl 명령어</h2>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-bold mb-3 text-blue-600">조회 명령어</h3>
              <div className="bg-gray-900 text-gray-100 p-3 rounded text-xs space-y-1">
                <div><span className="text-green-400"># 전체 리소스 확인</span></div>
                <div>kubectl get all</div>
                <div><span className="text-green-400"># Pod 상태 모니터링</span></div>
                <div>kubectl get pods -w</div>
                <div><span className="text-green-400"># 상세 정보</span></div>
                <div>kubectl describe pod nginx-pod</div>
                <div><span className="text-green-400"># 로그 확인</span></div>
                <div>kubectl logs -f nginx-pod</div>
              </div>
            </div>
            <div>
              <h3 className="font-bold mb-3 text-green-600">실행 명령어</h3>
              <div className="bg-gray-900 text-gray-100 p-3 rounded text-xs space-y-1">
                <div><span className="text-green-400"># Pod 내부 셸 접속</span></div>
                <div>kubectl exec -it nginx-pod -- /bin/bash</div>
                <div><span className="text-green-400"># 포트 포워딩</span></div>
                <div>kubectl port-forward pod/nginx-pod 8080:80</div>
                <div><span className="text-green-400"># 리소스 삭제</span></div>
                <div>kubectl delete deployment nginx-deployment</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4">💡 Kubernetes 모범 사례</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">리소스 제한 설정</h3>
                <p className="text-sm">requests/limits를 통해 CPU/메모리 보장 및 제한</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">Liveness/Readiness Probe</h3>
                <p className="text-sm">컨테이너 헬스체크로 자동 재시작 및 트래픽 제어</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">Namespace 분리</h3>
                <p className="text-sm">dev/staging/prod 환경 또는 팀별 격리</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">Label 체계화</h3>
                <p className="text-sm">app, environment, version 등 일관된 라벨링</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Next Steps */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-2">다음 단계: Kubernetes 고급</h3>
        <p className="text-gray-700 dark:text-gray-300">
          StatefulSet, DaemonSet, Ingress, HPA(Auto Scaling), RBAC 권한 관리 등
          고급 기능과 프로덕션 배포 전략을 학습합니다.
        </p>
      </div>
    </div>
  );
}
