'use client';

import React from 'react';
import { Server, Shield, Gauge } from 'lucide-react';

export default function Chapter5() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-2xl p-8 mb-8 border border-purple-200 dark:border-purple-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center">
            <Server className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Kubernetes 고급</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          StatefulSet, Ingress, HPA, RBAC - 프로덕션급 운영
        </p>
      </div>

      <section className="my-8">
        <h2>StatefulSet & DaemonSet</h2>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# StatefulSet for databases
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 3
  template:
    spec:
      containers:
      - name: postgres
        image: postgres:15
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi`}
            </pre>
          </div>
        </div>
      </section>

      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Gauge className="text-orange-600" />
          HPA - Auto Scaling
        </h2>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

kubectl autoscale deployment app --cpu-percent=70 --min=2 --max=10`}
            </pre>
          </div>
        </div>
      </section>

      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Shield className="text-red-600" />
          RBAC - Role Based Access Control
        </h2>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Role for pod reader
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]

# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
subjects:
- kind: User
  name: jane
roleRef:
  kind: Role
  name: pod-reader`}
            </pre>
          </div>
        </div>
      </section>

      <div className="bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-500 p-6 rounded-lg">
        <h3>다음: CI/CD 파이프라인</h3>
        <p>자동화된 빌드/테스트/배포</p>
      </div>
    </div>
  )
}