'use client';

import React from 'react';
import { GitBranch, Rocket, CheckCircle, XCircle, AlertCircle, Zap, Shield } from 'lucide-react';

export default function Chapter6() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-2xl p-8 mb-8 border border-green-200 dark:border-green-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center">
            <GitBranch className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">CI/CD 파이프라인 구축</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          GitHub Actions, GitLab CI, Jenkins를 활용한 엔터프라이즈급 자동화 파이프라인
        </p>
      </div>

      {/* CI/CD Fundamentals */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4">CI/CD 핵심 개념</h2>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3 text-blue-600">Continuous Integration (CI)</h3>
            <p className="mb-4">코드 변경사항을 자동으로 빌드하고 테스트하여 메인 브랜치에 통합</p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>코드 커밋 시 자동 빌드/테스트</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>정적 분석, 린트, 유닛 테스트 실행</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>조기 버그 발견 및 빠른 피드백</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>병합 충돌 최소화</span>
              </li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3 text-green-600">Continuous Deployment/Delivery (CD)</h3>
            <p className="mb-4">검증된 코드를 자동으로 프로덕션에 배포 (또는 배포 준비 완료)</p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span><strong>Continuous Delivery</strong>: 배포 버튼 하나로 릴리스</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span><strong>Continuous Deployment</strong>: 완전 자동 배포</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>통합 테스트, E2E 테스트 자동화</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={16} />
                <span>빠른 릴리스 주기 (1일 수십 번 배포 가능)</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
          <h3 className="text-lg font-bold mb-3">CI/CD 파이프라인 단계</h3>
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg font-semibold text-sm whitespace-nowrap">1. Commit</div>
            <span className="text-gray-400">→</span>
            <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg font-semibold text-sm whitespace-nowrap">2. Build</div>
            <span className="text-gray-400">→</span>
            <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg font-semibold text-sm whitespace-nowrap">3. Test</div>
            <span className="text-gray-400">→</span>
            <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg font-semibold text-sm whitespace-nowrap">4. Security Scan</div>
            <span className="text-gray-400">→</span>
            <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg font-semibold text-sm whitespace-nowrap">5. Package</div>
            <span className="text-gray-400">→</span>
            <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg font-semibold text-sm whitespace-nowrap">6. Deploy</div>
          </div>
        </div>
      </section>

      {/* GitHub Actions */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Rocket className="text-purple-600" />
          GitHub Actions - 완전한 CI/CD 워크플로우
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Node.js 애플리케이션 CI/CD 파이프라인</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:  # 수동 실행 가능

env:
  NODE_VERSION: '18.x'
  REGISTRY: ghcr.io
  IMAGE_NAME: $\{{ github.repository }}

jobs:
  # Job 1: 코드 품질 검사
  lint-and-test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: $\{{ env.NODE_VERSION }}
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run linter
      run: npm run lint

    - name: Run unit tests
      run: npm test -- --coverage

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage/lcov.info
        fail_ci_if_error: true

  # Job 2: 보안 스캔
  security-scan:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy results to GitHub Security
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

    - name: Check for npm vulnerabilities
      run: npm audit --audit-level=high

  # Job 3: Docker 이미지 빌드 및 푸시
  build-and-push:
    needs: [lint-and-test, security-scan]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to GitHub Container Registry
      uses: docker/login-action@v3
      with:
        registry: $\{{ env.REGISTRY }}
        username: $\{{ github.actor }}
        password: $\{{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata (tags, labels)
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: $\{{ env.REGISTRY }}/$\{{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: $\{{ steps.meta.outputs.tags }}
        labels: $\{{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # Job 4: Kubernetes 배포 (main 브랜치만)
  deploy-to-k8s:
    needs: build-and-push
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://app.example.com

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: $\{{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: $\{{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --name production-cluster --region us-west-2

    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/app \\
          app=$\{{ env.REGISTRY }}/$\{{ env.IMAGE_NAME }}:main \\
          -n production

        kubectl rollout status deployment/app -n production --timeout=5m

    - name: Run smoke tests
      run: |
        kubectl run smoke-test --image=curlimages/curl:latest \\
          --rm -i --restart=Never -- \\
          curl -f https://app.example.com/health || exit 1

    - name: Notify Slack on success
      if: success()
      uses: slackapi/slack-github-action@v1
      with:
        webhook-url: $\{{ secrets.SLACK_WEBHOOK }}
        payload: |
          {
            "text": "🚀 Deployment to production succeeded!",
            "blocks": [
              {
                "type": "section",
                "text": {
                  "type": "mrkdwn",
                  "text": "*Deployment Status: SUCCESS* ✅\\n*Commit:* $\{{ github.sha }}\\n*Author:* $\{{ github.actor }}"
                }
              }
            ]
          }

    - name: Notify Slack on failure
      if: failure()
      uses: slackapi/slack-github-action@v1
      with:
        webhook-url: $\{{ secrets.SLACK_WEBHOOK }}
        payload: |
          {
            "text": "❌ Deployment to production failed!",
            "blocks": [
              {
                "type": "section",
                "text": {
                  "type": "mrkdwn",
                  "text": "*Deployment Status: FAILED* ❌\\n*Commit:* $\{{ github.sha }}\\n*Author:* $\{{ github.actor }}"
                }
              }
            ]
          }`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">GitHub Actions 고급 기능</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">Matrix Strategy</h4>
              <p className="text-sm mb-2">여러 버전/환경에서 동시 테스트</p>
              <div className="bg-gray-900 text-gray-100 p-2 rounded text-xs">
{`strategy:
  matrix:
    node: [16, 18, 20]
    os: [ubuntu, windows, macos]
runs-on: $\{{ matrix.os }}-latest`}
              </div>
            </div>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">Reusable Workflows</h4>
              <p className="text-sm mb-2">공통 워크플로우 재사용</p>
              <div className="bg-gray-900 text-gray-100 p-2 rounded text-xs">
{`jobs:
  call-workflow:
    uses: org/repo/.github/workflows/test.yml@main
    with:
      node-version: 18`}
              </div>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">Environments</h4>
              <p className="text-sm mb-2">배포 승인 및 시크릿 관리</p>
              <div className="bg-gray-900 text-gray-100 p-2 rounded text-xs">
{`environment:
  name: production
  url: https://app.com
# 수동 승인 필요 설정 가능`}
              </div>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h4 className="font-bold mb-2">Caching</h4>
              <p className="text-sm mb-2">의존성 캐싱으로 빌드 가속</p>
              <div className="bg-gray-900 text-gray-100 p-2 rounded text-xs">
{`- uses: actions/cache@v3
  with:
    path: ~/.npm
    key: $\{{ runner.os }}-node-$\{{ hashFiles('**/package-lock.json') }}`}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* GitLab CI/CD */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Zap className="text-orange-600" />
          GitLab CI/CD - 통합 DevOps 플랫폼
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">GitLab CI/CD 파이프라인</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# .gitlab-ci.yml
stages:
  - build
  - test
  - security
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

# 빌드 Job
build:
  stage: build
  image: docker:24-dind
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $IMAGE_TAG .
    - docker push $IMAGE_TAG
  only:
    - main
    - develop
  tags:
    - docker

# 유닛 테스트
unit-test:
  stage: test
  image: node:18
  cache:
    key: $CI_COMMIT_REF_SLUG
    paths:
      - node_modules/
  script:
    - npm ci
    - npm run test:unit -- --coverage
  coverage: '/Lines\\s*:\\s*(\\d+\\.\\d+)%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml
  only:
    - merge_requests
    - main

# E2E 테스트
e2e-test:
  stage: test
  image: cypress/browsers:node18.12.0-chrome107
  services:
    - name: postgres:15
      alias: db
  variables:
    DATABASE_URL: "postgresql://user:pass@db:5432/testdb"
  script:
    - npm ci
    - npm run db:migrate
    - npm run test:e2e
  artifacts:
    when: on_failure
    paths:
      - cypress/screenshots/
      - cypress/videos/
    expire_in: 1 week
  only:
    - merge_requests

# SAST (Static Application Security Testing)
sast:
  stage: security
  image: returntocorp/semgrep
  script:
    - semgrep --config=auto --json -o sast-report.json
  artifacts:
    reports:
      sast: sast-report.json
  allow_failure: true

# Container Scanning
container-scan:
  stage: security
  image:
    name: aquasec/trivy:latest
    entrypoint: [""]
  script:
    - trivy image --exit-code 1 --severity HIGH,CRITICAL $IMAGE_TAG
  dependencies:
    - build
  allow_failure: false

# Production 배포 (수동 승인)
deploy-production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://app.example.com
    on_stop: stop-production
  before_script:
    - kubectl config set-cluster k8s --server="$KUBE_URL" --insecure-skip-tls-verify=true
    - kubectl config set-credentials admin --token="$KUBE_TOKEN"
    - kubectl config set-context default --cluster=k8s --user=admin
    - kubectl config use-context default
  script:
    - kubectl set image deployment/app app=$IMAGE_TAG -n production
    - kubectl rollout status deployment/app -n production --timeout=5m
  when: manual
  only:
    - main

# Rollback Job
stop-production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    action: stop
  script:
    - kubectl rollout undo deployment/app -n production
  when: manual
  only:
    - main`}
            </pre>
          </div>
        </div>
      </section>

      {/* Jenkins Pipeline */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4">Jenkins - 확장 가능한 자동화 서버</h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">Jenkinsfile (Declarative Pipeline)</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`// Jenkinsfile
pipeline {
    agent {
        kubernetes {
            yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: docker
    image: docker:24-dind
    securityContext:
      privileged: true
  - name: kubectl
    image: bitnami/kubectl:latest
    command: ['sleep']
    args: ['infinity']
'''
        }
    }

    environment {
        DOCKER_REGISTRY = 'docker.io'
        IMAGE_NAME = 'myorg/myapp'
        KUBE_NAMESPACE = 'production'
    }

    parameters {
        choice(name: 'DEPLOY_ENV', choices: ['dev', 'staging', 'production'], description: 'Deployment environment')
        booleanParam(name: 'RUN_SMOKE_TESTS', defaultValue: true, description: 'Run smoke tests after deployment')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(returnStdout: true, script: 'git rev-parse --short HEAD').trim()
                }
            }
        }

        stage('Build') {
            steps {
                container('docker') {
                    sh '''
                        docker build -t \${DOCKER_REGISTRY}/\${IMAGE_NAME}:\${GIT_COMMIT_SHORT} .
                        docker tag \${DOCKER_REGISTRY}/\${IMAGE_NAME}:\${GIT_COMMIT_SHORT} \\
                                   \${DOCKER_REGISTRY}/\${IMAGE_NAME}:latest
                    '''
                }
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'npm run test:unit'
                    }
                    post {
                        always {
                            junit 'test-results/junit.xml'
                            publishCoverage adapters: [coberturaAdapter('coverage/cobertura-coverage.xml')]
                        }
                    }
                }

                stage('Integration Tests') {
                    steps {
                        sh 'npm run test:integration'
                    }
                }

                stage('Security Scan') {
                    steps {
                        container('docker') {
                            sh '''
                                trivy image --severity HIGH,CRITICAL \\
                                  \${DOCKER_REGISTRY}/\${IMAGE_NAME}:\${GIT_COMMIT_SHORT}
                            '''
                        }
                    }
                }
            }
        }

        stage('Push Image') {
            when {
                branch 'main'
            }
            steps {
                container('docker') {
                    withCredentials([usernamePassword(
                        credentialsId: 'docker-registry-creds',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh '''
                            echo "\$DOCKER_PASS" | docker login -u "\$DOCKER_USER" --password-stdin \${DOCKER_REGISTRY}
                            docker push \${DOCKER_REGISTRY}/\${IMAGE_NAME}:\${GIT_COMMIT_SHORT}
                            docker push \${DOCKER_REGISTRY}/\${IMAGE_NAME}:latest
                        '''
                    }
                }
            }
        }

        stage('Deploy') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                script {
                    timeout(time: 15, unit: 'MINUTES') {
                        input message: "Deploy to \${params.DEPLOY_ENV}?", ok: 'Deploy'
                    }
                }

                container('kubectl') {
                    withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                        sh """
                            kubectl set image deployment/myapp \\
                              app=\${DOCKER_REGISTRY}/\${IMAGE_NAME}:\${GIT_COMMIT_SHORT} \\
                              -n \${KUBE_NAMESPACE}

                            kubectl rollout status deployment/myapp -n \${KUBE_NAMESPACE} --timeout=5m
                        """
                    }
                }
            }
        }

        stage('Smoke Tests') {
            when {
                expression { params.RUN_SMOKE_TESTS }
            }
            steps {
                sh 'npm run test:smoke'
            }
        }
    }

    post {
        success {
            slackSend(
                color: 'good',
                message: "✅ Pipeline SUCCESS: \${env.JOB_NAME} #\${env.BUILD_NUMBER}\\nCommit: \${env.GIT_COMMIT_SHORT}"
            )
        }
        failure {
            slackSend(
                color: 'danger',
                message: "❌ Pipeline FAILED: \${env.JOB_NAME} #\${env.BUILD_NUMBER}\\nCommit: \${env.GIT_COMMIT_SHORT}"
            )
        }
        always {
            cleanWs()
        }
    }
}`}
            </pre>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Shield className="text-indigo-600" />
          CI/CD 모범 사례
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-bold mb-1">빠른 피드백 루프</h3>
                <p className="text-sm">빌드는 10분 이내, 가장 중요한 테스트를 먼저 실행</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-bold mb-1">Branch Protection</h3>
                <p className="text-sm">main 브랜치는 CI 통과 + 코드 리뷰 필수</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-bold mb-1">시크릿 관리</h3>
                <p className="text-sm">환경 변수로 시크릿 관리, 코드에 하드코딩 금지</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-bold mb-1">멱등성 (Idempotency)</h3>
                <p className="text-sm">파이프라인을 여러 번 실행해도 동일한 결과</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-bold mb-1">Artifacts 보관</h3>
                <p className="text-sm">빌드 결과물, 테스트 리포트, 로그 저장</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-bold mb-1">Rollback 전략</h3>
                <p className="text-sm">배포 실패 시 이전 버전으로 즉시 롤백 가능</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Deployment Strategies */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4">배포 전략</h2>

        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3 text-blue-600">Rolling Deployment (무중단 배포)</h3>
            <p className="mb-3">Pod을 하나씩 순차적으로 업데이트 (Kubernetes 기본 전략)</p>
            <ul className="space-y-1 text-sm">
              <li>✅ 간단하고 안정적</li>
              <li>✅ 리소스 효율적</li>
              <li>⚠️ 구버전/신버전이 동시 실행되는 시간 존재</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3 text-green-600">Blue-Green Deployment</h3>
            <p className="mb-3">새 버전을 완전히 준비한 후 트래픽을 한 번에 전환</p>
            <ul className="space-y-1 text-sm">
              <li>✅ 즉시 롤백 가능 (트래픽만 다시 전환)</li>
              <li>✅ 구버전/신버전 혼재 없음</li>
              <li>⚠️ 2배의 리소스 필요</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <h3 className="text-xl font-bold mb-3 text-purple-600">Canary Deployment</h3>
            <p className="mb-3">일부 사용자에게만 신버전 노출 후 점진적 확대</p>
            <ul className="space-y-1 text-sm">
              <li>✅ 위험 최소화 (5% → 25% → 50% → 100%)</li>
              <li>✅ A/B 테스트 가능</li>
              <li>⚠️ 복잡한 트래픽 관리 필요 (Istio, Flagger)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Next Steps */}
      <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-2">다음 단계: GitOps 배포</h3>
        <p className="text-gray-700 dark:text-gray-300">
          ArgoCD, Flux를 활용한 Git 기반 선언적 배포 및 자동 동기화 전략
        </p>
      </div>
    </div>
  );
}