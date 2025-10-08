'use client';

import React, { useState } from 'react';
import { Rocket, Server, Cloud, Shield, Gauge, AlertTriangle, CheckCircle2 } from 'lucide-react';
import References from '@/components/common/References';

interface Chapter10Props {
  onComplete?: () => void
}

export default function Chapter10({ onComplete }: Chapter10Props) {
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-blue-600 dark:text-blue-400">Chapter 10: 모델 배포 (Model Deployment)</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          ML 모델을 프로덕션 환경에 배포하고 운영하는 방법을 학습합니다
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Rocket className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            모델 배포란?
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">정의</h3>
            <p className="text-gray-600 dark:text-gray-400">
              모델 배포는 학습된 머신러닝 모델을 실제 사용자가 접근할 수 있는 
              프로덕션 환경으로 이동시켜 예측 서비스를 제공하는 과정입니다.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">배포 방식</h4>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li>REST API 서비스</li>
                <li>배치 예측 (Batch Prediction)</li>
                <li>스트리밍 예측 (Real-time)</li>
                <li>엣지 디바이스 배포</li>
                <li>웹/모바일 앱 임베딩</li>
              </ul>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">주요 고려사항</h4>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li>모델 성능 및 지연시간</li>
                <li>확장성 (Scalability)</li>
                <li>모니터링 및 로깅</li>
                <li>버전 관리</li>
                <li>보안 및 접근 제어</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* 탭 네비게이션 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setActiveTab('overview')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === 'overview'
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            개요
          </button>
          <button
            onClick={() => setActiveTab('architecture')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === 'architecture'
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            아키텍처
          </button>
          <button
            onClick={() => setActiveTab('implementation')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === 'implementation'
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            구현
          </button>
          <button
            onClick={() => setActiveTab('mlops')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === 'mlops'
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            MLOps
          </button>
        </div>

        <div className="p-6">
          {activeTab === 'overview' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">배포 프로세스 개요</h3>
              <div className="space-y-4">
                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-semibold">1. 모델 준비</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    학습된 모델을 배포 가능한 형태로 패키징합니다.
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-900 p-3 rounded mt-2">
                    <pre className="text-sm font-mono overflow-x-auto">
{`# 모델 저장 예시
import joblib
import pickle

# scikit-learn 모델
joblib.dump(model, 'model.pkl')

# 딥러닝 모델
model.save('model.h5')  # Keras
torch.save(model.state_dict(), 'model.pth')  # PyTorch`}
                    </pre>
                  </div>
                </div>
                
                <div className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-semibold">2. 환경 설정</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    프로덕션 환경에 필요한 의존성과 인프라를 구성합니다.
                  </p>
                </div>
                
                <div className="border-l-4 border-green-500 pl-4">
                  <h4 className="font-semibold">3. API 개발</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    모델을 서비스할 API 엔드포인트를 개발합니다.
                  </p>
                </div>
                
                <div className="border-l-4 border-yellow-500 pl-4">
                  <h4 className="font-semibold">4. 테스트 및 검증</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    배포된 모델의 성능과 안정성을 검증합니다.
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'architecture' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <Server className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                배포 아키텍처
              </h3>
              <div className="space-y-4">
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <h4 className="font-semibold mb-2">마이크로서비스 아키텍처</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    모델 서빙을 독립적인 서비스로 분리하여 확장성과 유지보수성을 향상시킵니다.
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                      <span className="text-sm">API Gateway: 요청 라우팅 및 인증</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <span className="text-sm">Model Server: 모델 추론 서비스</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      <span className="text-sm">Feature Store: 특징 데이터 관리</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <span className="text-sm">Monitoring: 성능 모니터링</span>
                    </div>
                  </div>
                </div>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <h5 className="font-semibold mb-2 flex items-center gap-2">
                      <Cloud className="w-4 h-4" />
                      클라우드 서비스
                    </h5>
                    <ul className="text-sm space-y-1">
                      <li>• AWS SageMaker</li>
                      <li>• Google Cloud AI Platform</li>
                      <li>• Azure Machine Learning</li>
                      <li>• Hugging Face Endpoints</li>
                    </ul>
                  </div>
                  
                  <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <h5 className="font-semibold mb-2 flex items-center gap-2">
                      <Shield className="w-4 h-4" />
                      보안 고려사항
                    </h5>
                    <ul className="text-sm space-y-1">
                      <li>• API 키 인증</li>
                      <li>• HTTPS 암호화</li>
                      <li>• Rate Limiting</li>
                      <li>• 입력값 검증</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'implementation' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">실제 구현 예제</h3>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">1. FastAPI를 사용한 모델 서빙</h4>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# FastAPI 앱 초기화
app = FastAPI(title="ML Model API", version="1.0")

# 모델 로드
model = joblib.load("model.pkl")

# 입력 데이터 스키마
class PredictionRequest(BaseModel):
    features: list[float]
    
# 출력 데이터 스키마
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 입력 데이터 검증
        features = np.array(request.features).reshape(1, -1)
        
        # 예측 수행
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}`}
                  </pre>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">2. Docker 컨테이너화</h4>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 애플리케이션 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`}
                  </pre>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">3. Kubernetes 배포</h4>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-api
  template:
    metadata:
      labels:
        app: ml-model-api
    spec:
      containers:
      - name: model-server
        image: your-registry/ml-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10`}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'mlops' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <Gauge className="w-5 h-5 text-green-600 dark:text-green-400" />
                MLOps 실천 사항
              </h3>
              <div className="space-y-4">
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h4 className="font-semibold mb-2">CI/CD 파이프라인</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    모델 개발부터 배포까지 자동화된 파이프라인을 구축합니다.
                  </p>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-xs overflow-x-auto">
{`# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run tests
        run: |
          pytest tests/
          
      - name: Model validation
        run: |
          python scripts/validate_model.py
          
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build and push Docker image
        run: |
          docker build -t $REGISTRY/model:$GITHUB_SHA .
          docker push $REGISTRY/model:$GITHUB_SHA
          
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ml-model-api \
            model-server=$REGISTRY/model:$GITHUB_SHA`}
                  </pre>
                </div>
                
                <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <h4 className="font-semibold mb-2">모니터링 및 로깅</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    프로덕션 환경에서 모델의 성능과 동작을 지속적으로 모니터링합니다.
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                      <p className="text-xs font-semibold">지표 모니터링</p>
                      <ul className="text-xs mt-1 space-y-0.5">
                        <li>• 추론 지연시간</li>
                        <li>• 요청 처리량</li>
                        <li>• 에러율</li>
                        <li>• 리소스 사용량</li>
                      </ul>
                    </div>
                    <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                      <p className="text-xs font-semibold">모델 모니터링</p>
                      <ul className="text-xs mt-1 space-y-0.5">
                        <li>• 예측 분포 변화</li>
                        <li>• 입력 데이터 드리프트</li>
                        <li>• 성능 저하 감지</li>
                        <li>• A/B 테스트 결과</li>
                      </ul>
                    </div>
                  </div>
                </div>
                
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h4 className="font-semibold mb-2">모델 버전 관리</h4>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-xs overflow-x-auto">
{`import mlflow

# 모델 등록
with mlflow.start_run():
    # 모델 학습
    model = train_model(X_train, y_train)
    
    # 메트릭 로깅
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1_score)
    
    # 모델 저장
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="production_model"
    )
    
# 모델 배포
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="production_model",
    version=latest_version,
    stage="Production"
)`}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex gap-3">
        <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
        <div>
          <strong>주의사항:</strong> 모델 배포 시 고려해야 할 중요한 사항들:
          <ul className="mt-2 space-y-1 list-disc list-inside">
            <li>데이터 프라이버시 및 보안 규정 준수</li>
            <li>모델 편향성 및 공정성 검증</li>
            <li>장애 대응 및 롤백 계획 수립</li>
            <li>비용 최적화 및 리소스 관리</li>
          </ul>
        </div>
      </div>

      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 rounded-xl p-6 border-2 border-blue-200 dark:border-blue-800">
        <h3 className="text-xl font-semibold mb-2">실습 프로젝트: 실시간 추천 시스템 배포</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          사용자 행동 데이터를 기반으로 실시간 추천을 제공하는 시스템을 구축하고 배포해봅시다
        </p>
        <div className="space-y-4">
          <div className="space-y-3">
            <h4 className="font-semibold">시스템 구성요소:</h4>
            <ol className="space-y-2 list-decimal list-inside">
              <li>Feature Engineering 서비스: 실시간 특징 추출</li>
              <li>Model Serving API: 추천 모델 서빙</li>
              <li>Cache Layer: Redis를 활용한 결과 캐싱</li>
              <li>A/B Testing Framework: 모델 성능 비교</li>
              <li>Monitoring Dashboard: Grafana 기반 모니터링</li>
            </ol>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <h5 className="font-semibold text-sm mb-2">성능 목표</h5>
              <ul className="text-xs space-y-1">
                <li>• 응답 시간: &lt; 100ms (P95)</li>
                <li>• 처리량: 10,000 req/s</li>
                <li>• 가용성: 99.9%</li>
                <li>• 모델 업데이트: 일 1회</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <h5 className="font-semibold text-sm mb-2">기술 스택</h5>
              <ul className="text-xs space-y-1">
                <li>• FastAPI + Gunicorn</li>
                <li>• Docker + Kubernetes</li>
                <li>• Redis + PostgreSQL</li>
                <li>• Prometheus + Grafana</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: 'ML Deployment & MLOps',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Machine Learning Systems Design',
                authors: 'Chip Huyen',
                year: '2022',
                description: '프로덕션 ML 시스템 설계 완벽 가이드 - Stanford CS 329S',
                link: 'https://huyenchip.com/machine-learning-systems-design/toc.html'
              },
              {
                title: 'Hidden Technical Debt in Machine Learning Systems',
                authors: 'D. Sculley, et al.',
                year: '2015',
                description: 'ML 시스템의 기술 부채 - Google 경험 (NeurIPS 2015)',
                link: 'https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html'
              },
              {
                title: 'Challenges in Deploying Machine Learning',
                authors: 'Andrei Paleyes, et al.',
                year: '2022',
                description: 'ML 배포의 도전과제 완벽 정리 (ACM Computing Surveys)',
                link: 'https://arxiv.org/abs/2011.09926'
              }
            ]
          },
          {
            title: 'Model Serving & Monitoring',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'TFX: A TensorFlow-Based Production-Scale ML Platform',
                authors: 'Denis Baylor, et al.',
                year: '2017',
                description: 'Google TFX 플랫폼 - 프로덕션 ML 파이프라인 (KDD 2017)',
                link: 'https://dl.acm.org/doi/10.1145/3097983.3098021'
              },
              {
                title: 'Monitoring Machine Learning Models in Production',
                authors: 'Christopher Samiullah',
                year: '2020',
                description: '프로덕션 모델 모니터링 가이드',
                link: 'https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/'
              }
            ]
          },
          {
            title: 'Infrastructure & Platforms',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Kubeflow Documentation',
                description: 'Kubernetes 기반 ML 워크플로우 플랫폼',
                link: 'https://www.kubeflow.org/'
              },
              {
                title: 'MLflow',
                description: 'Databricks의 ML 라이프사이클 관리 플랫폼',
                link: 'https://mlflow.org/'
              },
              {
                title: 'BentoML',
                description: 'ML 모델 서빙 프레임워크 - 쉽고 빠른 배포',
                link: 'https://www.bentoml.com/'
              },
              {
                title: 'Seldon Core',
                description: 'Kubernetes 네이티브 ML 배포 플랫폼',
                link: 'https://www.seldon.io/products/core'
              }
            ]
          },
          {
            title: 'Production Best Practices',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Google ML Best Practices',
                description: 'Google의 프로덕션 ML 베스트 프랙티스',
                link: 'https://developers.google.com/machine-learning/guides/rules-of-ml'
              },
              {
                title: 'AWS SageMaker',
                description: 'AWS의 완전 관리형 ML 서비스',
                link: 'https://aws.amazon.com/sagemaker/'
              },
              {
                title: 'Azure ML',
                description: 'Microsoft Azure ML 플랫폼',
                link: 'https://azure.microsoft.com/en-us/products/machine-learning/'
              },
              {
                title: 'Evidently AI',
                description: 'ML 모델 모니터링 및 테스팅 도구',
                link: 'https://www.evidentlyai.com/'
              }
            ]
          }
        ]}
      />

      <div className="flex justify-between items-center pt-8">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          모델 배포는 ML 프로젝트의 가치를 실현하는 핵심 단계입니다
        </p>
        {onComplete && (
          <button
            onClick={onComplete}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            다음 챕터로
          </button>
        )}
      </div>
    </div>
  )
}