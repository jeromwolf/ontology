'use client'

import React from 'react'
import { BarChart3, Beaker, Package, GitBranch, TrendingUp, Database, Zap, Layers } from 'lucide-react'

export default function Chapter8() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <Beaker className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                실험 추적과 메타데이터 관리
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                MLflow, Weights & Biases, Neptune.ai로 ML 실험을 체계적으로 관리하기
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <GitBranch className="w-6 h-6 text-slate-700" />
              실험 추적이 필수인 이유
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                머신러닝 개발은 <strong>수백 번의 실험</strong>을 반복하는 과정입니다.
                학습률, 배치 크기, 모델 아키텍처 등 수십 개의 하이퍼파라미터를 조정하며,
                어떤 조합이 최고 성능을 냈는지 추적하지 않으면 <strong>재현 불가능</strong>한 결과를 얻게 됩니다.
                실험 추적 시스템은 모든 실험을 <strong>자동으로 기록</strong>하고 비교할 수 있게 합니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">실험 추적 시스템의 핵심 기능</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <BarChart3 className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>메트릭 기록</strong>: Loss, Accuracy, F1-Score 등 자동 로깅</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Package className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span><strong>아티팩트 저장</strong>: 모델, 체크포인트, 예측 결과 보관</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <GitBranch className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>버전 관리</strong>: 코드, 데이터, 모델의 정확한 버전 추적</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <TrendingUp className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" />
                    <span><strong>시각화 & 비교</strong>: 실험 간 성능 비교 대시보드</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* MLflow */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Package className="w-6 h-6 text-slate-700" />
            MLflow: 오픈소스 ML 라이프사이클 플랫폼
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                MLflow 설치 및 Tracking Server 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# MLflow 설치
pip install mlflow

# Tracking Server 시작 (PostgreSQL + S3)
mlflow server \\
    --backend-store-uri postgresql://user:pass@localhost/mlflow \\
    --default-artifact-root s3://my-bucket/mlflow-artifacts \\
    --host 0.0.0.0 \\
    --port 5000

# 환경 변수 설정
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                MLflow Tracking: 실험 로깅
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, f1_score

# Tracking URI 설정
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("image-classification")

# 실험 시작
with mlflow.start_run(run_name="resnet50-v1"):
    # 파라미터 기록
    mlflow.log_param("model", "resnet50")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 100)

    # 시스템 태그
    mlflow.set_tag("team", "cv-research")
    mlflow.set_tag("gpu", "A100")

    # 모델 훈련
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)

        # 메트릭 기록 (step별)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    # 최종 메트릭
    test_preds = model.predict(test_loader)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average='weighted')

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1", test_f1)

    # 모델 저장
    mlflow.pytorch.log_model(model, "model")

    # 아티팩트 저장 (그래프, 혼동 행렬 등)
    import matplotlib.pyplot as plt
    plt.plot(train_losses)
    plt.savefig("loss_curve.png")
    mlflow.log_artifact("loss_curve.png")

    # 텍스트 파일 저장
    with open("hyperparameters.txt", "w") as f:
        f.write(f"lr={learning_rate}, batch={batch_size}")
    mlflow.log_artifact("hyperparameters.txt")

    # 실행 ID 가져오기
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                MLflow Model Registry: 모델 버전 관리
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# 모델을 Registry에 등록
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "ImageClassifier")

# 모델 스테이지 관리
from mlflow.tracking import MlflowClient
client = MlflowClient()

# 특정 버전을 Staging으로 이동
client.transition_model_version_stage(
    name="ImageClassifier",
    version=3,
    stage="Staging"
)

# Production으로 승격
client.transition_model_version_stage(
    name="ImageClassifier",
    version=3,
    stage="Production",
    archive_existing_versions=True  # 기존 Prod 버전 Archive
)

# Production 모델 로드
import mlflow.pyfunc
model_name = "ImageClassifier"
stage = "Production"

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
predictions = model.predict(new_data)

# 모델 메타데이터 조회
latest_versions = client.get_latest_versions("ImageClassifier")
for version in latest_versions:
    print(f"Version {version.version}: Stage={version.current_stage}")`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">MLflow 4대 컴포넌트</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">MLflow Tracking</p>
                  <p className="text-sm text-slate-200">실험 파라미터, 메트릭, 아티팩트 기록</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">MLflow Projects</p>
                  <p className="text-sm text-slate-200">재현 가능한 실행 환경 (conda, docker)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">MLflow Models</p>
                  <p className="text-sm text-slate-200">다양한 프레임워크 모델 통합 포맷</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Model Registry</p>
                  <p className="text-sm text-slate-200">모델 버전 관리 및 스테이지 워크플로우</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Weights & Biases */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-slate-700" />
            Weights & Biases (W&B): 협업 중심 실험 추적
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                W&B 기본 사용법
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import wandb
import torch

# W&B 초기화
wandb.init(
    project="image-classification",
    name="resnet50-experiment",
    config={
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "optimizer": "adam",
        "architecture": "resnet50"
    },
    tags=["baseline", "resnet"]
)

# config 접근
config = wandb.config

# 모델 훈련
model = ResNet50()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(config.epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    # 메트릭 로깅 (자동으로 그래프 생성)
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# 모델 저장
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

# 실험 종료
wandb.finish()`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                W&B 고급 기능: 이미지/테이블/그래프 로깅
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# 1. 이미지 로깅 (예측 시각화)
images = []
for img, pred, label in zip(sample_images, predictions, labels):
    images.append(wandb.Image(
        img,
        caption=f"Pred: {pred}, True: {label}"
    ))
wandb.log({"predictions": images})

# 2. 테이블 로깅 (결과 분석)
import pandas as pd
results_table = wandb.Table(dataframe=pd.DataFrame({
    "image_id": image_ids,
    "prediction": predictions,
    "confidence": confidences,
    "label": labels
}))
wandb.log({"results": results_table})

# 3. Confusion Matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true,
        preds=y_pred,
        class_names=class_names
    )
})

# 4. PR Curve
wandb.log({
    "pr_curve": wandb.plot.pr_curve(
        y_true, y_probas, labels=class_names
    )
})

# 5. Histogram (가중치 분포)
wandb.log({
    "weights/conv1": wandb.Histogram(model.conv1.weight.data.cpu())
})

# 6. 3D Scatter Plot
wandb.log({
    "embeddings": wandb.Object3D(embeddings_array)
})

# 7. Audio/Video 로깅
wandb.log({
    "audio": wandb.Audio(audio_array, sample_rate=16000),
    "video": wandb.Video(video_path, fps=30)
})`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                W&B Sweeps: 하이퍼파라미터 튜닝
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# sweep_config.yaml
program: train.py
method: bayes  # random, grid, bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
    distribution: log_uniform_values
  batch_size:
    values: [16, 32, 64, 128]
  optimizer:
    values: ['adam', 'sgd', 'adamw']
  dropout:
    min: 0.1
    max: 0.5

# train.py
import wandb

def train():
    # 자동으로 sweep config에서 파라미터 주입
    wandb.init()
    config = wandb.config

    model = build_model(dropout=config.dropout)
    optimizer = get_optimizer(config.optimizer, config.learning_rate)

    # 훈련 로직
    for epoch in range(50):
        train_loss = train_epoch(model, config.batch_size)
        val_acc = validate(model)
        wandb.log({"val_accuracy": val_acc})

# Sweep 실행
sweep_id = wandb.sweep(sweep_config, project="hyperparameter-tuning")
wandb.agent(sweep_id, function=train, count=50)  # 50회 실험`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
                <h3 className="font-bold text-lg mb-3">W&B 핵심 장점</h3>
                <ul className="space-y-2 text-sm text-purple-100">
                  <li>• 실시간 협업 대시보드</li>
                  <li>• 자동 하이퍼파라미터 최적화</li>
                  <li>• 모델 성능 리포트 자동 생성</li>
                  <li>• 시각화 패널 커스터마이징</li>
                </ul>
              </div>
              <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-xl">
                <h3 className="font-bold text-lg mb-3">W&B Artifacts</h3>
                <ul className="space-y-2 text-sm text-blue-100">
                  <li>• 데이터셋 버전 관리</li>
                  <li>• 모델 체크포인트 추적</li>
                  <li>• Lineage 그래프 (데이터→모델 계보)</li>
                  <li>• 자동 중복 제거 (dedupe)</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Neptune.ai */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Neptune.ai: 엔터프라이즈 실험 관리
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Neptune.ai 통합 예제
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import neptune.new as neptune

# Neptune 초기화
run = neptune.init_run(
    project="team/image-classification",
    api_token="YOUR_API_TOKEN",
    name="resnet50-baseline",
    tags=["baseline", "resnet50"]
)

# 파라미터 기록
params = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adam"
}
run["parameters"] = params

# 시스템 정보 자동 수집
run["sys/name"] = "experiment-server-01"
run["sys/gpu"] = "NVIDIA A100"

# 훈련 중 메트릭 기록
for epoch in range(100):
    train_loss = train_epoch()
    val_metrics = validate()

    run["train/loss"].log(train_loss)
    run["val/accuracy"].log(val_metrics['accuracy'])
    run["val/f1"].log(val_metrics['f1'])

# 파일 업로드
run["model/checkpoints"].upload("model.pth")
run["visualizations/confusion_matrix"].upload("cm.png")

# 사용자 정의 객체 (JSON)
run["model/architecture"] = {
    "type": "resnet50",
    "layers": 50,
    "pretrained": True
}

# 실험 종료
run.stop()`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">Neptune.ai 차별점</h3>
              <div className="space-y-3">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-1">강력한 메타데이터 구조</p>
                  <p className="text-sm text-green-100">계층적 네임스페이스로 복잡한 실험 조직화</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-1">비교 기능</p>
                  <p className="text-sm text-green-100">수십 개 실험을 동시에 비교하는 강력한 UI</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-1">오프라인 모드</p>
                  <p className="text-sm text-green-100">인터넷 없이 실험 후 나중에 동기화</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Experiment Comparison Best Practices */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            실험 비교 및 분석 베스트 프랙티스
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                MLflow로 실험 비교
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()
experiment = client.get_experiment_by_name("image-classification")

# 모든 실행 조회
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.val_accuracy > 0.9",
    order_by=["metrics.val_accuracy DESC"],
    max_results=10
)

# 데이터프레임으로 변환
data = []
for run in runs:
    data.append({
        "run_id": run.info.run_id,
        "learning_rate": run.data.params.get("learning_rate"),
        "batch_size": run.data.params.get("batch_size"),
        "val_accuracy": run.data.metrics.get("val_accuracy"),
        "val_f1": run.data.metrics.get("val_f1"),
    })

comparison_df = pd.DataFrame(data)
print(comparison_df.sort_values("val_accuracy", ascending=False))

# 최고 성능 실행 찾기
best_run = runs[0]
best_model_uri = f"runs:/{best_run.info.run_id}/model"
print(f"Best model: {best_model_uri}")`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                실험 네이밍 컨벤션
              </h3>
              <div className="space-y-3">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <p className="font-mono text-sm text-slate-800 dark:text-white">
                    <strong>모델-버전-설명-날짜</strong>
                  </p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                    예: resnet50-v2-aug-dropout-2024-01-15
                  </p>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <p className="font-mono text-sm text-slate-800 dark:text-white">
                    <strong>태그 활용</strong>
                  </p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                    baseline, ablation, production, debug 등
                  </p>
                </div>
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <p className="font-mono text-sm text-slate-800 dark:text-white">
                    <strong>계층적 구조</strong>
                  </p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                    프로젝트/실험그룹/개별실험 (예: cv/resnet/experiment-001)
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 border-l-4 border-slate-700">
              <h3 className="font-bold text-slate-800 dark:text-white mb-3">실험 추적 체크리스트</h3>
              <ul className="space-y-2 text-slate-700 dark:text-slate-300 text-sm">
                <li>✓ 모든 하이퍼파라미터 기록 (학습률, 배치 크기, 옵티마이저 등)</li>
                <li>✓ 코드 버전 (Git commit SHA) 추적</li>
                <li>✓ 데이터셋 버전 및 전처리 스텝 기록</li>
                <li>✓ 시스템 환경 (GPU 타입, CUDA 버전, 라이브러리 버전)</li>
                <li>✓ 랜덤 시드 고정 및 기록 (재현성)</li>
                <li>✓ 훈련 시간, 메모리 사용량 측정</li>
                <li>✓ 체크포인트 저장 주기 설정</li>
                <li>✓ 실험 목적 및 가설 문서화</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8 border border-blue-200 dark:border-blue-800">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4">
              핵심 요점
            </h2>
            <ul className="space-y-3 text-slate-700 dark:text-slate-300">
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">1.</span>
                <span><strong>MLflow</strong>는 오픈소스로 자체 호스팅 가능하며, Model Registry로 모델 라이프사이클 관리를 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>Weights & Biases</strong>는 실시간 협업과 강력한 시각화 기능으로 팀 생산성을 높입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Neptune.ai</strong>는 계층적 메타데이터 구조로 대규모 실험을 체계적으로 관리합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>하이퍼파라미터 튜닝</strong>은 Sweep/Agent 패턴으로 자동화할 수 있습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span>실험 재현성을 위해 <strong>코드, 데이터, 환경 버전</strong>을 모두 기록해야 합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span>명확한 <strong>네이밍 컨벤션과 태그 전략</strong>은 수백 개의 실험을 효율적으로 관리하는 핵심입니다.</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Next Chapter Preview */}
        <section>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border-2 border-slate-300 dark:border-gray-600">
            <h3 className="text-lg font-bold text-slate-800 dark:text-white mb-2">
              다음 챕터 미리보기
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              <strong>Chapter 9: 대규모 모델 훈련</strong>
              <br />
              Multi-node Multi-GPU 분산 훈련, ZeRO/FSDP 최적화, 혼합 정밀도 훈련으로
              GPT 스타일의 초거대 모델을 효율적으로 훈련하는 실전 기법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
