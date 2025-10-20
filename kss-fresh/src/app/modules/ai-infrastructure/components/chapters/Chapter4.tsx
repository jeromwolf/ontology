'use client'

import React from 'react'
import { Code, Zap, Database, GitBranch, TrendingUp, CheckCircle, FileText, Layers } from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <Code className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                ML 프레임워크 기초
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                PyTorch와 TensorFlow로 확장 가능한 훈련 인프라 구축하기
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Layers className="w-6 h-6 text-slate-700" />
              프레임워크 선택: PyTorch vs TensorFlow
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-6">
                PyTorch와 TensorFlow는 현대 딥러닝의 양대 산맥입니다.
                두 프레임워크 모두 <strong>자동 미분, GPU 가속, 분산 훈련</strong>을 지원하지만,
                철학과 API 설계에서 차이가 있습니다.
              </p>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6 border border-orange-200 dark:border-orange-800">
                  <h3 className="font-bold text-orange-700 dark:text-orange-300 mb-3 flex items-center gap-2">
                    <Zap className="w-5 h-5" />
                    PyTorch
                  </h3>
                  <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                    <li>• <strong>Dynamic Computation Graph</strong> (Define-by-Run)</li>
                    <li>• Pythonic API, 직관적인 디버깅</li>
                    <li>• 연구 커뮤니티에서 압도적 인기</li>
                    <li>• TorchScript로 프로덕션 최적화</li>
                    <li>• Meta (Facebook) 주도 개발</li>
                  </ul>
                </div>

                <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
                  <h3 className="font-bold text-blue-700 dark:text-blue-300 mb-3 flex items-center gap-2">
                    <Layers className="w-5 h-5" />
                    TensorFlow
                  </h3>
                  <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                    <li>• <strong>Static Computation Graph</strong> (Define-then-Run)</li>
                    <li>• TensorFlow Lite/JS로 모바일/웹 배포</li>
                    <li>• 프로덕션 생태계 (TF Serving, TF Extended)</li>
                    <li>• Keras API로 간편한 고수준 인터페이스</li>
                    <li>• Google 주도 개발</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* PyTorch Setup */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Code className="w-6 h-6 text-slate-700" />
            PyTorch 분산 훈련 설정
          </h2>

          <div className="space-y-6">
            {/* DDP Setup */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                DistributedDataParallel (DDP) 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    """분산 환경 초기화"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # NCCL backend (GPU), Gloo backend (CPU)
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup_distributed(rank, world_size)

    # 모델 생성 및 DDP 래핑
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 분산 데이터 샘플러
    dataset = MyDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 훈련 루프
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 각 epoch마다 섞기

        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(rank)
            labels = labels.to(rank)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

# 멀티 프로세스 실행
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )`}
                </pre>
              </div>
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-sm font-bold text-slate-800 dark:text-white mb-2">TorchRun으로 실행</p>
                <div className="font-mono text-sm text-slate-700 dark:text-slate-300">
                  torchrun --nproc_per_node=4 train.py
                </div>
              </div>
            </div>

            {/* FSDP */}
            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Zap className="w-6 h-6" />
                FSDP: 초거대 모델 훈련
              </h3>
              <p className="text-slate-200 mb-4">
                Fully Sharded Data Parallel은 모델 파라미터, 그래디언트, 옵티마이저 상태를
                <strong> GPU 간 샤딩</strong>하여 메모리 사용량을 대폭 줄입니다.
              </p>
              <div className="bg-white/10 rounded-lg p-6 font-mono text-sm backdrop-blur overflow-x-auto">
                <pre className="text-white">
{`from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# FSDP 정책 설정
auto_wrap_policy = size_based_auto_wrap_policy(
    min_num_params=1e8  # 100M 파라미터 이상 모듈 샤딩
)

model = MyLargeModel()
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=...,  # FP16/BF16 혼합 정밀도
    sharding_strategy='FULL_SHARD',  # 완전 샤딩
    device_id=torch.cuda.current_device()
)

# 일반 DDP와 동일한 훈련 코드!
for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()`}
                </pre>
              </div>
              <div className="mt-4 grid md:grid-cols-3 gap-3">
                <div className="bg-white/10 rounded p-3">
                  <p className="font-bold text-sm mb-1">DDP</p>
                  <p className="text-xs text-slate-200">N × 모델 크기</p>
                </div>
                <div className="bg-white/10 rounded p-3">
                  <p className="font-bold text-sm mb-1">FSDP</p>
                  <p className="text-xs text-slate-200">모델 크기 / N</p>
                </div>
                <div className="bg-white/10 rounded p-3">
                  <p className="font-bold text-sm mb-1">메모리 절감</p>
                  <p className="text-xs text-slate-200">최대 N배</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* TensorFlow Distributed */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            TensorFlow 분산 전략
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                MirroredStrategy: 단일 머신 멀티 GPU
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import tensorflow as tf

# MirroredStrategy: 모든 GPU에 모델 복제
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# strategy.scope() 내에서 모델 생성
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# 데이터셋은 자동으로 샤딩됨
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(64)

# 분산 훈련 실행
model.fit(train_dataset, epochs=10)`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                MultiWorkerMirroredStrategy: 멀티 노드
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import json
import os

# TF_CONFIG 환경 변수로 클러스터 정보 제공
tf_config = {
    'cluster': {
        'worker': ['host1:12345', 'host2:12345', 'host3:12345']
    },
    'task': {'type': 'worker', 'index': 0}  # 각 워커마다 다른 index
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(...)

# 체크포인트 저장은 chief worker(index=0)만 수행
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints',
        save_weights_only=True
    )
]

model.fit(dataset, epochs=10, callbacks=callbacks)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Checkpointing */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Database className="w-6 h-6 text-slate-700" />
            체크포인팅 전략
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                PyTorch 체크포인트 저장/로드
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import torch
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, loss, path):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'torch_version': torch.__version__
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path):
    """체크포인트 로드"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# 사용 예시
save_checkpoint(
    model, optimizer, epoch, loss,
    path=f'checkpoints/model_epoch_{epoch}.pt'
)

# 훈련 재개
start_epoch, best_loss = load_checkpoint(
    model, optimizer,
    path='checkpoints/model_epoch_10.pt'
)`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Best Practice: 주기적 저장 + Best Model
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`best_val_loss = float('inf')
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # 매 epoch마다 저장
    save_checkpoint(
        model, optimizer, epoch, train_loss,
        checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    )

    # Best model 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            checkpoint_dir / 'best_model.pt'
        )
        print(f"New best model! Val loss: {val_loss:.4f}")

    # 오래된 체크포인트 정리 (최근 3개만 유지)
    all_checkpoints = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pt')
    )
    for old_ckpt in all_checkpoints[:-3]:
        old_ckpt.unlink()`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">분산 훈련 체크포인팅 주의사항</h3>
              <div className="space-y-3 text-purple-100">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                  <p><strong>Rank 0만 저장:</strong> 중복 저장 방지, 파일 충돌 회피</p>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                  <p><strong>동기화:</strong> 모든 GPU가 같은 epoch에 도달한 후 저장</p>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                  <p><strong>분산 스토리지:</strong> NFS, S3 등 공유 파일시스템 사용</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Logging */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <FileText className="w-6 h-6 text-slate-700" />
            실험 추적 및 로깅
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                TensorBoard 통합
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from torch.utils.tensorboard import SummaryWriter
import torchvision

# TensorBoard writer 생성
writer = SummaryWriter(log_dir='runs/experiment_1')

# 스칼라 로깅
for epoch in range(num_epochs):
    train_loss = train(...)
    val_loss = validate(...)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

# 이미지 로깅
writer.add_images('predictions', pred_images, epoch)

# 모델 그래프 로깅
writer.add_graph(model, input_sample)

# 하이퍼파라미터 로깅
hparams = {'lr': 1e-4, 'batch_size': 32, 'model': 'ResNet50'}
metrics = {'best_val_acc': 0.95}
writer.add_hparams(hparams, metrics)

writer.close()

# TensorBoard 실행: tensorboard --logdir=runs`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Weights & Biases (WandB) 통합
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import wandb

# 실험 초기화
wandb.init(
    project="my-awesome-project",
    config={
        "learning_rate": 1e-4,
        "architecture": "ResNet50",
        "dataset": "ImageNet",
        "epochs": 100
    }
)

# 모델 watch (그래디언트 추적)
wandb.watch(model, log='all', log_freq=100)

# 훈련 루프
for epoch in range(config.epochs):
    for batch in dataloader:
        loss = train_step(batch)

        # 메트릭 로깅
        wandb.log({
            "train/loss": loss,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

    # Validation
    val_metrics = validate(model, val_loader)
    wandb.log({
        "val/accuracy": val_metrics['accuracy'],
        "val/loss": val_metrics['loss']
    })

    # 모델 아티팩트 저장
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'model_{epoch}.pt')
        wandb.save(f'model_{epoch}.pt')

wandb.finish()`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <TrendingUp className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
                <p className="font-bold text-slate-800 dark:text-white mb-1">TensorBoard</p>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  오픈소스, 로컬 우선, PyTorch/TensorFlow 기본 지원
                </p>
              </div>
              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                <GitBranch className="w-6 h-6 text-purple-600 dark:text-purple-400 mb-2" />
                <p className="font-bold text-slate-800 dark:text-white mb-1">WandB</p>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  클라우드 기반, 팀 협업, 하이퍼파라미터 스윕
                </p>
              </div>
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                <Database className="w-6 h-6 text-green-600 dark:text-green-400 mb-2" />
                <p className="font-bold text-slate-800 dark:text-white mb-1">MLflow</p>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  End-to-end MLOps, 모델 레지스트리, 프레임워크 중립적
                </p>
              </div>
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
                <span><strong>PyTorch DDP</strong>는 가장 널리 사용되는 분산 훈련 방식으로, All-Reduce로 그래디언트를 동기화합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>FSDP</strong>는 모델을 샤딩하여 단일 GPU 메모리를 초과하는 거대 모델을 훈련합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>TensorFlow MirroredStrategy</strong>는 간단한 API로 멀티 GPU 훈련을 자동화합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>체크포인팅</strong>은 장애 복구와 실험 재현을 위해 필수이며, Best Model + 주기적 저장 전략을 사용합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>TensorBoard, WandB, MLflow</strong>는 실험 추적 및 하이퍼파라미터 비교를 위한 핵심 도구입니다.</span>
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
              <strong>Chapter 5: 분산 훈련 전략</strong>
              <br />
              DeepSpeed, Megatron-LM, ZeRO 최적화 등 초거대 모델 훈련을 위한 고급 분산 전략을 탐구합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
