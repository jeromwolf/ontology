'use client';

import { useState, useEffect, useRef } from 'react';
import {
  Boxes, Network, Zap, TrendingUp,
  Layers, GitBranch, Activity, Server,
  Clock, Database, ArrowRight, Cpu
} from 'lucide-react';

interface TrainingNode {
  id: string;
  name: string;
  rank: number;
  gpuCount: number;
  progress: number;
  throughput: number;
  status: 'training' | 'syncing' | 'idle';
  x: number;
  y: number;
}

interface GradientSync {
  from: string;
  to: string;
  progress: number;
  bandwidth: number;
}

type ParallelismMode = 'data' | 'model' | 'pipeline';

export default function DistributedTrainingVisualizer() {
  const [nodes, setNodes] = useState<TrainingNode[]>([]);
  const [parallelismMode, setParallelismMode] = useState<ParallelismMode>('data');
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(1);
  const [totalEpochs, setTotalEpochs] = useState(10);
  const [globalProgress, setGlobalProgress] = useState(0);
  const [gradientSyncs, setGradientSyncs] = useState<GradientSync[]>([]);
  const [syncFrequency, setSyncFrequency] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize nodes based on parallelism mode
  useEffect(() => {
    const nodeCount = parallelismMode === 'data' ? 8 : parallelismMode === 'model' ? 4 : 4;
    const newNodes: TrainingNode[] = [];

    if (parallelismMode === 'data') {
      // Data Parallelism: 2x4 grid
      for (let row = 0; row < 2; row++) {
        for (let col = 0; col < 4; col++) {
          const i = row * 4 + col;
          newNodes.push({
            id: `node-${i}`,
            name: `Worker ${i}`,
            rank: i,
            gpuCount: 1,
            progress: 0,
            throughput: 0,
            status: 'idle',
            x: 100 + col * 150,
            y: 100 + row * 120
          });
        }
      }
    } else if (parallelismMode === 'model') {
      // Model Parallelism: horizontal pipeline
      for (let i = 0; i < 4; i++) {
        newNodes.push({
          id: `node-${i}`,
          name: `Layer ${i + 1}`,
          rank: i,
          gpuCount: 2,
          progress: 0,
          throughput: 0,
          status: 'idle',
          x: 80 + i * 160,
          y: 150
        });
      }
    } else {
      // Pipeline Parallelism: stages
      const stages = ['Embedding', 'Encoder', 'Decoder', 'Output'];
      for (let i = 0; i < 4; i++) {
        newNodes.push({
          id: `node-${i}`,
          name: stages[i],
          rank: i,
          gpuCount: 2,
          progress: 0,
          throughput: 0,
          status: 'idle',
          x: 80 + i * 160,
          y: 150
        });
      }
    }

    setNodes(newNodes);
    setGlobalProgress(0);
  }, [parallelismMode]);

  // Training simulation
  useEffect(() => {
    if (!isTraining) return;

    const interval = setInterval(() => {
      setNodes(prev => {
        const updated = prev.map(node => {
          let newProgress = node.progress;
          let newStatus = node.status;
          let newThroughput = node.throughput;

          if (parallelismMode === 'data') {
            // Data parallel: all nodes train independently
            newProgress = Math.min(100, node.progress + Math.random() * 3);
            newStatus = newProgress < 100 ? 'training' : 'syncing';
            newThroughput = 80 + Math.random() * 40;
          } else if (parallelismMode === 'model') {
            // Model parallel: sequential processing
            if (node.rank === 0 || prev[node.rank - 1].progress > node.progress + 10) {
              newProgress = Math.min(100, node.progress + Math.random() * 2.5);
              newStatus = 'training';
              newThroughput = 60 + Math.random() * 30;
            } else {
              newStatus = 'idle';
            }
          } else {
            // Pipeline parallel: pipelined execution
            const canProcess = node.rank === 0 || prev[node.rank - 1].progress > 15;
            if (canProcess && node.progress < 100) {
              newProgress = Math.min(100, node.progress + Math.random() * 2);
              newStatus = 'training';
              newThroughput = 70 + Math.random() * 35;
            } else {
              newStatus = node.progress >= 100 ? 'syncing' : 'idle';
            }
          }

          return {
            ...node,
            progress: newProgress,
            status: newStatus,
            throughput: newThroughput
          };
        });

        // Check if epoch is complete
        const avgProgress = updated.reduce((sum, n) => sum + n.progress, 0) / updated.length;
        setGlobalProgress(avgProgress);

        if (avgProgress >= 99.5) {
          if (epoch < totalEpochs) {
            setEpoch(prev => prev + 1);
            return updated.map(n => ({ ...n, progress: 0, status: 'idle' as const }));
          } else {
            setIsTraining(false);
          }
        }

        return updated;
      });

      // Simulate gradient synchronization
      if (parallelismMode === 'data' && Math.random() < 0.3) {
        const syncs: GradientSync[] = [];
        for (let i = 0; i < nodes.length - 1; i++) {
          if (nodes[i].status === 'syncing') {
            syncs.push({
              from: nodes[i].id,
              to: nodes[i + 1].id,
              progress: Math.random() * 100,
              bandwidth: 10 + Math.random() * 15 // GB/s
            });
          }
        }
        setGradientSyncs(syncs);
      }
    }, syncFrequency);

    return () => clearInterval(interval);
  }, [isTraining, nodes, parallelismMode, syncFrequency, epoch, totalEpochs]);

  // Draw topology visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, width, height);

    // Draw connections
    if (parallelismMode !== 'data') {
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 2;
      for (let i = 0; i < nodes.length - 1; i++) {
        ctx.beginPath();
        ctx.moveTo(nodes[i].x + 50, nodes[i].y + 30);
        ctx.lineTo(nodes[i + 1].x, nodes[i + 1].y + 30);
        ctx.stroke();

        // Arrow
        const arrowX = nodes[i + 1].x - 10;
        const arrowY = nodes[i + 1].y + 30;
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(arrowX - 10, arrowY - 5);
        ctx.lineTo(arrowX - 10, arrowY + 5);
        ctx.closePath();
        ctx.fillStyle = '#475569';
        ctx.fill();
      }
    }

    // Draw gradient sync lines for data parallelism
    if (parallelismMode === 'data') {
      gradientSyncs.forEach(sync => {
        const fromNode = nodes.find(n => n.id === sync.from);
        const toNode = nodes.find(n => n.id === sync.to);
        if (!fromNode || !toNode) return;

        const gradient = ctx.createLinearGradient(
          fromNode.x + 50,
          fromNode.y + 30,
          toNode.x + 50,
          toNode.y + 30
        );
        gradient.addColorStop(0, '#3b82f6');
        gradient.addColorStop(1, '#8b5cf6');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(fromNode.x + 50, fromNode.y + 30);
        ctx.lineTo(toNode.x + 50, toNode.y + 30);
        ctx.stroke();
        ctx.setLineDash([]);

        // Bandwidth label
        ctx.fillStyle = '#a78bfa';
        ctx.font = '11px Inter';
        ctx.textAlign = 'center';
        const midX = (fromNode.x + toNode.x + 50) / 2;
        const midY = (fromNode.y + toNode.y + 30) / 2;
        ctx.fillText(`${sync.bandwidth.toFixed(1)} GB/s`, midX, midY - 5);
      });
    }

    // Draw nodes
    nodes.forEach(node => {
      // Node background
      let bgColor = '#334155';
      if (node.status === 'training') {
        bgColor = '#1e40af';
      } else if (node.status === 'syncing') {
        bgColor = '#7c3aed';
      }

      ctx.fillStyle = bgColor;
      ctx.fillRect(node.x, node.y, 100, 60);

      // Border
      ctx.strokeStyle = node.status === 'training' ? '#3b82f6' : '#475569';
      ctx.lineWidth = 2;
      ctx.strokeRect(node.x, node.y, 100, 60);

      // Node name
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 13px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(node.name, node.x + 50, node.y + 20);

      // Progress bar
      const barWidth = 80;
      const barHeight = 8;
      const barX = node.x + 10;
      const barY = node.y + 35;

      ctx.fillStyle = '#1e293b';
      ctx.fillRect(barX, barY, barWidth, barHeight);

      const progressWidth = (node.progress / 100) * barWidth;
      const progressGradient = ctx.createLinearGradient(barX, barY, barX + barWidth, barY);
      progressGradient.addColorStop(0, '#3b82f6');
      progressGradient.addColorStop(1, '#8b5cf6');
      ctx.fillStyle = progressGradient;
      ctx.fillRect(barX, barY, progressWidth, barHeight);

      // Progress text
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '10px Inter';
      ctx.fillText(`${node.progress.toFixed(0)}%`, node.x + 50, node.y + 53);
    });
  }, [nodes, gradientSyncs, parallelismMode]);

  const handleStartTraining = () => {
    setIsTraining(true);
    setEpoch(1);
    setGlobalProgress(0);
    setNodes(prev => prev.map(n => ({ ...n, progress: 0, status: 'idle' as const })));
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    setNodes(prev => prev.map(n => ({ ...n, status: 'idle' as const })));
  };

  const avgThroughput = nodes.reduce((sum, n) => sum + n.throughput, 0) / nodes.length;
  const totalGPUs = nodes.reduce((sum, n) => sum + n.gpuCount, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-purple-500 p-3 rounded-lg">
              <Network className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">분산 학습 시각화</h1>
              <p className="text-slate-300">Distributed Training Visualizer</p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Server className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-slate-300">노드</span>
              </div>
              <div className="text-2xl font-bold">{nodes.length}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Cpu className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-300">총 GPU</span>
              </div>
              <div className="text-2xl font-bold">{totalGPUs}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="w-4 h-4 text-green-400" />
                <span className="text-sm text-slate-300">Epoch</span>
              </div>
              <div className="text-2xl font-bold">
                {epoch}/{totalEpochs}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                <span className="text-sm text-slate-300">진행률</span>
              </div>
              <div className="text-2xl font-bold">{globalProgress.toFixed(1)}%</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Activity className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-slate-300">처리량</span>
              </div>
              <div className="text-2xl font-bold">{avgThroughput.toFixed(0)} img/s</div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">학습 설정</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <Boxes className="w-4 h-4 text-purple-400" />
                병렬화 전략
              </label>
              <select
                value={parallelismMode}
                onChange={(e) => setParallelismMode(e.target.value as ParallelismMode)}
                disabled={isTraining}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 disabled:opacity-50"
              >
                <option value="data">Data Parallelism</option>
                <option value="model">Model Parallelism</option>
                <option value="pipeline">Pipeline Parallelism</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <Clock className="w-4 h-4 text-green-400" />
                총 Epoch 수
              </label>
              <input
                type="number"
                value={totalEpochs}
                onChange={(e) => setTotalEpochs(Number(e.target.value))}
                disabled={isTraining}
                min={1}
                max={100}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 disabled:opacity-50"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                동기화 주기 (ms)
              </label>
              <input
                type="number"
                value={syncFrequency}
                onChange={(e) => setSyncFrequency(Number(e.target.value))}
                min={50}
                max={1000}
                step={50}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <Database className="w-4 h-4 text-cyan-400" />
                배치 크기
              </label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                disabled={isTraining}
                min={8}
                max={512}
                step={8}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 disabled:opacity-50"
              />
            </div>
          </div>

          <div className="flex gap-4 mt-6">
            <button
              onClick={handleStartTraining}
              disabled={isTraining}
              className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-slate-600 disabled:to-slate-600 text-white font-semibold py-3 px-6 rounded-lg transition-all disabled:cursor-not-allowed"
            >
              학습 시작
            </button>
            <button
              onClick={handleStopTraining}
              disabled={!isTraining}
              className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-slate-600 text-white font-semibold py-3 px-6 rounded-lg transition-all disabled:cursor-not-allowed"
            >
              학습 중지
            </button>
          </div>
        </div>

        {/* Topology Visualization */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <GitBranch className="w-6 h-6 text-purple-400" />
              네트워크 토폴로지
            </h2>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-600 rounded"></div>
                <span>Training</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-purple-600 rounded"></div>
                <span>Syncing</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-slate-600 rounded"></div>
                <span>Idle</span>
              </div>
            </div>
          </div>
          <canvas
            ref={canvasRef}
            className="w-full h-80 rounded-lg"
            style={{ width: '100%', height: '320px' }}
          />
        </div>

        {/* Parallelism Strategy Info */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Layers className="w-6 h-6 text-cyan-400" />
            병렬화 전략 설명
          </h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h3 className="font-bold text-lg mb-2 text-blue-400">Data Parallelism</h3>
              <p className="text-sm text-slate-300 mb-3">
                각 노드가 전체 모델 복사본을 가지고 서로 다른 데이터로 학습
              </p>
              <ul className="text-sm space-y-1 text-slate-400">
                <li>• 주기적 그래디언트 동기화</li>
                <li>• All-Reduce 통신 패턴</li>
                <li>• 확장성이 가장 우수</li>
              </ul>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h3 className="font-bold text-lg mb-2 text-purple-400">Model Parallelism</h3>
              <p className="text-sm text-slate-300 mb-3">
                모델을 여러 부분으로 나누어 각 노드에 분산 배치
              </p>
              <ul className="text-sm space-y-1 text-slate-400">
                <li>• 거대 모델 학습 가능</li>
                <li>• 레이어 간 활성화 전달</li>
                <li>• GPU 메모리 효율적</li>
              </ul>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h3 className="font-bold text-lg mb-2 text-green-400">Pipeline Parallelism</h3>
              <p className="text-sm text-slate-300 mb-3">
                모델 스테이지를 파이프라인으로 구성하여 처리
              </p>
              <ul className="text-sm space-y-1 text-slate-400">
                <li>• 마이크로 배치 처리</li>
                <li>• 버블 타임 최소화</li>
                <li>• 처리량 극대화</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Code Example */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">PyTorch 코드 예시</h2>
          <pre className="bg-slate-900 rounded-lg p-4 overflow-x-auto text-sm">
            <code className="text-slate-300">{`import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 분산 환경 초기화
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 모델을 DDP로 래핑
model = YourModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 분산 샘플러 사용
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=local_rank
)

# DataLoader 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=${batchSize},
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)

# 학습 루프
for epoch in range(${totalEpochs}):
    train_sampler.set_epoch(epoch)
    for batch in train_loader:
        outputs = model(batch)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 자동으로 그래디언트 동기화`}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}
