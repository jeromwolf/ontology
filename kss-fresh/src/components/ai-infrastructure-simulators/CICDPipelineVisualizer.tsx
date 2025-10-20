'use client';

import { useState, useEffect } from 'react';
import {
  GitBranch, Play, CheckCircle, XCircle,
  Clock, Code, TestTube, Package, Rocket, RotateCcw
} from 'lucide-react';

interface PipelineStage {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  duration: number;
  startTime?: Date;
  icon: any;
}

interface Deployment {
  version: string;
  traffic: number;
  status: 'healthy' | 'degraded' | 'unhealthy';
  replicas: number;
}

export default function CICDPipelineVisualizer() {
  const [stages, setStages] = useState<PipelineStage[]>([
    { id: 'checkout', name: 'Code Checkout', status: 'pending', duration: 0, icon: Code },
    { id: 'test', name: 'Run Tests', status: 'pending', duration: 0, icon: TestTube },
    { id: 'build', name: 'Build Model', status: 'pending', duration: 0, icon: Package },
    { id: 'validate', name: 'Validate Model', status: 'pending', duration: 0, icon: CheckCircle },
    { id: 'deploy', name: 'Deploy to Staging', status: 'pending', duration: 0, icon: Rocket }
  ]);

  const [isRunning, setIsRunning] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const [deployments, setDeployments] = useState<Deployment[]>([
    { version: 'v1.0', traffic: 80, status: 'healthy', replicas: 3 },
    { version: 'v2.0', traffic: 20, status: 'healthy', replicas: 1 }
  ]);
  const [canaryProgress, setCanaryProgress] = useState(20);
  const [rollbackEnabled, setRollbackEnabled] = useState(false);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setStages(prev => {
        const updated = [...prev];
        const current = updated[currentStage];

        if (!current) {
          setIsRunning(false);
          return prev;
        }

        if (current.status === 'pending') {
          updated[currentStage] = {
            ...current,
            status: 'running',
            startTime: new Date()
          };
        } else if (current.status === 'running') {
          const duration = current.duration + 1;
          const targetDuration = [3, 8, 12, 6, 5][currentStage];

          if (duration >= targetDuration) {
            const success = Math.random() > 0.1;
            updated[currentStage] = {
              ...current,
              status: success ? 'success' : 'failed',
              duration
            };

            if (success && currentStage < updated.length - 1) {
              setCurrentStage(prev => prev + 1);
            } else if (!success) {
              setIsRunning(false);
              setRollbackEnabled(true);
            } else if (currentStage === updated.length - 1) {
              setIsRunning(false);
              updateCanaryDeployment();
            }
          } else {
            updated[currentStage] = { ...current, duration };
          }
        }

        return updated;
      });
    }, 500);

    return () => clearInterval(interval);
  }, [isRunning, currentStage]);

  const updateCanaryDeployment = () => {
    if (canaryProgress < 100) {
      const newProgress = Math.min(100, canaryProgress + 20);
      setCanaryProgress(newProgress);
      setDeployments([
        { version: 'v1.0', traffic: 100 - newProgress, status: 'healthy', replicas: Math.ceil((100 - newProgress) / 25) },
        { version: 'v2.0', traffic: newProgress, status: 'healthy', replicas: Math.ceil(newProgress / 25) }
      ]);
    }
  };

  const handleRun = () => {
    setStages(prev => prev.map(s => ({ ...s, status: 'pending', duration: 0 })));
    setCurrentStage(0);
    setIsRunning(true);
    setRollbackEnabled(false);
  };

  const handleRollback = () => {
    setDeployments([
      { version: 'v1.0', traffic: 100, status: 'healthy', replicas: 4 },
      { version: 'v2.0', traffic: 0, status: 'unhealthy', replicas: 0 }
    ]);
    setCanaryProgress(0);
    setRollbackEnabled(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-slate-600 text-slate-300';
      case 'running': return 'bg-blue-600 text-white animate-pulse';
      case 'success': return 'bg-green-600 text-white';
      case 'failed': return 'bg-red-600 text-white';
      default: return 'bg-slate-600 text-slate-300';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Clock className="w-5 h-5 animate-spin" />;
      case 'success': return <CheckCircle className="w-5 h-5" />;
      case 'failed': return <XCircle className="w-5 h-5" />;
      default: return <Clock className="w-5 h-5" />;
    }
  };

  const totalDuration = stages.reduce((sum, s) => sum + s.duration, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="bg-blue-500 p-3 rounded-lg">
                <GitBranch className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">CI/CD 파이프라인 시각화</h1>
                <p className="text-slate-300">ML CI/CD Pipeline Visualizer</p>
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleRun}
                disabled={isRunning}
                className="flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-600 px-4 py-2 rounded-lg font-semibold disabled:cursor-not-allowed"
              >
                <Play className="w-5 h-5" />
                Run Pipeline
              </button>
              {rollbackEnabled && (
                <button
                  onClick={handleRollback}
                  className="flex items-center gap-2 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg font-semibold"
                >
                  <RotateCcw className="w-5 h-5" />
                  Rollback
                </button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">총 단계</div>
              <div className="text-2xl font-bold">{stages.length}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">성공</div>
              <div className="text-2xl font-bold text-green-400">
                {stages.filter(s => s.status === 'success').length}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">실패</div>
              <div className="text-2xl font-bold text-red-400">
                {stages.filter(s => s.status === 'failed').length}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">소요 시간</div>
              <div className="text-2xl font-bold">{totalDuration}s</div>
            </div>
          </div>
        </div>

        {/* Pipeline Stages */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">파이프라인 단계</h2>
          <div className="space-y-3">
            {stages.map((stage, idx) => {
              const Icon = stage.icon;
              return (
                <div
                  key={stage.id}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    stage.status === 'running'
                      ? 'border-blue-500 bg-blue-500/10'
                      : stage.status === 'success'
                      ? 'border-green-500 bg-green-500/10'
                      : stage.status === 'failed'
                      ? 'border-red-500 bg-red-500/10'
                      : 'border-slate-700 bg-slate-700/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className={`p-3 rounded-lg ${getStatusColor(stage.status)}`}>
                        <Icon className="w-6 h-6" />
                      </div>
                      <div>
                        <div className="font-semibold text-lg">{stage.name}</div>
                        <div className="text-sm text-slate-400">
                          {stage.status === 'running' && `진행 중... ${stage.duration}s`}
                          {stage.status === 'success' && `완료 (${stage.duration}s)`}
                          {stage.status === 'failed' && `실패 (${stage.duration}s)`}
                          {stage.status === 'pending' && '대기 중'}
                        </div>
                      </div>
                    </div>
                    <div>{getStatusIcon(stage.status)}</div>
                  </div>

                  {stage.status === 'running' && (
                    <div className="mt-3 bg-slate-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all"
                        style={{ width: `${(stage.duration / [3, 8, 12, 6, 5][idx]) * 100}%` }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Canary Deployment */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Rocket className="w-6 h-6 text-purple-400" />
            Canary 배포
          </h2>

          <div className="mb-6">
            <div className="flex justify-between mb-2">
              <span className="text-sm font-semibold">배포 진행률</span>
              <span className="text-sm font-semibold">{canaryProgress}%</span>
            </div>
            <div className="bg-slate-700 rounded-full h-4">
              <div
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-4 rounded-full transition-all"
                style={{ width: `${canaryProgress}%` }}
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {deployments.map(dep => (
              <div
                key={dep.version}
                className={`p-4 rounded-lg border-2 ${
                  dep.status === 'healthy'
                    ? 'border-green-500 bg-green-500/10'
                    : dep.status === 'degraded'
                    ? 'border-yellow-500 bg-yellow-500/10'
                    : 'border-red-500 bg-red-500/10'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-lg font-bold">{dep.version}</span>
                  <span
                    className={`px-2 py-1 rounded text-xs font-semibold ${
                      dep.status === 'healthy'
                        ? 'bg-green-600'
                        : dep.status === 'degraded'
                        ? 'bg-yellow-600'
                        : 'bg-red-600'
                    }`}
                  >
                    {dep.status}
                  </span>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">트래픽:</span>
                    <span className="font-semibold">{dep.traffic}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">레플리카:</span>
                    <span className="font-semibold">{dep.replicas}</span>
                  </div>
                </div>

                <div className="mt-3 bg-slate-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      dep.status === 'healthy' ? 'bg-green-500' : dep.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${dep.traffic}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Metrics */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">배포 메트릭</h2>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-1">평균 응답시간</div>
              <div className="text-2xl font-bold text-green-400">45ms</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-1">에러율</div>
              <div className="text-2xl font-bold text-blue-400">0.02%</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-1">QPS</div>
              <div className="text-2xl font-bold">1,245</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-1">성공률</div>
              <div className="text-2xl font-bold text-green-400">99.98%</div>
            </div>
          </div>
        </div>

        {/* GitHub Actions YAML */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">GitHub Actions Workflow</h2>
          <pre className="bg-slate-900 rounded-lg p-4 overflow-x-auto text-sm">
            <code className="text-green-400">{`name: ML Model CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build model
        run: python train.py
      - name: Upload model
        uses: actions/upload-artifact@v3

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl apply -f k8s/deployment.yaml
          kubectl rollout status deployment/model-server`}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}
