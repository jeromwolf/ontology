'use client';

import { useState, useEffect, useRef } from 'react';
import {
  Server, Activity, Zap, TrendingUp,
  Clock, Users, BarChart3, GitBranch,
  Gauge, AlertCircle, ArrowRight, Layers
} from 'lucide-react';

interface ModelInstance {
  id: string;
  version: string;
  status: 'active' | 'starting' | 'stopping';
  requests: number;
  latency: number;
  cpu: number;
  memory: number;
  x: number;
  y: number;
}

interface Request {
  id: string;
  timestamp: number;
  targetInstance: string;
  latency: number;
  status: 'pending' | 'processing' | 'completed';
}

interface TrafficSplit {
  version: string;
  percentage: number;
  color: string;
}

export default function ModelServingSimulator() {
  const [instances, setInstances] = useState<ModelInstance[]>([]);
  const [requests, setRequests] = useState<Request[]>([]);
  const [qps, setQps] = useState(10);
  const [autoScaling, setAutoScaling] = useState(true);
  const [minInstances, setMinInstances] = useState(2);
  const [maxInstances, setMaxInstances] = useState(8);
  const [targetLatency, setTargetLatency] = useState(100);
  const [trafficSplits, setTrafficSplits] = useState<TrafficSplit[]>([
    { version: 'v1.0', percentage: 80, color: '#3b82f6' },
    { version: 'v2.0', percentage: 20, color: '#8b5cf6' }
  ]);
  const [totalRequests, setTotalRequests] = useState(0);
  const [successfulRequests, setSuccessfulRequests] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const metricsCanvasRef = useRef<HTMLCanvasElement>(null);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  const [qpsHistory, setQpsHistory] = useState<number[]>([]);

  // Initialize instances
  useEffect(() => {
    const initialInstances: ModelInstance[] = [];
    for (let i = 0; i < minInstances; i++) {
      initialInstances.push({
        id: `instance-${i}`,
        version: i < minInstances - 1 ? 'v1.0' : 'v2.0',
        status: 'active',
        requests: 0,
        latency: 50 + Math.random() * 30,
        cpu: 30 + Math.random() * 20,
        memory: 40 + Math.random() * 20,
        x: 100 + (i % 4) * 150,
        y: 150 + Math.floor(i / 4) * 100
      });
    }
    setInstances(initialInstances);
  }, [minInstances]);

  // Generate requests
  useEffect(() => {
    const interval = setInterval(() => {
      const numRequests = Math.floor(Math.random() * qps * 0.3);
      const newRequests: Request[] = [];

      for (let i = 0; i < numRequests; i++) {
        // Route based on traffic split
        const random = Math.random() * 100;
        let cumulative = 0;
        let selectedVersion = trafficSplits[0].version;

        for (const split of trafficSplits) {
          cumulative += split.percentage;
          if (random <= cumulative) {
            selectedVersion = split.version;
            break;
          }
        }

        const availableInstances = instances.filter(
          inst => inst.status === 'active' && inst.version === selectedVersion
        );

        if (availableInstances.length > 0) {
          // Round-robin load balancing
          const targetInstance = availableInstances[
            Math.floor(Math.random() * availableInstances.length)
          ];

          newRequests.push({
            id: `req-${Date.now()}-${i}`,
            timestamp: Date.now(),
            targetInstance: targetInstance.id,
            latency: 0,
            status: 'pending'
          });
        }
      }

      setRequests(prev => [...prev, ...newRequests].slice(-100));
      setTotalRequests(prev => prev + newRequests.length);
    }, 1000);

    return () => clearInterval(interval);
  }, [qps, instances, trafficSplits]);

  // Process requests
  useEffect(() => {
    const interval = setInterval(() => {
      setRequests(prev => prev.map(req => {
        if (req.status === 'pending') {
          return { ...req, status: 'processing' as const };
        } else if (req.status === 'processing') {
          const instance = instances.find(i => i.id === req.targetInstance);
          const latency = instance ? instance.latency : 100;
          if (Date.now() - req.timestamp > latency) {
            setSuccessfulRequests(prev => prev + 1);
            return { ...req, status: 'completed' as const, latency };
          }
        }
        return req;
      }));

      // Update instance metrics
      setInstances(prev => prev.map(inst => {
        const activeRequests = requests.filter(
          r => r.targetInstance === inst.id && r.status !== 'completed'
        ).length;

        const newLatency = 40 + activeRequests * 5 + Math.random() * 20;
        const newCpu = 20 + activeRequests * 8 + Math.random() * 10;
        const newMemory = 40 + activeRequests * 3 + Math.random() * 10;

        return {
          ...inst,
          requests: activeRequests,
          latency: newLatency,
          cpu: Math.min(100, newCpu),
          memory: Math.min(100, newMemory)
        };
      }));
    }, 100);

    return () => clearInterval(interval);
  }, [requests, instances]);

  // Auto-scaling logic
  useEffect(() => {
    if (!autoScaling) return;

    const interval = setInterval(() => {
      const avgLatency = instances.reduce((sum, i) => sum + i.latency, 0) / instances.length;
      const avgCpu = instances.reduce((sum, i) => sum + i.cpu, 0) / instances.length;

      setInstances(prev => {
        let updated = [...prev];

        // Scale up if latency is high or CPU is high
        if ((avgLatency > targetLatency || avgCpu > 70) && updated.length < maxInstances) {
          const newInstance: ModelInstance = {
            id: `instance-${updated.length}`,
            version: Math.random() < 0.8 ? 'v1.0' : 'v2.0',
            status: 'starting',
            requests: 0,
            latency: 50,
            cpu: 30,
            memory: 40,
            x: 100 + (updated.length % 4) * 150,
            y: 150 + Math.floor(updated.length / 4) * 100
          };
          updated.push(newInstance);

          setTimeout(() => {
            setInstances(prev => prev.map(i =>
              i.id === newInstance.id ? { ...i, status: 'active' as const } : i
            ));
          }, 2000);
        }

        // Scale down if latency is low and CPU is low
        if (avgLatency < targetLatency * 0.6 && avgCpu < 40 && updated.length > minInstances) {
          const instanceToRemove = updated[updated.length - 1];
          updated = updated.map(i =>
            i.id === instanceToRemove.id ? { ...i, status: 'stopping' as const } : i
          );

          setTimeout(() => {
            setInstances(prev => prev.filter(i => i.id !== instanceToRemove.id));
          }, 2000);
        }

        return updated;
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [autoScaling, targetLatency, minInstances, maxInstances, instances]);

  // Track metrics history
  useEffect(() => {
    const interval = setInterval(() => {
      const avgLatency = instances.reduce((sum, i) => sum + i.latency, 0) / instances.length;
      const currentQps = requests.filter(
        r => r.status === 'completed' && Date.now() - r.timestamp < 1000
      ).length;

      setLatencyHistory(prev => [...prev, avgLatency].slice(-60));
      setQpsHistory(prev => [...prev, currentQps].slice(-60));
    }, 1000);

    return () => clearInterval(interval);
  }, [instances, requests]);

  // Draw architecture diagram
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || instances.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, width, height);

    // Draw load balancer
    ctx.fillStyle = '#334155';
    ctx.fillRect(width / 2 - 60, 30, 120, 60);
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.strokeRect(width / 2 - 60, 30, 120, 60);

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Load Balancer', width / 2, 55);
    ctx.font = '11px Inter';
    ctx.fillText(`${qps} QPS`, width / 2, 75);

    // Draw instances
    instances.forEach(inst => {
      let bgColor = '#334155';
      let borderColor = '#475569';

      if (inst.status === 'starting') {
        bgColor = '#1e40af';
        borderColor = '#3b82f6';
      } else if (inst.status === 'stopping') {
        bgColor = '#7f1d1d';
        borderColor = '#ef4444';
      } else if (inst.version === 'v2.0') {
        borderColor = '#8b5cf6';
      }

      ctx.fillStyle = bgColor;
      ctx.fillRect(inst.x, inst.y, 120, 80);
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(inst.x, inst.y, 120, 80);

      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(inst.version, inst.x + 60, inst.y + 20);
      ctx.font = '10px Inter';
      ctx.fillText(inst.status, inst.x + 60, inst.y + 35);
      ctx.fillText(`${inst.requests} reqs`, inst.x + 60, inst.y + 50);
      ctx.fillText(`${inst.latency.toFixed(0)}ms`, inst.x + 60, inst.y + 65);

      // Draw line from load balancer to instance
      ctx.strokeStyle = inst.version === 'v1.0' ? '#3b82f6' : '#8b5cf6';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(width / 2, 90);
      ctx.lineTo(inst.x + 60, inst.y);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  }, [instances, qps]);

  // Draw metrics chart
  useEffect(() => {
    const canvas = metricsCanvasRef.current;
    if (!canvas || latencyHistory.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw latency line
    if (latencyHistory.length > 0) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const maxLatency = Math.max(...latencyHistory, targetLatency);
      const xStep = width / (latencyHistory.length - 1);
      latencyHistory.forEach((latency, i) => {
        const x = i * xStep;
        const y = height - (latency / maxLatency) * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Draw target latency line
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    const targetY = height - (targetLatency / Math.max(...latencyHistory, targetLatency)) * height;
    ctx.beginPath();
    ctx.moveTo(0, targetY);
    ctx.lineTo(width, targetY);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [latencyHistory, targetLatency]);

  const avgLatency = instances.reduce((sum, i) => sum + i.latency, 0) / instances.length;
  const successRate = totalRequests > 0 ? (successfulRequests / totalRequests) * 100 : 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-blue-500 p-3 rounded-lg">
              <Server className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">모델 서빙 시뮬레이터</h1>
              <p className="text-slate-300">Model Serving Infrastructure Simulator</p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Activity className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-slate-300">QPS</span>
              </div>
              <div className="text-2xl font-bold">{qpsHistory[qpsHistory.length - 1] || 0}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-300">평균 지연시간</span>
              </div>
              <div className="text-2xl font-bold">{avgLatency.toFixed(0)}ms</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Server className="w-4 h-4 text-green-400" />
                <span className="text-sm text-slate-300">인스턴스</span>
              </div>
              <div className="text-2xl font-bold">{instances.length}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                <span className="text-sm text-slate-300">성공률</span>
              </div>
              <div className="text-2xl font-bold">{successRate.toFixed(1)}%</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Users className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-slate-300">총 요청</span>
              </div>
              <div className="text-2xl font-bold">{totalRequests}</div>
            </div>
          </div>
        </div>

        {/* Configuration */}
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4">Auto-Scaling 설정</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="font-semibold">Auto-Scaling</label>
                <button
                  onClick={() => setAutoScaling(!autoScaling)}
                  className={`px-4 py-2 rounded-lg font-semibold ${
                    autoScaling ? 'bg-green-600' : 'bg-slate-600'
                  }`}
                >
                  {autoScaling ? 'ON' : 'OFF'}
                </button>
              </div>

              <div>
                <label className="block text-sm mb-2">QPS (요청/초)</label>
                <input
                  type="range"
                  min={1}
                  max={100}
                  value={qps}
                  onChange={(e) => setQps(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-right text-sm text-slate-400">{qps}</div>
              </div>

              <div>
                <label className="block text-sm mb-2">목표 지연시간 (ms)</label>
                <input
                  type="range"
                  min={50}
                  max={200}
                  value={targetLatency}
                  onChange={(e) => setTargetLatency(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-right text-sm text-slate-400">{targetLatency}ms</div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm mb-2">최소 인스턴스</label>
                  <input
                    type="number"
                    min={1}
                    max={4}
                    value={minInstances}
                    onChange={(e) => setMinInstances(Number(e.target.value))}
                    className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-2">최대 인스턴스</label>
                  <input
                    type="number"
                    min={4}
                    max={12}
                    value={maxInstances}
                    onChange={(e) => setMaxInstances(Number(e.target.value))}
                    className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4">A/B 테스트 트래픽 분배</h2>
            <div className="space-y-4">
              {trafficSplits.map((split, idx) => (
                <div key={split.version}>
                  <div className="flex items-center justify-between mb-2">
                    <label className="font-semibold">{split.version}</label>
                    <span className="text-lg font-bold">{split.percentage}%</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={split.percentage}
                    onChange={(e) => {
                      const newValue = Number(e.target.value);
                      setTrafficSplits(prev => {
                        const updated = [...prev];
                        const otherIdx = idx === 0 ? 1 : 0;
                        updated[idx].percentage = newValue;
                        updated[otherIdx].percentage = 100 - newValue;
                        return updated;
                      });
                    }}
                    className="w-full"
                  />
                  <div
                    className="h-2 rounded-full mt-2"
                    style={{
                      width: `${split.percentage}%`,
                      backgroundColor: split.color
                    }}
                  />
                </div>
              ))}
            </div>

            <div className="mt-6 bg-slate-700/50 rounded-lg p-4">
              <h3 className="font-semibold mb-2">트래픽 분배 현황</h3>
              <div className="flex h-4 rounded-full overflow-hidden">
                {trafficSplits.map(split => (
                  <div
                    key={split.version}
                    style={{
                      width: `${split.percentage}%`,
                      backgroundColor: split.color
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Architecture Diagram */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-blue-400" />
            서빙 아키텍처
          </h2>
          <canvas
            ref={canvasRef}
            className="w-full h-96 rounded-lg"
            style={{ width: '100%', height: '384px' }}
          />
        </div>

        {/* Metrics */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-purple-400" />
            지연시간 모니터링
          </h2>
          <div className="mb-2 flex gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <span>실제 지연시간</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span>목표 지연시간</span>
            </div>
          </div>
          <canvas
            ref={metricsCanvasRef}
            className="w-full h-48 rounded-lg"
            style={{ width: '100%', height: '192px' }}
          />
        </div>
      </div>
    </div>
  );
}
