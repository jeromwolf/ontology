'use client';

import { useState, useEffect, useRef } from 'react';
import {
  Cpu, Zap, Thermometer, Activity,
  Server, AlertTriangle, CheckCircle, XCircle,
  TrendingUp, BarChart3, Gauge, HardDrive
} from 'lucide-react';

interface GPUNode {
  id: string;
  name: string;
  gpuCount: number;
  utilization: number;
  memoryUsed: number;
  memoryTotal: number;
  temperature: number;
  powerUsage: number;
  powerLimit: number;
  status: 'healthy' | 'warning' | 'error';
  cudaCores: number;
  tensorCores: number;
}

interface MetricHistory {
  timestamp: number;
  utilization: number;
  memory: number;
  temperature: number;
}

export default function GPUClusterMonitor() {
  const [nodes, setNodes] = useState<GPUNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [metricHistory, setMetricHistory] = useState<MetricHistory[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(1000);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyCanvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize GPU cluster
  useEffect(() => {
    const initialNodes: GPUNode[] = Array.from({ length: 8 }, (_, i) => ({
      id: `gpu-node-${i + 1}`,
      name: `GPU-${String(i + 1).padStart(2, '0')}`,
      gpuCount: i < 4 ? 8 : 4,
      utilization: Math.random() * 100,
      memoryUsed: Math.random() * 80,
      memoryTotal: 80,
      temperature: 45 + Math.random() * 30,
      powerUsage: 250 + Math.random() * 150,
      powerLimit: 400,
      status: Math.random() > 0.9 ? 'warning' : 'healthy',
      cudaCores: 10752,
      tensorCores: 336
    }));
    setNodes(initialNodes);
    setSelectedNode(initialNodes[0].id);
  }, []);

  // Auto-refresh metrics
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      setNodes(prev => prev.map(node => {
        const newUtil = Math.max(0, Math.min(100, node.utilization + (Math.random() - 0.5) * 10));
        const newMem = Math.max(0, Math.min(node.memoryTotal, node.memoryUsed + (Math.random() - 0.5) * 5));
        const newTemp = Math.max(40, Math.min(85, node.temperature + (Math.random() - 0.5) * 3));
        const newPower = Math.max(100, Math.min(node.powerLimit, node.powerUsage + (Math.random() - 0.5) * 20));

        let status: 'healthy' | 'warning' | 'error' = 'healthy';
        if (newTemp > 80 || newUtil > 95 || newMem > node.memoryTotal * 0.95) {
          status = 'error';
        } else if (newTemp > 70 || newUtil > 85 || newMem > node.memoryTotal * 0.85) {
          status = 'warning';
        }

        return {
          ...node,
          utilization: newUtil,
          memoryUsed: newMem,
          temperature: newTemp,
          powerUsage: newPower,
          status
        };
      }));

      // Update history for selected node
      const selected = nodes.find(n => n.id === selectedNode);
      if (selected) {
        setMetricHistory(prev => {
          const newHistory = [
            ...prev,
            {
              timestamp: Date.now(),
              utilization: selected.utilization,
              memory: (selected.memoryUsed / selected.memoryTotal) * 100,
              temperature: selected.temperature
            }
          ];
          return newHistory.slice(-60); // Keep last 60 data points
        });
      }
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, selectedNode, nodes]);

  // Draw GPU utilization visualization
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

    // Draw grid
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(width - 20, y);
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = '#94a3b8';
      ctx.font = '12px Inter';
      ctx.textAlign = 'right';
      ctx.fillText(`${100 - i * 25}%`, 35, y + 4);
    }

    // Draw bars for each GPU
    const barWidth = (width - 80) / nodes.length;
    const padding = barWidth * 0.2;

    nodes.forEach((node, i) => {
      const x = 50 + i * barWidth;
      const barHeight = (node.utilization / 100) * (height - 40);
      const y = height - 20 - barHeight;

      // Bar color based on status
      let gradient = ctx.createLinearGradient(x, height - 20, x, y);
      if (node.status === 'error') {
        gradient.addColorStop(0, '#ef4444');
        gradient.addColorStop(1, '#dc2626');
      } else if (node.status === 'warning') {
        gradient.addColorStop(0, '#f59e0b');
        gradient.addColorStop(1, '#d97706');
      } else {
        gradient.addColorStop(0, '#3b82f6');
        gradient.addColorStop(1, '#2563eb');
      }

      ctx.fillStyle = gradient;
      ctx.fillRect(x + padding, y, barWidth - padding * 2, barHeight);

      // Node label
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '11px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(node.name, x + barWidth / 2, height - 5);

      // Utilization percentage
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px Inter';
      ctx.fillText(`${node.utilization.toFixed(0)}%`, x + barWidth / 2, y - 5);

      // Highlight selected node
      if (node.id === selectedNode) {
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 3;
        ctx.strokeRect(x + padding, y, barWidth - padding * 2, barHeight);
      }
    });
  }, [nodes, selectedNode]);

  // Draw metric history chart
  useEffect(() => {
    const canvas = historyCanvasRef.current;
    if (!canvas || metricHistory.length < 2) return;

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

    const drawLine = (data: number[], color: string, label: string) => {
      if (data.length < 2) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const xStep = width / (data.length - 1);
      data.forEach((value, i) => {
        const x = i * xStep;
        const y = height - (value / 100) * height;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();
    };

    // Draw lines
    drawLine(
      metricHistory.map(m => m.utilization),
      '#3b82f6',
      'Utilization'
    );
    drawLine(
      metricHistory.map(m => m.memory),
      '#10b981',
      'Memory'
    );
    drawLine(
      metricHistory.map(m => (m.temperature - 40) * 2.5), // Scale temperature to 0-100
      '#ef4444',
      'Temperature'
    );
  }, [metricHistory]);

  const selectedNodeData = nodes.find(n => n.id === selectedNode);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return null;
    }
  };

  const clusterStats = {
    totalGPUs: nodes.reduce((sum, n) => sum + n.gpuCount, 0),
    avgUtilization: nodes.reduce((sum, n) => sum + n.utilization, 0) / nodes.length,
    healthyNodes: nodes.filter(n => n.status === 'healthy').length,
    warningNodes: nodes.filter(n => n.status === 'warning').length,
    errorNodes: nodes.filter(n => n.status === 'error').length,
    totalMemory: nodes.reduce((sum, n) => sum + n.memoryTotal, 0),
    usedMemory: nodes.reduce((sum, n) => sum + n.memoryUsed, 0),
    totalPower: nodes.reduce((sum, n) => sum + n.powerUsage, 0)
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="bg-blue-500 p-3 rounded-lg">
                <Server className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">GPU 클러스터 모니터</h1>
                <p className="text-slate-300">Multi-GPU Cluster Monitoring Dashboard</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  autoRefresh
                    ? 'bg-green-600 hover:bg-green-700'
                    : 'bg-slate-600 hover:bg-slate-500'
                }`}
              >
                {autoRefresh ? 'Auto-Refresh ON' : 'Auto-Refresh OFF'}
              </button>
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
              >
                <option value={500}>0.5초</option>
                <option value={1000}>1초</option>
                <option value={2000}>2초</option>
                <option value={5000}>5초</option>
              </select>
            </div>
          </div>

          {/* Cluster Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-slate-300">총 GPU</span>
              </div>
              <div className="text-2xl font-bold">{clusterStats.totalGPUs}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-300">평균 사용률</span>
              </div>
              <div className="text-2xl font-bold">{clusterStats.avgUtilization.toFixed(1)}%</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-sm text-slate-300">정상</span>
              </div>
              <div className="text-2xl font-bold text-green-400">{clusterStats.healthyNodes}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-slate-300">경고</span>
              </div>
              <div className="text-2xl font-bold text-yellow-400">{clusterStats.warningNodes}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-4 h-4 text-red-400" />
                <span className="text-sm text-slate-300">오류</span>
              </div>
              <div className="text-2xl font-bold text-red-400">{clusterStats.errorNodes}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <HardDrive className="w-4 h-4 text-cyan-400" />
                <span className="text-sm text-slate-300">메모리</span>
              </div>
              <div className="text-2xl font-bold">
                {clusterStats.usedMemory.toFixed(0)}/{clusterStats.totalMemory}GB
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-slate-300">전력</span>
              </div>
              <div className="text-2xl font-bold">{(clusterStats.totalPower / 1000).toFixed(1)}kW</div>
            </div>
          </div>
        </div>

        {/* GPU Utilization Chart */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-blue-400" />
              GPU 사용률 현황
            </h2>
            <div className="flex gap-2">
              <button
                onClick={() => setViewMode('grid')}
                className={`px-3 py-1 rounded ${
                  viewMode === 'grid' ? 'bg-blue-600' : 'bg-slate-700'
                }`}
              >
                Grid
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`px-3 py-1 rounded ${
                  viewMode === 'list' ? 'bg-blue-600' : 'bg-slate-700'
                }`}
              >
                List
              </button>
            </div>
          </div>
          <canvas
            ref={canvasRef}
            className="w-full h-64 rounded-lg"
            style={{ width: '100%', height: '256px' }}
          />
        </div>

        {/* Node Details and History */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Selected Node Details */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Gauge className="w-6 h-6 text-purple-400" />
              노드 상세 정보
            </h2>
            {selectedNodeData && (
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-700">
                  <span className="text-lg font-semibold">{selectedNodeData.name}</span>
                  {getStatusIcon(selectedNodeData.status)}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <div className="text-sm text-slate-400">GPU Count</div>
                    <div className="text-xl font-bold">{selectedNodeData.gpuCount}x GPU</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-slate-400">Utilization</div>
                    <div className="text-xl font-bold text-blue-400">
                      {selectedNodeData.utilization.toFixed(1)}%
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-slate-400">Memory</div>
                    <div className="text-xl font-bold text-green-400">
                      {selectedNodeData.memoryUsed.toFixed(1)}/{selectedNodeData.memoryTotal}GB
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-slate-400">Temperature</div>
                    <div className="text-xl font-bold text-orange-400">
                      {selectedNodeData.temperature.toFixed(1)}°C
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-slate-400">Power</div>
                    <div className="text-xl font-bold text-yellow-400">
                      {selectedNodeData.powerUsage.toFixed(0)}/{selectedNodeData.powerLimit}W
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-slate-400">CUDA Cores</div>
                    <div className="text-xl font-bold">{selectedNodeData.cudaCores.toLocaleString()}</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-slate-400">Tensor Cores</div>
                    <div className="text-xl font-bold">{selectedNodeData.tensorCores}</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-slate-400">Memory %</div>
                    <div className="text-xl font-bold">
                      {((selectedNodeData.memoryUsed / selectedNodeData.memoryTotal) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Metric History */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-green-400" />
              실시간 메트릭 추이
            </h2>
            <div className="mb-4 flex gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded"></div>
                <span>사용률</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded"></div>
                <span>메모리</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded"></div>
                <span>온도</span>
              </div>
            </div>
            <canvas
              ref={historyCanvasRef}
              className="w-full h-48 rounded-lg"
              style={{ width: '100%', height: '192px' }}
            />
          </div>
        </div>

        {/* Node List */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">모든 노드</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {nodes.map(node => (
              <button
                key={node.id}
                onClick={() => setSelectedNode(node.id)}
                className={`text-left p-4 rounded-lg border-2 transition-all ${
                  selectedNode === node.id
                    ? 'border-blue-500 bg-slate-700'
                    : 'border-slate-700 bg-slate-800 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold">{node.name}</span>
                  {getStatusIcon(node.status)}
                </div>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">사용률:</span>
                    <span className="font-semibold">{node.utilization.toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">메모리:</span>
                    <span className="font-semibold">
                      {((node.memoryUsed / node.memoryTotal) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">온도:</span>
                    <span className="font-semibold">{node.temperature.toFixed(0)}°C</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
