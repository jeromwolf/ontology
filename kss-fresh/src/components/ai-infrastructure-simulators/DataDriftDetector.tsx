'use client';

import { useState, useEffect, useRef } from 'react';
import {
  AlertTriangle, TrendingUp, BarChart3, Activity,
  Calendar, Filter, Bell, Eye
} from 'lucide-react';

interface FeatureDrift {
  name: string;
  ksStatistic: number;
  pValue: number;
  driftScore: number;
  status: 'normal' | 'warning' | 'critical';
}

interface DriftEvent {
  timestamp: Date;
  feature: string;
  severity: 'low' | 'medium' | 'high';
  description: string;
}

export default function DataDriftDetector() {
  const [features, setFeatures] = useState<FeatureDrift[]>([]);
  const [driftHistory, setDriftHistory] = useState<number[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<FeatureDrift | null>(null);
  const [threshold, setThreshold] = useState(0.05);
  const [events, setEvents] = useState<DriftEvent[]>([]);
  const [monitoring, setMonitoring] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const distCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const featureNames = ['age', 'income', 'credit_score', 'purchase_amount', 'session_duration', 'click_rate'];
    const initialFeatures: FeatureDrift[] = featureNames.map(name => ({
      name,
      ksStatistic: Math.random() * 0.15,
      pValue: Math.random(),
      driftScore: Math.random() * 100,
      status: Math.random() > 0.8 ? 'warning' : 'normal'
    }));
    setFeatures(initialFeatures);
    setSelectedFeature(initialFeatures[0]);
    setDriftHistory(Array.from({ length: 30 }, () => Math.random() * 80));
  }, []);

  useEffect(() => {
    if (!monitoring) return;

    const interval = setInterval(() => {
      setFeatures(prev => prev.map(f => {
        const newKS = Math.max(0, Math.min(0.3, f.ksStatistic + (Math.random() - 0.5) * 0.03));
        const newPValue = Math.random();
        const newScore = newKS * 100;

        let status: 'normal' | 'warning' | 'critical' = 'normal';
        if (newKS > threshold * 2) status = 'critical';
        else if (newKS > threshold) status = 'warning';

        if (status !== 'normal' && f.status === 'normal') {
          setEvents(prev => [{
            timestamp: new Date(),
            feature: f.name,
            severity: status === 'critical' ? 'high' : 'medium',
            description: `${f.name} 피처에서 데이터 드리프트 감지됨 (KS=${newKS.toFixed(3)})`
          }, ...prev].slice(0, 10));
        }

        return { ...f, ksStatistic: newKS, pValue: newPValue, driftScore: newScore, status };
      }));

      setDriftHistory(prev => [...prev, Math.random() * 100].slice(-30));
    }, 2000);

    return () => clearInterval(interval);
  }, [monitoring, threshold]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || driftHistory.length === 0) return;

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

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const xStep = width / (driftHistory.length - 1);
    driftHistory.forEach((value, i) => {
      const x = i * xStep;
      const y = height - (value / 100) * height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const thresholdY = height - (threshold * 100);
    ctx.strokeStyle = '#ef4444';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, thresholdY);
    ctx.lineTo(width, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [driftHistory, threshold]);

  useEffect(() => {
    const canvas = distCanvasRef.current;
    if (!canvas || !selectedFeature) return;

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

    const drawDist = (mean: number, std: number, color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let x = 0; x < width; x++) {
        const z = ((x / width) * 6 - 3);
        const y = height - (height * Math.exp(-0.5 * Math.pow((z - mean) / std, 2)) / (std * Math.sqrt(2 * Math.PI)) * 2);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    drawDist(0, 1, '#3b82f6');
    drawDist(selectedFeature.ksStatistic * 10 - 1, 1.2, '#ef4444');
  }, [selectedFeature]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'bg-green-500/20 text-green-400';
      case 'warning': return 'bg-yellow-500/20 text-yellow-400';
      case 'critical': return 'bg-red-500/20 text-red-400';
      default: return 'bg-slate-500/20 text-slate-400';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-blue-400';
      default: return 'text-slate-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="bg-orange-500 p-3 rounded-lg">
                <AlertTriangle className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">데이터 드리프트 탐지기</h1>
                <p className="text-slate-300">Data Drift Detection Simulator</p>
              </div>
            </div>
            <button
              onClick={() => setMonitoring(!monitoring)}
              className={`px-4 py-2 rounded-lg font-semibold ${
                monitoring ? 'bg-green-600' : 'bg-slate-600'
              }`}
            >
              {monitoring ? '모니터링 중' : '일시정지'}
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">총 피처</div>
              <div className="text-2xl font-bold">{features.length}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">정상</div>
              <div className="text-2xl font-bold text-green-400">
                {features.filter(f => f.status === 'normal').length}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">경고</div>
              <div className="text-2xl font-bold text-yellow-400">
                {features.filter(f => f.status === 'warning').length}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">위험</div>
              <div className="text-2xl font-bold text-red-400">
                {features.filter(f => f.status === 'critical').length}
              </div>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-blue-400" />
              피처별 드리프트 스코어
            </h2>
            <div className="space-y-3">
              {features.map(feature => (
                <button
                  key={feature.name}
                  onClick={() => setSelectedFeature(feature)}
                  className={`w-full text-left p-4 rounded-lg border transition-all ${
                    selectedFeature?.name === feature.name
                      ? 'border-cyan-500 bg-slate-700'
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">{feature.name}</span>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${getStatusColor(feature.status)}`}>
                      {feature.status}
                    </span>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">KS Statistic:</span>
                      <span className="font-bold">{feature.ksStatistic.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">P-Value:</span>
                      <span className="font-bold">{feature.pValue.toFixed(3)}</span>
                    </div>
                    <div className="bg-slate-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          feature.status === 'critical' ? 'bg-red-500' :
                          feature.status === 'warning' ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${Math.min(100, feature.driftScore)}%` }}
                      />
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold">드리프트 임계값</h2>
                <span className="text-lg font-bold text-cyan-400">{threshold.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min={0.01}
                max={0.15}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-slate-400 mt-2">
                <span>0.01 (민감)</span>
                <span>0.15 (관대)</span>
              </div>
            </div>

            <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Activity className="w-6 h-6 text-purple-400" />
                드리프트 추이
              </h2>
              <canvas
                ref={canvasRef}
                className="w-full h-48 rounded-lg"
                style={{ width: '100%', height: '192px' }}
              />
              <div className="mt-2 flex gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded"></div>
                  <span>실제값</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 rounded"></div>
                  <span>임계값</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {selectedFeature && (
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-green-400" />
              분포 비교: {selectedFeature.name}
            </h2>
            <canvas
              ref={distCanvasRef}
              className="w-full h-64 rounded-lg"
              style={{ width: '100%', height: '256px' }}
            />
            <div className="mt-4 grid md:grid-cols-2 gap-4">
              <div className="flex gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded"></div>
                  <span>학습 데이터 분포</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 rounded"></div>
                  <span>운영 데이터 분포</span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-slate-400">KS Statistic</div>
                <div className="text-2xl font-bold text-orange-400">{selectedFeature.ksStatistic.toFixed(3)}</div>
              </div>
            </div>
          </div>
        )}

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Bell className="w-6 h-6 text-yellow-400" />
            알림 이벤트
          </h2>
          {events.length > 0 ? (
            <div className="space-y-2">
              {events.map((event, idx) => (
                <div key={idx} className="bg-slate-700/50 rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <AlertTriangle className={`w-4 h-4 ${getSeverityColor(event.severity)}`} />
                        <span className={`font-semibold ${getSeverityColor(event.severity)}`}>
                          {event.severity.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm">{event.description}</p>
                    </div>
                    <div className="text-xs text-slate-400">
                      {event.timestamp.toLocaleTimeString('ko-KR')}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-slate-400 py-8">
              드리프트 이벤트 없음
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
