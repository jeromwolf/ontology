'use client';

import { useState, useEffect, useRef } from 'react';
import {
  Video, Mic, Brain, Zap, Activity, Clock,
  Play, Pause, BarChart3, AlertCircle, CheckCircle
} from 'lucide-react';

interface PipelineStage {
  id: string;
  name: string;
  status: 'idle' | 'processing' | 'complete' | 'error';
  latency: number;
  throughput: number;
}

interface PerformanceMetric {
  timestamp: number;
  fps: number;
  latency: number;
  cpuUsage: number;
  gpuUsage: number;
}

export default function RealtimePipeline() {
  const [isRunning, setIsRunning] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [stages, setStages] = useState<PipelineStage[]>([
    { id: 'capture', name: '입력 캡처', status: 'idle', latency: 0, throughput: 0 },
    { id: 'encode-video', name: '비디오 인코딩', status: 'idle', latency: 0, throughput: 0 },
    { id: 'encode-audio', name: '오디오 인코딩', status: 'idle', latency: 0, throughput: 0 },
    { id: 'fusion', name: '모달 융합', status: 'idle', latency: 0, throughput: 0 },
    { id: 'decode', name: '디코딩', status: 'idle', latency: 0, throughput: 0 },
    { id: 'output', name: '출력 렌더링', status: 'idle', latency: 0, throughput: 0 }
  ]);
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [totalLatency, setTotalLatency] = useState(0);
  const [fps, setFps] = useState(0);
  const [qualityMode, setQualityMode] = useState<'low' | 'medium' | 'high'>('medium');
  const videoCanvasRef = useRef<HTMLCanvasElement>(null);
  const audioCanvasRef = useRef<HTMLCanvasElement>(null);
  const metricsCanvasRef = useRef<HTMLCanvasElement>(null);

  // Simulate video stream
  useEffect(() => {
    const canvas = videoCanvasRef.current;
    if (!canvas || !isRunning) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      if (!isRunning) return;

      // Generate synthetic video frame
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      gradient.addColorStop(0, `hsl(${currentFrame % 360}, 70%, 60%)`);
      gradient.addColorStop(1, `hsl(${(currentFrame + 180) % 360}, 70%, 40%)`);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add "objects" moving across frame
      const objects = 3;
      for (let i = 0; i < objects; i++) {
        const x = ((currentFrame * 2 + i * 100) % canvas.width);
        const y = 50 + i * 60;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.beginPath();
        ctx.arc(x, y, 20, 0, Math.PI * 2);
        ctx.fill();
      }

      // Frame counter
      ctx.fillStyle = 'white';
      ctx.font = 'bold 16px monospace';
      ctx.fillText(`Frame: ${currentFrame}`, 10, 30);

      setCurrentFrame(prev => prev + 1);
    };

    const interval = setInterval(animate, 1000 / 30); // 30 FPS
    return () => clearInterval(interval);
  }, [isRunning, currentFrame]);

  // Simulate audio waveform
  useEffect(() => {
    const canvas = audioCanvasRef.current;
    if (!canvas || !isRunning) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawWaveform = () => {
      if (!isRunning) return;

      ctx.fillStyle = '#1f2937';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const bars = 50;
      const barWidth = canvas.width / bars;
      const centerY = canvas.height / 2;

      ctx.fillStyle = '#10b981';
      for (let i = 0; i < bars; i++) {
        const height = Math.sin(currentFrame * 0.1 + i * 0.5) * 30 + Math.random() * 20;
        ctx.fillRect(i * barWidth, centerY - height / 2, barWidth - 2, height);
      }
    };

    const interval = setInterval(drawWaveform, 1000 / 30);
    return () => clearInterval(interval);
  }, [isRunning, currentFrame]);

  // Simulate pipeline processing
  useEffect(() => {
    if (!isRunning) return;

    const processStages = async () => {
      const latencyFactors = {
        low: 0.5,
        medium: 1,
        high: 1.5
      };
      const factor = latencyFactors[qualityMode];

      for (let i = 0; i < stages.length; i++) {
        // Update stage status
        setStages(prev => prev.map((stage, idx) => ({
          ...stage,
          status: idx === i ? 'processing' : idx < i ? 'complete' : 'idle',
          latency: idx <= i ? Math.random() * 10 * factor + 5 : 0,
          throughput: idx <= i ? Math.random() * 50 + 100 : 0
        })));

        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // All stages complete
      setStages(prev => prev.map(stage => ({ ...stage, status: 'complete' })));

      // Calculate metrics
      const latency = stages.reduce((sum, s) => sum + s.latency, 0);
      setTotalLatency(latency);
      setFps(Math.min(60, 1000 / latency));

      const metric: PerformanceMetric = {
        timestamp: Date.now(),
        fps: 1000 / latency,
        latency,
        cpuUsage: Math.random() * 30 + 40,
        gpuUsage: Math.random() * 40 + 50
      };

      setMetrics(prev => [...prev.slice(-30), metric]);

      setTimeout(() => {
        setStages(prev => prev.map(stage => ({ ...stage, status: 'idle' })));
      }, 200);
    };

    const interval = setInterval(processStages, 500);
    return () => clearInterval(interval);
  }, [isRunning, qualityMode]);

  // Draw performance metrics chart
  useEffect(() => {
    const canvas = metricsCanvasRef.current;
    if (!canvas || metrics.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = (rect.height / 5) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(rect.width, y);
      ctx.stroke();
    }

    // Draw FPS line
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    metrics.forEach((metric, idx) => {
      const x = (rect.width / metrics.length) * idx;
      const y = rect.height - (metric.fps / 60) * rect.height;
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw latency line
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    metrics.forEach((metric, idx) => {
      const x = (rect.width / metrics.length) * idx;
      const y = rect.height - (metric.latency / 100) * rect.height;
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Legend
    ctx.fillStyle = '#10b981';
    ctx.fillRect(10, 10, 20, 4);
    ctx.fillStyle = 'white';
    ctx.font = '12px sans-serif';
    ctx.fillText('FPS', 35, 15);

    ctx.fillStyle = '#ef4444';
    ctx.fillRect(80, 10, 20, 4);
    ctx.fillStyle = 'white';
    ctx.fillText('Latency', 105, 15);
  }, [metrics]);

  const getStageIcon = (stageId: string) => {
    switch (stageId) {
      case 'capture': return Video;
      case 'encode-video': return Video;
      case 'encode-audio': return Mic;
      case 'fusion': return Brain;
      case 'decode': return Zap;
      case 'output': return Activity;
      default: return Clock;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processing': return 'text-blue-600 bg-blue-50 dark:bg-blue-900/30';
      case 'complete': return 'text-green-600 bg-green-50 dark:bg-green-900/30';
      case 'error': return 'text-red-600 bg-red-50 dark:bg-red-900/30';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Activity size={32} />
          <h2 className="text-2xl font-bold">실시간 멀티모달 파이프라인</h2>
        </div>
        <p className="text-violet-100">
          비디오와 오디오를 실시간으로 처리하는 멀티모달 AI 파이프라인을 시뮬레이션합니다
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                isRunning
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
            >
              {isRunning ? <Pause size={20} /> : <Play size={20} />}
              {isRunning ? '일시정지' : '시작'}
            </button>

            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold">품질:</span>
              <select
                value={qualityMode}
                onChange={(e) => setQualityMode(e.target.value as any)}
                disabled={isRunning}
                className="px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
              >
                <option value="low">낮음 (Fast)</option>
                <option value="medium">중간 (Balanced)</option>
                <option value="high">높음 (Quality)</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{fps.toFixed(1)}</div>
              <div className="text-gray-500">FPS</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{totalLatency.toFixed(1)}</div>
              <div className="text-gray-500">ms 지연</div>
            </div>
          </div>
        </div>
      </div>

      {/* Input Streams */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Video className="text-blue-600" />
            비디오 스트림
          </h3>
          <canvas
            ref={videoCanvasRef}
            className="w-full bg-gray-900 rounded-lg"
            width={400}
            height={200}
          />
          <div className="mt-2 text-sm text-gray-500">
            해상도: 1920x1080 | 30 FPS | H.264
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Mic className="text-green-600" />
            오디오 스트림
          </h3>
          <canvas
            ref={audioCanvasRef}
            className="w-full bg-gray-900 rounded-lg"
            width={400}
            height={200}
          />
          <div className="mt-2 text-sm text-gray-500">
            샘플레이트: 48kHz | 채널: Stereo | AAC
          </div>
        </div>
      </div>

      {/* Pipeline Stages */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Zap className="text-purple-600" />
          파이프라인 단계
        </h3>
        <div className="space-y-3">
          {stages.map((stage, idx) => {
            const Icon = getStageIcon(stage.id);
            return (
              <div key={stage.id} className="relative">
                <div className={`flex items-center gap-4 p-4 rounded-lg border-2 ${
                  stage.status === 'processing'
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : stage.status === 'complete'
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                    : 'border-gray-200 dark:border-gray-700'
                }`}>
                  <div className={`p-2 rounded-lg ${getStatusColor(stage.status)}`}>
                    <Icon size={24} />
                  </div>
                  <div className="flex-1">
                    <div className="font-semibold">{stage.name}</div>
                    <div className="text-sm text-gray-500">
                      {stage.latency > 0 ? `${stage.latency.toFixed(1)}ms` : '대기중'}
                    </div>
                  </div>
                  <div className="text-right">
                    {stage.status === 'processing' && (
                      <div className="flex items-center gap-2 text-blue-600">
                        <div className="animate-spin">⚙️</div>
                        <span className="text-sm font-semibold">처리중...</span>
                      </div>
                    )}
                    {stage.status === 'complete' && (
                      <CheckCircle className="text-green-600" size={24} />
                    )}
                    {stage.status === 'error' && (
                      <AlertCircle className="text-red-600" size={24} />
                    )}
                  </div>
                </div>
                {idx < stages.length - 1 && (
                  <div className="absolute left-8 bottom-0 transform translate-y-full h-3 w-0.5 bg-gray-300 dark:bg-gray-600" />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <BarChart3 className="text-purple-600" />
          성능 모니터링
        </h3>
        <canvas
          ref={metricsCanvasRef}
          className="w-full rounded-lg"
          style={{ height: 200 }}
        />
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {metrics.length > 0 ? metrics[metrics.length - 1].cpuUsage.toFixed(0) : 0}%
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">CPU 사용률</div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {metrics.length > 0 ? metrics[metrics.length - 1].gpuUsage.toFixed(0) : 0}%
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">GPU 사용률</div>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/30 p-4 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{currentFrame}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">처리된 프레임</div>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/30 p-4 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">
              {metrics.length > 0 ? (metrics[metrics.length - 1].throughput || 150).toFixed(0) : 0}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Throughput (MB/s)</div>
          </div>
        </div>
      </div>

      {/* Optimization Tips */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">⚡ 실시간 처리 최적화 전략</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
            <h4 className="font-semibold mb-2 text-purple-600">1. 파이프라인 병렬화</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              비디오와 오디오 인코딩을 동시에 처리하여 전체 지연시간 감소
            </p>
          </div>
          <div className="p-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
            <h4 className="font-semibold mb-2 text-purple-600">2. 배치 처리</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              여러 프레임을 배치로 묶어 GPU 활용률을 극대화
            </p>
          </div>
          <div className="p-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
            <h4 className="font-semibold mb-2 text-purple-600">3. 모델 양자화</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              INT8 또는 FP16 정밀도 사용으로 추론 속도 2-4배 향상
            </p>
          </div>
          <div className="p-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
            <h4 className="font-semibold mb-2 text-purple-600">4. 프레임 스키핑</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              지연시간 증가 시 일부 프레임을 건너뛰어 실시간성 유지
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
