'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { AlertTriangle, Activity, Shield, TrendingUp, Clock, Database, Zap, Eye } from 'lucide-react';

interface ThreatEvent {
  id: string;
  timestamp: Date;
  type: 'adversarial' | 'extraction' | 'poisoning' | 'backdoor' | 'privacy';
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  description: string;
  status: 'detected' | 'mitigated' | 'active';
}

interface MetricData {
  time: string;
  queries: number;
  threats: number;
  blocked: number;
}

const threatTypes = {
  adversarial: { name: '적대적 공격', color: '#EF4444' },
  extraction: { name: '모델 추출', color: '#F59E0B' },
  poisoning: { name: '데이터 중독', color: '#8B5CF6' },
  backdoor: { name: '백도어', color: '#EC4899' },
  privacy: { name: '프라이버시 침해', color: '#3B82F6' }
};

export default function ThreatDetectionDashboard() {
  const [threats, setThreats] = useState<ThreatEvent[]>([]);
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [totalQueries, setTotalQueries] = useState(0);
  const [blockedRequests, setBlockedRequests] = useState(0);
  const [threatCount, setThreatCount] = useState(0);
  const [systemHealth, setSystemHealth] = useState(100);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  // 위협 생성 시뮬레이션
  const generateThreat = useCallback((): ThreatEvent => {
    const types: ThreatEvent['type'][] = ['adversarial', 'extraction', 'poisoning', 'backdoor', 'privacy'];
    const severities: ThreatEvent['severity'][] = ['low', 'medium', 'high', 'critical'];
    const sources = ['API-Client-234', 'External-IP-123.45.67.89', 'User-Session-456', 'Unknown-Source'];
    
    const type = types[Math.floor(Math.random() * types.length)];
    const severity = severities[Math.floor(Math.random() * severities.length)];
    
    const descriptions = {
      adversarial: '비정상적인 입력 패턴 감지',
      extraction: '과도한 API 쿼리 패턴',
      poisoning: '의심스러운 데이터 업로드',
      backdoor: '트리거 패턴 감지',
      privacy: '민감 정보 추출 시도'
    };

    return {
      id: `threat-${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
      type,
      severity,
      source: sources[Math.floor(Math.random() * sources.length)],
      description: descriptions[type],
      status: Math.random() > 0.3 ? 'mitigated' : 'active'
    };
  }, []);

  // 실시간 모니터링 시뮬레이션
  const startMonitoring = useCallback(() => {
    setIsMonitoring(true);
    
    const interval = setInterval(() => {
      // 메트릭 업데이트
      setTotalQueries(prev => prev + Math.floor(Math.random() * 50) + 10);
      
      // 위협 감지 (20% 확률)
      if (Math.random() < 0.2) {
        const newThreat = generateThreat();
        setThreats(prev => [newThreat, ...prev].slice(0, 10));
        setThreatCount(prev => prev + 1);
        
        if (newThreat.status === 'mitigated') {
          setBlockedRequests(prev => prev + 1);
        }
        
        // 시스템 건강도 업데이트
        if (newThreat.severity === 'critical') {
          setSystemHealth(prev => Math.max(0, prev - 10));
        } else if (newThreat.severity === 'high') {
          setSystemHealth(prev => Math.max(0, prev - 5));
        }
      }
      
      // 시스템 건강도 회복
      setSystemHealth(prev => Math.min(100, prev + 1));

      // 메트릭 히스토리 업데이트
      const now = new Date();
      const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
      
      setMetrics(prev => [...prev.slice(-19), {
        time: timeStr,
        queries: Math.floor(Math.random() * 100) + 50,
        threats: Math.floor(Math.random() * 10),
        blocked: Math.floor(Math.random() * 5)
      }]);
    }, 2000);

    return () => clearInterval(interval);
  }, [generateThreat]);

  // 그래프 그리기
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const graphWidth = width - padding * 2;
    const graphHeight = height - padding * 2;

    // 캔버스 초기화
    ctx.clearRect(0, 0, width, height);

    // 배경
    ctx.fillStyle = 'rgba(0, 0, 0, 0.02)';
    ctx.fillRect(padding, padding, graphWidth, graphHeight);

    // 그리드
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding + (graphHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    if (metrics.length < 2) return;

    // 데이터 그리기
    const maxValue = 150;
    const stepX = graphWidth / (metrics.length - 1);

    // 쿼리 라인 (파란색)
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    metrics.forEach((metric, index) => {
      const x = padding + index * stepX;
      const y = padding + graphHeight - (metric.queries / maxValue) * graphHeight;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 위협 라인 (빨간색)
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    metrics.forEach((metric, index) => {
      const x = padding + index * stepX;
      const y = padding + graphHeight - (metric.threats * 10 / maxValue) * graphHeight;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 범례
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#3B82F6';
    ctx.fillRect(width - 120, 10, 12, 12);
    ctx.fillStyle = '#000';
    ctx.fillText('쿼리', width - 100, 20);
    
    ctx.fillStyle = '#EF4444';
    ctx.fillRect(width - 120, 30, 12, 12);
    ctx.fillStyle = '#000';
    ctx.fillText('위협', width - 100, 40);
  }, [metrics]);

  // 애니메이션 루프
  useEffect(() => {
    const animate = () => {
      drawChart();
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [drawChart]);

  // 자동 시작
  useEffect(() => {
    const cleanup = startMonitoring();
    return cleanup;
  }, [startMonitoring]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50 dark:bg-red-900/30';
      case 'high': return 'text-orange-600 bg-orange-50 dark:bg-orange-900/30';
      case 'medium': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/30';
      case 'low': return 'text-blue-600 bg-blue-50 dark:bg-blue-900/30';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getHealthColor = () => {
    if (systemHealth >= 80) return 'text-green-600';
    if (systemHealth >= 60) return 'text-yellow-600';
    if (systemHealth >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
          실시간 위협 탐지 대시보드
        </h3>
        <div className="flex items-center">
          <div className={`w-3 h-3 rounded-full ${isMonitoring ? 'bg-green-500 animate-pulse' : 'bg-gray-400'} mr-2`} />
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {isMonitoring ? '모니터링 중' : '오프라인'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-5 h-5 text-blue-600" />
            <span className="text-xs text-gray-600 dark:text-gray-400">총 쿼리</span>
          </div>
          <p className="text-2xl font-bold text-blue-600">{totalQueries.toLocaleString()}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            <TrendingUp className="w-3 h-3 inline mr-1" />
            분당 {Math.floor(totalQueries / 10)}
          </p>
        </div>

        <div className="bg-red-50 dark:bg-red-900/30 p-4 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <span className="text-xs text-gray-600 dark:text-gray-400">위협 감지</span>
          </div>
          <p className="text-2xl font-bold text-red-600">{threatCount}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            활성: {threats.filter(t => t.status === 'active').length}
          </p>
        </div>

        <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <Shield className="w-5 h-5 text-green-600" />
            <span className="text-xs text-gray-600 dark:text-gray-400">차단됨</span>
          </div>
          <p className="text-2xl font-bold text-green-600">{blockedRequests}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            성공률: {totalQueries > 0 ? ((blockedRequests / threatCount) * 100).toFixed(1) : 0}%
          </p>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <Database className="w-5 h-5 text-gray-600" />
            <span className="text-xs text-gray-600 dark:text-gray-400">시스템 상태</span>
          </div>
          <p className={`text-2xl font-bold ${getHealthColor()}`}>{systemHealth}%</p>
          <div className="w-full bg-gray-200 dark:bg-gray-600 h-2 rounded-full mt-2">
            <div 
              className={`h-full rounded-full transition-all ${
                systemHealth >= 80 ? 'bg-green-500' :
                systemHealth >= 60 ? 'bg-yellow-500' :
                systemHealth >= 40 ? 'bg-orange-500' :
                'bg-red-500'
              }`}
              style={{ width: `${systemHealth}%` }}
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold mb-3">실시간 메트릭</h4>
          <canvas
            ref={canvasRef}
            width={500}
            height={250}
            className="w-full border border-gray-200 dark:border-gray-700 rounded-lg"
          />
        </div>

        <div>
          <h4 className="text-lg font-semibold mb-3">최근 위협</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {threats.length === 0 ? (
              <p className="text-gray-500 text-center py-8">아직 감지된 위협이 없습니다</p>
            ) : (
              threats.map(threat => (
                <div
                  key={threat.id}
                  className="p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center mb-1">
                        <span
                          className="w-3 h-3 rounded-full mr-2"
                          style={{ backgroundColor: threatTypes[threat.type].color }}
                        />
                        <span className="font-medium text-sm">
                          {threatTypes[threat.type].name}
                        </span>
                        <span className={`ml-2 px-2 py-1 text-xs rounded ${getSeverityColor(threat.severity)}`}>
                          {threat.severity}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {threat.description}
                      </p>
                      <div className="flex items-center mt-1 text-xs text-gray-500">
                        <Clock className="w-3 h-3 mr-1" />
                        {threat.timestamp.toLocaleTimeString()}
                        <span className="mx-2">•</span>
                        <Eye className="w-3 h-3 mr-1" />
                        {threat.source}
                      </div>
                    </div>
                    <div className="ml-3">
                      {threat.status === 'mitigated' ? (
                        <span className="text-green-600 text-xs bg-green-50 dark:bg-green-900/30 px-2 py-1 rounded">
                          차단됨
                        </span>
                      ) : (
                        <span className="text-red-600 text-xs bg-red-50 dark:bg-red-900/30 px-2 py-1 rounded">
                          활성
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}