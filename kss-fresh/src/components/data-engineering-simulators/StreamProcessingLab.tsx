'use client';

import { useState, useEffect, useRef } from 'react';
import {
  Radio, TrendingUp, Zap, Activity,
  Play, Pause, RotateCcw, Settings, AlertTriangle
} from 'lucide-react';

interface StreamEvent {
  id: string;
  timestamp: number;
  userId: string;
  eventType: string;
  value: number;
}

interface WindowedMetric {
  window: string;
  count: number;
  sum: number;
  avg: number;
}

export default function StreamProcessingLab() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [windowType, setWindowType] = useState<'tumbling' | 'sliding' | 'session'>('tumbling');
  const [windowSize, setWindowSize] = useState(5);
  const [metrics, setMetrics] = useState<WindowedMetric[]>([]);
  const [throughput, setThroughput] = useState(0);
  const [latency, setLatency] = useState(0);
  const streamInterval = useRef<NodeJS.Timeout | null>(null);

  const eventTypes = ['click', 'view', 'purchase', 'add_to_cart', 'search'];

  useEffect(() => {
    if (isStreaming) {
      streamInterval.current = setInterval(() => {
        generateEvent();
      }, 200);
    } else {
      if (streamInterval.current) {
        clearInterval(streamInterval.current);
      }
    }

    return () => {
      if (streamInterval.current) {
        clearInterval(streamInterval.current);
      }
    };
  }, [isStreaming]);

  useEffect(() => {
    if (events.length > 0) {
      calculateMetrics();
    }
  }, [events, windowType, windowSize]);

  const generateEvent = () => {
    const newEvent: StreamEvent = {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: Date.now(),
      userId: `user_${Math.floor(Math.random() * 1000)}`,
      eventType: eventTypes[Math.floor(Math.random() * eventTypes.length)],
      value: Math.floor(Math.random() * 100) + 10,
    };

    setEvents(prev => {
      const updated = [...prev, newEvent];
      // Keep only last 100 events
      return updated.slice(-100);
    });

    // Update throughput (events per second)
    setThroughput(prev => Math.min(prev + 1, 50));
    setLatency(Math.random() * 50 + 10); // Simulated latency in ms
  };

  const calculateMetrics = () => {
    const now = Date.now();
    const windowMs = windowSize * 1000;

    let windows: WindowedMetric[] = [];

    if (windowType === 'tumbling') {
      // Non-overlapping windows
      const windowStart = Math.floor(now / windowMs) * windowMs;
      const windowEvents = events.filter(e => e.timestamp >= windowStart);

      if (windowEvents.length > 0) {
        windows = [{
          window: new Date(windowStart).toLocaleTimeString(),
          count: windowEvents.length,
          sum: windowEvents.reduce((acc, e) => acc + e.value, 0),
          avg: windowEvents.reduce((acc, e) => acc + e.value, 0) / windowEvents.length,
        }];
      }
    } else if (windowType === 'sliding') {
      // Overlapping windows (slide = half of window size)
      const slideMs = windowMs / 2;
      const numWindows = 3;

      for (let i = 0; i < numWindows; i++) {
        const windowEnd = now - (i * slideMs);
        const windowStart = windowEnd - windowMs;
        const windowEvents = events.filter(e => e.timestamp >= windowStart && e.timestamp < windowEnd);

        if (windowEvents.length > 0) {
          windows.push({
            window: `${new Date(windowStart).toLocaleTimeString()} - ${new Date(windowEnd).toLocaleTimeString()}`,
            count: windowEvents.length,
            sum: windowEvents.reduce((acc, e) => acc + e.value, 0),
            avg: windowEvents.reduce((acc, e) => acc + e.value, 0) / windowEvents.length,
          });
        }
      }
    } else {
      // Session window (gap-based)
      const gapMs = 3000; // 3 second gap
      let sessionStart = events[0]?.timestamp || now;
      let sessionEvents: StreamEvent[] = [];

      events.forEach((event, idx) => {
        if (idx > 0 && event.timestamp - events[idx - 1].timestamp > gapMs) {
          // New session
          if (sessionEvents.length > 0) {
            windows.push({
              window: `Session ${new Date(sessionStart).toLocaleTimeString()}`,
              count: sessionEvents.length,
              sum: sessionEvents.reduce((acc, e) => acc + e.value, 0),
              avg: sessionEvents.reduce((acc, e) => acc + e.value, 0) / sessionEvents.length,
            });
          }
          sessionStart = event.timestamp;
          sessionEvents = [event];
        } else {
          sessionEvents.push(event);
        }
      });

      // Add last session
      if (sessionEvents.length > 0) {
        windows.push({
          window: `Session ${new Date(sessionStart).toLocaleTimeString()}`,
          count: sessionEvents.length,
          sum: sessionEvents.reduce((acc, e) => acc + e.value, 0),
          avg: sessionEvents.reduce((acc, e) => acc + e.value, 0) / sessionEvents.length,
        });
      }
    }

    setMetrics(windows.slice(0, 5));
  };

  const reset = () => {
    setIsStreaming(false);
    setEvents([]);
    setMetrics([]);
    setThroughput(0);
    setLatency(0);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 to-red-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Radio size={32} />
          <h2 className="text-2xl font-bold">실시간 스트림 처리 실험실</h2>
        </div>
        <p className="text-orange-100">
          Kafka + Flink 스타일의 윈도우 기반 스트림 처리 시뮬레이션
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">⚙️ 스트림 설정</h3>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Window Type */}
          <div>
            <label className="block text-sm font-semibold mb-2">윈도우 타입</label>
            <div className="space-y-2">
              {[
                { value: 'tumbling' as const, label: 'Tumbling Window', desc: '겹치지 않는 고정 윈도우' },
                { value: 'sliding' as const, label: 'Sliding Window', desc: '겹치는 이동 윈도우' },
                { value: 'session' as const, label: 'Session Window', desc: '갭 기반 세션 윈도우' },
              ].map((option) => (
                <button
                  key={option.value}
                  onClick={() => setWindowType(option.value)}
                  className={`w-full p-3 rounded-lg border-2 text-left transition-all ${
                    windowType === option.value
                      ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="font-semibold">{option.label}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">{option.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Window Size */}
          <div>
            <label className="block text-sm font-semibold mb-2">윈도우 크기 (초)</label>
            <input
              type="range"
              min="1"
              max="10"
              value={windowSize}
              onChange={(e) => setWindowSize(Number(e.target.value))}
              className="w-full"
            />
            <div className="text-center text-2xl font-bold text-orange-600 mt-2">{windowSize}s</div>
          </div>

          {/* Stream Controls */}
          <div className="space-y-3">
            <button
              onClick={() => setIsStreaming(!isStreaming)}
              className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                isStreaming
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
            >
              {isStreaming ? (
                <>
                  <Pause size={20} /> 스트림 중지
                </>
              ) : (
                <>
                  <Play size={20} /> 스트림 시작
                </>
              )}
            </button>
            <button
              onClick={reset}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-lg font-semibold transition-all"
            >
              <RotateCcw size={20} /> 초기화
            </button>
          </div>
        </div>
      </div>

      {/* Real-time Metrics */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="text-blue-500" />
            <h3 className="font-bold">처리량</h3>
          </div>
          <div className="text-3xl font-bold text-blue-600">{throughput.toFixed(0)}</div>
          <div className="text-sm text-gray-600 dark:text-gray-400">events/sec</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="text-yellow-500" />
            <h3 className="font-bold">레이턴시</h3>
          </div>
          <div className="text-3xl font-bold text-yellow-600">{latency.toFixed(1)}</div>
          <div className="text-sm text-gray-600 dark:text-gray-400">ms</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="text-green-500" />
            <h3 className="font-bold">총 이벤트</h3>
          </div>
          <div className="text-3xl font-bold text-green-600">{events.length}</div>
          <div className="text-sm text-gray-600 dark:text-gray-400">in buffer</div>
        </div>
      </div>

      {/* Event Stream */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">📡 실시간 이벤트 스트림</h3>
        <div className="bg-gray-900 text-green-400 rounded-lg p-4 h-48 overflow-y-auto font-mono text-xs">
          {events.slice(-20).reverse().map((event) => (
            <div key={event.id} className="mb-1">
              {new Date(event.timestamp).toLocaleTimeString()} | {event.userId} | {event.eventType} | value: {event.value}
            </div>
          ))}
          {events.length === 0 && (
            <div className="text-gray-500 text-center py-8">스트림을 시작하면 이벤트가 표시됩니다...</div>
          )}
        </div>
      </div>

      {/* Windowed Aggregations */}
      {metrics.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4">📊 윈도우별 집계 결과</h3>
          <div className="space-y-3">
            {metrics.map((metric, idx) => (
              <div key={idx} className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 p-4 rounded-lg border-l-4 border-orange-500">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{metric.window}</div>
                  <div className="flex gap-6 text-sm">
                    <span>Count: <strong className="text-blue-600">{metric.count}</strong></span>
                    <span>Sum: <strong className="text-green-600">{metric.sum.toFixed(0)}</strong></span>
                    <span>Avg: <strong className="text-purple-600">{metric.avg.toFixed(2)}</strong></span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Flink SQL Example */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">🔧 Apache Flink SQL 예제</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`-- Tumbling Window (5초 고정 윈도우)
SELECT
    TUMBLE_START(event_time, INTERVAL '${windowSize}' SECOND) as window_start,
    event_type,
    COUNT(*) as event_count,
    SUM(value) as total_value,
    AVG(value) as avg_value
FROM user_events
GROUP BY
    TUMBLE(event_time, INTERVAL '${windowSize}' SECOND),
    event_type;

-- Sliding Window (5초 윈도우, 2.5초 슬라이드)
SELECT
    HOP_START(event_time, INTERVAL '${windowSize / 2}' SECOND, INTERVAL '${windowSize}' SECOND) as window_start,
    event_type,
    COUNT(*) as event_count
FROM user_events
GROUP BY
    HOP(event_time, INTERVAL '${windowSize / 2}' SECOND, INTERVAL '${windowSize}' SECOND),
    event_type;

-- Session Window (3초 갭)
SELECT
    SESSION_START(event_time, INTERVAL '3' SECOND) as session_start,
    user_id,
    COUNT(*) as events_in_session
FROM user_events
GROUP BY
    SESSION(event_time, INTERVAL '3' SECOND),
    user_id;`}
        </pre>
      </div>
    </div>
  );
}
