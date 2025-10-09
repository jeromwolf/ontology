'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Users, Brain, MessageCircle, Zap, Target, CheckCircle2, AlertCircle, Play, Pause, RefreshCw, Plus, Settings, BarChart3, Network } from 'lucide-react';

interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'idle' | 'thinking' | 'working' | 'communicating' | 'done' | 'error';
  color: string;
  progress: number;
  x?: number;
  y?: number;
  tasksCompleted?: number;
  avgResponseTime?: number;
}

interface AgentMessage {
  from: string;
  to: string;
  content: string;
  timestamp: number;
  type: 'request' | 'response' | 'notification';
}

interface WorkflowPattern {
  id: string;
  name: string;
  description: string;
  agents: string[];
  steps: WorkflowStep[];
}

interface WorkflowStep {
  agentId: string;
  duration: number;
  nextAgents: string[];
  message: string;
  condition?: string;
}

interface Metrics {
  totalTasks: number;
  successRate: number;
  avgResponseTime: number;
  messagesPerSecond: number;
  activeAgents: number;
}

const workflowPatterns: WorkflowPattern[] = [
  {
    id: 'sequential',
    name: '순차 처리 (Sequential)',
    description: '단일 경로로 순서대로 처리',
    agents: ['researcher', 'analyzer', 'writer', 'reviewer'],
    steps: [
      { agentId: 'researcher', duration: 2000, nextAgents: ['analyzer'], message: '데이터 수집 완료 (100개 문서)' },
      { agentId: 'analyzer', duration: 2500, nextAgents: ['writer'], message: '3가지 핵심 인사이트 도출' },
      { agentId: 'writer', duration: 2000, nextAgents: ['reviewer'], message: '2페이지 요약 보고서 생성' },
      { agentId: 'reviewer', duration: 1500, nextAgents: [], message: '품질 검토 완료 (95/100)' }
    ]
  },
  {
    id: 'parallel',
    name: '병렬 처리 (Parallel)',
    description: '여러 에이전트가 동시에 작업',
    agents: ['researcher', 'analyzer', 'coder', 'designer', 'writer'],
    steps: [
      { agentId: 'researcher', duration: 2000, nextAgents: ['analyzer', 'coder', 'designer'], message: '요구사항 분석 완료' },
      { agentId: 'analyzer', duration: 2500, nextAgents: ['writer'], message: '데이터 분석 완료' },
      { agentId: 'coder', duration: 3000, nextAgents: ['writer'], message: '프로토타입 구현 완료' },
      { agentId: 'designer', duration: 2800, nextAgents: ['writer'], message: 'UI 디자인 완료' },
      { agentId: 'writer', duration: 1500, nextAgents: [], message: '통합 문서 작성 완료' }
    ]
  },
  {
    id: 'conditional',
    name: '조건 분기 (Conditional)',
    description: '조건에 따라 다른 경로로 분기',
    agents: ['validator', 'processor', 'advanced', 'simple', 'finalizer'],
    steps: [
      { agentId: 'validator', duration: 1500, nextAgents: ['processor'], message: '입력 데이터 검증 완료' },
      { agentId: 'processor', duration: 2000, nextAgents: ['advanced', 'simple'], message: '복잡도 분석 완료', condition: 'complexity > 0.5' },
      { agentId: 'advanced', duration: 3500, nextAgents: ['finalizer'], message: '고급 처리 완료' },
      { agentId: 'simple', duration: 1500, nextAgents: ['finalizer'], message: '간단 처리 완료' },
      { agentId: 'finalizer', duration: 1000, nextAgents: [], message: '최종 검증 완료' }
    ]
  },
  {
    id: 'microservices',
    name: '마이크로서비스 (Microservices)',
    description: '독립적인 서비스 간 협업',
    agents: ['gateway', 'auth', 'business', 'database', 'cache', 'logger'],
    steps: [
      { agentId: 'gateway', duration: 500, nextAgents: ['auth'], message: 'API 요청 수신' },
      { agentId: 'auth', duration: 800, nextAgents: ['business'], message: '인증 성공 (user_123)' },
      { agentId: 'business', duration: 1500, nextAgents: ['database', 'cache'], message: '비즈니스 로직 처리' },
      { agentId: 'database', duration: 2000, nextAgents: ['logger'], message: 'DB 쿼리 완료 (50ms)' },
      { agentId: 'cache', duration: 200, nextAgents: ['logger'], message: '캐시 업데이트 완료' },
      { agentId: 'logger', duration: 300, nextAgents: [], message: '로그 기록 완료' }
    ]
  }
];

const agentTypes = [
  { id: 'researcher', name: 'Researcher', role: '정보 수집', color: 'blue' },
  { id: 'analyzer', name: 'Analyzer', role: '데이터 분석', color: 'green' },
  { id: 'writer', name: 'Writer', role: '보고서 작성', color: 'purple' },
  { id: 'reviewer', name: 'Reviewer', role: '품질 검토', color: 'orange' },
  { id: 'coder', name: 'Coder', role: '코드 구현', color: 'indigo' },
  { id: 'designer', name: 'Designer', role: 'UI 디자인', color: 'pink' },
  { id: 'validator', name: 'Validator', role: '검증', color: 'red' },
  { id: 'processor', name: 'Processor', role: '처리', color: 'yellow' },
  { id: 'advanced', name: 'Advanced', role: '고급 처리', color: 'teal' },
  { id: 'simple', name: 'Simple', role: '간단 처리', color: 'cyan' },
  { id: 'finalizer', name: 'Finalizer', role: '최종화', color: 'lime' },
  { id: 'gateway', name: 'Gateway', role: 'API 게이트웨이', color: 'violet' },
  { id: 'auth', name: 'Auth', role: '인증 서비스', color: 'fuchsia' },
  { id: 'business', name: 'Business', role: '비즈니스 로직', color: 'rose' },
  { id: 'database', name: 'Database', role: '데이터베이스', color: 'amber' },
  { id: 'cache', name: 'Cache', role: '캐시 서비스', color: 'emerald' },
  { id: 'logger', name: 'Logger', role: '로깅 서비스', color: 'slate' }
];

export default function A2AOrchestrator() {
  const [selectedPattern, setSelectedPattern] = useState<WorkflowPattern>(workflowPatterns[0]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [task, setTask] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [showMetrics, setShowMetrics] = useState(true);
  const [showNetwork, setShowNetwork] = useState(false);
  const [metrics, setMetrics] = useState<Metrics>({
    totalTasks: 0,
    successRate: 100,
    avgResponseTime: 0,
    messagesPerSecond: 0,
    activeAgents: 0
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    initializeAgents(selectedPattern);
  }, [selectedPattern]);

  useEffect(() => {
    if (showNetwork && canvasRef.current) {
      drawNetwork();
    }
  }, [agents, messages, showNetwork]);

  const initializeAgents = (pattern: WorkflowPattern) => {
    const newAgents = pattern.agents.map((agentId, index) => {
      const agentType = agentTypes.find(t => t.id === agentId) || agentTypes[0];
      const angle = (2 * Math.PI * index) / pattern.agents.length;
      const radius = 120;
      return {
        id: agentId,
        name: agentType.name,
        role: agentType.role,
        status: 'idle' as const,
        color: agentType.color,
        progress: 0,
        x: 200 + radius * Math.cos(angle),
        y: 150 + radius * Math.sin(angle),
        tasksCompleted: 0,
        avgResponseTime: 0
      };
    });
    setAgents(newAgents);
    setMessages([]);
  };

  const drawNetwork = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Make canvas responsive
    canvas.width = canvas.offsetWidth;
    canvas.height = 400;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections (messages)
    messages.slice(-10).forEach((msg, index) => {
      const fromAgent = agents.find(a => a.id === msg.from);
      const toAgent = agents.find(a => a.id === msg.to);

      if (fromAgent && toAgent && fromAgent.x && fromAgent.y && toAgent.x && toAgent.y) {
        const opacity = 1 - (index / 10) * 0.7;
        ctx.strokeStyle = `rgba(249, 115, 22, ${opacity})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(fromAgent.x, fromAgent.y);
        ctx.lineTo(toAgent.x, toAgent.y);
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(toAgent.y - fromAgent.y, toAgent.x - fromAgent.x);
        const arrowLength = 10;
        ctx.fillStyle = `rgba(249, 115, 22, ${opacity})`;
        ctx.beginPath();
        ctx.moveTo(
          toAgent.x - arrowLength * Math.cos(angle - Math.PI / 6),
          toAgent.y - arrowLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(toAgent.x, toAgent.y);
        ctx.lineTo(
          toAgent.x - arrowLength * Math.cos(angle + Math.PI / 6),
          toAgent.y - arrowLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.fill();
      }
    });

    // Draw agents as nodes
    agents.forEach(agent => {
      if (agent.x && agent.y) {
        // Node circle
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, 25, 0, 2 * Math.PI);

        let fillColor = '#e5e7eb'; // idle
        if (agent.status === 'working' || agent.status === 'thinking') fillColor = '#fbbf24'; // yellow
        if (agent.status === 'communicating') fillColor = '#3b82f6'; // blue
        if (agent.status === 'done') fillColor = '#10b981'; // green
        if (agent.status === 'error') fillColor = '#ef4444'; // red

        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Node label
        ctx.fillStyle = '#1f2937';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(agent.name, agent.x, agent.y + 45);
      }
    });
  };

  const startWorkflow = async () => {
    if (!task.trim()) return;

    setIsRunning(true);
    setIsPaused(false);
    setMessages([]);

    // Reset agents
    setAgents(agents.map(a => ({ ...a, status: 'idle', progress: 0 })));

    const startTime = Date.now();
    let messageCount = 0;

    for (const step of selectedPattern.steps) {
      if (isPaused) {
        await new Promise(resolve => {
          const checkInterval = setInterval(() => {
            if (!isPaused) {
              clearInterval(checkInterval);
              resolve(null);
            }
          }, 100);
        });
      }

      // Start agent work
      setAgents(prev => prev.map(a =>
        a.id === step.agentId
          ? { ...a, status: 'thinking', progress: 0 }
          : a
      ));

      await new Promise(resolve => setTimeout(resolve, 500));

      // Agent working
      setAgents(prev => prev.map(a =>
        a.id === step.agentId
          ? { ...a, status: 'working', progress: 50 }
          : a
      ));

      await new Promise(resolve => setTimeout(resolve, step.duration / 2));

      // Agent communicating
      if (step.nextAgents.length > 0) {
        setAgents(prev => prev.map(a =>
          a.id === step.agentId
            ? { ...a, status: 'communicating', progress: 80 }
            : a
        ));

        for (const nextAgent of step.nextAgents) {
          const msg: AgentMessage = {
            from: step.agentId,
            to: nextAgent,
            content: step.message,
            timestamp: Date.now(),
            type: 'request'
          };
          setMessages(prev => [...prev, msg]);
          messageCount++;
        }
      }

      await new Promise(resolve => setTimeout(resolve, 500));

      // Agent done
      setAgents(prev => prev.map(a =>
        a.id === step.agentId
          ? { ...a, status: 'done', progress: 100, tasksCompleted: (a.tasksCompleted || 0) + 1 }
          : a
      ));
    }

    const totalTime = Date.now() - startTime;
    const avgTime = totalTime / selectedPattern.steps.length;
    const mps = (messageCount / (totalTime / 1000)).toFixed(2);

    setMetrics({
      totalTasks: metrics.totalTasks + 1,
      successRate: 100,
      avgResponseTime: Math.round(avgTime),
      messagesPerSecond: parseFloat(mps),
      activeAgents: selectedPattern.agents.length
    });

    setIsRunning(false);
  };

  const resetWorkflow = () => {
    setIsRunning(false);
    setIsPaused(false);
    setMessages([]);
    setTask('');
    initializeAgents(selectedPattern);
  };

  const getStatusIcon = (status: Agent['status']) => {
    switch(status) {
      case 'thinking':
        return <Brain className="w-4 h-4 animate-pulse" />;
      case 'working':
        return <Zap className="w-4 h-4 animate-spin" />;
      case 'communicating':
        return <MessageCircle className="w-4 h-4" />;
      case 'done':
        return <CheckCircle2 className="w-4 h-4" />;
      case 'error':
        return <AlertCircle className="w-4 h-4" />;
      default:
        return <Target className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-gradient-to-br from-orange-900 via-red-900 to-orange-900 rounded-xl p-6 text-white">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">A2A (Agent-to-Agent) Orchestrator</h3>
        <p className="text-orange-200">에이전트 간 통신과 협업 워크플로우를 시각화합니다</p>
      </div>

      {/* Pattern Selection */}
      <div className="mb-6">
        <h4 className="text-sm font-semibold text-orange-200 mb-2">워크플로우 패턴 선택</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {workflowPatterns.map(pattern => (
            <button
              key={pattern.id}
              onClick={() => {
                setSelectedPattern(pattern);
                resetWorkflow();
              }}
              disabled={isRunning}
              className={`p-3 rounded-lg text-left transition-colors ${
                selectedPattern.id === pattern.id
                  ? 'bg-orange-600'
                  : 'bg-orange-800 hover:bg-orange-700'
              } disabled:opacity-50`}
            >
              <div className="font-semibold text-sm">{pattern.name}</div>
              <div className="text-xs text-orange-200 mt-1">{pattern.description}</div>
              <div className="text-xs text-orange-300 mt-1">{pattern.agents.length} agents</div>
            </button>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="mb-6">
        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="작업을 입력하세요... (예: AI 트렌드 리포트 작성)"
            className="flex-1 px-4 py-2 bg-orange-800 border border-orange-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500 text-white placeholder-orange-300"
            disabled={isRunning}
          />
          <button
            onClick={startWorkflow}
            disabled={isRunning || !task.trim()}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 rounded-lg flex items-center gap-2 transition-colors"
          >
            <Play className="w-4 h-4" />
            실행
          </button>
          <button
            onClick={resetWorkflow}
            disabled={isRunning}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 rounded-lg flex items-center gap-2 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            리셋
          </button>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className={`px-3 py-1 rounded-lg text-sm flex items-center gap-2 ${
              showMetrics ? 'bg-orange-600' : 'bg-orange-800'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            메트릭
          </button>
          <button
            onClick={() => setShowNetwork(!showNetwork)}
            className={`px-3 py-1 rounded-lg text-sm flex items-center gap-2 ${
              showNetwork ? 'bg-orange-600' : 'bg-orange-800'
            }`}
          >
            <Network className="w-4 h-4" />
            네트워크
          </button>
        </div>
      </div>

      {/* Metrics Dashboard */}
      {showMetrics && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
          <div className="bg-orange-800/50 rounded-lg p-3">
            <div className="text-xs text-orange-300 mb-1">완료 작업</div>
            <div className="text-2xl font-bold">{metrics.totalTasks}</div>
          </div>
          <div className="bg-orange-800/50 rounded-lg p-3">
            <div className="text-xs text-orange-300 mb-1">성공률</div>
            <div className="text-2xl font-bold">{metrics.successRate}%</div>
          </div>
          <div className="bg-orange-800/50 rounded-lg p-3">
            <div className="text-xs text-orange-300 mb-1">평균 응답시간</div>
            <div className="text-2xl font-bold">{metrics.avgResponseTime}ms</div>
          </div>
          <div className="bg-orange-800/50 rounded-lg p-3">
            <div className="text-xs text-orange-300 mb-1">메시지/초</div>
            <div className="text-2xl font-bold">{metrics.messagesPerSecond}</div>
          </div>
          <div className="bg-orange-800/50 rounded-lg p-3">
            <div className="text-xs text-orange-300 mb-1">활성 에이전트</div>
            <div className="text-2xl font-bold">{metrics.activeAgents}</div>
          </div>
        </div>
      )}

      {/* Network Visualization */}
      {showNetwork && (
        <div className="mb-6 bg-white rounded-lg p-4">
          <canvas
            ref={canvasRef}
            className="w-full h-[400px]"
          />
        </div>
      )}

      {/* Agents Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 mb-6">
        {agents.map(agent => (
          <div
            key={agent.id}
            className={`p-3 rounded-lg border-2 transition-all ${
              agent.status !== 'idle'
                ? 'border-orange-400 shadow-lg bg-orange-800/50'
                : 'border-orange-700 bg-orange-800/30'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4" />
                <span className="font-semibold text-sm">{agent.name}</span>
              </div>
              {getStatusIcon(agent.status)}
            </div>
            <p className="text-xs text-orange-200 mb-2">{agent.role}</p>
            <div className="w-full bg-orange-900 rounded-full h-1.5 mb-1">
              <div
                className="bg-orange-400 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${agent.progress}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-orange-300">
              <span>작업: {agent.tasksCompleted || 0}</span>
              <span className="capitalize">{agent.status}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Communication Log */}
      {messages.length > 0 && (
        <div className="bg-orange-800/30 rounded-lg p-4">
          <h4 className="text-sm font-semibold mb-3">통신 로그 ({messages.length} messages)</h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {messages.slice(-15).map((msg, index) => (
              <div key={index} className="flex items-center gap-2 text-sm bg-orange-900/30 rounded p-2">
                <span className="text-orange-300 font-medium text-xs">{msg.from}</span>
                <span className="text-orange-500">→</span>
                <span className="text-orange-300 font-medium text-xs">{msg.to}</span>
                <span className="text-orange-500">:</span>
                <span className="text-orange-100 flex-1 text-xs">{msg.content}</span>
                <span className="text-xs text-orange-400">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
