'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  Users, Vote, Gavel, TrendingUp, TrendingDown,
  CheckCircle, XCircle, Clock, Play, RefreshCw,
  BarChart, PieChart, Activity, AlertTriangle,
  Zap, Shield, Network, MessageSquare, Pause,
  WifiOff, AlertCircle, Crown
} from 'lucide-react';

// Types
interface Node {
  id: string;
  name: string;
  role: 'follower' | 'candidate' | 'leader';
  term: number;
  votedFor?: string;
  log: LogEntry[];
  commitIndex: number;
  isFaulty: boolean;
  isDown: boolean;
  x?: number;
  y?: number;
  voteReceived?: boolean;
}

interface LogEntry {
  term: number;
  command: string;
  index: number;
}

interface Message {
  from: string;
  to: string;
  type: 'RequestVote' | 'VoteResponse' | 'AppendEntries' | 'Propose' | 'Accept' | 'Commit';
  term: number;
  value?: string;
  granted?: boolean;
}

interface Metrics {
  totalMessages: number;
  consensusTime: number;
  leaderElections: number;
  failedNodes: number;
  successRate: number;
}

interface AlgorithmConfig {
  id: string;
  name: string;
  description: string;
  faultTolerance: string;
  complexity: string;
}

const ALGORITHMS: AlgorithmConfig[] = [
  {
    id: 'raft',
    name: 'Raft Consensus',
    description: '리더 선출과 로그 복제를 통한 합의 알고리즘',
    faultTolerance: 'f < n/2 (과반수 필요)',
    complexity: 'O(n) 메시지 복잡도'
  },
  {
    id: 'paxos',
    name: 'Paxos',
    description: '2단계 프로토콜로 안정적인 합의 달성',
    faultTolerance: 'f < n/2 (다수결)',
    complexity: 'O(n²) 메시지 복잡도'
  },
  {
    id: 'pbft',
    name: 'Practical Byzantine Fault Tolerance',
    description: '악의적 노드가 있어도 합의 가능',
    faultTolerance: 'f < n/3 (2/3 이상 필요)',
    complexity: 'O(n²) 메시지 복잡도'
  },
  {
    id: 'pow',
    name: 'Proof of Work',
    description: '계산 난이도 기반 합의 메커니즘',
    faultTolerance: '51% 공격 저항',
    complexity: '높은 계산 비용'
  }
];

const SCENARIOS = [
  { id: 'normal', name: '정상 작동', description: '모든 노드가 정상 작동' },
  { id: 'single-failure', name: '단일 노드 장애', description: '1개 노드 다운' },
  { id: 'network-partition', name: '네트워크 분리', description: '네트워크가 2개 파티션으로 분리' },
  { id: 'byzantine', name: '악의적 노드', description: '1개 노드가 잘못된 값 전파' },
  { id: 'leader-crash', name: '리더 장애', description: '리더 노드 갑작스런 중단' }
];

export default function ConsensusSimulator() {
  // State
  const [nodes, setNodes] = useState<Node[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(ALGORITHMS[0]);
  const [selectedScenario, setSelectedScenario] = useState(SCENARIOS[0]);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentPhase, setCurrentPhase] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [metrics, setMetrics] = useState<Metrics>({
    totalMessages: 0,
    consensusTime: 0,
    leaderElections: 0,
    failedNodes: 0,
    successRate: 0
  });
  const [executionLog, setExecutionLog] = useState<string[]>([]);
  const [showNetwork, setShowNetwork] = useState(true);
  const [showMetrics, setShowMetrics] = useState(true);
  const [consensusValue, setConsensusValue] = useState<string>('');
  const [consensusReached, setConsensusReached] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize nodes
  useEffect(() => {
    const nodeCount = selectedAlgorithm.id === 'pbft' ? 7 : 5;
    const initialNodes: Node[] = Array.from({ length: nodeCount }, (_, i) => ({
      id: `node-${i + 1}`,
      name: `Node ${i + 1}`,
      role: i === 0 ? 'leader' : 'follower',
      term: 0,
      log: [],
      commitIndex: 0,
      isFaulty: false,
      isDown: false
    }));
    setNodes(initialNodes);
  }, [selectedAlgorithm]);

  // Draw network visualization
  useEffect(() => {
    if (!showNetwork || !canvasRef.current || nodes.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.offsetWidth;
    canvas.height = 400;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate node positions in circle
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.35;

    const nodesWithPos = nodes.map((node, index) => {
      const angle = (index / nodes.length) * 2 * Math.PI - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      return { ...node, x, y };
    });

    // Draw connections (messages)
    messages.slice(-5).forEach((msg, index) => {
      const fromNode = nodesWithPos.find(n => n.id === msg.from);
      const toNode = nodesWithPos.find(n => n.id === msg.to);

      if (fromNode && toNode && fromNode.x && fromNode.y && toNode.x && toNode.y) {
        const opacity = 1 - (index / 5) * 0.6;
        ctx.strokeStyle = `rgba(249, 115, 22, ${opacity})`;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Arrow
        const angle = Math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x);
        const arrowLength = 10;
        ctx.fillStyle = `rgba(249, 115, 22, ${opacity})`;
        ctx.beginPath();
        ctx.moveTo(
          toNode.x - arrowLength * Math.cos(angle - Math.PI / 6),
          toNode.y - arrowLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(toNode.x, toNode.y);
        ctx.lineTo(
          toNode.x - arrowLength * Math.cos(angle + Math.PI / 6),
          toNode.y - arrowLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.fill();
      }
    });

    // Draw nodes
    nodesWithPos.forEach(node => {
      if (!node.x || !node.y) return;

      ctx.beginPath();
      ctx.arc(node.x, node.y, 35, 0, 2 * Math.PI);

      // Color based on role and status
      let fillColor = '#3b82f6'; // follower
      if (node.isDown) fillColor = '#6b7280';
      else if (node.isFaulty) fillColor = '#ef4444';
      else if (node.role === 'leader') fillColor = '#f59e0b';
      else if (node.role === 'candidate') fillColor = '#8b5cf6';

      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = node.voteReceived ? 5 : 3;
      ctx.stroke();

      // Leader crown
      if (node.role === 'leader' && !node.isDown) {
        ctx.fillStyle = '#fbbf24';
        ctx.font = '20px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('👑', node.x, node.y - 45);
      }

      // Node label
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.name.split(' ')[1], node.x, node.y);

      // Role label
      ctx.fillStyle = '#1f2937';
      ctx.font = '11px sans-serif';
      ctx.fillText(
        node.isDown ? 'DOWN' : node.isFaulty ? 'FAULTY' : node.role.toUpperCase(),
        node.x,
        node.y + 50
      );

      // Term badge
      if (node.term > 0 && selectedAlgorithm.id === 'raft') {
        ctx.fillStyle = '#374151';
        ctx.font = '10px sans-serif';
        ctx.fillText(`T${node.term}`, node.x, node.y + 65);
      }
    });

  }, [nodes, messages, showNetwork, selectedAlgorithm]);

  const log = (message: string) => {
    setExecutionLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
  };

  // Raft Leader Election
  const runRaftLeaderElection = async () => {
    log('🚀 Raft 리더 선출 시작...');
    setCurrentPhase('리더 선출');

    // Reset all nodes to followers
    setNodes(prev => prev.map(n => ({
      ...n,
      role: 'follower',
      term: 0,
      votedFor: undefined,
      voteReceived: false
    })));

    await new Promise(resolve => setTimeout(resolve, 500));

    // Random node becomes candidate
    const candidateIndex = Math.floor(Math.random() * nodes.length);
    const candidateId = `node-${candidateIndex + 1}`;

    log(`📢 ${candidateId}이(가) Candidate로 전환 (Term 1)`);
    setNodes(prev => prev.map(n =>
      n.id === candidateId ? { ...n, role: 'candidate', term: 1, votedFor: n.id } : { ...n, term: 1 }
    ));

    await new Promise(resolve => setTimeout(resolve, 800));

    // Request votes
    let votesReceived = 1; // self-vote
    setCurrentPhase('투표 요청');
    log(`🗳️  ${candidateId}이(가) RequestVote 메시지 전송`);

    for (const node of nodes) {
      if (node.id === candidateId || node.isDown) continue;

      const message: Message = {
        from: candidateId,
        to: node.id,
        type: 'RequestVote',
        term: 1
      };
      setMessages(prev => [...prev, message]);
      await new Promise(resolve => setTimeout(resolve, 300));

      // Vote response
      if (!node.isFaulty && Math.random() > 0.2) {
        const response: Message = {
          from: node.id,
          to: candidateId,
          type: 'VoteResponse',
          term: 1,
          granted: true
        };
        setMessages(prev => [...prev, response]);
        setNodes(prev => prev.map(n =>
          n.id === node.id ? { ...n, votedFor: candidateId, voteReceived: true } : n
        ));
        votesReceived++;
        log(`✅ ${node.id}가 ${candidateId}에게 투표`);
      } else {
        log(`❌ ${node.id}가 투표 거부`);
      }

      await new Promise(resolve => setTimeout(resolve, 200));
    }

    // Check majority
    const majority = Math.floor(nodes.filter(n => !n.isDown).length / 2) + 1;

    if (votesReceived >= majority) {
      setCurrentPhase('리더 당선');
      log(`👑 ${candidateId}가 리더로 선출됨 (${votesReceived}/${nodes.length} 투표)`);
      setNodes(prev => prev.map(n =>
        n.id === candidateId ? { ...n, role: 'leader' } : { ...n, role: 'follower' }
      ));
      setMetrics(prev => ({
        ...prev,
        leaderElections: prev.leaderElections + 1
      }));
      return candidateId;
    } else {
      log(`⚠️  과반수 확보 실패 - 재선거 필요`);
      return null;
    }
  };

  // Raft Log Replication
  const runRaftLogReplication = async (leaderId: string, value: string) => {
    setCurrentPhase('로그 복제');
    log(`📝 리더가 명령 제안: "${value}"`);

    const logEntry: LogEntry = {
      term: 1,
      command: value,
      index: 1
    };

    // Leader appends to own log
    setNodes(prev => prev.map(n =>
      n.id === leaderId ? { ...n, log: [...n.log, logEntry] } : n
    ));

    await new Promise(resolve => setTimeout(resolve, 500));

    // Send AppendEntries to followers
    let replicationCount = 1; // leader itself

    for (const node of nodes) {
      if (node.id === leaderId || node.isDown) continue;

      const message: Message = {
        from: leaderId,
        to: node.id,
        type: 'AppendEntries',
        term: 1,
        value: value
      };
      setMessages(prev => [...prev, message]);
      await new Promise(resolve => setTimeout(resolve, 300));

      if (!node.isFaulty) {
        setNodes(prev => prev.map(n =>
          n.id === node.id ? { ...n, log: [...n.log, logEntry] } : n
        ));
        replicationCount++;
        log(`✅ ${node.id}가 로그 복제 완료`);
      } else {
        log(`❌ ${node.id}가 로그 복제 실패 (악의적 노드)`);
      }

      await new Promise(resolve => setTimeout(resolve, 200));
    }

    const majority = Math.floor(nodes.filter(n => !n.isDown).length / 2) + 1;

    if (replicationCount >= majority) {
      setCurrentPhase('커밋');
      log(`🎉 과반수 복제 완료 - 로그 커밋 (${replicationCount}/${nodes.length})`);
      setNodes(prev => prev.map(n => ({
        ...n,
        commitIndex: 1
      })));
      setConsensusValue(value);
      setConsensusReached(true);
      return true;
    } else {
      log(`⚠️  과반수 미달 - 커밋 실패`);
      return false;
    }
  };

  // Paxos Algorithm
  const runPaxos = async (value: string) => {
    log('🚀 Paxos 합의 시작...');

    // Phase 1: Prepare
    setCurrentPhase('Phase 1: Prepare');
    const proposerId = nodes[0].id;
    const proposalNumber = Date.now();

    log(`📢 Proposer ${proposerId}가 Prepare(${proposalNumber}) 전송`);

    let promiseCount = 0;

    for (const node of nodes) {
      if (node.id === proposerId || node.isDown) continue;

      const message: Message = {
        from: proposerId,
        to: node.id,
        type: 'Propose',
        term: proposalNumber,
        value: value
      };
      setMessages(prev => [...prev, message]);
      await new Promise(resolve => setTimeout(resolve, 300));

      if (!node.isFaulty) {
        promiseCount++;
        log(`✅ ${node.id}가 Promise 응답`);
      }

      await new Promise(resolve => setTimeout(resolve, 200));
    }

    const majority = Math.floor(nodes.filter(n => !n.isDown).length / 2) + 1;

    if (promiseCount >= majority - 1) {
      // Phase 2: Accept
      setCurrentPhase('Phase 2: Accept');
      log(`📝 Phase 2: Accept 요청 전송`);

      let acceptCount = 1; // self

      for (const node of nodes) {
        if (node.id === proposerId || node.isDown) continue;

        const message: Message = {
          from: proposerId,
          to: node.id,
          type: 'Accept',
          term: proposalNumber,
          value: value
        };
        setMessages(prev => [...prev, message]);
        await new Promise(resolve => setTimeout(resolve, 300));

        if (!node.isFaulty) {
          acceptCount++;
          log(`✅ ${node.id}가 Accept 완료`);
        }

        await new Promise(resolve => setTimeout(resolve, 200));
      }

      if (acceptCount >= majority) {
        setCurrentPhase('합의 완료');
        log(`🎉 Paxos 합의 성공: "${value}"`);
        setConsensusValue(value);
        setConsensusReached(true);
        return true;
      }
    }

    log(`⚠️  합의 실패`);
    return false;
  };

  // PBFT Algorithm
  const runPBFT = async (value: string) => {
    log('🚀 PBFT 합의 시작...');

    setCurrentPhase('Pre-Prepare');
    const primaryId = nodes.find(n => n.role === 'leader')?.id || nodes[0].id;

    log(`📢 Primary ${primaryId}가 Pre-Prepare 메시지 전송`);

    // Pre-Prepare phase
    for (const node of nodes) {
      if (node.id === primaryId || node.isDown) continue;

      const message: Message = {
        from: primaryId,
        to: node.id,
        type: 'Propose',
        term: 1,
        value: value
      };
      setMessages(prev => [...prev, message]);
      await new Promise(resolve => setTimeout(resolve, 200));
    }

    await new Promise(resolve => setTimeout(resolve, 500));

    // Prepare phase
    setCurrentPhase('Prepare');
    log(`📝 Prepare 단계 - 노드 간 메시지 교환`);

    let prepareCount = 0;
    const requiredVotes = Math.floor((nodes.filter(n => !n.isDown).length * 2) / 3) + 1;

    for (const node of nodes) {
      if (node.isDown || node.isFaulty) continue;
      prepareCount++;
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    if (prepareCount >= requiredVotes) {
      // Commit phase
      setCurrentPhase('Commit');
      log(`✅ Prepare 완료 (${prepareCount}/${nodes.length})`);
      log(`📝 Commit 단계 시작`);

      let commitCount = 0;
      for (const node of nodes) {
        if (node.isDown || node.isFaulty) continue;
        commitCount++;
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      if (commitCount >= requiredVotes) {
        setCurrentPhase('합의 완료');
        log(`🎉 PBFT 합의 성공: "${value}" (2/3 이상 동의)`);
        setConsensusValue(value);
        setConsensusReached(true);
        return true;
      }
    }

    log(`⚠️  합의 실패 - 2/3 미달`);
    return false;
  };

  // Apply scenario
  const applyScenario = () => {
    setNodes(prev => prev.map(n => ({
      ...n,
      isDown: false,
      isFaulty: false,
      role: n.id === 'node-1' ? 'leader' : 'follower'
    })));

    switch (selectedScenario.id) {
      case 'single-failure':
        setNodes(prev => prev.map((n, i) =>
          i === prev.length - 1 ? { ...n, isDown: true } : n
        ));
        log(`⚠️  시나리오: ${prev.length - 1}번 노드 다운`);
        break;

      case 'network-partition':
        const half = Math.floor(nodes.length / 2);
        setNodes(prev => prev.map((n, i) =>
          i >= half ? { ...n, isDown: true } : n
        ));
        log(`⚠️  시나리오: 네트워크 파티션 (${half}개 노드 격리)`);
        break;

      case 'byzantine':
        setNodes(prev => prev.map((n, i) =>
          i === 1 ? { ...n, isFaulty: true } : n
        ));
        log(`⚠️  시나리오: node-2가 악의적 노드로 설정`);
        break;

      case 'leader-crash':
        setNodes(prev => prev.map(n =>
          n.role === 'leader' ? { ...n, isDown: true } : n
        ));
        log(`⚠️  시나리오: 리더 노드 크래시`);
        break;

      default:
        log(`✅ 시나리오: 정상 작동`);
    }
  };

  // Run simulation
  const runSimulation = async () => {
    setIsRunning(true);
    setIsPaused(false);
    setConsensusReached(false);
    setConsensusValue('');
    setMessages([]);
    setExecutionLog([]);
    setMetrics({
      totalMessages: 0,
      consensusTime: 0,
      leaderElections: 0,
      failedNodes: 0,
      successRate: 0
    });

    const startTime = Date.now();
    const value = `Command-${Date.now()}`;

    applyScenario();
    await new Promise(resolve => setTimeout(resolve, 1000));

    let success = false;

    try {
      if (selectedAlgorithm.id === 'raft') {
        const leaderId = await runRaftLeaderElection();
        if (leaderId) {
          success = await runRaftLogReplication(leaderId, value);
        }
      } else if (selectedAlgorithm.id === 'paxos') {
        success = await runPaxos(value);
      } else if (selectedAlgorithm.id === 'pbft') {
        success = await runPBFT(value);
      } else if (selectedAlgorithm.id === 'pow') {
        setCurrentPhase('마이닝');
        log('⛏️  Proof of Work 마이닝 시작...');
        await new Promise(resolve => setTimeout(resolve, 3000));
        log('🎉 블록 채굴 성공!');
        setConsensusValue(value);
        setConsensusReached(true);
        success = true;
      }

      const endTime = Date.now();
      const failedCount = nodes.filter(n => n.isDown).length;

      setMetrics({
        totalMessages: messages.length,
        consensusTime: endTime - startTime,
        leaderElections: selectedAlgorithm.id === 'raft' ? 1 : 0,
        failedNodes: failedCount,
        successRate: success ? 100 : 0
      });

      if (success) {
        log(`✨ 합의 완료! 총 시간: ${((endTime - startTime) / 1000).toFixed(2)}s`);
      } else {
        log(`❌ 합의 실패`);
      }

    } catch (error) {
      log(`⚠️  에러 발생: ${error}`);
    }

    setIsRunning(false);
    setCurrentPhase('');
  };

  const resetSimulation = () => {
    setConsensusReached(false);
    setConsensusValue('');
    setMessages([]);
    setExecutionLog([]);
    setCurrentPhase('');
    setNodes(prev => prev.map(n => ({
      ...n,
      role: n.id === 'node-1' ? 'leader' : 'follower',
      term: 0,
      votedFor: undefined,
      log: [],
      commitIndex: 0,
      isDown: false,
      isFaulty: false,
      voteReceived: false
    })));
  };

  return (
    <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
          🔄 Distributed Consensus Simulator
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Raft, Paxos, PBFT 등 분산 합의 알고리즘을 실시간으로 시뮬레이션하고 학습하세요
        </p>

        {/* Toggle Controls */}
        <div className="flex flex-wrap gap-2 mb-4">
          <button
            onClick={() => setShowNetwork(!showNetwork)}
            className={`px-3 py-1.5 rounded-lg transition-colors text-sm ${
              showNetwork
                ? 'bg-orange-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            <Network className="w-4 h-4 inline mr-1" />
            네트워크 시각화
          </button>
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className={`px-3 py-1.5 rounded-lg transition-colors text-sm ${
              showMetrics
                ? 'bg-orange-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            <BarChart className="w-4 h-4 inline mr-1" />
            성능 지표
          </button>
        </div>
      </div>

      {/* Metrics Dashboard */}
      {showMetrics && metrics.totalMessages > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">메시지 수</div>
            <div className="text-3xl font-bold">{metrics.totalMessages}</div>
          </div>
          <div className="bg-gradient-to-br from-green-500 to-green-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">합의 시간</div>
            <div className="text-3xl font-bold">{(metrics.consensusTime / 1000).toFixed(1)}s</div>
          </div>
          <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">리더 선출</div>
            <div className="text-3xl font-bold">{metrics.leaderElections}</div>
          </div>
          <div className="bg-gradient-to-br from-red-500 to-red-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">장애 노드</div>
            <div className="text-3xl font-bold">{metrics.failedNodes}</div>
          </div>
          <div className="bg-gradient-to-br from-orange-500 to-orange-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">성공률</div>
            <div className="text-3xl font-bold">{metrics.successRate}%</div>
          </div>
        </div>
      )}

      {/* Network Visualization */}
      {showNetwork && nodes.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              네트워크 토폴로지
            </h4>
            {currentPhase && (
              <div className="px-3 py-1 bg-orange-100 dark:bg-orange-900/30 rounded-full text-xs font-semibold text-orange-700 dark:text-orange-300">
                {currentPhase}
              </div>
            )}
          </div>
          <canvas
            ref={canvasRef}
            className="w-full rounded-lg bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800"
          />
          <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-orange-500"></div>
              <span>Leader</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-purple-500"></div>
              <span>Candidate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-blue-500"></div>
              <span>Follower</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <span>Faulty</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-gray-500"></div>
              <span>Down</span>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-12 gap-4">
        {/* Configuration Panel */}
        <div className="col-span-12 md:col-span-4 space-y-4">
          {/* Algorithm Selection */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
              <Zap className="w-4 h-4 text-orange-600" />
              Consensus Algorithm
            </h4>
            <div className="space-y-2">
              {ALGORITHMS.map(algo => (
                <button
                  key={algo.id}
                  onClick={() => setSelectedAlgorithm(algo)}
                  disabled={isRunning}
                  className={`w-full text-left p-3 rounded-lg transition-colors disabled:opacity-50 ${
                    selectedAlgorithm.id === algo.id
                      ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white'
                      : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-900 dark:text-white'
                  }`}
                >
                  <div className="font-medium text-sm">{algo.name}</div>
                  <div className={`text-xs mt-1 ${
                    selectedAlgorithm.id === algo.id ? 'text-orange-100' : 'text-gray-600 dark:text-gray-400'
                  }`}>
                    {algo.faultTolerance}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Scenario Selection */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
              <Shield className="w-4 h-4 text-blue-600" />
              Failure Scenario
            </h4>
            <div className="space-y-2">
              {SCENARIOS.map(scenario => (
                <button
                  key={scenario.id}
                  onClick={() => setSelectedScenario(scenario)}
                  disabled={isRunning}
                  className={`w-full text-left p-2 rounded-lg transition-colors text-sm disabled:opacity-50 ${
                    selectedScenario.id === scenario.id
                      ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                      : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-900 dark:text-white'
                  }`}
                >
                  <div className="font-medium">{scenario.name}</div>
                  <div className="text-xs opacity-75">{scenario.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Control Buttons */}
          <div className="space-y-2">
            <button
              onClick={runSimulation}
              disabled={isRunning}
              className="w-full px-4 py-3 bg-gradient-to-r from-orange-600 to-red-600 text-white rounded-lg hover:from-orange-700 hover:to-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 font-semibold"
            >
              {isRunning ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  실행 중...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  시뮬레이션 시작
                </>
              )}
            </button>

            <button
              onClick={resetSimulation}
              disabled={isRunning}
              className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors"
            >
              초기화
            </button>
          </div>

          {/* Consensus Result */}
          {consensusReached && (
            <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-green-800 dark:text-green-200">
                  ✅ 합의 성공
                </h4>
                <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <div className="text-sm text-green-700 dark:text-green-300">
                <p><strong>합의 값:</strong></p>
                <p className="font-mono text-xs bg-green-100 dark:bg-green-900/30 p-2 rounded mt-1">
                  {consensusValue}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Node Status Panel */}
        <div className="col-span-12 md:col-span-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 h-full">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
              <Users className="w-4 h-4" />
              Nodes ({nodes.length})
            </h4>

            <div className="space-y-2 max-h-96 overflow-y-auto">
              {nodes.map(node => (
                <div
                  key={node.id}
                  className={`p-3 rounded-lg border-2 ${
                    node.isDown
                      ? 'border-gray-400 bg-gray-100 dark:bg-gray-700 opacity-50'
                      : node.isFaulty
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : node.role === 'leader'
                      ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20'
                      : node.role === 'candidate'
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                      : 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm text-gray-900 dark:text-white">
                        {node.name}
                      </span>
                      {node.role === 'leader' && !node.isDown && (
                        <Crown className="w-4 h-4 text-yellow-600" />
                      )}
                      {node.isDown && (
                        <WifiOff className="w-4 h-4 text-gray-600" />
                      )}
                      {node.isFaulty && (
                        <AlertCircle className="w-4 h-4 text-red-600" />
                      )}
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      node.role === 'leader'
                        ? 'bg-orange-200 dark:bg-orange-900 text-orange-800 dark:text-orange-200'
                        : node.role === 'candidate'
                        ? 'bg-purple-200 dark:bg-purple-900 text-purple-800 dark:text-purple-200'
                        : 'bg-blue-200 dark:bg-blue-900 text-blue-800 dark:text-blue-200'
                    }`}>
                      {node.isDown ? 'DOWN' : node.role.toUpperCase()}
                    </span>
                  </div>

                  {selectedAlgorithm.id === 'raft' && !node.isDown && (
                    <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1 mt-2">
                      <div>Term: {node.term}</div>
                      {node.votedFor && <div>Voted for: {node.votedFor}</div>}
                      <div>Log entries: {node.log.length}</div>
                      <div>Commit index: {node.commitIndex}</div>
                    </div>
                  )}

                  {node.isFaulty && (
                    <div className="text-xs text-red-600 dark:text-red-400 mt-2">
                      ⚠️ 악의적 동작 가능성
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Execution Log Panel */}
        <div className="col-span-12 md:col-span-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 h-full">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
              <MessageSquare className="w-4 h-4" />
              Execution Log
            </h4>

            <div className="h-96 overflow-y-auto space-y-1 bg-gray-900 rounded-lg p-3">
              {executionLog.length === 0 ? (
                <p className="text-xs text-gray-500">
                  시뮬레이션을 시작하면 실행 로그가 여기에 표시됩니다
                </p>
              ) : (
                executionLog.map((log, idx) => (
                  <p key={idx} className="text-xs text-green-400 font-mono leading-relaxed">
                    {log}
                  </p>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Algorithm Details */}
      <div className="mt-6 bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-orange-800 dark:text-orange-200 space-y-1">
            <p className="font-semibold mb-2">📚 {selectedAlgorithm.name} 알고리즘</p>
            <p><strong>설명:</strong> {selectedAlgorithm.description}</p>
            <p><strong>장애 허용성:</strong> {selectedAlgorithm.faultTolerance}</p>
            <p><strong>복잡도:</strong> {selectedAlgorithm.complexity}</p>

            {selectedAlgorithm.id === 'raft' && (
              <>
                <p className="mt-2"><strong>동작 과정:</strong></p>
                <p>1️⃣ 리더 선출: Candidate가 과반수 투표 획득</p>
                <p>2️⃣ 로그 복제: 리더가 명령을 Follower에게 복제</p>
                <p>3️⃣ 커밋: 과반수 복제 확인 후 커밋</p>
              </>
            )}

            {selectedAlgorithm.id === 'paxos' && (
              <>
                <p className="mt-2"><strong>동작 과정:</strong></p>
                <p>1️⃣ Phase 1 Prepare: Proposer가 제안 번호 획득</p>
                <p>2️⃣ Phase 1 Promise: Acceptor가 약속 응답</p>
                <p>3️⃣ Phase 2 Accept: 값을 제안하고 수락받음</p>
              </>
            )}

            {selectedAlgorithm.id === 'pbft' && (
              <>
                <p className="mt-2"><strong>동작 과정:</strong></p>
                <p>1️⃣ Pre-Prepare: Primary가 요청 순서 지정</p>
                <p>2️⃣ Prepare: 2/3 이상 노드 간 합의</p>
                <p>3️⃣ Commit: 최종 커밋 및 실행</p>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
