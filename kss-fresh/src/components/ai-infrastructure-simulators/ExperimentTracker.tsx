'use client';

import { useState, useEffect } from 'react';
import {
  FlaskConical, BarChart3, TrendingUp, Filter,
  Search, Star, Clock, GitCompare, Download, Eye
} from 'lucide-react';

interface Experiment {
  id: string;
  name: string;
  timestamp: Date;
  params: Record<string, any>;
  metrics: {
    train_loss: number;
    val_loss: number;
    train_acc: number;
    val_acc: number;
    f1_score: number;
  };
  status: 'running' | 'completed' | 'failed';
  duration: number;
  starred: boolean;
}

export default function ExperimentTracker() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<'timestamp' | 'val_acc' | 'val_loss'>('timestamp');
  const [filterStatus, setFilterStatus] = useState<'all' | 'running' | 'completed' | 'failed'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'table' | 'comparison'>('table');

  useEffect(() => {
    const sampleExperiments: Experiment[] = Array.from({ length: 12 }, (_, i) => ({
      id: `exp-${i + 1}`,
      name: `experiment-${i + 1}`,
      timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000),
      params: {
        learning_rate: (Math.random() * 0.01).toFixed(4),
        batch_size: [16, 32, 64, 128][Math.floor(Math.random() * 4)],
        epochs: [50, 100, 150, 200][Math.floor(Math.random() * 4)],
        optimizer: ['adam', 'sgd', 'rmsprop'][Math.floor(Math.random() * 3)],
        model: ['resnet50', 'efficientnet', 'vit'][Math.floor(Math.random() * 3)]
      },
      metrics: {
        train_loss: 0.1 + Math.random() * 0.4,
        val_loss: 0.15 + Math.random() * 0.5,
        train_acc: 0.85 + Math.random() * 0.13,
        val_acc: 0.80 + Math.random() * 0.15,
        f1_score: 0.75 + Math.random() * 0.20
      },
      status: ['completed', 'running', 'failed'][Math.floor(Math.random() * 3)] as any,
      duration: 300 + Math.random() * 1200,
      starred: Math.random() > 0.7
    }));
    setExperiments(sampleExperiments);
  }, []);

  const filteredExperiments = experiments
    .filter(exp => {
      const matchesStatus = filterStatus === 'all' || exp.status === filterStatus;
      const matchesSearch = exp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        exp.params.model.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesStatus && matchesSearch;
    })
    .sort((a, b) => {
      if (sortBy === 'timestamp') return b.timestamp.getTime() - a.timestamp.getTime();
      if (sortBy === 'val_acc') return b.metrics.val_acc - a.metrics.val_acc;
      if (sortBy === 'val_loss') return a.metrics.val_loss - b.metrics.val_loss;
      return 0;
    });

  const toggleExperimentSelection = (id: string) => {
    setSelectedExperiments(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const toggleStar = (id: string) => {
    setExperiments(prev => prev.map(exp =>
      exp.id === id ? { ...exp, starred: !exp.starred } : exp
    ));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-blue-400 bg-blue-500/20';
      case 'completed': return 'text-green-400 bg-green-500/20';
      case 'failed': return 'text-red-400 bg-red-500/20';
      default: return 'text-slate-400 bg-slate-500/20';
    }
  };

  const selectedExpsData = experiments.filter(exp => selectedExperiments.includes(exp.id));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="bg-purple-500 p-3 rounded-lg">
                <FlaskConical className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">실험 추적기</h1>
                <p className="text-slate-300">MLflow-style Experiment Tracker</p>
              </div>
            </div>
            <div className="flex gap-3">
              <button className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg font-semibold">
                <Download className="w-5 h-5" />
                Export CSV
              </button>
              <button
                onClick={() => setViewMode(viewMode === 'table' ? 'comparison' : 'table')}
                className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg font-semibold"
              >
                {viewMode === 'table' ? <GitCompare className="w-5 h-5" /> : <BarChart3 className="w-5 h-5" />}
                {viewMode === 'table' ? 'Compare' : 'Table'}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">총 실험</div>
              <div className="text-2xl font-bold">{experiments.length}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">진행중</div>
              <div className="text-2xl font-bold text-blue-400">
                {experiments.filter(e => e.status === 'running').length}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">완료</div>
              <div className="text-2xl font-bold text-green-400">
                {experiments.filter(e => e.status === 'completed').length}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-300 mb-1">최고 정확도</div>
              <div className="text-2xl font-bold text-yellow-400">
                {Math.max(...experiments.map(e => e.metrics.val_acc)).toFixed(3)}
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <Search className="w-4 h-4 text-blue-400" />
                검색
              </label>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="실험명 또는 모델..."
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <Filter className="w-4 h-4 text-purple-400" />
                상태 필터
              </label>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value as any)}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
              >
                <option value="all">전체</option>
                <option value="running">진행중</option>
                <option value="completed">완료</option>
                <option value="failed">실패</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-400" />
                정렬
              </label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
              >
                <option value="timestamp">최신순</option>
                <option value="val_acc">정확도 높은순</option>
                <option value="val_loss">손실 낮은순</option>
              </select>
            </div>
          </div>
        </div>

        {viewMode === 'table' ? (
          /* Experiments Table */
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 overflow-x-auto">
            <h2 className="text-xl font-bold mb-4">실험 목록</h2>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-3 px-2">
                    <input type="checkbox" className="rounded" />
                  </th>
                  <th className="text-left py-3 px-2"></th>
                  <th className="text-left py-3 px-2">이름</th>
                  <th className="text-left py-3 px-2">모델</th>
                  <th className="text-left py-3 px-2">LR</th>
                  <th className="text-left py-3 px-2">배치</th>
                  <th className="text-left py-3 px-2">Train Loss</th>
                  <th className="text-left py-3 px-2">Val Loss</th>
                  <th className="text-left py-3 px-2">Val Acc</th>
                  <th className="text-left py-3 px-2">F1</th>
                  <th className="text-left py-3 px-2">상태</th>
                  <th className="text-left py-3 px-2">시간</th>
                </tr>
              </thead>
              <tbody>
                {filteredExperiments.map(exp => (
                  <tr
                    key={exp.id}
                    className={`border-b border-slate-700 hover:bg-slate-700/50 ${
                      selectedExperiments.includes(exp.id) ? 'bg-slate-700/30' : ''
                    }`}
                  >
                    <td className="py-3 px-2">
                      <input
                        type="checkbox"
                        checked={selectedExperiments.includes(exp.id)}
                        onChange={() => toggleExperimentSelection(exp.id)}
                        className="rounded"
                      />
                    </td>
                    <td className="py-3 px-2">
                      <button onClick={() => toggleStar(exp.id)}>
                        <Star className={`w-4 h-4 ${exp.starred ? 'fill-yellow-400 text-yellow-400' : 'text-slate-500'}`} />
                      </button>
                    </td>
                    <td className="py-3 px-2 font-semibold">{exp.name}</td>
                    <td className="py-3 px-2 text-slate-300">{exp.params.model}</td>
                    <td className="py-3 px-2 text-slate-400">{exp.params.learning_rate}</td>
                    <td className="py-3 px-2 text-slate-400">{exp.params.batch_size}</td>
                    <td className="py-3 px-2">{exp.metrics.train_loss.toFixed(4)}</td>
                    <td className="py-3 px-2">{exp.metrics.val_loss.toFixed(4)}</td>
                    <td className="py-3 px-2 font-semibold text-green-400">
                      {(exp.metrics.val_acc * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-2">{exp.metrics.f1_score.toFixed(3)}</td>
                    <td className="py-3 px-2">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${getStatusColor(exp.status)}`}>
                        {exp.status}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-slate-400 text-xs">
                      {Math.floor(exp.duration / 60)}m {Math.floor(exp.duration % 60)}s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          /* Comparison View */
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <GitCompare className="w-6 h-6 text-purple-400" />
              실험 비교 ({selectedExpsData.length}개 선택됨)
            </h2>
            {selectedExpsData.length > 0 ? (
              <div className="space-y-6">
                {/* Metrics Comparison */}
                <div>
                  <h3 className="font-semibold mb-3">메트릭 비교</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-700">
                          <th className="text-left py-2 px-2">실험</th>
                          <th className="text-left py-2 px-2">Train Loss</th>
                          <th className="text-left py-2 px-2">Val Loss</th>
                          <th className="text-left py-2 px-2">Val Acc</th>
                          <th className="text-left py-2 px-2">F1 Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedExpsData.map(exp => (
                          <tr key={exp.id} className="border-b border-slate-700">
                            <td className="py-2 px-2 font-semibold">{exp.name}</td>
                            <td className="py-2 px-2">{exp.metrics.train_loss.toFixed(4)}</td>
                            <td className="py-2 px-2">{exp.metrics.val_loss.toFixed(4)}</td>
                            <td className="py-2 px-2 text-green-400 font-semibold">
                              {(exp.metrics.val_acc * 100).toFixed(2)}%
                            </td>
                            <td className="py-2 px-2">{exp.metrics.f1_score.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Hyperparameters Comparison */}
                <div>
                  <h3 className="font-semibold mb-3">하이퍼파라미터 비교</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-700">
                          <th className="text-left py-2 px-2">실험</th>
                          <th className="text-left py-2 px-2">모델</th>
                          <th className="text-left py-2 px-2">Learning Rate</th>
                          <th className="text-left py-2 px-2">Batch Size</th>
                          <th className="text-left py-2 px-2">Optimizer</th>
                          <th className="text-left py-2 px-2">Epochs</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedExpsData.map(exp => (
                          <tr key={exp.id} className="border-b border-slate-700">
                            <td className="py-2 px-2 font-semibold">{exp.name}</td>
                            <td className="py-2 px-2 text-cyan-400">{exp.params.model}</td>
                            <td className="py-2 px-2">{exp.params.learning_rate}</td>
                            <td className="py-2 px-2">{exp.params.batch_size}</td>
                            <td className="py-2 px-2">{exp.params.optimizer}</td>
                            <td className="py-2 px-2">{exp.params.epochs}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Best Model Highlight */}
                {(() => {
                  const bestExp = selectedExpsData.reduce((best, exp) =>
                    exp.metrics.val_acc > best.metrics.val_acc ? exp : best
                  );
                  return (
                    <div className="bg-green-500/20 border border-green-500 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="w-5 h-5 text-green-400" />
                        <span className="font-bold text-green-400">최고 성능 모델</span>
                      </div>
                      <div className="text-sm space-y-1">
                        <div><span className="text-slate-400">실험:</span> {bestExp.name}</div>
                        <div><span className="text-slate-400">정확도:</span> <span className="font-bold">{(bestExp.metrics.val_acc * 100).toFixed(2)}%</span></div>
                        <div><span className="text-slate-400">모델:</span> {bestExp.params.model}</div>
                        <div><span className="text-slate-400">Learning Rate:</span> {bestExp.params.learning_rate}</div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            ) : (
              <div className="text-center text-slate-400 py-12">
                비교할 실험을 선택하세요 (테이블 뷰에서 체크박스 선택)
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
