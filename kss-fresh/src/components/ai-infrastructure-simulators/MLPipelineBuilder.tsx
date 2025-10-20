'use client';

import { useState } from 'react';
import {
  GitBranch, Database, Cpu, Server,
  CheckCircle, Code, Download, Play,
  Settings, FileText, Box, Layers
} from 'lucide-react';

interface PipelineComponent {
  id: string;
  type: 'data' | 'preprocessing' | 'training' | 'evaluation' | 'deployment';
  name: string;
  config: Record<string, any>;
  x: number;
  y: number;
}

interface Connection {
  from: string;
  to: string;
}

export default function MLPipelineBuilder() {
  const [components, setComponents] = useState<PipelineComponent[]>([
    {
      id: 'data-1',
      type: 'data',
      name: 'Data Ingestion',
      config: { source: 's3://data-bucket', format: 'parquet' },
      x: 50,
      y: 100
    },
    {
      id: 'preprocess-1',
      type: 'preprocessing',
      name: 'Data Preprocessing',
      config: { normalization: 'standard', encoding: 'onehot' },
      x: 250,
      y: 100
    },
    {
      id: 'train-1',
      type: 'training',
      name: 'Model Training',
      config: { algorithm: 'xgboost', epochs: 100, batch_size: 32 },
      x: 450,
      y: 100
    },
    {
      id: 'eval-1',
      type: 'evaluation',
      name: 'Model Evaluation',
      config: { metrics: ['accuracy', 'f1', 'auc'] },
      x: 650,
      y: 100
    },
    {
      id: 'deploy-1',
      type: 'deployment',
      name: 'Model Deployment',
      config: { platform: 'sagemaker', replicas: 3 },
      x: 850,
      y: 100
    }
  ]);

  const [connections] = useState<Connection[]>([
    { from: 'data-1', to: 'preprocess-1' },
    { from: 'preprocess-1', to: 'train-1' },
    { from: 'train-1', to: 'eval-1' },
    { from: 'eval-1', to: 'deploy-1' }
  ]);

  const [selectedComponent, setSelectedComponent] = useState<PipelineComponent | null>(null);
  const [showYAML, setShowYAML] = useState(false);

  const componentLibrary = [
    { type: 'data', name: 'Data Source', icon: Database, color: 'bg-blue-600' },
    { type: 'preprocessing', name: 'Preprocessing', icon: Settings, color: 'bg-purple-600' },
    { type: 'training', name: 'Training', icon: Cpu, color: 'bg-green-600' },
    { type: 'evaluation', name: 'Evaluation', icon: CheckCircle, color: 'bg-yellow-600' },
    { type: 'deployment', name: 'Deployment', icon: Server, color: 'bg-red-600' }
  ];

  const generateYAML = () => {
    const yaml = `# Kubeflow Pipeline Definition
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: ml-pipeline-
spec:
  entrypoint: ml-pipeline
  templates:
${components.map(comp => `  - name: ${comp.id}
    container:
      image: ml-ops/${comp.type}:latest
      command: ["python"]
      args: ["/${comp.type}.py"]
      env:
${Object.entries(comp.config).map(([k, v]) => `      - name: ${k.toUpperCase()}
        value: "${v}"`).join('\n')}
`).join('\n')}
  - name: ml-pipeline
    dag:
      tasks:
${components.map(comp => `      - name: ${comp.id}
        template: ${comp.id}
        dependencies: [${connections.filter(c => c.to === comp.id).map(c => c.from).join(', ')}]`).join('\n')}`;
    return yaml;
  };

  const getComponentIcon = (type: string) => {
    const item = componentLibrary.find(c => c.type === type);
    if (!item) return Database;
    return item.icon;
  };

  const getComponentColor = (type: string) => {
    const item = componentLibrary.find(c => c.type === type);
    return item?.color || 'bg-slate-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-green-500 p-3 rounded-lg">
                <GitBranch className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">ML 파이프라인 빌더</h1>
                <p className="text-slate-300">Drag-and-Drop ML Pipeline Builder</p>
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setShowYAML(!showYAML)}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg font-semibold"
              >
                <FileText className="w-5 h-5" />
                {showYAML ? 'Hide YAML' : 'Show YAML'}
              </button>
              <button className="flex items-center gap-2 bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg font-semibold">
                <Download className="w-5 h-5" />
                Export
              </button>
              <button className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg font-semibold">
                <Play className="w-5 h-5" />
                Run Pipeline
              </button>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Component Library */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Box className="w-6 h-6 text-cyan-400" />
              컴포넌트 라이브러리
            </h2>
            <div className="space-y-3">
              {componentLibrary.map(item => {
                const Icon = item.icon;
                return (
                  <div
                    key={item.type}
                    className={`${item.color} rounded-lg p-4 cursor-move hover:opacity-80 transition-opacity`}
                    draggable
                  >
                    <div className="flex items-center gap-3">
                      <Icon className="w-6 h-6" />
                      <span className="font-semibold">{item.name}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-6 p-4 bg-slate-700/50 rounded-lg">
              <h3 className="font-semibold mb-2 text-sm">Pipeline Stats</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Components:</span>
                  <span className="font-bold">{components.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Connections:</span>
                  <span className="font-bold">{connections.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Estimated Time:</span>
                  <span className="font-bold">~15 min</span>
                </div>
              </div>
            </div>
          </div>

          {/* Pipeline Canvas */}
          <div className="lg:col-span-3 space-y-6">
            <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Layers className="w-6 h-6 text-purple-400" />
                파이프라인 DAG
              </h2>
              <div className="bg-slate-900 rounded-lg p-6 min-h-96 relative overflow-x-auto">
                <svg className="w-full h-96">
                  {/* Draw connections */}
                  {connections.map((conn, idx) => {
                    const fromComp = components.find(c => c.id === conn.from);
                    const toComp = components.find(c => c.id === conn.to);
                    if (!fromComp || !toComp) return null;

                    return (
                      <g key={idx}>
                        <line
                          x1={fromComp.x + 90}
                          y1={fromComp.y + 30}
                          x2={toComp.x}
                          y2={toComp.y + 30}
                          stroke="#475569"
                          strokeWidth="2"
                        />
                        <polygon
                          points={`${toComp.x},${toComp.y + 30} ${toComp.x - 8},${toComp.y + 25} ${toComp.x - 8},${toComp.y + 35}`}
                          fill="#475569"
                        />
                      </g>
                    );
                  })}
                </svg>

                {/* Draw components */}
                {components.map(comp => {
                  const Icon = getComponentIcon(comp.type);
                  const color = getComponentColor(comp.type);

                  return (
                    <div
                      key={comp.id}
                      onClick={() => setSelectedComponent(comp)}
                      className={`absolute cursor-pointer transition-all ${
                        selectedComponent?.id === comp.id ? 'ring-2 ring-cyan-500' : ''
                      }`}
                      style={{ left: comp.x, top: comp.y }}
                    >
                      <div className={`${color} rounded-lg p-3 w-32 shadow-lg`}>
                        <Icon className="w-5 h-5 mb-1" />
                        <div className="text-sm font-semibold truncate">{comp.name}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Component Configuration */}
            {selectedComponent && (
              <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
                <h2 className="text-xl font-bold mb-4">컴포넌트 설정</h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-semibold mb-2">이름</label>
                    <input
                      type="text"
                      value={selectedComponent.name}
                      onChange={(e) => {
                        setComponents(prev => prev.map(c =>
                          c.id === selectedComponent.id ? { ...c, name: e.target.value } : c
                        ));
                        setSelectedComponent({ ...selectedComponent, name: e.target.value });
                      }}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-semibold mb-2">타입</label>
                    <div className="text-lg font-bold text-slate-300">{selectedComponent.type}</div>
                  </div>

                  <div>
                    <label className="block text-sm font-semibold mb-2">설정</label>
                    <div className="bg-slate-900 rounded-lg p-4 space-y-2">
                      {Object.entries(selectedComponent.config).map(([key, value]) => (
                        <div key={key} className="flex justify-between items-center">
                          <span className="text-sm text-slate-400">{key}:</span>
                          <input
                            type="text"
                            value={String(value)}
                            onChange={(e) => {
                              const newConfig = { ...selectedComponent.config, [key]: e.target.value };
                              setComponents(prev => prev.map(c =>
                                c.id === selectedComponent.id ? { ...c, config: newConfig } : c
                              ));
                              setSelectedComponent({ ...selectedComponent, config: newConfig });
                            }}
                            className="bg-slate-700 border border-slate-600 rounded px-3 py-1 text-sm w-64"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* YAML Output */}
            {showYAML && (
              <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold flex items-center gap-2">
                    <Code className="w-6 h-6 text-green-400" />
                    Kubeflow Pipeline YAML
                  </h2>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(generateYAML());
                    }}
                    className="text-sm bg-slate-700 hover:bg-slate-600 px-3 py-1 rounded"
                  >
                    Copy
                  </button>
                </div>
                <pre className="bg-slate-900 rounded-lg p-4 overflow-x-auto text-sm max-h-96">
                  <code className="text-green-400">{generateYAML()}</code>
                </pre>
              </div>
            )}
          </div>
        </div>

        {/* Quick Start Templates */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">템플릿</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-slate-700 rounded-lg p-4 cursor-pointer hover:bg-slate-600 transition-colors">
              <h3 className="font-bold mb-2">Classification Pipeline</h3>
              <p className="text-sm text-slate-400">
                데이터 로드 → 전처리 → 분류 모델 학습 → 평가 → 배포
              </p>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 cursor-pointer hover:bg-slate-600 transition-colors">
              <h3 className="font-bold mb-2">Time Series Forecasting</h3>
              <p className="text-sm text-slate-400">
                시계열 데이터 → Feature Engineering → LSTM/Prophet → 평가
              </p>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 cursor-pointer hover:bg-slate-600 transition-colors">
              <h3 className="font-bold mb-2">AutoML Pipeline</h3>
              <p className="text-sm text-slate-400">
                데이터 → AutoML Search → 하이퍼파라미터 튜닝 → 배포
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
