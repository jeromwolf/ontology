'use client';

import { useState } from 'react';
import {
  Layers, Database, Server, GitBranch,
  BarChart3, Bell, Shield, Zap,
  CheckCircle, Download, Code, DollarSign
} from 'lucide-react';

interface Component {
  id: string;
  category: 'data' | 'training' | 'serving' | 'monitoring';
  name: string;
  selected: boolean;
  cost: number;
  complexity: number;
}

export default function MLOpsArchitectDashboard() {
  const [components, setComponents] = useState<Component[]>([
    // Data
    { id: 'feature-store', category: 'data', name: 'Feature Store (Feast)', selected: true, cost: 500, complexity: 3 },
    { id: 'data-pipeline', category: 'data', name: 'Data Pipeline (Airflow)', selected: true, cost: 300, complexity: 4 },
    { id: 'data-validation', category: 'data', name: 'Data Validation (Great Expectations)', selected: false, cost: 100, complexity: 2 },

    // Training
    { id: 'experiment-tracking', category: 'training', name: 'Experiment Tracking (MLflow)', selected: true, cost: 200, complexity: 2 },
    { id: 'distributed-training', category: 'training', name: 'Distributed Training (Kubeflow)', selected: true, cost: 1000, complexity: 5 },
    { id: 'hyperparameter-tuning', category: 'training', name: 'HP Tuning (Optuna)', selected: false, cost: 150, complexity: 3 },

    // Serving
    { id: 'model-serving', category: 'serving', name: 'Model Serving (Seldon/KServe)', selected: true, cost: 600, complexity: 4 },
    { id: 'api-gateway', category: 'serving', name: 'API Gateway', selected: true, cost: 200, complexity: 2 },
    { id: 'ab-testing', category: 'serving', name: 'A/B Testing Framework', selected: false, cost: 250, complexity: 3 },

    // Monitoring
    { id: 'monitoring', category: 'monitoring', name: 'Monitoring (Prometheus)', selected: true, cost: 300, complexity: 3 },
    { id: 'drift-detection', category: 'monitoring', name: 'Drift Detection (Evidently)', selected: true, cost: 200, complexity: 3 },
    { id: 'alerting', category: 'monitoring', name: 'Alerting (PagerDuty)', selected: false, cost: 150, complexity: 2 }
  ]);

  const [scale, setScale] = useState<'small' | 'medium' | 'large'>('medium');
  const [cloudProvider, setCloudProvider] = useState<'aws' | 'gcp' | 'azure'>('aws');

  const toggleComponent = (id: string) => {
    setComponents(prev => prev.map(c =>
      c.id === id ? { ...c, selected: !c.selected } : c
    ));
  };

  const selectedComponents = components.filter(c => c.selected);
  const totalCost = selectedComponents.reduce((sum, c) => sum + c.cost, 0);
  const avgComplexity = selectedComponents.reduce((sum, c) => sum + c.complexity, 0) / selectedComponents.length;

  const scaleMultiplier = scale === 'small' ? 0.5 : scale === 'medium' ? 1 : 2;
  const estimatedMonthlyCost = totalCost * scaleMultiplier;

  const bestPractices = [
    { id: 'versioning', name: '모델 버저닝', checked: selectedComponents.some(c => c.id === 'experiment-tracking'), required: true },
    { id: 'cicd', name: 'CI/CD 파이프라인', checked: true, required: true },
    { id: 'monitoring', name: '모니터링 & 알림', checked: selectedComponents.some(c => c.id === 'monitoring'), required: true },
    { id: 'drift', name: '데이터 드리프트 탐지', checked: selectedComponents.some(c => c.id === 'drift-detection'), required: false },
    { id: 'feature-store', name: 'Feature Store', checked: selectedComponents.some(c => c.id === 'feature-store'), required: false },
    { id: 'ab-testing', name: 'A/B 테스팅', checked: selectedComponents.some(c => c.id === 'ab-testing'), required: false }
  ];

  const completionRate = (bestPractices.filter(bp => bp.checked).length / bestPractices.length) * 100;

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'data': return Database;
      case 'training': return Zap;
      case 'serving': return Server;
      case 'monitoring': return BarChart3;
      default: return Layers;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'data': return 'from-blue-600 to-cyan-600';
      case 'training': return 'from-purple-600 to-pink-600';
      case 'serving': return 'from-green-600 to-emerald-600';
      case 'monitoring': return 'from-orange-600 to-red-600';
      default: return 'from-slate-600 to-slate-700';
    }
  };

  const generateArchitectureDiagram = () => {
    return `# MLOps Architecture

${selectedComponents.filter(c => c.category === 'data').length > 0 ? `
## Data Layer
${selectedComponents.filter(c => c.category === 'data').map(c => `- ${c.name}`).join('\n')}
` : ''}
${selectedComponents.filter(c => c.category === 'training').length > 0 ? `
## Training Layer
${selectedComponents.filter(c => c.category === 'training').map(c => `- ${c.name}`).join('\n')}
` : ''}
${selectedComponents.filter(c => c.category === 'serving').length > 0 ? `
## Serving Layer
${selectedComponents.filter(c => c.category === 'serving').map(c => `- ${c.name}`).join('\n')}
` : ''}
${selectedComponents.filter(c => c.category === 'monitoring').length > 0 ? `
## Monitoring Layer
${selectedComponents.filter(c => c.category === 'monitoring').map(c => `- ${c.name}`).join('\n')}
` : ''}

## Cost Estimation
- Scale: ${scale}
- Monthly Cost: $${estimatedMonthlyCost.toLocaleString()}
- Cloud Provider: ${cloudProvider.toUpperCase()}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-purple-500 p-3 rounded-lg">
              <Layers className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">MLOps 아키텍처 설계 대시보드</h1>
              <p className="text-slate-300">Complete MLOps Architecture Designer</p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-sm text-slate-300">선택된 컴포넌트</span>
              </div>
              <div className="text-2xl font-bold">{selectedComponents.length}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <DollarSign className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-slate-300">월 예상 비용</span>
              </div>
              <div className="text-2xl font-bold text-yellow-400">
                ${estimatedMonthlyCost.toLocaleString()}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <BarChart3 className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-slate-300">복잡도</span>
              </div>
              <div className="text-2xl font-bold">{avgComplexity.toFixed(1)}/5</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Shield className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-300">Best Practices</span>
              </div>
              <div className="text-2xl font-bold text-purple-400">{completionRate.toFixed(0)}%</div>
            </div>
          </div>
        </div>

        {/* Configuration */}
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4">규모</h2>
            <div className="space-y-2">
              {(['small', 'medium', 'large'] as const).map(s => (
                <button
                  key={s}
                  onClick={() => setScale(s)}
                  className={`w-full p-3 rounded-lg font-semibold transition-all ${
                    scale === s
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {s === 'small' && 'Small (0.5x)'}
                  {s === 'medium' && 'Medium (1x)'}
                  {s === 'large' && 'Large (2x)'}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4">클라우드 제공자</h2>
            <div className="space-y-2">
              {(['aws', 'gcp', 'azure'] as const).map(cp => (
                <button
                  key={cp}
                  onClick={() => setCloudProvider(cp)}
                  className={`w-full p-3 rounded-lg font-semibold transition-all ${
                    cloudProvider === cp
                      ? 'bg-purple-600 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {cp.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4">비용 상세</h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">기본 비용:</span>
                <span className="font-bold">${totalCost}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">규모 배수:</span>
                <span className="font-bold">{scaleMultiplier}x</span>
              </div>
              <div className="h-px bg-slate-700 my-2"></div>
              <div className="flex justify-between text-lg">
                <span className="text-slate-300">월 총액:</span>
                <span className="font-bold text-yellow-400">
                  ${estimatedMonthlyCost.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">연 예상:</span>
                <span className="font-bold">
                  ${(estimatedMonthlyCost * 12).toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Component Selection */}
        <div className="grid md:grid-cols-2 gap-6">
          {['data', 'training', 'serving', 'monitoring'].map(category => {
            const Icon = getCategoryIcon(category);
            const categoryComponents = components.filter(c => c.category === category);

            return (
              <div key={category} className="bg-slate-800 rounded-xl p-6 border border-slate-700">
                <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <Icon className="w-6 h-6" />
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </h2>
                <div className="space-y-2">
                  {categoryComponents.map(comp => (
                    <button
                      key={comp.id}
                      onClick={() => toggleComponent(comp.id)}
                      className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                        comp.selected
                          ? 'border-cyan-500 bg-cyan-500/10'
                          : 'border-slate-700 bg-slate-700/50 hover:border-slate-600'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold">{comp.name}</span>
                        <div className="flex items-center gap-2">
                          {comp.selected && <CheckCircle className="w-5 h-5 text-cyan-400" />}
                        </div>
                      </div>
                      <div className="flex justify-between text-sm text-slate-400">
                        <span>비용: ${comp.cost}/월</span>
                        <span>복잡도: {comp.complexity}/5</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Best Practices Checklist */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Shield className="w-6 h-6 text-purple-400" />
            Best Practices 체크리스트
          </h2>
          <div className="grid md:grid-cols-2 gap-3">
            {bestPractices.map(bp => (
              <div
                key={bp.id}
                className={`p-4 rounded-lg border ${
                  bp.checked
                    ? 'border-green-500 bg-green-500/10'
                    : bp.required
                    ? 'border-red-500 bg-red-500/10'
                    : 'border-yellow-500 bg-yellow-500/10'
                }`}
              >
                <div className="flex items-center gap-3">
                  {bp.checked ? (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  ) : (
                    <div className="w-5 h-5 border-2 border-slate-500 rounded"></div>
                  )}
                  <div className="flex-1">
                    <div className="font-semibold">{bp.name}</div>
                    {bp.required && !bp.checked && (
                      <div className="text-xs text-red-400">필수 항목</div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 bg-slate-700/50 rounded-lg p-4">
            <div className="flex justify-between mb-2">
              <span className="font-semibold">완성도</span>
              <span className="font-bold text-purple-400">{completionRate.toFixed(0)}%</span>
            </div>
            <div className="bg-slate-700 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all"
                style={{ width: `${completionRate}%` }}
              />
            </div>
          </div>
        </div>

        {/* Scalability Calculator */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">확장성 계산기</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-2">예상 QPS</div>
              <div className="text-3xl font-bold">{scale === 'small' ? '100' : scale === 'medium' ? '1,000' : '10,000'}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-2">모델 수</div>
              <div className="text-3xl font-bold">{scale === 'small' ? '1-5' : scale === 'medium' ? '5-20' : '20+'}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-2">예상 팀 크기</div>
              <div className="text-3xl font-bold">{scale === 'small' ? '2-3' : scale === 'medium' ? '5-10' : '10+'}</div>
            </div>
          </div>
        </div>

        {/* Architecture Export */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Code className="w-6 h-6 text-green-400" />
              아키텍처 다이어그램
            </h2>
            <button
              onClick={() => {
                const blob = new Blob([generateArchitectureDiagram()], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'mlops-architecture.md';
                a.click();
              }}
              className="flex items-center gap-2 bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg font-semibold"
            >
              <Download className="w-5 h-5" />
              Export
            </button>
          </div>
          <pre className="bg-slate-900 rounded-lg p-4 overflow-x-auto text-sm max-h-96">
            <code className="text-green-400">{generateArchitectureDiagram()}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}
