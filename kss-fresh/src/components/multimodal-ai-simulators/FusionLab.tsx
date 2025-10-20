'use client';

import { useState } from 'react';
import {
  Merge, TrendingUp, Zap, Clock, HardDrive,
  CheckCircle, AlertCircle, Info, Sparkles
} from 'lucide-react';

interface FusionStrategy {
  id: string;
  name: string;
  description: string;
  accuracy: number;
  latency: number;
  memory: number;
  complexity: 'low' | 'medium' | 'high';
  useCases: string[];
}

interface ComparisonMetric {
  name: string;
  unit: string;
  early: number;
  late: number;
  hybrid: number;
  better: 'higher' | 'lower';
}

export default function FusionLab() {
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(['late']);
  const [taskType, setTaskType] = useState<'classification' | 'generation' | 'retrieval'>('classification');
  const [datasetSize, setDatasetSize] = useState<'small' | 'medium' | 'large'>('medium');
  const [showRecommendation, setShowRecommendation] = useState(false);

  const fusionStrategies: FusionStrategy[] = [
    {
      id: 'early',
      name: 'Early Fusion',
      description: '모달리티를 인코딩 전에 원시 데이터 수준에서 결합',
      accuracy: 0.82,
      latency: 45,
      memory: 2.1,
      complexity: 'low',
      useCases: ['단순 분류', '실시간 처리', '제한된 리소스']
    },
    {
      id: 'late',
      name: 'Late Fusion',
      description: '각 모달리티를 독립적으로 인코딩 후 최종 단계에서 결합',
      accuracy: 0.89,
      latency: 62,
      memory: 3.8,
      complexity: 'medium',
      useCases: ['고정밀 분류', '모달별 사전학습 활용', '불완전한 입력 대응']
    },
    {
      id: 'hybrid',
      name: 'Hybrid Fusion',
      description: 'Early와 Late를 조합하여 중간 단계에서도 상호작용',
      accuracy: 0.93,
      latency: 78,
      memory: 4.5,
      complexity: 'high',
      useCases: ['최고 성능 요구', '복잡한 추론', '풍부한 리소스']
    },
    {
      id: 'attention',
      name: 'Cross-Attention Fusion',
      description: '어텐션 메커니즘으로 모달 간 동적 상호작용',
      accuracy: 0.91,
      latency: 85,
      memory: 4.2,
      complexity: 'high',
      useCases: ['VQA', '이미지-텍스트 매칭', '멀티모달 생성']
    },
    {
      id: 'hierarchical',
      name: 'Hierarchical Fusion',
      description: '여러 단계에서 점진적으로 정보를 융합',
      accuracy: 0.90,
      latency: 70,
      memory: 3.9,
      complexity: 'high',
      useCases: ['계층적 태스크', '다단계 추론', '세밀한 제어']
    }
  ];

  const comparisonMetrics: ComparisonMetric[] = [
    { name: '정확도', unit: '%', early: 82, late: 89, hybrid: 93, better: 'higher' },
    { name: '지연시간', unit: 'ms', early: 45, late: 62, hybrid: 78, better: 'lower' },
    { name: '메모리', unit: 'GB', early: 2.1, late: 3.8, hybrid: 4.5, better: 'lower' },
    { name: '처리량', unit: 'samples/s', early: 220, late: 160, hybrid: 128, better: 'higher' },
    { name: '학습 시간', unit: 'hours', early: 12, late: 18, hybrid: 24, better: 'lower' },
    { name: '파라미터 수', unit: 'M', early: 85, late: 150, hybrid: 210, better: 'lower' }
  ];

  const toggleStrategy = (id: string) => {
    if (selectedStrategies.includes(id)) {
      setSelectedStrategies(selectedStrategies.filter(s => s !== id));
    } else {
      if (selectedStrategies.length < 3) {
        setSelectedStrategies([...selectedStrategies, id]);
      } else {
        alert('최대 3개까지 비교 가능합니다.');
      }
    }
  };

  const getRecommendation = (): FusionStrategy => {
    if (taskType === 'generation') {
      return fusionStrategies.find(s => s.id === 'attention')!;
    } else if (taskType === 'retrieval') {
      return fusionStrategies.find(s => s.id === 'late')!;
    } else {
      // classification
      if (datasetSize === 'small') {
        return fusionStrategies.find(s => s.id === 'early')!;
      } else if (datasetSize === 'large') {
        return fusionStrategies.find(s => s.id === 'hybrid')!;
      } else {
        return fusionStrategies.find(s => s.id === 'late')!;
      }
    }
  };

  const getMetricColor = (strategy: string, metricName: string, better: string) => {
    const metric = comparisonMetrics.find(m => m.name === metricName);
    if (!metric) return 'text-gray-600';

    const value = metric[strategy as keyof typeof metric] as number;
    const values = [metric.early, metric.late, metric.hybrid];
    const best = better === 'higher' ? Math.max(...values) : Math.min(...values);

    if (value === best) return 'text-green-600 font-bold';
    if (better === 'higher') {
      return value < best * 0.9 ? 'text-red-600' : 'text-orange-600';
    } else {
      return value > best * 1.1 ? 'text-red-600' : 'text-orange-600';
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300';
      case 'medium': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'high': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Merge size={32} />
          <h2 className="text-2xl font-bold">모달 퓨전 실험실</h2>
        </div>
        <p className="text-violet-100">
          다양한 융합 전략을 비교하고 최적의 방법을 찾아보세요
        </p>
      </div>

      {/* Task Configuration */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Info className="text-purple-600" />
          태스크 설정
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold mb-2">태스크 유형</label>
            <select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value as any)}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
            >
              <option value="classification">분류 (Classification)</option>
              <option value="generation">생성 (Generation)</option>
              <option value="retrieval">검색 (Retrieval)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-2">데이터셋 크기</label>
            <select
              value={datasetSize}
              onChange={(e) => setDatasetSize(e.target.value as any)}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
            >
              <option value="small">소형 (&lt;10K)</option>
              <option value="medium">중형 (10K-100K)</option>
              <option value="large">대형 (&gt;100K)</option>
            </select>
          </div>
        </div>
        <button
          onClick={() => setShowRecommendation(!showRecommendation)}
          className="mt-4 flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
        >
          <Sparkles size={18} />
          추천 전략 보기
        </button>
      </div>

      {/* Recommendation */}
      {showRecommendation && (
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/30 dark:to-pink-900/30 rounded-lg p-6 border-2 border-purple-300 dark:border-purple-700">
          <div className="flex items-start gap-3">
            <Sparkles className="text-purple-600 flex-shrink-0" size={24} />
            <div>
              <h4 className="font-bold text-lg mb-2">추천 전략: {getRecommendation().name}</h4>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                {getRecommendation().description}
              </p>
              <div className="flex flex-wrap gap-2">
                <span className={`px-3 py-1 rounded-full text-sm ${getComplexityColor(getRecommendation().complexity)}`}>
                  복잡도: {getRecommendation().complexity.toUpperCase()}
                </span>
                <span className="px-3 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300 rounded-full text-sm">
                  정확도: {(getRecommendation().accuracy * 100).toFixed(0)}%
                </span>
                <span className="px-3 py-1 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 rounded-full text-sm">
                  지연: {getRecommendation().latency}ms
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Strategy Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">융합 전략 선택 (최대 3개)</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {fusionStrategies.map(strategy => (
            <button
              key={strategy.id}
              onClick={() => toggleStrategy(strategy.id)}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                selectedStrategies.includes(strategy.id)
                  ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                  : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <h4 className="font-bold">{strategy.name}</h4>
                {selectedStrategies.includes(strategy.id) && (
                  <CheckCircle className="text-purple-600 flex-shrink-0" size={20} />
                )}
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {strategy.description}
              </p>
              <div className="flex items-center gap-2 mb-2">
                <span className={`px-2 py-0.5 rounded-full text-xs ${getComplexityColor(strategy.complexity)}`}>
                  {strategy.complexity}
                </span>
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">정확도:</span>
                  <span className="font-semibold">{(strategy.accuracy * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">지연:</span>
                  <span className="font-semibold">{strategy.latency}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">메모리:</span>
                  <span className="font-semibold">{strategy.memory}GB</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Comparison Table */}
      {selectedStrategies.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="text-purple-600" />
            성능 비교표
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="p-3 text-left">지표</th>
                  <th className="p-3 text-center">Early Fusion</th>
                  <th className="p-3 text-center">Late Fusion</th>
                  <th className="p-3 text-center">Hybrid Fusion</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {comparisonMetrics.map(metric => (
                  <tr key={metric.name} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="p-3 font-semibold">
                      {metric.name}
                      {metric.better === 'higher' ? ' ⬆️' : ' ⬇️'}
                    </td>
                    <td className={`p-3 text-center ${getMetricColor('early', metric.name, metric.better)}`}>
                      {metric.early} {metric.unit}
                    </td>
                    <td className={`p-3 text-center ${getMetricColor('late', metric.name, metric.better)}`}>
                      {metric.late} {metric.unit}
                    </td>
                    <td className={`p-3 text-center ${getMetricColor('hybrid', metric.name, metric.better)}`}>
                      {metric.hybrid} {metric.unit}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-sm text-gray-500">
            💡 <strong>초록</strong>: 최고 성능, <strong>주황</strong>: 평균, <strong>빨강</strong>: 낮은 성능
          </div>
        </div>
      )}

      {/* Use Cases */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">사용 사례 및 권장사항</h3>
        <div className="space-y-4">
          {fusionStrategies.slice(0, 3).map(strategy => (
            <div key={strategy.id} className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <h4 className="font-semibold mb-2 text-purple-600">{strategy.name}</h4>
              <div className="flex flex-wrap gap-2">
                {strategy.useCases.map(useCase => (
                  <span
                    key={useCase}
                    className="px-3 py-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full text-sm"
                  >
                    {useCase}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Visual Comparison */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">시각적 비교</h3>
        <div className="grid md:grid-cols-3 gap-4">
          {/* Early Fusion */}
          <div className="p-4 bg-green-50 dark:bg-green-900/30 rounded-lg">
            <h4 className="font-semibold mb-3 text-green-600">Early Fusion</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-16 h-8 bg-blue-400 rounded"></div>
                <span>Video</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-16 h-8 bg-green-400 rounded"></div>
                <span>Audio</span>
              </div>
              <div className="text-center text-xl">⬇️ Concat</div>
              <div className="flex items-center gap-2">
                <div className="w-full h-12 bg-gradient-to-r from-blue-400 to-green-400 rounded"></div>
              </div>
              <div className="text-center text-xl">⬇️ Model</div>
              <div className="flex items-center gap-2">
                <div className="w-full h-8 bg-purple-400 rounded text-center text-white">Output</div>
              </div>
            </div>
          </div>

          {/* Late Fusion */}
          <div className="p-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
            <h4 className="font-semibold mb-3 text-blue-600">Late Fusion</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="space-y-2">
                <div className="w-full h-8 bg-blue-400 rounded text-center">Video</div>
                <div className="text-center">⬇️</div>
                <div className="w-full h-12 bg-blue-500 rounded text-center text-white text-xs flex items-center justify-center">
                  Encoder
                </div>
                <div className="text-center">⬇️</div>
                <div className="w-full h-8 bg-blue-600 rounded"></div>
              </div>
              <div className="space-y-2">
                <div className="w-full h-8 bg-green-400 rounded text-center">Audio</div>
                <div className="text-center">⬇️</div>
                <div className="w-full h-12 bg-green-500 rounded text-center text-white text-xs flex items-center justify-center">
                  Encoder
                </div>
                <div className="text-center">⬇️</div>
                <div className="w-full h-8 bg-green-600 rounded"></div>
              </div>
            </div>
            <div className="text-center text-xl mt-2">⬇️ Fusion</div>
            <div className="w-full h-8 bg-purple-400 rounded mt-2 text-center text-white">Output</div>
          </div>

          {/* Hybrid Fusion */}
          <div className="p-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
            <h4 className="font-semibold mb-3 text-purple-600">Hybrid Fusion</h4>
            <div className="space-y-2 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div className="w-full h-8 bg-blue-400 rounded text-center">Video</div>
                <div className="w-full h-8 bg-green-400 rounded text-center">Audio</div>
              </div>
              <div className="text-center">⬇️ ⬅️➡️ ⬇️</div>
              <div className="w-full h-10 bg-gradient-to-r from-blue-500 to-green-500 rounded text-center text-white text-xs flex items-center justify-center">
                Cross-Attention
              </div>
              <div className="text-center">⬇️</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="w-full h-8 bg-blue-600 rounded"></div>
                <div className="w-full h-8 bg-green-600 rounded"></div>
              </div>
              <div className="text-center">⬇️ Fusion</div>
              <div className="w-full h-8 bg-purple-400 rounded text-center text-white">Output</div>
            </div>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">💻 PyTorch 구현 예제</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`import torch
import torch.nn as nn

# Late Fusion 예제
class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.audio_encoder = AudioEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, video, audio):
        v_emb = self.vision_encoder(video)  # [B, 512]
        a_emb = self.audio_encoder(audio)   # [B, 512]

        # Concatenate embeddings
        combined = torch.cat([v_emb, a_emb], dim=-1)  # [B, 1024]
        output = self.fusion(combined)
        return output

# Cross-Attention Fusion 예제
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        # Video attends to Audio
        v2a, _ = self.cross_attn(video_feat, audio_feat, audio_feat)
        video_feat = self.norm(video_feat + v2a)

        # Audio attends to Video
        a2v, _ = self.cross_attn(audio_feat, video_feat, video_feat)
        audio_feat = self.norm(audio_feat + a2v)

        return video_feat, audio_feat`}
        </pre>
      </div>

      {/* Performance Stats */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">82-93%</div>
          <div className="text-sm text-green-100">정확도 범위</div>
        </div>
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">45-85ms</div>
          <div className="text-sm text-blue-100">지연시간 범위</div>
        </div>
        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{selectedStrategies.length}</div>
          <div className="text-sm text-purple-100">선택된 전략</div>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">5</div>
          <div className="text-sm text-orange-100">전략 옵션</div>
        </div>
      </div>
    </div>
  );
}
