'use client';

import { useState } from 'react';
import {
  DollarSign, Cpu, Zap, TrendingDown,
  BarChart3, Settings, Gauge, Cloud
} from 'lucide-react';

interface Instance {
  name: string;
  type: 'gpu' | 'cpu';
  specs: string;
  pricePerHour: number;
  performance: number;
}

export default function ResourceOptimizer() {
  const [cloudProvider, setCloudProvider] = useState<'aws' | 'gcp' | 'azure'>('aws');
  const [instanceType, setInstanceType] = useState('p3.2xlarge');
  const [quantity, setQuantity] = useState(4);
  const [hoursPerMonth, setHoursPerMonth] = useState(720);
  const [spotInstance, setSpotInstance] = useState(false);
  const [batchSize, setBatchSize] = useState(32);
  const [mixedPrecision, setMixedPrecision] = useState(false);

  const instances: Record<string, Instance[]> = {
    aws: [
      { name: 'p3.2xlarge', type: 'gpu', specs: '1x V100, 8 vCPU, 61GB RAM', pricePerHour: 3.06, performance: 100 },
      { name: 'p3.8xlarge', type: 'gpu', specs: '4x V100, 32 vCPU, 244GB RAM', pricePerHour: 12.24, performance: 380 },
      { name: 'p4d.24xlarge', type: 'gpu', specs: '8x A100, 96 vCPU, 1152GB RAM', pricePerHour: 32.77, performance: 800 },
      { name: 'g4dn.xlarge', type: 'gpu', specs: '1x T4, 4 vCPU, 16GB RAM', pricePerHour: 0.526, performance: 40 },
      { name: 'c5.9xlarge', type: 'cpu', specs: '36 vCPU, 72GB RAM', pricePerHour: 1.53, performance: 25 }
    ],
    gcp: [
      { name: 'n1-highmem-8', type: 'gpu', specs: '1x V100, 8 vCPU, 52GB RAM', pricePerHour: 2.48, performance: 100 },
      { name: 'a2-highgpu-1g', type: 'gpu', specs: '1x A100, 12 vCPU, 85GB RAM', pricePerHour: 3.67, performance: 110 },
      { name: 'a2-ultragpu-8g', type: 'gpu', specs: '8x A100, 96 vCPU, 1360GB RAM', pricePerHour: 29.39, performance: 820 }
    ],
    azure: [
      { name: 'NC6s_v3', type: 'gpu', specs: '1x V100, 6 vCPU, 112GB RAM', pricePerHour: 3.06, performance: 100 },
      { name: 'NC24s_v3', type: 'gpu', specs: '4x V100, 24 vCPU, 448GB RAM', pricePerHour: 12.24, performance: 380 },
      { name: 'ND96asr_v4', type: 'gpu', specs: '8x A100, 96 vCPU, 900GB RAM', pricePerHour: 27.20, performance: 800 }
    ]
  };

  const selectedInstance = instances[cloudProvider].find(i => i.name === instanceType) || instances[cloudProvider][0];

  const spotDiscount = spotInstance ? 0.7 : 0;
  const effectivePrice = selectedInstance.pricePerHour * (1 - spotDiscount);
  const monthlyCost = effectivePrice * quantity * hoursPerMonth;

  const mixedPrecisionSpeedup = mixedPrecision ? 1.5 : 1;
  const batchSizeSpeedup = Math.log2(batchSize / 16) * 0.2 + 1;
  const totalSpeedup = mixedPrecisionSpeedup * batchSizeSpeedup;

  const throughput = (selectedInstance.performance * quantity * totalSpeedup);
  const costPerUnit = monthlyCost / throughput;

  const recommendations = [
    {
      title: 'Spot Instance 사용',
      savings: `~${(spotInstance ? 0 : 30)}%`,
      description: 'Spot Instance로 전환하여 비용 절감',
      enabled: !spotInstance
    },
    {
      title: 'Mixed Precision 활성화',
      speedup: '1.5x',
      description: 'FP16 사용으로 학습 속도 향상',
      enabled: !mixedPrecision
    },
    {
      title: '배치 크기 최적화',
      impact: batchSize < 64 ? '증가 권장' : '최적',
      description: `현재: ${batchSize}, 권장: 64-128`
    },
    {
      title: '리전 최적화',
      savings: '~10%',
      description: '저렴한 리전 선택으로 비용 절감'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-green-500 p-3 rounded-lg">
              <DollarSign className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">리소스 최적화 도구</h1>
              <p className="text-slate-300">GPU/CPU Resource Optimizer</p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <DollarSign className="w-4 h-4 text-green-400" />
                <span className="text-sm text-slate-300">월 비용</span>
              </div>
              <div className="text-2xl font-bold text-green-400">
                ${monthlyCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Gauge className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-slate-300">처리량</span>
              </div>
              <div className="text-2xl font-bold">{throughput.toFixed(0)} img/s</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <TrendingDown className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-300">단위 비용</span>
              </div>
              <div className="text-2xl font-bold">${costPerUnit.toFixed(4)}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-slate-300">속도 향상</span>
              </div>
              <div className="text-2xl font-bold text-yellow-400">{totalSpeedup.toFixed(2)}x</div>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Settings className="w-6 h-6 text-cyan-400" />
              인프라 설정
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                  <Cloud className="w-4 h-4 text-blue-400" />
                  클라우드 제공자
                </label>
                <select
                  value={cloudProvider}
                  onChange={(e) => {
                    setCloudProvider(e.target.value as any);
                    setInstanceType(instances[e.target.value as 'aws' | 'gcp' | 'azure'][0].name);
                  }}
                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
                >
                  <option value="aws">AWS</option>
                  <option value="gcp">Google Cloud</option>
                  <option value="azure">Azure</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-purple-400" />
                  인스턴스 타입
                </label>
                <select
                  value={instanceType}
                  onChange={(e) => setInstanceType(e.target.value)}
                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
                >
                  {instances[cloudProvider].map(inst => (
                    <option key={inst.name} value={inst.name}>
                      {inst.name} - {inst.specs} - ${inst.pricePerHour}/hr
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold mb-2">
                  인스턴스 수량: {quantity}
                </label>
                <input
                  type="range"
                  min={1}
                  max={16}
                  value={quantity}
                  onChange={(e) => setQuantity(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold mb-2">
                  월 사용 시간: {hoursPerMonth}h
                </label>
                <input
                  type="range"
                  min={100}
                  max={744}
                  step={10}
                  value={hoursPerMonth}
                  onChange={(e) => setHoursPerMonth(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-slate-700/50 rounded-lg">
                <div>
                  <div className="font-semibold">Spot Instance</div>
                  <div className="text-sm text-slate-400">~70% 할인</div>
                </div>
                <button
                  onClick={() => setSpotInstance(!spotInstance)}
                  className={`px-4 py-2 rounded-lg font-semibold ${
                    spotInstance ? 'bg-green-600' : 'bg-slate-600'
                  }`}
                >
                  {spotInstance ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>
          </div>

          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Zap className="w-6 h-6 text-yellow-400" />
              성능 최적화
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-semibold mb-2">
                  배치 크기: {batchSize}
                </label>
                <input
                  type="range"
                  min={8}
                  max={256}
                  step={8}
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full"
                />
                <div className="mt-2 text-sm text-slate-400">
                  속도 향상: {batchSizeSpeedup.toFixed(2)}x
                </div>
              </div>

              <div className="flex items-center justify-between p-4 bg-slate-700/50 rounded-lg">
                <div>
                  <div className="font-semibold">Mixed Precision (FP16)</div>
                  <div className="text-sm text-slate-400">~1.5x 속도 향상</div>
                </div>
                <button
                  onClick={() => setMixedPrecision(!mixedPrecision)}
                  className={`px-4 py-2 rounded-lg font-semibold ${
                    mixedPrecision ? 'bg-green-600' : 'bg-slate-600'
                  }`}
                >
                  {mixedPrecision ? 'ON' : 'OFF'}
                </button>
              </div>

              <div className="bg-blue-500/20 border border-blue-500 rounded-lg p-4">
                <h3 className="font-semibold mb-2 text-blue-400">총 성능 향상</h3>
                <div className="text-3xl font-bold text-blue-400">{totalSpeedup.toFixed(2)}x</div>
                <div className="mt-2 text-sm text-slate-300">
                  Mixed Precision: {mixedPrecisionSpeedup}x<br />
                  Batch Size: {batchSizeSpeedup.toFixed(2)}x
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-green-400" />
            비용 분석
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-2">시간당 비용</div>
              <div className="text-3xl font-bold mb-1">
                ${(effectivePrice * quantity).toFixed(2)}
              </div>
              <div className="text-sm text-slate-400">
                {spotInstance ? '(Spot Instance)' : '(On-Demand)'}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-2">일일 비용</div>
              <div className="text-3xl font-bold mb-1">
                ${(effectivePrice * quantity * 24).toFixed(0)}
              </div>
              <div className="text-sm text-slate-400">
                24시간 기준
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-2">월별 비용</div>
              <div className="text-3xl font-bold mb-1 text-green-400">
                ${monthlyCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
              <div className="text-sm text-slate-400">
                {hoursPerMonth}시간 기준
              </div>
            </div>
          </div>

          {spotInstance && (
            <div className="mt-4 bg-green-500/20 border border-green-500 rounded-lg p-4">
              <div className="font-semibold text-green-400 mb-1">비용 절감</div>
              <div className="text-2xl font-bold text-green-400">
                ${((selectedInstance.pricePerHour * quantity * hoursPerMonth * 0.7)).toLocaleString(undefined, { maximumFractionDigits: 0 })}/월 절약
              </div>
            </div>
          )}
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">최적화 권장사항</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {recommendations.map((rec, idx) => (
              <div
                key={idx}
                className={`p-4 rounded-lg border ${
                  rec.enabled === false
                    ? 'border-green-500 bg-green-500/10'
                    : 'border-slate-600 bg-slate-700/50'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold">{rec.title}</h3>
                  {rec.savings && (
                    <span className="px-2 py-1 bg-green-600 rounded text-xs font-bold">
                      {rec.savings}
                    </span>
                  )}
                  {rec.speedup && (
                    <span className="px-2 py-1 bg-blue-600 rounded text-xs font-bold">
                      {rec.speedup}
                    </span>
                  )}
                </div>
                <p className="text-sm text-slate-400">{rec.description}</p>
                {rec.impact && (
                  <div className="mt-2 text-sm">
                    <span className="text-slate-400">영향도: </span>
                    <span className="font-semibold">{rec.impact}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">ROI 계산</h2>
          <div className="bg-slate-900 rounded-lg p-4">
            <pre className="text-sm text-slate-300">
              {`현재 설정:
  인스턴스: ${quantity}x ${selectedInstance.name}
  월 사용: ${hoursPerMonth}시간
  배치 크기: ${batchSize}
  Mixed Precision: ${mixedPrecision ? 'ON' : 'OFF'}
  Spot Instance: ${spotInstance ? 'ON' : 'OFF'}

비용 분석:
  시간당: $${(effectivePrice * quantity).toFixed(2)}
  일일: $${(effectivePrice * quantity * 24).toFixed(2)}
  월별: $${monthlyCost.toFixed(2)}

성능:
  처리량: ${throughput.toFixed(0)} images/second
  속도 향상: ${totalSpeedup.toFixed(2)}x
  단위 비용: $${costPerUnit.toFixed(6)} per image

ROI:
  ${spotInstance ? '✓' : '✗'} Spot Instance 활성화 ${spotInstance ? `(~$${(monthlyCost * 0.43).toFixed(0)} 절약)` : ''}
  ${mixedPrecision ? '✓' : '✗'} Mixed Precision 활성화 ${mixedPrecision ? '(1.5x 속도)' : ''}
  ${batchSize >= 64 ? '✓' : '✗'} 배치 크기 최적화 ${batchSize >= 64 ? '(최적)' : '(증가 권장)'}`}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
