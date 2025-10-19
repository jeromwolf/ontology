'use client';

import React, { useState } from 'react';
import { Cloud, DollarSign, Gauge, Globe } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

interface CloudProvider {
  id: string;
  name: string;
  region: string;
  services: ServiceAllocation[];
  cost: number;
  uptime: number;
  latency: number;
}

interface ServiceAllocation {
  service: string;
  percentage: number;
  reason: string;
}

export default function MultiCloudManager() {
  const [providers, setProviders] = useState<CloudProvider[]>([
    {
      id: 'aws',
      name: 'AWS',
      region: 'us-east-1',
      services: [
        { service: 'Compute', percentage: 60, reason: 'Best EC2 pricing' },
        { service: 'Storage', percentage: 40, reason: 'S3 ecosystem' }
      ],
      cost: 15000,
      uptime: 99.99,
      latency: 45
    },
    {
      id: 'azure',
      name: 'Azure',
      region: 'eastus',
      services: [
        { service: 'Database', percentage: 70, reason: 'Strong SQL offerings' },
        { service: 'AI/ML', percentage: 50, reason: 'Azure OpenAI Service' }
      ],
      cost: 12000,
      uptime: 99.95,
      latency: 52
    },
    {
      id: 'gcp',
      name: 'GCP',
      region: 'us-central1',
      services: [
        { service: 'Analytics', percentage: 80, reason: 'BigQuery leadership' },
        { service: 'Kubernetes', percentage: 60, reason: 'GKE native' }
      ],
      cost: 10000,
      uptime: 99.97,
      latency: 48
    }
  ]);

  const [strategy, setStrategy] = useState<'cost-optimization' | 'high-availability' | 'performance' | 'vendor-diversity'>('cost-optimization');

  const workloads = [
    { name: 'Web Application', provider: 'AWS', cost: 5000, traffic: '60%' },
    { name: 'Data Analytics', provider: 'GCP', cost: 8000, traffic: '25%' },
    { name: 'AI Services', provider: 'Azure', cost: 4000, traffic: '15%' }
  ];

  const totalCost = providers.reduce((sum, p) => sum + p.cost, 0);
  const avgUptime = providers.reduce((sum, p) => sum + p.uptime, 0) / providers.length;
  const avgLatency = providers.reduce((sum, p) => sum + p.latency, 0) / providers.length;

  const strategyRecommendations = {
    'cost-optimization': {
      title: '비용 최적화 전략',
      description: '각 클라우드의 최저가 서비스 활용',
      recommendations: [
        'AWS: Spot Instances for batch jobs (90% savings)',
        'GCP: Committed Use Discounts for BigQuery',
        'Azure: Reserved Instances for databases (60% savings)'
      ],
      estimatedSavings: '$12,000/month (32%)'
    },
    'high-availability': {
      title: '고가용성 전략',
      description: '다중 클라우드 failover 구성',
      recommendations: [
        'Active-Active replication across AWS and Azure',
        'GCP as disaster recovery site',
        'Cross-cloud DNS failover with Route 53 / Cloud DNS'
      ],
      estimatedSavings: 'N/A (focus on uptime)'
    },
    'performance': {
      title: '성능 최적화 전략',
      description: '지역별 최적 클라우드 선택',
      recommendations: [
        'Use CloudFront (AWS) + Cloudflare for global CDN',
        'Deploy compute close to users (multi-region)',
        'Direct Connect / ExpressRoute for hybrid cloud'
      ],
      estimatedSavings: 'Improved latency: -30ms avg'
    },
    'vendor-diversity': {
      title: '벤더 다각화 전략',
      description: '종속성 최소화 및 리스크 분산',
      recommendations: [
        'Split critical workloads 40/30/30 across providers',
        'Use Terraform for infrastructure as code',
        'Kubernetes for portable container orchestration'
      ],
      estimatedSavings: 'Reduced vendor lock-in risk'
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        <SimulatorNav />

        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent mb-2">
                멀티 클라우드 매니저
              </h1>
              <p className="text-gray-600 dark:text-gray-300">
                AWS, Azure, GCP를 효율적으로 관리하세요
              </p>
            </div>

            <div>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="cost-optimization">비용 최적화</option>
                <option value="high-availability">고가용성</option>
                <option value="performance">성능 최적화</option>
                <option value="vendor-diversity">벤더 다각화</option>
              </select>
            </div>
          </div>

          {/* Total Metrics */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Total Monthly Cost</div>
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                ${totalCost.toLocaleString()}
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Avg Uptime</div>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                {avgUptime.toFixed(2)}%
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Avg Latency</div>
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                {avgLatency.toFixed(0)}ms
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Providers</div>
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                {providers.length}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Cloud Providers */}
          <div className="lg:col-span-2 space-y-6">
            {providers.map((provider) => (
              <div key={provider.id} className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <Cloud className={`w-8 h-8 ${
                      provider.id === 'aws' ? 'text-orange-500' :
                      provider.id === 'azure' ? 'text-blue-500' : 'text-green-500'
                    }`} />
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">{provider.name}</h3>
                      <p className="text-sm text-gray-500">{provider.region}</p>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                      ${provider.cost.toLocaleString()}/mo
                    </div>
                    <div className="text-sm text-gray-500">
                      {((provider.cost / totalCost) * 100).toFixed(0)}% of total
                    </div>
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <Gauge className="w-4 h-4 text-green-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Uptime</span>
                    </div>
                    <div className="text-xl font-bold text-gray-900 dark:text-gray-100">{provider.uptime}%</div>
                  </div>

                  <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <Globe className="w-4 h-4 text-blue-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Latency</span>
                    </div>
                    <div className="text-xl font-bold text-gray-900 dark:text-gray-100">{provider.latency}ms</div>
                  </div>
                </div>

                {/* Service Allocation */}
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">Service Allocation</h4>
                  <div className="space-y-3">
                    {provider.services.map((service, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-700 dark:text-gray-300">{service.service}</span>
                          <span className="font-semibold text-gray-900 dark:text-gray-100">{service.percentage}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              provider.id === 'aws' ? 'bg-orange-500' :
                              provider.id === 'azure' ? 'bg-blue-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${service.percentage}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-500 mt-1">{service.reason}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}

            {/* Workload Distribution */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">Workload Distribution</h3>

              <div className="space-y-3">
                {workloads.map((workload, idx) => (
                  <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-gray-900 dark:text-gray-100">{workload.name}</span>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        workload.provider === 'AWS' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' :
                        workload.provider === 'Azure' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                        'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                      }`}>
                        {workload.provider}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Cost:</span>
                        <span className="ml-2 font-semibold text-gray-900 dark:text-gray-100">
                          ${workload.cost.toLocaleString()}/mo
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">Traffic:</span>
                        <span className="ml-2 font-semibold text-gray-900 dark:text-gray-100">{workload.traffic}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Strategy Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sticky top-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">
                {strategyRecommendations[strategy].title}
              </h3>

              <p className="text-gray-600 dark:text-gray-300 mb-4">
                {strategyRecommendations[strategy].description}
              </p>

              <div className="mb-6">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">권장 사항</h4>
                <ul className="space-y-2">
                  {strategyRecommendations[strategy].recommendations.map((rec, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Estimated Impact</div>
                <div className="text-xl font-bold text-green-600 dark:text-green-400">
                  {strategyRecommendations[strategy].estimatedSavings}
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">관리 도구</h4>

                <div className="space-y-2">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="font-semibold text-sm text-gray-900 dark:text-gray-100">Terraform</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Infrastructure as Code</div>
                  </div>

                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="font-semibold text-sm text-gray-900 dark:text-gray-100">Kubernetes</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Container Orchestration</div>
                  </div>

                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="font-semibold text-sm text-gray-900 dark:text-gray-100">Datadog</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Multi-Cloud Monitoring</div>
                  </div>

                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="font-semibold text-sm text-gray-900 dark:text-gray-100">CloudHealth</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">Cost Management</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
