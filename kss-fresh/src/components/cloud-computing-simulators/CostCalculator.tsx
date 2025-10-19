'use client';

import React, { useState, useMemo } from 'react';
import { DollarSign, TrendingUp, TrendingDown, Calculator, Download } from 'lucide-react';

interface ServiceConfig {
  ec2Instances: number;
  ec2Type: string;
  ec2Hours: number;
  s3Storage: number;
  s3Requests: number;
  rdsInstances: number;
  rdsType: string;
  dataTransfer: number;
  lambdaInvocations: number;
  lambdaMemory: number;
}

export default function CostCalculator() {
  const [provider, setProvider] = useState<'AWS' | 'Azure' | 'GCP'>('AWS');
  const [config, setConfig] = useState<ServiceConfig>({
    ec2Instances: 2,
    ec2Type: 't3.medium',
    ec2Hours: 730,
    s3Storage: 100,
    s3Requests: 1000000,
    rdsInstances: 1,
    rdsType: 'db.t3.medium',
    dataTransfer: 1000,
    lambdaInvocations: 10000000,
    lambdaMemory: 512
  });

  const [savingsOptions, setSavingsOptions] = useState({
    reservedInstances: false,
    savingsPlans: false,
    spotInstances: false,
    autoScaling: false
  });

  // AWS Pricing (simplified)
  const awsPricing = {
    ec2: {
      't3.micro': 0.0104,
      't3.small': 0.0208,
      't3.medium': 0.0416,
      't3.large': 0.0832,
      'm5.large': 0.096,
      'm5.xlarge': 0.192,
      'c5.large': 0.085,
      'c5.xlarge': 0.17
    },
    rds: {
      'db.t3.micro': 0.017,
      'db.t3.small': 0.034,
      'db.t3.medium': 0.068,
      'db.t3.large': 0.136,
      'db.m5.large': 0.192,
      'db.m5.xlarge': 0.384
    },
    s3: {
      storage: 0.023, // per GB
      requests: 0.0004 // per 1000 requests
    },
    dataTransfer: 0.09, // per GB
    lambda: {
      requests: 0.20, // per 1M requests
      compute: 0.0000166667 // per GB-second
    }
  };

  const calculateCost = useMemo(() => {
    let ec2Cost = 0;
    let s3Cost = 0;
    let rdsCost = 0;
    let dataTransferCost = 0;
    let lambdaCost = 0;

    if (provider === 'AWS') {
      // EC2 Cost
      const ec2HourlyRate = awsPricing.ec2[config.ec2Type as keyof typeof awsPricing.ec2] || 0;
      ec2Cost = config.ec2Instances * ec2HourlyRate * config.ec2Hours;

      // Apply savings
      if (savingsOptions.reservedInstances) {
        ec2Cost *= 0.40; // 60% discount
      } else if (savingsOptions.savingsPlans) {
        ec2Cost *= 0.28; // 72% discount
      } else if (savingsOptions.spotInstances) {
        ec2Cost *= 0.10; // 90% discount
      }

      if (savingsOptions.autoScaling) {
        ec2Cost *= 0.70; // 30% reduction from optimization
      }

      // S3 Cost
      s3Cost = (config.s3Storage * awsPricing.s3.storage) +
               (config.s3Requests / 1000 * awsPricing.s3.requests);

      // RDS Cost
      const rdsHourlyRate = awsPricing.rds[config.rdsType as keyof typeof awsPricing.rds] || 0;
      rdsCost = config.rdsInstances * rdsHourlyRate * 730;

      // Data Transfer
      dataTransferCost = config.dataTransfer * awsPricing.dataTransfer;

      // Lambda Cost
      const requestCost = (config.lambdaInvocations / 1000000) * awsPricing.lambda.requests;
      const computeSeconds = config.lambdaInvocations * 0.2; // avg 200ms
      const gbSeconds = (config.lambdaMemory / 1024) * computeSeconds;
      const computeCost = gbSeconds * awsPricing.lambda.compute;
      lambdaCost = requestCost + computeCost;
    }

    const total = ec2Cost + s3Cost + rdsCost + dataTransferCost + lambdaCost;

    return {
      ec2: ec2Cost,
      s3: s3Cost,
      rds: rdsCost,
      dataTransfer: dataTransferCost,
      lambda: lambdaCost,
      total,
      monthly: total,
      yearly: total * 12
    };
  }, [provider, config, savingsOptions]);

  const potentialSavings = useMemo(() => {
    const baseConfig = { ...config };
    const baseSavings = { ...savingsOptions };

    // Calculate base cost without any optimizations
    const tempSavings = {
      reservedInstances: false,
      savingsPlans: false,
      spotInstances: false,
      autoScaling: false
    };

    let baseCost = 0;
    const ec2HourlyRate = awsPricing.ec2[config.ec2Type as keyof typeof awsPricing.ec2] || 0;
    baseCost = config.ec2Instances * ec2HourlyRate * config.ec2Hours;

    const riSavings = baseCost * 0.60;
    const spSavings = baseCost * 0.72;
    const spotSavings = baseCost * 0.90;
    const autoScalingSavings = baseCost * 0.30;

    return {
      reservedInstances: riSavings,
      savingsPlans: spSavings,
      spotInstances: spotSavings,
      autoScaling: autoScalingSavings
    };
  }, [config]);

  const exportCostReport = () => {
    const report = {
      provider,
      configuration: config,
      savingsOptions,
      costs: calculateCost,
      potentialSavings,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cloud-cost-report-${Date.now()}.json`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent mb-2">
                클라우드 비용 계산기
              </h1>
              <p className="text-gray-600 dark:text-gray-300">
                클라우드 서비스 비용을 정확하게 예측하고 최적화 방안을 찾으세요
              </p>
            </div>

            <div className="flex items-center gap-3">
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="AWS">AWS</option>
                <option value="Azure">Azure (Coming Soon)</option>
                <option value="GCP">GCP (Coming Soon)</option>
              </select>

              <button
                onClick={exportCostReport}
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <Download className="w-4 h-4" />
                Export Report
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* EC2 Configuration */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5 text-blue-500" />
                EC2 컴퓨팅 인스턴스
              </h3>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    인스턴스 수
                  </label>
                  <input
                    type="number"
                    value={config.ec2Instances}
                    onChange={(e) => setConfig({ ...config, ec2Instances: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    min="1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    인스턴스 타입
                  </label>
                  <select
                    value={config.ec2Type}
                    onChange={(e) => setConfig({ ...config, ec2Type: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    {Object.keys(awsPricing.ec2).map((type) => (
                      <option key={type} value={type}>{type}</option>
                    ))}
                  </select>
                </div>

                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    월간 실행 시간: {config.ec2Hours} hours
                  </label>
                  <input
                    type="range"
                    value={config.ec2Hours}
                    onChange={(e) => setConfig({ ...config, ec2Hours: Number(e.target.value) })}
                    className="w-full"
                    min="0"
                    max="730"
                    step="10"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0h (0%)</span>
                    <span>365h (50%)</span>
                    <span>730h (100%)</span>
                  </div>
                </div>
              </div>
            </div>

            {/* S3 Storage */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5 text-green-500" />
                S3 스토리지
              </h3>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    스토리지 (GB)
                  </label>
                  <input
                    type="number"
                    value={config.s3Storage}
                    onChange={(e) => setConfig({ ...config, s3Storage: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    min="0"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    월간 요청 수
                  </label>
                  <input
                    type="number"
                    value={config.s3Requests}
                    onChange={(e) => setConfig({ ...config, s3Requests: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    min="0"
                    step="100000"
                  />
                </div>
              </div>
            </div>

            {/* RDS Database */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5 text-purple-500" />
                RDS 데이터베이스
              </h3>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    인스턴스 수
                  </label>
                  <input
                    type="number"
                    value={config.rdsInstances}
                    onChange={(e) => setConfig({ ...config, rdsInstances: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    min="0"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    인스턴스 타입
                  </label>
                  <select
                    value={config.rdsType}
                    onChange={(e) => setConfig({ ...config, rdsType: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    {Object.keys(awsPricing.rds).map((type) => (
                      <option key={type} value={type}>{type}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Lambda & Data Transfer */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                <Calculator className="w-5 h-5 text-orange-500" />
                Lambda & 데이터 전송
              </h3>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Lambda 호출 수 (월)
                  </label>
                  <input
                    type="number"
                    value={config.lambdaInvocations}
                    onChange={(e) => setConfig({ ...config, lambdaInvocations: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    min="0"
                    step="1000000"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Lambda 메모리 (MB)
                  </label>
                  <select
                    value={config.lambdaMemory}
                    onChange={(e) => setConfig({ ...config, lambdaMemory: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    <option value="128">128 MB</option>
                    <option value="256">256 MB</option>
                    <option value="512">512 MB</option>
                    <option value="1024">1024 MB</option>
                    <option value="2048">2048 MB</option>
                  </select>
                </div>

                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    데이터 전송 (GB/월)
                  </label>
                  <input
                    type="number"
                    value={config.dataTransfer}
                    onChange={(e) => setConfig({ ...config, dataTransfer: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    min="0"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Cost Summary */}
          <div className="lg:col-span-1 space-y-6">
            {/* Total Cost */}
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg p-6 text-white sticky top-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold">예상 비용</h3>
                <DollarSign className="w-6 h-6" />
              </div>

              <div className="mb-6">
                <div className="text-sm opacity-90 mb-1">월간 총 비용</div>
                <div className="text-4xl font-bold">${calculateCost.monthly.toFixed(2)}</div>
              </div>

              <div className="mb-6">
                <div className="text-sm opacity-90 mb-1">연간 총 비용</div>
                <div className="text-2xl font-bold">${calculateCost.yearly.toFixed(2)}</div>
              </div>

              <div className="border-t border-white/20 pt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span>EC2</span>
                  <span className="font-semibold">${calculateCost.ec2.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>S3</span>
                  <span className="font-semibold">${calculateCost.s3.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>RDS</span>
                  <span className="font-semibold">${calculateCost.rds.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Lambda</span>
                  <span className="font-semibold">${calculateCost.lambda.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Data Transfer</span>
                  <span className="font-semibold">${calculateCost.dataTransfer.toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Savings Options */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                <TrendingDown className="w-5 h-5 text-green-500" />
                비용 절감 옵션
              </h3>

              <div className="space-y-4">
                <label className="flex items-center justify-between cursor-pointer">
                  <div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">Reserved Instances</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">최대 60% 할인</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={savingsOptions.reservedInstances}
                    onChange={(e) => setSavingsOptions({ ...savingsOptions, reservedInstances: e.target.checked })}
                    className="w-5 h-5"
                  />
                </label>

                <label className="flex items-center justify-between cursor-pointer">
                  <div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">Savings Plans</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">최대 72% 할인</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={savingsOptions.savingsPlans}
                    onChange={(e) => setSavingsOptions({ ...savingsOptions, savingsPlans: e.target.checked })}
                    className="w-5 h-5"
                  />
                </label>

                <label className="flex items-center justify-between cursor-pointer">
                  <div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">Spot Instances</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">최대 90% 할인</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={savingsOptions.spotInstances}
                    onChange={(e) => setSavingsOptions({ ...savingsOptions, spotInstances: e.target.checked })}
                    className="w-5 h-5"
                  />
                </label>

                <label className="flex items-center justify-between cursor-pointer">
                  <div>
                    <div className="font-semibold text-gray-900 dark:text-gray-100">Auto Scaling</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">약 30% 절감</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={savingsOptions.autoScaling}
                    onChange={(e) => setSavingsOptions({ ...savingsOptions, autoScaling: e.target.checked })}
                    className="w-5 h-5"
                  />
                </label>
              </div>

              <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">예상 절감액 (월간)</div>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  ${(
                    (savingsOptions.reservedInstances ? potentialSavings.reservedInstances : 0) +
                    (savingsOptions.savingsPlans ? potentialSavings.savingsPlans : 0) +
                    (savingsOptions.spotInstances ? potentialSavings.spotInstances : 0) +
                    (savingsOptions.autoScaling ? potentialSavings.autoScaling : 0)
                  ).toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
