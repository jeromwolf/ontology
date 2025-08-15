'use client';

import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, Activity, BarChart3, AlertCircle, Info, Play, Pause } from 'lucide-react';

interface SimulationParams {
  initialValue: number;
  expectedReturn: number;
  volatility: number;
  timeHorizon: number;
  numSimulations: number;
  confidenceLevel: number;
  contributionMonthly: number;
  inflationRate: number;
}

interface SimulationResults {
  paths: number[][];
  finalValues: number[];
  percentiles: {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
  };
  probabilityOfSuccess: number;
  expectedValue: number;
  var: number;
  cvar: number;
  maxDrawdown: number;
  timeToTarget: number[];
}

interface DistributionData {
  value: number;
  frequency: number;
  cumulative: number;
}

export default function MonteCarloSimulator() {
  const [params, setParams] = useState<SimulationParams>({
    initialValue: 1000000,
    expectedReturn: 8,
    volatility: 15,
    timeHorizon: 20,
    numSimulations: 10000,
    confidenceLevel: 95,
    contributionMonthly: 10000,
    inflationRate: 2.5
  });
  
  const [targetValue, setTargetValue] = useState(3000000);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<SimulationResults | null>(null);
  const [viewMode, setViewMode] = useState<'paths' | 'distribution' | 'statistics'>('paths');
  const [selectedPaths, setSelectedPaths] = useState<number[]>([]);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // 몬테카를로 시뮬레이션 실행
  const runSimulation = async () => {
    setIsRunning(true);
    setProgress(0);
    
    const paths: number[][] = [];
    const finalValues: number[] = [];
    const timeToTarget: number[] = [];
    const dt = 1 / 12; // 월간 단위
    const periods = params.timeHorizon * 12;
    
    // Geometric Brownian Motion with monthly contributions
    for (let sim = 0; sim < params.numSimulations; sim++) {
      const path: number[] = [params.initialValue];
      let currentValue = params.initialValue;
      let targetReached = false;
      
      for (let t = 1; t <= periods; t++) {
        // GBM with drift and volatility
        const drift = (params.expectedReturn - params.inflationRate) / 100;
        const vol = params.volatility / 100;
        const randomShock = Math.sqrt(dt) * normalRandom();
        
        // 가격 변동
        const returns = drift * dt + vol * randomShock;
        currentValue = currentValue * (1 + returns) + params.contributionMonthly;
        
        path.push(currentValue);
        
        // 목표 도달 시간 기록
        if (!targetReached && currentValue >= targetValue) {
          timeToTarget.push(t / 12);
          targetReached = true;
        }
      }
      
      paths.push(path);
      finalValues.push(currentValue);
      
      // 진행률 업데이트
      if (sim % 100 === 0) {
        setProgress((sim / params.numSimulations) * 100);
        await new Promise(resolve => setTimeout(resolve, 0)); // UI 업데이트를 위한 비동기 처리
      }
    }
    
    // 결과 분석
    finalValues.sort((a, b) => a - b);
    
    const percentiles = {
      p5: finalValues[Math.floor(params.numSimulations * 0.05)],
      p25: finalValues[Math.floor(params.numSimulations * 0.25)],
      p50: finalValues[Math.floor(params.numSimulations * 0.50)],
      p75: finalValues[Math.floor(params.numSimulations * 0.75)],
      p95: finalValues[Math.floor(params.numSimulations * 0.95)]
    };
    
    // VaR와 CVaR 계산
    const varIndex = Math.floor(params.numSimulations * (1 - params.confidenceLevel / 100));
    const varValue = finalValues[varIndex];
    const cvarValues = finalValues.slice(0, varIndex);
    const cvar = cvarValues.reduce((sum, val) => sum + val, 0) / cvarValues.length;
    
    // 최대 낙폭 계산
    let maxDrawdown = 0;
    paths.forEach(path => {
      let peak = path[0];
      for (let i = 1; i < path.length; i++) {
        if (path[i] > peak) peak = path[i];
        const drawdown = (peak - path[i]) / peak * 100;
        if (drawdown > maxDrawdown) maxDrawdown = drawdown;
      }
    });
    
    const results: SimulationResults = {
      paths: paths.slice(0, 100), // 시각화를 위해 100개만 저장
      finalValues,
      percentiles,
      probabilityOfSuccess: finalValues.filter(v => v >= targetValue).length / params.numSimulations,
      expectedValue: finalValues.reduce((sum, val) => sum + val, 0) / params.numSimulations,
      var: varValue,
      cvar,
      maxDrawdown,
      timeToTarget
    };
    
    setResults(results);
    setIsRunning(false);
    setProgress(100);
    
    // 대표 경로 선택 (5%, 25%, 50%, 75%, 95%)
    setSelectedPaths([5, 25, 50, 75, 95]);
  };
  
  // 정규분포 난수 생성 (Box-Muller transform)
  const normalRandom = (): number => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };
  
  // 경로 시각화
  useEffect(() => {
    if (!results || viewMode !== 'paths' || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // 캔버스 크기 설정
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    // 배경 클리어
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    // 축 그리기
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, rect.height - 50);
    ctx.lineTo(rect.width - 20, rect.height - 50);
    ctx.moveTo(50, rect.height - 50);
    ctx.lineTo(50, 20);
    ctx.stroke();
    
    // 경로 그리기
    const maxValue = Math.max(...results.paths.flat());
    const xScale = (rect.width - 70) / (params.timeHorizon * 12);
    const yScale = (rect.height - 70) / maxValue;
    
    // 모든 경로 (반투명)
    ctx.globalAlpha = 0.1;
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 0.5;
    
    results.paths.forEach(path => {
      ctx.beginPath();
      path.forEach((value, i) => {
        const x = 50 + i * xScale;
        const y = rect.height - 50 - value * yScale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
    
    // 선택된 백분위 경로
    ctx.globalAlpha = 1;
    ctx.lineWidth = 2;
    const colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'];
    
    selectedPaths.forEach((percentile, idx) => {
      const pathIndex = Math.floor(results.paths.length * percentile / 100);
      const path = results.paths[pathIndex];
      if (!path) return;
      
      ctx.strokeStyle = colors[idx];
      ctx.beginPath();
      path.forEach((value, i) => {
        const x = 50 + i * xScale;
        const y = rect.height - 50 - value * yScale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
    
    // 목표선
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    const targetY = rect.height - 50 - targetValue * yScale;
    ctx.moveTo(50, targetY);
    ctx.lineTo(rect.width - 20, targetY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 레이블
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '12px sans-serif';
    ctx.fillText('시간 (년)', rect.width / 2, rect.height - 10);
    ctx.save();
    ctx.translate(15, rect.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('포트폴리오 가치', 0, 0);
    ctx.restore();
  }, [results, viewMode, selectedPaths, params.timeHorizon, targetValue]);
  
  // 분포 히스토그램 생성
  const getDistributionData = (): DistributionData[] => {
    if (!results) return [];
    
    const bins = 50;
    const min = Math.min(...results.finalValues);
    const max = Math.max(...results.finalValues);
    const binWidth = (max - min) / bins;
    
    const histogram: DistributionData[] = [];
    let cumulative = 0;
    
    for (let i = 0; i < bins; i++) {
      const binStart = min + i * binWidth;
      const binEnd = binStart + binWidth;
      const frequency = results.finalValues.filter(v => v >= binStart && v < binEnd).length;
      cumulative += frequency;
      
      histogram.push({
        value: binStart + binWidth / 2,
        frequency: frequency / results.finalValues.length,
        cumulative: cumulative / results.finalValues.length
      });
    }
    
    return histogram;
  };

  return (
    <div className="space-y-6">
      {/* 시뮬레이션 파라미터 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4">시뮬레이션 설정</h3>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">초기 투자금</label>
            <input
              type="number"
              value={params.initialValue}
              onChange={(e) => setParams({ ...params, initialValue: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
              step="100000"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">기대 수익률 (%)</label>
            <input
              type="number"
              value={params.expectedReturn}
              onChange={(e) => setParams({ ...params, expectedReturn: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
              step="0.5"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">변동성 (%)</label>
            <input
              type="number"
              value={params.volatility}
              onChange={(e) => setParams({ ...params, volatility: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
              step="1"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">투자 기간 (년)</label>
            <input
              type="number"
              value={params.timeHorizon}
              onChange={(e) => setParams({ ...params, timeHorizon: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
              min="1"
              max="50"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">월 적립금</label>
            <input
              type="number"
              value={params.contributionMonthly}
              onChange={(e) => setParams({ ...params, contributionMonthly: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
              step="10000"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">물가상승률 (%)</label>
            <input
              type="number"
              value={params.inflationRate}
              onChange={(e) => setParams({ ...params, inflationRate: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
              step="0.1"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">시뮬레이션 횟수</label>
            <select
              value={params.numSimulations}
              onChange={(e) => setParams({ ...params, numSimulations: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
            >
              <option value={1000}>1,000회</option>
              <option value={5000}>5,000회</option>
              <option value={10000}>10,000회</option>
              <option value={50000}>50,000회</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">목표 금액</label>
            <input
              type="number"
              value={targetValue}
              onChange={(e) => setTargetValue(Number(e.target.value))}
              className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900"
              step="100000"
            />
          </div>
        </div>
        
        <button
          onClick={runSimulation}
          disabled={isRunning}
          className="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isRunning ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              시뮬레이션 실행 중... ({progress.toFixed(0)}%)
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              시뮬레이션 실행
            </>
          )}
        </button>
      </div>

      {results && (
        <>
          {/* 결과 요약 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">목표 달성 확률</p>
              <p className={`text-2xl font-bold ${
                results.probabilityOfSuccess > 0.8 ? 'text-green-600' : 
                results.probabilityOfSuccess > 0.5 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {(results.probabilityOfSuccess * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">기대값</p>
              <p className="text-2xl font-bold">
                ₩{(results.expectedValue / 10000).toFixed(0)}만원
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                VaR ({params.confidenceLevel}%)
              </p>
              <p className="text-2xl font-bold text-red-600">
                ₩{(results.var / 10000).toFixed(0)}만원
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">CVaR</p>
              <p className="text-2xl font-bold text-red-600">
                ₩{(results.cvar / 10000).toFixed(0)}만원
              </p>
            </div>
          </div>

          {/* 백분위수 요약 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">시나리오별 최종 자산</h3>
            
            <div className="grid grid-cols-5 gap-4 text-center">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">최악 (5%)</p>
                <p className="text-xl font-bold text-red-600">
                  ₩{(results.percentiles.p5 / 10000).toFixed(0)}만원
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">하위 (25%)</p>
                <p className="text-xl font-bold text-yellow-600">
                  ₩{(results.percentiles.p25 / 10000).toFixed(0)}만원
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">중간 (50%)</p>
                <p className="text-xl font-bold text-green-600">
                  ₩{(results.percentiles.p50 / 10000).toFixed(0)}만원
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">상위 (75%)</p>
                <p className="text-xl font-bold text-blue-600">
                  ₩{(results.percentiles.p75 / 10000).toFixed(0)}만원
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">최선 (95%)</p>
                <p className="text-xl font-bold text-purple-600">
                  ₩{(results.percentiles.p95 / 10000).toFixed(0)}만원
                </p>
              </div>
            </div>
          </div>

          {/* 뷰 모드 선택 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-2 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex gap-2">
              <button
                onClick={() => setViewMode('paths')}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  viewMode === 'paths'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700'
                }`}
              >
                시뮬레이션 경로
              </button>
              <button
                onClick={() => setViewMode('distribution')}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  viewMode === 'distribution'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700'
                }`}
              >
                확률 분포
              </button>
              <button
                onClick={() => setViewMode('statistics')}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  viewMode === 'statistics'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700'
                }`}
              >
                상세 통계
              </button>
            </div>
          </div>

          {/* 시각화 영역 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            {viewMode === 'paths' && (
              <div>
                <h3 className="text-lg font-semibold mb-4">시뮬레이션 경로</h3>
                <canvas
                  ref={canvasRef}
                  className="w-full h-96 bg-gray-900 rounded-lg"
                />
                <div className="mt-4 flex flex-wrap gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-500 rounded" />
                    <span>5% (최악)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-yellow-500 rounded" />
                    <span>25% (하위)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-green-500 rounded" />
                    <span>50% (중간)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-blue-500 rounded" />
                    <span>75% (상위)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-purple-500 rounded" />
                    <span>95% (최선)</span>
                  </div>
                </div>
              </div>
            )}
            
            {viewMode === 'distribution' && (
              <div>
                <h3 className="text-lg font-semibold mb-4">최종 자산 분포</h3>
                <div className="h-64 flex items-end gap-0.5">
                  {getDistributionData().map((bin, idx) => (
                    <div
                      key={idx}
                      className="flex-1 bg-blue-500 hover:bg-blue-600 transition-colors relative group"
                      style={{ height: `${bin.frequency * 2000}%` }}
                    >
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                        ₩{(bin.value / 10000).toFixed(0)}만원
                        <br />
                        확률: {(bin.frequency * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
                  최종 포트폴리오 가치 분포
                </div>
              </div>
            )}
            
            {viewMode === 'statistics' && (
              <div>
                <h3 className="text-lg font-semibold mb-4">상세 통계</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium mb-3">리스크 지표</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>최대 낙폭 (MDD)</span>
                        <span className="font-medium text-red-600">
                          -{results.maxDrawdown.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>표준편차</span>
                        <span className="font-medium">
                          ₩{(Math.sqrt(
                            results.finalValues.reduce((sum, val) => 
                              sum + Math.pow(val - results.expectedValue, 2), 0
                            ) / results.finalValues.length
                          ) / 10000).toFixed(0)}만원
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>왜도 (Skewness)</span>
                        <span className="font-medium">
                          {/* 간단한 왜도 계산 */}
                          {(results.finalValues.filter(v => v > results.expectedValue).length / 
                            results.finalValues.length - 0.5).toFixed(3)}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-3">목표 달성 분석</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>평균 달성 시간</span>
                        <span className="font-medium">
                          {results.timeToTarget.length > 0
                            ? `${(results.timeToTarget.reduce((a, b) => a + b, 0) / 
                                results.timeToTarget.length).toFixed(1)}년`
                            : '미달성'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>조기 달성 확률 (10년 내)</span>
                        <span className="font-medium">
                          {(results.timeToTarget.filter(t => t <= 10).length / 
                            params.numSimulations * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>필요 추가 투자액</span>
                        <span className="font-medium">
                          {results.probabilityOfSuccess < 0.8
                            ? `월 ${Math.ceil((targetValue - results.percentiles.p50) / 
                                (params.timeHorizon * 12) / 10000)}만원`
                            : '충분'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* 몬테카를로 시뮬레이션 가이드 */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          몬테카를로 시뮬레이션 가이드
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">주요 개념</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>VaR</strong>: 특정 신뢰수준에서의 최대 예상 손실</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>CVaR</strong>: VaR를 초과하는 손실의 평균 (꼬리 리스크)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>GBM</strong>: 주가의 로그정규분포를 가정한 확률 모델</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>경로 의존성</strong>: 최종 결과뿐만 아니라 과정도 중요</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">활용 팁</h4>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>시뮬레이션 횟수가 많을수록 정확하지만 시간이 오래 걸림</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>변동성은 과거 데이터 기반으로 보수적으로 설정</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>목표 달성 확률 80% 이상을 권장</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span>정기적으로 파라미터를 재조정하여 현실 반영</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}