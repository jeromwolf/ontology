'use client';

import React, { useState } from 'react';
import { Cloud, Server, Database, ArrowRight, CheckCircle, AlertTriangle, Info } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

interface MigrationPhase {
  id: string;
  name: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  progress: number;
  tasks: string[];
}

interface Application {
  id: string;
  name: string;
  type: string;
  complexity: 'low' | 'medium' | 'high';
  dependencies: string[];
  migrationStrategy: '6R' | null;
}

export default function CloudMigration() {
  const [currentPhase, setCurrentPhase] = useState(0);
  const [applications, setApplications] = useState<Application[]>([
    {
      id: 'app1',
      name: 'Web Frontend',
      type: 'React SPA',
      complexity: 'low',
      dependencies: ['API Backend'],
      migrationStrategy: null
    },
    {
      id: 'app2',
      name: 'API Backend',
      type: 'Node.js',
      complexity: 'medium',
      dependencies: ['MySQL DB', 'Redis Cache'],
      migrationStrategy: null
    },
    {
      id: 'app3',
      name: 'MySQL DB',
      type: 'Database',
      complexity: 'high',
      dependencies: [],
      migrationStrategy: null
    },
    {
      id: 'app4',
      name: 'Batch Jobs',
      type: 'Python Scripts',
      complexity: 'medium',
      dependencies: ['MySQL DB'],
      migrationStrategy: null
    }
  ]);

  const [phases, setPhases] = useState<MigrationPhase[]>([
    {
      id: 'assess',
      name: '평가 (Assess)',
      status: 'completed',
      progress: 100,
      tasks: [
        '현재 인프라 분석',
        '애플리케이션 목록 작성',
        '종속성 맵핑',
        '비용 분석'
      ]
    },
    {
      id: 'plan',
      name: '계획 (Plan)',
      status: 'in-progress',
      progress: 60,
      tasks: [
        '6R 전략 선택',
        '마이그레이션 순서 결정',
        '리스크 평가',
        '타임라인 수립'
      ]
    },
    {
      id: 'design',
      name: '설계 (Design)',
      status: 'pending',
      progress: 0,
      tasks: [
        '클라우드 아키텍처 설계',
        '네트워크 구성',
        '보안 설정',
        'DR 계획 수립'
      ]
    },
    {
      id: 'migrate',
      name: '마이그레이션 (Migrate)',
      status: 'pending',
      progress: 0,
      tasks: [
        '파일럿 마이그레이션',
        '데이터 동기화',
        '애플리케이션 이전',
        '테스트 및 검증'
      ]
    },
    {
      id: 'operate',
      name: '운영 (Operate)',
      status: 'pending',
      progress: 0,
      tasks: [
        '모니터링 설정',
        '백업 자동화',
        '비용 최적화',
        '성능 튜닝'
      ]
    }
  ]);

  const sixRStrategies = {
    rehost: {
      name: 'Rehost (Lift & Shift)',
      description: '최소한의 변경으로 클라우드로 이전',
      complexity: 'Low',
      timeframe: '1-3 months',
      costSavings: '10-20%',
      icon: '🚀',
      bestFor: ['레거시 앱', '빠른 마이그레이션 필요']
    },
    replatform: {
      name: 'Replatform (Lift & Reshape)',
      description: '일부 최적화를 통해 클라우드 서비스 활용',
      complexity: 'Medium',
      timeframe: '2-4 months',
      costSavings: '20-40%',
      icon: '🔧',
      bestFor: ['기존 앱 개선', 'Managed 서비스 활용']
    },
    repurchase: {
      name: 'Repurchase (Drop & Shop)',
      description: 'SaaS 솔루션으로 전환',
      complexity: 'Medium',
      timeframe: '1-2 months',
      costSavings: '30-50%',
      icon: '🛒',
      bestFor: ['라이센스 제품', '표준 솔루션']
    },
    refactor: {
      name: 'Refactor (Re-architect)',
      description: '클라우드 네이티브로 재설계',
      complexity: 'High',
      timeframe: '4-12 months',
      costSavings: '40-60%',
      icon: '🏗️',
      bestFor: ['핵심 비즈니스 앱', '확장성 필요']
    },
    retire: {
      name: 'Retire',
      description: '더 이상 필요 없는 시스템 폐기',
      complexity: 'Low',
      timeframe: '< 1 month',
      costSavings: '100%',
      icon: '🗑️',
      bestFor: ['중복 시스템', '사용하지 않는 앱']
    },
    retain: {
      name: 'Retain',
      description: '현재 상태 유지 (온프레미스)',
      complexity: 'N/A',
      timeframe: 'N/A',
      costSavings: '0%',
      icon: '📦',
      bestFor: ['규제 요구사항', '기술적 제약']
    }
  };

  const selectStrategy = (appId: string, strategy: '6R') => {
    setApplications(prev => prev.map(app =>
      app.id === appId ? { ...app, migrationStrategy: strategy } : app
    ));
  };

  const startMigration = () => {
    if (applications.every(app => app.migrationStrategy)) {
      setPhases(prev => prev.map((phase, idx) =>
        idx === currentPhase ? { ...phase, status: 'completed', progress: 100 } :
        idx === currentPhase + 1 ? { ...phase, status: 'in-progress', progress: 0 } :
        phase
      ));
      setCurrentPhase(prev => Math.min(prev + 1, phases.length - 1));
    } else {
      alert('모든 애플리케이션에 대해 마이그레이션 전략을 선택해주세요.');
    }
  };

  const estimatedCost = applications.reduce((sum, app) => {
    const baseCost = app.complexity === 'low' ? 5000 : app.complexity === 'medium' ? 15000 : 30000;
    return sum + baseCost;
  }, 0);

  const estimatedTimeline = applications.reduce((sum, app) => {
    const baseTime = app.complexity === 'low' ? 1 : app.complexity === 'medium' ? 2 : 4;
    return Math.max(sum, baseTime);
  }, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        <SimulatorNav />

        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
            클라우드 마이그레이션 플래너
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            체계적인 클라우드 마이그레이션 계획 수립 및 실행
          </p>

          {/* Progress Timeline */}
          <div className="mt-6 flex items-center justify-between">
            {phases.map((phase, idx) => (
              <React.Fragment key={phase.id}>
                <div className="flex flex-col items-center">
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white ${
                    phase.status === 'completed' ? 'bg-green-500' :
                    phase.status === 'in-progress' ? 'bg-blue-500' :
                    phase.status === 'error' ? 'bg-red-500' : 'bg-gray-300'
                  }`}>
                    {idx + 1}
                  </div>
                  <div className={`text-sm mt-2 font-semibold ${
                    phase.status === 'in-progress' ? 'text-blue-600 dark:text-blue-400' : 'text-gray-600 dark:text-gray-400'
                  }`}>
                    {phase.name}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">{phase.progress}%</div>
                </div>
                {idx < phases.length - 1 && (
                  <div className="flex-1 h-1 bg-gray-300 mx-2">
                    <div
                      className="h-full bg-green-500 transition-all"
                      style={{ width: `${phase.status === 'completed' ? 100 : 0}%` }}
                    />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Applications */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">애플리케이션 목록</h3>

              <div className="space-y-4">
                {applications.map((app) => (
                  <div key={app.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <div className="font-bold text-gray-900 dark:text-gray-100">{app.name}</div>
                        <div className="text-sm text-gray-500">{app.type}</div>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        app.complexity === 'low' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                        app.complexity === 'medium' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                        'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                      }`}>
                        {app.complexity} complexity
                      </span>
                    </div>

                    {app.dependencies.length > 0 && (
                      <div className="mb-3">
                        <div className="text-xs text-gray-500 mb-1">Dependencies:</div>
                        <div className="flex flex-wrap gap-1">
                          {app.dependencies.map((dep, idx) => (
                            <span key={idx} className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs text-gray-700 dark:text-gray-300">
                              {dep}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    <div>
                      <div className="text-xs text-gray-500 mb-2">Migration Strategy:</div>
                      <div className="grid grid-cols-3 gap-2">
                        {Object.entries(sixRStrategies).slice(0, 3).map(([key, strategy]) => (
                          <button
                            key={key}
                            onClick={() => selectStrategy(app.id, key as '6R')}
                            className={`px-3 py-2 text-xs rounded-lg border-2 transition-all ${
                              app.migrationStrategy === key
                                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                            }`}
                          >
                            <div className="text-lg mb-1">{strategy.icon}</div>
                            <div className="font-semibold">{key}</div>
                          </button>
                        ))}
                      </div>
                      <div className="grid grid-cols-3 gap-2 mt-2">
                        {Object.entries(sixRStrategies).slice(3).map(([key, strategy]) => (
                          <button
                            key={key}
                            onClick={() => selectStrategy(app.id, key as '6R')}
                            className={`px-3 py-2 text-xs rounded-lg border-2 transition-all ${
                              app.migrationStrategy === key
                                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                            }`}
                          >
                            <div className="text-lg mb-1">{strategy.icon}</div>
                            <div className="font-semibold">{key}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={startMigration}
                className="w-full mt-6 px-4 py-3 bg-purple-500 hover:bg-purple-600 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
              >
                <ArrowRight className="w-5 h-5" />
                다음 단계로 진행
              </button>
            </div>

            {/* Current Phase Tasks */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">
                현재 단계: {phases[currentPhase]?.name}
              </h3>

              <div className="space-y-2">
                {phases[currentPhase]?.tasks.map((task, idx) => (
                  <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span className="text-gray-900 dark:text-gray-100">{task}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 6R Strategies & Estimates */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sticky top-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">예상 결과</h3>

              <div className="space-y-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">예상 비용</div>
                  <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                    ${estimatedCost.toLocaleString()}
                  </div>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">예상 기간</div>
                  <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                    {estimatedTimeline} months
                  </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">애플리케이션</div>
                  <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                    {applications.length}
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <h4 className="font-bold text-gray-900 dark:text-gray-100 mb-3">6R 전략 가이드</h4>

                <div className="space-y-3">
                  {Object.entries(sixRStrategies).map(([key, strategy]) => (
                    <div key={key} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-lg">{strategy.icon}</span>
                        <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">{strategy.name}</span>
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">{strategy.description}</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-500">복잡도:</span>
                          <span className="ml-1 font-semibold text-gray-900 dark:text-gray-100">{strategy.complexity}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">기간:</span>
                          <span className="ml-1 font-semibold text-gray-900 dark:text-gray-100">{strategy.timeframe}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
