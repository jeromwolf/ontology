'use client';

import React from 'react';
import { useParams } from 'next/navigation';
import dynamic from 'next/dynamic';

// Dynamic imports for all simulators
const BiasDetector = dynamic(() => import('@/components/ai-ethics-simulators/BiasDetector'), {
  ssr: false,
});
const FairnessAnalyzer = dynamic(
  () => import('@/components/ai-ethics-simulators/FairnessAnalyzer'),
  { ssr: false }
);
const EthicsFramework = dynamic(
  () => import('@/components/ai-ethics-simulators/EthicsFramework'),
  { ssr: false }
);
const ImpactAssessment = dynamic(
  () => import('@/components/ai-ethics-simulators/ImpactAssessment'),
  { ssr: false }
);

export default function SimulatorPage() {
  const params = useParams();
  const simulatorId = params.simulatorId as string;

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'bias-detector':
        return <BiasDetector />;
      case 'fairness-analyzer':
        return <FairnessAnalyzer />;
      case 'ethics-framework':
        return <EthicsFramework />;
      case 'impact-assessment':
        return <ImpactAssessment />;
      default:
        return (
          <div className="min-h-screen bg-gradient-to-br from-rose-50 to-pink-50 dark:from-gray-900 dark:to-rose-950 flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-4xl font-bold mb-4 text-gray-900 dark:text-white">시뮬레이터를 찾을 수 없습니다</h1>
              <p className="text-gray-600 dark:text-gray-400 mb-6">요청하신 시뮬레이터가 존재하지 않습니다.</p>
              <a
                href="/modules/ai-ethics"
                className="bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-600 hover:to-pink-700 text-white px-6 py-3 rounded-lg font-semibold inline-block"
              >
                모듈로 돌아가기
              </a>
            </div>
          </div>
        );
    }
  };

  return getSimulatorComponent();
}
