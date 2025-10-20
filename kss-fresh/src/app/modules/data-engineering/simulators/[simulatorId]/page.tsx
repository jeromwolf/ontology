'use client';

import { use } from 'react';
import dynamic from 'next/dynamic';

// 동적 임포트로 시뮬레이터 컴포넌트 로드
const EDAPlayground = dynamic(() => import('@/components/data-engineering-simulators/EDAPlayground'), { ssr: false });
const ETLPipelineDesigner = dynamic(() => import('@/components/data-engineering-simulators/ETLPipelineDesigner'), { ssr: false });
const StreamProcessingLab = dynamic(() => import('@/components/data-engineering-simulators/StreamProcessingLab'), { ssr: false });
const DataLakehouseArchitect = dynamic(() => import('@/components/data-engineering-simulators/DataLakehouseArchitect'), { ssr: false });
const AirflowDAGBuilder = dynamic(() => import('@/components/data-engineering-simulators/AirflowDAGBuilder'), { ssr: false });
const SparkOptimizer = dynamic(() => import('@/components/data-engineering-simulators/SparkOptimizer'), { ssr: false });
const DataQualitySuite = dynamic(() => import('@/components/data-engineering-simulators/DataQualitySuite'), { ssr: false });
const CloudCostCalculator = dynamic(() => import('@/components/data-engineering-simulators/CloudCostCalculator'), { ssr: false });
const DataLineageExplorer = dynamic(() => import('@/components/data-engineering-simulators/DataLineageExplorer'), { ssr: false });
const SQLPerformanceTuner = dynamic(() => import('@/components/data-engineering-simulators/SQLPerformanceTuner'), { ssr: false });

interface PageProps {
  params: Promise<{
    simulatorId: string;
  }>;
}

export default function SimulatorPage({ params }: PageProps) {
  const { simulatorId } = use(params);

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'eda-playground':
        return <EDAPlayground />;
      case 'etl-pipeline-designer':
        return <ETLPipelineDesigner />;
      case 'stream-processing-lab':
        return <StreamProcessingLab />;
      case 'data-lakehouse-architect':
        return <DataLakehouseArchitect />;
      case 'airflow-dag-builder':
        return <AirflowDAGBuilder />;
      case 'spark-optimizer':
        return <SparkOptimizer />;
      case 'data-quality-suite':
        return <DataQualitySuite />;
      case 'cloud-cost-calculator':
        return <CloudCostCalculator />;
      case 'data-lineage-explorer':
        return <DataLineageExplorer />;
      case 'sql-performance-tuner':
        return <SQLPerformanceTuner />;
      default:
        return (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">시뮬레이터를 찾을 수 없습니다</h2>
              <p className="text-gray-600 dark:text-gray-400">
                요청하신 시뮬레이터가 존재하지 않습니다.
              </p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {getSimulatorComponent()}
    </div>
  );
}
