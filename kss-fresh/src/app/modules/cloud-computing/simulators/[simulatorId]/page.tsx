'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { useParams } from 'next/navigation';

// Dynamic imports for all simulators
const CloudArchitect = dynamic(() => import('@/components/cloud-computing-simulators/CloudArchitect'), { ssr: false });
const CostCalculator = dynamic(() => import('@/components/cloud-computing-simulators/CostCalculator'), { ssr: false });
const ServerlessLab = dynamic(() => import('@/components/cloud-computing-simulators/ServerlessLab'), { ssr: false });
const ContainerOrchestrator = dynamic(() => import('@/components/cloud-computing-simulators/ContainerOrchestrator'), { ssr: false });
const CloudMigration = dynamic(() => import('@/components/cloud-computing-simulators/CloudMigration'), { ssr: false });
const MultiCloudManager = dynamic(() => import('@/components/cloud-computing-simulators/MultiCloudManager'), { ssr: false });
const CloudSecurityLab = dynamic(() => import('@/components/cloud-computing-simulators/CloudSecurityLab'), { ssr: false });
const InfrastructureAsCode = dynamic(() => import('@/components/cloud-computing-simulators/InfrastructureAsCode'), { ssr: false });

export default function SimulatorPage() {
  const params = useParams();
  const simulatorId = params?.simulatorId as string;

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'cloud-architect':
        return <CloudArchitect />;
      case 'cost-calculator':
        return <CostCalculator />;
      case 'serverless-lab':
        return <ServerlessLab />;
      case 'container-orchestrator':
        return <ContainerOrchestrator />;
      case 'cloud-migration':
        return <CloudMigration />;
      case 'multi-cloud-manager':
        return <MultiCloudManager />;
      case 'cloud-security-lab':
        return <CloudSecurityLab />;
      case 'infrastructure-as-code':
        return <InfrastructureAsCode />;
      default:
        return (
          <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
            <div className="text-center">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                Simulator Not Found
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                The simulator "{simulatorId}" does not exist.
              </p>
            </div>
          </div>
        );
    }
  };

  return getSimulatorComponent();
}
