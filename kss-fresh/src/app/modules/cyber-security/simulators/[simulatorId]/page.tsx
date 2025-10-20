'use client';

import React from 'react';
import { useParams } from 'next/navigation';
import dynamic from 'next/dynamic';

// Dynamic imports for all simulators
const HackingLab = dynamic(() => import('@/components/cyber-security-simulators/HackingLab'), {
  ssr: false,
});
const VulnerabilityScanner = dynamic(
  () => import('@/components/cyber-security-simulators/VulnerabilityScanner'),
  { ssr: false }
);
const FirewallConfig = dynamic(
  () => import('@/components/cyber-security-simulators/FirewallConfig'),
  { ssr: false }
);
const ZeroTrustArchitect = dynamic(
  () => import('@/components/cyber-security-simulators/ZeroTrustArchitect'),
  { ssr: false }
);
const IncidentSimulator = dynamic(
  () => import('@/components/cyber-security-simulators/IncidentSimulator'),
  { ssr: false }
);
const CryptoAnalyzer = dynamic(
  () => import('@/components/cyber-security-simulators/CryptoAnalyzer'),
  { ssr: false }
);

export default function SimulatorPage() {
  const params = useParams();
  const simulatorId = params.simulatorId as string;

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'hacking-lab':
        return <HackingLab />;
      case 'vulnerability-scanner':
        return <VulnerabilityScanner />;
      case 'firewall-config':
        return <FirewallConfig />;
      case 'zero-trust-architect':
        return <ZeroTrustArchitect />;
      case 'incident-simulator':
        return <IncidentSimulator />;
      case 'crypto-analyzer':
        return <CryptoAnalyzer />;
      default:
        return (
          <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-4xl font-bold mb-4">시뮬레이터를 찾을 수 없습니다</h1>
              <p className="text-gray-400 mb-6">요청하신 시뮬레이터가 존재하지 않습니다.</p>
              <a
                href="/modules/cyber-security"
                className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-semibold inline-block"
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
