import React, { useState } from 'react';
import { Shield, Users, Laptop, Network, Lock } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

interface ZeroTrustComponent {
  id: string;
  name: string;
  category: 'identity' | 'device' | 'network' | 'data';
  enabled: boolean;
  description: string;
}

export default function ZeroTrustArchitect() {
  const [components, setComponents] = useState<ZeroTrustComponent[]>([
    {
      id: '1',
      name: 'Multi-Factor Authentication (MFA)',
      category: 'identity',
      enabled: true,
      description: '2단계 이상의 인증 요구',
    },
    {
      id: '2',
      name: 'Single Sign-On (SSO)',
      category: 'identity',
      enabled: true,
      description: '중앙화된 인증 관리',
    },
    {
      id: '3',
      name: 'Just-In-Time (JIT) Access',
      category: 'identity',
      enabled: false,
      description: '필요 시점에만 권한 부여',
    },
    {
      id: '4',
      name: 'Endpoint Security',
      category: 'device',
      enabled: true,
      description: '엔드포인트 보안 솔루션',
    },
    {
      id: '5',
      name: 'MDM/UEM',
      category: 'device',
      enabled: false,
      description: '모바일 기기 관리',
    },
    {
      id: '6',
      name: 'Device Compliance Check',
      category: 'device',
      enabled: true,
      description: '기기 규정 준수 검증',
    },
    {
      id: '7',
      name: 'Micro-Segmentation',
      category: 'network',
      enabled: true,
      description: '네트워크 세밀 분할',
    },
    {
      id: '8',
      name: 'Software-Defined Perimeter (SDP)',
      category: 'network',
      enabled: false,
      description: '소프트웨어 정의 경계',
    },
    {
      id: '9',
      name: 'ZTNA (Zero Trust Network Access)',
      category: 'network',
      enabled: true,
      description: '제로 트러스트 네트워크 접근',
    },
    {
      id: '10',
      name: 'Data Encryption at Rest',
      category: 'data',
      enabled: true,
      description: '저장 데이터 암호화',
    },
    {
      id: '11',
      name: 'Data Loss Prevention (DLP)',
      category: 'data',
      enabled: false,
      description: '데이터 유출 방지',
    },
  ]);

  const [accessRequest, setAccessRequest] = useState({
    user: 'john.doe@company.com',
    device: 'Laptop-001',
    resource: 'HR Database',
    location: 'Office Network',
  });

  const [verificationResult, setVerificationResult] = useState<{
    passed: boolean;
    checks: { name: string; status: boolean }[];
  } | null>(null);

  const toggleComponent = (id: string) => {
    setComponents(components.map((c) => (c.id === id ? { ...c, enabled: !c.enabled } : c)));
  };

  const verifyAccess = () => {
    const identityComponents = components.filter((c) => c.category === 'identity' && c.enabled);
    const deviceComponents = components.filter((c) => c.category === 'device' && c.enabled);
    const networkComponents = components.filter((c) => c.category === 'network' && c.enabled);
    const dataComponents = components.filter((c) => c.category === 'data' && c.enabled);

    const checks = [
      {
        name: 'Identity Verification',
        status: identityComponents.some((c) => c.name.includes('MFA')),
      },
      {
        name: 'Device Compliance',
        status: deviceComponents.some((c) => c.name.includes('Compliance')),
      },
      {
        name: 'Network Segmentation',
        status: networkComponents.some((c) => c.name.includes('Micro-Segmentation')),
      },
      {
        name: 'Data Protection',
        status: dataComponents.some((c) => c.name.includes('Encryption')),
      },
      {
        name: 'Least Privilege',
        status: identityComponents.some((c) => c.name.includes('JIT')) || identityComponents.length >= 2,
      },
    ];

    const passed = checks.every((c) => c.status);
    setVerificationResult({ passed, checks });
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'identity':
        return <Users className="w-5 h-5" />;
      case 'device':
        return <Laptop className="w-5 h-5" />;
      case 'network':
        return <Network className="w-5 h-5" />;
      case 'data':
        return <Lock className="w-5 h-5" />;
      default:
        return null;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'identity':
        return 'border-blue-500 bg-blue-900/20';
      case 'device':
        return 'border-green-500 bg-green-900/20';
      case 'network':
        return 'border-purple-500 bg-purple-900/20';
      case 'data':
        return 'border-yellow-500 bg-yellow-900/20';
      default:
        return 'border-gray-500 bg-gray-900/20';
    }
  };

  const stats = {
    identity: components.filter((c) => c.category === 'identity' && c.enabled).length,
    device: components.filter((c) => c.category === 'device' && c.enabled).length,
    network: components.filter((c) => c.category === 'network' && c.enabled).length,
    data: components.filter((c) => c.category === 'data' && c.enabled).length,
    total: components.filter((c) => c.enabled).length,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <SimulatorNav />
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
            <Shield className="w-10 h-10 text-indigo-500" />
            Zero Trust Architect
          </h1>
          <p className="text-xl text-gray-300">제로 트러스트 보안 아키텍처 설계 시뮬레이터</p>
        </div>

        {/* Statistics */}
        <div className="grid md:grid-cols-5 gap-4 mb-6">
          <div className="bg-gray-800 rounded-xl p-4">
            <div className="text-2xl font-bold text-indigo-400">{stats.total}</div>
            <div className="text-sm text-gray-400">Total Active</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 border-l-4 border-blue-500">
            <div className="text-2xl font-bold text-blue-400">{stats.identity}</div>
            <div className="text-sm text-gray-400">Identity</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 border-l-4 border-green-500">
            <div className="text-2xl font-bold text-green-400">{stats.device}</div>
            <div className="text-sm text-gray-400">Device</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 border-l-4 border-purple-500">
            <div className="text-2xl font-bold text-purple-400">{stats.network}</div>
            <div className="text-sm text-gray-400">Network</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 border-l-4 border-yellow-500">
            <div className="text-2xl font-bold text-yellow-400">{stats.data}</div>
            <div className="text-sm text-gray-400">Data</div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Components Configuration */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-6">보안 구성요소</h2>

            {['identity', 'device', 'network', 'data'].map((category) => (
              <div key={category} className="mb-6">
                <h3 className="font-bold text-lg mb-3 flex items-center gap-2 capitalize">
                  {getCategoryIcon(category)}
                  {category}
                </h3>
                <div className="space-y-2">
                  {components
                    .filter((c) => c.category === category)
                    .map((component) => (
                      <div
                        key={component.id}
                        className={`border-2 rounded-lg p-3 ${getCategoryColor(category)} ${
                          !component.enabled && 'opacity-50'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <input
                                type="checkbox"
                                checked={component.enabled}
                                onChange={() => toggleComponent(component.id)}
                                className="w-4 h-4"
                              />
                              <span className="font-semibold text-sm">{component.name}</span>
                            </div>
                            <p className="text-xs text-gray-400 ml-6">{component.description}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            ))}
          </div>

          {/* Access Verification */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-6">접근 권한 검증</h2>

            <div className="bg-gray-900 rounded-lg p-4 mb-4">
              <h3 className="font-bold mb-3">접근 요청 정보</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <label className="text-gray-400 block mb-1">User</label>
                  <input
                    type="text"
                    value={accessRequest.user}
                    onChange={(e) => setAccessRequest({ ...accessRequest, user: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                  />
                </div>
                <div>
                  <label className="text-gray-400 block mb-1">Device</label>
                  <input
                    type="text"
                    value={accessRequest.device}
                    onChange={(e) => setAccessRequest({ ...accessRequest, device: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                  />
                </div>
                <div>
                  <label className="text-gray-400 block mb-1">Resource</label>
                  <input
                    type="text"
                    value={accessRequest.resource}
                    onChange={(e) => setAccessRequest({ ...accessRequest, resource: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                  />
                </div>
                <div>
                  <label className="text-gray-400 block mb-1">Location</label>
                  <input
                    type="text"
                    value={accessRequest.location}
                    onChange={(e) => setAccessRequest({ ...accessRequest, location: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2"
                  />
                </div>
              </div>
              <button
                onClick={verifyAccess}
                className="w-full mt-4 bg-indigo-600 hover:bg-indigo-700 py-3 rounded-lg font-semibold"
              >
                검증 시작
              </button>
            </div>

            {verificationResult && (
              <div
                className={`border-2 rounded-lg p-4 ${
                  verificationResult.passed
                    ? 'border-green-500 bg-green-900/30'
                    : 'border-red-500 bg-red-900/30'
                }`}
              >
                <h3 className="font-bold text-lg mb-4">
                  {verificationResult.passed ? '✅ 접근 허용' : '❌ 접근 거부'}
                </h3>
                <div className="space-y-2">
                  {verificationResult.checks.map((check, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center justify-between p-2 rounded ${
                        check.status ? 'bg-green-900/30' : 'bg-red-900/30'
                      }`}
                    >
                      <span className="text-sm">{check.name}</span>
                      <span className="text-sm font-bold">
                        {check.status ? '✓ PASS' : '✗ FAIL'}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t border-gray-700 text-sm text-gray-300">
                  {verificationResult.passed ? (
                    <p>모든 보안 요구사항을 충족했습니다. 접근이 허용됩니다.</p>
                  ) : (
                    <p>일부 보안 요구사항을 충족하지 못했습니다. 추가 보안 구성요소를 활성화하세요.</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Zero Trust Principles */}
        <div className="mt-6 bg-gradient-to-r from-indigo-900/50 to-purple-900/50 rounded-xl p-6">
          <h2 className="text-2xl font-bold mb-4">💡 제로 트러스트 핵심 원칙</h2>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div className="bg-black/30 rounded-lg p-4">
              <h3 className="font-bold text-indigo-400 mb-2">Never Trust, Always Verify</h3>
              <p className="text-gray-300">모든 접근 요청을 항상 검증</p>
            </div>
            <div className="bg-black/30 rounded-lg p-4">
              <h3 className="font-bold text-purple-400 mb-2">Least Privilege Access</h3>
              <p className="text-gray-300">최소 권한 원칙 적용</p>
            </div>
            <div className="bg-black/30 rounded-lg p-4">
              <h3 className="font-bold text-pink-400 mb-2">Assume Breach</h3>
              <p className="text-gray-300">침해를 전제한 설계</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
