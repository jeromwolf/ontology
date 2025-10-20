import React, { useState } from 'react';
import { Shield, Plus, Trash2, Check, X } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

interface FirewallRule {
  id: string;
  name: string;
  action: 'allow' | 'deny';
  protocol: 'TCP' | 'UDP' | 'ICMP' | 'Any';
  sourceIp: string;
  destPort: string;
  enabled: boolean;
}

export default function FirewallConfig() {
  const [rules, setRules] = useState<FirewallRule[]>([
    {
      id: '1',
      name: 'Allow HTTP',
      action: 'allow',
      protocol: 'TCP',
      sourceIp: '0.0.0.0/0',
      destPort: '80',
      enabled: true,
    },
    {
      id: '2',
      name: 'Allow HTTPS',
      action: 'allow',
      protocol: 'TCP',
      sourceIp: '0.0.0.0/0',
      destPort: '443',
      enabled: true,
    },
    {
      id: '3',
      name: 'Allow SSH (Admin Only)',
      action: 'allow',
      protocol: 'TCP',
      sourceIp: '192.168.1.0/24',
      destPort: '22',
      enabled: true,
    },
    {
      id: '4',
      name: 'Block Telnet',
      action: 'deny',
      protocol: 'TCP',
      sourceIp: '0.0.0.0/0',
      destPort: '23',
      enabled: true,
    },
  ]);

  const [newRule, setNewRule] = useState<Omit<FirewallRule, 'id' | 'enabled'>>({
    name: '',
    action: 'allow',
    protocol: 'TCP',
    sourceIp: '0.0.0.0/0',
    destPort: '',
  });

  const [testPacket, setTestPacket] = useState({
    sourceIp: '192.168.1.100',
    destPort: '80',
    protocol: 'TCP',
  });

  const [testResult, setTestResult] = useState<{
    allowed: boolean;
    matchedRule: string;
  } | null>(null);

  const addRule = () => {
    if (!newRule.name || !newRule.destPort) {
      alert('규칙 이름과 포트를 입력하세요');
      return;
    }

    const rule: FirewallRule = {
      ...newRule,
      id: Date.now().toString(),
      enabled: true,
    };

    setRules([...rules, rule]);
    setNewRule({
      name: '',
      action: 'allow',
      protocol: 'TCP',
      sourceIp: '0.0.0.0/0',
      destPort: '',
    });
  };

  const deleteRule = (id: string) => {
    setRules(rules.filter((r) => r.id !== id));
  };

  const toggleRule = (id: string) => {
    setRules(
      rules.map((r) => (r.id === id ? { ...r, enabled: !r.enabled } : r))
    );
  };

  const testFirewall = () => {
    // Find matching rule
    const matchedRule = rules.find(
      (r) =>
        r.enabled &&
        r.protocol === testPacket.protocol &&
        r.destPort === testPacket.destPort
    );

    if (matchedRule) {
      setTestResult({
        allowed: matchedRule.action === 'allow',
        matchedRule: matchedRule.name,
      });
    } else {
      // Default deny
      setTestResult({
        allowed: false,
        matchedRule: 'Default Deny Policy',
      });
    }
  };

  const stats = {
    total: rules.length,
    enabled: rules.filter((r) => r.enabled).length,
    allow: rules.filter((r) => r.action === 'allow').length,
    deny: rules.filter((r) => r.action === 'deny').length,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <SimulatorNav />
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
            <Shield className="w-10 h-10 text-blue-500" />
            Firewall Configuration
          </h1>
          <p className="text-xl text-gray-300">방화벽 규칙 설정 및 테스트 시뮬레이터</p>
        </div>

        {/* Statistics */}
        <div className="grid md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-800 rounded-xl p-4">
            <div className="text-2xl font-bold text-blue-400">{stats.total}</div>
            <div className="text-sm text-gray-400">Total Rules</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4">
            <div className="text-2xl font-bold text-green-400">{stats.enabled}</div>
            <div className="text-sm text-gray-400">Enabled</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4">
            <div className="text-2xl font-bold text-emerald-400">{stats.allow}</div>
            <div className="text-sm text-gray-400">Allow Rules</div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4">
            <div className="text-2xl font-bold text-red-400">{stats.deny}</div>
            <div className="text-sm text-gray-400">Deny Rules</div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Firewall Rules */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-6">방화벽 규칙</h2>

            {/* Add New Rule */}
            <div className="bg-gray-900 rounded-lg p-4 mb-4">
              <h3 className="font-bold mb-3">새 규칙 추가</h3>
              <div className="space-y-3">
                <input
                  type="text"
                  placeholder="규칙 이름"
                  value={newRule.name}
                  onChange={(e) => setNewRule({ ...newRule, name: e.target.value })}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
                />
                <div className="grid grid-cols-2 gap-2">
                  <select
                    value={newRule.action}
                    onChange={(e) =>
                      setNewRule({ ...newRule, action: e.target.value as 'allow' | 'deny' })
                    }
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
                  >
                    <option value="allow">Allow</option>
                    <option value="deny">Deny</option>
                  </select>
                  <select
                    value={newRule.protocol}
                    onChange={(e) =>
                      setNewRule({
                        ...newRule,
                        protocol: e.target.value as 'TCP' | 'UDP' | 'ICMP' | 'Any',
                      })
                    }
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
                  >
                    <option value="TCP">TCP</option>
                    <option value="UDP">UDP</option>
                    <option value="ICMP">ICMP</option>
                    <option value="Any">Any</option>
                  </select>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <input
                    type="text"
                    placeholder="Source IP"
                    value={newRule.sourceIp}
                    onChange={(e) => setNewRule({ ...newRule, sourceIp: e.target.value })}
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                  <input
                    type="text"
                    placeholder="Dest Port"
                    value={newRule.destPort}
                    onChange={(e) => setNewRule({ ...newRule, destPort: e.target.value })}
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
                <button
                  onClick={addRule}
                  className="w-full bg-blue-600 hover:bg-blue-700 py-2 rounded font-semibold flex items-center justify-center gap-2"
                >
                  <Plus className="w-4 h-4" />
                  규칙 추가
                </button>
              </div>
            </div>

            {/* Rules List */}
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {rules.map((rule) => (
                <div
                  key={rule.id}
                  className={`border-2 rounded-lg p-3 ${
                    rule.enabled
                      ? rule.action === 'allow'
                        ? 'border-green-500 bg-green-900/20'
                        : 'border-red-500 bg-red-900/20'
                      : 'border-gray-700 bg-gray-900/50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={rule.enabled}
                        onChange={() => toggleRule(rule.id)}
                        className="w-4 h-4"
                      />
                      <span className="font-bold">{rule.name}</span>
                    </div>
                    <button
                      onClick={() => deleteRule(rule.id)}
                      className="text-red-400 hover:text-red-300"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="text-xs text-gray-400 space-y-1 ml-6">
                    <div>
                      Action: <span className="text-white">{rule.action.toUpperCase()}</span>
                    </div>
                    <div>
                      Protocol: <span className="text-white">{rule.protocol}</span>
                    </div>
                    <div>
                      Source: <span className="text-white">{rule.sourceIp}</span>
                    </div>
                    <div>
                      Port: <span className="text-white">{rule.destPort}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Test Panel */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-6">패킷 테스트</h2>

            <div className="bg-gray-900 rounded-lg p-4 mb-4">
              <h3 className="font-bold mb-3">테스트 패킷 설정</h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-gray-400 block mb-1">Source IP</label>
                  <input
                    type="text"
                    value={testPacket.sourceIp}
                    onChange={(e) => setTestPacket({ ...testPacket, sourceIp: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-400 block mb-1">Destination Port</label>
                  <input
                    type="text"
                    value={testPacket.destPort}
                    onChange={(e) => setTestPacket({ ...testPacket, destPort: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-400 block mb-1">Protocol</label>
                  <select
                    value={testPacket.protocol}
                    onChange={(e) => setTestPacket({ ...testPacket, protocol: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
                  >
                    <option value="TCP">TCP</option>
                    <option value="UDP">UDP</option>
                    <option value="ICMP">ICMP</option>
                  </select>
                </div>
                <button
                  onClick={testFirewall}
                  className="w-full bg-purple-600 hover:bg-purple-700 py-2 rounded font-semibold"
                >
                  테스트 실행
                </button>
              </div>
            </div>

            {testResult && (
              <div
                className={`border-2 rounded-lg p-4 ${
                  testResult.allowed
                    ? 'border-green-500 bg-green-900/30'
                    : 'border-red-500 bg-red-900/30'
                }`}
              >
                <div className="flex items-center gap-3 mb-3">
                  {testResult.allowed ? (
                    <Check className="w-8 h-8 text-green-400" />
                  ) : (
                    <X className="w-8 h-8 text-red-400" />
                  )}
                  <div>
                    <h3 className="font-bold text-lg">
                      {testResult.allowed ? '패킷 허용' : '패킷 차단'}
                    </h3>
                    <p className="text-sm text-gray-400">
                      Matched Rule: {testResult.matchedRule}
                    </p>
                  </div>
                </div>
                <div className="text-sm bg-black/30 rounded p-3 font-mono">
                  <div>Source: {testPacket.sourceIp}</div>
                  <div>Protocol: {testPacket.protocol}</div>
                  <div>Port: {testPacket.destPort}</div>
                  <div className="mt-2 pt-2 border-t border-gray-700">
                    Result:{' '}
                    <span className={testResult.allowed ? 'text-green-400' : 'text-red-400'}>
                      {testResult.allowed ? 'ALLOWED' : 'DENIED'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {!testResult && (
              <div className="bg-gray-900 rounded-lg p-8 text-center text-gray-500">
                테스트 패킷을 설정하고 실행 버튼을 클릭하세요
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
