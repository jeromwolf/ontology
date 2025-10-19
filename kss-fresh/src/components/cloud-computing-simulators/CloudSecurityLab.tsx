'use client';

import React, { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, XCircle, Lock, Key, Eye } from 'lucide-react';

interface SecurityCheck {
  id: string;
  category: string;
  name: string;
  status: 'pass' | 'fail' | 'warning';
  severity: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  remediation: string;
}

interface IAMPolicy {
  id: string;
  principal: string;
  actions: string[];
  resources: string[];
  effect: 'Allow' | 'Deny';
}

export default function CloudSecurityLab() {
  const [securityChecks, setSecurityChecks] = useState<SecurityCheck[]>([
    {
      id: '1',
      category: 'IAM',
      name: 'Root account MFA enabled',
      status: 'pass',
      severity: 'critical',
      description: 'Root account has MFA configured',
      remediation: 'Enable MFA for root account'
    },
    {
      id: '2',
      category: 'IAM',
      name: 'No overly permissive policies',
      status: 'fail',
      severity: 'high',
      description: 'Found policies with * actions on * resources',
      remediation: 'Apply principle of least privilege'
    },
    {
      id: '3',
      category: 'Network',
      name: 'Security groups ingress rules',
      status: 'warning',
      severity: 'medium',
      description: 'Some security groups allow 0.0.0.0/0',
      remediation: 'Restrict ingress to specific IP ranges'
    },
    {
      id: '4',
      category: 'Encryption',
      name: 'S3 buckets encrypted',
      status: 'pass',
      severity: 'high',
      description: 'All S3 buckets have encryption enabled',
      remediation: 'N/A'
    },
    {
      id: '5',
      category: 'Encryption',
      name: 'RDS encryption at rest',
      status: 'fail',
      severity: 'critical',
      description: 'RDS instances without encryption',
      remediation: 'Enable encryption for all RDS instances'
    },
    {
      id: '6',
      category: 'Monitoring',
      name: 'CloudTrail enabled',
      status: 'pass',
      severity: 'critical',
      description: 'CloudTrail logging is active',
      remediation: 'N/A'
    },
    {
      id: '7',
      category: 'Network',
      name: 'VPC flow logs enabled',
      status: 'warning',
      severity: 'medium',
      description: 'VPC flow logs not enabled for all VPCs',
      remediation: 'Enable flow logs for network monitoring'
    },
    {
      id: '8',
      category: 'Access',
      name: 'Public S3 buckets',
      status: 'fail',
      severity: 'critical',
      description: 'Found publicly accessible S3 buckets',
      remediation: 'Block public access to S3 buckets'
    }
  ]);

  const [iamPolicies, setIamPolicies] = useState<IAMPolicy[]>([
    {
      id: 'p1',
      principal: 'user/admin',
      actions: ['*'],
      resources: ['*'],
      effect: 'Allow'
    },
    {
      id: 'p2',
      principal: 'role/developer',
      actions: ['s3:GetObject', 's3:PutObject'],
      resources: ['arn:aws:s3:::my-app-bucket/*'],
      effect: 'Allow'
    },
    {
      id: 'p3',
      principal: 'role/read-only',
      actions: ['s3:GetObject', 'ec2:Describe*'],
      resources: ['*'],
      effect: 'Allow'
    }
  ]);

  const [selectedTab, setSelectedTab] = useState<'checks' | 'iam' | 'compliance'>('checks');

  const runSecurityScan = () => {
    // Simulate security scan
    setSecurityChecks(prev => prev.map(check => ({
      ...check,
      status: Math.random() > 0.7 ? 'fail' : Math.random() > 0.5 ? 'warning' : 'pass'
    })));
  };

  const fixIssue = (id: string) => {
    setSecurityChecks(prev => prev.map(check =>
      check.id === id ? { ...check, status: 'pass' as const } : check
    ));
  };

  const categorySummary = securityChecks.reduce((acc, check) => {
    if (!acc[check.category]) {
      acc[check.category] = { pass: 0, fail: 0, warning: 0 };
    }
    acc[check.category][check.status]++;
    return acc;
  }, {} as Record<string, { pass: number; fail: number; warning: number }>);

  const totalPass = securityChecks.filter(c => c.status === 'pass').length;
  const totalFail = securityChecks.filter(c => c.status === 'fail').length;
  const totalWarning = securityChecks.filter(c => c.status === 'warning').length;
  const securityScore = ((totalPass / securityChecks.length) * 100).toFixed(0);

  const complianceFrameworks = [
    { name: 'CIS AWS Foundations', compliance: 75, color: 'bg-blue-500' },
    { name: 'PCI DSS', compliance: 60, color: 'bg-purple-500' },
    { name: 'HIPAA', compliance: 85, color: 'bg-green-500' },
    { name: 'SOC 2', compliance: 70, color: 'bg-orange-500' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 to-orange-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent mb-2">
                클라우드 보안 랩
              </h1>
              <p className="text-gray-600 dark:text-gray-300">
                클라우드 보안 취약점 스캔 및 보안 정책 관리
              </p>
            </div>

            <button
              onClick={runSecurityScan}
              className="px-6 py-3 bg-red-500 hover:bg-red-600 text-white rounded-lg font-semibold transition-colors flex items-center gap-2"
            >
              <Shield className="w-5 h-5" />
              Security Scan
            </button>
          </div>

          {/* Security Score */}
          <div className="grid grid-cols-4 gap-4 mt-6">
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-4 rounded-lg">
              <div className="text-sm opacity-90 mb-1">Security Score</div>
              <div className="text-4xl font-bold">{securityScore}%</div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Passing</div>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">{totalPass}</div>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Failing</div>
              <div className="text-3xl font-bold text-red-600 dark:text-red-400">{totalFail}</div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Warnings</div>
              <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400">{totalWarning}</div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mt-6">
            <button
              onClick={() => setSelectedTab('checks')}
              className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                selectedTab === 'checks'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              Security Checks
            </button>
            <button
              onClick={() => setSelectedTab('iam')}
              className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                selectedTab === 'iam'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              IAM Policies
            </button>
            <button
              onClick={() => setSelectedTab('compliance')}
              className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                selectedTab === 'compliance'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              Compliance
            </button>
          </div>
        </div>

        {/* Security Checks Tab */}
        {selectedTab === 'checks' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              {Object.entries(categorySummary).map(([category, stats]) => (
                <div key={category} className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">{category}</h3>
                    <div className="flex gap-2 text-sm">
                      <span className="px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded">
                        {stats.pass} pass
                      </span>
                      {stats.fail > 0 && (
                        <span className="px-2 py-1 bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 rounded">
                          {stats.fail} fail
                        </span>
                      )}
                      {stats.warning > 0 && (
                        <span className="px-2 py-1 bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400 rounded">
                          {stats.warning} warning
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="space-y-3">
                    {securityChecks.filter(c => c.category === category).map((check) => (
                      <div
                        key={check.id}
                        className={`p-4 rounded-lg border-2 ${
                          check.status === 'pass' ? 'border-green-200 bg-green-50 dark:bg-green-900/20 dark:border-green-800' :
                          check.status === 'fail' ? 'border-red-200 bg-red-50 dark:bg-red-900/20 dark:border-red-800' :
                          'border-yellow-200 bg-yellow-50 dark:bg-yellow-900/20 dark:border-yellow-800'
                        }`}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-start gap-3">
                            {check.status === 'pass' && <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />}
                            {check.status === 'fail' && <XCircle className="w-5 h-5 text-red-600 mt-0.5" />}
                            {check.status === 'warning' && <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />}
                            <div>
                              <div className="font-semibold text-gray-900 dark:text-gray-100">{check.name}</div>
                              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">{check.description}</div>
                            </div>
                          </div>

                          <span className={`px-2 py-1 text-xs font-semibold rounded ${
                            check.severity === 'critical' ? 'bg-red-200 text-red-800 dark:bg-red-900/50 dark:text-red-300' :
                            check.severity === 'high' ? 'bg-orange-200 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300' :
                            check.severity === 'medium' ? 'bg-yellow-200 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300' :
                            'bg-blue-200 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300'
                          }`}>
                            {check.severity}
                          </span>
                        </div>

                        {check.status !== 'pass' && (
                          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                            <div className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                              <strong>Remediation:</strong> {check.remediation}
                            </div>
                            <button
                              onClick={() => fixIssue(check.id)}
                              className="px-3 py-1 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors"
                            >
                              Fix Issue
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Quick Stats */}
            <div className="lg:col-span-1">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sticky top-6">
                <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">보안 체크리스트</h3>

                <div className="space-y-3">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Lock className="w-4 h-4 text-blue-500" />
                      <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">Encryption</span>
                    </div>
                    <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1 ml-6">
                      <li>• Data at rest encryption</li>
                      <li>• Data in transit (TLS/SSL)</li>
                      <li>• Key management (KMS)</li>
                    </ul>
                  </div>

                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Key className="w-4 h-4 text-purple-500" />
                      <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">Access Control</span>
                    </div>
                    <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1 ml-6">
                      <li>• Least privilege principle</li>
                      <li>• MFA enabled</li>
                      <li>• Regular access reviews</li>
                    </ul>
                  </div>

                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Eye className="w-4 h-4 text-green-500" />
                      <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">Monitoring</span>
                    </div>
                    <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1 ml-6">
                      <li>• CloudTrail logging</li>
                      <li>• VPC Flow Logs</li>
                      <li>• GuardDuty enabled</li>
                    </ul>
                  </div>

                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="w-4 h-4 text-red-500" />
                      <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">Network Security</span>
                    </div>
                    <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1 ml-6">
                      <li>• Security groups configured</li>
                      <li>• NACLs in place</li>
                      <li>• WAF protection</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* IAM Policies Tab */}
        {selectedTab === 'iam' && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">IAM Policies</h3>

            <div className="space-y-4">
              {iamPolicies.map((policy) => (
                <div key={policy.id} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <div className="font-semibold text-gray-900 dark:text-gray-100">{policy.principal}</div>
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      policy.effect === 'Allow' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    }`}>
                      {policy.effect}
                    </span>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="text-gray-500 mb-1">Actions:</div>
                      <div className="font-mono bg-gray-50 dark:bg-gray-700 p-2 rounded text-xs">
                        {policy.actions.join(', ')}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-500 mb-1">Resources:</div>
                      <div className="font-mono bg-gray-50 dark:bg-gray-700 p-2 rounded text-xs break-all">
                        {policy.resources.join(', ')}
                      </div>
                    </div>
                  </div>

                  {policy.actions.includes('*') && policy.resources.includes('*') && (
                    <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-red-600" />
                      <span className="text-sm text-red-700 dark:text-red-400">
                        ⚠️ Overly permissive policy - violates least privilege principle
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Compliance Tab */}
        {selectedTab === 'compliance' && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">Compliance Frameworks</h3>

            <div className="space-y-6">
              {complianceFrameworks.map((framework) => (
                <div key={framework.name}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-gray-900 dark:text-gray-100">{framework.name}</span>
                    <span className="text-sm font-bold text-gray-900 dark:text-gray-100">{framework.compliance}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                    <div
                      className={`${framework.color} h-3 rounded-full transition-all`}
                      style={{ width: `${framework.compliance}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
