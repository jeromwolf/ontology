'use client';

import React, { useState } from 'react';
import { Server, Database, Network, Shield, Zap, Globe, HardDrive, Cpu } from 'lucide-react';

interface Component {
  id: string;
  type: string;
  name: string;
  x: number;
  y: number;
  color: string;
  icon: string;
}

interface Connection {
  from: string;
  to: string;
  label: string;
}

export default function CloudArchitect() {
  const [components, setComponents] = useState<Component[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedService, setSelectedService] = useState<string>('');
  const [selectedProvider, setSelectedProvider] = useState<'AWS' | 'Azure' | 'GCP'>('AWS');
  const [draggedComponent, setDraggedComponent] = useState<string | null>(null);

  const serviceTypes = {
    AWS: {
      compute: ['EC2', 'Lambda', 'ECS', 'EKS'],
      storage: ['S3', 'EBS', 'EFS', 'Glacier'],
      database: ['RDS', 'DynamoDB', 'Aurora', 'ElastiCache'],
      network: ['VPC', 'CloudFront', 'Route 53', 'API Gateway'],
      security: ['IAM', 'WAF', 'Shield', 'KMS'],
      analytics: ['Athena', 'EMR', 'Kinesis', 'Redshift']
    },
    Azure: {
      compute: ['Virtual Machines', 'Functions', 'AKS', 'App Service'],
      storage: ['Blob Storage', 'Disk Storage', 'File Storage', 'Archive'],
      database: ['SQL Database', 'Cosmos DB', 'PostgreSQL', 'Cache for Redis'],
      network: ['Virtual Network', 'CDN', 'DNS', 'Application Gateway'],
      security: ['Active Directory', 'Key Vault', 'Security Center', 'DDoS Protection'],
      analytics: ['Synapse', 'HDInsight', 'Stream Analytics', 'Data Factory']
    },
    GCP: {
      compute: ['Compute Engine', 'Cloud Functions', 'GKE', 'App Engine'],
      storage: ['Cloud Storage', 'Persistent Disk', 'Filestore', 'Nearline/Coldline'],
      database: ['Cloud SQL', 'Firestore', 'Bigtable', 'Memorystore'],
      network: ['VPC', 'Cloud CDN', 'Cloud DNS', 'API Gateway'],
      security: ['IAM', 'Cloud Armor', 'KMS', 'Security Command Center'],
      analytics: ['BigQuery', 'Dataproc', 'Dataflow', 'Pub/Sub']
    }
  };

  const categoryColors = {
    compute: '#3b82f6',
    storage: '#10b981',
    database: '#8b5cf6',
    network: '#f59e0b',
    security: '#ef4444',
    analytics: '#06b6d4'
  };

  const categoryIcons: Record<string, any> = {
    compute: Cpu,
    storage: HardDrive,
    database: Database,
    network: Globe,
    security: Shield,
    analytics: Zap
  };

  const addComponent = (type: string, category: string) => {
    const newComponent: Component = {
      id: `${type}-${Date.now()}`,
      type,
      name: type,
      x: 50 + Math.random() * 300,
      y: 50 + Math.random() * 200,
      color: categoryColors[category as keyof typeof categoryColors],
      icon: category
    };
    setComponents([...components, newComponent]);
  };

  const removeComponent = (id: string) => {
    setComponents(components.filter(c => c.id !== id));
    setConnections(connections.filter(conn => conn.from !== id && conn.to !== id));
  };

  const addConnection = (fromId: string, toId: string) => {
    const newConnection: Connection = {
      from: fromId,
      to: toId,
      label: 'data flow'
    };
    setConnections([...connections, newConnection]);
  };

  const clearArchitecture = () => {
    setComponents([]);
    setConnections([]);
  };

  const exportArchitecture = () => {
    const architecture = {
      provider: selectedProvider,
      components,
      connections,
      timestamp: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(architecture, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cloud-architecture-${selectedProvider}-${Date.now()}.json`;
    a.click();
  };

  const loadTemplate = (template: string) => {
    clearArchitecture();

    if (template === 'three-tier') {
      const newComponents = [
        { id: 'lb-1', type: 'Load Balancer', name: 'ALB', x: 200, y: 50, color: categoryColors.network, icon: 'network' },
        { id: 'web-1', type: 'Web Server', name: 'EC2', x: 100, y: 150, color: categoryColors.compute, icon: 'compute' },
        { id: 'web-2', type: 'Web Server', name: 'EC2', x: 200, y: 150, color: categoryColors.compute, icon: 'compute' },
        { id: 'web-3', type: 'Web Server', name: 'EC2', x: 300, y: 150, color: categoryColors.compute, icon: 'compute' },
        { id: 'app-1', type: 'App Server', name: 'Lambda', x: 150, y: 250, color: categoryColors.compute, icon: 'compute' },
        { id: 'app-2', type: 'App Server', name: 'Lambda', x: 250, y: 250, color: categoryColors.compute, icon: 'compute' },
        { id: 'db-1', type: 'Database', name: 'RDS', x: 200, y: 350, color: categoryColors.database, icon: 'database' }
      ];

      const newConnections = [
        { from: 'lb-1', to: 'web-1', label: 'HTTP' },
        { from: 'lb-1', to: 'web-2', label: 'HTTP' },
        { from: 'lb-1', to: 'web-3', label: 'HTTP' },
        { from: 'web-1', to: 'app-1', label: 'API' },
        { from: 'web-2', to: 'app-1', label: 'API' },
        { from: 'web-3', to: 'app-2', label: 'API' },
        { from: 'app-1', to: 'db-1', label: 'SQL' },
        { from: 'app-2', to: 'db-1', label: 'SQL' }
      ];

      setComponents(newComponents);
      setConnections(newConnections);
    } else if (template === 'microservices') {
      const newComponents = [
        { id: 'api-gw', type: 'API Gateway', name: 'API Gateway', x: 200, y: 50, color: categoryColors.network, icon: 'network' },
        { id: 'auth-svc', type: 'Auth Service', name: 'Lambda', x: 100, y: 150, color: categoryColors.compute, icon: 'compute' },
        { id: 'user-svc', type: 'User Service', name: 'ECS', x: 200, y: 150, color: categoryColors.compute, icon: 'compute' },
        { id: 'order-svc', type: 'Order Service', name: 'EKS', x: 300, y: 150, color: categoryColors.compute, icon: 'compute' },
        { id: 'cache', type: 'Cache', name: 'ElastiCache', x: 150, y: 250, color: categoryColors.database, icon: 'database' },
        { id: 'queue', type: 'Queue', name: 'SQS', x: 250, y: 250, color: categoryColors.analytics, icon: 'analytics' },
        { id: 'db-auth', type: 'Auth DB', name: 'DynamoDB', x: 100, y: 350, color: categoryColors.database, icon: 'database' },
        { id: 'db-user', type: 'User DB', name: 'RDS', x: 200, y: 350, color: categoryColors.database, icon: 'database' },
        { id: 'db-order', type: 'Order DB', name: 'Aurora', x: 300, y: 350, color: categoryColors.database, icon: 'database' }
      ];

      const newConnections = [
        { from: 'api-gw', to: 'auth-svc', label: 'auth' },
        { from: 'api-gw', to: 'user-svc', label: 'user' },
        { from: 'api-gw', to: 'order-svc', label: 'order' },
        { from: 'auth-svc', to: 'db-auth', label: 'data' },
        { from: 'user-svc', to: 'cache', label: 'cache' },
        { from: 'user-svc', to: 'db-user', label: 'data' },
        { from: 'order-svc', to: 'queue', label: 'event' },
        { from: 'order-svc', to: 'db-order', label: 'data' }
      ];

      setComponents(newComponents);
      setConnections(newConnections);
    } else if (template === 'serverless') {
      const newComponents = [
        { id: 'cdn', type: 'CDN', name: 'CloudFront', x: 200, y: 50, color: categoryColors.network, icon: 'network' },
        { id: 's3-web', type: 'Static Hosting', name: 'S3', x: 200, y: 150, color: categoryColors.storage, icon: 'storage' },
        { id: 'api-gw', type: 'API Gateway', name: 'API Gateway', x: 200, y: 250, color: categoryColors.network, icon: 'network' },
        { id: 'lambda-1', type: 'Function', name: 'Lambda', x: 100, y: 350, color: categoryColors.compute, icon: 'compute' },
        { id: 'lambda-2', type: 'Function', name: 'Lambda', x: 200, y: 350, color: categoryColors.compute, icon: 'compute' },
        { id: 'lambda-3', type: 'Function', name: 'Lambda', x: 300, y: 350, color: categoryColors.compute, icon: 'compute' },
        { id: 'db', type: 'Database', name: 'DynamoDB', x: 200, y: 450, color: categoryColors.database, icon: 'database' }
      ];

      const newConnections = [
        { from: 'cdn', to: 's3-web', label: 'static' },
        { from: 's3-web', to: 'api-gw', label: 'API' },
        { from: 'api-gw', to: 'lambda-1', label: 'invoke' },
        { from: 'api-gw', to: 'lambda-2', label: 'invoke' },
        { from: 'api-gw', to: 'lambda-3', label: 'invoke' },
        { from: 'lambda-1', to: 'db', label: 'query' },
        { from: 'lambda-2', to: 'db', label: 'query' },
        { from: 'lambda-3', to: 'db', label: 'query' }
      ];

      setComponents(newComponents);
      setConnections(newConnections);
    }
  };

  const IconComponent = ({ iconName }: { iconName: string }) => {
    const Icon = categoryIcons[iconName] || Server;
    return <Icon className="w-4 h-4" />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
                클라우드 아키텍처 디자이너
              </h1>
              <p className="text-gray-600 dark:text-gray-300">
                드래그 앤 드롭으로 클라우드 아키텍처를 설계하세요
              </p>
            </div>

            <div className="flex items-center gap-3">
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="AWS">AWS</option>
                <option value="Azure">Azure</option>
                <option value="GCP">GCP</option>
              </select>
            </div>
          </div>

          {/* Templates */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => loadTemplate('three-tier')}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
            >
              3-Tier 템플릿
            </button>
            <button
              onClick={() => loadTemplate('microservices')}
              className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
            >
              마이크로서비스 템플릿
            </button>
            <button
              onClick={() => loadTemplate('serverless')}
              className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors"
            >
              서버리스 템플릿
            </button>
            <button
              onClick={clearArchitecture}
              className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
            >
              Clear
            </button>
            <button
              onClick={exportArchitecture}
              className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-lg transition-colors"
            >
              Export JSON
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Service Palette */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 sticky top-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">
                서비스 팔레트
              </h3>

              <div className="space-y-3">
                {Object.entries(serviceTypes[selectedProvider]).map(([category, services]) => (
                  <div key={category}>
                    <div className="flex items-center gap-2 mb-2">
                      <IconComponent iconName={category} />
                      <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 capitalize">
                        {category}
                      </h4>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {services.map((service) => (
                        <button
                          key={service}
                          onClick={() => addComponent(service, category)}
                          className="px-2 py-1.5 text-xs rounded border-2 transition-all hover:scale-105"
                          style={{
                            borderColor: categoryColors[category as keyof typeof categoryColors],
                            color: categoryColors[category as keyof typeof categoryColors]
                          }}
                        >
                          {service}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Canvas */}
          <div className="lg:col-span-3">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="relative w-full h-[600px] border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden bg-gray-50 dark:bg-gray-900">
                {/* Grid Background */}
                <div className="absolute inset-0" style={{
                  backgroundImage: 'radial-gradient(circle, #e5e7eb 1px, transparent 1px)',
                  backgroundSize: '20px 20px'
                }} />

                {/* Connections */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                  {connections.map((conn, idx) => {
                    const fromComp = components.find(c => c.id === conn.from);
                    const toComp = components.find(c => c.id === conn.to);
                    if (!fromComp || !toComp) return null;

                    return (
                      <g key={idx}>
                        <line
                          x1={fromComp.x + 40}
                          y1={fromComp.y + 40}
                          x2={toComp.x + 40}
                          y2={toComp.y + 40}
                          stroke="#9ca3af"
                          strokeWidth="2"
                          strokeDasharray="5,5"
                          markerEnd="url(#arrowhead)"
                        />
                        <text
                          x={(fromComp.x + toComp.x) / 2 + 40}
                          y={(fromComp.y + toComp.y) / 2 + 40}
                          fill="#6b7280"
                          fontSize="10"
                          textAnchor="middle"
                        >
                          {conn.label}
                        </text>
                      </g>
                    );
                  })}
                  <defs>
                    <marker
                      id="arrowhead"
                      markerWidth="10"
                      markerHeight="10"
                      refX="9"
                      refY="3"
                      orient="auto"
                    >
                      <polygon points="0 0, 10 3, 0 6" fill="#9ca3af" />
                    </marker>
                  </defs>
                </svg>

                {/* Components */}
                {components.map((comp) => (
                  <div
                    key={comp.id}
                    className="absolute cursor-move group"
                    style={{ left: comp.x, top: comp.y }}
                    draggable
                    onDragStart={() => setDraggedComponent(comp.id)}
                    onDragEnd={(e) => {
                      const rect = e.currentTarget.parentElement?.getBoundingClientRect();
                      if (rect) {
                        const newX = e.clientX - rect.left - 40;
                        const newY = e.clientY - rect.top - 40;
                        setComponents(components.map(c =>
                          c.id === comp.id ? { ...c, x: Math.max(0, newX), y: Math.max(0, newY) } : c
                        ));
                      }
                      setDraggedComponent(null);
                    }}
                  >
                    <div
                      className="w-20 h-20 rounded-lg shadow-lg flex flex-col items-center justify-center text-white transition-transform group-hover:scale-110"
                      style={{ backgroundColor: comp.color }}
                    >
                      <IconComponent iconName={comp.icon} />
                      <span className="text-xs mt-1 font-semibold text-center px-1">
                        {comp.type}
                      </span>
                    </div>
                    <button
                      onClick={() => removeComponent(comp.id)}
                      className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white rounded-full text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      ×
                    </button>
                  </div>
                ))}

                {/* Empty State */}
                {components.length === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-500">
                    <div className="text-center">
                      <Server className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-lg font-semibold">클라우드 서비스를 추가하세요</p>
                      <p className="text-sm mt-2">왼쪽 팔레트에서 서비스를 선택하거나 템플릿을 로드하세요</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Stats */}
              <div className="mt-4 grid grid-cols-3 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">Total Components</div>
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{components.length}</div>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">Connections</div>
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{connections.length}</div>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400">Provider</div>
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">{selectedProvider}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
