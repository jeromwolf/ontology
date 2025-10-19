'use client';

import React, { useState, useEffect } from 'react';
import { Box, Activity, AlertCircle, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

interface Container {
  id: string;
  name: string;
  image: string;
  status: 'running' | 'stopped' | 'error' | 'starting';
  cpu: number;
  memory: number;
  port: number;
  replicas: number;
}

interface Node {
  id: string;
  name: string;
  status: 'ready' | 'not-ready';
  cpu: number;
  memory: number;
  containers: string[];
}

export default function ContainerOrchestrator() {
  const [containers, setContainers] = useState<Container[]>([
    {
      id: 'c1',
      name: 'web-frontend',
      image: 'nginx:latest',
      status: 'running',
      cpu: 25,
      memory: 512,
      port: 80,
      replicas: 3
    },
    {
      id: 'c2',
      name: 'api-backend',
      image: 'node:18-alpine',
      status: 'running',
      cpu: 50,
      memory: 1024,
      port: 3000,
      replicas: 5
    },
    {
      id: 'c3',
      name: 'database',
      image: 'postgres:15',
      status: 'running',
      cpu: 75,
      memory: 2048,
      port: 5432,
      replicas: 1
    }
  ]);

  const [nodes, setNodes] = useState<Node[]>([
    { id: 'n1', name: 'node-1', status: 'ready', cpu: 40, memory: 60, containers: ['c1', 'c2'] },
    { id: 'n2', name: 'node-2', status: 'ready', cpu: 35, memory: 55, containers: ['c1', 'c3'] },
    { id: 'n3', name: 'node-3', status: 'ready', cpu: 30, memory: 45, containers: ['c2'] }
  ]);

  const [selectedContainer, setSelectedContainer] = useState<string | null>(null);
  const [isAutoScaling, setIsAutoScaling] = useState(true);
  const [newContainerForm, setNewContainerForm] = useState({
    name: '',
    image: '',
    cpu: 50,
    memory: 512,
    port: 8080,
    replicas: 1
  });

  // Auto-scaling simulation
  useEffect(() => {
    if (!isAutoScaling) return;

    const interval = setInterval(() => {
      setContainers(prev => prev.map(container => {
        const cpuUsage = container.cpu + (Math.random() - 0.5) * 20;
        const newCpu = Math.max(0, Math.min(100, cpuUsage));

        // Scale up if CPU > 80%
        if (newCpu > 80 && container.replicas < 10) {
          return { ...container, cpu: newCpu, replicas: container.replicas + 1 };
        }

        // Scale down if CPU < 30%
        if (newCpu < 30 && container.replicas > 1) {
          return { ...container, cpu: newCpu, replicas: container.replicas - 1 };
        }

        return { ...container, cpu: newCpu };
      }));

      // Update node metrics
      setNodes(prev => prev.map(node => ({
        ...node,
        cpu: Math.max(0, Math.min(100, node.cpu + (Math.random() - 0.5) * 15)),
        memory: Math.max(0, Math.min(100, node.memory + (Math.random() - 0.5) * 10))
      })));
    }, 2000);

    return () => clearInterval(interval);
  }, [isAutoScaling]);

  const addContainer = () => {
    if (!newContainerForm.name || !newContainerForm.image) {
      alert('Please fill in container name and image');
      return;
    }

    const newContainer: Container = {
      id: `c${Date.now()}`,
      name: newContainerForm.name,
      image: newContainerForm.image,
      status: 'starting',
      cpu: newContainerForm.cpu,
      memory: newContainerForm.memory,
      port: newContainerForm.port,
      replicas: newContainerForm.replicas
    };

    setContainers([...containers, newContainer]);

    // Simulate container startup
    setTimeout(() => {
      setContainers(prev => prev.map(c =>
        c.id === newContainer.id ? { ...c, status: 'running' } : c
      ));
    }, 2000);

    // Reset form
    setNewContainerForm({
      name: '',
      image: '',
      cpu: 50,
      memory: 512,
      port: 8080,
      replicas: 1
    });
  };

  const removeContainer = (id: string) => {
    setContainers(containers.filter(c => c.id !== id));
  };

  const restartContainer = (id: string) => {
    setContainers(prev => prev.map(c =>
      c.id === id ? { ...c, status: 'starting' as const } : c
    ));

    setTimeout(() => {
      setContainers(prev => prev.map(c =>
        c.id === id ? { ...c, status: 'running' as const } : c
      ));
    }, 1500);
  };

  const scaleContainer = (id: string, delta: number) => {
    setContainers(prev => prev.map(c =>
      c.id === id ? { ...c, replicas: Math.max(1, c.replicas + delta) } : c
    ));
  };

  const totalContainers = containers.reduce((sum, c) => sum + c.replicas, 0);
  const runningContainers = containers.filter(c => c.status === 'running').reduce((sum, c) => sum + c.replicas, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        <SimulatorNav />

        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-2">
                컨테이너 오케스트레이터
              </h1>
              <p className="text-gray-600 dark:text-gray-300">
                Kubernetes 스타일 컨테이너 관리 및 오토스케일링
              </p>
            </div>

            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={isAutoScaling}
                  onChange={(e) => setIsAutoScaling(e.target.checked)}
                  className="w-4 h-4"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">Auto-scaling</span>
              </label>
            </div>
          </div>

          {/* Cluster Stats */}
          <div className="grid grid-cols-4 gap-4 mt-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400">Total Pods</div>
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">{totalContainers}</div>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400">Running</div>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">{runningContainers}</div>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400">Services</div>
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">{containers.length}</div>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400">Nodes</div>
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">{nodes.length}</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Containers List */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">Running Services</h3>

              <div className="space-y-3">
                {containers.map((container) => (
                  <div
                    key={container.id}
                    className={`border-2 rounded-lg p-4 transition-all cursor-pointer ${
                      selectedContainer === container.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedContainer(container.id)}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <Box className="w-5 h-5 text-blue-500" />
                        <div>
                          <div className="font-bold text-gray-900 dark:text-gray-100">{container.name}</div>
                          <div className="text-sm text-gray-500">{container.image}</div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {container.status === 'running' && <CheckCircle className="w-5 h-5 text-green-500" />}
                        {container.status === 'starting' && <RefreshCw className="w-5 h-5 text-yellow-500 animate-spin" />}
                        {container.status === 'stopped' && <AlertCircle className="w-5 h-5 text-gray-500" />}
                        {container.status === 'error' && <XCircle className="w-5 h-5 text-red-500" />}
                        <span className={`text-sm font-semibold ${
                          container.status === 'running' ? 'text-green-600' :
                          container.status === 'starting' ? 'text-yellow-600' :
                          container.status === 'error' ? 'text-red-600' : 'text-gray-600'
                        }`}>
                          {container.status}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-3 mb-3">
                      <div>
                        <div className="text-xs text-gray-500 mb-1">CPU Usage</div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full transition-all ${
                              container.cpu > 80 ? 'bg-red-500' :
                              container.cpu > 60 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${container.cpu}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-500 mt-1">{container.cpu.toFixed(0)}%</div>
                      </div>

                      <div>
                        <div className="text-xs text-gray-500 mb-1">Memory</div>
                        <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                          {container.memory} MB
                        </div>
                      </div>

                      <div>
                        <div className="text-xs text-gray-500 mb-1">Port</div>
                        <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                          :{container.port}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={(e) => { e.stopPropagation(); scaleContainer(container.id, -1); }}
                          className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600 text-sm"
                          disabled={container.replicas <= 1}
                        >
                          −
                        </button>
                        <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                          {container.replicas} replicas
                        </span>
                        <button
                          onClick={(e) => { e.stopPropagation(); scaleContainer(container.id, 1); }}
                          className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600 text-sm"
                          disabled={container.replicas >= 10}
                        >
                          +
                        </button>
                      </div>

                      <div className="flex gap-2">
                        <button
                          onClick={(e) => { e.stopPropagation(); restartContainer(container.id); }}
                          className="px-3 py-1 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors"
                        >
                          Restart
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); removeContainer(container.id); }}
                          className="px-3 py-1 text-sm bg-red-500 hover:bg-red-600 text-white rounded transition-colors"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Nodes */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">Cluster Nodes</h3>

              <div className="grid gap-3">
                {nodes.map((node) => (
                  <div key={node.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Activity className="w-5 h-5 text-green-500" />
                        <span className="font-bold text-gray-900 dark:text-gray-100">{node.name}</span>
                      </div>
                      <span className={`text-sm font-semibold ${
                        node.status === 'ready' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {node.status}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-xs text-gray-500 mb-1">CPU</div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full transition-all"
                            style={{ width: `${node.cpu}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-500 mt-1">{node.cpu.toFixed(0)}%</div>
                      </div>

                      <div>
                        <div className="text-xs text-gray-500 mb-1">Memory</div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-purple-500 h-2 rounded-full transition-all"
                            style={{ width: `${node.memory}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-500 mt-1">{node.memory.toFixed(0)}%</div>
                      </div>
                    </div>

                    <div className="mt-2 text-xs text-gray-500">
                      {node.containers.length} containers running
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Add Container Form */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sticky top-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">Deploy New Service</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Service Name
                  </label>
                  <input
                    type="text"
                    value={newContainerForm.name}
                    onChange={(e) => setNewContainerForm({ ...newContainerForm, name: e.target.value })}
                    placeholder="my-service"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Image
                  </label>
                  <input
                    type="text"
                    value={newContainerForm.image}
                    onChange={(e) => setNewContainerForm({ ...newContainerForm, image: e.target.value })}
                    placeholder="nginx:latest"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    CPU Limit: {newContainerForm.cpu}%
                  </label>
                  <input
                    type="range"
                    value={newContainerForm.cpu}
                    onChange={(e) => setNewContainerForm({ ...newContainerForm, cpu: Number(e.target.value) })}
                    min="10"
                    max="100"
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Memory (MB)
                  </label>
                  <select
                    value={newContainerForm.memory}
                    onChange={(e) => setNewContainerForm({ ...newContainerForm, memory: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    <option value="128">128</option>
                    <option value="256">256</option>
                    <option value="512">512</option>
                    <option value="1024">1024</option>
                    <option value="2048">2048</option>
                    <option value="4096">4096</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Port
                  </label>
                  <input
                    type="number"
                    value={newContainerForm.port}
                    onChange={(e) => setNewContainerForm({ ...newContainerForm, port: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    min="1"
                    max="65535"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Replicas
                  </label>
                  <input
                    type="number"
                    value={newContainerForm.replicas}
                    onChange={(e) => setNewContainerForm({ ...newContainerForm, replicas: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    min="1"
                    max="10"
                  />
                </div>

                <button
                  onClick={addContainer}
                  className="w-full px-4 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold transition-colors"
                >
                  Deploy Service
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
