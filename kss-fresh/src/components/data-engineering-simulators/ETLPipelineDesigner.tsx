'use client';

import { useState } from 'react';
import {
  Database, ArrowRight, Filter, RefreshCw,
  Download, Settings, Play, Check, X,
  GitBranch, Zap, Cloud, HardDrive
} from 'lucide-react';

interface PipelineNode {
  id: string;
  type: 'source' | 'transform' | 'destination';
  name: string;
  icon: string;
  config: Record<string, any>;
}

interface PipelineConnection {
  from: string;
  to: string;
}

export default function ETLPipelineDesigner() {
  const [nodes, setNodes] = useState<PipelineNode[]>([
    {
      id: 'source-1',
      type: 'source',
      name: 'PostgreSQL',
      icon: '🐘',
      config: { host: 'localhost', db: 'production', table: 'orders' },
    },
  ]);
  const [connections, setConnections] = useState<PipelineConnection[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [pipelineMode, setPipelineMode] = useState<'etl' | 'elt'>('etl');
  const [isRunning, setIsRunning] = useState(false);
  const [executionLog, setExecutionLog] = useState<string[]>([]);

  const availableNodes = {
    sources: [
      { id: 'postgres', name: 'PostgreSQL', icon: '🐘', type: 'source' as const },
      { id: 'mysql', name: 'MySQL', icon: '🐬', type: 'source' as const },
      { id: 's3', name: 'Amazon S3', icon: '📦', type: 'source' as const },
      { id: 'api', name: 'REST API', icon: '🌐', type: 'source' as const },
      { id: 'csv', name: 'CSV Files', icon: '📄', type: 'source' as const },
    ],
    transforms: [
      { id: 'filter', name: 'Filter Rows', icon: '🔍', type: 'transform' as const },
      { id: 'join', name: 'Join Tables', icon: '🔗', type: 'transform' as const },
      { id: 'aggregate', name: 'Aggregate', icon: '📊', type: 'transform' as const },
      { id: 'dedupe', name: 'Deduplicate', icon: '🗑️', type: 'transform' as const },
      { id: 'enrich', name: 'Enrich Data', icon: '✨', type: 'transform' as const },
      { id: 'validate', name: 'Data Quality', icon: '✅', type: 'transform' as const },
    ],
    destinations: [
      { id: 'snowflake', name: 'Snowflake', icon: '❄️', type: 'destination' as const },
      { id: 'bigquery', name: 'BigQuery', icon: '🔷', type: 'destination' as const },
      { id: 'redshift', name: 'Redshift', icon: '🔴', type: 'destination' as const },
      { id: 'delta', name: 'Delta Lake', icon: '🔺', type: 'destination' as const },
      { id: 'parquet', name: 'Parquet', icon: '📋', type: 'destination' as const },
    ],
  };

  const addNode = (template: typeof availableNodes.sources[0]) => {
    const newNode: PipelineNode = {
      id: `${template.type}-${Date.now()}`,
      type: template.type,
      name: template.name,
      icon: template.icon,
      config: {},
    };
    setNodes([...nodes, newNode]);
  };

  const connectNodes = (fromId: string, toId: string) => {
    const newConnection = { from: fromId, to: toId };
    setConnections([...connections, newConnection]);
  };

  const runPipeline = () => {
    setIsRunning(true);
    setExecutionLog([]);

    const logs: string[] = [];
    let delay = 0;

    nodes.forEach((node, idx) => {
      setTimeout(() => {
        if (node.type === 'source') {
          logs.push(`[${new Date().toLocaleTimeString()}] 📥 Extracting from ${node.name}...`);
          logs.push(`[${new Date().toLocaleTimeString()}] ✅ Extracted 10,000 rows from ${node.name}`);
        } else if (node.type === 'transform') {
          logs.push(`[${new Date().toLocaleTimeString()}] ⚙️ Applying transformation: ${node.name}...`);
          logs.push(`[${new Date().toLocaleTimeString()}] ✅ Transformation complete (8,500 rows)`);
        } else if (node.type === 'destination') {
          logs.push(`[${new Date().toLocaleTimeString()}] 📤 Loading to ${node.name}...`);
          logs.push(`[${new Date().toLocaleTimeString()}] ✅ Successfully loaded 8,500 rows to ${node.name}`);
        }
        setExecutionLog([...logs]);

        if (idx === nodes.length - 1) {
          setTimeout(() => {
            logs.push(`[${new Date().toLocaleTimeString()}] 🎉 Pipeline execution completed!`);
            setExecutionLog([...logs]);
            setIsRunning(false);
          }, 500);
        }
      }, delay);
      delay += 1000;
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <GitBranch size={32} />
          <h2 className="text-2xl font-bold">ETL/ELT 파이프라인 디자이너</h2>
        </div>
        <p className="text-purple-100">
          드래그 앤 드롭으로 데이터 파이프라인을 시각적으로 설계하고 실행하세요
        </p>
      </div>

      {/* Mode Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">파이프라인 모드 선택</h3>
        <div className="flex gap-4">
          <button
            onClick={() => setPipelineMode('etl')}
            className={`flex-1 p-4 rounded-lg border-2 transition-all ${
              pipelineMode === 'etl'
                ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                : 'border-gray-200 dark:border-gray-700'
            }`}
          >
            <div className="text-2xl mb-2">⚙️</div>
            <h4 className="font-bold mb-1">ETL (Extract-Transform-Load)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              추출 → 변환 → 적재 (전통적 방식, 스테이징 필요)
            </p>
          </button>
          <button
            onClick={() => setPipelineMode('elt')}
            className={`flex-1 p-4 rounded-lg border-2 transition-all ${
              pipelineMode === 'elt'
                ? 'border-pink-500 bg-pink-50 dark:bg-pink-900/20'
                : 'border-gray-200 dark:border-gray-700'
            }`}
          >
            <div className="text-2xl mb-2">⚡</div>
            <h4 className="font-bold mb-1">ELT (Extract-Load-Transform)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              추출 → 적재 → 변환 (현대적 방식, 클라우드 DW 활용)
            </p>
          </button>
        </div>
      </div>

      {/* Node Palette */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">📦 컴포넌트 팔레트</h3>

        <div className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2 text-sm text-gray-600 dark:text-gray-400">Sources (데이터 소스)</h4>
            <div className="flex flex-wrap gap-2">
              {availableNodes.sources.map((node) => (
                <button
                  key={node.id}
                  onClick={() => addNode(node)}
                  className="px-3 py-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors text-sm"
                >
                  {node.icon} {node.name}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2 text-sm text-gray-600 dark:text-gray-400">Transforms (데이터 변환)</h4>
            <div className="flex flex-wrap gap-2">
              {availableNodes.transforms.map((node) => (
                <button
                  key={node.id}
                  onClick={() => addNode(node)}
                  className="px-3 py-2 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/40 transition-colors text-sm"
                >
                  {node.icon} {node.name}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2 text-sm text-gray-600 dark:text-gray-400">Destinations (데이터 적재)</h4>
            <div className="flex flex-wrap gap-2">
              {availableNodes.destinations.map((node) => (
                <button
                  key={node.id}
                  onClick={() => addNode(node)}
                  className="px-3 py-2 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/40 transition-colors text-sm"
                >
                  {node.icon} {node.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Canvas */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold">🎨 파이프라인 캔버스</h3>
          <button
            onClick={runPipeline}
            disabled={isRunning || nodes.length === 0}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-all ${
              isRunning || nodes.length === 0
                ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            <Play size={18} />
            {isRunning ? '실행 중...' : '파이프라인 실행'}
          </button>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 min-h-[300px] border-2 border-dashed border-gray-300 dark:border-gray-700">
          {nodes.length === 0 ? (
            <div className="flex items-center justify-center h-[200px] text-gray-400">
              위의 컴포넌트를 클릭하여 파이프라인을 구성하세요
            </div>
          ) : (
            <div className="flex items-center gap-4 flex-wrap">
              {nodes.map((node, idx) => (
                <div key={node.id} className="flex items-center gap-4">
                  <div
                    onClick={() => setSelectedNode(node.id)}
                    className={`p-4 rounded-lg cursor-pointer transition-all ${
                      node.type === 'source'
                        ? 'bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-300'
                        : node.type === 'transform'
                        ? 'bg-green-100 dark:bg-green-900/30 border-2 border-green-300'
                        : 'bg-purple-100 dark:bg-purple-900/30 border-2 border-purple-300'
                    } ${selectedNode === node.id ? 'ring-4 ring-yellow-400' : ''}`}
                  >
                    <div className="text-2xl mb-1">{node.icon}</div>
                    <div className="text-sm font-semibold">{node.name}</div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setNodes(nodes.filter(n => n.id !== node.id));
                      }}
                      className="mt-2 text-xs text-red-500 hover:text-red-700"
                    >
                      <X size={14} className="inline" /> 삭제
                    </button>
                  </div>
                  {idx < nodes.length - 1 && (
                    <ArrowRight className="text-gray-400" size={24} />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Execution Log */}
      {executionLog.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4">📜 실행 로그</h3>
          <div className="bg-gray-900 text-green-400 rounded-lg p-4 font-mono text-sm h-64 overflow-y-auto">
            {executionLog.map((log, idx) => (
              <div key={idx} className="mb-1">{log}</div>
            ))}
          </div>
        </div>
      )}

      {/* Code Generation */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">🔧 생성된 Airflow DAG 코드</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'custom_etl_pipeline',
    default_args=default_args,
    schedule_interval='0 2 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
${nodes.map((node, idx) => {
  if (node.type === 'source') {
    return `
    # Extract from ${node.name}
    extract_${idx} = PythonOperator(
        task_id='extract_from_${node.name.toLowerCase().replace(/\\s+/g, '_')}',
        python_callable=extract_from_postgres,
    )`;
  } else if (node.type === 'transform') {
    return `
    # Transform: ${node.name}
    transform_${idx} = PythonOperator(
        task_id='${node.name.toLowerCase().replace(/\\s+/g, '_')}',
        python_callable=apply_transformation,
    )`;
  } else {
    return `
    # Load to ${node.name}
    load_${idx} = PythonOperator(
        task_id='load_to_${node.name.toLowerCase().replace(/\\s+/g, '_')}',
        python_callable=load_to_warehouse,
    )`;
  }
}).join('')}

    # Define dependencies
    ${nodes.map((_, idx) => idx > 0 ? `extract_0 >> transform_${idx}` : '').filter(Boolean).join(' >> ')}
`}
        </pre>
      </div>
    </div>
  );
}
