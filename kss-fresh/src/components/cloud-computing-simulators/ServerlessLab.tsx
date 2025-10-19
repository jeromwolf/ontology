'use client';

import React, { useState, useEffect } from 'react';
import { Zap, Play, Pause, RotateCcw, Activity, Clock, TrendingUp } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

interface LogEntry {
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR' | 'SUCCESS';
  message: string;
}

interface Metrics {
  invocations: number;
  avgDuration: number;
  errors: number;
  coldStarts: number;
  totalCost: number;
}

export default function ServerlessLab() {
  const [isRunning, setIsRunning] = useState(false);
  const [code, setCode] = useState(`exports.handler = async (event) => {
  console.log('Event received:', event);

  // Your serverless logic here
  const result = {
    statusCode: 200,
    body: JSON.stringify({
      message: 'Hello from Lambda!',
      timestamp: new Date().toISOString()
    })
  };

  return result;
};`);

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<Metrics>({
    invocations: 0,
    avgDuration: 0,
    errors: 0,
    coldStarts: 0,
    totalCost: 0
  });

  const [config, setConfig] = useState({
    memory: 512,
    timeout: 30,
    concurrency: 10,
    runtime: 'nodejs18.x'
  });

  const addLog = (level: LogEntry['level'], message: string) => {
    const newLog: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message
    };
    setLogs(prev => [newLog, ...prev].slice(0, 100));
  };

  const runFunction = () => {
    const startTime = Date.now();
    const isColdStart = Math.random() < 0.1; // 10% chance of cold start

    addLog('INFO', `Lambda function invoked (${isColdStart ? 'COLD START' : 'warm'})`);

    setTimeout(() => {
      try {
        // Simulate function execution
        const duration = Math.random() * 500 + (isColdStart ? 1000 : 100);
        const success = Math.random() > 0.05; // 95% success rate

        if (success) {
          addLog('SUCCESS', `Execution completed in ${duration.toFixed(0)}ms`);
          addLog('INFO', `Response: {"statusCode":200,"body":"{\\"message\\":\\"Hello from Lambda!\\"}"}"`);
        } else {
          addLog('ERROR', 'Function execution failed: Timeout exceeded');
        }

        // Update metrics
        setMetrics(prev => ({
          invocations: prev.invocations + 1,
          avgDuration: (prev.avgDuration * prev.invocations + duration) / (prev.invocations + 1),
          errors: success ? prev.errors : prev.errors + 1,
          coldStarts: isColdStart ? prev.coldStarts + 1 : prev.coldStarts,
          totalCost: prev.totalCost + calculateCost(duration, config.memory)
        }));

      } catch (error) {
        addLog('ERROR', `Execution error: ${error}`);
      }
    }, 500);
  };

  const calculateCost = (duration: number, memory: number) => {
    // AWS Lambda pricing: $0.0000166667 per GB-second
    const gbSeconds = (memory / 1024) * (duration / 1000);
    const requestCost = 0.0000002; // $0.20 per 1M requests
    return gbSeconds * 0.0000166667 + requestCost;
  };

  const resetMetrics = () => {
    setMetrics({
      invocations: 0,
      avgDuration: 0,
      errors: 0,
      coldStarts: 0,
      totalCost: 0
    });
    setLogs([]);
    addLog('INFO', 'Metrics reset');
  };

  const loadExample = (example: string) => {
    if (example === 'api-gateway') {
      setCode(`exports.handler = async (event) => {
  const { httpMethod, path, body } = event;

  console.log(\`\${httpMethod} \${path}\`);

  if (httpMethod === 'GET') {
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: 'GET request successful' })
    };
  }

  if (httpMethod === 'POST') {
    const data = JSON.parse(body);
    return {
      statusCode: 201,
      body: JSON.stringify({ created: true, data })
    };
  }

  return { statusCode: 405, body: 'Method Not Allowed' };
};`);
    } else if (example === 's3-trigger') {
      setCode(`exports.handler = async (event) => {
  const s3Event = event.Records[0].s3;
  const bucket = s3Event.bucket.name;
  const key = decodeURIComponent(s3Event.object.key);

  console.log(\`Processing file: \${key} from bucket: \${bucket}\`);

  // Process the file (e.g., resize image, analyze data)
  const result = await processFile(bucket, key);

  return {
    statusCode: 200,
    body: JSON.stringify({ processed: true, result })
  };
};

async function processFile(bucket, key) {
  // Simulated file processing
  return { bucket, key, timestamp: new Date().toISOString() };
}`);
    } else if (example === 'scheduled') {
      setCode(`exports.handler = async (event) => {
  console.log('Scheduled task running at:', new Date().toISOString());

  // Daily cleanup task
  const cleanupResult = await performCleanup();

  // Send notification
  await sendNotification({
    subject: 'Daily Cleanup Complete',
    message: \`Cleaned up \${cleanupResult.itemsRemoved} items\`
  });

  return {
    statusCode: 200,
    body: JSON.stringify(cleanupResult)
  };
};

async function performCleanup() {
  // Simulated cleanup logic
  return { itemsRemoved: Math.floor(Math.random() * 100) };
}

async function sendNotification(params) {
  console.log('Notification sent:', params);
}`);
    } else if (example === 'dynamodb-stream') {
      setCode(`exports.handler = async (event) => {
  for (const record of event.Records) {
    const eventName = record.eventName;
    const newData = record.dynamodb.NewImage;
    const oldData = record.dynamodb.OldImage;

    console.log(\`DynamoDB \${eventName}\`, { newData, oldData });

    if (eventName === 'INSERT') {
      await handleInsert(newData);
    } else if (eventName === 'MODIFY') {
      await handleUpdate(oldData, newData);
    } else if (eventName === 'REMOVE') {
      await handleDelete(oldData);
    }
  }

  return { statusCode: 200, body: 'Processed successfully' };
};

async function handleInsert(data) {
  console.log('New record created:', data);
}

async function handleUpdate(oldData, newData) {
  console.log('Record updated:', { from: oldData, to: newData });
}

async function handleDelete(data) {
  console.log('Record deleted:', data);
}`);
    }
    addLog('INFO', `Example loaded: ${example}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        <SimulatorNav />

        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-yellow-600 to-orange-600 bg-clip-text text-transparent mb-2">
                서버리스 실습 환경
              </h1>
              <p className="text-gray-600 dark:text-gray-300">
                AWS Lambda 함수를 작성하고 테스트하세요
              </p>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={runFunction}
                className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <Play className="w-4 h-4" />
                Invoke
              </button>
              <button
                onClick={resetMetrics}
                className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                Reset
              </button>
            </div>
          </div>

          {/* Example Templates */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => loadExample('api-gateway')}
              className="px-3 py-1.5 text-sm bg-blue-100 hover:bg-blue-200 dark:bg-blue-900/30 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded-lg transition-colors"
            >
              API Gateway
            </button>
            <button
              onClick={() => loadExample('s3-trigger')}
              className="px-3 py-1.5 text-sm bg-green-100 hover:bg-green-200 dark:bg-green-900/30 dark:hover:bg-green-900/50 text-green-700 dark:text-green-300 rounded-lg transition-colors"
            >
              S3 Trigger
            </button>
            <button
              onClick={() => loadExample('scheduled')}
              className="px-3 py-1.5 text-sm bg-purple-100 hover:bg-purple-200 dark:bg-purple-900/30 dark:hover:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded-lg transition-colors"
            >
              Scheduled Task
            </button>
            <button
              onClick={() => loadExample('dynamodb-stream')}
              className="px-3 py-1.5 text-sm bg-orange-100 hover:bg-orange-200 dark:bg-orange-900/30 dark:hover:bg-orange-900/50 text-orange-700 dark:text-orange-300 rounded-lg transition-colors"
            >
              DynamoDB Stream
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Code Editor */}
          <div className="lg:col-span-2 space-y-6">
            {/* Configuration */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">Function Configuration</h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Memory (MB)
                  </label>
                  <select
                    value={config.memory}
                    onChange={(e) => setConfig({ ...config, memory: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                  >
                    <option value="128">128</option>
                    <option value="256">256</option>
                    <option value="512">512</option>
                    <option value="1024">1024</option>
                    <option value="2048">2048</option>
                    <option value="3008">3008</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Timeout (s)
                  </label>
                  <input
                    type="number"
                    value={config.timeout}
                    onChange={(e) => setConfig({ ...config, timeout: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                    min="1"
                    max="900"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Concurrency
                  </label>
                  <input
                    type="number"
                    value={config.concurrency}
                    onChange={(e) => setConfig({ ...config, concurrency: Number(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                    min="1"
                    max="1000"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Runtime
                  </label>
                  <select
                    value={config.runtime}
                    onChange={(e) => setConfig({ ...config, runtime: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                  >
                    <option value="nodejs18.x">Node.js 18</option>
                    <option value="python3.11">Python 3.11</option>
                    <option value="go1.x">Go 1.x</option>
                    <option value="java17">Java 17</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Code Editor */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">Function Code</h3>
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="w-full h-96 px-4 py-3 font-mono text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                spellCheck="false"
              />
            </div>

            {/* Logs */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">Execution Logs</h3>
              <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
                {logs.length === 0 ? (
                  <div className="text-gray-500 text-center py-8">No logs yet. Click "Invoke" to run the function.</div>
                ) : (
                  logs.map((log, idx) => (
                    <div key={idx} className="mb-1">
                      <span className="text-gray-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
                      {' '}
                      <span className={
                        log.level === 'ERROR' ? 'text-red-400' :
                        log.level === 'WARN' ? 'text-yellow-400' :
                        log.level === 'SUCCESS' ? 'text-green-400' :
                        'text-blue-400'
                      }>
                        [{log.level}]
                      </span>
                      {' '}
                      <span className="text-gray-300">{log.message}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Metrics Panel */}
          <div className="lg:col-span-1 space-y-6">
            {/* Metrics */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sticky top-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-500" />
                Metrics
              </h3>

              <div className="space-y-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Total Invocations</div>
                  <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                    {metrics.invocations.toLocaleString()}
                  </div>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Avg Duration</div>
                  <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                    {metrics.avgDuration.toFixed(0)}ms
                  </div>
                </div>

                <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Errors</div>
                  <div className="text-3xl font-bold text-red-600 dark:text-red-400">
                    {metrics.errors}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {metrics.invocations > 0 ? ((metrics.errors / metrics.invocations) * 100).toFixed(1) : 0}% error rate
                  </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Cold Starts</div>
                  <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                    {metrics.coldStarts}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {metrics.invocations > 0 ? ((metrics.coldStarts / metrics.invocations) * 100).toFixed(1) : 0}% cold start rate
                  </div>
                </div>

                <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Total Cost</div>
                  <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                    ${metrics.totalCost.toFixed(6)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    ${(metrics.totalCost * 720).toFixed(2)}/month estimate
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
