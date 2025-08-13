'use client';

import React from 'react';

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          MCP란 무엇인가?
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            <strong>Model Context Protocol (MCP)</strong>는 Anthropic이 개발한 오픈 프로토콜로,
            AI 모델과 외부 도구/데이터 소스를 표준화된 방식으로 연결합니다.
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h3 className="font-semibold mb-2">MCP의 핵심 컴포넌트</h3>
            <ul className="space-y-2">
              <li>📦 <strong>Resources</strong>: 파일, 데이터베이스, API 등 데이터 소스</li>
              <li>🔧 <strong>Tools</strong>: Agent가 사용할 수 있는 함수와 명령</li>
              <li>💬 <strong>Prompts</strong>: 재사용 가능한 프롬프트 템플릿</li>
              <li>🔄 <strong>Sampling</strong>: LLM과의 상호작용 관리</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          MCP Server 구현
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            MCP Server는 도구와 리소스를 제공하는 백엔드입니다. TypeScript로 간단한 MCP 서버를 구현해봅시다:
          </p>
          
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`import { Server } from '@modelcontextprotocol/sdk/server';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio';

const server = new Server({
  name: 'my-mcp-server',
  version: '1.0.0',
});

// Tool 등록
server.setRequestHandler('tools/list', async () => ({
  tools: [{
    name: 'calculate',
    description: '수학 계산을 수행합니다',
    inputSchema: {
      type: 'object',
      properties: {
        expression: { type: 'string' }
      }
    }
  }]
}));

// Tool 실행
server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'calculate') {
    const result = eval(request.params.arguments.expression);
    return { content: [{ type: 'text', text: result.toString() }] };
  }
});

// 서버 시작
const transport = new StdioServerTransport();
await server.connect(transport);`}
          </pre>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          🎮 MCP Server 시뮬레이터
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          MCP 서버와 클라이언트 간의 통신을 실시간으로 시각화합니다.
        </p>
        <div className="text-center p-8 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            시뮬레이터를 보려면 전체 시뮬레이터 페이지를 방문하세요.
          </p>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          MCP 통신 프로토콜
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            MCP는 JSON-RPC 2.0 기반의 양방향 통신을 사용합니다:
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Request</h4>
              <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded overflow-x-auto">
{`{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "MCP protocol"
    }
  },
  "id": 1
}`}
              </pre>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Response</h4>
              <pre className="text-xs bg-gray-900 text-gray-100 p-2 rounded overflow-x-auto">
{`{
  "jsonrpc": "2.0",
  "result": {
    "content": [{
      "type": "text",
      "text": "검색 결과..."
    }]
  },
  "id": 1
}`}
              </pre>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}