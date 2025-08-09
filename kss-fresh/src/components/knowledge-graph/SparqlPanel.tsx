'use client';

import React, { useState, useCallback } from 'react';
import { Play, Copy, Download, Save } from 'lucide-react';
import { Triple } from './types';

interface SparqlPanelProps {
  query: string;
  onQueryChange: (query: string) => void;
  onExecute: (query: string) => void;
  triples: Triple[];
}

export const SparqlPanel: React.FC<SparqlPanelProps> = ({
  query,
  onQueryChange,
  onExecute,
  triples
}) => {
  const [results, setResults] = useState<any[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExecute = useCallback(() => {
    setIsExecuting(true);
    setError(null);
    
    try {
      // Simple SPARQL execution simulation
      // In a real implementation, this would use a proper SPARQL engine
      const mockResults = triples.slice(0, 5).map(t => ({
        subject: t.subject,
        predicate: t.predicate,
        object: t.object
      }));
      
      setTimeout(() => {
        setResults(mockResults);
        setIsExecuting(false);
        onExecute(query);
      }, 500);
    } catch (err) {
      setError('쿼리 실행 중 오류가 발생했습니다.');
      setIsExecuting(false);
    }
  }, [query, triples, onExecute]);

  const handleCopyQuery = () => {
    navigator.clipboard.writeText(query);
  };

  const handleSaveQuery = () => {
    const blob = new Blob([query], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'query.sparql';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full flex flex-col bg-gray-800 text-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold">SPARQL 쿼리</h2>
      </div>

      {/* Query Editor */}
      <div className="flex-1 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-medium">쿼리 편집기</span>
            <div className="flex gap-2">
              <button
                onClick={handleCopyQuery}
                className="p-1 hover:bg-gray-700 rounded"
                title="복사"
              >
                <Copy className="w-4 h-4" />
              </button>
              <button
                onClick={handleSaveQuery}
                className="p-1 hover:bg-gray-700 rounded"
                title="저장"
              >
                <Save className="w-4 h-4" />
              </button>
            </div>
          </div>
          <textarea
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            className="w-full h-32 p-3 bg-gray-900 border border-gray-700 rounded text-sm font-mono text-gray-300 focus:outline-none focus:border-blue-500"
            placeholder="SPARQL 쿼리를 입력하세요..."
          />
          <button
            onClick={handleExecute}
            disabled={isExecuting}
            className="mt-3 w-full flex items-center justify-center gap-2 py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded transition-colors"
          >
            <Play className="w-4 h-4" />
            <span>{isExecuting ? '실행 중...' : '실행'}</span>
          </button>
        </div>

        {/* Results */}
        <div className="flex-1 p-4 overflow-y-auto">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-medium">실행 결과</span>
            {results.length > 0 && (
              <span className="text-xs text-gray-400">{results.length}개 결과</span>
            )}
          </div>
          
          {error ? (
            <div className="p-3 bg-red-900/20 border border-red-700 rounded text-sm text-red-400">
              {error}
            </div>
          ) : results.length > 0 ? (
            <div className="bg-gray-900 border border-gray-700 rounded overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-800">
                    <th className="p-2 text-left border-b border-gray-700">Subject</th>
                    <th className="p-2 text-left border-b border-gray-700">Predicate</th>
                    <th className="p-2 text-left border-b border-gray-700">Object</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, idx) => (
                    <tr key={idx} className="hover:bg-gray-800/50">
                      <td className="p-2 border-b border-gray-700/50 text-blue-400">
                        {result.subject}
                      </td>
                      <td className="p-2 border-b border-gray-700/50 text-green-400">
                        {result.predicate}
                      </td>
                      <td className="p-2 border-b border-gray-700/50 text-orange-400">
                        {result.object}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-8 text-center text-gray-500 text-sm">
              쿼리를 실행하여 결과를 확인하세요
            </div>
          )}
        </div>

        {/* Status Bar */}
        <div className="p-4 border-t border-gray-700 text-xs text-gray-400">
          <div className="flex items-center justify-between">
            <span>준비됨</span>
            <span>실행 시간: {isExecuting ? '계산 중...' : '0.5초'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};