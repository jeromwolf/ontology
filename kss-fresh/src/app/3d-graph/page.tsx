'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Loader, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

interface Triple {
  subject: string;
  predicate: string;
  object: string;
  type?: 'resource' | 'literal';
}

// KnowledgeGraphContainer 컴포넌트를 동적으로 로드
const KnowledgeGraphContainer = dynamic(
  () => import('@/components/knowledge-graph/KnowledgeGraphContainer').then(mod => ({ default: mod.KnowledgeGraphContainer })),
  { 
    ssr: false,
    loading: () => (
      <div className="w-full h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <Loader className="w-8 h-8 animate-spin text-gray-400 mx-auto mb-4" />
          <p className="text-gray-400">지식그래프 시뮬레이터 로딩중...</p>
        </div>
      </div>
    )
  }
);

// 샘플 데이터
const sampleTriples: Triple[] = [
  { subject: ':지식관리시스템', predicate: ':type', object: ':System' },
  { subject: ':지식관리시스템', predicate: ':hasComponent', object: ':데이터베이스' },
  { subject: ':지식관리시스템', predicate: ':hasComponent', object: ':추론엔진' },
  { subject: ':지식관리시스템', predicate: ':hasComponent', object: ':시각화도구' },
  
  { subject: ':데이터베이스', predicate: ':type', object: ':Component' },
  { subject: ':데이터베이스', predicate: ':stores', object: ':RDF트리플' },
  { subject: ':데이터베이스', predicate: ':hasFeature', object: ':SPARQL지원' },
  
  { subject: ':추론엔진', predicate: ':type', object: ':Component' },
  { subject: ':추론엔진', predicate: ':performs', object: ':추론' },
  { subject: ':추론엔진', predicate: ':uses', object: ':규칙' },
  
  { subject: ':시각화도구', predicate: ':type', object: ':Component' },
  { subject: ':시각화도구', predicate: ':creates', object: ':그래프' },
  { subject: ':시각화도구', predicate: ':supports', object: ':3D렌더링' },
  
  { subject: ':사용자', predicate: ':uses', object: ':지식관리시스템' },
  { subject: ':사용자', predicate: ':hasRole', object: '연구원', type: 'literal' as const },
  { subject: ':사용자', predicate: ':hasName', object: '김철수', type: 'literal' as const },
];

export default function Graph3DPage() {
  const [isFromRDFEditor, setIsFromRDFEditor] = useState(false);
  const [triples, setTriples] = useState<Triple[]>(sampleTriples);
  const [hasChanges, setHasChanges] = useState(false);
  const [labelType, setLabelType] = useState<'html' | 'sprite' | 'text' | 'billboard'>('html');

  useEffect(() => {
    // Check URL params for labelType
    const searchParams = new URLSearchParams(window.location.search);
    const lt = searchParams.get('labelType') as 'html' | 'sprite' | 'text' | 'billboard';
    if (lt && ['html', 'sprite', 'text', 'billboard'].includes(lt)) {
      setLabelType(lt);
    }
  }, []);

  useEffect(() => {
    // Check for saved triples from RDF Editor
    const savedData = localStorage.getItem('rdf-editor-triples');
    if (savedData) {
      try {
        const parsed = JSON.parse(savedData);
        // Clear the data after loading (one-time use)
        localStorage.removeItem('rdf-editor-triples');
        
        // Check if data is recent (within last 5 minutes)
        const timestamp = new Date(parsed.timestamp);
        const now = new Date();
        const diffMinutes = (now.getTime() - timestamp.getTime()) / (1000 * 60);
        
        if (diffMinutes < 5 && parsed.triples && Array.isArray(parsed.triples)) {
          console.log('Loaded triples from RDF Editor:', parsed.triples.length);
          setIsFromRDFEditor(true);
          setTriples(parsed.triples);
        }
      } catch (error) {
        console.error('Error loading saved triples:', error);
      }
    }
  }, []);

  const handleTriplesChange = (newTriples: Triple[]) => {
    setTriples(newTriples);
    setHasChanges(true);
  };

  const handleSaveToRDFEditor = () => {
    // Save current triples for RDF Editor to pick up
    localStorage.setItem('3d-graph-triples', JSON.stringify({
      triples: triples,
      timestamp: new Date().toISOString(),
      source: '3d-graph'
    }));
    
    // Navigate back to RDF Editor
    window.location.href = '/rdf-editor';
  };

  return (
    <div className="h-screen bg-gray-50 dark:bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">3D 지식그래프 시뮬레이터</h1>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                {isFromRDFEditor ? (
                  <span className="text-green-600 dark:text-green-400">
                    ✓ RDF 에디터에서 {triples.length}개의 트리플을 불러왔습니다
                  </span>
                ) : (
                  '전문적인 지식그래프 편집, 시각화, 분석 도구'
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isFromRDFEditor && hasChanges && (
              <button 
                onClick={handleSaveToRDFEditor}
                className="px-3 py-1 text-sm bg-green-100 dark:bg-green-900/30 hover:bg-green-200 dark:hover:bg-green-900/50 text-green-700 dark:text-green-300 rounded transition-colors flex items-center gap-1"
              >
                <ArrowLeft className="w-3 h-3" />
                변경사항 저장하고 돌아가기
              </button>
            )}
            {isFromRDFEditor && !hasChanges && (
              <Link 
                href="/rdf-editor"
                className="px-3 py-1 text-sm bg-blue-100 dark:bg-blue-900/30 hover:bg-blue-200 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded transition-colors flex items-center gap-1"
              >
                <ArrowLeft className="w-3 h-3" />
                RDF 에디터로 돌아가기
              </Link>
            )}
            <button 
              onClick={() => window.close()}
              className="px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded transition-colors"
            >
              닫기
            </button>
          </div>
        </div>
      </div>
      
      {/* Full Height Container */}
      <div className="flex-1 relative">
        <KnowledgeGraphContainer 
          initialTriples={triples}
          onTriplesChange={handleTriplesChange}
          labelType={labelType}
        />
      </div>
    </div>
  );
}