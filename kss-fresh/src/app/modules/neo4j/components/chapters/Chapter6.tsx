'use client';

import React from 'react';
import { Database, Network, Zap, Globe } from 'lucide-react';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">KSS 도메인 통합 🌐</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          Knowledge Space Simulator의 다양한 도메인 데이터를 
          Neo4j 그래프로 통합하여 지식의 연결성을 탐험하세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 KSS 지식 그래프 아키텍처</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">모든 학습 도메인을 하나의 그래프로</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <Database className="text-blue-500 mb-2" size={24} />
              <h4 className="font-medium">온톨로지</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">RDF → Property Graph</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <Network className="text-green-500 mb-2" size={24} />
              <h4 className="font-medium">스마트 팩토리</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">IoT 센서 연결</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <Zap className="text-purple-500 mb-2" size={24} />
              <h4 className="font-medium">AI/ML</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">모델 관계 매핑</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 실시간 데이터 통합</h2>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">라이브 데이터 스트림</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span>주식 시장 데이터 → Neo4j 실시간 업데이트</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
              <span>공장 센서 데이터 → 이상 패턴 감지</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-purple-500 rounded-full animate-pulse"></div>
              <span>학습자 진도 → 개인화된 경로 추천</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔮 미래 확장 계획</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl">
            <Globe className="text-indigo-500 mb-3" size={28} />
            <h3 className="font-semibold mb-2">글로벌 지식 네트워크</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              전 세계 대학과 연구소의 지식을 연결하여 
              인류 최대의 학습 그래프 구축
            </p>
          </div>
          <div className="bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 p-6 rounded-xl">
            <Zap className="text-emerald-500 mb-3" size={28} />
            <h3 className="font-semibold mb-2">AI 기반 자동 큐레이션</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              학습자의 목표와 진도에 맞춰 
              최적의 학습 경로를 실시간 생성
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}