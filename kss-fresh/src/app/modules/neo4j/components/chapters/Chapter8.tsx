'use client';

import React from 'react';
import { Rocket, Target, Trophy, Sparkles } from 'lucide-react';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">실전 프로젝트 🚀</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          지금까지 학습한 모든 지식을 활용하여 
          실제 현업에서 사용할 수 있는 Neo4j 프로젝트를 구축해보세요!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 프로젝트 선택</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl cursor-pointer hover:shadow-lg transition-shadow">
            <Target className="text-blue-500 mb-3" size={28} />
            <h3 className="font-semibold mb-2">추천 시스템</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              사용자 행동과 아이템 특성을 그래프로 모델링하여 
              개인화된 추천 엔진 구축
            </p>
            <div className="mt-3 text-xs text-blue-600 dark:text-blue-400">
              난이도: 중급 | 기간: 2주
            </div>
          </div>

          <div className="bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 p-6 rounded-xl cursor-pointer hover:shadow-lg transition-shadow">
            <Sparkles className="text-emerald-500 mb-3" size={28} />
            <h3 className="font-semibold mb-2">사기 탐지 시스템</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              금융 거래 네트워크에서 패턴 분석을 통한 
              실시간 사기 거래 탐지 시스템
            </p>
            <div className="mt-3 text-xs text-emerald-600 dark:text-emerald-400">
              난이도: 고급 | 기간: 3주
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl cursor-pointer hover:shadow-lg transition-shadow">
            <Trophy className="text-purple-500 mb-3" size={28} />
            <h3 className="font-semibold mb-2">지식 그래프 AI</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              기업 내부 문서와 데이터를 연결하여 
              ChatGPT 수준의 질의응답 시스템 구축
            </p>
            <div className="mt-3 text-xs text-purple-600 dark:text-purple-400">
              난이도: 전문가 | 기간: 4주
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📋 프로젝트 진행 단계</h2>
        <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl p-6">
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                1
              </div>
              <div>
                <h3 className="font-semibold">요구사항 분석 및 설계</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  비즈니스 요구사항을 그래프 모델로 변환하고 데이터 스키마 설계
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-emerald-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                2
              </div>
              <div>
                <h3 className="font-semibold">데이터 수집 및 전처리</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  실제 데이터 소스에서 데이터를 추출하고 Neo4j 적재를 위한 변환
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                3
              </div>
              <div>
                <h3 className="font-semibold">핵심 알고리즘 구현</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Cypher 쿼리와 그래프 알고리즘을 활용한 비즈니스 로직 개발
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                4
              </div>
              <div>
                <h3 className="font-semibold">API 개발 및 배포</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  RESTful API 구축 및 클라우드 환경에 프로덕션 배포
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🏆 성과 측정</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6">
            <h3 className="font-semibold mb-4 text-emerald-700 dark:text-emerald-300">기술적 성과</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                쿼리 성능 최적화 (응답시간 100ms 이하)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                높은 가용성 달성 (99.9% 업타임)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                확장 가능한 아키텍처 설계
              </li>
            </ul>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
            <h3 className="font-semibold mb-4 text-blue-700 dark:text-blue-300">비즈니스 임팩트</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                사용자 만족도 향상 (평점 4.5/5 이상)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                운영 효율성 개선 (작업 시간 30% 단축)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                비용 절감 효과 (연간 20% 절약)
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎓 다음 학습 로드맵</h2>
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">Neo4j 마스터가 되는 길</h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center">
              <Rocket className="text-indigo-500 mx-auto mb-2" size={24} />
              <h4 className="font-medium">고급 개발</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                커스텀 프로시저<br/>
                Neo4j 플러그인 개발
              </p>
            </div>
            <div className="text-center">
              <Target className="text-purple-500 mx-auto mb-2" size={24} />
              <h4 className="font-medium">전문 분야</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                그래프 ML<br/>
                실시간 분석 시스템
              </p>
            </div>
            <div className="text-center">
              <Trophy className="text-pink-500 mx-auto mb-2" size={24} />
              <h4 className="font-medium">커뮤니티</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                오픈소스 기여<br/>
                기술 컨퍼런스 발표
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}