'use client';

import React from 'react';
import Link from 'next/link';
import { BookOpen, FlaskConical, Lightbulb, Target } from 'lucide-react';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4 flex items-center gap-2">
          <BookOpen className="w-6 h-6" />
          LLM이란 무엇인가?
        </h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-lg text-gray-700 dark:text-gray-300 mb-4">
            <strong>대형 언어 모델(Large Language Model, LLM)</strong>은 수십억 개의 매개변수를 가진 딥러닝 모델로, 
            인간의 언어를 이해하고 생성할 수 있는 인공지능입니다.
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Large</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">수십억~수조 개의 매개변수</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Language</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">자연어 처리에 특화</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Model</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">Transformer 아키텍처 기반</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">LLM의 역사적 발전</h3>
        <div className="space-y-4">
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2017: Transformer의 등장</h4>
            <p className="text-gray-600 dark:text-gray-400">
              "Attention Is All You Need" 논문으로 Transformer 아키텍처 소개
            </p>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2018: BERT & GPT-1</h4>
            <p className="text-gray-600 dark:text-gray-400">
              양방향 인코더(BERT)와 단방향 디코더(GPT) 모델의 성공
            </p>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2019-2020: GPT-2, GPT-3</h4>
            <p className="text-gray-600 dark:text-gray-400">
              스케일링의 힘 - 모델 크기가 성능을 좌우한다는 것을 증명
            </p>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2022: ChatGPT의 혁명</h4>
            <p className="text-gray-600 dark:text-gray-400">
              RLHF를 통한 사용자 친화적 AI의 탄생, 전 세계적 AI 붐 시작
            </p>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2023: GPT-4, Claude 2, Llama 2</h4>
            <p className="text-gray-600 dark:text-gray-400">
              멀티모달 지원, 오픈소스 모델의 부상, 기업별 경쟁 심화
            </p>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2024 상반기: Grok, Llama 3, Gemini 1.5</h4>
            <p className="text-gray-600 dark:text-gray-400">
              xAI의 Grok 등장, Meta Llama 3 405B 오픈소스 공개, 100만+ 토큰 컨텍스트 시대
            </p>
          </div>
          <div className="border-l-4 border-orange-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2024.09: o1 추론 모델 등장</h4>
            <p className="text-gray-600 dark:text-gray-400">
              OpenAI의 첫 추론 특화 모델, 수학/코딩에 특화된 체인 오브 생각 방식
            </p>
          </div>
          <div className="border-l-4 border-red-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2024.12: o3 모델 충격적 등장 🔥</h4>
            <p className="text-gray-600 dark:text-gray-400">
              o1의 후속 추론 모델, ARC-AGI 87.5%, 코딩 대회 2727 ELO로 AGI에 근접
            </p>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2025 상반기: Claude Opus 4, Grok 3, Gemini 2.5</h4>
            <p className="text-gray-600 dark:text-gray-400">
              코딩 특화 모델 발전 (SWE-bench 72.5%), 200만 토큰 컨텍스트, 자율 에이전트 시대 개막
            </p>
          </div>
          <div className="border-l-4 border-green-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">2025.08: GPT-5 공식 출시 🎯</h4>
            <p className="text-gray-600 dark:text-gray-400">
              OpenAI의 가장 스마트하고 빠른 모델, 정확도/속도/추론/문제해결 획기적 발전, 500만 기업 사용자
            </p>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">2025년 최신 LLM 현황</h3>
        
        {/* GPT-5 특별 공지 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl border-2 border-green-300 dark:border-green-700">
          <div className="flex items-start gap-3">
            <span className="text-2xl">🎯</span>
            <div>
              <h4 className="font-bold text-green-900 dark:text-green-200 mb-1">Breaking: GPT-5 공식 출시!</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                2025년 8월 OpenAI가 드디어 GPT-5를 공식 출시했습니다. "가장 스마트하고 빠른" 모델로, 
                정확도, 속도, 추론, 문제 해결 능력에서 획기적인 발전을 이루었으며, 이미 500만 기업이 사용 중입니다.
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-indigo-700 dark:text-indigo-300 mb-3">최강 성능 모델</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-green-500">🎯</span>
                  <div>
                    <strong>GPT-5</strong> (OpenAI, 2025.08)
                    <p className="text-gray-600 dark:text-gray-400">가장 스마트하고 빠른 모델, 정확도/속도/추론 획기적 발전</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500">🔥</span>
                  <div>
                    <strong>o3</strong> (OpenAI, 2024.12)
                    <p className="text-gray-600 dark:text-gray-400">추론 특화 모델, ARC-AGI 87.5%, 코딩 대회 2727 ELO</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">⚡</span>
                  <div>
                    <strong>o1</strong> (OpenAI, 2024.09)
                    <p className="text-gray-600 dark:text-gray-400">첫 추론 모델, 수학/코딩 특화, 체인 오브 생각</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-500">•</span>
                  <div>
                    <strong>Claude Opus 4</strong> (Anthropic, 2025.01)
                    <p className="text-gray-600 dark:text-gray-400">코딩 최강 (SWE-bench 72.5%), 7시간 자율 작업</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-500">•</span>
                  <div>
                    <strong>Grok 3</strong> (xAI, 2025.02)
                    <p className="text-gray-600 dark:text-gray-400">200K H100 GPU 학습, AIME 93.3% 달성</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-500">•</span>
                  <div>
                    <strong>Gemini 2.5 Pro</strong> (Google, 2025.03)
                    <p className="text-gray-600 dark:text-gray-400">200만 토큰 컨텍스트, 최고 가성비</p>
                  </div>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-purple-700 dark:text-purple-300 mb-3">오픈소스 혁신</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <div>
                    <strong>Llama 3.3 70B</strong> (Meta, 2024.12)
                    <p className="text-gray-600 dark:text-gray-400">405B급 성능을 70B 크기로 구현</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <div>
                    <strong>Mixtral 8x22B</strong> (Mistral, 2024)
                    <p className="text-gray-600 dark:text-gray-400">MoE 아키텍처, 효율적인 추론</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <div>
                    <strong>Qwen 2.5</strong> (Alibaba, 2024)
                    <p className="text-gray-600 dark:text-gray-400">중국어 최강, 다국어 지원</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <div>
                    <strong>DeepSeek V3</strong> (DeepSeek, 2024)
                    <p className="text-gray-600 dark:text-gray-400">코딩 특화, 저비용 고효율</p>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">최신 LLM 현황</h3>
        
        {/* 시뮬레이터 링크 추가 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-indigo-900 dark:text-indigo-200 mb-1">🎮 모델 비교 시뮬레이터</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                GPT, Claude, Gemini 등 주요 LLM 모델들의 특성과 성능을 비교해보세요
              </p>
            </div>
            <Link 
              href="/modules/llm/simulators/model-comparison"
              className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              <FlaskConical className="w-4 h-4" />
              시뮬레이터 실행
            </Link>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">주요 기술 트렌드</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">🔥 추론 혁명</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              o3: 새로운 추론 패러다임<br/>
              체인 오브 생각 내재화<br/>
              복잡한 문제 해결 능력
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">초거대 컨텍스트</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Gemini 2.5: 200만 토큰<br/>
              Claude 3: 20만 토큰<br/>
              전체 코드베이스 분석 가능
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">자율 에이전트</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Claude Opus 4: 7시간 자율작업<br/>
              Tool Use & Function Calling<br/>
              복잡한 작업 자동화
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">멀티모달 통합</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              이미지, 비디오, 오디오 이해<br/>
              실시간 스트리밍 처리<br/>
              크로스모달 추론
            </p>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">LLM이 가져온 패러다임 변화</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-3 flex items-center gap-2">
              <Lightbulb className="w-5 h-5" />
              Before LLM
            </h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 작업별 특화 모델</li>
              <li>• 대량의 라벨링 데이터 필요</li>
              <li>• 긴 개발 주기</li>
              <li>• 제한적인 일반화 능력</li>
            </ul>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-3 flex items-center gap-2">
              <Target className="w-5 h-5" />
              After LLM
            </h4>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 하나의 모델로 다양한 작업</li>
              <li>• Few-shot, Zero-shot 학습</li>
              <li>• 프롬프트만으로 빠른 개발</li>
              <li>• 강력한 일반화 능력</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}