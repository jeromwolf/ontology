'use client';

import EmbeddingVisualizer from '../EmbeddingVisualizer';
import { Brain, Zap, Database, TrendingUp } from 'lucide-react';

// Chapter 3: Embeddings and Vector Search
export default function Chapter3() {
  return (
    <div className="space-y-8">
      {/* 페이지 헤더 */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 3: 임베딩과 벡터 검색의 마법</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          텍스트를 숫자로, 의미를 거리로 - AI가 언어를 이해하는 핵심 기술
        </p>
      </div>

      {/* 임베딩 개념 설명 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Brain className="text-purple-500" />
          임베딩(Embedding)이란?
        </h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            임베딩은 텍스트를 고차원 벡터 공간의 점으로 변환하는 과정입니다.
            쉽게 말해, 단어나 문장을 숫자의 배열로 표현하는 것입니다.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">입력: 텍스트</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                "AI를 배우고 싶어요"
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
              <Zap className="text-purple-500 mx-auto mb-2" size={24} />
              <p className="text-sm text-gray-600 dark:text-gray-400">변환</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">출력: 벡터</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                [0.21, -0.15, 0.88, ...]
              </p>
            </div>
          </div>
        </div>

        {/* 핵심 원리 */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">🎯 핵심 원리</h3>
            <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-300">
              <li>• <strong>의미 유사성:</strong> 비슷한 의미의 텍스트는 가까운 벡터로 표현</li>
              <li>• <strong>거리 계산:</strong> 코사인 유사도로 텍스트 간 유사성 측정</li>
              <li>• <strong>차원 축소:</strong> 복잡한 언어를 컴퓨터가 이해할 수 있는 형태로</li>
            </ul>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">💡 실제 활용</h3>
            <ul className="space-y-2 text-sm text-green-700 dark:text-green-300">
              <li>• <strong>의미 검색:</strong> "자동차"로 검색하면 "승용차", "차량"도 찾음</li>
              <li>• <strong>추천 시스템:</strong> 유사한 콘텐츠 자동 추천</li>
              <li>• <strong>번역/요약:</strong> 언어 간 의미 매핑</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 인터랙티브 시각화 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Database className="text-blue-500" />
          임베딩 시각화 체험
        </h2>
        <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl p-6">
          <p className="text-gray-600 dark:text-gray-400 mb-4 text-center">
            아래에서 다양한 텍스트가 벡터 공간에서 어떻게 배치되는지 직접 확인해보세요!
          </p>
          <EmbeddingVisualizer />
        </div>
      </section>

      {/* 임베딩 모델 상세 비교 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <TrendingUp className="text-emerald-500" />
          최신 임베딩 모델 완벽 가이드
        </h2>
        
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse bg-white dark:bg-gray-800 rounded-lg overflow-hidden shadow-sm">
            <thead>
              <tr className="bg-gradient-to-r from-emerald-500 to-green-600 text-white">
                <th className="px-6 py-3 text-left">모델명</th>
                <th className="px-6 py-3 text-left">차원</th>
                <th className="px-6 py-3 text-left">특징</th>
                <th className="px-6 py-3 text-left">장점</th>
                <th className="px-6 py-3 text-left">비용</th>
                <th className="px-6 py-3 text-left">추천 용도</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <td className="px-6 py-4">
                  <div>
                    <div className="font-semibold">OpenAI text-embedding-3-small</div>
                    <div className="text-xs text-gray-500">최신 모델</div>
                  </div>
                </td>
                <td className="px-6 py-4">1536</td>
                <td className="px-6 py-4 text-sm">
                  • 빠른 속도<br/>
                  • 높은 정확도
                </td>
                <td className="px-6 py-4 text-sm text-green-600 dark:text-green-400">
                  성능-비용 최적
                </td>
                <td className="px-6 py-4 text-sm font-mono">$0.00002/1K</td>
                <td className="px-6 py-4 text-sm">
                  일반적인 RAG
                </td>
              </tr>
              
              <tr className="bg-gray-50 dark:bg-gray-700/50 border-b border-gray-200 dark:border-gray-700">
                <td className="px-6 py-4">
                  <div>
                    <div className="font-semibold">OpenAI text-embedding-3-large</div>
                    <div className="text-xs text-gray-500">고성능</div>
                  </div>
                </td>
                <td className="px-6 py-4">3072</td>
                <td className="px-6 py-4 text-sm">
                  • 최고 정확도<br/>
                  • 세밀한 구분
                </td>
                <td className="px-6 py-4 text-sm text-blue-600 dark:text-blue-400">
                  최고 품질
                </td>
                <td className="px-6 py-4 text-sm font-mono">$0.00013/1K</td>
                <td className="px-6 py-4 text-sm">
                  정밀 검색 필요
                </td>
              </tr>

              <tr className="border-b border-gray-200 dark:border-gray-700">
                <td className="px-6 py-4">
                  <div>
                    <div className="font-semibold">Cohere embed-v3</div>
                    <div className="text-xs text-gray-500">다국어 특화</div>
                  </div>
                </td>
                <td className="px-6 py-4">1024</td>
                <td className="px-6 py-4 text-sm">
                  • 100+ 언어<br/>
                  • 압축 지원
                </td>
                <td className="px-6 py-4 text-sm text-purple-600 dark:text-purple-400">
                  다국어 최강
                </td>
                <td className="px-6 py-4 text-sm font-mono">$0.00010/1K</td>
                <td className="px-6 py-4 text-sm">
                  글로벌 서비스
                </td>
              </tr>

              <tr className="bg-gray-50 dark:bg-gray-700/50 border-b border-gray-200 dark:border-gray-700">
                <td className="px-6 py-4">
                  <div>
                    <div className="font-semibold">BGE-M3</div>
                    <div className="text-xs text-gray-500">오픈소스</div>
                  </div>
                </td>
                <td className="px-6 py-4">1024</td>
                <td className="px-6 py-4 text-sm">
                  • 다중 검색<br/>
                  • 한국어 우수
                </td>
                <td className="px-6 py-4 text-sm text-orange-600 dark:text-orange-400">
                  무료 + 성능
                </td>
                <td className="px-6 py-4 text-sm font-mono">무료</td>
                <td className="px-6 py-4 text-sm">
                  온프레미스
                </td>
              </tr>

              <tr className="border-b border-gray-200 dark:border-gray-700">
                <td className="px-6 py-4">
                  <div>
                    <div className="font-semibold">Sentence-BERT</div>
                    <div className="text-xs text-gray-500">경량화</div>
                  </div>
                </td>
                <td className="px-6 py-4">768</td>
                <td className="px-6 py-4 text-sm">
                  • 빠른 추론<br/>
                  • 작은 모델
                </td>
                <td className="px-6 py-4 text-sm text-gray-600 dark:text-gray-400">
                  속도 우선
                </td>
                <td className="px-6 py-4 text-sm font-mono">무료</td>
                <td className="px-6 py-4 text-sm">
                  실시간 처리
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* 모델 선택 가이드 */}
        <div className="mt-6 bg-amber-50 dark:bg-amber-900/20 rounded-xl p-6">
          <h3 className="font-semibold text-amber-800 dark:text-amber-200 mb-3">
            🎯 어떤 모델을 선택해야 할까요?
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium text-amber-700 dark:text-amber-300 mb-2">시작하는 경우</h4>
              <p className="text-sm text-amber-600 dark:text-amber-400">
                → OpenAI text-embedding-3-small로 시작하세요. 
                안정적이고 비용 효율적입니다.
              </p>
            </div>
            <div>
              <h4 className="font-medium text-amber-700 dark:text-amber-300 mb-2">비용이 중요한 경우</h4>
              <p className="text-sm text-amber-600 dark:text-amber-400">
                → BGE-M3 같은 오픈소스 모델을 자체 호스팅하세요. 
                초기 설정만 하면 무료입니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 실습 코드 */}
      <section className="bg-gray-900 rounded-xl p-6">
        <h3 className="text-white font-bold mb-4">🔥 5분 만에 임베딩 시작하기</h3>
        <pre className="text-sm text-gray-300 overflow-x-auto">
          <code>{`# 1. OpenAI 임베딩 생성
from openai import OpenAI

client = OpenAI()

def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 2. 유사도 계산
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

# 3. 실제 사용
text1 = "AI를 배우고 싶어요"
text2 = "인공지능 학습에 관심있어요"
text3 = "오늘 날씨가 좋네요"

emb1 = create_embedding(text1)
emb2 = create_embedding(text2)
emb3 = create_embedding(text3)

print(f"유사도 (AI-인공지능): {cosine_similarity(emb1, emb2):.3f}")
print(f"유사도 (AI-날씨): {cosine_similarity(emb1, emb3):.3f}")

# 결과:
# 유사도 (AI-인공지능): 0.923  <- 높음!
# 유사도 (AI-날씨): 0.234      <- 낮음!`}</code>
        </pre>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-purple-800 dark:text-purple-200">📚 핵심 정리</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h3 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">이해한 개념</h3>
            <ul className="space-y-1 text-sm">
              <li>✓ 임베딩은 텍스트를 벡터로 변환</li>
              <li>✓ 유사한 의미는 가까운 벡터</li>
              <li>✓ 다양한 모델의 특징과 선택 기준</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">실습한 내용</h3>
            <ul className="space-y-1 text-sm">
              <li>✓ 임베딩 시각화 체험</li>
              <li>✓ OpenAI API로 임베딩 생성</li>
              <li>✓ 코사인 유사도 계산</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}