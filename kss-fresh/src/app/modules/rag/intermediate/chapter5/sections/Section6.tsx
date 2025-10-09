'use client'

import { Shuffle } from 'lucide-react'
import References from '@/components/common/References'

export default function Section6() {
  return (
    <>
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl bg-teal-100 dark:bg-teal-900/20 flex items-center justify-center">
            <Shuffle className="text-teal-600" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.6 크로스 모달 검색 전략</h2>
            <p className="text-gray-600 dark:text-gray-400">모달리티 간 연관성 활용한 고급 검색</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-teal-50 dark:bg-teal-900/20 p-6 rounded-xl border border-teal-200 dark:border-teal-700">
            <h3 className="font-bold text-teal-800 dark:text-teal-200 mb-4">통합 멀티모달 검색 전략</h3>

            <div className="prose prose-sm dark:prose-invert mb-4">
              <p className="text-gray-700 dark:text-gray-300">
                <strong>크로스모달 검색은 서로 다른 데이터 타입 간의 의미적 연결을 활용하여 통합된 검색 경험을 제공합니다.</strong>
                텍스트 쿼리로 이미지를 찾거나, 이미지로 관련 음성을 검색하는 등
                기존 단일 모달 검색의 한계를 뛰어넘는 혁신적 접근법입니다.
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <strong>핵심 융합 전략:</strong>
              </p>
              <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
                <li><strong>Feature-level Fusion</strong>: 임베딩 차원에서 특성 벡터 결합</li>
                <li><strong>Score-level Fusion</strong>: 각 모달리티 검색 점수의 가중 평균</li>
                <li><strong>Decision-level Fusion</strong>: 최종 결정 단계에서 결과 통합</li>
                <li><strong>Adaptive Weighting</strong>: 쿼리 특성에 따른 동적 가중치 조정</li>
              </ul>

              <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg border border-cyan-200 dark:border-cyan-700 mt-4">
                <h4 className="font-bold text-cyan-800 dark:text-cyan-200 mb-2">🎯 실험 결과 비교</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-cyan-300 dark:border-cyan-600">
                        <th className="text-left py-2 text-cyan-800 dark:text-cyan-200">검색 방식</th>
                        <th className="text-left py-2 text-cyan-800 dark:text-cyan-200">텍스트→이미지</th>
                        <th className="text-left py-2 text-cyan-800 dark:text-cyan-200">이미지→텍스트</th>
                        <th className="text-left py-2 text-cyan-800 dark:text-cyan-200">통합 성능</th>
                      </tr>
                    </thead>
                    <tbody className="text-cyan-700 dark:text-cyan-300">
                      <tr>
                        <td className="py-1">단일 모달</td>
                        <td className="py-1">82.4%</td>
                        <td className="py-1">79.1%</td>
                        <td className="py-1">80.8%</td>
                      </tr>
                      <tr>
                        <td className="py-1">Early Fusion</td>
                        <td className="py-1">89.2%</td>
                        <td className="py-1">86.7%</td>
                        <td className="py-1">88.0%</td>
                      </tr>
                      <tr>
                        <td className="py-1">Late Fusion</td>
                        <td className="py-1">91.5%</td>
                        <td className="py-1">88.9%</td>
                        <td className="py-1">90.2%</td>
                      </tr>
                      <tr>
                        <td className="py-1">Adaptive</td>
                        <td className="py-1">93.8%</td>
                        <td className="py-1">91.4%</td>
                        <td className="py-1">92.6%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg border border-emerald-200 dark:border-emerald-700 mt-4">
                <h4 className="font-bold text-emerald-800 dark:text-emerald-200 mb-2">🚀 차세대 기술 동향</h4>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong className="text-emerald-800 dark:text-emerald-200">GPT-4V 통합</strong>
                    <p className="text-emerald-700 dark:text-emerald-300 mt-1">
                      OpenAI의 Vision 모델과 RAG 결합으로 이미지 이해도 대폭 향상.
                      복잡한 차트, 다이어그램도 정확한 텍스트 설명으로 변환.
                    </p>
                  </div>
                  <div>
                    <strong className="text-emerald-800 dark:text-emerald-200">DALL-E 3 역검색</strong>
                    <p className="text-emerald-700 dark:text-emerald-300 mt-1">
                      텍스트 설명으로 유사한 이미지 생성 후,
                      생성 이미지와 실제 이미지 간 유사도로 검색 정확도 향상.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">🔄 모달리티 융합 방식</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="font-medium text-teal-600 mb-1">Early Fusion</p>
                    <p className="text-gray-600 dark:text-gray-400">
                      임베딩 레벨에서 직접 결합
                    </p>
                  </div>
                  <div>
                    <p className="font-medium text-teal-600 mb-1">Late Fusion</p>
                    <p className="text-gray-600 dark:text-gray-400">
                      검색 결과 수준에서 가중치 결합
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">⚖️ 동적 가중치 조정</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  쿼리 타입과 컨텍스트에 따른 모달리티별 중요도 자동 조정
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">🎯 컨텍스트 인식 검색</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  이전 검색 결과와 사용자 의도를 고려한 개인화된 멀티모달 검색
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Practical Exercise */}
      <section className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">실습 과제</h2>

        <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
          <h3 className="font-bold mb-4">멀티모달 RAG 구축 실습</h3>

          <div className="space-y-4">
            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🎥 과제 1: 비디오 기반 QA 시스템</h4>
              <ol className="space-y-2 text-sm">
                <li>1. 교육 비디오에서 키프레임 및 전사 추출</li>
                <li>2. 시각적 내용과 음성 내용 통합 인덱싱</li>
                <li>3. "이 부분에서 설명하는 개념은?" 타입 질의 처리</li>
                <li>4. 정확한 타임스탬프와 함께 답변 제공</li>
              </ol>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">📊 과제 2: 문서 내 차트 분석 RAG</h4>
              <ul className="space-y-1 text-sm">
                <li>• PDF에서 차트/그래프 자동 추출</li>
                <li>• 차트 데이터를 텍스트로 변환</li>
                <li>• "수익이 가장 높은 분기는?" 등 데이터 질의 처리</li>
                <li>• 시각적 증거와 함께 답변 생성</li>
              </ul>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🔄 과제 3: 크로스모달 검색 엔진</h4>
              <ul className="space-y-1 text-sm">
                <li>• 텍스트 쿼리로 관련 이미지 검색</li>
                <li>• 이미지 업로드로 관련 텍스트 검색</li>
                <li>• 오디오 클립으로 관련 문서 검색</li>
                <li>• 검색 결과의 신뢰도 평가 시스템</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 멀티모달 AI & CLIP',
            icon: 'web' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'OpenAI CLIP Documentation',
                authors: 'OpenAI',
                year: '2021',
                description: '이미지-텍스트 통합 임베딩 - 4억 쌍 학습',
                link: 'https://github.com/openai/CLIP'
              },
              {
                title: 'Hugging Face Transformers - Vision',
                authors: 'Hugging Face',
                year: '2025',
                description: 'ViT, CLIP, BLIP 등 멀티모달 모델 라이브러리',
                link: 'https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder'
              },
              {
                title: 'LangChain Multi-Modal RAG',
                authors: 'LangChain',
                year: '2025',
                description: '이미지/비디오/오디오 처리 - 통합 RAG 파이프라인',
                link: 'https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector'
              },
              {
                title: 'GPT-4 Vision API',
                authors: 'OpenAI',
                year: '2024',
                description: '이미지 이해 및 분석 - RAG 응답 생성',
                link: 'https://platform.openai.com/docs/guides/vision'
              },
              {
                title: 'Gemini Pro Vision',
                authors: 'Google DeepMind',
                year: '2024',
                description: '네이티브 멀티모달 LLM - 이미지, 비디오, 오디오 통합',
                link: 'https://ai.google.dev/tutorials/multimodal'
              }
            ]
          },
          {
            title: '📖 멀티모달 학습 & 검색 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'CLIP: Learning Transferable Visual Models',
                authors: 'Radford et al., OpenAI',
                year: '2021',
                description: 'Contrastive Learning - 제로샷 이미지 분류',
                link: 'https://arxiv.org/abs/2103.00020'
              },
              {
                title: 'BLIP-2: Bootstrapping Vision-Language',
                authors: 'Li et al., Salesforce',
                year: '2023',
                description: 'Q-Former로 효율적인 멀티모달 학습',
                link: 'https://arxiv.org/abs/2301.12597'
              },
              {
                title: 'Flamingo: Visual Language Model',
                authors: 'Alayrac et al., DeepMind',
                year: '2022',
                description: '이미지/비디오/텍스트 인터리빙 처리',
                link: 'https://arxiv.org/abs/2204.14198'
              },
              {
                title: 'Wav2Vec 2.0: Self-Supervised Audio',
                authors: 'Baevski et al., Meta',
                year: '2020',
                description: '오디오 표현 학습 - 음성 검색 기반',
                link: 'https://arxiv.org/abs/2006.11477'
              }
            ]
          },
          {
            title: '🛠️ 멀티모달 RAG 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Unstructured.io',
                authors: 'Unstructured',
                year: '2025',
                description: 'PDF/이미지/표 추출 - RAG용 문서 전처리',
                link: 'https://unstructured.io/'
              },
              {
                title: 'Twelve Labs Video Understanding',
                authors: 'Twelve Labs',
                year: '2024',
                description: '비디오 검색 & 분석 API - 장면 기반 검색',
                link: 'https://docs.twelvelabs.io/'
              },
              {
                title: 'AssemblyAI Audio Intelligence',
                authors: 'AssemblyAI',
                year: '2025',
                description: '음성-텍스트 변환 - 감정, 화자 분리, 요약',
                link: 'https://www.assemblyai.com/docs'
              },
              {
                title: 'Pinecone Namespaces',
                authors: 'Pinecone',
                year: '2025',
                description: '멀티모달 벡터 저장 - 타입별 네임스페이스 분리',
                link: 'https://docs.pinecone.io/docs/namespaces'
              },
              {
                title: 'LlamaIndex ImageNode',
                authors: 'LlamaIndex',
                year: '2025',
                description: '이미지 노드 처리 - 텍스트와 이미지 통합 인덱싱',
                link: 'https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html'
              }
            ]
          }
        ]}
      />
    </>
  )
}
