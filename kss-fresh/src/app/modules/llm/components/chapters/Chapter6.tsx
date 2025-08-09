'use client'

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4">
          고급 기법과 최신 동향
        </h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-lg text-gray-700 dark:text-gray-300">
            2024-2025년 AI의 최전선: Multimodal, Diffusion, 차세대 아키텍처
          </p>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Multimodal AI 시스템</h3>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h4 className="font-bold text-purple-700 dark:text-purple-300 mb-4">Vision-Language Models</h4>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h5 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">최신 모델들</h5>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <div>
                    <strong>GPT-4V/4o</strong>: 이미지 이해 + 생성
                    <p className="text-gray-600 dark:text-gray-400">스크린샷 분석, 차트 해석, 코드 이미지 읽기</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <div>
                    <strong>Claude 3 Vision</strong>: 고정밀 이미지 분석
                    <p className="text-gray-600 dark:text-gray-400">문서 OCR, 다이어그램 이해, 의료 영상</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <div>
                    <strong>Gemini Ultra</strong>: 비디오 이해
                    <p className="text-gray-600 dark:text-gray-400">동영상 요약, 실시간 스트림 분석</p>
                  </div>
                </li>
              </ul>
            </div>
            <div>
              <h5 className="font-semibold text-pink-600 dark:text-pink-400 mb-2">핵심 기술</h5>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-pink-500">•</span>
                  <div>
                    <strong>CLIP</strong>: 이미지-텍스트 임베딩 정렬
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-pink-500">•</span>
                  <div>
                    <strong>BLIP-2</strong>: 효율적인 비전-언어 사전학습
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-pink-500">•</span>
                  <div>
                    <strong>LLaVA</strong>: 오픈소스 멀티모달 LLM
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-pink-500">•</span>
                  <div>
                    <strong>Flamingo</strong>: Few-shot 비전 학습
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Diffusion Models</h3>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
          <h4 className="font-bold text-green-700 dark:text-green-300 mb-4">이미지 생성의 혁명</h4>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h5 className="font-semibold text-green-600 dark:text-green-400 mb-2">Stable Diffusion 3</h5>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Multimodal Diffusion Transformer (MMDiT)</li>
                <li>• 텍스트 렌더링 개선, 고해상도 생성</li>
                <li>• ControlNet, LoRA 호환</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h5 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-2">DALL-E 3</h5>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• ChatGPT 통합으로 프롬프트 자동 개선</li>
                <li>• 텍스트 정확도 99%+</li>
                <li>• 일관된 캐릭터 생성</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h5 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">Midjourney V6</h5>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 포토리얼리즘 극대화</li>
                <li>• 프롬프트 이해도 향상</li>
                <li>• 스타일 일관성 유지</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">차세대 아키텍처</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">Mamba (SSM)</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              State Space Model 기반<br/>
              선형 시간 복잡도<br/>
              무한 컨텍스트 가능성
            </p>
          </div>
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
            <h4 className="font-semibold text-indigo-700 dark:text-indigo-300 mb-2">RWKV</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              RNN + Transformer 하이브리드<br/>
              메모리 효율적<br/>
              스트리밍 추론 지원
            </p>
          </div>
          <div className="bg-violet-50 dark:bg-violet-900/20 p-4 rounded-lg">
            <h4 className="font-semibold text-violet-700 dark:text-violet-300 mb-2">Flash Attention</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              IO-aware 알고리즘<br/>
              메모리 사용량 10배 감소<br/>
              속도 2-4배 향상
            </p>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Hugging Face 생태계</h3>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h4 className="font-bold text-orange-700 dark:text-orange-300 mb-4">🤗 통합 플랫폼</h4>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h5 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">핵심 라이브러리</h5>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <div>
                    <strong>Transformers</strong>: 20만+ 모델 접근
                    <p className="text-gray-600 dark:text-gray-400">from transformers import AutoModel</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <div>
                    <strong>Datasets</strong>: 10만+ 데이터셋
                    <p className="text-gray-600 dark:text-gray-400">load_dataset("squad")</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <div>
                    <strong>Accelerate</strong>: 분산 학습 간소화
                    <p className="text-gray-600 dark:text-gray-400">Multi-GPU, TPU 지원</p>
                  </div>
                </li>
              </ul>
            </div>
            <div>
              <h5 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">서비스 & 도구</h5>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-yellow-500">•</span>
                  <div>
                    <strong>Spaces</strong>: 모델 데모 배포
                    <p className="text-gray-600 dark:text-gray-400">Gradio, Streamlit 앱 호스팅</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-yellow-500">•</span>
                  <div>
                    <strong>AutoTrain</strong>: No-code 파인튜닝
                    <p className="text-gray-600 dark:text-gray-400">GUI로 모델 학습</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-yellow-500">•</span>
                  <div>
                    <strong>Inference API</strong>: 즉시 사용 가능
                    <p className="text-gray-600 dark:text-gray-400">REST API로 모델 호출</p>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">AI 서비스 생태계</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">엔터프라이즈</h4>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• OpenAI API</li>
              <li>• Anthropic Claude API</li>
              <li>• Google Vertex AI</li>
              <li>• AWS Bedrock</li>
              <li>• Azure OpenAI Service</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">개발자 도구</h4>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• LangChain</li>
              <li>• LlamaIndex</li>
              <li>• Pinecone (Vector DB)</li>
              <li>• Weights & Biases</li>
              <li>• Cohere Rerank</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">특화 서비스</h4>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• Perplexity (검색 AI)</li>
              <li>• GitHub Copilot</li>
              <li>• Cursor (AI IDE)</li>
              <li>• RunwayML (비디오)</li>
              <li>• ElevenLabs (음성)</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">효율화 기술</h3>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6">
          <h4 className="font-bold text-gray-700 dark:text-gray-300 mb-4">모델 최적화 기법</h4>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h5 className="font-semibold text-gray-600 dark:text-gray-400 mb-2">Parameter Efficient Fine-tuning</h5>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• <strong>LoRA</strong>: Low-Rank Adaptation (0.1% 파라미터)</li>
                <li>• <strong>QLoRA</strong>: 4-bit Quantized LoRA</li>
                <li>• <strong>Prefix Tuning</strong>: 프롬프트 임베딩 학습</li>
                <li>• <strong>Adapter</strong>: 작은 모듈 삽입</li>
              </ul>
            </div>
            <div>
              <h5 className="font-semibold text-gray-600 dark:text-gray-400 mb-2">Quantization & Compression</h5>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• <strong>GPTQ</strong>: 4-bit weight quantization</li>
                <li>• <strong>AWQ</strong>: Activation-aware quantization</li>
                <li>• <strong>Pruning</strong>: 불필요한 연결 제거</li>
                <li>• <strong>Distillation</strong>: 작은 모델로 지식 전달</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}