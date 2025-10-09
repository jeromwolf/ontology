'use client'

import { Eye } from 'lucide-react'

export default function Section3() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
          <Eye className="text-green-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.3 Multimodal RAG: 텍스트를 넘어서</h2>
          <p className="text-gray-600 dark:text-gray-400">이미지, 비디오, 오디오를 통합한 차세대 검색</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">멀티모달 RAG의 최신 동향</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">🖼️ Visual RAG</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• CLIP 기반 이미지-텍스트 검색</li>
                <li>• LayoutLM을 활용한 문서 이해</li>
                <li>• Scene Graph 기반 추론</li>
                <li>• OCR + RAG 통합</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">🎥 Video RAG</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 시간적 정보 인덱싱</li>
                <li>• 키프레임 추출 및 검색</li>
                <li>• 비디오 요약과 QA</li>
                <li>• 실시간 스트리밍 RAG</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">최근 연구 하이라이트</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li><strong>• MM-RAG (Meta, 2024):</strong> 30B 파라미터 멀티모달 RAG, 이미지와 텍스트 동시 검색</li>
              <li><strong>• VideoChat-RAG (2024):</strong> 비디오 대화를 위한 시간 인식 RAG</li>
              <li><strong>• AudioRAG (Google, 2024):</strong> 음성/음악 검색과 생성 통합</li>
            </ul>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">통합 멀티모달 RAG 아키텍처</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`class UnifiedMultimodalRAG:
    """통합 멀티모달 RAG 시스템"""

    def __init__(self):
        # 모달리티별 인코더
        self.text_encoder = AutoModel.from_pretrained("bert-base")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2")

        # 통합 프로젝션 레이어
        self.projection = nn.Linear(768, 512)  # 공통 임베딩 공간

        # 크로스모달 어텐션
        self.cross_attention = nn.MultiheadAttention(512, 8)

    def encode_multimodal_query(self, query: Dict[str, Any]) -> torch.Tensor:
        """멀티모달 쿼리 인코딩"""
        embeddings = []

        if 'text' in query:
            text_emb = self.text_encoder(query['text'])
            embeddings.append(self.projection(text_emb))

        if 'image' in query:
            image_emb = self.image_encoder.get_image_features(query['image'])
            embeddings.append(self.projection(image_emb))

        if 'audio' in query:
            audio_emb = self.audio_encoder(query['audio']).last_hidden_state
            embeddings.append(self.projection(audio_emb.mean(dim=1)))

        # 크로스모달 융합
        if len(embeddings) > 1:
            fused = torch.stack(embeddings)
            attended, _ = self.cross_attention(fused, fused, fused)
            return attended.mean(dim=0)
        else:
            return embeddings[0]

    def retrieve_and_generate(self, query: Dict[str, Any]) -> str:
        """멀티모달 검색 및 생성"""
        # 1. 멀티모달 쿼리 인코딩
        query_embedding = self.encode_multimodal_query(query)

        # 2. 크로스모달 검색
        retrieved_items = self.cross_modal_search(query_embedding)

        # 3. 멀티모달 컨텍스트 구성
        context = self.build_multimodal_context(retrieved_items)

        # 4. 멀티모달 생성
        response = self.generate_with_multimodal_context(query, context)

        return response`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
