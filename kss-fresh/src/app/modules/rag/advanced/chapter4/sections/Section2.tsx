'use client'

import { Zap } from 'lucide-react'

export default function Section2() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
          <Zap className="text-purple-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.2 ColBERT: Late Interaction의 혁명</h2>
          <p className="text-gray-600 dark:text-gray-400">토큰 레벨 상호작용으로 속도와 정확도를 모두 잡다</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">ColBERT의 혁신적 아키텍처</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>ColBERT (Contextualized Late Interaction over BERT)는 Stanford에서 개발한
              획기적인 검색 모델입니다.</strong> Bi-Encoder의 효율성과 Cross-Encoder의 정확성 사이의
              최적점을 찾아, 토큰 레벨에서 늦은 상호작용(Late Interaction)을 수행합니다.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">핵심 혁신 포인트</h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-purple-600 font-bold">1</span>
                </div>
                <div>
                  <h5 className="font-medium text-gray-900 dark:text-white">Token-level Encoding</h5>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    각 토큰을 개별적으로 인코딩하여 세밀한 의미 포착
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-purple-600 font-bold">2</span>
                </div>
                <div>
                  <h5 className="font-medium text-gray-900 dark:text-white">MaxSim Operation</h5>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    쿼리의 각 토큰에 대해 문서에서 가장 유사한 토큰 매칭
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-purple-600 font-bold">3</span>
                </div>
                <div>
                  <h5 className="font-medium text-gray-900 dark:text-white">Precomputable Document Representations</h5>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    문서 표현을 사전 계산하여 검색 속도 대폭 향상
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">ColBERT v2 구현 및 최적화</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Tuple, Dict, Optional
import faiss
from dataclasses import dataclass

@dataclass
class ColBERTConfig:
    """ColBERT 설정"""
    model_name: str = 'bert-base-uncased'
    max_query_len: int = 32
    max_doc_len: int = 180
    dim: int = 128
    similarity: str = 'cosine'
    compression_level: int = 2  # 압축 수준 (1-4)

class ColBERTv2:
    def __init__(self, config: ColBERTConfig):
        """
        ColBERT v2 구현
        - Residual Compression 적용
        - Denoised Supervision
        - Improved Training Efficiency
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # BERT 모델 로드
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Linear projection for dimension reduction
        self.linear = nn.Linear(768, config.dim)

        # 압축을 위한 Centroid 학습
        self.centroids = None
        self.residual_encoder = None

        self.bert.to(self.device)
        self.linear.to(self.device)

    def encode_query(self, query: str) -> torch.Tensor:
        """쿼리 인코딩"""
        # 토크나이징
        encoded = self.tokenizer(
            query,
            max_length=self.config.max_query_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # BERT 인코딩
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        # 차원 축소
        embeddings = self.linear(embeddings)

        # L2 정규화
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Mask expansion (쿼리 확장)
        # [MASK] 토큰을 추가하여 더 많은 매칭 기회 제공
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids == self.tokenizer.pad_token_id).nonzero()[:, 1]

        if len(mask_positions) > 0:
            # 일부 패딩 토큰을 [MASK]로 변경
            num_masks = min(5, len(mask_positions))
            for i in range(num_masks):
                input_ids[0, mask_positions[i]] = mask_token_id

        return embeddings, attention_mask

    def encode_document(self, document: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """문서 인코딩 with Residual Compression"""
        # 토크나이징
        encoded = self.tokenizer(
            document,
            max_length=self.config.max_doc_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # BERT 인코딩
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        # 차원 축소
        embeddings = self.linear(embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Residual Compression (ColBERT v2의 핵심)
        if self.centroids is not None:
            compressed_embeddings = self._compress_embeddings(embeddings)
            return compressed_embeddings, attention_mask

        return embeddings, attention_mask

    def _compress_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Residual Compression
        - Centroid 기반 압축으로 메모리 사용량 75% 감소
        - 성능 손실 최소화
        """
        batch_size, seq_len, dim = embeddings.shape

        # 1. Centroid 할당
        embeddings_flat = embeddings.view(-1, dim)
        distances = torch.cdist(embeddings_flat, self.centroids)
        centroid_ids = distances.argmin(dim=1)

        # 2. Residual 계산
        assigned_centroids = self.centroids[centroid_ids]
        residuals = embeddings_flat - assigned_centroids

        # 3. Residual 양자화 (2-bit)
        if self.config.compression_level >= 2:
            # 각 차원에 대해 2-bit 양자화
            residual_scale = residuals.abs().max(dim=0)[0]
            residuals_normalized = residuals / (residual_scale + 1e-6)
            residuals_quantized = torch.round(residuals_normalized * 3) / 3
            residuals = residuals_quantized * residual_scale

        # 4. 압축된 표현 반환
        compressed = {
            'centroid_ids': centroid_ids.view(batch_size, seq_len),
            'residuals': residuals.view(batch_size, seq_len, dim),
            'shape': embeddings.shape
        }

        return compressed

    def late_interaction(self, query_embeddings: torch.Tensor,
                        doc_embeddings: torch.Tensor,
                        query_mask: torch.Tensor,
                        doc_mask: torch.Tensor) -> float:
        """
        Late Interaction (MaxSim) 수행
        각 쿼리 토큰에 대해 문서에서 가장 유사한 토큰을 찾아 점수 계산
        """
        # 압축된 경우 복원
        if isinstance(doc_embeddings, dict):
            doc_embeddings = self._decompress_embeddings(doc_embeddings)

        # 유효한 토큰만 고려
        query_embeddings = query_embeddings[0]  # batch size 1 가정
        doc_embeddings = doc_embeddings[0]

        query_valid = query_mask[0].bool()
        doc_valid = doc_mask[0].bool()

        # 유효한 임베딩만 선택
        q_embs = query_embeddings[query_valid]
        d_embs = doc_embeddings[doc_valid]

        # Cosine similarity matrix
        scores = torch.matmul(q_embs, d_embs.transpose(0, 1))

        # MaxSim: 각 쿼리 토큰에 대해 최대 유사도 선택
        max_scores = scores.max(dim=1)[0]

        # 평균 점수 반환
        return max_scores.mean().item()

    def build_index(self, documents: List[str], batch_size: int = 32) -> faiss.Index:
        """
        FAISS 인덱스 구축 (대규모 검색용)
        """
        all_embeddings = []
        doc_boundaries = [0]  # 각 문서의 시작 위치

        print("Building ColBERT index...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            for doc in batch:
                embs, mask = self.encode_document(doc)
                valid_embs = embs[0][mask[0].bool()]  # 유효한 토큰만
                all_embeddings.append(valid_embs.cpu().numpy())
                doc_boundaries.append(doc_boundaries[-1] + len(valid_embs))

        # 모든 임베딩을 하나로 결합
        all_embeddings = np.vstack(all_embeddings)

        # FAISS 인덱스 생성
        index = faiss.IndexFlatIP(self.config.dim)  # Inner Product

        # IVF for faster search
        if len(all_embeddings) > 10000:
            nlist = int(np.sqrt(len(all_embeddings)))
            quantizer = faiss.IndexFlatIP(self.config.dim)
            index = faiss.IndexIVFFlat(quantizer, self.config.dim, nlist)
            index.train(all_embeddings)

        index.add(all_embeddings)

        return index, doc_boundaries

    def retrieve(self, query: str, index: faiss.Index,
                doc_boundaries: List[int], k: int = 10) -> List[Tuple[int, float]]:
        """
        ColBERT 검색 수행
        """
        # 쿼리 인코딩
        q_embs, q_mask = self.encode_query(query)
        valid_q_embs = q_embs[0][q_mask[0].bool()].cpu().numpy()

        # 각 쿼리 토큰에 대해 검색
        doc_scores = {}

        for q_emb in valid_q_embs:
            # k*10 개의 가장 유사한 토큰 검색
            scores, indices = index.search(q_emb.reshape(1, -1), k * 10)

            # 문서별로 점수 집계
            for score, idx in zip(scores[0], indices[0]):
                # 어느 문서에 속하는지 찾기
                doc_id = np.searchsorted(doc_boundaries[1:], idx, side='right')

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = []
                doc_scores[doc_id].append(score)

        # 문서별 최종 점수 계산 (MaxSim 평균)
        final_scores = []
        for doc_id, scores in doc_scores.items():
            # 각 쿼리 토큰의 최대 점수 평균
            avg_score = np.mean(scores)
            final_scores.append((doc_id, avg_score))

        # 점수순 정렬
        final_scores.sort(key=lambda x: x[1], reverse=True)

        return final_scores[:k]

# 사용 예제
config = ColBERTConfig(
    dim=128,
    compression_level=2  # Residual compression 활성화
)
colbert = ColBERTv2(config)

# 문서 인덱싱
documents = [
    "ColBERT는 효율적인 신경망 검색 모델입니다.",
    "BERT 기반의 late interaction을 사용합니다.",
    "토큰 레벨의 상호작용으로 정확도를 높입니다."
]

index, boundaries = colbert.build_index(documents)

# 검색 수행
query = "ColBERT의 작동 원리"
results = colbert.retrieve(query, index, boundaries, k=3)

print("=== ColBERT 검색 결과 ===")
for doc_id, score in results:
    print(f"문서 {doc_id}: {documents[doc_id][:50]}... (점수: {score:.3f})")`}
            </pre>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl border border-yellow-200 dark:border-yellow-700">
          <h3 className="font-bold text-yellow-800 dark:text-yellow-200 mb-4">성능 벤치마크: MS MARCO에서의 결과</h3>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">MRR@10</h4>
              <p className="text-3xl font-bold text-purple-600">39.1</p>
              <p className="text-xs text-gray-500 mt-1">BM25: 18.7</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">Recall@1000</h4>
              <p className="text-3xl font-bold text-green-600">97.3</p>
              <p className="text-xs text-gray-500 mt-1">DPR: 95.2</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">Latency</h4>
              <p className="text-3xl font-bold text-orange-600">58ms</p>
              <p className="text-xs text-gray-500 mt-1">BERT: 430ms</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
