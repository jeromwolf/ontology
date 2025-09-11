'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Trophy, BarChart3, Zap, Brain, Sparkles, Target } from 'lucide-react'

export default function Chapter4Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/advanced"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          고급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-orange-500 to-red-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Trophy size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 4: 고급 Reranking 전략</h1>
              <p className="text-orange-100 text-lg">검색 품질을 극대화하는 최신 재순위화 기법</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Cross-Encoder의 혁신 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
              <Brain className="text-orange-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.1 Cross-Encoder: Bi-Encoder의 한계를 넘어서</h2>
              <p className="text-gray-600 dark:text-gray-400">쿼리와 문서를 함께 인코딩하는 혁신적 접근</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
              <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">Bi-Encoder vs Cross-Encoder 심층 비교</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>기존 Bi-Encoder의 근본적 한계:</strong> 쿼리와 문서를 독립적으로 인코딩하기 때문에
                  상호작용(interaction) 정보를 포착할 수 없습니다. Cross-Encoder는 이를 해결하여
                  훨씬 정확한 관련성 평가를 가능하게 합니다.
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-3">Bi-Encoder (기존)</h4>
                  <div className="space-y-2 text-sm">
                    <div className="bg-red-50 dark:bg-red-900/30 p-2 rounded">
                      <strong>독립 인코딩:</strong> query → embedding, doc → embedding
                    </div>
                    <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                      <strong>속도:</strong> 매우 빠름 (사전 계산 가능)
                    </div>
                    <div className="bg-blue-50 dark:bg-blue-900/30 p-2 rounded">
                      <strong>확장성:</strong> 수백만 문서 처리 가능
                    </div>
                    <div className="bg-orange-50 dark:bg-orange-900/30 p-2 rounded">
                      <strong>정확도:</strong> 상대적으로 낮음
                    </div>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-3">Cross-Encoder (고급)</h4>
                  <div className="space-y-2 text-sm">
                    <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                      <strong>공동 인코딩:</strong> [query, doc] → score
                    </div>
                    <div className="bg-orange-50 dark:bg-orange-900/30 p-2 rounded">
                      <strong>속도:</strong> 느림 (실시간 계산 필요)
                    </div>
                    <div className="bg-red-50 dark:bg-red-900/30 p-2 rounded">
                      <strong>확장성:</strong> Top-K 재순위화에만 사용
                    </div>
                    <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                      <strong>정확도:</strong> 매우 높음 (SOTA)
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">Cross-Encoder 구현: ms-marco-MiniLM 활용</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple, Dict
import torch
from dataclasses import dataclass
import time

@dataclass
class RerankingResult:
    """재순위화 결과"""
    doc_id: str
    original_score: float
    reranked_score: float
    original_rank: int
    new_rank: int
    content: str

class AdvancedCrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        """
        고급 Cross-Encoder 재순위화 시스템
        - MS MARCO로 학습된 최신 모델 사용
        - GPU 가속 지원
        - 배치 처리 최적화
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(model_name, device=self.device)
        
        # 성능 메트릭
        self.stats = {
            'total_reranked': 0,
            'avg_latency': 0,
            'score_improvements': []
        }
    
    def rerank(self, query: str, documents: List[Dict], 
               top_k: int = 10, batch_size: int = 32) -> List[RerankingResult]:
        """
        문서 재순위화 수행
        
        Args:
            query: 검색 쿼리
            documents: [{'id': str, 'content': str, 'score': float}]
            top_k: 재순위화할 상위 문서 수
            batch_size: 배치 크기
        """
        start_time = time.time()
        
        # 1단계: 초기 순위 기록
        documents = sorted(documents, key=lambda x: x['score'], reverse=True)
        docs_to_rerank = documents[:top_k]
        
        # 2단계: Cross-Encoder 점수 계산
        pairs = [[query, doc['content']] for doc in docs_to_rerank]
        
        # 배치 처리로 효율성 향상
        cross_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            batch_scores = self.model.predict(batch)
            cross_scores.extend(batch_scores)
        
        # 3단계: 점수 정규화 및 결합
        results = []
        for i, (doc, cross_score) in enumerate(zip(docs_to_rerank, cross_scores)):
            # 원본 점수와 Cross-Encoder 점수 결합
            # α를 조정하여 가중치 조절 가능
            alpha = 0.7  # Cross-Encoder 가중치
            combined_score = alpha * cross_score + (1 - alpha) * doc['score']
            
            results.append(RerankingResult(
                doc_id=doc['id'],
                original_score=doc['score'],
                reranked_score=combined_score,
                original_rank=i + 1,
                new_rank=-1,  # 나중에 업데이트
                content=doc['content'][:200] + '...'
            ))
        
        # 4단계: 재순위화
        results.sort(key=lambda x: x.reranked_score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i + 1
        
        # 5단계: 나머지 문서 추가 (재순위화하지 않은 문서들)
        remaining_docs = documents[top_k:]
        for i, doc in enumerate(remaining_docs):
            results.append(RerankingResult(
                doc_id=doc['id'],
                original_score=doc['score'],
                reranked_score=doc['score'],
                original_rank=top_k + i + 1,
                new_rank=top_k + i + 1,
                content=doc['content'][:200] + '...'
            ))
        
        # 성능 통계 업데이트
        latency = time.time() - start_time
        self._update_stats(results, latency)
        
        return results
    
    def _update_stats(self, results: List[RerankingResult], latency: float):
        """성능 통계 업데이트"""
        self.stats['total_reranked'] += len(results)
        self.stats['avg_latency'] = (
            (self.stats['avg_latency'] * (self.stats['total_reranked'] - len(results)) + 
             latency * len(results)) / self.stats['total_reranked']
        )
        
        # 점수 개선 추적
        for result in results:
            if result.original_rank <= 10:  # Top-10 내에서의 변화만 추적
                improvement = result.original_rank - result.new_rank
                self.stats['score_improvements'].append(improvement)
    
    def get_performance_report(self) -> Dict:
        """성능 리포트 생성"""
        if not self.stats['score_improvements']:
            return self.stats
        
        improvements = self.stats['score_improvements']
        return {
            **self.stats,
            'avg_rank_improvement': np.mean(improvements),
            'median_rank_improvement': np.median(improvements),
            'improvement_rate': len([x for x in improvements if x > 0]) / len(improvements)
        }

# 사용 예제
reranker = AdvancedCrossEncoderReranker()

# 검색 결과 (Bi-Encoder에서 나온 초기 결과)
search_results = [
    {'id': 'doc1', 'content': 'Python은 인터프리터 언어입니다...', 'score': 0.89},
    {'id': 'doc2', 'content': 'Python 프로그래밍의 기초...', 'score': 0.87},
    {'id': 'doc3', 'content': 'Python으로 웹 개발하기...', 'score': 0.85},
    # ... 더 많은 문서
]

# 재순위화 수행
query = "Python 인터프리터의 작동 원리"
reranked = reranker.rerank(query, search_results, top_k=20)

# 결과 분석
print("=== 재순위화 결과 ===")
for result in reranked[:10]:
    if result.original_rank != result.new_rank:
        change = result.original_rank - result.new_rank
        symbol = "↑" if change > 0 else "↓"
        print(f"{result.new_rank}. {result.doc_id} "
              f"(원래: {result.original_rank}위 {symbol}{abs(change)}) "
              f"점수: {result.reranked_score:.3f}")

# 성능 리포트
report = reranker.get_performance_report()
print(f"\\n평균 순위 개선: {report['avg_rank_improvement']:.2f}")
print(f"개선율: {report['improvement_rate']*100:.1f}%")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: ColBERT - 효율성과 정확성의 균형 */}
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

        {/* Section 3: Diversity-aware Reranking */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
              <Sparkles className="text-green-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.3 다양성 인식 재순위화</h2>
              <p className="text-gray-600 dark:text-gray-400">관련성과 다양성의 최적 균형점 찾기</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">MMR (Maximal Marginal Relevance) 고급 구현</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>검색 결과의 다양성은 사용자 경험에 매우 중요합니다.</strong>
                  특히 모호한 쿼리나 다면적 정보 요구에서는 단순히 관련성이 높은 문서만
                  보여주는 것보다 다양한 관점을 제공하는 것이 효과적입니다.
                </p>
              </div>

              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer

class DiversityAwareReranker:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        다양성 인식 재순위화 시스템
        - MMR (Maximal Marginal Relevance)
        - 주제 클러스터링 기반 다양성
        - 사용자 피드백 기반 학습
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.lambda_param = 0.7  # 관련성 vs 다양성 가중치
        
        # 주제 클러스터 정보 (사전 학습된 것으로 가정)
        self.topic_clusters = None
        self.aspect_keywords = self._load_aspect_keywords()
    
    def _load_aspect_keywords(self) -> Dict[str, List[str]]:
        """각 측면에 대한 키워드 로드"""
        return {
            'technical': ['algorithm', 'implementation', 'performance', 'code'],
            'conceptual': ['theory', 'concept', 'principle', 'fundamental'],
            'practical': ['example', 'use case', 'application', 'real-world'],
            'comparison': ['vs', 'compare', 'difference', 'better'],
            'tutorial': ['how to', 'guide', 'step by step', 'learn']
        }
    
    def mmr_rerank(self, query: str, documents: List[Dict], 
                   lambda_param: float = None, top_k: int = 10) -> List[Dict]:
        """
        Maximal Marginal Relevance 기반 재순위화
        
        MMR = λ * Rel(d) - (1-λ) * max(Sim(d, d_i))
        """
        if lambda_param is None:
            lambda_param = self.lambda_param
        
        # 쿼리와 문서 임베딩
        query_emb = self.encoder.encode([query])
        doc_contents = [doc['content'] for doc in documents]
        doc_embs = self.encoder.encode(doc_contents)
        
        # 쿼리-문서 관련성
        relevance_scores = cosine_similarity(query_emb, doc_embs)[0]
        
        # MMR 알고리즘
        selected = []
        candidates = list(range(len(documents)))
        
        while len(selected) < top_k and candidates:
            mmr_scores = []
            
            for idx in candidates:
                # 관련성 점수
                rel_score = relevance_scores[idx]
                
                # 이미 선택된 문서들과의 최대 유사도
                if selected:
                    selected_embs = doc_embs[selected]
                    similarities = cosine_similarity(
                        [doc_embs[idx]], selected_embs
                    )[0]
                    max_sim = similarities.max()
                else:
                    max_sim = 0
                
                # MMR 점수 계산
                mmr = lambda_param * rel_score - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            
            # 최고 MMR 점수를 가진 문서 선택
            best_idx = candidates[np.argmax(mmr_scores)]
            selected.append(best_idx)
            candidates.remove(best_idx)
        
        # 선택된 문서들 반환
        reranked = []
        for rank, idx in enumerate(selected):
            doc = documents[idx].copy()
            doc['mmr_score'] = relevance_scores[idx]
            doc['diversity_rank'] = rank + 1
            reranked.append(doc)
        
        return reranked
    
    def aspect_aware_rerank(self, query: str, documents: List[Dict], 
                           ensure_aspects: int = 3) -> List[Dict]:
        """
        측면 인식 재순위화
        다양한 관점/측면이 포함되도록 보장
        """
        # 각 문서의 측면 분류
        doc_aspects = []
        for doc in documents:
            aspects = self._classify_aspects(doc['content'])
            doc_aspects.append(aspects)
        
        # 측면별 최상위 문서 선택
        aspect_best = defaultdict(list)
        for i, (doc, aspects) in enumerate(zip(documents, doc_aspects)):
            for aspect in aspects:
                aspect_best[aspect].append((doc['score'], i))
        
        # 각 측면별로 정렬
        for aspect in aspect_best:
            aspect_best[aspect].sort(reverse=True)
        
        # 다양한 측면을 커버하도록 선택
        selected = []
        selected_indices = set()
        aspect_counts = defaultdict(int)
        
        # Round-robin 방식으로 각 측면에서 선택
        aspect_cycle = list(aspect_best.keys())
        aspect_idx = 0
        
        while len(selected) < min(len(documents), ensure_aspects * 3):
            aspect = aspect_cycle[aspect_idx % len(aspect_cycle)]
            
            # 해당 측면에서 아직 선택되지 않은 최상위 문서 찾기
            for score, doc_idx in aspect_best[aspect]:
                if doc_idx not in selected_indices:
                    selected.append(documents[doc_idx])
                    selected_indices.add(doc_idx)
                    aspect_counts[aspect] += 1
                    break
            
            aspect_idx += 1
            
            # 모든 측면이 최소 1개씩 포함되었는지 확인
            if len(selected) >= ensure_aspects and \
               all(count > 0 for count in aspect_counts.values()):
                break
        
        # 나머지 문서들은 점수순으로 추가
        for i, doc in enumerate(documents):
            if i not in selected_indices and len(selected) < 10:
                selected.append(doc)
        
        return selected
    
    def _classify_aspects(self, text: str) -> List[str]:
        """문서의 측면 분류"""
        text_lower = text.lower()
        aspects = []
        
        for aspect, keywords in self.aspect_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                aspects.append(aspect)
        
        # 측면이 없으면 기본값
        if not aspects:
            aspects = ['general']
        
        return aspects
    
    def learning_to_diversify(self, query: str, documents: List[Dict],
                            user_feedback: List[int] = None) -> List[Dict]:
        """
        사용자 피드백 기반 다양성 학습
        클릭률, 체류시간 등을 활용하여 최적 λ 값 학습
        """
        if user_feedback is None:
            # 피드백이 없으면 기본 MMR 사용
            return self.mmr_rerank(query, documents)
        
        # 피드백 기반 λ 조정
        # 상위 문서들의 유사도가 높은데 클릭률이 낮으면 λ 감소 (다양성 증가)
        top_5_similarity = self._calculate_diversity_score(documents[:5])
        click_rate = sum(user_feedback[:5]) / min(5, len(user_feedback))
        
        if top_5_similarity > 0.8 and click_rate < 0.4:
            # 너무 유사한 결과, 다양성 필요
            adjusted_lambda = max(0.3, self.lambda_param - 0.2)
        elif top_5_similarity < 0.5 and click_rate > 0.7:
            # 충분히 다양함, 관련성 증가 필요
            adjusted_lambda = min(0.9, self.lambda_param + 0.1)
        else:
            adjusted_lambda = self.lambda_param
        
        return self.mmr_rerank(query, documents, lambda_param=adjusted_lambda)
    
    def _calculate_diversity_score(self, documents: List[Dict]) -> float:
        """문서 집합의 다양성 점수 계산 (0-1, 낮을수록 다양)"""
        if len(documents) < 2:
            return 0.0
        
        contents = [doc['content'] for doc in documents]
        embeddings = self.encoder.encode(contents)
        
        # 모든 쌍의 유사도 평균
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0

# 고급 재순위화 파이프라인
class AdvancedRerankingPipeline:
    def __init__(self):
        """
        다단계 재순위화 파이프라인
        1. Cross-Encoder로 관련성 재평가
        2. MMR로 다양성 확보
        3. 사용자 선호도 반영
        """
        self.cross_encoder = AdvancedCrossEncoderReranker()
        self.diversity_reranker = DiversityAwareReranker()
        self.user_preferences = {}
    
    def rerank(self, query: str, documents: List[Dict], 
              user_id: str = None, ensure_diversity: bool = True) -> List[Dict]:
        """
        통합 재순위화 수행
        """
        # 1단계: Cross-Encoder 재순위화
        ce_results = self.cross_encoder.rerank(query, documents, top_k=30)
        
        # RerankingResult를 Dict로 변환
        reranked_docs = []
        for result in ce_results:
            reranked_docs.append({
                'id': result.doc_id,
                'content': result.content,
                'score': result.reranked_score,
                'original_rank': result.original_rank
            })
        
        # 2단계: 다양성 재순위화
        if ensure_diversity:
            # 사용자별 선호도 반영
            lambda_param = 0.7
            if user_id and user_id in self.user_preferences:
                lambda_param = self.user_preferences[user_id].get('lambda', 0.7)
            
            final_results = self.diversity_reranker.mmr_rerank(
                query, reranked_docs[:20], 
                lambda_param=lambda_param, 
                top_k=10
            )
        else:
            final_results = reranked_docs[:10]
        
        return final_results
    
    def update_user_preference(self, user_id: str, feedback: Dict):
        """사용자 선호도 업데이트"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {'lambda': 0.7}
        
        # 피드백 기반으로 lambda 조정
        if feedback.get('wanted_more_diversity'):
            self.user_preferences[user_id]['lambda'] = max(
                0.3, self.user_preferences[user_id]['lambda'] - 0.1
            )
        elif feedback.get('wanted_more_relevance'):
            self.user_preferences[user_id]['lambda'] = min(
                0.9, self.user_preferences[user_id]['lambda'] + 0.1
            )

# 사용 예제
pipeline = AdvancedRerankingPipeline()

# 초기 검색 결과
search_results = [
    {'id': '1', 'content': 'Python 프로그래밍 기초 튜토리얼...', 'score': 0.92},
    {'id': '2', 'content': 'Python 입문자를 위한 가이드...', 'score': 0.91},
    {'id': '3', 'content': 'Python vs Java 성능 비교...', 'score': 0.88},
    {'id': '4', 'content': 'Python 고급 기법과 최적화...', 'score': 0.87},
    {'id': '5', 'content': 'Python으로 웹 개발하기...', 'score': 0.86},
]

# 재순위화 수행
query = "Python 프로그래밍 배우기"
final_results = pipeline.rerank(query, search_results, ensure_diversity=True)

print("=== 다양성을 고려한 재순위화 결과 ===")
for i, doc in enumerate(final_results, 1):
    print(f"{i}. {doc['id']}: {doc['content'][:50]}...")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Learning to Rank */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
              <Target className="text-blue-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.4 Learning to Rank (LTR) for RAG</h2>
              <p className="text-gray-600 dark:text-gray-400">기계학습을 활용한 최적 순위 학습</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">LambdaMART를 활용한 RAG 최적화</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>Learning to Rank는 검색 품질을 극대화하는 핵심 기술입니다.</strong>
                  특히 RAG 시스템에서는 단순한 벡터 유사도를 넘어서 다양한 특징을 종합적으로
                  고려하여 최적의 순위를 학습할 수 있습니다.
                </p>
              </div>

              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
                <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class RankingFeatures:
    """랭킹을 위한 특징들"""
    # 텍스트 유사도 특징
    bm25_score: float
    vector_similarity: float
    cross_encoder_score: float
    
    # 문서 특징
    doc_length: int
    doc_freshness: float  # 최신성 (일 단위)
    doc_popularity: float  # 클릭률, 조회수 등
    
    # 쿼리-문서 매칭 특징
    exact_match_count: int
    synonym_match_count: int
    named_entity_overlap: float
    
    # 의미적 특징
    topic_similarity: float
    sentiment_alignment: float
    
    # 사용자 특징
    user_click_history: float  # 이 문서 타입에 대한 과거 클릭률
    session_dwell_time: float  # 세션 내 평균 체류시간

class LearningToRankRAG:
    def __init__(self, model_path: Optional[str] = None):
        """
        Learning to Rank 기반 RAG 재순위화
        - LambdaMART (XGBoost) 사용
        - 온라인 학습 지원
        - Feature importance 분석
        """
        self.scaler = StandardScaler()
        self.feature_names = [
            'bm25_score', 'vector_similarity', 'cross_encoder_score',
            'doc_length_norm', 'doc_freshness', 'doc_popularity',
            'exact_match_count', 'synonym_match_count', 'named_entity_overlap',
            'topic_similarity', 'sentiment_alignment',
            'user_click_history', 'session_dwell_time'
        ]
        
        if model_path:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        else:
            self.model = None
            
        # 온라인 학습을 위한 버퍼
        self.training_buffer = []
        self.buffer_size = 1000
    
    def extract_features(self, query: str, document: Dict, 
                        user_context: Optional[Dict] = None) -> RankingFeatures:
        """
        쿼리-문서 쌍에서 랭킹 특징 추출
        """
        # 기본 점수들 (이미 계산되어 있다고 가정)
        features = RankingFeatures(
            bm25_score=document.get('bm25_score', 0.0),
            vector_similarity=document.get('vector_score', 0.0),
            cross_encoder_score=document.get('ce_score', 0.0),
            doc_length=len(document['content'].split()),
            doc_freshness=self._calculate_freshness(document),
            doc_popularity=document.get('popularity', 0.0),
            exact_match_count=self._count_exact_matches(query, document['content']),
            synonym_match_count=self._count_synonym_matches(query, document['content']),
            named_entity_overlap=self._calculate_entity_overlap(query, document['content']),
            topic_similarity=document.get('topic_similarity', 0.0),
            sentiment_alignment=self._calculate_sentiment_alignment(query, document),
            user_click_history=0.0,
            session_dwell_time=0.0
        )
        
        # 사용자 컨텍스트가 있으면 개인화 특징 추가
        if user_context:
            features.user_click_history = user_context.get('doc_type_ctr', {}).get(
                document.get('type', 'general'), 0.0
            )
            features.session_dwell_time = user_context.get('avg_dwell_time', 0.0)
        
        return features
    
    def _calculate_freshness(self, document: Dict) -> float:
        """문서 최신성 계산 (0-1)"""
        from datetime import datetime, timedelta
        
        if 'created_at' not in document:
            return 0.5
        
        created = datetime.fromisoformat(document['created_at'])
        age_days = (datetime.now() - created).days
        
        # 지수적 감쇠 (30일 반감기)
        return np.exp(-age_days / 30.0)
    
    def _count_exact_matches(self, query: str, content: str) -> int:
        """정확한 단어 매칭 수"""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        return len(query_terms.intersection(content_terms))
    
    def _count_synonym_matches(self, query: str, content: str) -> int:
        """동의어 매칭 수 (간단한 예제)"""
        # 실제로는 WordNet이나 사전 학습된 동의어 사전 사용
        synonym_dict = {
            'python': ['파이썬', 'python3', 'py'],
            'machine learning': ['ml', '머신러닝', '기계학습'],
            'deep learning': ['dl', '딥러닝', '심층학습']
        }
        
        count = 0
        query_lower = query.lower()
        content_lower = content.lower()
        
        for term, synonyms in synonym_dict.items():
            if term in query_lower:
                for syn in synonyms:
                    if syn in content_lower:
                        count += 1
        
        return count
    
    def _calculate_entity_overlap(self, query: str, content: str) -> float:
        """명명된 개체 중첩도 (간단한 버전)"""
        # 실제로는 NER 모델 사용
        # 여기서는 대문자로 시작하는 단어를 개체로 가정
        query_entities = set(w for w in query.split() if w[0].isupper())
        content_entities = set(w for w in content.split() if w[0].isupper())
        
        if not query_entities:
            return 0.0
        
        overlap = len(query_entities.intersection(content_entities))
        return overlap / len(query_entities)
    
    def _calculate_sentiment_alignment(self, query: str, document: Dict) -> float:
        """감성 정렬도 (질문과 답변의 톤 일치)"""
        # 간단한 규칙 기반 (실제로는 감성 분석 모델 사용)
        positive_words = {'good', 'best', 'excellent', 'great', '좋은', '최고'}
        negative_words = {'bad', 'worst', 'poor', 'terrible', '나쁜', '최악'}
        
        query_sentiment = sum(1 for w in query.split() if w in positive_words) - \
                         sum(1 for w in query.split() if w in negative_words)
        
        doc_sentiment = sum(1 for w in document['content'].split() if w in positive_words) - \
                       sum(1 for w in document['content'].split() if w in negative_words)
        
        # 같은 부호면 1, 다른 부호면 0
        if query_sentiment * doc_sentiment >= 0:
            return 1.0
        return 0.0
    
    def features_to_vector(self, features: RankingFeatures) -> np.ndarray:
        """특징을 벡터로 변환"""
        return np.array([
            features.bm25_score,
            features.vector_similarity,
            features.cross_encoder_score,
            features.doc_length / 1000.0,  # 정규화
            features.doc_freshness,
            features.doc_popularity,
            features.exact_match_count / 10.0,  # 정규화
            features.synonym_match_count / 5.0,  # 정규화
            features.named_entity_overlap,
            features.topic_similarity,
            features.sentiment_alignment,
            features.user_click_history,
            features.session_dwell_time / 60.0  # 분 단위로 정규화
        ])
    
    def train_model(self, training_data: List[Tuple[str, List[Dict], List[int]]],
                   validation_data: Optional[List] = None):
        """
        LambdaMART 모델 학습
        
        Args:
            training_data: [(query, documents, relevance_labels)]
            validation_data: 검증 데이터 (선택사항)
        """
        # 특징 추출
        X_train = []
        y_train = []
        qids = []
        
        for qid, (query, documents, labels) in enumerate(training_data):
            for doc, label in zip(documents, labels):
                features = self.extract_features(query, doc)
                X_train.append(self.features_to_vector(features))
                y_train.append(label)
                qids.append(qid)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        qids = np.array(qids)
        
        # 특징 스케일링
        X_train = self.scaler.fit_transform(X_train)
        
        # XGBoost DMatrix 생성
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(np.bincount(qids))
        
        # LambdaMART 파라미터
        params = {
            'objective': 'rank:ndcg',
            'eval_metric': ['ndcg@10', 'map@10'],
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.0
        }
        
        # 모델 학습
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        # Feature importance 분석
        self._analyze_feature_importance()
    
    def _analyze_feature_importance(self):
        """특징 중요도 분석"""
        if self.model is None:
            return
        
        importance = self.model.get_score(importance_type='gain')
        
        # Feature name 매핑
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            key = f'f{i}'
            if key in importance:
                feature_importance[name] = importance[key]
        
        # 중요도 순으로 정렬
        sorted_importance = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("=== Feature Importance ===")
        for feature, score in sorted_importance[:10]:
            print(f"{feature}: {score:.2f}")
    
    def predict_ranking_scores(self, query: str, documents: List[Dict],
                             user_context: Optional[Dict] = None) -> List[float]:
        """
        문서들의 랭킹 점수 예측
        """
        if self.model is None:
            # 모델이 없으면 기본 점수 사용
            return [doc.get('score', 0.0) for doc in documents]
        
        # 특징 추출
        features = []
        for doc in documents:
            feat = self.extract_features(query, doc, user_context)
            features.append(self.features_to_vector(feat))
        
        # 스케일링
        X = self.scaler.transform(np.array(features))
        
        # 예측
        dtest = xgb.DMatrix(X)
        scores = self.model.predict(dtest)
        
        return scores.tolist()
    
    def rerank_with_ltr(self, query: str, documents: List[Dict],
                       user_context: Optional[Dict] = None) -> List[Dict]:
        """
        Learning to Rank를 사용한 재순위화
        """
        # 점수 예측
        scores = self.predict_ranking_scores(query, documents, user_context)
        
        # 문서와 점수 결합
        for doc, score in zip(documents, scores):
            doc['ltr_score'] = score
        
        # 점수순 정렬
        reranked = sorted(documents, key=lambda x: x['ltr_score'], reverse=True)
        
        return reranked
    
    def collect_feedback(self, query: str, documents: List[Dict], 
                        clicks: List[int], dwell_times: List[float]):
        """
        사용자 피드백 수집 (온라인 학습용)
        """
        # 클릭과 체류시간을 기반으로 관련성 레이블 생성
        relevance_labels = []
        for click, dwell_time in zip(clicks, dwell_times):
            if not click:
                label = 0
            elif dwell_time < 10:  # 10초 미만
                label = 1
            elif dwell_time < 30:  # 30초 미만
                label = 2
            else:
                label = 3  # 높은 관련성
            
            relevance_labels.append(label)
        
        # 학습 버퍼에 추가
        self.training_buffer.append((query, documents, relevance_labels))
        
        # 버퍼가 가득 차면 재학습
        if len(self.training_buffer) >= self.buffer_size:
            self._retrain_online()
    
    def _retrain_online(self):
        """온라인 재학습"""
        print("Online retraining with {} examples...".format(len(self.training_buffer)))
        
        # 기존 모델을 베이스로 추가 학습
        self.train_model(self.training_buffer)
        
        # 버퍼 초기화
        self.training_buffer = []

# 통합 재순위화 시스템
class UnifiedRerankingSystem:
    def __init__(self):
        """
        모든 재순위화 기법을 통합한 시스템
        """
        self.cross_encoder = AdvancedCrossEncoderReranker()
        self.colbert = ColBERTv2(ColBERTConfig())
        self.diversity_reranker = DiversityAwareReranker()
        self.ltr_model = LearningToRankRAG()
    
    def rerank(self, query: str, initial_results: List[Dict], 
              reranking_strategy: str = 'hybrid',
              user_context: Optional[Dict] = None) -> List[Dict]:
        """
        통합 재순위화 수행
        
        Strategies:
        - 'cross_encoder': Cross-Encoder만 사용
        - 'colbert': ColBERT만 사용
        - 'diversity': MMR 다양성 재순위화
        - 'ltr': Learning to Rank
        - 'hybrid': 모든 기법 조합 (기본값)
        """
        if reranking_strategy == 'cross_encoder':
            return self._rerank_cross_encoder(query, initial_results)
        
        elif reranking_strategy == 'colbert':
            return self._rerank_colbert(query, initial_results)
        
        elif reranking_strategy == 'diversity':
            return self.diversity_reranker.mmr_rerank(query, initial_results)
        
        elif reranking_strategy == 'ltr':
            return self.ltr_model.rerank_with_ltr(query, initial_results, user_context)
        
        elif reranking_strategy == 'hybrid':
            # 1단계: Cross-Encoder로 Top-30 재순위화
            ce_results = self._rerank_cross_encoder(query, initial_results[:50])[:30]
            
            # 2단계: LTR 모델로 점수 조정
            if self.ltr_model.model is not None:
                ltr_results = self.ltr_model.rerank_with_ltr(query, ce_results, user_context)
            else:
                ltr_results = ce_results
            
            # 3단계: MMR로 최종 다양성 확보
            final_results = self.diversity_reranker.mmr_rerank(
                query, ltr_results[:20], top_k=10
            )
            
            return final_results
        
        else:
            raise ValueError(f"Unknown strategy: {reranking_strategy}")
    
    def _rerank_cross_encoder(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Cross-Encoder 재순위화 래퍼"""
        results = self.cross_encoder.rerank(query, documents)
        return [
            {
                'id': r.doc_id,
                'content': r.content,
                'score': r.reranked_score
            }
            for r in results
        ]
    
    def _rerank_colbert(self, query: str, documents: List[Dict]) -> List[Dict]:
        """ColBERT 재순위화"""
        # ColBERT 인덱스 구축 (실제로는 사전 구축)
        doc_contents = [d['content'] for d in documents]
        index, boundaries = self.colbert.build_index(doc_contents)
        
        # 검색 수행
        results = self.colbert.retrieve(query, index, boundaries)
        
        # 결과 포맷팅
        reranked = []
        for doc_id, score in results:
            reranked.append({
                'id': documents[doc_id]['id'],
                'content': documents[doc_id]['content'],
                'score': score
            })
        
        return reranked

# 사용 예제
print("=== 고급 재순위화 시스템 데모 ===\\n")

# 초기 검색 결과
initial_results = [
    {'id': '1', 'content': 'Cross-Encoder는 쿼리와 문서를 함께 인코딩합니다.', 'score': 0.89},
    {'id': '2', 'content': 'ColBERT는 토큰 레벨 상호작용을 사용합니다.', 'score': 0.87},
    {'id': '3', 'content': 'MMR은 다양성을 위한 재순위화 알고리즘입니다.', 'score': 0.85},
    {'id': '4', 'content': 'Learning to Rank는 기계학습 기반 순위 최적화입니다.', 'score': 0.83},
    {'id': '5', 'content': 'BERT 기반 검색 모델들의 성능 비교.', 'score': 0.82},
]

# 통합 시스템 초기화
unified_system = UnifiedRerankingSystem()

# 다양한 전략으로 재순위화
strategies = ['cross_encoder', 'diversity', 'hybrid']

for strategy in strategies:
    print(f"\\n=== {strategy.upper()} 전략 결과 ===")
    results = unified_system.rerank(
        "재순위화 알고리즘의 종류", 
        initial_results, 
        reranking_strategy=strategy
    )
    
    for i, doc in enumerate(results[:5], 1):
        print(f"{i}. [{doc['id']}] {doc['content'][:50]}... (점수: {doc.get('score', 0):.3f})")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Practical Exercise */}
        <section className="bg-gradient-to-r from-orange-500 to-red-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">실습 과제</h2>
          
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">고급 재순위화 시스템 구축</h3>
            
            <div className="space-y-4">
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">📋 요구사항</h4>
                <ol className="space-y-2 text-sm">
                  <li>1. Cross-Encoder와 ColBERT를 모두 활용한 하이브리드 시스템 구현</li>
                  <li>2. 쿼리 유형별 최적 재순위화 전략 자동 선택</li>
                  <li>3. A/B 테스트를 통한 λ 파라미터 최적화</li>
                  <li>4. 실시간 사용자 피드백 기반 온라인 학습</li>
                  <li>5. 재순위화 성능 모니터링 대시보드 구축</li>
                </ol>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🎯 성능 목표</h4>
                <ul className="space-y-1 text-sm">
                  <li>• MRR@10: 0.4 이상</li>
                  <li>• NDCG@10: 0.6 이상</li>
                  <li>• 재순위화 레이턴시: &lt; 100ms (P95)</li>
                  <li>• 다양성 점수: 0.7 이상</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">💡 도전 과제</h4>
                <p className="text-sm">
                  다국어 재순위화 지원을 추가하여 한국어, 영어, 일본어 쿼리에 대해
                  언어별 최적화된 재순위화를 수행하는 시스템으로 확장해보세요.
                  특히 Cross-lingual 검색에서의 재순위화 전략을 고민해보세요.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter3"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 분산 RAG 시스템
          </Link>
          
          <Link
            href="/modules/rag/advanced/chapter5"
            className="inline-flex items-center gap-2 bg-orange-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-orange-600 transition-colors"
          >
            다음: RAG 평가와 모니터링
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}