'use client'

import { Layers } from 'lucide-react'

export default function Section2() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
          <Layers className="text-orange-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.2 RAPTOR: 재귀적 트리 구조의 검색</h2>
          <p className="text-gray-600 dark:text-gray-400">Stanford의 계층적 문서 구조화 연구 (2024.01)</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
          <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">RAPTOR의 혁신: 재귀적 요약 트리</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>RAPTOR(Recursive Abstractive Processing for Tree-Organized Retrieval)는
              문서를 계층적으로 요약하여 다양한 추상화 수준에서 검색을 가능하게 합니다.</strong>
              이는 긴 문서나 복잡한 주제에 대한 질문에 특히 효과적이며, 전체적인 맥락과
              세부 정보를 모두 포착할 수 있습니다.
            </p>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
            <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import torch
from transformers import AutoModel, AutoTokenizer

@dataclass
class RAPTORNode:
    """RAPTOR 트리의 노드"""
    level: int
    content: str
    summary: str
    children: List['RAPTORNode']
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None

class RAPTOR:
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",
                 summarization_model: str = "facebook/bart-large-cnn",
                 max_cluster_size: int = 10):
        """
        RAPTOR: 재귀적 트리 기반 검색
        - 계층적 문서 구조화
        - 다중 수준 요약
        - 적응적 검색
        """
        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.summarizer = self._init_summarizer(summarization_model)
        self.max_cluster_size = max_cluster_size
        self.tree_root = None

    def build_tree(self, documents: List[str]) -> RAPTORNode:
        """문서 집합으로부터 RAPTOR 트리 구축"""
        print("Building RAPTOR tree...")

        # 1단계: 리프 노드 생성 (원본 문서)
        leaf_nodes = []
        for i, doc in enumerate(documents):
            node = RAPTORNode(
                level=0,
                content=doc,
                summary=doc[:200] + "...",  # 초기에는 앞부분만
                children=[],
                embedding=self._get_embedding(doc)
            )
            leaf_nodes.append(node)

        # 2단계: 재귀적으로 상위 레벨 구축
        current_level_nodes = leaf_nodes
        level = 0

        while len(current_level_nodes) > 1:
            level += 1
            print(f"Building level {level} with {len(current_level_nodes)} nodes")

            # 클러스터링
            clusters = self._cluster_nodes(current_level_nodes)

            # 각 클러스터에 대해 부모 노드 생성
            parent_nodes = []
            for cluster_id, nodes in clusters.items():
                parent_node = self._create_parent_node(nodes, level)
                parent_nodes.append(parent_node)

            current_level_nodes = parent_nodes

        # 루트 노드
        self.tree_root = current_level_nodes[0] if current_level_nodes else None
        return self.tree_root

    def _cluster_nodes(self, nodes: List[RAPTORNode]) -> Dict[int, List[RAPTORNode]]:
        """노드 클러스터링"""
        if len(nodes) <= self.max_cluster_size:
            return {0: nodes}

        # 임베딩 추출
        embeddings = np.array([node.embedding for node in nodes])

        # 최적 클러스터 수 결정
        n_clusters = max(2, len(nodes) // self.max_cluster_size)

        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # 클러스터별로 그룹화
        clusters = defaultdict(list)
        for node, label in zip(nodes, cluster_labels):
            node.cluster_id = label
            clusters[label].append(node)

        return clusters

    def _create_parent_node(self, children: List[RAPTORNode], level: int) -> RAPTORNode:
        """자식 노드들로부터 부모 노드 생성"""
        # 자식들의 내용 결합
        combined_content = "\n\n".join([child.summary for child in children])

        # 요약 생성
        summary = self._generate_summary(combined_content)

        # 부모 노드 생성
        parent = RAPTORNode(
            level=level,
            content=combined_content,
            summary=summary,
            children=children,
            embedding=self._get_embedding(summary)
        )

        return parent

    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """텍스트 요약 생성"""
        # 실제로는 BART 등의 요약 모델 사용
        # 여기서는 간단한 추출적 요약
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text

        # TF-IDF 기반 중요 문장 선택 (시뮬레이션)
        important_sentences = sentences[:3]  # 처음 3문장
        return '. '.join(important_sentences) + '.'

    def _get_embedding(self, text: str) -> np.ndarray:
        """텍스트 임베딩 생성"""
        inputs = self.tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512, padding=True)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.numpy()[0]

    def search(self, query: str, top_k: int = 5,
              collapse_threshold: float = 0.5) -> List[Tuple[RAPTORNode, float]]:
        """
        RAPTOR 검색
        - 트리의 모든 레벨에서 검색
        - 관련성에 따라 노드 확장/축소
        """
        if not self.tree_root:
            return []

        query_embedding = self._get_embedding(query)

        # 모든 노드와 점수 계산
        all_nodes_scores = []
        self._collect_relevant_nodes(
            self.tree_root, query_embedding,
            all_nodes_scores, collapse_threshold
        )

        # 점수순 정렬
        all_nodes_scores.sort(key=lambda x: x[1], reverse=True)

        # 중복 제거 (자식이 선택되면 부모는 제외)
        selected_nodes = []
        selected_contents = set()

        for node, score in all_nodes_scores:
            # 이미 선택된 내용과 중복되지 않는지 확인
            if node.content not in selected_contents:
                selected_nodes.append((node, score))
                selected_contents.add(node.content)

                # 하위 노드들의 내용도 추가 (중복 방지)
                self._add_descendant_contents(node, selected_contents)

            if len(selected_nodes) >= top_k:
                break

        return selected_nodes

    def _collect_relevant_nodes(self, node: RAPTORNode, query_embedding: np.ndarray,
                               results: List[Tuple[RAPTORNode, float]],
                               threshold: float):
        """관련 노드 수집 (재귀적)"""
        # 현재 노드와의 유사도 계산
        similarity = self._cosine_similarity(query_embedding, node.embedding)

        # 임계값 이상이면 결과에 추가
        if similarity >= threshold:
            results.append((node, similarity))

            # 높은 관련성이면 자식 노드도 탐색
            if similarity >= threshold + 0.2:  # 더 높은 임계값
                for child in node.children:
                    self._collect_relevant_nodes(
                        child, query_embedding, results, threshold
                    )
        else:
            # 낮은 관련성이어도 자식 중 일부는 관련될 수 있음
            # 샘플링하여 탐색
            if node.children and np.random.random() < 0.3:
                sample_size = min(3, len(node.children))
                sampled_children = np.random.choice(
                    node.children, size=sample_size, replace=False
                )
                for child in sampled_children:
                    self._collect_relevant_nodes(
                        child, query_embedding, results, threshold
                    )

    def _add_descendant_contents(self, node: RAPTORNode, contents_set: set):
        """노드의 모든 하위 내용 추가"""
        for child in node.children:
            contents_set.add(child.content)
            self._add_descendant_contents(child, contents_set)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _init_summarizer(self, model_name: str):
        """요약 모델 초기화"""
        # 실제로는 transformers pipeline 사용
        from transformers import pipeline
        return pipeline("summarization", model=model_name)

    def visualize_tree(self, max_depth: int = 3) -> str:
        """트리 구조 시각화"""
        if not self.tree_root:
            return "Tree not built"

        lines = []
        self._visualize_node(self.tree_root, lines, "", True, max_depth)
        return "\n".join(lines)

    def _visualize_node(self, node: RAPTORNode, lines: List[str],
                       prefix: str, is_last: bool, max_depth: int):
        """노드 시각화 (재귀적)"""
        if max_depth <= 0:
            return

        # 현재 노드 출력
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}Level {node.level}: {node.summary[:50]}...")

        # 자식 노드들
        if node.children:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self._visualize_node(
                    child, lines, prefix + extension,
                    is_last_child, max_depth - 1
                )

# RAPTOR 개선: 동적 트리 업데이트
class DynamicRAPTOR(RAPTOR):
    def __init__(self, *args, **kwargs):
        """동적 업데이트가 가능한 RAPTOR"""
        super().__init__(*args, **kwargs)
        self.update_threshold = 0.3  # 재구조화 임계값

    def add_documents(self, new_documents: List[str]):
        """새로운 문서 추가"""
        # 새 문서들을 리프 노드로 추가
        new_nodes = []
        for doc in new_documents:
            node = RAPTORNode(
                level=0,
                content=doc,
                summary=doc[:200] + "...",
                children=[],
                embedding=self._get_embedding(doc)
            )
            new_nodes.append(node)

        # 기존 트리와 병합
        self._merge_nodes(new_nodes)

    def _merge_nodes(self, new_nodes: List[RAPTORNode]):
        """새 노드들을 기존 트리에 병합"""
        # 가장 유사한 기존 클러스터 찾기
        for node in new_nodes:
            best_cluster = self._find_best_cluster(node)

            if best_cluster:
                # 기존 클러스터에 추가
                self._add_to_cluster(node, best_cluster)
            else:
                # 새 클러스터 생성
                self._create_new_cluster(node)

        # 트리 재균형화 체크
        if self._needs_rebalancing():
            self._rebalance_tree()

    def _find_best_cluster(self, node: RAPTORNode) -> Optional[RAPTORNode]:
        """가장 적합한 클러스터 찾기"""
        # 레벨 0의 모든 부모 노드들과 비교
        # 실제 구현은 더 복잡함
        return None

    def _needs_rebalancing(self) -> bool:
        """트리 재균형화 필요 여부 확인"""
        # 클러스터 크기 불균형 체크
        return False

    def _rebalance_tree(self):
        """트리 재균형화"""
        print("Rebalancing RAPTOR tree...")
        # 전체 트리 재구축 또는 부분 재구조화

# 사용 예제
print("=== RAPTOR 데모 ===\n")

# 샘플 문서
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Transfer learning reuses knowledge from one task for another.",
    "Supervised learning requires labeled training data.",
    "Unsupervised learning finds patterns without labels.",
    "Semi-supervised learning uses both labeled and unlabeled data.",
    "Active learning selects the most informative samples for labeling."
]

# RAPTOR 트리 구축
raptor = RAPTOR(max_cluster_size=3)
root = raptor.build_tree(documents)

# 트리 시각화
print("RAPTOR Tree Structure:")
print(raptor.visualize_tree(max_depth=3))

# 검색 테스트
queries = [
    "What are the types of machine learning?",
    "How does deep learning work?",
    "Explain learning without labels"
]

print("\n\n=== RAPTOR Search Results ===")
for query in queries:
    print(f"\nQuery: {query}")
    results = raptor.search(query, top_k=3)

    for i, (node, score) in enumerate(results, 1):
        print(f"\n{i}. Level {node.level} (Score: {score:.3f})")
        print(f"   Summary: {node.summary[:100]}...")
        if node.level == 0:
            print(f"   Type: Leaf node (original document)")
        else:
            print(f"   Type: Internal node (summary of {len(node.children)} children)")`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
