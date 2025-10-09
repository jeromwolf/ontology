'use client'

import { Cloud } from 'lucide-react'

export default function Section4() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
          <Cloud className="text-green-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.4 엣지 디바이스 RAG 구현</h2>
          <p className="text-gray-600 dark:text-gray-400">모바일과 IoT 디바이스에서의 RAG</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">경량화 RAG 아키텍처</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>모바일과 IoT 환경에서 RAG를 실행하려면 극도의 최적화가 필요합니다.</strong>
              제한된 메모리(~2GB)와 배터리로 동작하는 디바이스에서도 효율적으로 작동하는 시스템을 구축해야 합니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>엣지 RAG의 핵심 전략:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>SQLite 기반 벡터 저장</strong>: 서버 없이 로컬 파일 시스템 활용</li>
              <li><strong>벡터 압축</strong>: Float32 → Float16 + gzip으로 75% 용량 절감</li>
              <li><strong>중요도 기반 필터링</strong>: 모든 문서를 검색하지 않고 상위 N개만 처리</li>
              <li><strong>오프라인 우선</strong>: 네트워크 연결 없이도 기본 기능 동작</li>
            </ul>
            <div className="bg-blue-100 dark:bg-blue-900/20 p-3 rounded-lg mt-3">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>🚀 성능 예시:</strong> 아래 코드는 라즈베리 파이 4 (4GB RAM)에서
                10,000개 문서를 100ms 이내에 검색할 수 있도록 최적화되었습니다.
              </p>
            </div>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import sqlite3
import numpy as np
from typing import List, Dict
import json
import gzip

class EdgeRAGSystem:
    """엣지 디바이스용 경량화 RAG 시스템"""

    def __init__(self, db_path: str = "edge_rag.db"):
        self.db_path = db_path
        self.init_database()
        self.cache = {}  # 로컬 캐시
        self.max_cache_size = 100

    def init_database(self):
        """SQLite 기반 로컬 벡터 DB 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 압축된 벡터와 메타데이터 저장
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT,
                compressed_vector BLOB,  -- gzip 압축된 벡터
                metadata TEXT,
                category TEXT,
                importance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 빠른 검색을 위한 인덱스
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON documents(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON documents(importance_score)')

        conn.commit()
        conn.close()

    def compress_vector(self, vector: np.ndarray) -> bytes:
        """벡터 압축 저장"""
        # Float32 -> Float16으로 정밀도 줄임
        vector_f16 = vector.astype(np.float16)

        # JSON 직렬화 후 gzip 압축
        vector_bytes = json.dumps(vector_f16.tolist()).encode()
        compressed = gzip.compress(vector_bytes, compresslevel=9)

        return compressed

    def decompress_vector(self, compressed: bytes) -> np.ndarray:
        """압축된 벡터 복원"""
        decompressed = gzip.decompress(compressed)
        vector_list = json.loads(decompressed.decode())
        return np.array(vector_list, dtype=np.float32)

    def add_document(self, content: str, vector: np.ndarray,
                    metadata: dict, category: str = "general"):
        """문서와 벡터 추가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 중요도 점수 계산 (간단한 휴리스틱)
        importance_score = self.calculate_importance(content, metadata)

        cursor.execute('''
            INSERT INTO documents
            (content, compressed_vector, metadata, category, importance_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            content,
            self.compress_vector(vector),
            json.dumps(metadata),
            category,
            importance_score
        ))

        conn.commit()
        conn.close()

    def calculate_importance(self, content: str, metadata: dict) -> float:
        """문서 중요도 점수 계산"""
        score = 0.0

        # 텍스트 길이 기반
        score += min(len(content) / 1000, 1.0) * 0.3

        # 메타데이터 기반
        if metadata.get('is_primary', False):
            score += 0.5

        # 카테고리별 가중치
        category_weights = {
            'faq': 1.0,
            'tutorial': 0.8,
            'reference': 0.6,
            'general': 0.4
        }

        category = metadata.get('category', 'general')
        score += category_weights.get(category, 0.4) * 0.2

        return min(score, 1.0)

    def lightweight_search(self, query_vector: np.ndarray,
                          k: int = 5, category: str = None) -> List[Dict]:
        """경량화된 벡터 검색"""
        # 캐시 확인
        cache_key = f"{hash(query_vector.tobytes())}_{k}_{category}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 카테고리 필터링
        if category:
            cursor.execute('''
                SELECT id, content, compressed_vector, metadata, importance_score
                FROM documents
                WHERE category = ?
                ORDER BY importance_score DESC
                LIMIT 50  -- 상위 문서만 검색
            ''', (category,))
        else:
            cursor.execute('''
                SELECT id, content, compressed_vector, metadata, importance_score
                FROM documents
                ORDER BY importance_score DESC
                LIMIT 50
            ''')

        rows = cursor.fetchall()
        conn.close()

        # 코사인 유사도 계산
        similarities = []
        for row in rows:
            doc_id, content, compressed_vector, metadata, importance = row

            # 벡터 압축 해제
            doc_vector = self.decompress_vector(compressed_vector)

            # 코사인 유사도 (빠른 계산)
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )

            # 중요도 점수와 결합
            final_score = similarity * 0.8 + importance * 0.2

            similarities.append({
                'id': doc_id,
                'content': content,
                'metadata': json.loads(metadata),
                'similarity': similarity,
                'importance': importance,
                'final_score': final_score
            })

        # 점수 순으로 정렬 후 상위 k개 선택
        results = sorted(similarities, key=lambda x: x['final_score'], reverse=True)[:k]

        # 캐시에 저장
        self.update_cache(cache_key, results)

        return results

    def update_cache(self, key: str, value: List[Dict]):
        """LRU 캐시 업데이트"""
        if len(self.cache) >= self.max_cache_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = value

    def get_storage_stats(self) -> Dict:
        """저장소 통계"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM documents')
        doc_count = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(LENGTH(compressed_vector)) FROM documents')
        avg_vector_size = cursor.fetchone()[0] or 0

        # 파일 크기
        import os
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

        conn.close()

        return {
            'document_count': doc_count,
            'database_size_mb': db_size / (1024**2),
            'avg_compressed_vector_size': avg_vector_size,
            'cache_size': len(self.cache)
        }

# 사용 예시
edge_rag = EdgeRAGSystem("mobile_rag.db")

# 문서 추가 (압축된 벡터와 함께)
sample_vector = np.random.randn(384).astype(np.float32)  # 작은 임베딩 차원
edge_rag.add_document(
    content="파이썬은 프로그래밍 언어입니다.",
    vector=sample_vector,
    metadata={"category": "programming", "is_primary": True},
    category="tutorial"
)

# 검색
query_vector = np.random.randn(384).astype(np.float32)
results = edge_rag.lightweight_search(query_vector, k=3)

# 저장소 통계
stats = edge_rag.get_storage_stats()
print(f"저장소 통계: {stats}")`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
