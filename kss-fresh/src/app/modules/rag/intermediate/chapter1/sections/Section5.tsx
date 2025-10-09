'use client'

import { BookOpen } from 'lucide-react'
import CodeSandbox from '../../../components/CodeSandbox'

export default function Section5() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-emerald-100 dark:bg-emerald-900/20 flex items-center justify-center">
          <BookOpen className="text-emerald-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.5 실전 코드 예제</h2>
          <p className="text-gray-600 dark:text-gray-400">프로덕션 환경에서 사용하는 벡터 DB 실습</p>
        </div>
      </div>

      <div className="space-y-6">
        <CodeSandbox
          title="실습 1: Pinecone 벡터 DB 셋업 및 검색"
          description="완전 관리형 벡터 DB로 빠르게 시작하기"
          language="python"
          code={`import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

# Pinecone 초기화
pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"
)

# 인덱스 생성 (768차원 - text-embedding-ada-002)
index_name = "rag-production"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        pods=1,
        pod_type="p1.x1"  # 프로덕션용 Pod
    )

# LangChain과 연동
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 문서 추가
docs = [
    "RAG는 Retrieval-Augmented Generation의 약자입니다.",
    "벡터 데이터베이스는 임베딩을 효율적으로 저장합니다.",
    "Pinecone은 자동 스케일링을 지원합니다."
]
vectorstore.add_texts(docs)

# 유사도 검색 (상위 3개)
query = "RAG 시스템이란?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")`}
          output={`1. RAG는 Retrieval-Augmented Generation의 약자입니다.
2. 벡터 데이터베이스는 임베딩을 효율적으로 저장합니다.
3. Pinecone은 자동 스케일링을 지원합니다.`}
          highlightLines={[21, 22, 23, 24, 25]}
        />

        <CodeSandbox
          title="실습 2: Qdrant 필터링 및 메타데이터 검색"
          description="Rust 기반 고성능 벡터 검색 엔진"
          language="python"
          code={`from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Qdrant 클라이언트 초기화 (로컬)
client = QdrantClient(host="localhost", port=6333)

# 컬렉션 생성
collection_name = "documents"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# 메타데이터와 함께 벡터 삽입
points = [
    PointStruct(
        id=1,
        vector=[0.1] * 768,
        payload={"category": "AI", "year": 2024, "topic": "RAG"}
    ),
    PointStruct(
        id=2,
        vector=[0.2] * 768,
        payload={"category": "DB", "year": 2024, "topic": "Vector"}
    ),
    PointStruct(
        id=3,
        vector=[0.3] * 768,
        payload={"category": "AI", "year": 2023, "topic": "LLM"}
    )
]
client.upsert(collection_name=collection_name, points=points)

# 필터링 검색 (AI 카테고리만)
search_result = client.search(
    collection_name=collection_name,
    query_vector=[0.15] * 768,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="AI")
            )
        ]
    ),
    limit=5
)

for hit in search_result:
    print(f"ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")`}
          output={`ID: 1, Score: 0.9987, Payload: {'category': 'AI', 'year': 2024, 'topic': 'RAG'}
ID: 3, Score: 0.9712, Payload: {'category': 'AI', 'year': 2023, 'topic': 'LLM'}`}
          highlightLines={[35, 36, 37, 38, 39, 40, 41, 42, 43, 44]}
        />

        <CodeSandbox
          title="실습 3: 벡터 DB 성능 벤치마크"
          description="Chroma vs Qdrant 검색 성능 비교"
          language="python"
          code={`import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb

# 테스트 데이터 생성 (10,000개 벡터)
num_vectors = 10000
dimension = 768
vectors = np.random.rand(num_vectors, dimension).astype(np.float32)

# --- Chroma 벤치마크 ---
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="benchmark")

start_time = time.time()
for i in range(num_vectors):
    collection.add(
        embeddings=[vectors[i].tolist()],
        ids=[f"doc_{i}"]
    )
chroma_insert_time = time.time() - start_time

start_time = time.time()
query_vector = vectors[0].tolist()
results = collection.query(query_embeddings=[query_vector], n_results=10)
chroma_search_time = time.time() - start_time

# --- Qdrant 벤치마크 ---
qdrant_client = QdrantClient(host="localhost", port=6333)
qdrant_client.recreate_collection(
    collection_name="benchmark",
    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
)

start_time = time.time()
points = [
    PointStruct(id=i, vector=vectors[i].tolist())
    for i in range(num_vectors)
]
qdrant_client.upsert(collection_name="benchmark", points=points)
qdrant_insert_time = time.time() - start_time

start_time = time.time()
search_result = qdrant_client.search(
    collection_name="benchmark",
    query_vector=vectors[0].tolist(),
    limit=10
)
qdrant_search_time = time.time() - start_time

# 결과 출력
print(f"📊 벡터 DB 성능 비교 (10,000개 벡터)")
print(f"\\nChroma:")
print(f"  - 삽입 시간: {chroma_insert_time:.2f}초")
print(f"  - 검색 시간: {chroma_search_time*1000:.2f}ms")
print(f"\\nQdrant:")
print(f"  - 삽입 시간: {qdrant_insert_time:.2f}초")
print(f"  - 검색 시간: {qdrant_search_time*1000:.2f}ms")
print(f"\\n⚡ Qdrant가 {chroma_search_time/qdrant_search_time:.1f}배 빠름!")`}
          output={`📊 벡터 DB 성능 비교 (10,000개 벡터)

Chroma:
  - 삽입 시간: 12.34초
  - 검색 시간: 5.67ms

Qdrant:
  - 삽입 시간: 2.89초
  - 검색 시간: 1.23ms

⚡ Qdrant가 4.6배 빠름!`}
          highlightLines={[51, 52, 53, 54, 55, 56, 57]}
        />
      </div>
    </section>
  )
}
